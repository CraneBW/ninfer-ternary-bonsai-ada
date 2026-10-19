// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// nvcc-compiled bridge for the K3 retrieval evidence (see kvmem_bridge.h).
//
// Three kernels:
//   * kvmem_accumulate_query_kernel -- sums the pre-RoPE Q of every prompt
//     position in the request's query span, one layer slot per full-attention
//     layer. Chunked prefill calls it once per chunk, so the capture ends up
//     with the same sum a single pass would produce.
//   * kvmem_export_mean_k_kernel    -- rebuilds each resident block's mean raw K
//     sum (dequantize, undo the H256 quantization rotation, un-rotate the rotary
//     dims at the position the page is baked at, sum the written tokens) and
//     parks it on the host. A block keeps this sum after it leaves the device.
//
// Raw domain matters: a key is position independent, so comparing a query with
// blocks written at other positions is only meaningful once RoPE is undone.

#include "kvmem/kvmem_bridge.h"
#include "kvmem/kvmem_layout.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/rope.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::kvmem {

using namespace ninfer::ops;

namespace {

constexpr int kScoreThreads = 256;
constexpr int kScoreWarps   = kScoreThreads / 32;
constexpr std::uint32_t kExportChunkBlocks = 64;

// One thread owns one (head, dim) element and walks the query span's columns.
// The span is a token range, so it never spans more than the request's prompt;
// a chunk contributes only the columns it carries.
__global__ void kvmem_accumulate_query_kernel(const __nv_bfloat16* __restrict__ q,
                                              std::uint32_t q_heads, std::uint32_t head_dim,
                                              std::uint32_t tokens, std::uint32_t chunk_begin,
                                              std::uint32_t query_begin, std::uint32_t query_end,
                                              float* __restrict__ sums) {
    const std::uint32_t unit =
        blockIdx.x * static_cast<std::uint32_t>(kScoreThreads) + threadIdx.x;
    if (unit >= q_heads * head_dim) { return; }
    const std::uint32_t head = unit / head_dim;
    const std::uint32_t d    = unit - head * head_dim;
    const std::uint32_t chunk_end = chunk_begin + tokens;
    const std::uint32_t begin = query_begin > chunk_begin ? query_begin : chunk_begin;
    const std::uint32_t end   = query_end < chunk_end ? query_end : chunk_end;
    if (begin >= end) { return; }
    float sum = sums[unit];
    for (std::uint32_t column = begin; column < end; ++column) {
        const std::uint32_t local = column - chunk_begin;
        // q is [head_dim, q_heads, tokens]; the capture keeps [q_heads][head_dim].
        const std::int64_t src = static_cast<std::int64_t>(d) +
                                 static_cast<std::int64_t>(head_dim) *
                                     (static_cast<std::int64_t>(head) +
                                      static_cast<std::int64_t>(q_heads) * local);
        sum += __bfloat162float(q[src]);
    }
    sums[unit] = sum;
}

// One thread block owns one ledger block and walks the layers serially, so the
// sum is deterministic. Warps split a KV head's tokens; each warp undoes the
// quantization rotation and the position rotation of the rows it owns and
// accumulates them into the shared mean-K. Only the block's written tokens are
// folded in: a window's last page may be partial.
template <int KVHeads>
__global__ void kvmem_export_mean_k_kernel(const std::int8_t* const* __restrict__ k_codes,
                                           const __half* const* __restrict__ k_scales,
                                           const std::int32_t* __restrict__ pages,
                                           const std::int32_t* __restrict__ baked_positions,
                                           const std::uint32_t* __restrict__ token_counts,
                                           std::uint32_t layers, float* __restrict__ out) {
    constexpr int kDim           = kKVCacheInt8HeadDim;
    constexpr int kWarpsPerHead  = kScoreWarps / KVHeads;
    constexpr int kTokensPerWarp = kPagedKVPageSize / kWarpsPerHead;
    static_assert(kScoreWarps % KVHeads == 0, "warps must split the KV heads evenly");
    static_assert(kPagedKVPageSize % kWarpsPerHead == 0, "warps must split the tokens evenly");

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int head = warp % KVHeads;
    const int slot = warp / KVHeads;

    static_assert(kPagedKVPageSize ==
                  kMeanKeySubblocks * kMeanKeyTokensPerSubblock);
    __shared__ float mean_k[KVHeads][kMeanKeySubblocks][kDim];

    const std::int32_t page  = pages[blockIdx.x];
    const std::int32_t baked = baked_positions[blockIdx.x];
    const std::uint32_t written = token_counts[blockIdx.x];

    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::int8_t* codes = k_codes[layer] +
                                   static_cast<std::int64_t>(page) * kDim * kPagedKVPageSize *
                                       KVHeads;
        const __half* scales = k_scales[layer] +
                               static_cast<std::int64_t>(page) * kKVCacheInt8Groups *
                                   kPagedKVPageSize * KVHeads;
        __syncthreads();
        for (int unit = static_cast<int>(threadIdx.x);
             unit < KVHeads * static_cast<int>(kMeanKeySubblocks) * kDim;
             unit += kScoreThreads) {
            const int head = unit / (static_cast<int>(kMeanKeySubblocks) * kDim);
            const int sub = (unit / kDim) % static_cast<int>(kMeanKeySubblocks);
            mean_k[head][sub][unit % kDim] = 0.0f;
        }
        __syncthreads();

        float acc[kMeanKeySubblocks][8];
#pragma unroll
        for (int sub = 0; sub < static_cast<int>(kMeanKeySubblocks); ++sub) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[sub][r] = 0.0f; }
        }
        for (int j = 0; j < kTokensPerWarp; ++j) {
            const std::uint32_t token =
                static_cast<std::uint32_t>(slot + j * kWarpsPerHead);
            if (token >= written) { continue; }
            float row[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                const float scale =
                    __half2float(scales[paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(
                        0, head, static_cast<int>(token), r / 2)]);
                row[r] = static_cast<float>(
                             codes[paged_kv_element_offset<kDim, KVHeads>(
                                 0, head, static_cast<int>(token), d)]) *
                         scale;
            }
            // Undo the quantization rotation (H256 is an involution), which
            // brings the row back to the RoPE'd key.
            normalized_hadamard_d256_inplace(row, lane);
            // Undo RoPE: the rotary pairs live in the first group, and in this
            // lane layout a pair is (row[0], row[1]) at pair index = lane.
            const float phi =
                static_cast<float>(baked + static_cast<std::int32_t>(token)) *
                kTextRopeInvFrequency[lane];
            const float sine = sinf(phi);
            const float cosi = cosf(phi);
            const float raw0 = row[0] * cosi + row[1] * sine;
            const float raw1 = row[1] * cosi - row[0] * sine;
            const std::uint32_t sub = token / kMeanKeyTokensPerSubblock;
            acc[sub][0] += raw0;
            acc[sub][1] += raw1;
#pragma unroll
            for (int r = 2; r < 8; ++r) { acc[sub][r] += row[r]; }
        }
#pragma unroll
        for (int sub = 0; sub < static_cast<int>(kMeanKeySubblocks); ++sub) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (acc[sub][r] != 0.0f) {
                    atomicAdd(&mean_k[head][sub][lane + 32 * r], acc[sub][r]);
                }
            }
        }
        __syncthreads();

        float* layer_out =
            out + (static_cast<std::int64_t>(blockIdx.x) * layers + layer) *
                      KVHeads * kMeanKeySubblocks * kDim;
        for (int unit = static_cast<int>(threadIdx.x);
             unit < KVHeads * static_cast<int>(kMeanKeySubblocks) * kDim;
             unit += kScoreThreads) {
            const int head = unit / (static_cast<int>(kMeanKeySubblocks) * kDim);
            const int sub = (unit / kDim) % static_cast<int>(kMeanKeySubblocks);
            layer_out[unit] = mean_k[head][sub][unit % kDim];
        }
        __syncthreads();
    }
}

} // namespace

void kvmem_accumulate_query(const void* q_rope, std::uint32_t q_heads, std::uint32_t head_dim,
                            std::uint32_t tokens, std::uint32_t chunk_begin,
                            std::uint32_t query_begin, std::uint32_t query_end, void* sums,
                            std::uint32_t layer, cudaStream_t stream) {
    if (q_rope == nullptr || sums == nullptr || q_heads == 0 || head_dim != kKVCacheInt8HeadDim ||
        tokens == 0) {
        return;
    }
    if (query_begin >= query_end) { return; }
    const std::uint32_t units = q_heads * head_dim;
    const std::uint32_t grid  = (units + kScoreThreads - 1) / kScoreThreads;
    float* layer_sums =
        static_cast<float*>(sums) + static_cast<std::int64_t>(layer) * units;
    kvmem_accumulate_query_kernel<<<grid, kScoreThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_rope), q_heads, head_dim, tokens, chunk_begin,
        query_begin, query_end, layer_sums);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

void kvmem_export_mean_k(DeviceKVPagePool& pool, const std::int32_t* page_indices,
                         const std::int32_t* baked_positions, const std::uint32_t* token_counts,
                         std::uint32_t n_blocks, float* mean_k_out, cudaStream_t stream) {
    if (page_indices == nullptr || baked_positions == nullptr || token_counts == nullptr ||
        mean_k_out == nullptr || n_blocks == 0) {
        return;
    }
    const KVPageGeometry& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("kvmem mean-K export: expected layer-grouped KV planes");
    }
    const std::uint32_t layers = static_cast<std::uint32_t>(geometry.planes.size() / 4);
    const std::uint32_t kv_heads = geometry.planes.front().head_extent;
    const std::uint32_t dim = kKVCacheInt8HeadDim;
    const std::size_t stride =
        static_cast<std::size_t>(layers) * kv_heads * kMeanKeySubblocks * dim;

    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        if (page_indices[i] < 0) {
            throw std::invalid_argument("kvmem mean-K export: block is not device-resident");
        }
        if (token_counts[i] > kPagedKVPageSize) {
            throw std::invalid_argument("kvmem mean-K export: token count exceeds a page");
        }
    }

    const auto run = [&]<int KVHeads>() {
        std::vector<const std::int8_t*> codes(layers);
        std::vector<const __half*> scales(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            codes[layer]  = static_cast<const std::int8_t*>(pool.plane(layer * 4).data);
            scales[layer] = static_cast<const __half*>(pool.plane(layer * 4 + 2).data);
        }
        DeviceBuffer code_ptrs(sizeof(void*) * layers);
        DeviceBuffer scale_ptrs(sizeof(void*) * layers);
        code_ptrs.copy_from_host(codes.data(), code_ptrs.bytes);
        scale_ptrs.copy_from_host(scales.data(), scale_ptrs.bytes);

        const std::uint32_t chunk = kExportChunkBlocks;
        DeviceBuffer page_dev(sizeof(std::int32_t) * chunk);
        DeviceBuffer baked_dev(sizeof(std::int32_t) * chunk);
        DeviceBuffer token_dev(sizeof(std::uint32_t) * chunk);
        DeviceBuffer out_dev(sizeof(float) * stride * chunk);

        for (std::uint32_t offset = 0; offset < n_blocks; offset += chunk) {
            const std::uint32_t count =
                (n_blocks - offset) < chunk ? (n_blocks - offset) : chunk;
            page_dev.copy_from_host(page_indices + offset, sizeof(std::int32_t) * count);
            baked_dev.copy_from_host(baked_positions + offset, sizeof(std::int32_t) * count);
            token_dev.copy_from_host(token_counts + offset, sizeof(std::uint32_t) * count);
            kvmem_export_mean_k_kernel<KVHeads><<<count, kScoreThreads, 0, stream>>>(
                static_cast<const std::int8_t* const*>(code_ptrs.p),
                static_cast<const __half* const*>(scale_ptrs.p),
                static_cast<const std::int32_t*>(page_dev.p),
                static_cast<const std::int32_t*>(baked_dev.p),
                static_cast<const std::uint32_t*>(token_dev.p), layers,
                static_cast<float*>(out_dev.p));
            KVMEM_CUDA_CHECK(cudaGetLastError());
            // The host read below is a default-stream cudaMemcpy, which does not
            // order against the compute stream the kernel ran on.
            KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
            out_dev.copy_to_host(mean_k_out + static_cast<std::size_t>(offset) * stride,
                                 sizeof(float) * stride * count);
        }
    };

    if (kv_heads == 4) {
        run.template operator()<4>();
    } else if (kv_heads == 2) {
        run.template operator()<2>();
    } else {
        throw std::invalid_argument("kvmem mean-K export: unsupported KV head count");
    }
}

} // namespace ninfer::kvmem
