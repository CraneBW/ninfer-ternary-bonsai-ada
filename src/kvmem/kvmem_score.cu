// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// nvcc-compiled bridge for the K3 retrieval scorer (see kvmem_bridge.h).
//
// Two kernels:
//   * kvmem_capture_query_kernel -- lifts one column of a pre-RoPE Q tensor into
//     the capture buffer, one layer slot per full-attention layer.
//   * kvmem_score_block_kernel   -- one block per ledger block: the block's
//     mean-K rebuilt in the RAW domain (dequantize, undo the H256 quantization
//     rotation, un-rotate the rotary dims at the position the page is baked at)
//     and the cosine between it and the captured raw query of every layer,
//     averaged over the KV heads and the layers.
//
// Raw domain matters: a key is position independent, so comparing a query with
// blocks written at other positions is only meaningful once RoPE is undone.

#include "kvmem/kvmem_bridge.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/rope.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::kvmem {

using namespace ninfer::ops;

namespace {

constexpr int kScoreThreads = 256;
constexpr int kScoreWarps   = kScoreThreads / 32;

__global__ void kvmem_capture_query_kernel(const __nv_bfloat16* __restrict__ q,
                                           std::uint32_t q_heads, std::uint32_t head_dim,
                                           std::uint32_t column, __nv_bfloat16* __restrict__ dst) {
    const std::uint32_t unit =
        blockIdx.x * static_cast<std::uint32_t>(kScoreThreads) + threadIdx.x;
    if (unit >= q_heads * head_dim) { return; }
    const std::uint32_t head = unit / head_dim;
    const std::uint32_t d    = unit - head * head_dim;
    // q is [head_dim, q_heads, tokens]; the capture keeps [q_heads][head_dim].
    const std::int64_t src = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(head_dim) *
                                 (static_cast<std::int64_t>(head) +
                                  static_cast<std::int64_t>(q_heads) * column);
    dst[unit] = q[src];
}

// One thread block owns one ledger block and walks the layers serially, so the
// sum is deterministic. Warps split a KV head's 64 tokens; each warp undoes the
// quantization rotation and the position rotation of the rows it owns and
// accumulates them into the shared mean-K, after which one warp per KV head
// takes the cosine against the query averaged over that head's GQA group.
template <int KVHeads, int QHeads>
__global__ void kvmem_score_block_kernel(const __nv_bfloat16* __restrict__ query,
                                        const std::int8_t* const* __restrict__ k_codes,
                                        const __half* const* __restrict__ k_scales,
                                        const std::int32_t* __restrict__ pages,
                                        const std::int32_t* __restrict__ baked_positions,
                                        std::uint32_t layers, float* __restrict__ out) {
    static_assert(QHeads % KVHeads == 0, "GQA needs whole query-head groups");
    constexpr int kGroup        = QHeads / KVHeads;
    constexpr int kDim          = kKVCacheInt8HeadDim;
    constexpr int kWarpsPerHead = kScoreWarps / KVHeads;
    constexpr int kTokensPerWarp = kPagedKVPageSize / kWarpsPerHead;
    static_assert(kScoreWarps % KVHeads == 0, "warps must split the KV heads evenly");
    static_assert(kPagedKVPageSize % kWarpsPerHead == 0, "warps must split the tokens evenly");

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int head = warp % KVHeads;
    const int slot = warp / KVHeads;

    __shared__ float mean_k[KVHeads][kDim];
    __shared__ float warp_cos[kScoreWarps];

    const std::int32_t page  = pages[blockIdx.x];
    const std::int32_t baked = baked_positions[blockIdx.x];

    float total = 0.0f;  // thread 0 only
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::int8_t* codes = k_codes[layer] +
                                   static_cast<std::int64_t>(page) * kDim * kPagedKVPageSize *
                                       KVHeads;
        const __half* scales = k_scales[layer] +
                               static_cast<std::int64_t>(page) * kKVCacheInt8Groups *
                                   kPagedKVPageSize * KVHeads;
        __syncthreads();  // the previous layer's readers are done with mean_k
        for (int unit = static_cast<int>(threadIdx.x); unit < KVHeads * kDim;
             unit += kScoreThreads) {
            mean_k[unit / kDim][unit % kDim] = 0.0f;
        }
        __syncthreads();

        // ---- sum the block's raw K over its tokens ----
        float acc[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
        for (int j = 0; j < kTokensPerWarp; ++j) {
            const int token = slot + j * kWarpsPerHead;
            float row[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                const float scale =
                    __half2float(scales[paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(
                        0, head, token, r / 2)]);
                row[r] = static_cast<float>(
                             codes[paged_kv_element_offset<kDim, KVHeads>(0, head, token, d)]) *
                         scale;
            }
            // Undo the quantization rotation (H256 is an involution), which
            // brings the row back to the RoPE'd key.
            normalized_hadamard_d256_inplace(row, lane);
            // Undo RoPE: the rotary pairs live in the first group, and in this
            // lane layout a pair is (row[0], row[1]) at pair index = lane.
            const float phi = static_cast<float>(baked + token) * kTextRopeInvFrequency[lane];
            const float sine = sinf(phi);
            const float cosi = cosf(phi);
            const float raw0 = row[0] * cosi + row[1] * sine;
            const float raw1 = row[1] * cosi - row[0] * sine;
            acc[0] += raw0;
            acc[1] += raw1;
#pragma unroll
            for (int r = 2; r < 8; ++r) { acc[r] += row[r]; }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            atomicAdd(&mean_k[head][lane + 32 * r], acc[r]);
        }
        __syncthreads();

        // ---- cosine against the query averaged over this head's GQA group ----
        float cos_head = 0.0f;
        if (warp < KVHeads) {
            const int h = warp;
            float dot = 0.0f, q_norm = 0.0f, k_norm = 0.0f;
            for (int d = lane; d < kDim; d += 32) {
                float q_bar = 0.0f;
#pragma unroll
                for (int gi = 0; gi < kGroup; ++gi) {
                    q_bar += __bfloat162float(
                        query[(static_cast<std::int64_t>(layer) * QHeads + h * kGroup + gi) * kDim +
                              d]);
                }
                q_bar /= static_cast<float>(kGroup);
                const float m = mean_k[h][d];
                dot    = fmaf(q_bar, m, dot);
                q_norm = fmaf(q_bar, q_bar, q_norm);
                k_norm = fmaf(m, m, k_norm);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot    += __shfl_xor_sync(0xffffffffu, dot, offset);
                q_norm += __shfl_xor_sync(0xffffffffu, q_norm, offset);
                k_norm += __shfl_xor_sync(0xffffffffu, k_norm, offset);
            }
            if (lane == 0) {
                const float den = q_norm * k_norm;
                cos_head        = den > 0.0f ? dot * rsqrtf(den) : 0.0f;
            }
        }
        // Only lane 0 holds the warp's finished cosine; the other lanes must not
        // store their zero.
        if (lane == 0) { warp_cos[warp] = cos_head; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int w = 0; w < KVHeads; ++w) { sum += warp_cos[w]; }
            total += sum / static_cast<float>(KVHeads);
        }
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = layers > 0 ? total / static_cast<float>(layers) : 0.0f;
    }
}

} // namespace

void kvmem_capture_query(const void* q_rope, std::uint32_t q_heads, std::uint32_t head_dim,
                         std::uint32_t tokens, std::uint32_t column, void* dst,
                         std::uint32_t layer, cudaStream_t stream) {
    if (q_rope == nullptr || dst == nullptr) { return; }
    if (q_heads == 0 || head_dim != kKVCacheInt8HeadDim || column >= tokens) {
        throw std::invalid_argument("kvmem query capture: query geometry is out of domain");
    }
    const std::uint32_t units = q_heads * head_dim;
    const std::uint32_t grid  = (units + kScoreThreads - 1) / kScoreThreads;
    __nv_bfloat16* layer_dst =
        static_cast<__nv_bfloat16*>(dst) + static_cast<std::int64_t>(layer) * units;
    kvmem_capture_query_kernel<<<grid, kScoreThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_rope), q_heads, head_dim, column, layer_dst);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

void kvmem_score_blocks(DeviceKVPagePool& pool, const KvmemQueryCapture& query,
                        const std::int32_t* page_indices, const std::int32_t* baked_positions,
                        std::uint32_t n_blocks, float* scores_out, cudaStream_t stream) {
    if (!query.valid() || page_indices == nullptr || baked_positions == nullptr ||
        scores_out == nullptr || n_blocks == 0) {
        return;
    }
    const KVPageGeometry& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("kvmem score: expected layer-grouped KV planes");
    }
    const std::uint32_t layers = static_cast<std::uint32_t>(geometry.planes.size() / 4);
    if (query.layers != layers || query.head_dim != kKVCacheInt8HeadDim) {
        throw std::invalid_argument("kvmem score: query capture does not match the KV geometry");
    }
    const int kv_heads = geometry.planes.front().head_extent;

    std::vector<std::uint32_t> blocks;
    std::vector<std::int32_t> pages;
    std::vector<std::int32_t> baked;
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        if (page_indices[i] < 0) { continue; }
        blocks.push_back(i);
        pages.push_back(page_indices[i]);
        baked.push_back(baked_positions[i]);
    }
    if (blocks.empty()) { return; }

    const auto run = [&]<int KVHeads, int QHeads>() {
        std::vector<const std::int8_t*> codes(layers);
        std::vector<const __half*> scales(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            codes[layer]  = static_cast<const std::int8_t*>(pool.plane(layer * 4).data);
            scales[layer] = static_cast<const __half*>(pool.plane(layer * 4 + 2).data);
        }
        DeviceBuffer code_ptrs(sizeof(void*) * layers);
        DeviceBuffer scale_ptrs(sizeof(void*) * layers);
        DeviceBuffer page_dev(sizeof(std::int32_t) * pages.size());
        DeviceBuffer baked_dev(sizeof(std::int32_t) * baked.size());
        DeviceBuffer scores_dev(sizeof(float) * pages.size());
        code_ptrs.copy_from_host(codes.data(), code_ptrs.bytes);
        scale_ptrs.copy_from_host(scales.data(), scale_ptrs.bytes);
        page_dev.copy_from_host(pages.data(), page_dev.bytes);
        baked_dev.copy_from_host(baked.data(), baked_dev.bytes);
        kvmem_score_block_kernel<KVHeads, QHeads><<<static_cast<unsigned>(pages.size()),
                                                   kScoreThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data),
            static_cast<const std::int8_t* const*>(code_ptrs.p),
            static_cast<const __half* const*>(scale_ptrs.p),
            static_cast<const std::int32_t*>(page_dev.p),
            static_cast<const std::int32_t*>(baked_dev.p), layers,
            static_cast<float*>(scores_dev.p));
        KVMEM_CUDA_CHECK(cudaGetLastError());
        // The host read below is a default-stream cudaMemcpy, which does not
        // order against the compute stream the kernel ran on.
        KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> scored(pages.size(), 0.0f);
        scores_dev.copy_to_host(scored.data(), scores_dev.bytes);
        for (std::size_t i = 0; i < scored.size(); ++i) { scores_out[blocks[i]] = scored[i]; }
    };

    if (kv_heads == 4 && query.q_heads == 24) {
        run.template operator()<4, 24>();
    } else if (kv_heads == 2 && query.q_heads == 16) {
        run.template operator()<2, 16>();
    } else {
        throw std::invalid_argument("kvmem score: unsupported query/KV head counts");
    }
}

} // namespace ninfer::kvmem
