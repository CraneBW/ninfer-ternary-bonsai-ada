// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// KVMem K2 prototype: in-place K phase re-rotation for int8-group64 cached pages.
//
// When the KVMem window compacts selected blocks into new slots, every moved
// block's cached K carries RoPE phases of its OLD window slot while the
// attention kernel will read it at the NEW slot (the kernel derives positions
// from block-table indices). This kernel re-bakes the phases in the shared
// D256 Hadamard basis used by INT8-G64 append and attention:
//
//   stored   = Quant_G64(H256(RoPE_src(K_raw)))
//   re-baked = Quant_G64(H256(RoPE_dst(RoPE_src^-1(H256(Dequant_G64(stored))))))
//
// RoPE here is Qwen3.6 Text MRoPE: rotary_dim 64, split-half pairs (p, p+32)
// for p in [0,32), angle phi = positions[axis(p)] * kTextRopeInvFrequency[p],
// axis = p % 3 (src/ops/kernel/rope.cuh). H256 is the warp-wide normalized
// Sylvester Hadamard (normalized_hadamard_d256_inplace) -- an involution, so
// de-rotation and re-rotation use the same helper. Because H256 mixes the four
// 64-element quantization groups, all four groups are dequantized into the
// warp's 8-register row, un-rotated to the RoPE domain where dims [0,64) are
// re-phased while dims [64,256) remain invariant, and then re-rotated and
// re-quantized across all four groups.
//
// One warp owns one (page, token, head): lane l holds dims l + 32*r for r in
// [0,8), so (row[0], row[1]) is simultaneously the rotary pair (l, l+32) in the
// un-rotated domain. Everything uses the exact production helpers
// (normalized_hadamard_d256_inplace, kv_cache_int8_quant_*) so a re-baked page
// is bit-consistent with what the append path writes for the same K at the new
// position (up to single-step quantization and fp reassociation).

#pragma once

#include <cuda_runtime.h>

#include <cstdio>

#include "ops/kernel/rope.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::kvmem {

// Local check: kvmem stays self-contained (core/device.h carries the SM gate).
inline void kvmem_cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expr, file, line,
                     cudaGetErrorString(err));
    }
}

#ifndef KVMEM_CUDA_CHECK
#define KVMEM_CUDA_CHECK(expr) ::ninfer::kvmem::kvmem_cuda_check((expr), #expr, __FILE__, __LINE__)
#endif

// The re-phase pipeline is intentionally built from the EXACT production helpers
// (codec, Hadamard, RoPE tables) so a re-baked page is bit-consistent with what
// the append path writes for the same raw K at the new position.
using namespace ninfer::ops;

template <int KVHeads>
__device__ __forceinline__ void kvmem_rerope_int8_g64_token(
    std::int8_t* codes, __half* scales, int page, int head, int token,
    const std::int32_t pos_src[3], const std::int32_t pos_dst[3], int lane) {
    const std::int64_t code_base =
        paged_kv_element_offset<kKVCacheInt8HeadDim, KVHeads>(page, head, token, 0);
    const std::int64_t scale_base =
        paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(page, head, token, 0);

    // -- dequant all 4 groups (dims lane + 32*r for r in [0,8)) --
    float row[8];
#pragma unroll
    for (int g = 0; g < kKVCacheInt8Groups; ++g) {
        const float s      = __half2float(scales[scale_base + g]);
        const int g_offset = g * kKVCacheInt8Group;
        row[2 * g]         = static_cast<float>(codes[code_base + g_offset + lane]) * s;
        row[2 * g + 1]     = static_cast<float>(codes[code_base + g_offset + lane + 32]) * s;
    }

    // -- undo the H256 quantization rotation (involution) --
    normalized_hadamard_d256_inplace(row, lane);

    // -- un-RoPE at the baked slot, re-RoPE at the new slot (dims [0,64)) --
    const int pair      = lane;  // rotary pairs are (p, p+32) for p in [0,32)
    const int axis      = pair % 3;
    const float phi_src = static_cast<float>(pos_src[axis]) * kTextRopeInvFrequency[pair];
    const float phi_dst = static_cast<float>(pos_dst[axis]) * kTextRopeInvFrequency[pair];
    float ss = 0.0f, cs = 0.0f, sd = 0.0f, cd = 0.0f;
    sincosf(phi_src, &ss, &cs);
    sincosf(phi_dst, &sd, &cd);
    // stored pair convention (apply_rope_head): x0' = x0*cos - x1*sin; x1' = x1*cos + x0*sin
    const float u0 = row[0] * cs + row[1] * ss;  // inverse: sin flips sign
    const float u1 = row[1] * cs - row[0] * ss;
    row[0]         = u0 * cd - u1 * sd;
    row[1]         = u1 * cd + u0 * sd;

    // -- re-apply H256 and re-quantize all 4 groups --
    normalized_hadamard_d256_inplace(row, lane);
#pragma unroll
    for (int g = 0; g < kKVCacheInt8Groups; ++g) {
        const float v0     = row[2 * g];
        const float v1     = row[2 * g + 1];
        const int g_offset = g * kKVCacheInt8Group;
        float absmax       = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, offset));
        }
        const KVCacheInt8QuantParams qp         = kv_cache_int8_quant_params(absmax);
        codes[code_base + g_offset + lane]      = kv_cache_int8_quant_code(v0, qp.inverse_scale);
        codes[code_base + g_offset + lane + 32] = kv_cache_int8_quant_code(v1, qp.inverse_scale);
        scales[scale_base + g]                  = qp.scale;
    }
}

// One warp re-bakes one (page, token, head) in the D256 Hadamard basis.
//   codes/scales : the device page (page-major layout of paged_kv_element_offset)
//   src_pos/dst_pos : [axis * (pages * valid_tokens) + page * valid_tokens + token]
//                     absolute MRoPE positions
//   valid_tokens : tokens < valid_tokens are re-baked (tail pages).
template <int KVHeads>
__global__ void kvmem_rerope_int8_g64_kernel(std::int8_t* codes, __half* scales,
                                             const std::int32_t* src_pos,
                                             const std::int32_t* dst_pos, int pages,
                                             int valid_tokens) {
    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int page  = static_cast<int>(blockIdx.z);
    const int lane  = static_cast<int>(threadIdx.x);  // 32 lanes == 32 RoPE pairs
    if (token >= valid_tokens) { return; }

    const int stride          = pages * valid_tokens;
    const int col             = page * valid_tokens + token;
    const std::int32_t ps[3]  = {src_pos[0 * stride + col], src_pos[1 * stride + col],
                                 src_pos[2 * stride + col]};
    const std::int32_t pd[3]  = {dst_pos[0 * stride + col], dst_pos[1 * stride + col],
                                 dst_pos[2 * stride + col]};
    kvmem_rerope_int8_g64_token<KVHeads>(codes, scales, page, head, token, ps, pd, lane);
}

// Batched indexed variant: re-bakes `blockIdx.z` moved blocks in a single
// launch per layer plane, deriving each token's MRoPE position from the block's
// scalar `from_bases[blk]` and `to_bases[blk]`.
template <int KVHeads>
__global__ void kvmem_rerope_int8_g64_indexed_kernel(std::int8_t* codes_plane,
                                                     __half* scales_plane,
                                                     const std::int32_t* page_indices,
                                                     const std::int32_t* from_bases,
                                                     const std::int32_t* to_bases,
                                                     int valid_tokens) {
    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int blk   = static_cast<int>(blockIdx.z);
    const int lane  = static_cast<int>(threadIdx.x);
    if (token >= valid_tokens) { return; }

    const int page           = page_indices[blk];
    const std::int32_t s_pos = from_bases[blk] + token;
    const std::int32_t d_pos = to_bases[blk] + token;
    const std::int32_t ps[3] = {s_pos, s_pos, s_pos};
    const std::int32_t pd[3] = {d_pos, d_pos, d_pos};
    kvmem_rerope_int8_g64_token<KVHeads>(codes_plane, scales_plane, page, head, token, ps, pd,
                                         lane);
}

// Launcher: re-bakes `pages` whole pages (all 64 tokens). A partially filled
// tail page passes valid_tokens < 64 for that page only by launching it
// separately; the production executor (K1b) will own that split.
template <int KVHeads>
void kvmem_rerope_int8_g64_pages(std::int8_t* codes, __half* scales, int pages,
                                 const std::int32_t* src_pos, const std::int32_t* dst_pos,
                                 cudaStream_t stream) {
    static_assert(kKVCacheInt8HeadDim == 256, "re-phase assumes the 256-dim MRoPE head");
    const dim3 grid(kPagedKVPageSize, KVHeads, pages);
    const dim3 block(32, 1u, 1u);
    kvmem_rerope_int8_g64_kernel<KVHeads>
        <<<grid, block, 0, stream>>>(codes, scales, src_pos, dst_pos, pages, kPagedKVPageSize);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

// Per-plane variant: the executor addresses one layer's K-codes / K-scales plane
// tensors directly. `page_index` is the physical page within the plane (planes
// are page-major: page stride = 256 * 64 * KVHeads for codes, 4 * 64 * KVHeads
// for scales). src/dst positions are page-local [axis * 64 + token].
template <int KVHeads>
void kvmem_rerope_int8_g64_plane(std::int8_t* codes_plane, __half* scales_plane,
                                 std::int32_t page_index, const std::int32_t* src_pos,
                                 const std::int32_t* dst_pos, cudaStream_t stream) {
    constexpr std::int64_t kCodeStride =
        static_cast<std::int64_t>(kKVCacheInt8HeadDim) * kPagedKVPageSize * KVHeads;
    constexpr std::int64_t kScaleStride =
        static_cast<std::int64_t>(kKVCacheInt8Groups) * kPagedKVPageSize * KVHeads;
    const dim3 grid(kPagedKVPageSize, KVHeads, 1);
    const dim3 block(32, 1u, 1u);
    kvmem_rerope_int8_g64_kernel<KVHeads><<<grid, block, 0, stream>>>(
        codes_plane + static_cast<std::int64_t>(page_index) * kCodeStride,
        scales_plane + static_cast<std::int64_t>(page_index) * kScaleStride, src_pos, dst_pos, 1,
        kPagedKVPageSize);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

// Batched per-plane launcher: re-bakes `n_blocks` physical pages on one layer's
// K-codes / K-scales plane in a single launch.
template <int KVHeads>
void kvmem_rerope_int8_g64_planes(std::int8_t* codes_plane, __half* scales_plane,
                                  const std::int32_t* d_page_indices,
                                  const std::int32_t* d_from_bases, const std::int32_t* d_to_bases,
                                  std::uint32_t n_blocks, cudaStream_t stream) {
    if (n_blocks == 0) { return; }
    const dim3 grid(kPagedKVPageSize, KVHeads, n_blocks);
    const dim3 block(32, 1u, 1u);
    kvmem_rerope_int8_g64_indexed_kernel<KVHeads><<<grid, block, 0, stream>>>(
        codes_plane, scales_plane, d_page_indices, d_from_bases, d_to_bases, kPagedKVPageSize);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kvmem
