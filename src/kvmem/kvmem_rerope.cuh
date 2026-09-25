// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// KVMem K2 prototype: in-place K phase re-rotation for int8-group64 cached pages.
//
// When the KVMem window compacts selected blocks into new slots, every moved
// block's cached K carries RoPE phases of its OLD window slot while the
// attention kernel will read it at the NEW slot (the kernel derives positions
// from block-table indices). This kernel re-bakes the phases:
//
//   stored = Quant(H64(RoPE_src(K_raw)))          (group 0 only; RoPE domain is
//   re-baked = Quant(H64(RoPE_dst(H64(Dequant(stored)))))   dims [0,64) = group 0)
//
// RoPE here is Qwen3.6 Text MRoPE: rotary_dim 64, split-half pairs (p, p+32)
// for p in [0,32), angle phi = positions[axis(p)] * kTextRopeInvFrequency[p],
// axis = p % 3 (src/ops/kernel/rope.cuh). H64 is the warp-shuffle normalized
// Hadamard (kv_cache_hadamard64) -- an involution, so de-rotation and
// re-rotation use the same helper. Groups 1..3 are position-free and pass
// through untouched.
//
// One warp owns one (page, token, head): lane l holds dims l and l+32, which is
// simultaneously one H64 butterfly lane-pair and one RoPE pair. Everything uses
// the exact production helpers (kv_cache_hadamard64, kv_cache_int8_quant_*)
// so a re-baked page is bit-consistent with what the append path would have
// written for the same raw K at the new position (up to fp reassociation).

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

#define KVMEM_CUDA_CHECK(expr) ::ninfer::kvmem::kvmem_cuda_check((expr), #expr, __FILE__, __LINE__)

// The re-phase pipeline is intentionally built from the EXACT production helpers
// (codec, Hadamard, RoPE tables) so a re-baked page is bit-consistent with what
// the append path writes for the same raw K at the new position.
using namespace ninfer::ops;

// One warp re-bakes group 0 of one (page, token, head).
//   codes/scales : the device page (page-major layout of paged_kv_element_offset)
//   src_pos/dst_pos : [axis * 64 + token] absolute MRoPE positions, page-local
//                     token index (layout matches the engine's positions[T,3]
//                     with T narrowed to one page)
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

    const std::int64_t code_base =
        paged_kv_element_offset<kKVCacheInt8HeadDim, KVHeads>(page, head, token, 0);
    const std::int64_t scale_base =
        paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(page, head, token, 0);

    // -- dequant group 0 (dims lane, lane+32) --
    const float s = __half2float(scales[scale_base]);
    float x0 = static_cast<float>(codes[code_base + lane]) * s;
    float x1 = static_cast<float>(codes[code_base + lane + 32]) * s;

    // -- undo the H64 quantization rotation (involution) --
    kv_cache_hadamard64(x0, x1);

    // -- un-RoPE at the baked slot, re-RoPE at the new slot --
    const int pair  = lane;  // rotary pairs are (p, p+32) for p in [0,32)
    const int axis  = pair % 3;
    const float phi_src =
        static_cast<float>(src_pos[axis * valid_tokens + token]) * kTextRopeInvFrequency[pair];
    const float phi_dst =
        static_cast<float>(dst_pos[axis * valid_tokens + token]) * kTextRopeInvFrequency[pair];
    float ss = 0.0f, cs = 0.0f, sd = 0.0f, cd = 0.0f;
    sincosf(phi_src, &ss, &cs);
    sincosf(phi_dst, &sd, &cd);
    // stored pair convention (apply_rope_head): x0' = x0*cos - x1*sin; x1' = x1*cos + x0*sin
    const float u0 = x0 * cs + x1 * ss;  // inverse: sin flips sign
    const float u1 = x1 * cs - x0 * ss;
    float v0 = u0 * cd - u1 * sd;
    float v1 = u1 * cd + u0 * sd;

    // -- re-apply H64 and re-quantize group 0 --
    kv_cache_hadamard64(v0, v1);
    float absmax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, offset));
    }
    const KVCacheInt8QuantParams qp = kv_cache_int8_quant_params(absmax);
    codes[code_base + lane]      = kv_cache_int8_quant_code(v0, qp.inverse_scale);
    codes[code_base + lane + 32] = kv_cache_int8_quant_code(v1, qp.inverse_scale);
    scales[scale_base]           = qp.scale;
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

} // namespace ninfer::kvmem
