// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself (measured: 15.7 t/s with MTP N=2 against 62.1 t/s without). This kernel
// keeps the weights single-read: the code byte and scale are loaded once per group and reused for
// all kT tokens, while each token contributes its own four activations.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// Token-blocked variant, for PREFILL: the prefill chunk.
//
// The verify kernel above takes the whole T at once and is capped at kT <= 8 by register
// pressure, so it cannot serve a prefill chunk -- the CLI requires that chunk to be a multiple
// of 128 (apps/cli/options.cpp: "--prefill-chunk must be a multiple of 128"). Without this
// kernel, every prefill therefore falls through to the correctness-first reference kernel in
// ternary_rowsplit_gemm.cuh, whose shape is one 128-thread CTA per output row plus seven
// block-wide barriers per row and a four-way duplicated code load (threads index>>2 share a
// byte). Published patch state measured that kernel at 42 GiB/s against this card's 638.7 GB/s
// sustained-read ceiling -- 7%.
//
// This kernel keeps the GEMV shape -- lane l reads code byte l and so covers columns 4l..4l+3,
// weights are decoded once per group and reused for the whole token tile, and the only reduction
// is five shuffle steps -- but tiles BOTH axes: the token axis across blockIdx.y (kT tokens) and
// the row axis across kR rows per warp.
//
// Why the row axis matters: measured on the target card the R=1/kT=8 shape reaches 125.7 t/s at
// an effective weight bandwidth of only 112 GB/s, while the T=1 GEMV reads at 422 GB/s. The
// kernel is therefore NOT weight-bandwidth-bound -- it is bound by the activation side, which
// loads kT*4 elements per lane per group against a single weight byte. Since activations are
// identical for every output row of the same token, widening the warp over rows amortises them:
//
//   loads per FMA  =  (2*kT + kR) / (4 * kR * kT)
//
// R=1,kT=8 gives 0.53; R=2,kT=8 gives 0.28; R=4,kT=4 gives 0.19. The cost is registers
// (kR*kT accumulators plus 2*kT activation pairs), which is why both axes are capped small.
template <int kR, int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_block_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                        std::int32_t groups_per_row, std::int32_t tokens,
                                        std::int32_t out_row_stride) {
    static_assert(kR >= 1 && kR <= 4, "row block must keep accumulators in registers");
    static_assert(kT >= 1 && kT <= 8, "token block must keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    const int row0    = warp * kR;
    const int token0  = static_cast<int>(blockIdx.y) * kT;
    if (row0 >= rows) { return; }

    float accumulator[kR][kT];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kT; ++t) { accumulator[r][t] = 0.0f; }
    }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;

        // Activations: loaded ONCE per group and reused by every row this warp owns. This is the
        // whole point of kR -- with kR == 1 they are re-loaded for each row the warp covers.
        float2 low[kT];
        float2 high[kT];
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            const std::int32_t token = token0 + t;
            low[t]  = make_float2(0.0f, 0.0f);
            high[t] = make_float2(0.0f, 0.0f);
            if (token < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(token) * groups_per_row * kGemvGroupK;
                low[t]  = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                high[t] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
            }
        }

#pragma unroll
        for (int r = 0; r < kR; ++r) {
            const int row = row0 + r;
            if (row < rows) {
                const std::uint8_t* code_row =
                    codes + static_cast<std::int64_t>(row) * groups_per_row *
                                kGemvCodeBytesPerGroup;
                const std::uint8_t* scale_row =
                    scales + static_cast<std::int64_t>(row) * groups_per_row *
                                 kGemvScaleBytesPerGroup;
                const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
                const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
                const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
                const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
                const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
                const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    if (token0 + t < tokens) {
                        const float dot =
                            fmaf(weight0, low[t].x,
                                 fmaf(weight1, low[t].y,
                                      fmaf(weight2, high[t].x, weight3 * high[t].y)));
                        accumulator[r][t] = fmaf(scale, dot, accumulator[r][t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            float value = accumulator[r][t];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            const int row   = row0 + r;
            const int token = token0 + t;
            if (lane == 0 && row < rows && token < tokens) {
                // Token-major output: element (row, token) lives at token * out_row_stride + row.
                out[static_cast<std::int64_t>(token) * out_row_stride + row] =
                    __float2bfloat16_rn(value);
            }
        }
    }
}

} // namespace ninfer::ops::detail
