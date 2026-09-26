// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// PTQ1_0 variants of the decode-shaped GEMV family in ternary_rowsplit_gemv.cuh.
//
// Why this exists: the fast GEMV family was written for PQ2_0's packing, where a 128-weight
// group is 32 two-bit code bytes and lane l's four weights live in byte l -- a perfect
// lane/activation alignment. PTQ1_0 kept falling through to the reference kernel, whose
// 128-thread CTA + seven barriers per output row measured 14.9 t/s prefill / 4.8 t/s decode
// against 396.8 / 26.4 for PQ2_0 on the same card (docs/perf-baseline-sm86.md).
//
// PTQ1_0's group is 24 base-3 code bytes (five trits per byte, mixed-radix:
// digit k of byte b is (((b * 3^k) mod 256) * 3) >> 8, minus one) plus a 2-byte high plane
// carrying weights 120..127, plus the same 2-byte fp16 scale. The stage walk is asymmetric
// (weights 0..79 from qs[0..15], 80..119 from qs[16..23], 120..127 from qh[0..1]), which is
// what blocked a naive reuse of the PQ2 lane map.
//
// The lane map below restores the alignment. 128 weights = 32 lanes x 4 weights:
//
//   lanes  0..19  weights 16n + 4g .. +3, byte qs[4g + j] at trit n   (n = i>>2, g = i&3, i = lane)
//   lanes 20..29  weights 80 + 8n + 4g .. +3, byte qs[16+4g+j] at n   (n = i>>1, g = i&1, i = lane-20)
//   lanes 30..31  weights 120 + 4k .. +3, digits interleaved across qh[0..1]
//
// The activation offset collapses to exactly 4*lane / 80+4*(lane-20) / 120+4*(lane-30), so a
// warp still reads x[group*128 .. group*128+127] as one contiguous 256-byte span -- the same
// coalescing the PQ2 GEMV enjoys. Decoding uses the widened byte-perm trick from the PrismML
// reference CUDA dequantizer (vec_dot_ptq1_0_q8_1): bytes are multiplied by 3^n in 16-bit
// lanes where the product cannot carry across lanes, truncated to 8 bits exactly like the
// scalar `(uint8_t)(raw * pow3[n])` cast, and the digit falls out of one more x3/shift pair.

#include "ternary_rowsplit_gemv.cuh"
#include "ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kPTQ1CodeBytes = PTQ1RowSplitStorage::kCodeBytesPerGroup;  // 24
inline constexpr int kPTQ1HighBytes = PTQ1RowSplitStorage::kHighBytesPerGroup;  // 2
inline constexpr int kPTQ1GroupK    = PTQ1RowSplitStorage::kGroupK;             // 128

// Activation offset of this lane's four weights inside one 128-weight group.
__device__ __forceinline__ int ptq1_lane_act0(int lane) {
    if (lane < 20) { return 4 * lane; }
    if (lane < 30) { return 80 + 4 * (lane - 20); }
    return 120 + 4 * (lane - 30);
}

// One 8-bit base-3 digit: literal match with PTQ1SimtDecodeAtom -- (uint8)(raw * 3^t)
// truncation, then digit = ((uint16)q * 3) >> 8, mapped to {-1, 0, +1}.
__device__ __forceinline__ float ptq1_digit(std::uint8_t raw, int trit) {
    const std::uint8_t q = static_cast<std::uint8_t>(raw * ternary_pow3(trit));
    const int xi         = static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8);
    return static_cast<float>(xi - 1);
}

// Per-lane decode result: four consecutive ternary weights and their activation offset.
struct PTQ1LaneSlice {
    float w0;
    float w1;
    float w2;
    float w3;
    std::int32_t act0;
};

// Decode this lane's four weights of `group` from the row's base/high planes.
__device__ __forceinline__ PTQ1LaneSlice ptq1_lane_decode(const std::uint8_t* __restrict__ code_row,
                                                          const std::uint8_t* __restrict__ high_row,
                                                          int lane, int group) {
    PTQ1LaneSlice slice;
    if (lane < 30) {
        // Stages c=16 (lanes 0..19) and c=8 (lanes 20..29) share one shape: four whole
        // code bytes, one trit level, four consecutive weights.
        const bool c16 = lane < 20;
        const int i    = c16 ? lane : lane - 20;
        const int n    = c16 ? i >> 2 : i >> 1;
        const int off  = c16 ? 4 * (i & 3) : 16 + 4 * (i & 1);
        slice.act0     = ptq1_lane_act0(lane);

        // Four code bytes as one aligned 32-bit load. The wide multiply must not carry
        // across lanes: raw <= 255 and 3^n <= 81, so the 16-bit lane product fits in 15
        // bits and the 0x00FF00FF mask reproduces the scalar uint8 truncation exactly.
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + group * kPTQ1CodeBytes + off);
        const std::uint32_t p = ternary_pow3(n);
        // After the x3 that turns a truncated code into its digit, lane 0's digit sits at
        // bits 0-1 and lane 1's at bits 16-17 (NOT 8-9: the second lane's product lives at
        // bits 16-25 of the intermediate, so its /256 part lands at 24-25 pre-shift).
        const std::uint32_t t_lo =
            ((__byte_perm(packed, 0u, 0x4140) * p) & 0x00FF00FFu) * 3u >> 8;
        const std::uint32_t t_hi =
            ((__byte_perm(packed, 0u, 0x4342) * p) & 0x00FF00FFu) * 3u >> 8;
        slice.w0 = static_cast<float>(t_lo & 0x3u) - 1.0f;
        slice.w1 = static_cast<float>((t_lo >> 16) & 0x3u) - 1.0f;
        slice.w2 = static_cast<float>(t_hi & 0x3u) - 1.0f;
        slice.w3 = static_cast<float>((t_hi >> 16) & 0x3u) - 1.0f;
    } else {
        // qh tail: weights 120 + 4k + j read digit (j>>1) of byte qh[j&1]. Two lanes,
        // scalar decode -- 1/16th of the group's work does not justify vectorising.
        const int k       = lane - 30;
        slice.act0        = ptq1_lane_act0(lane);
        const std::uint16_t pair =
            *reinterpret_cast<const std::uint16_t*>(high_row + group * kPTQ1HighBytes);
        const std::uint8_t b0 = static_cast<std::uint8_t>(pair & 0xFFu);
        const std::uint8_t b1 = static_cast<std::uint8_t>(pair >> 8);
        slice.w0              = ptq1_digit(b0, 2 * k);
        slice.w1              = ptq1_digit(b1, 2 * k);
        slice.w2              = ptq1_digit(b0, 2 * k + 1);
        slice.w3              = ptq1_digit(b1, 2 * k + 1);
    }
    return slice;
}

// Dot one lane's four weights with x[group*128 + act0 .. +3] (bf16, 4-byte aligned).
__device__ __forceinline__ float ptq1_lane_dot(const PTQ1LaneSlice& slice,
                                               const __nv_bfloat16* __restrict__ x,
                                               int group) {
    const std::int32_t base = group * kPTQ1GroupK + slice.act0;
    const float2 low        = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base));
    const float2 high =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base + 2));
    return fmaf(slice.w0, low.x, fmaf(slice.w1, low.y, fmaf(slice.w2, high.x, slice.w3 * high.y)));
}

// T == 1 decode GEMV, PTQ1_0. One warp per output row, no __syncthreads.
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ highs,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, std::int32_t rows,
                              std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPTQ1CodeBytes;
    const std::uint8_t* high_row =
        highs + static_cast<std::int64_t>(warp) * groups_per_row * kPTQ1HighBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;
    for (int group = 0; group < groups_per_row; ++group) {
        const PTQ1LaneSlice slice = ptq1_lane_decode(code_row, high_row, lane, group);
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup),
                           ptq1_lane_dot(slice, x, group), accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant for the verify pass (T = draft + 1): codes are decoded once per
// (row, group) and reused across the whole token tile, exactly like the PQ2 tile kernel.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ highs,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row, std::int32_t tokens,
                                   std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPTQ1CodeBytes;
    const std::uint8_t* high_row =
        highs + static_cast<std::int64_t>(warp) * groups_per_row * kPTQ1HighBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const PTQ1LaneSlice slice = ptq1_lane_decode(code_row, high_row, lane, group);
        const float scale         = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const std::int32_t base   = group * kPTQ1GroupK + slice.act0;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kPTQ1GroupK;
                const float2 low =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(slice.w0, low.x, fmaf(slice.w1, low.y,
                                                             fmaf(slice.w2, high.x,
                                                                  slice.w3 * high.y)));
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
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] = __float2bfloat16_rn(value);
        }
    }
}

// Token- and row-blocked variant for prefill, mirroring ternary_pq2_gemv_tile_block_kernel:
// activations are loaded once per group and amortised over kR rows, results staged in
// shared memory and flushed coalesced per token.
template <int kR, int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kBlockMinCtasPerSm)
void ternary_ptq1_gemv_tile_block_kernel(const __nv_bfloat16* __restrict__ x,
                                         const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ highs,
                                         const std::uint8_t* __restrict__ scales,
                                         __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                         std::int32_t groups_per_row, std::int32_t tokens,
                                         std::int32_t out_row_stride) {
    static_assert(kR >= 1 && kR <= 8, "row block must keep accumulators in registers");
    static_assert(kT >= 1 && kT <= 8, "token block must keep accumulators in registers");
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
    const int warp = static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + warp_in_block;
    const int row0   = warp * kR;
    const int token0 = static_cast<int>(blockIdx.y) * kT;
    if (static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock * kR >= rows) { return; }
    const bool active = row0 < rows;

    __shared__ __nv_bfloat16 staged[kT][kGemvWarpsPerBlock * kR];

    float accumulator[kR][kT];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kT; ++t) { accumulator[r][t] = 0.0f; }
    }

    // Per-lane activation offset inside the group: fixed for the kernel's lifetime.
    const std::int32_t act0 = ptq1_lane_act0(lane);

    for (int group = 0; active && group < groups_per_row; ++group) {
        const std::int32_t base = group * kPTQ1GroupK;

        // Activations: one load per (token, lane) shared by every row this warp owns.
        float2 low[kT];
        float2 high[kT];
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            low[t]  = make_float2(0.0f, 0.0f);
            high[t] = make_float2(0.0f, 0.0f);
            const int token = token0 + t;
            if (token < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(token) * groups_per_row * kPTQ1GroupK;
                low[t]  = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + act0));
                high[t] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + act0 + 2));
            }
        }

#pragma unroll
        for (int r = 0; r < kR; ++r) {
            const int row = row0 + r;
            if (row < rows) {
                const std::uint8_t* code_row =
                    codes + static_cast<std::int64_t>(row) * groups_per_row * kPTQ1CodeBytes;
                const std::uint8_t* high_row =
                    highs + static_cast<std::int64_t>(row) * groups_per_row * kPTQ1HighBytes;
                const std::uint8_t* scale_row =
                    scales + static_cast<std::int64_t>(row) * groups_per_row *
                                 kGemvScaleBytesPerGroup;
                const PTQ1LaneSlice slice = ptq1_lane_decode(code_row, high_row, lane, group);
                const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    if (token0 + t < tokens) {
                        const float dot = fmaf(slice.w0, low[t].x,
                                               fmaf(slice.w1, low[t].y,
                                                    fmaf(slice.w2, high[t].x,
                                                         slice.w3 * high[t].y)));
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
            if (lane == 0 && active) {
                staged[t][warp_in_block * kR + r] = __float2bfloat16_rn(value);
            }
        }
    }
    __syncthreads();

    constexpr int kRowsPerBlock = kGemvWarpsPerBlock * kR;
    const int row_base          = static_cast<int>(blockIdx.x) * kRowsPerBlock;
#pragma unroll
    for (int i = threadIdx.x; i < kT * kRowsPerBlock; i += kGemvWarpsPerBlock * 32) {
        const int t     = i / kRowsPerBlock;
        const int r     = i % kRowsPerBlock;
        const int token = token0 + t;
        const int row   = row_base + r;
        if (token < tokens && row < rows) {
            out[static_cast<std::int64_t>(token) * out_row_stride + row] = staged[t][r];
        }
    }
}

} // namespace ninfer::ops::detail
