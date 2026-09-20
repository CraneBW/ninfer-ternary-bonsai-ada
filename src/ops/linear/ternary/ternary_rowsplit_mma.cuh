#pragma once

// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.

// PQ2_0 RowSplit x BF16 tensor-core GEMM, for the prefill shape.
//
// out[rows, cols] = W[rows, k] * x[k, cols]
//
// Same physical skeleton as q4_rowsplit_gemm_mma.cuh: a CTA owns a BlockRows x BlockCols output
// tile and walks K a tile at a time, staging raw codes and fp16 scales through a cp.async pipeline,
// decoding them to BF16 in shared memory, then consuming them with m16n8k16 BF16 MMA.
//
// Why this path exists. The SIMT prefill GEMV spends four FMA instructions per weight, which pins
// it to the ALU issue pipe -- NCU on the 248320-row head: ALU the top utilizer, 95.47% occupancy,
// 909e6 instructions against a 1.32 ms pure-issue floor, and no occupancy left to buy. MMA removes
// the multiplies (one m16n8k16 carries 2048 MACs) but not the decode, so the decode is what has to
// get cheap.
//
// Decoding a byte. PQ2_0 packs four two-bit codes per byte, and the codes are {0,1,2} meaning
// {-1,0,+1}. Two shifts and two masks split a byte into two nibble pairs, each pair separated into
// one byte per code; __byte_perm then interleaves half(1024.0) so a single __hsub2 by half(1025.0)
// lands on exactly {-1,0,+1}. Six instructions per byte, 1.5 per weight, against the SIMT path's
// per-weight FMA. The bias must be 1024, not 2048: half's ULP at 2048 is 2.0 and 2049 would round.
// Selectors must stay in 0..7, because prmt reads 8..15 as its sign-replicate modes.
//
// K tile = 64, half a group. The ternary group is 128 wide, which is what the scale is per. A
// 128-wide tile works and was measured, but its activation staging alone is 64 KiB at BlockCols
// 128, which forces dynamic shared memory (>48 KiB), one CTA per SM, and 16.7% occupancy. NCU on
// that shape: DRAM 9.6%, tensor 28%, issue 23% -- latency-bound for want of resident warps.
// Halving the tile halves the staging and buys a second CTA; measured 48.3 -> 37.0 ms on the
// 248320x5120 head at T=1024. It also keeps every schedule under the static 48 KiB budget, which
// matters because cudaFuncSetAttribute is illegal inside a CUDA graph capture and prefill tiles
// are captured.
//
// The scale is folded into the decoded value rather than applied after accumulation. That costs
// one bit of mantissa relative to the SIMT path (fp16 scale rounded to bf16), so the numerics are
// NOT bit-identical to the reference kernel: max relative deviation 5-6e-3 against the reference
// kernel's 2.5-2.9e-3, both dominated by the bf16 output rounding. Engine-side the two paths land
// within 0.015% on perplexity (9.69192 against the reference kernel's 9.6905 over 110 windows).
//
// What was tried and did not help, so it is not here:
//
//   - Double-buffered A/B fragments, so the ldmatrix for k-step ki+1 issues before the multiplies
//     of step ki (the trick q4_rowsplit_gemm_mma.cuh uses). Measured exactly flat: 37.024 ms
//     against 37.029 on the 248320x5120 head. The ldmatrix is evidently not what the warps wait
//     on. NCU says the wait is on the execution pipe (4.7 of the 11.83 cycles between issues), and
//     at 4 warps per scheduler that pipe is the tensor unit itself.
//   - BlockRows 96 instead of 64, which cuts the activation traffic by a third. 34.8 ms, 6%.
//   - 16 and 32 warps per CTA: 42.0 and 61.8 ms. More warps make it worse, not better.
//
// The standing limit: 248320x5120 at T=1024 runs 37.0 ms with tensor at 37.9% of its pipe and
// every other pipe under 42%. Instruction count is not the constraint (issue slots 23%), nor is
// DRAM (12.9%). It is latency across a chain that is only 8 independent mma deep per warp, and
// nothing tried so far lengthens that chain.

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// XOR swizzle for a [rows][BlockK] BF16 tile -- identical in spirit to q4_mma_swizzle_k64, and
// valid for any BlockK that is a multiple of 8.
__device__ __forceinline__ int ternary_mma_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// One byte of four 2-bit codes -> two BF16 pairs, values already scaled.
//
//   byte layout   c0 | c1<<2 | c2<<4 | c3<<6      (c in {0,1,2} meaning {-1,0,+1})
//   a             c0 in byte 0, c1 in byte 1
//   c             c2 in byte 0, c3 in byte 1
//   __byte_perm   interleaves each with half(1024.0) -> {1024+c, 1024+c}
//   __hsub2       by half(1025.0) -> exactly {c-1, c-1}
//
// half(1024.0) is 0x6400 and the exponent puts the ULP at 1.0, so 1024, 1025 and 1026 are all
// exactly representable; that is why the bias is 1024 and not 2048 (whose ULP is 2.0 and would
// round 2049).
__device__ __forceinline__ void ternary_mma_decode_byte(std::uint8_t byte, __half2 scale2,
                                                        unsigned& lo_bits, unsigned& hi_bits) {
    const unsigned v = static_cast<unsigned>(byte);
    const unsigned a = (v & 0x03u) | ((v & 0x0Cu) << 6);       // c0 -> byte 0, c1 -> byte 1
    const unsigned c = ((v >> 4) & 0x03u) | ((v & 0xC0u) << 2); // c2 -> byte 0, c3 -> byte 1

    constexpr unsigned kMagic = 0x64006400u; // half 1024.0 in both lanes
    constexpr unsigned kBias  = 0x64016401u; // half 1025.0 in both lanes

    // Selectors must stay in 0..7: prmt treats 8..15 as the sign-replicate modes, not as an
    // index into the second operand. 0x5150 = {a[0], b[1], a[1], b[1]}.
    const unsigned wa = __byte_perm(a, kMagic, 0x5150u); // byte order: c0, 0x64, c1, 0x64
    const unsigned wc = __byte_perm(c, kMagic, 0x5150u);

    const __half2 bias = *reinterpret_cast<const __half2*>(&kBias);
    __half2 ha = *reinterpret_cast<const __half2*>(&wa);
    __half2 hc = *reinterpret_cast<const __half2*>(&wc);
    ha = __hmul2(__hsub2(ha, bias), scale2);
    hc = __hmul2(__hsub2(hc, bias), scale2);

    const __nv_bfloat162 ba = __float22bfloat162_rn(__half22float2(ha));
    const __nv_bfloat162 bc = __float22bfloat162_rn(__half22float2(hc));
    lo_bits = *reinterpret_cast<const unsigned*>(&ba);
    hi_bits = *reinterpret_cast<const unsigned*>(&bc);
}

template <int BlockRows_, int BlockCols_, int BlockK_, int WarpRows_, int WarpCols_,
          int PipelineStages_, int LaunchBoundsMinBlocks_>
struct TernaryMmaSchedule {
    static constexpr int kBlockRows = BlockRows_;
    static constexpr int kBlockCols = BlockCols_;
    static constexpr int kBlockK    = BlockK_;
    static constexpr int kWarpRows  = WarpRows_;
    static constexpr int kWarpCols  = WarpCols_;

    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows    = kWarpRows / 16;
    static constexpr int kMmaCols    = kWarpCols / 8;
    static constexpr int kMmaKSteps  = kBlockK / 16;
    static constexpr int kGroupK     = 128; // the ternary quant group is 128 wide
    static constexpr int kGroupBytes = 32;

    // Codes staged per row for one K tile, and the scale is the group's.
    static constexpr int kCodeBytesPerTile = kBlockK / 4;
    static constexpr int kScaleBytes       = 2;

    // How many rows one warp decodes per step: 32 lanes / bytes per row.
    static constexpr int kRowsPerDecodeStep = 32 / kCodeBytesPerTile;

    // Staged shared memory: the decoded BF16 A tile, the BF16 activation pipeline, the raw codes
    // and the fp16 scales. Kept as a class constant so the kernel can size a static array with it.
    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * 2 + kPipelineStages * kBlockCols * kBlockK * 2 +
        kPipelineStages * kBlockRows * kCodeBytesPerTile + kPipelineStages * kBlockRows * kScaleBytes;

    static_assert(kSharedBytes <= 48 * 1024, "ternary MMA staged shared exceeds the 48 KiB budget");
    static_assert(kGroupK % kBlockK == 0, "K tile must divide the 128-wide ternary group");
    // The codes are staged with cp_async<16>, so a K tile has to carry whole 16-byte chunks. A
    // half-group tile is exactly one chunk; anything smaller would stage nothing at all.
    static_assert(kCodeBytesPerTile >= 16 && (kCodeBytesPerTile % 16) == 0,
                  "a K tile must stage whole 16-byte code chunks");
    static_assert((32 % kCodeBytesPerTile) == 0,
                  "a warp must split evenly over the K tile's code bytes");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0);
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0);
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8);
    static_assert(kWarps >= 1 && kThreads <= 1024);
};

// clang-format off
template <class Schedule_, bool Full>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kLaunchBoundsMinBlocks)
void ternary_rowsplit_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k,
    std::int32_t out_row_stride) {
    // clang-format on
    using Schedule   = Schedule_;
    constexpr int BM = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockCols;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpCols;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaCols;
    constexpr int KSUB   = Schedule::kMmaKSteps;
    constexpr int S      = Schedule::kPipelineStages;
    constexpr int CBT    = Schedule::kCodeBytesPerTile;

    // Static, not dynamic: the schedule is fixed at compile time and the budget stays under the
    // engine's 48 KiB convention, which keeps cudaFuncSetAttribute out of the picture entirely --
    // that call is illegal inside a CUDA graph capture, and prefill tiles are captured.
    __shared__ __align__(16) std::uint8_t smem_raw[Schedule::kSharedBytes];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Bs = As + BM * BK;
    std::uint8_t* Cr  = reinterpret_cast<std::uint8_t*>(Bs + S * BN * BK);
    std::uint8_t* Sr  = Cr + S * BM * CBT;

    const int groups_per_row = padded_k / 128;
    const int k_tiles        = padded_k / BK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    float accum[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            accum[mi][ni][0] = 0.0f;
            accum[mi][ni][1] = 0.0f;
            accum[mi][ni][2] = 0.0f;
            accum[mi][ni][3] = 0.0f;
        }
    }

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst = &Bs[(stage * BN + local_col) * BK +
                            ternary_mma_swizzle(local_col, k8 * 8)];
            if constexpr (Full) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < cols && kk + 8 <= k) {
                    cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        }
    };

    // A K tile is one or two halves of a 128-wide group, so its codes start at a byte offset
    // inside the group and every row of the tile shares that group's single scale.
    auto stage_quant = [&](int stage, int k_tile) {
        const int group0   = (k_tile * BK) / Schedule::kGroupK;
        const int byte0    = ((k_tile * BK) % Schedule::kGroupK) / 4;
        constexpr int kChunks = CBT / 16;
#pragma unroll 1
        for (int item = tid; item < BM * kChunks; item += Schedule::kThreads) {
            const int local_row = item / kChunks;
            const int chunk     = item - local_row * kChunks;
            const int row       = row0 + local_row;
            auto* dst           = &Cr[(stage * BM + local_row) * CBT + chunk * 16];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(row) * groups_per_row + group0;
                cp_async<16>(dst, &codes[gi * Schedule::kGroupBytes + byte0 + chunk * 16]);
            } else {
                if (row < rows) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(row) * groups_per_row + group0;
                    cp_async<16>(dst, &codes[gi * Schedule::kGroupBytes + byte0 + chunk * 16]);
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        }

#pragma unroll 1
        for (int local_row = tid; local_row < BM; local_row += Schedule::kThreads) {
            const int row = row0 + local_row;
            auto* dst     = &Sr[(stage * BM + local_row) * Schedule::kScaleBytes];
            if (row < rows) {
                const std::int64_t gi = static_cast<std::int64_t>(row) * groups_per_row + group0;
                *reinterpret_cast<std::uint16_t*>(dst) =
                    *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
            } else {
                *reinterpret_cast<std::uint16_t*>(dst) = 0;
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    // One warp decodes whole rows of A: lane l owns code byte l, which is weights 4l..4l+3. When
    // the K tile is half a group there are only 16 bytes per row, so the warp splits into two
    // 16-lane halves and covers two rows per step instead of idling half its lanes.
    auto decode_weight = [&](int stage) {
        constexpr int kStep = Schedule::kRowsPerDecodeStep;
        for (int base = warp * kStep; base < BM; base += Schedule::kWarps * kStep) {
            const int local_row = base + lane / CBT;
            const int byte_col  = lane % CBT;
            if (local_row >= BM) { continue; }
            const std::uint8_t* codes_row = &Cr[(stage * BM + local_row) * CBT];
            const std::uint16_t scale_bits =
                *reinterpret_cast<const std::uint16_t*>(&Sr[(stage * BM + local_row) *
                                                            Schedule::kScaleBytes]);
            const __half2 scale2 = __half2half2(__ushort_as_half(scale_bits));

            unsigned lo_bits = 0, hi_bits = 0;
            ternary_mma_decode_byte(codes_row[byte_col], scale2, lo_bits, hi_bits);

            const int base_col = 4 * byte_col;
            auto* dst          = &As[local_row * BK];
            *reinterpret_cast<unsigned*>(&dst[ternary_mma_swizzle(local_row, base_col)]) = lo_bits;
            *reinterpret_cast<unsigned*>(&dst[ternary_mma_swizzle(local_row, base_col + 2)]) =
                hi_bits;
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage);
        __syncthreads();

        unsigned a_frag[MT][4];
        unsigned b_frag[NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = ks + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&As[row * BK + ternary_mma_swizzle(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = ks + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs[(stage * BN + row) * BK +
                                          ternary_mma_swizzle(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                             accum[mi][ni][3], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2],
                             a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + warp_row * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = accum[mi][ni];
            auto store            = [&](int row, int col, float value) {
                out[static_cast<std::int64_t>(col) * out_row_stride + row] =
                    __float2bfloat16_rn(value);
            };
            if constexpr (Full) {
                store(output_row0, output_col0, values[0]);
                store(output_row0, output_col1, values[1]);
                store(output_row1, output_col0, values[2]);
                store(output_row1, output_col1, values[3]);
            } else {
                if (output_row0 < rows) {
                    if (output_col0 < cols) { store(output_row0, output_col0, values[0]); }
                    if (output_col1 < cols) { store(output_row0, output_col1, values[1]); }
                }
                if (output_row1 < rows) {
                    if (output_col0 < cols) { store(output_row1, output_col0, values[2]); }
                    if (output_col1 < cols) { store(output_row1, output_col1, values[3]); }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
