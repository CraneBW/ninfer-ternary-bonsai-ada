// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.

#pragma once

// PQ2_0 RowSplit x BF16 tensor-core GEMV for the SPECULATIVE VERIFY pass (T = draft + 1, so 2..4).
//
// Why a second tensor-core entry point. The prefill kernel in ternary_rowsplit_mma.cuh tiles the
// token axis at 128, which is right when T is a prefill chunk and wrong when T is 3: a 128-wide
// output tile would be 97% empty. What the verify pass needs is the opposite shape -- all of K
// inside one CTA, so the weights are read exactly once, with the token axis kept tiny.
//
// Skeleton follows q4_small_t_mma.cuh: the weight is the A operand and the activation is the B
// operand, the CTA owns RowsPerCta output rows, and each of the eight warps takes one slice of K
// and reduces at the end with a shared-memory tree.
//
// Why the tensor cores matter here. The SIMT tile kernel this replaces is not slow because T is 3
// -- measured, its cost is flat in T (1.572 ms at T=1 against 1.719 ms at T=4 on the 248320-row
// head) because the weights are read once either way. It is slow because it leaves the card at
// 196 GB/s, 31% of the measured ceiling, with FMA and decode both crowding the issue slots. On
// that same head NCU put it at 95.47% achieved occupancy with ALU as the top pipe and no occupancy
// left to buy. Tensors move the multiplies off the ALU pipe; that is the whole change.
//
// The one slice of the prefill kernel that is better here: because a warp takes exactly one
// 128-wide quant group, the scale can be applied AFTER the K reduction, in fp32, instead of being
// folded into each decoded weight. So the A operand is exactly {-1,0,+1} in bf16 -- no rounding at
// all -- where the prefill path pays one mantissa bit for the same thing.

#include "ops/common/mma.cuh"
#include "ops/linear/ternary/ternary_epilogue.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One code byte -> two weights, exact, already in bf16.
//
// Differs from ternary_mma_decode_byte only in that it returns one pair instead of two and leaves
// the group scale to the caller. The nibble select is by lane parity, because a m16n8k16 A
// fragment wants columns 2l..2l+1 while a PQ2_0 byte holds columns 4b..4b+3: lanes 2k and 2k+1
// read the same byte and take opposite halves. `nibble` is that parity.
//
// Two things here are load-bearing and easy to undo by accident:
//
// 1. The nibble select is a variable shift, not a ternary. Both halves assemble the same way --
//    shift the wanted two-bit field down to bits 1:0, copy it to 1:0, and copy its neighbour to
//    9:8 -- so `v >> 4*nibble` covers both and the branch disappears. Written as a conditional the
//    compiler emits both arms plus a SEL, which is one extra instruction on a path that runs four
//    times per K step.
//
// 2. The magic is bf16 at exponent 7, not fp16 and not bf16 at exponent 10. bf16 stores seven
//    mantissa bits, so at exponent 7 its ULP is exactly 1: splicing the code into the mantissa's
//    low bits gives 128 + c, whose bias byte is 0x00 -- so the splice needs no OR to prime it.
//    One bf16 subtract of 129 gives c - 1 in {0,+/-1}, exactly, and the pair is already the width
//    the mma wants. The prefill kernel's fp16 version instead lands on {0,+/-1} in half precision
//    and pays two converts to get to bf16; the exponent-10 version lands on 8*(c-1) and pays an OR
//    per pair plus a 1/8 premultiply on the scale. This costs neither.
//
// Worth ~8% on the verify shapes. The whole decode is 24% of the kernel -- measured by rebuilding
// with the arithmetic compiled out, which puts 34816x5120 at 644 GB/s, this card's measured read
// ceiling -- and this is what came off cheaply. What is left is the PRMT and the assembly.
__device__ __forceinline__ unsigned ternary_small_t_decode_pair(std::uint32_t v, unsigned nibble) {
    const unsigned t = v >> (nibble * 4);
    const unsigned a = (t & 0x03u) | ((t & 0x0Cu) << 6);
    constexpr unsigned kMagic = 0x43004300u; // bf16 0x4300 in both lanes: exponent of 128
    constexpr unsigned kBias  = 0x43014301u; // bf16 129.0 in both lanes
    const unsigned w          = __byte_perm(a, kMagic, 0x5150u);
    const __nv_bfloat162 bias = *reinterpret_cast<const __nv_bfloat162*>(&kBias);
    const __nv_bfloat162 h = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&w), bias);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Rows per CTA is the lever that sets this kernel's load mix, and the ratio is not guessable from
// the shape. Per K group a CTA stages
//
//     codes       = kRowsPerCta * kGroupK / 4        bytes, from DRAM (each row is read once)
//     activations = kGroupK * live_cols * 2          bytes, from L2 (every CTA reads the same slice)
//
// so activations/codes = 8 * live_cols / kRowsPerCta. At 16 rows and 4 live tokens that is 2.0 --
// the activations are TWICE the weight traffic -- and measured, four shapes with wildly different
// grids (2176, 320, 384, 256 CTAs) and group counts (5, 17, 5, 5) all land on the same 1.3-1.6 TB/s
// of COMBINED traffic. That is an L2 ceiling, not the measured 637 GB/s DRAM one, and it shows up
// as DRAM rates of only 453-571 GB/s. Raising the row block to 32 takes the ratio to 1.0, which
// puts the L2 ceiling at ~2x1.45 TB/s and leaves DRAM as the binding constraint again.
//
// The row block is a kernel template parameter rather than a member of a templated schedule
// because templating the struct makes cudafe++ die outright under this build's -rdc=true
// ("memory region allocation must not occur after front end processing has ended").
struct TernarySmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kTileKPerWarp      = 128; // exactly one ternary quant group
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kCodeBytesPerGroup = 32;
    static constexpr int kCodeBytesPerWarp  = kTileKPerWarp / 4;
    static constexpr int kCodeRowBytes      = kGroupK / 4;
    // A row holds all eight warps' groups (8 x 32 bytes = 256). That is 64 four-byte words, and
    // 64 % 32 == 0, so every row starts on the same bank: the eight lanes with distinct gid read
    // the same byte column of eight different rows and collide eight ways. Padding by 16 bytes
    // (keeping the 16-byte alignment cp_async needs) makes the row stride 272 = 4 mod 32 words,
    // which spreads the eight rows over eight distinct banks. NCU before this: L1/TEX throughput
    // 95.02% with DRAM at 43.89%.
    static constexpr int kCodeStride = kCodeRowBytes + 16;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kMmaKSteps         = kTileKPerWarp / 16;

    static_assert(kGroupK % 128 == 0, "the K slice per warp must be a whole number of groups");
    static_assert(kRowsPerCta % kKWarps == 0, "rows per CTA must divide evenly over the warps");
    static_assert(kCodeBytesPerWarp % 16 == 0, "group staging moves whole 16-byte chunks");
};

// TileCols is the token tile; mma n is 8, so 8 is the smallest useful value and is what the verify
// pass (T <= 4) uses. A larger tile amortises the weight decode over more tokens.
//
// Epilogue is the value transform applied at each store. It is defaulted, and that default is
// load-bearing: the three append-only row-block instantiations the launcher selects between
// (rows 16 / 32 / 48) are spelled <8, kMinBlocks, kRows> and have to keep compiling AND keep
// meaning exactly what they meant before the parameter existed. Same trick, same intent as the
// trailing `bool kPtq1 = false` on TernaryS8Storage. The check that the default is really a no-op
// and not merely a small change is the numeric gate the port already runs: ninfer-perplexity
// --text <corpus> --context 512 --stride 256 must read 9.693396 to the digit, because the identity
// is inlined to its own argument and ptxas is expected to see the pre-parameter program.
//
// The default is also why the parameter is the LAST one and the value argument is last in the
// parameter list -- any earlier position changes what an existing <8, 4, 32> means.
template <int TileCols, int LaunchBoundsMinBlocks,
          int kRowsPerCta = TernarySmallTSchedule::kRowsPerCta,
          class Epilogue  = TernaryIdentityEpilogue>
__launch_bounds__(TernarySmallTSchedule::kThreads, LaunchBoundsMinBlocks) __global__
void ternary_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::uint8_t* __restrict__ codes,
                                const std::uint8_t* __restrict__ scales,
                                __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                std::int32_t k, std::int32_t tokens,
                                std::int32_t out_row_stride, Epilogue epilogue = {}) {
    using Schedule = TernarySmallTSchedule;
    static_assert(TileCols >= 8 && TileCols <= 32 && (TileCols % 8) == 0);
    static_assert(kRowsPerCta % TernarySmallTSchedule::kKWarps == 0,
                  "rows per CTA must divide evenly over the warps");

    constexpr int kWarps     = Schedule::kKWarps;
    constexpr int kTileK     = Schedule::kTileKPerWarp;
    constexpr int kGroupK    = Schedule::kGroupK;
    constexpr int kNt        = TileCols / 8;
    constexpr int kMmaKSteps = Schedule::kMmaKSteps;
    // mma m is 16, so one A-fragment pass covers exactly 16 output rows; a wider CTA owns several
    // such row blocks and runs the SAME staged activations against each of them. That reuse is the
    // entire point of the parameter -- activations are L2-resident and cost more of the load pipe
    // than the codes do.
    constexpr int kRowBlock   = 16;
    constexpr int kRowBlocks  = kRowsPerCta / kRowBlock;
    // Derived from the template parameter, NOT from Schedule: the schedule's own
    // kRowsPerLoaderWarp is pinned at 16/8 and silently stages only half the rows when the
    // kernel is instantiated at a wider row block. That reads as a 1.6x speedup with wrong
    // output, which is exactly the shape of bug the harness comparison is here to catch.
    constexpr int kRowsPerLoaderWarp = kRowsPerCta / kWarps;
    static_assert((kRowsPerCta % kRowBlock) == 0, "rows per CTA must be a whole number of mma m-tiles");

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][Schedule::kCodeStride];
            __nv_bfloat16 activations[kWarps][TileCols * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;
        float partial[kRowBlocks * kWarps * kNt * 32 * 4];
    };
    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int gid   = lane >> 2;
    const int lid   = lane & 3;
    const int row0  = static_cast<int>(blockIdx.x) * kRowsPerCta;
    const int k_groups = k / kGroupK;
    const int groups_per_row = k / 128;

    // Columns at or past the real token count are never stored by the epilogue, so staging them
    // every group is pure waste -- and it is the larger half of the staging. A warp stages
    // TileCols columns of its 128-wide K slice per group, so at T=3 the mma's eight-wide n axis
    // makes five of those columns dead: 10 KB of the 16 KB a CTA moves per group, and it is
    // re-moved on every one of the k/1024 group iterations. Measured on the 5120x17408 shape,
    // small_t16 costs 30% more than small_t8 while doing identical work, which is this term.
    //
    // The dead columns still have to hold *something* the mma can read, so they are zeroed once
    // before the group loop. They are outside the staging loop from then on, so the zero survives
    // every group and no stale value from a previous kernel can reach an mma operand.
    const int live_cols = tokens < TileCols ? tokens : TileCols;
    constexpr int kItemsPerSplit = TileCols * (kTileK / 8);
    if (live_cols < TileCols) {
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            if (col >= live_cols) {
                *reinterpret_cast<int4*>(
                    &x_shared[warp][col * kTileK + ternary_mma_swizzle(col, k8 * 8)]) =
                    make_int4(0, 0, 0, 0);
            }
        }
    }

    const auto stage_x = [&](int group_k0) {
        const int items = live_cols * (kTileK / 8);
#pragma unroll 1
        for (int item = lane; item < items; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + ternary_mma_swizzle(col, k8 * 8)];
            const int kk  = group_k0 + warp * kTileK + k8 * 8;
            if (kk + 8 <= k) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
    };

    // Codes are shared by every warp: each warp decodes a different 128-wide group out of the same
    // staged rows, so ONE row of code_shared holds all eight groups (kCodeRowBytes = 256) and the
    // warps split the staging by row rather than by column.
    const auto stage_weight = [&](int group_k0) {
        constexpr int kChunksPerRow = Schedule::kCodeRowBytes / 16;
#pragma unroll
        for (int item = lane; item < kRowsPerLoaderWarp * kChunksPerRow; item += 32) {
            const int row_item = item / kChunksPerRow;
            const int chunk    = item - row_item * kChunksPerRow;
            const int row      = warp * kRowsPerLoaderWarp + row_item;
            const int grow     = row0 + row;
            auto* dst          = &code_shared[row][chunk * 16];
            if (grow < rows) {
                cp_async<16>(dst, &codes[static_cast<std::int64_t>(grow) * groups_per_row *
                                            Schedule::kCodeBytesPerGroup +
                                        group_k0 / 4 + chunk * 16]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        // One 16-byte load covers the eight consecutive groups the eight warps each need.
        for (int row = tid; row < kRowsPerCta; row += Schedule::kThreads) {
            const int grow = row0 + row;
            auto* dst      = &scale_shared[row][0];
            if (grow < rows && group_k0 / 128 + kWarps <= groups_per_row) {
                cp_async<16>(dst, &scales[(static_cast<std::int64_t>(grow) * groups_per_row +
                                           group_k0 / 128) *
                                          2]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    float acc[kRowBlocks][kNt][4] = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int gi = 0; gi < k_groups; ++gi) {
        const int group_k0      = gi * kGroupK;

#pragma unroll
        for (int rb = 0; rb < kRowBlocks; ++rb) {
        const int r0            = rb * kRowBlock;
        float group_acc[kNt][4] = {};

        // This lane's two code rows and its byte column within them, hoisted out of the K loop so
        // that with the K axis unrolled every offset is a compile-time constant.
        //
        // The decode below is 24% of this kernel, measured by rebuilding with the arithmetic
        // compiled out (gate_up then runs at 644 GB/s, the card's measured ceiling -- so the
        // memory path is not what is left). Two ways to take more of it were tried and both lost:
        // slicing whole words in registers instead of loading bytes, which trades two LDS for four
        // shifts and costs 6% (0.055 against 0.052 on 5120x17408), and hoisting these row pointers
        // still further, which changed nothing at all because nvcc had already done it. What
        // remains is the byte assembly plus the perm, four times per K step, and it is issue-bound.
        const std::uint8_t* const code_lo = &code_shared[r0 + gid][0];
        const std::uint8_t* const code_hi = &code_shared[r0 + gid + 8][0];
        const unsigned nibble = lid & 1;
        const int coff0       = warp * Schedule::kCodeBytesPerWarp + (lid >> 1);

#pragma unroll
        for (int ks = 0; ks < kMmaKSteps; ++ks) {
            // A fragment: lane (gid, lid) needs rows gid and gid+8, columns 2*lid (already in the
            // low or high nibble of the byte) and 2*lid+8 (two bytes further along the row).
            // The warp offset is what selects this warp's own group out of the shared row.
            const int coff     = coff0 + ks * 4;
            const unsigned af0 = ternary_small_t_decode_pair(code_lo[coff], nibble);
            const unsigned af1 = ternary_small_t_decode_pair(code_hi[coff], nibble);
            const unsigned af2 = ternary_small_t_decode_pair(code_lo[coff + 2], nibble);
            const unsigned af3 = ternary_small_t_decode_pair(code_hi[coff + 2], nibble);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&x_shared[warp][br * kTileK +
                                                    ternary_mma_swizzle(br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        // The warp owned exactly one group, so its scale is a single fp32 multiply after the K
        // reduction -- and the weights it multiplied against were exact.
        const float top_scale =
            __half2float(__ushort_as_half(scale_shared[r0 + gid][warp]));
        const float bot_scale =
            __half2float(__ushort_as_half(scale_shared[r0 + gid + 8][warp]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[rb][nt][0] = fmaf(group_acc[nt][0], top_scale, acc[rb][nt][0]);
            acc[rb][nt][1] = fmaf(group_acc[nt][1], top_scale, acc[rb][nt][1]);
            acc[rb][nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[rb][nt][2]);
            acc[rb][nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[rb][nt][3]);
        }
        } // row block

        if (gi + 1 < k_groups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    // partial is indexed [rb][warp][kNt][lane], so each row block reduces over the same eight
    // warps independently. The tree is otherwise untouched -- only the slot addressing moved.
    const auto pslot = [&](int rb, int w, int nt) {
        return partial + (((rb * kWarps + w) * kNt + nt) * 32 + lane) * 4;
    };

#pragma unroll
    for (int rb = 0; rb < kRowBlocks; ++rb) {
        if ((warp & 1) != 0) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                store_vec(pslot(rb, warp, nt),
                          make_float4(acc[rb][nt][0], acc[rb][nt][1], acc[rb][nt][2],
                                      acc[rb][nt][3]));
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int rb = 0; rb < kRowBlocks; ++rb) {
        if ((warp & 1) == 0) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float4 partner = load_vec<float4>(pslot(rb, warp + 1, nt));
                acc[rb][nt][0] += partner.x;
                acc[rb][nt][1] += partner.y;
                acc[rb][nt][2] += partner.z;
                acc[rb][nt][3] += partner.w;
                if (warp != 0) {
                    store_vec(pslot(rb, warp, nt),
                              make_float4(acc[rb][nt][0], acc[rb][nt][1], acc[rb][nt][2],
                                          acc[rb][nt][3]));
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[rb][nt][0], acc[rb][nt][1], acc[rb][nt][2],
                                     acc[rb][nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value = load_vec<float4>(pslot(rb, split, nt));
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = nt * 8 + 2 * lid;
            const int row_lo = row0 + rb * kRowBlock + gid;
            const int row_hi = row0 + rb * kRowBlock + gid + 8;
            // Every store goes through the epilogue, and each call is INSIDE the store's own bounds
            // guard rather than hoisted above it: the identity ignores its arguments, but the
            // residual epilogue this parameter exists for dereferences the tensor it is about to
            // overwrite, so an unguarded call would touch a row or a token the store may not write.
            // The (row, token) order and the fact that out_row_stride is the stride an epilogue
            // would rebuild the same address from are the contract spelled out in
            // ternary_epilogue.cuh.
            if (col0 < tokens && row_lo < rows) {
                out[static_cast<std::int64_t>(col0) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(epilogue.apply(row_lo, col0, sum.x));
            }
            if (col0 < tokens && row_hi < rows) {
                out[static_cast<std::int64_t>(col0) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(epilogue.apply(row_hi, col0, sum.z));
            }
            if (col0 + 1 < tokens && row_lo < rows) {
                out[static_cast<std::int64_t>(col0 + 1) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(epilogue.apply(row_lo, col0 + 1, sum.y));
            }
            if (col0 + 1 < tokens && row_hi < rows) {
                out[static_cast<std::int64_t>(col0 + 1) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(epilogue.apply(row_hi, col0 + 1, sum.w));
            }
        }
        } // row block
    }
}

} // namespace ninfer::ops::detail
