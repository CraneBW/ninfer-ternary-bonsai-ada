// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_epilogue.cuh"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_s8.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh"
#include "ops/linear/ternary/ternary_small_t_plan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// Which kernel serves prefill (T >= 5).
//
//   unset / mma  tensor-core path (ternary_rowsplit_mma.cuh) -- the measured default
//   block        token-blocked SIMT GEMV, kept as the fallback and as an A/B arm
//   ref          the correctness-first reference kernel, the oracle for both
//
// A two-run A/B against `ref` is what qualifies a new path engine-side: it is the only comparison
// that can catch a token-tile or activation-layout error, because at T == 1 the token-major and
// row-major activation layouts coincide exactly. Read once, because the choice decides which
// kernel enters a captured CUDA graph.
enum class PrefillRoute { Mma, Block, Reference };

PrefillRoute prefill_route() {
    static const PrefillRoute route = [] {
        const char* value = std::getenv("NINFER_TERNARY_PREFILL");
        if (value == nullptr) { return PrefillRoute::Mma; }
        const std::string text(value);
        if (text == "ref") { return PrefillRoute::Reference; }
        if (text == "block") { return PrefillRoute::Block; }
        return PrefillRoute::Mma;
    }();
    return route;
}


// Decode (T == 1) takes the warp-per-row GEMV for PQ2_0. K is a whole number of 128-groups for
// every width in this model, so that kernel needs no column guard.
void launch_pq2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((w.k % 128) != 0 || x.ne[1] != 1) {
        throw std::invalid_argument("ternary gemv: expected one token and a whole-group K");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    ternary_pq2_gemv_kernel<<<grid, kGemvWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row);
    CUDA_CHECK(cudaGetLastError());
}

// Small-token-tile GEMV: weights are read once for up to 4 tokens, which is what makes the
// speculative verify pass (T = draft + 1) cheap. Falls back to the reference tiled kernel beyond
// that, and for PTQ1_0 / padded-K weights.
void launch_pq2_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, std::int32_t tokens,
                          cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_pq2_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row);
    } else {
        ternary_pq2_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, tokens, out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Storage, class Atom, int kTileT>
void launch_gemm(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                 cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t t    = x.ne[1];
    if (k % Storage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary linear: output row stride is smaller than the tile");
    }
    const std::int32_t groups_per_row = k / Storage::kGroupK;

    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(div_up(t, kTileT)), 1u);
    constexpr dim3 block(Storage::kGroupK, 1u, 1u);

    ternary_rowsplit_gemm_kernel<Storage, Atom, kTileT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, t, groups_per_row, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <int kTileT>
void launch_by_qtype(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::PTQ1_0_G128:
        launch_gemm<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                    stream);
        return;
    case QType::PQ2_0_G128:
        launch_gemm<PQ2RowSplitStorage, PQ2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                  stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("ternary linear: unsupported weight qtype");
}

} // namespace

// A PQ2_0 weight can take the fast family whenever the layout did not pad K past the real width --
// true for every width in this model (5120/6144/10240/17408 are all whole 128-groups), and checked
// here rather than assumed, because these kernels read whole groups without a column guard.
//
// Every one of them walks the activation as x[token * w.k + column], so the x row pitch has to BE
// w.k. That holds for both callers today (the raw path's hidden and the folded activation both
// report w.k as ne[0]), but it is assumed rather than enforced anywhere else, and a mismatch would
// read the wrong rows instead of failing. Checked here so all three routes agree on the gate.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    return w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr && w.padded_shape[1] == w.k &&
           (w.k % 128) == 0 && x.ne[1] >= 1 && x.ne[1] <= max_tokens && x.ne[0] == w.k;
}

void launch_small_t(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                    cudaStream_t stream, std::int32_t rows_override, TernaryS8Scratch scratch);

// Defined next to note_rung, declared here because the split runs inside launch_small_t.
void note_split(std::int32_t rows, std::int32_t k, std::int32_t tokens, std::int32_t row_block,
                std::int32_t slices);

// The row block and the split-K policy live in ternary_small_t_plan.h rather than here: the
// workspace allocator in ternary_dispatch.cpp has to size the split-K reduction buffer for the same
// slice count this file will ask for, and the two agreeing by construction is the whole point.
static_assert(kSmallTSplitGroupK == TernarySmallTSchedule::kGroupK,
              "the plan header's K step and the kernel's must be the same group count");

// A warp takes one whole 128-wide quant group, so K has to hold whole K groups. Declared here
// rather than next to the verify route that first needed it, so both entry points name it.
inline constexpr std::int32_t kSmallTGroupK = 8 * 128;

// Decode (T == 1) takes the tensor-core kernel by default; NINFER_TERNARY_DECODE=gemv restores the
// warp-per-row GEMV as the A/B arm.
//
// The GEMV is the one this path has always used -- one warp per row, no barriers -- but it predates
// the tensor-core verify path and had never been measured against it. It loses. Two like-for-like
// numbers, both under nsys on the en-code fixture: the 34816x5120 shape costs 96.2 us on the GEMV
// against 83.4 on the tensor-core kernel, and the matmul half of a whole forward costs 14.45 ms at
// T=1 against 13.36 ms at T=3 -- the tensor-core kernel does MORE work (three tokens) in LESS time.
// (Compare those two against each other, not a matmul time against a whole-forward time; the
// non-matmul kernels -- rotation, GDN, attention -- add roughly 2 ms either way.) End to end the
// switch is 60.0 -> 65.8 t/s, reproduced across runs; the generated text stays fluent and PPL,
// unchanged at 9.69192, only exercises prefill so it does not cover this path.
//
// Worth noting why it wins at T=1 even though its token tile is eight wide: the kernel's cost is
// governed by weight traffic rather than by token count, so seven empty columns cost almost
// nothing, while the SIMT decode pays four integer instructions per weight on a core that is
// already issue-bound.
bool decode_uses_small_t() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_DECODE");
        return value == nullptr || std::string(value) != "gemv";
    }();
    return enabled;
}

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            // Decode is one token wide, three orders of magnitude below the int8
                            // rung's threshold, so this entry never needs the scratch. The parameter
                            // exists because TernaryLaunch is one function-pointer type for both
                            // entry points.
                            TernaryS8Scratch /*scratch*/) {
    if (gemv_admits(x, w, 1)) {
        if (decode_uses_small_t() && (w.k % kSmallTGroupK) == 0) {
            // One token wide, so the activation each CTA re-stages is a quarter of the verify
            // path's: 32, and not the per-shape table -- that table was measured at T = 4 only.
            // Empty scratch on purpose: split-K was measured on the verify shape, and T = 1 is a
            // different work point (a slice would cut a k loop that is already the whole kernel).
            launch_small_t(x, w, out, out_row_stride, stream, 32, TernaryS8Scratch{});
            return;
        }
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    launch_by_qtype<1>(x, w, out, out_row_stride, stream);
}

// One (kR, kT) instantiation of the blocked GEMV: grid.x tiles the rows in strides of
// warps*kR, grid.y tiles the tokens in kT.
template <int kR, int kT>
void launch_pq2_gemv_tile_block_shape(const Tensor& x, const Weight& w, Tensor& out,
                                      std::int32_t out_row_stride, std::int32_t groups_per_row,
                                      std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock * kR)),
                    static_cast<unsigned>(div_up(tokens, kT)), 1u);
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    ternary_pq2_gemv_tile_block_kernel<kR, kT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row, tokens, out_row_stride);
}

// Blocked GEMV launcher, for prefill. Both block shapes are occupancy levers rather than fixed
// constants, and the measured behaviour on the target card is that the kernel is
// activation-bound, not weight-bound (R=1/kT=8 reached 125.7 t/s at an effective weight
// bandwidth of only 112 GB/s, against 422 GB/s for the T=1 GEMV). So the row block -- which
// amortises activation loads across output rows -- is the stronger knob, and the token block
// then trades weight traffic against register pressure. Both are env-selectable so one build
// can sweep; the defaults are the measured winners.
void launch_pq2_gemv_tile_block(const Tensor& x, const Weight& w, Tensor& out,
                                std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t groups_per_row = w.k / 128;
    const std::int32_t tokens         = x.ne[1];
    static const int rows_block = [] {
        const char* value = std::getenv("NINFER_TERNARY_ROWS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8) ? parsed : 4;
    }();
    static const int token_block = [] {
        const char* value = std::getenv("NINFER_TERNARY_TILE");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 2 || parsed == 4 || parsed == 8) ? parsed : 8;
    }();

    const auto shape = [&](auto rows_tag, auto token_tag) {
        launch_pq2_gemv_tile_block_shape<decltype(rows_tag)::value, decltype(token_tag)::value>(
            x, w, out, out_row_stride, groups_per_row, tokens, stream);
        CUDA_CHECK(cudaGetLastError());
    };
    using std::integral_constant;

    if (rows_block == 1 && token_block == 2) { shape(integral_constant<int, 1>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 1 && token_block == 4) { shape(integral_constant<int, 1>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 1) { shape(integral_constant<int, 1>{}, integral_constant<int, 8>{}); }
    else if (rows_block == 2 && token_block == 2) { shape(integral_constant<int, 2>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 2 && token_block == 4) { shape(integral_constant<int, 2>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 2) { shape(integral_constant<int, 2>{}, integral_constant<int, 8>{}); }
    else if (rows_block == 8 && token_block == 2) { shape(integral_constant<int, 8>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 8 && token_block == 4) { shape(integral_constant<int, 8>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 8) { shape(integral_constant<int, 8>{}, integral_constant<int, 8>{}); }
    else if (token_block == 2) { shape(integral_constant<int, 4>{}, integral_constant<int, 2>{}); }
    else if (token_block == 4) { shape(integral_constant<int, 4>{}, integral_constant<int, 4>{}); }
    else { shape(integral_constant<int, 4>{}, integral_constant<int, 8>{}); }
}

// The speculative verify pass (T = 2..4) gets its own tensor-core entry point, because the prefill
// one tiles the token axis at 128 and would be 97% empty on three tokens. NINFER_TERNARY_VERIFY=tile
// forces the SIMT token-tile GEMV back, as the A/B arm.
bool verify_uses_small_t() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_VERIFY");
        return value == nullptr || std::string(value) != "tile";
    }();
    return enabled;
}

// How many tokens the verify-shaped entry admits. It was 4 because the draft window was 3
// (T = draft + 1) and nothing wider had been tried; the small-T kernel's token tile is 8 wide and
// it already handles a partial tile -- `live_cols = tokens < TileCols ? tokens : TileCols`, dead
// columns zeroed, store gated on `col0 < tokens` -- so 4 was a policy choice, not a kernel limit.
//
// Measured end to end at draft 4 (T = 5), which the old cap pushed onto the blocked GEMV: 28.6 t/s
// against 101.6 at draft 3, while acceptance kept rising (2.08 -> 2.11 tok/round). The cap was the
// only thing standing between draft 4 and the tensor-core kernel.
//
// Clamped to the token tile (8): past it the kernel would stage eight columns and leave the rest of
// the output unwritten, silently.
int verify_token_cap() {
    static const int cap = [] {
        const char* value = std::getenv("NINFER_TERNARY_VERIFY_CAP");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed >= 1 && parsed <= 8) ? parsed : 8;
    }();
    return cap;
}

void launch_small_t(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                    cudaStream_t stream, std::int32_t rows_override, TernaryS8Scratch scratch) {
    const std::int32_t rows = w.n;
    const std::int32_t tokens = x.ne[1];
    const int row_block = small_t_rows_for(rows, rows_override);
    // Split-K. Slices go in blockIdx.z; each slice sums a sub-range of the K steps and writes raw
    // fp32 partials, which ternary_s8_reduce_kernel then adds up. This is NOT bit-identical to the
    // unsplit path -- the K terms are combined in a different order -- so the rollback arm
    // (NINFER_TERNARY_SMALL_T_KSPLIT=0) is part of the gate, not a convenience.
    //
    // The bound the arena declared: partial_floats is what the allocator actually handed over, and
    // the launcher caps its OWN slice count against it rather than assuming it got what it asked
    // for. That is the same contract launch_pq2_mma_s8_tile keeps with the same buffer.
    std::int32_t slices = small_t_slices(rows, w.k, tokens, row_block);
    if (slices > 1) {
        const std::int64_t per_slice = static_cast<std::int64_t>(rows) * tokens;
        const std::int64_t fits = per_slice > 0 && scratch.partial_floats > 0
                                      ? scratch.partial_floats / per_slice
                                      : 0;
        if (slices > fits) { slices = static_cast<std::int32_t>(fits); }
        if (slices < 2 || scratch.partial == nullptr) { slices = 1; }
    }

    const auto* x_ptr   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes   = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales  = static_cast<const std::uint8_t*>(w.scales);
    auto* out_ptr       = static_cast<__nv_bfloat16*>(out.data);
    const auto grid_for = [rows, slices](int block) {
        return dim3(static_cast<unsigned>(div_up(rows, block)), 1u,
                    static_cast<unsigned>(slices));
    };
    const auto args = [&](auto tag) {
        constexpr int kRows = decltype(tag)::value;
        // The epilogue is named here rather than left to the kernel's default, even though the two
        // are the same type today. The launcher is the only place that knows WHICH projection is
        // being computed -- and therefore whether the consumer wants a residual, a silu, or nothing
        // -- so it is the call site the fused epilogue is going to have to be named at. Naming the
        // identity now keeps that change to this expression and leaves the kernel template alone.
        ternary_small_t_mma_kernel<8, (kRows == 16 ? 4 : (kRows == 32 ? 3 : 2)), kRows>
            <<<grid_for(kRows), TernarySmallTSchedule::kThreads, 0, stream>>>(
                x_ptr, codes, scales, out_ptr, rows, w.k, tokens, out_row_stride,
                TernaryIdentityEpilogue{}, slices > 1 ? scratch.partial : nullptr);
    };
    using std::integral_constant;
    switch (row_block) {
    case 16: args(integral_constant<int, 16>{}); break;
    case 48: args(integral_constant<int, 48>{}); break;
    default: args(integral_constant<int, 32>{}); break;
    }
    CUDA_CHECK(cudaGetLastError());

    if (slices > 1) {
        // The split is invisible in the rung trace -- the rung name is still "small_t", and the row
        // block is whatever the policy picked -- so it says so itself rather than leaving a reader
        // to infer it from a speed change. Same NINFER_TERNARY_S8_DEBUG switch as note_rung, and the
        // same budget, so it cannot become a printf in a hot loop by accident.
        note_split(rows, w.k, tokens, row_block, slices);
        // One thread per (token, row) element. The kernel writes with `rows` as the row stride --
        // NOT out_row_stride -- and the reduce reads the same layout back, which is why the two can
        // share one buffer without the caller having to know which stride it was built at. On this
        // route the two are equal anyway (ternary_dispatch_basis_strided passes w.n), but the
        // reduce is written against `rows` regardless so the pair stays consistent if that changes.
        const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
        const int threads = 256;
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
        ternary_s8_reduce_kernel<<<blocks, threads, 0, stream>>>(
            scratch.partial, out_ptr, out_row_stride, rows, tokens, slices);
        CUDA_CHECK(cudaGetLastError());
    }
}

// small_t with a token loop, expressed at the LAUNCHER rather than in the kernel.
//
// The kernel's token tile is eight wide and it has no outer loop -- live_cols = min(tokens, 8) and
// the epilogue guards every store -- so a T wider than eight is served by re-entering it once per
// tile. The reference port puts the loop inside its kernel; this reaches the same place with no
// kernel change and no new correctness surface, at the cost of one extra launch per tile
// (~8 us on a tile that costs milliseconds).
//
// The weight re-staging is the point, not a side effect: re-entering re-reads the weight window,
// which is exactly the reference port's measured model for this rung (small_t == 10.2 ms *
// ceil(T/8) against a wide tile that is flat at ~54 ms). At 8 tokens per pass the two cross at
// T = 41 on their shapes -- but that crossing is between THEIR two kernels, and it is the MMA
// threshold here that gets to decide, not this loop.
//
// Slicing rather than a kernel parameter: x is ne[0]-contiguous, so slicing the token axis gives a
// view whose data pointer is already t0 * k elements in, and out's nb[1] already carries the
// parent's row stride -- which is the stride the launcher is handed. Nothing here has to know the
// width of either.
void launch_small_t_tiled(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, cudaStream_t stream) {
    constexpr std::int32_t kTile = 8;
    const std::int32_t tokens    = x.ne[1];
    for (std::int32_t t0 = 0; t0 < tokens; t0 += kTile) {
        const std::int32_t remaining = tokens - t0;
        const std::int32_t tile      = remaining < kTile ? remaining : kTile;
        // Named, not inline: launch_small_t takes `Tensor& out` and a temporary cannot bind to it.
        Tensor x_tile   = x.slice(1, t0, tile);
        Tensor out_tile = out.slice(1, t0, tile);
        // 32, not the per-shape table: P4 swept the prefill work points and 32 won, and a tile is
        // eight tokens wide, which is twice the width the table was measured at. The empty scratch
        // also keeps split-K off this route: its tiles are 8 tokens wide, which would pass the
        // split's own token gate, and the split was measured on the verify shape rather than here.
        launch_small_t(x_tile, w, out_tile, out_row_stride, stream, 32, TernaryS8Scratch{});
    }
}

// The measured winner of the K=64 sweep on the target card: 64 output rows, a 128-token tile, 8
// warps laid out 2x4, two cp.async stages, two CTAs per SM. It beat 64x64, 128x64, 16-warp and
// 32-warp variants (37.0 ms against 41.4-54.8 on the 248320x5120 head at T=1024).
//
// That sweep was run AT T=1024, where a 128-token tile is full. Below 128 it is not, and the tile
// width is not a throughput knob there -- it is a tax. This kernel's only token split is
// grid.y = div_up(T, BN) (col0 = blockIdx.y * BN) and its outer loop is over K, not over tokens:
// a CTA handed fewer than 128 tokens stages and multiplies the whole 128-wide tile and merely
// guards the STORE. So the fill fraction is T / (128 * ceil(T/128)), and at T=58 that is 45%
// against the 64-wide variant's 91%.
//
// The port's own measurement already contained this number and read it the other way: the
// three-way split that fitted kMmaMinTokens reports "MMA is flat in T (0.149 s at T=39 and T=128)"
// and used the flatness to justify lowering the threshold. Flat in T is the symptom, not the
// virtue -- T=39 was paying for 128 columns. Cross-engine it shows up as T=42 reading 232.7 t/s
// here against the reference port's bf16 521.5 on the same fixture (2.24x), and T=58 reading 363
// against 606 (1.67x).
//
// So the tile is chosen by T: at most kMmaShortTileTokens takes the 64-wide schedule (26880 B of
// shared, three CTAs per SM at 384 threads, which leaves ~170 registers per thread -- no register
// squeeze), above it takes the 128-wide one. Both keep a whole 128-wide quant group per K tile, so
// the K accumulation order is untouched and the result is expected bit-identical.
using TernaryMmaPrefillSchedule = TernaryMmaSchedule<64, 128, 64, 32, 32, 2, 2>;
using TernaryMmaShortSchedule   = TernaryMmaSchedule<64, 64, 64, 32, 32, 2, 3>;

// Token count up to which the 64-wide tile is used. 64 is where the two fills cross: at T=64 the
// short tile is 100% full and the wide one 50%; at T=65 the short schedule needs a SECOND weight
// pass (ceil(65/64) = 2) while the wide one still needs one, so the wide tile wins from 65 up
// until its own fill drops again. Overridable so the crossing can be swept without a rebuild.
inline constexpr std::int32_t kMmaShortTileTokens = 64;

int mma_short_tile_tokens() {
    static const int threshold = [] {
        const char* value = std::getenv("NINFER_TERNARY_SHORT_TILE_TOKENS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return parsed > 0 ? parsed : static_cast<int>(kMmaShortTileTokens);
    }();
    return threshold;
}

// A token tile narrower than half the block wastes the pipeline, so short verification-shaped T
// stays on the SIMT path even though the kernel would be correct there.
//
// The 64 was inherited from the 128-wide token tile and had never been re-fitted on this card. It
// is also the cliff between the MMA route and the blocked GEMV: at T=58 this engine measures
// 167 t/s, while the same engine's MMA at T=84 measures 550 -- and the reference port's bf16
// rungs measure 606 at T=58. The token tile is not full either way, which is what the policy
// assumed would favour the SIMT path; the measurement says otherwise.
//
// Re-fitted on the 4070 Ti SUPER by the reference port's two-line method (fit each rung, take the
// crossing). Three-way split on a 167-token prompt at chunk=128 separates the two chunks: all-GEMV
// 1.067 s, all-MMA 0.2985 s, 128-on-MMA + 39-on-GEMV 0.4210 s, which solves to MMA 0.149 s and
// GEMV 0.272 s at T=39 -- the MMA still wins by 1.83x there. MMA is flat in T (0.149 s at T=39 and
// T=128), GEMV is linear at 6.4 ms/token, so the crossing sits near T=23. 32 is that fit rounded
// up to the token tile's own 8-token granularity, chosen on the conservative side of the estimate.
// Measured directly: T=39/54/58/65 all favour the MMA by 1.8-2.6x, and T >= 64 was already on it.
//
// Overridable so the crossing can be swept without a rebuild. Same lever the reference port
// exposes as NINFER_TERNARY_WIDE_MIN_TOKENS, and its manual is explicit that the value is a
// property of the card and has to be re-measured rather than copied.
inline constexpr std::int32_t kMmaMinTokens = 32;

int mma_min_tokens() {
    static const int threshold = [] {
        const char* value = std::getenv("NINFER_TERNARY_MMA_MIN_TOKENS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return parsed > 0 ? parsed : static_cast<int>(kMmaMinTokens);
    }();
    return threshold;
}

// Which rung serves the band between the 8-wide small-T tile and the MMA threshold -- T = 9..31.
//
// This band is the dispatch's own hole and it is the one a SHORT PROMPT lands in. The token count
// the linear ops see is prompt_tokens - 4 (measured across seven fixtures, consistently), so
// kMmaMinTokens = 32 really means "prompts of 36 or more", and an ordinary question -- "Write a
// bash script that finds the ten largest files under a directory", 35 prompt tokens -- sees T = 31
// and used to fall all the way through to the blocked SIMT GEMV. That GEMV is the rung this file's
// own note measures at 8.3-9.8x slower than the tensor-core path at T=1024.
//
// Which arm serves it is a measurement and not an argument, so all of them are reachable:
//   auto    (default) -- small_t up to kGapSmallTMaxTokens, int8 above it. The two cost models
//                        cross there, and unlike the reference port's ranking this one is measured
//                        on this port's own kernels.
//   small_t           -- the 8-wide tensor-core kernel re-entered once per tile, forced
//   s8                -- the int8 rung, with its own threshold bypassed, forced
//   gemv              -- the blocked GEMV, i.e. the behaviour before this branch existed
// The reference port's T sweep puts ITS small_t 1.8-2.5x ahead of its wide tile over this exact
// range, and its s8 table puts small_t ahead of s8 everywhere below T=32 -- but both of those are
// its kernels, and its s8 table's own cell for T=16 matches this port's s8 to within 5%, so the
// ranking was re-run rather than copied. On this port small_t wins T=9..16 and int8 wins T=17..31.
// Above this token count the gap band prefers int8 over the 8-wide tensor-core kernel.
//
// The two cost models are different in kind, which is what makes the crossing real rather than a
// tuning artifact. small_t re-reads the weight window once per 8-token tile -- the reference port's
// fitted form is `10.2 ms * ceil(T/8)` -- while s8 stages its tile once and is therefore nearly
// flat: measured here, 44.3 ms at T=15 against 50.7 ms at T=27.
//
// On this card one small_t pass costs about 20 ms, so at ceil(T/8) = 2 (T <= 16) it wins narrowly,
// and from ceil(T/8) = 3 (T >= 17) it loses to a flat ~45-50 ms. Measured: T=16 gives small_t
// 41.1 ms against s8 44.3 ms, T=27 gives small_t 68.5 ms against s8 50.7 ms. Overridable so the
// crossing can be swept rather than trusted.
inline constexpr std::int32_t kGapSmallTMaxTokens = 16;

int gap_small_t_max_tokens() {
    static const int threshold = [] {
        const char* value = std::getenv("NINFER_TERNARY_GAP_SMALL_T_MAX");
        if (value == nullptr) { return static_cast<int>(kGapSmallTMaxTokens); }
        return std::atoi(value);
    }();
    return threshold;
}

enum class GapRoute { Auto, SmallT, S8, Gemv };

GapRoute gap_route() {
    // Read once: the choice decides which kernel enters a captured CUDA graph.
    static const GapRoute route = [] {
        const char* value = std::getenv("NINFER_TERNARY_GAP");
        if (value == nullptr) { return GapRoute::Auto; }
        const std::string text(value);
        if (text == "s8") { return GapRoute::S8; }
        if (text == "gemv") { return GapRoute::Gemv; }
        if (text == "small_t") { return GapRoute::SmallT; }
        return GapRoute::Auto;
    }();
    return route;
}

template <class Schedule>
void launch_ternary_mma(const Tensor& x, const Weight& w, Tensor& out,
                        std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t rows   = w.n;
    const std::int32_t k      = w.k;
    const std::int32_t tokens = x.ne[1];
    // The reference launcher checks this too (launch_gemm does), and an undersized stride would
    // silently scatter the tile across neighbouring rows rather than fail.
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary mma: output row stride is smaller than the tile");
    }
    // Restated here rather than left to gemv_admits alone: the kernel stages whole 64-wide K tiles
    // and reads exactly two planes, so a padded K or a high plane would index out of the group.
    // Both call sites pass through gemv_admits today; this is the guard for the next one.
    if (w.padded_shape[1] != k || (k % 128) != 0) {
        throw std::invalid_argument("ternary mma: needs a whole-group K with no padding");
    }
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(tokens, Schedule::kBlockCols)), 1u);
    const bool full = (rows % Schedule::kBlockRows) == 0 && (tokens % Schedule::kBlockCols) == 0;

    const auto* x_ptr     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes     = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales    = static_cast<const std::uint8_t*>(w.scales);
    auto* out_ptr         = static_cast<__nv_bfloat16*>(out.data);

    if (full) {
        ternary_rowsplit_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, rows, k, tokens, w.padded_shape[1], out_row_stride);
    } else {
        ternary_rowsplit_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, rows, k, tokens, w.padded_shape[1], out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// int8 rung: quantize the activation row per token, then run the s8 tensor-core kernel. Requires a
// caller-provided scratch (see TernaryS8Scratch) -- the quantization pass needs one int8 code row
// per token plus one fp32 scale per token, and it has to come from the arena, because this op runs
// inside captured CUDA graphs where a lazy cudaMalloc is illegal.
//
// Only PQ2_0 is admitted. The reference port also runs its PTQ1_0 format through this same kernel
// with a shared-memory repack of the raw base-3 rows; this artifact is PQ2_0 throughout, so that
// instantiation would be carried unexercised and is left out deliberately.
// MEASURED NEGATIVE RESULT -- do not "fix" the tile to follow T without re-measuring.
//
// The waste is real: everything downstream of the tile is sized for the TILE rather than for the
// token count, so kSubTiles = kTokens/8 mma sub-tiles are walked unconditionally (dead columns are
// zero-filled into the activation plane, only the store is guarded) and a T=27 launch spends 37 of
// its 64 columns on padding. The token count IS known here -- prefill is eager, so this rung is
// never inside a captured graph -- so narrowing the tile to follow T looked free.
//
// It is not. Instantiating 16/32/64 and picking by T made every fixture in 27..32 SLOWER by ~1%:
//
//   fixture (T)      auto   tile 64   tile 32     7 reps, true alternating, median model elapsed
//   gap-sm   27      112      110       112
//   gap-mid  28      112      111       112
//   prose    28      112      111       113
//   zh-code  29      112      111       112
//   bash     31      113      111       113
//   gap-t32  32      113      111       113
//   en-code  38      112      112       133   <- negative control: T>32, auto already picks 64
//
// The control matters: en-code's two arms are IDENTICAL (112 = 112), so the ~1% is the change and
// not the harness. (Forcing 32 at T=38 costs 19%, because 38 > 32 makes the outer tok_base loop
// run twice and read every weight twice -- so a mis-set override is far worse than no override.)
//
// Why narrowing loses, from ncu on this kernel at T=32: it is NOT mma-bound. The tensor pipe is
// only 17-35% active and the small-n layers are grid-limited (grid = div_up(n,64); [5120,17408] is
// 80 CTAs on 66 SMs = 1.2 waves, sm__warps_active 10-12%). Halving the tile halves the useful work
// per CTA while the per-CTA fixed costs -- prologue, per-chunk weight staging, and the synchronous
// scale read that sits on the mma's critical path -- are untouched. The 16-column instantiation is
// also unreachable at kTernaryS8MinTokens = 17, so it could only ever have been dead code.
//
// Kept as an env-selectable arm (the repo's convention for measured-and-rejected variants) so the
// measurement can be repeated; 64 is the default.
inline int s8_tile_tokens(std::int32_t tokens) {
    (void)tokens; // kept in the signature because a future re-fit would key on it
    static const int forced = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_TILE_TOKENS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return parsed;
    }();
    return (forced == 16 || forced == 32 || forced == 64) ? forced : 64;
}

// MinBlocks_ is the second half of __launch_bounds__(128, N) and therefore a REGISTER CAP: N=3
// lets ptxas keep the 159 registers it currently wants (65536 / (3*128) = 170 available), N=4 forces
// it down to 128. ncu says that is the difference between 3 and 4 resident CTAs per SM --
// sm__warps_active reads 24% against a sm__maximum_warps_per_active_cycle of 25%, i.e. the machine
// is held at three quarters of the occupancy it is configured to allow, and the kernel is
// latency-bound (long_scoreboard 22% + barrier 19.5% on the large layer, DRAM only reaching 44.7%).
//
// This is the cheap test of the occupancy hypothesis, and the one SKILL rejects for three OTHER
// kernels (wide_t, GDN record, blocked GEMV -- all measured monotonically worse). It has never been
// tried on this one, so it gets its own env arm rather than a verdict inherited from its neighbours:
// if 159 registers were being spent on nothing, 4 CTAs should win.
inline int s8_min_blocks() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_MIN_BLOCKS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return parsed;
    }();
    return (forced == 2 || forced == 3 || forced == 4) ? forced : 3;
}

// blockIdx.y width for the s8 kernel: how many token tiles to spread across blocks.
//
// 1 is the shipped schedule -- every CTA walks all ceil(T/kTokens) tiles in sequence. Anything above
// 1 hands each CTA one tile instead, which multiplies the CTA count by the tile count at NO cost in
// weight traffic: the weights are re-staged per tile either way (stage_weights is inside the chunk
// loop, which is inside the tile loop), so splitting the loop across blocks moves the same bytes and
// simply gives the card more blocks to hide latency with. The arithmetic is untouched -- each output
// element is still summed over the same chunks in the same order by one CTA -- so this arm is
// bit-identical to the one it replaces and needs no numeric gate.
//
//   unset / auto / -1 -> split where the row-block grid alone under-fills the card (the default)
//   0                 -> the pre-gridDim.y schedule, kept as the rollback and A/B arm
//   N > 0             -> gridDim.y = min(tiles, N)
//
// MEASURED (2026-09-26, `~/ninfer-work/tokgrid-ab.py`, one binary, true alternation, 6 reps,
// ninfer-perplexity at --context 512 so every forward sees T=508 and 8 token tiles):
//   prefill throughput  1190.3 vs 943.5 tok/s  = +26.2%
//   wall clock            25.0 vs 31.2 s       = -19.9%
//   PPL                   9.693396 both arms, identical over repeated runs (the arm is bit-inert)
// Per-launch s8 time from the paired nsys traces: gridX 16 -> 5.4x, 64 -> 1.67x, 80 -> 1.80x,
// 96 -> 1.44x, and gridX 544 unchanged at 1.01x, which is the gate holding exactly where it should.
inline int s8_token_grid(int grid_x, std::int32_t tokens, int tile_tokens) {
    const int tiles = div_up(static_cast<int>(tokens), tile_tokens);
    if (tiles <= 1) { return 1; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_TOKGRID");
        if (value == nullptr) { return -1; }
        const std::string text(value);
        if (text == "auto" || text == "-1") { return -1; }
        return std::atoi(value);
    }();
    if (forced == 0) { return 1; }
    if (forced > 0) { return tiles < forced ? tiles : forced; }
    return grid_x < kTernaryS8ResidentCtas ? tiles : 1;
}

// kRowsPerCta does not depend on the token tile (it is rows-per-warp times warps), so all three
// instantiations launch the same row grid. Returns the number of K-slices actually used, so the
// caller can run the reduction.
template <int Tokens, int MinBlocks>
int launch_pq2_mma_s8_tile(const Weight& w, Tensor& out, std::int32_t out_row_stride,
                           std::int32_t tokens, TernaryS8Scratch scratch, cudaStream_t stream) {
    constexpr int kWarps = 4;
    const int grid_x = div_up(w.n, TernaryS8Storage<Tokens, kWarps>::kRowsPerCta);
    // K is cut only where the token axis cannot add blocks (ternary_s8_slices decides), and only as
    // far as the caller's reduction buffer actually reaches -- the launcher never assumes the arena
    // granted what it asked for.
    int slices = ternary_s8_slices(w.n, tokens, w.k);
    if (slices > 1) {
        const std::int64_t per_slice =
            static_cast<std::int64_t>(w.n) * (tokens < 64 ? tokens : 64);
        const std::int64_t fits = per_slice > 0 ? scratch.partial_floats / per_slice : 0;
        if (slices > fits) { slices = static_cast<int>(fits); }
        if (slices < 2 || scratch.partial == nullptr) { slices = 1; }
    }
    const dim3 grid(static_cast<unsigned>(grid_x),
                    static_cast<unsigned>(s8_token_grid(grid_x, tokens, Tokens)),
                    static_cast<unsigned>(slices));
    ternary_pq2_mma_s8_kernel<Tokens, kWarps, MinBlocks><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride, nullptr, slices > 1 ? scratch.partial : nullptr);
    return slices;
}

// One block per token. The absmax over the whole K row is reduced first, so the scale is already
// known when the codes are written (that is why this is two passes and not one atomic pass).
static void launch_s8_quantize(const Tensor& x, TernaryS8Scratch& scratch, std::int32_t k,
                               std::int32_t tokens, cudaStream_t stream) {
    ternary_s8_quantize_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), scratch.codes, scratch.scales, k);
    CUDA_CHECK(cudaGetLastError());
    scratch.quantized_x      = x.data;
    scratch.quantized_k      = k;
    scratch.quantized_tokens = tokens;
}

void quantize_ternary_s8_activation(const Tensor& x, TernaryS8Scratch& scratch,
                                    cudaStream_t stream) {
    if (scratch.codes == nullptr || scratch.scales == nullptr) { return; }
    if (x.ne[0] <= 0 || x.ne[1] <= 0) { return; }
    launch_s8_quantize(x, scratch, x.ne[0], x.ne[1], stream);
}

void launch_pq2_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                       std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                       cudaStream_t stream) {
    // SKIPPED WHEN THE SCRATCH ALREADY HOLDS THIS SHAPE. The fused parents feed one folded
    // activation to several projections (four in attn_input_proj, three in gdn_input_proj), and
    // without this each of them redid the absmax pass and the code pass over the same k x T bytes.
    // Measured: 144 of the 400 quantize launches in one prefill were redundant, 36% of a kernel
    // that is itself 2.2% of prefill.
    //
    // Keyed on the SHAPE, not on a bare "already done" flag: a scratch that is handed to a weight
    // with a different k or a different token count still quantizes, so a caller that reuses a
    // scratch across two different activations gets correct codes rather than stale ones. The
    // quantization itself is unchanged, so this is bit-identical to the redundant version.
    if (scratch.quantized_x != x.data || scratch.quantized_k != w.k ||
        scratch.quantized_tokens != tokens) {
        launch_s8_quantize(x, scratch, w.k, tokens, stream);
    }

    // 4 warps at every tile: the instantiation the reference port measured as the winner of this
    // shape. Even at 64 columns its shared footprint is 25856 B, well under this card's 48 KiB
    // static limit, so none of the three needs a cudaFuncSetAttribute opt-in.
    //
    // Only the combinations that can actually be reached are instantiated: tile 64 is the default
    // and gets all three MinBlocks arms, the two narrower tiles keep one each.
    const int tile = s8_tile_tokens(tokens);
    const int mb   = s8_min_blocks();
    int slices     = 1;
    if (tile == 16) {
        slices = launch_pq2_mma_s8_tile<16, 3>(w, out, out_row_stride, tokens, scratch, stream);
    } else if (tile == 32) {
        if (mb == 4) {
            slices = launch_pq2_mma_s8_tile<32, 4>(w, out, out_row_stride, tokens, scratch, stream);
        } else {
            slices = launch_pq2_mma_s8_tile<32, 3>(w, out, out_row_stride, tokens, scratch, stream);
        }
    } else if (mb == 2) {
        slices = launch_pq2_mma_s8_tile<64, 2>(w, out, out_row_stride, tokens, scratch, stream);
    } else if (mb == 4) {
        slices = launch_pq2_mma_s8_tile<64, 4>(w, out, out_row_stride, tokens, scratch, stream);
    } else {
        slices = launch_pq2_mma_s8_tile<64, 3>(w, out, out_row_stride, tokens, scratch, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    if (slices > 1) {
        // Same stream, so this is ordered after every slice. Deterministic by construction: the
        // slices are summed in a fixed order, which is what keeps the PPL gate meaningful.
        const std::int64_t total  = static_cast<std::int64_t>(tokens) * w.n;
        const unsigned     blocks = static_cast<unsigned>((total + 255) / 256);
        ternary_s8_reduce_kernel<<<blocks, 256, 0, stream>>>(
            scratch.partial, static_cast<__nv_bfloat16*>(out.data), out_row_stride, w.n, tokens,
            slices);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Diagnostic for the rung dispatch: names the path each call ACTUALLY takes, so a dispatch change can
// be verified instead of assumed. Env-gated and capped, so it costs nothing in a normal run. It fires
// at graph CAPTURE time, which is exactly where the choice is made; the captured graph then replays
// that same branch without re-deciding.
//
// Why it is written this way: the reference port's first version of this probe printed the
// CONDITIONS rather than the branch taken, which is how its PPL gate came back bit-identical with
// the int8 rung switched on and still left "was the rung even reached" unanswerable. Same hazard
// applies here -- the s8 rung is the one change in this dispatch that moves the numerics, so being
// able to see it fire is what makes its A/B mean anything.
//   NINFER_TERNARY_S8_DEBUG=1      -> stderr lines
//   NINFER_TERNARY_S8_DEBUG_BUDGET (default 24 lines)
//   NINFER_TERNARY_S8_DEBUG_MIN_T  (default 0: print every call)
void note_rung(const char* rung, const Tensor& x, const Weight& w, std::int32_t out_row_stride) {
    static const int min_t = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_DEBUG_MIN_T");
        return value == nullptr ? 0 : std::atoi(value);
    }();
    static int budget = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_DEBUG_BUDGET");
        return value == nullptr ? 24 : std::atoi(value);
    }();
    if (budget <= 0 || std::getenv("NINFER_TERNARY_S8_DEBUG") == nullptr) { return; }
    if (static_cast<int>(x.ne[1]) < min_t) { return; }
    --budget;
    std::fprintf(stderr, "[ternary] rung=%-9s qtype=%d T=%d k=%d n=%d stride=%d\n", rung,
                 static_cast<int>(w.qtype), static_cast<int>(x.ne[1]), static_cast<int>(w.k),
                 static_cast<int>(w.n), static_cast<int>(out_row_stride));
}

// The split-K arm of the small-t rung, which the rung line above cannot show: the rung name stays
// "small_t" and the row block is whatever the shape policy picked. Shares note_rung's switch and
// budget so it cannot turn into a printf in a hot loop by accident, and so one env var turns on
// everything that describes which kernel actually ran.
void note_split(std::int32_t rows, std::int32_t k, std::int32_t tokens, std::int32_t row_block,
                std::int32_t slices) {
    static const int min_t = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_DEBUG_MIN_T");
        return value == nullptr ? 0 : std::atoi(value);
    }();
    static int budget = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_DEBUG_BUDGET");
        return value == nullptr ? 24 : std::atoi(value);
    }();
    if (budget <= 0 || std::getenv("NINFER_TERNARY_S8_DEBUG") == nullptr) { return; }
    if (tokens < min_t) { return; }
    --budget;
    std::fprintf(stderr, "[ternary] split     n=%d k=%d T=%d rows/cta=%d slices=%d ctas=%d\n",
                 static_cast<int>(rows), static_cast<int>(k), static_cast<int>(tokens),
                 static_cast<int>(row_block), static_cast<int>(slices),
                 static_cast<int>(div_up(rows, row_block) * slices));
}

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch) {
    // The speculative verify pass runs T = draft + 1 (2..4 here). The reference tiled kernel wastes
    // five of its eight token slots at that size and needs a 128-thread CTA plus seven barriers per
    // output row, which cost more than the whole decode step it was verifying. The small-tile GEMV
    // reads each weight once for all four tokens instead.
    //
    // The row-blocked kernel below was measured on this same path and LOST: routing T = 2..4 to it
    // dropped the MTP decode from 43.0 to 32.8 t/s (en) and 57.7 to 43.9 (zh). NCU says why, and
    // the reason rules the whole family out for the verify shape rather than just that one config.
    // On the 248320-row head the tile kernel already sits at 95.47% achieved occupancy, 36
    // registers and 78% SM throughput -- the ALU pipe ("integer and logic operations") is the top
    // utilizer, 909e6 instructions in 1.70 ms against a 1.32 ms pure-issue floor. It is issue-bound
    // with no occupancy left to buy, so trading registers for fewer loads can only lose: the same
    // head under the row block runs at 63 registers, 65.67% occupancy and 47.95% SM throughput.
    //
    // The verify pass is nonetheless NOT re-reading weights -- nsys shows one forward pass, 401
    // launches, against 407 for a T=1 decode step -- so the cost is per-token issue work, not
    // weight traffic. That is why raising the draft count, which amortises the per-group 2-bit
    // decode over more tokens, is the productive lever here; see the plan document.
    if (gemv_admits(x, w, verify_token_cap())) {
        if (verify_uses_small_t() && (w.k % kSmallTGroupK) == 0) {
            note_rung("small_t", x, w, out_row_stride);
            // 0 = pick the row block per shape, and the scratch decides whether the shape also
            // splits K: both were measured on this path and on no other.
            launch_small_t(x, w, out, out_row_stride, stream, 0, scratch);
            return;
        }
        // The A/B arm below is NOT interchangeable with the small-T kernel over the whole cap.
        // ternary_pq2_gemv_tile_kernel<4> writes exactly four columns -- its `t < tokens` guard
        // skips work but there is no token-tile loop and grid.y is 1 -- so past four tokens it
        // would leave the tail of the output unwritten and say nothing. It was safe while the cap
        // was 4 for the whole entry; now that the cap is 8 the guard has to be here instead, and
        // T = 5..8 under NINFER_TERNARY_VERIFY=tile falls through to the general path below.
        if (x.ne[1] <= 4) {
            note_rung("gemv_tile", x, w, out_row_stride);
            launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
            return;
        }
    }
    // The verify-shaped entry above requires T <= 4, so everything from a short prompt (T as low
    // as 5) to a full prefill chunk lands here. The CLI constrains the CHUNK to a multiple of 128
    // (apps/cli/options.cpp), not the actual token count, so T is only a multiple of 128 for a
    // prompt that fills a whole chunk.
    //
    // The tensor-core path is the default because the SIMT one is issue-bound and has no occupancy
    // left: NCU on the 248320-row head put the token-blocked GEMV at 95.47% occupancy, ALU the top
    // pipe, and 909e6 instructions against a 1.32 ms pure-issue floor. Measured across every
    // prefill shape in this model at T=1024, MMA is 8.3-9.8x faster than that GEMV.
    //
    // int8 rung, placed ABOVE the tensor-core rung deliberately. Its threshold (33) sits one token
    // past kMmaMinTokens (32), so checking it after the MMA arm would make it unreachable: the MMA
    // branch would swallow the whole T >= 33 range. The 8-wide verify block above cannot take these
    // calls either, so this is the only position where both gates stay live.
    //
    // Keeping the two thresholds independent is the point, not an accident of ordering: the s8
    // crossing is a card property that gets swept (ternary_s8_min_tokens in ternary_s8_scratch.h),
    // and sweeping it below 32 has to let s8 cover the T = 9..32 band the blocked GEMV holds today.
    const bool s8_ready = scratch.codes != nullptr && scratch.scales != nullptr;
    if (s8_ready && ternary_s8_enabled() && out_row_stride >= w.n &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max()) &&
        x.ne[1] >= ternary_s8_min_tokens()) {
        note_rung("s8", x, w, out_row_stride);
        launch_pq2_mma_s8(x, w, out, out_row_stride, x.ne[1], scratch, stream);
        return;
    }
    // The band between the 8-wide small-T tile and the MMA threshold -- see gap_route(). It sits
    // AFTER the s8 arm on purpose: pushing NINFER_TERNARY_S8_MIN_TOKENS down is still the way to
    // hand this band to int8, and when that is done this branch never fires.
    if (x.ne[1] > verify_token_cap() && x.ne[1] < mma_min_tokens() &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        const GapRoute route = gap_route();
        const bool small_t_ok = (w.k % kSmallTGroupK) == 0;
        const bool s8_ok      = s8_ready && ternary_s8_enabled() && out_row_stride >= w.n;
        // Gemv is an explicit veto of both; the forced arms veto the other one. Auto prefers by T.
        const bool ban_small_t = route == GapRoute::S8 || route == GapRoute::Gemv;
        const bool ban_s8      = route == GapRoute::SmallT || route == GapRoute::Gemv;
        const bool prefer_small_t =
            route == GapRoute::SmallT ||
            (route == GapRoute::Auto && x.ne[1] <= gap_small_t_max_tokens());

        // The preferred arm first, then the other one if a shape gate vetoed it, and only then the
        // blocked GEMV below -- which is always correct, just slow, so falling through is right.
        if (!ban_small_t && prefer_small_t && small_t_ok) {
            note_rung("small_t_tiled", x, w, out_row_stride);
            launch_small_t_tiled(x, w, out, out_row_stride, stream);
            return;
        }
        if (!ban_s8 && s8_ok) {
            note_rung("s8", x, w, out_row_stride);
            launch_pq2_mma_s8(x, w, out, out_row_stride, x.ne[1], scratch, stream);
            return;
        }
        if (!ban_small_t && small_t_ok) {
            note_rung("small_t_tiled", x, w, out_row_stride);
            launch_small_t_tiled(x, w, out, out_row_stride, stream);
            return;
        }
    }
    if (prefill_route() == PrefillRoute::Mma && x.ne[1] >= mma_min_tokens() &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        // Two tiles, chosen by T: see TernaryMmaShortSchedule. Prefill runs eagerly (only the three
        // decode-batch entry points capture graphs), so this choice is free to follow the actual
        // token count of the chunk rather than being frozen at capture time.
        if (x.ne[1] <= mma_short_tile_tokens()) {
            note_rung("short_mma", x, w, out_row_stride);
            launch_ternary_mma<TernaryMmaShortSchedule>(x, w, out, out_row_stride, stream);
            return;
        }
        note_rung("wide_mma", x, w, out_row_stride);
        launch_ternary_mma<TernaryMmaPrefillSchedule>(x, w, out, out_row_stride, stream);
        return;
    }
    // Fallback and A/B arm. It still beats the correctness-first reference kernel, which measured
    // 7% of the card's sustained read ceiling where the GEMV shape reaches 66% on the same
    // weights. NINFER_TERNARY_PREFILL=ref forces the reference kernel, so an A/B run can qualify
    // either fast path engine-side (T = 1 alone cannot catch a token-tile or layout error).
    if (prefill_route() != PrefillRoute::Reference &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        // This arm used to be the ONLY one without a probe, and that gap cost real debugging time:
        // a 35-prompt-token fixture printed no prefill rung at all, which read as "the dispatch is
        // broken" rather than "this call landed here". Every arm that can be taken now names itself.
        note_rung("block_gemv", x, w, out_row_stride);
        launch_pq2_gemv_tile_block(x, w, out, out_row_stride, stream);
        return;
    }
    note_rung("reference", x, w, out_row_stride);
    launch_by_qtype<8>(x, w, out, out_row_stride, stream);
}

} // namespace ninfer::ops::detail
