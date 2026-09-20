// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma.cuh"

#include <cstdint>
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

// A PQ2_0 weight can take the GEMV family whenever the layout did not pad K past the real width --
// true for every width in this model (5120/6144/10240/17408 are all whole 128-groups), and checked
// here rather than assumed, because the GEMV reads whole groups without a column guard.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    return w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr && w.padded_shape[1] == w.k &&
           (w.k % 128) == 0 && x.ne[1] >= 1 && x.ne[1] <= max_tokens;
}

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream) {
    if (gemv_admits(x, w, 1)) {
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

// The measured winner of the K=64 sweep on the target card: 64 output rows, a 128-token tile, 8
// warps laid out 2x4, two cp.async stages, two CTAs per SM. It beat 64x64, 128x64, 16-warp and
// 32-warp variants (37.0 ms against 41.4-54.8 on the 248320x5120 head at T=1024).
using TernaryMmaPrefillSchedule = TernaryMmaSchedule<64, 128, 64, 32, 32, 2, 2>;

// A token tile narrower than half the block wastes the pipeline, so short verification-shaped T
// stays on the SIMT path even though the kernel would be correct there.
inline constexpr std::int32_t kMmaMinTokens = 64;

template <class Schedule>
void launch_ternary_mma(const Tensor& x, const Weight& w, Tensor& out,
                        std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t rows   = w.n;
    const std::int32_t k      = w.k;
    const std::int32_t tokens = x.ne[1];
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

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream) {
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
    if (gemv_admits(x, w, 4)) {
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // Prefill reaches this branch with T >= 128: the CLI requires the prefill chunk to be a
    // multiple of 128, so the verify-shaped entry above never covers it.
    //
    // The tensor-core path is the default because the SIMT one is issue-bound and has no occupancy
    // left: NCU on the 248320-row head put the token-blocked GEMV at 95.47% occupancy, ALU the top
    // pipe, and 909e6 instructions against a 1.32 ms pure-issue floor. Measured across every
    // prefill shape in this model at T=1024, MMA is 8.3-9.8x faster than that GEMV.
    if (prefill_route() == PrefillRoute::Mma && x.ne[1] >= kMmaMinTokens &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        launch_ternary_mma<TernaryMmaPrefillSchedule>(x, w, out, out_row_stride, stream);
        return;
    }
    // Fallback and A/B arm. It still beats the correctness-first reference kernel, which measured
    // 7% of the card's sustained read ceiling where the GEMV shape reaches 66% on the same
    // weights. NINFER_TERNARY_PREFILL=ref forces the reference kernel, so an A/B run can qualify
    // either fast path engine-side (T = 1 alone cannot catch a token-tile or layout error).
    if (prefill_route() != PrefillRoute::Reference &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        launch_pq2_gemv_tile_block(x, w, out, out_row_stride, stream);
        return;
    }
    launch_by_qtype<8>(x, w, out, out_row_stride, stream);
}

} // namespace ninfer::ops::detail
