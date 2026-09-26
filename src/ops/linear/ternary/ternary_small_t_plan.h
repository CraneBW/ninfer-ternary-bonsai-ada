// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.

#pragma once

// Host-safe half of the small-t rung: the row block and the split-K policy.
//
// A separate header for the same reason ternary_s8_scratch.h is one: the two decisions below are
// made in TWO places that must agree. ternary_rowsplit_gemm.cu launches the kernel, and
// ternary_dispatch.cpp allocates the split-K reduction buffer out of the workspace arena -- and the
// arena's declaration is the thing that goes wrong silently when the two disagree (see the
// bad_alloc in ternary_rotation.cpp's folded_activation, which was exactly a reservation that did
// not match what was later taken from the same arena). Keeping one function for both callers makes
// agreement structural rather than a review note.
//
// Nothing here touches CUDA, so it can be included from a host translation unit.

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

// ceil-div, local rather than ops/common/math.h's div_up: this header is included by host TUs that
// do not otherwise pull in the kernel-side headers, and one three-line helper is cheaper than
// widening those includes.
constexpr std::int32_t small_t_ceil_div(std::int32_t x, std::int32_t d) {
    return (x + d - 1) / d;
}

// ---------------------------------------------------------------- row block

// The verify path picks its row block per shape, because the response to it is shape-dependent and
// not monotone -- so no single value can be right, and that (not the size of the effect) is why
// P4's global sweep came out flat and kept 32. Bucketing the decode window of three nsys traces by
// output rows, on the T = 4 verify path, microseconds per round:
//
//     n        rows=16   rows=32   rows=48
//     248320      2243      2173      2193
//      34816      5668      5228      5466
//       6144      2273      2195         -
//       5120      4293      4573      4604
//       4096       629       613       767
//       1024       180       237       296
//      total     15286     15019     15681
//
// 16 wins at or below 5120 and 32 wins above it, worth 337 us a round over all-32 (1.6%, engine A/B
// +1.8% both arm orders). The 5120 bucket is not a mixture artifact: it improves at both of its k
// values (k=17408 50.1 -> 46.7 us, k=6144 19.5 -> 17.7 us). WHY is not pinned down -- ncu puts the
// family at 56 registers with DRAM the top utilizer (87.6% on the 34816 head, 63.9% on the 5120
// one) and L2 at only 22-27%, so it is neither L2-bound nor short of occupancy in the usual sense.
// The threshold is measured, not derived.
inline constexpr std::int32_t kSmallTNarrowRowMax = 5120;

// How many output rows a warp takes in the widest row block this rung ever uses. The split-K
// allocation sizes itself against this, so it has to be the SMALLEST row block in play (a smaller
// block means a wider grid means more slices).
inline constexpr std::int32_t kSmallTNarrowRowBlock = 16;

// 0 = no override. Read once: this is on the path of every small_t launch.
inline int small_t_rows_env() {
    static const int value = [] {
        const char* raw  = std::getenv("NINFER_TERNARY_SMALL_T_ROWS");
        const int parsed = raw == nullptr ? 0 : std::atoi(raw);
        return (parsed == 16 || parsed == 32 || parsed == 48) ? parsed : 0;
    }();
    return value;
}

// rows_override != 0 pins one path's value (the T = 1 decode entry and the tiled prefill entry);
// 0 means "ask this table". The env var outranks both so the sweep that produced the table is
// still reproducible.
inline int small_t_rows_for(std::int32_t output_rows, std::int32_t rows_override) {
    if (const int forced = small_t_rows_env(); forced != 0) { return forced; }
    if (rows_override != 0) { return rows_override; }
    return output_rows <= kSmallTNarrowRowMax ? kSmallTNarrowRowBlock : 32;
}

// ---------------------------------------------------------------- split-K

// One K step of the kernel: eight warps, one 128-wide ternary quant group each. Duplicated here as
// a constant only because this header cannot see the device-side schedule; the .cu static_asserts
// that the two agree, so a change to either cannot drift silently.
inline constexpr std::int32_t kSmallTSplitGroupK = 8 * 128;

// Below this the row-block grid leaves the card under-filled and K is the only axis left to cut.
// Measured with ncu on the decode verify shapes: the grid at n = 34816 sits at 87.6% DRAM (this
// card's practical ceiling) while 6144 / 5120 / 4096 / 1024 sit at 68.7 / 63.9 / 61.4 / 25.8%,
// with L2 at only 22-27% and achieved occupancy at 30-41%. So those four are short of CONCURRENCY,
// not of bandwidth and not of L2 -- which is what K-splitting adds, and what shrinking the row block
// does not: a smaller block doubles the activation restaging too (measured, the global 16 arm loses
// 0.53%), while a K-slice only ever reads its own share of the activation.
inline constexpr std::int32_t kSmallTSplitMaxRows = 6144;

// The grid size the measured-good shapes sit at. Both shapes that reach the ceiling present more
// than a thousand CTAs (34816 -> 1088, 248320 -> 7760), so that is the target for the under-filled
// ones. It is a target and not a derivation: with the caps below the mid shapes land at 768-1280.
inline constexpr std::int32_t kSmallTSplitTargetCtas = 1024;

// How much K a slice must keep before it is worth handing out. A slice pays a pipeline prologue and
// a round of partial traffic, so a slice that is one or two K steps long spends more time filling
// than mma-ing -- measured, and this is the whole difference between the shape that gains and the
// shapes that lose:
//
//   decode verify, T=4, per-round microseconds in the small_t kernel, split off -> on
//     n=5120 k=17408  (17 K steps, 4 slices, 4-5 steps each)   4360 -> 4135   -225  (5.2%)
//     n=6144 k=5120   ( 5 K steps, 2 slices, 2-3 steps each)   2268 -> 2309   +41
//     n=4096 k=5120   ( 5 K steps, 4 slices, 1-2 steps each)    596 ->  629   +33
//
// The same rule the int8 rung already keeps (chunks/4 there, on 128-wide chunks) and for the same
// reason. Four K steps is 4096 weights per slice against a slice's own staging and prologue.
inline constexpr std::int32_t kSmallTSplitMinStepsPerSlice = 4;

// The number of K-slices the small-t rung should take for this shape. 1 means "do not split", and
// that is the answer everywhere the row grid already fills the card, so the split path only runs
// where K is the only way to add blocks.
//
// `rows` is the row block the row policy picked, so the grid the slices multiply is the real one.
// The caller passes the ACTUAL row block rather than re-deriving it, because a slice count computed
// against a grid that is not the launched one would overrun the buffer it was sized for.
//
// The cap is overridable so the value can be swept rather than argued about: 0 disables the split
// entirely, which is also the rollback arm for the numeric gate (the split changes the order the K
// terms are summed in, so it is NOT bit-identical -- see the note in the .cu).
inline int small_t_split_cap() {
    static const int cap = [] {
        const char* value = std::getenv("NINFER_TERNARY_SMALL_T_KSPLIT");
        if (value == nullptr) { return 8; }
        const int parsed = std::atoi(value);
        return parsed < 0 ? 0 : parsed;
    }();
    return cap;
}

inline std::int32_t small_t_slices(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                   std::int32_t rows) {
    const int cap = small_t_split_cap();
    if (cap <= 1) { return 1; }
    // The verify path only. The tiled path (T > 8) re-enters this rung once per 8-token tile at the
    // caller's level, and prefill's own shapes are served by the MMA rungs; slicing here would be
    // multiplying a grid that is already 8-wide in tokens.
    if (tokens > 8) { return 1; }
    if (n <= 0 || k <= 0 || rows <= 0 || n > kSmallTSplitMaxRows) { return 1; }
    std::int32_t slices = small_t_ceil_div(kSmallTSplitTargetCtas, small_t_ceil_div(n, rows));
    // Never past what the arena's partial budget can hold (kTernaryS8PartialMaxRows rows in total,
    // declared by ternary_rotation_workspace_bytes), and never more slices than there are K steps
    // to hand out: a slice with no group would still cost a CTA and a prologue.
    const std::int32_t by_budget = static_cast<std::int32_t>(25280 / n);
    const std::int32_t by_groups = (k / kSmallTSplitGroupK) / kSmallTSplitMinStepsPerSlice;
    if (by_budget < slices) { slices = by_budget; }
    if (by_groups < slices) { slices = by_groups; }
    if (cap < slices) { slices = cap; }
    return slices < 2 ? 1 : slices;
}

} // namespace ninfer::ops::detail
