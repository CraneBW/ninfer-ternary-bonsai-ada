// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.

#pragma once

// Host-safe half of the int8 rung (#14): the token threshold, the scratch descriptor and its sizes.
//
// Separate from ternary_rowsplit_mma_s8.cuh on purpose: that header carries the CUDA kernel and
// device-only syntax, and it must never be included by a host translation unit. ternary_dispatch.cpp
// and ternary_rotation.cpp are host TUs, and including the kernel header there drags
// cuda_pipeline_helpers.h into MSVC, which fails with "unexpected volatile" -- measured, that is
// exactly how the first build of this rung broke.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// A/B and rollback switch for the int8 rung, in the same style as NINFER_TERNARY_MMA / _HADAMARD.
// It is needed because this rung CHANGES THE NUMERICS (activations get quantized to int8), so it must
// be comparable inside one binary and switchable off without a rebuild:
//   NINFER_TERNARY_S8=0   -> stay on the bf16 rungs
// Read once, because the choice decides which kernel enters the captured CUDA graph.
[[nodiscard]] inline bool ternary_s8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// First token count at which the int8 rung beats both bf16 rungs.
//
// The reference port measured 33 on a 4090, against ITS bf16 rungs. Re-fitted here, because both
// of the things that number depends on are different on this card: the bf16 opponent (this port's
// own short/wide MMA schedules) and the kernel's own cost.
//
// This port's numbers, absolute per-call time at the same shape mix:
//   T   bf16 small_t(x8 tiles)   bf16 short tile(64w)   s8
//   16  41.1 ms                  84.8 ms               45.1 ms
//   28  68.5 ms                  94.6 ms               51.3 ms
// s8 is flat (44.3 ms at T=15 against 51.3 at T=28) while small_t steps by ~14-20 ms per 8-token
// tile, so the two cross between T=16 and T=17. From there up s8 wins against both bf16 rungs,
// including the wide one (measured T=127: 1.38k against 803.9 t/s; T=845: 1.91k against 1.20k).
//
// 17 and not 33 also closes a cliff: the gap branch covers T < kMmaMinTokens (32) and the MMA rung
// takes T >= 32, so with the old 33 the two met badly -- T=31 was served by s8 at ~51 ms and T=32
// by the bf16 short tile at ~95 ms, i.e. one more token cost nearly twice as much.
inline constexpr int kTernaryS8MinTokens = 17;

// The lowest token count at which the rung can be ASKED to run -- which is NOT the count at which
// it is chosen, and the difference is a live bug that was measured.
//
// kTernaryS8MinTokens is a policy threshold: it decides when int8 is worth taking. But the dispatch
// has a second way in. The T = 9..31 band between the small-T tile and the MMA threshold can be
// routed to int8 explicitly (NINFER_TERNARY_GAP=s8), and that route does not consult this
// threshold -- so the scratch has to exist across the whole band. With the allocation gated on the
// policy threshold instead, the arm silently fell through to the blocked GEMV: gap-tiny read
// 136.7 t/s with GAP=s8 against 136.3 for the GEMV arm, i.e. the route was not running at all,
// while the same fixture read 338.6 with the threshold itself pushed down to 9.
//
// 9 is exactly one past the verify cap, and the cap is clamped to [1,8] so the two cannot drift.
// Allocating below the policy threshold costs the arena nothing it has not already reserved:
// ternary_rotation_workspace_bytes() counts these bytes unconditionally, so this only moves where
// the bump pointer sits inside a scope that was sized for them anyway.
inline constexpr int kTernaryS8ScratchMinTokens = 9;

// The 33 above is a property of the CARD, not of the rung: it is where this kernel's ~43 ms per
// weight pass meets the cost of the bf16 rung it replaces. And the bf16 rung it replaces on THIS
// port is not the one the 4090 measurement used -- this port's prefill runs its own 64x128 MMA
// schedule, measured 25-30% faster than the reference port's bf16 path on this card. The crossing
// therefore has to be re-measured here rather than inherited, which is exactly what the reference
// port's manual says about NINFER_TERNARY_WIDE_MIN_TOKENS. Overridable so it is an A/B knob rather
// than a rebuild, in the same style as every other threshold in this dispatch.
[[nodiscard]] inline int ternary_s8_min_tokens() {
    static const int threshold = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_MIN_TOKENS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return parsed > 0 ? parsed : kTernaryS8MinTokens;
    }();
    return threshold;
}

// How many s8 CTAs this card can hold at once: 66 SMs x 3 resident CTAs/SM. The 3 is not a guess --
// it is where the kernel's own shared footprint lands it (25856 B x 3 = 77568 B of the 100 KB an
// sm_89 SM offers; a fourth would need 103424 B). It is a CARD constant: re-derive it before quoting
// any of the grid arithmetic below on different hardware.
//
// It is used to decide when a layer's row-block grid (`div_up(n, 64)`, a function of n alone) leaves
// the card empty, which is the case for every shape in this artifact with n <= 12288 -- 16, 64, 80,
// 96 and 192 CTAs against these 198 slots -- and those shapes are 48% of the ternary weight bytes
// and 65% of the measured s8 time.
inline constexpr int kTernaryS8ResidentCtas = 198;

// Activation-quantization scratch for the int8 rung: one int8 code row per token (token-major, the
// same layout as the activation it is built from) plus one fp32 scale per token.
//
// The caller owns this memory. The ternary linear op runs inside captured CUDA graphs, so a lazy
// cudaMalloc at launch time is illegal; it must come from the op's workspace arena, and the arena's
// capacity must count it (see ternary_rotation_workspace_bytes()).
struct TernaryS8Scratch {
    std::int8_t* codes  = nullptr;
    float*       scales = nullptr;
    // Split-K reduction buffer, present only when the launcher decided to slice K. Element
    // [slice][token][row] is an fp32 partial the epilogue has already scaled by the token's
    // activation scale, so the reduce is a plain deterministic sum -- NOT an atomicAdd, which would
    // make the result depend on scheduling order and take the numeric gate with it.
    float*       partial        = nullptr;
    std::int32_t partial_floats = 0;   // its capacity, so the launcher can bound the slice count
    // WHAT `codes` CURRENTLY HOLDS, or null for "nothing quantized yet". Several projections read
    // ONE folded activation -- the fused attention parent runs four (q/gate/k/v) and the GDN parent
    // three (qk/value/z) -- and each used to re-run the whole absmax + quantize pass over the same
    // k x T bytes. Recording the source lets the later readers skip it.
    //
    // All three fields, not a bare "already done" flag: a caller that hands the scratch to a
    // different weight (different k) or a different activation must still get correct codes, and
    // comparing the source pointer as well as the shape is what makes a stale hit impossible rather
    // than merely unlikely. The pointer is the only one of the three that a same-shaped, different
    // activation could not also satisfy.
    const void*  quantized_x      = nullptr;
    std::int32_t quantized_k      = 0;
    std::int32_t quantized_tokens = 0;
};

inline constexpr std::size_t ternary_s8_codes_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
}

inline constexpr std::size_t ternary_s8_scales_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * sizeof(float);
}

// Split-K is only taken when the token axis cannot supply blocks (one tile, i.e. T <= 64) AND the row
// axis under-fills the card (gridX < 198), so the slice count is ceil(198 / gridX) with gridX =
// ceil(n / 64). Both facts bound the partial buffer without needing n in the capacity query:
//
//   gridX < 198  =>  n <= 197 * 64 = 12608
//   slices <= 198 / gridX + 1  =>  slices * n <= 12672 + n <= 25280
//
// and the buffer is slices * n * min(tokens, 64) floats. 25280 * 64 * 4 = 6.47 MB, so the budget
// below is a real upper bound on anything the launcher can ask for -- it is declared as a constant in
// ternary_rotation_workspace_bytes() (which only receives k and tokens) and asserted at allocation.
inline constexpr std::size_t kTernaryS8PartialMaxRows = 25280;
inline constexpr std::size_t ternary_s8_partial_budget_bytes(std::int32_t tokens) {
    const std::int32_t tile = tokens < 64 ? tokens : 64;
    return kTernaryS8PartialMaxRows * static_cast<std::size_t>(tile) * sizeof(float);
}

inline constexpr std::size_t ternary_s8_partial_bytes(std::int32_t n, std::int32_t tokens,
                                                      std::int32_t slices) {
    const std::int32_t tile = tokens < 64 ? tokens : 64;
    return static_cast<std::size_t>(slices) * static_cast<std::size_t>(n) *
           static_cast<std::size_t>(tile) * sizeof(float);
}

// How many K-slices the s8 rung should take for this shape. One means "do not split", and that is
// the answer for every shape the token grid already fills, so the split path only ever runs where
// it is the only way to add blocks.
//
// The cap of 8 is not derived, it is chosen: each slice pays one pipeline prologue and one round of
// partial traffic, and a slice shorter than a few chunks spends more time filling its pipeline than
// doing mma. Overridable so the value can be swept rather than argued about.
inline int ternary_s8_slices(std::int32_t n, std::int32_t tokens, std::int32_t k) {
    static const int cap = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_KSPLIT");
        if (value == nullptr) { return 8; }
        const int parsed = std::atoi(value);
        return parsed < 0 ? 0 : parsed;   // 0 disables the split entirely
    }();
    if (cap <= 1) { return 1; }
    if (tokens > 64) { return 1; }        // more than one tile: the token grid is the cheaper lever
    if (n <= 0 || k <= 0) { return 1; }
    const std::int32_t grid_x = (n + 63) / 64;
    if (grid_x >= kTernaryS8ResidentCtas) { return 1; }
    std::int32_t slices = (kTernaryS8ResidentCtas + grid_x - 1) / grid_x;
    const std::int32_t chunks       = k / 128;          // one chunk == one ternary group
    const std::int32_t by_chunks    = chunks / 4;       // keep at least 4 chunks per slice
    if (by_chunks < slices) { slices = by_chunks; }
    if (slices > cap) { slices = cap; }
    return slices < 2 ? 1 : slices;
}

} // namespace ninfer::ops::detail
