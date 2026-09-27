// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// The ternary arm of `linear_add`: the residual add folded into the store of the ternary rung that
// computes the projection, instead of materialising it and adding it back (G4-S3).
//
// Host-safe on purpose, like ternary_small_t_plan.h and for the same reason: the caller is the
// wrapper in ops/wrapper/linear_add.cpp, a .cpp, and it needs the switch and the entry point without
// dragging a kernel header into a host translation unit (see the opening paragraph of
// ternary_s8_scratch.h for what that costs on MSVC). Nothing here names a device type -- the
// epilogue itself is constructed on the other side of that line, inside ternary_rowsplit_gemm.cu,
// from a route value and the tensor being updated.
//
// WHY A SEPARATE ARM AT ALL, rather than folding it into ternary_dispatch(). The residual is not a
// property of the GEMM: it belongs to the CONSUMER. `linear` and `attn_input_proj` run the same
// ternary kernels and want no residual at all, and the s8 and MMA rungs -- reachable through the
// same dispatch for larger token counts -- carry no epilogue yet. Making `linear_add` a distinct
// entry point is what keeps "the consumer wants a residual" a fact the routing can act on, instead
// of a flag threaded through a path that mostly does not care.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// A/B and rollback switch for the fused residual, in the same style as the rest of the ternary
// switches. It is needed for two reasons and both are structural rather than cosmetic:
//
//   * It CHANGES THE NUMERICS, and by more than one rounding. The unfused arm rounds the projection
//     to bf16 on its way into the scratch and then adds it to a bf16 residual:
//     `out = bf16(bf16(res) + bf16(acc))`. The fused arm reads the residual, adds it to the fp32
//     accumulator the GEMM already has, and rounds once: `out = bf16(res + acc)`. So the residual
//     is no longer rounded on the way in and the projection is no longer rounded at all. That is a
//     small change, but it is a change at every element of every `o_proj` / `down` output, so no PPL
//     gate on this path can be bit-identical: the gate for turning it on is a measured OFFSET, not
//     equality, and the offset has to be recorded before it becomes a default.
//   * It decides which kernel enters a captured CUDA graph, so it has to be readable before the
//     capture -- hence a read-once static, like every other switch in this dispatch.
//
// DEFAULT OFF, and it is now MEASURED off rather than merely unmeasured -- the same rule the rest
// of this port follows, applied to a rung that lost. On the 4070 Ti SUPER, 400-token greedy decode,
// MTP K=3, alternating arms over nine rounds: the fused arm was slower in eight of the nine
// (-0.67% paired median, ranges overlapping on the one round it tied) and its draft acceptance fell
// a full point, 68.00% -> 67.00%, reproducibly and with each arm's token ids stable across rounds.
// The prediction this was built against was +1.1%. Both axes are negative.
//
// The likely reason, not yet verified: fusion replaced a dedicated, vectorised elementwise kernel
// (read two contiguous arrays, write one) with a per-element `residual[token * stride + row]` load
// inside a store loop, and every shape this model presents here goes through the split-K reduce
// (n = 5120 <= kSmallTSplitMaxRows = 6144), where that gather is not what the kernel was shaped
// for. What it saves is the projection scratch's write and read, at a size that stays in L2.
// A shape that does NOT split is the untested case and the one place this could still pay.
//
// So the two original reasons both still hold, and a third joins them:
//   1. It changes the numerics (above) -- and the change is measurably for the worse in acceptance.
//   2. Until it is on, the default path is bit-identical by construction -- there is no code path
//      from a default-configured process to the fused launch. Verified, not assumed: an unpatched
//      serve and a patched one with the gate off returned a byte-identical 400-token completion.
//   3. Measured, it loses on both axes.
//
// Flipping the default is still the one line below (`return value == nullptr || ...`), and the
// reason it is not flipped is now a measurement rather than a promise to measure. The offsets, the
// split-K coverage check and the raw logs are in results/G4-fused-epilogue/ of the port's work tree.
// What keeps this code worth carrying -- rather than deleting a rung that lost -- is that the s8 and
// MMA rungs (S5/S6) and the swiglu consumer (S4) reuse its plumbing, and A/B/C above establish that
// carrying it costs the default path exactly nothing.
[[nodiscard]] inline bool ternary_linear_add_fused_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_LINEAR_ADD_FUSED");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

// `residual_out += W * x`, with the add folded into the ternary rung's store when that rung carries
// an epilogue and the add run as its own kernel when it does not.
//
// Returns TRUE when it handled the call, and FALSE when the caller must fall back to the
// materialise-then-add path -- which is the pre-existing behaviour and is what the workspace
// capacity is still sized for. Both arms are complete implementations; this is a routing decision,
// not an error path, so nothing here throws for a shape it declines.
//
// Preconditions, all of them the caller's and all of them checked there in ops/wrapper/linear_add.cpp
// before this is called: a ternary qtype with A16Only, `x` [w.k, T] and `residual_out` [w.n, T] both
// contiguous bf16, and no overlap between them. `workspace` is the op's arena; the rotation buffer
// and the int8 / split-K scratch come out of it, scoped to this call, exactly as on the fallback
// path.
[[nodiscard]] bool ternary_linear_add_try_fused(const Tensor& x, const Weight& w,
                                                Tensor& residual_out, WorkspaceArena& workspace,
                                                cudaStream_t stream);

} // namespace ninfer::ops::detail
