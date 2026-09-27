// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear_add/ternary/ternary_linear_add_plan.h"

#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rotation.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Everything this op needs beyond the caller's validation, refused without touching the arena.
//
// It is a SUBSET of what ternary_small_t_route() decides -- the route function is the authority and
// is asked again after the scratch exists, because two of its terms (the int8 scratch, and through
// it the gap band's s8 sub-arm) cannot be evaluated before the arena has been touched. Keeping the
// cheap shape gate here means a shape the fused arm can never serve does not even open a scope.
bool ternary_linear_add_fused_admits(const Tensor& x, const Weight& w, const Tensor& residual_out) {
    if (w.qtype != QType::PTQ1_0_G128 && w.qtype != QType::PQ2_0_G128) { return false; }
    if (x.data == nullptr || residual_out.data == nullptr) { return false; }
    if (x.dtype != DType::BF16 || residual_out.dtype != DType::BF16) { return false; }
    if (x.ne[0] != w.k || x.ne[1] != residual_out.ne[1] || residual_out.ne[0] != w.n) { return false; }
    if (!x.is_contiguous() || !residual_out.is_contiguous()) { return false; }
    // The kernel stages whole 128-wide quant groups, and its K loop and its group-relative split are
    // built from 8-group steps. Both are already checked by ternary_small_t_route(); repeated here
    // only so that the shape gate is self-contained for a reader who has not opened the .cu.
    if (w.k <= 0 || w.n <= 0 || x.ne[1] <= 0) { return false; }
    if ((w.k % 128) != 0 || (w.padded_shape[1] != w.k)) { return false; }
    return true;
}

} // namespace

bool ternary_linear_add_try_fused(const Tensor& x, const Weight& w, Tensor& residual_out,
                                  WorkspaceArena& workspace, cudaStream_t stream) {
    if (!ternary_linear_add_fused_enabled()) { return false; }
    if (!ternary_linear_add_fused_admits(x, w, residual_out)) { return false; }

    // Scoped, and opened BEFORE the gate that needs the scratch: the route function has to see the
    // same int8 / split-K scratch the launch will be given, because whether the gap band's int8
    // sub-arm is live is one of the terms that decides the route. A declined call unwinds the scope
    // and leaves the arena where it found it -- the scratch is only a bump-pointer reservation until
    // something writes it, so declining costs no device work at all.
    auto scope = workspace.scope();
    const TernaryS8Scratch scratch = allocate_ternary_s8_scratch(workspace, w.n, w.k, x.ne[1]);
    const TernarySmallTRoute route = ternary_small_t_route(x, w, w.n, scratch);
    if (route == TernarySmallTRoute::None) { return false; }

    // Into the folded basis, by the same call ternary_dispatch() makes -- including its
    // NINFER_TERNARY_HADAMARD=0 arm, which hands back the raw activation on purpose so that a broken
    // rotation and a broken decode can be told apart. Whichever branch it takes, the GEMM that
    // follows sees exactly the activation the unfused arm would have fed it.
    const Tensor activation = folded_activation(x, w, workspace, stream);

    // w.n is the output's row count and also its token stride: residual_out is [w.n, T] contiguous
    // (validated by the caller), which is the shape launch_small_t hardcodes.
    return launch_ternary_add_residual(route, activation, w, residual_out, w.n, stream, scratch);
}

} // namespace ninfer::ops::detail
