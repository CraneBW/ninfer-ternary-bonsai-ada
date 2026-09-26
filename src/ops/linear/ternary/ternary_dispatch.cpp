// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

TernaryS8Scratch allocate_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t n,
                                             std::int32_t k, std::int32_t tokens) {
    TernaryS8Scratch scratch{};
    if (tokens < kTernaryS8ScratchMinTokens) {
        // Below the lowest count the rung can be ASKED for, not below the count it is chosen at --
        // see kTernaryS8ScratchMinTokens for why those two differ and what gating on the wrong one
        // cost. Nothing above this line can ask for int8, so nothing is allocated.
        return scratch;
    }
    const DeviceSpan codes  = workspace.alloc_bytes(ternary_s8_codes_bytes(k, tokens));
    const DeviceSpan scales = workspace.alloc_bytes(ternary_s8_scales_bytes(tokens));
    scratch.codes           = static_cast<std::int8_t*>(codes.data);
    scratch.scales          = static_cast<float*>(scales.data);
    // Split-K reduction buffer, taken only for the shapes that will really slice. The launcher caps
    // its own slice count against partial_floats, so an arena that under-delivers degrades to a
    // single slice rather than overrunning; and the size here is bounded by
    // ternary_s8_partial_budget_bytes(), which is the term ternary_rotation_workspace_bytes()
    // declares for it (SKILL trap 8: any new scratch has to be counted where the arena is sized).
    const int slices = ternary_s8_slices(n, tokens, k);
    if (slices > 1) {
        const std::size_t bytes = ternary_s8_partial_bytes(n, tokens, slices);
        // The capacity statement declares the BOUND (ternary_s8_partial_budget_bytes), not this
        // size, because it cannot see n. Assert the derivation instead of trusting it: a shape that
        // broke it would overrun the arena rather than fail.
        if (bytes > ternary_s8_partial_budget_bytes(tokens)) {
            throw std::invalid_argument(
                "ternary s8 split-K: the reduction buffer exceeds the capacity declared for it "
                "[N=" + std::to_string(n) + ", K=" + std::to_string(k) + ", T=" +
                std::to_string(tokens) + ", slices=" + std::to_string(slices) + ", bytes=" +
                std::to_string(bytes) + "]");
        }
        const DeviceSpan  partial = workspace.alloc_bytes(bytes);
        scratch.partial           = static_cast<float*>(partial.data);
        scratch.partial_floats    = static_cast<std::int32_t>(bytes / sizeof(float));
    }
    return scratch;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream, TernaryS8Scratch scratch) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    // No workspace on this entry point: the CALLER already folded the activation, and may hand in
    // the int8 scratch it allocated from its own arena (see allocate_ternary_s8_scratch). With an
    // empty scratch these calls stay on the bf16 rungs.
    launch(x_folded, w, out, out_row_stride, stream, scratch);
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream, TernaryS8Scratch scratch) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, stream, scratch);
}

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the M4 decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream, TernaryS8Scratch{});
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope              = workspace->scope();
    const Tensor activation = folded_activation(x, w, *workspace, stream);

    // int8 rung scratch: one int8 code row per token plus one fp32 scale per token. Taken from the
    // same arena the rotation just used, and counted by ternary_rotation_workspace_bytes() so the
    // planner sizes the arena for it -- allocating it lazily inside the launch would be illegal,
    // because this op runs inside captured CUDA graphs.
    const TernaryS8Scratch scratch = allocate_ternary_s8_scratch(*workspace, w.n, w.k, x.ne[1]);
    launch(activation, w, out, w.n, stream, scratch);
}

} // namespace ninfer::ops::detail
