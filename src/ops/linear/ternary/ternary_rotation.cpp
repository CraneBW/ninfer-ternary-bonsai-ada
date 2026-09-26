// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_rotation.h"

#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kActivationBytesPerElement = 2; // BF16

bool ternary_qtype(QType qtype) noexcept {
    return qtype == QType::PTQ1_0_G128 || qtype == QType::PQ2_0_G128;
}

} // namespace

bool ternary_weight_is_folded(const Weight& weight) noexcept {
    return ternary_qtype(weight.qtype) && weight.hadamard_signs != nullptr &&
           weight.hadamard_n_blk > 0;
}

bool ternary_rotation_enabled() {
    // Read once: the value must not change between graph construction and graph replay, because
    // the rotation decides whether a kernel appears in the captured graph at all.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_HADAMARD");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

bool ternary_gdn_perm_enabled() {
    // Default OFF: this port's packer already normalized every GDN tensor to the grouped order,
    // so the reference's feature permutation would be a second permutation. See the header.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_GDN_PERM");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

// The rotation buffer alone: what folded_activation reserves and hands back. Kept beside the
// capacity statement so the two cannot drift -- the statement must equal this plus everything else
// the op allocates, and for a while it did not (see folded_activation).
constexpr std::size_t rotation_buffer_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) *
           kActivationBytesPerElement;
}

std::size_t ternary_rotation_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    if (k <= 0 || tokens <= 0) { return 0; }
    // The rotation buffer plus the int8 rung's activation-quantization scratch (one int8 code row
    // per token, one fp32 scale per token, and a little slack for the arena's alignment).
    // They are declared together because this function IS the capacity statement for the ternary
    // linear op (see linear.cpp), so anything the op allocates from the arena must be counted here.
    // The quantization pass itself only runs from ternary_s8_min_tokens() up, but the capacity is
    // the max over token counts the plan allows, which is what the planner asks for.
    const std::size_t rotated = static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) *
                                kActivationBytesPerElement;
    const std::size_t s8_scratch = ternary_s8_codes_bytes(k, tokens) +
                                   ternary_s8_scales_bytes(tokens) +
                                   // The split-K reduction buffer. Declared as its provable upper
                                   // bound rather than its actual size because this function is
                                   // handed k and tokens but not n, and the real size depends on n
                                   // through the slice count. The bound is derived at
                                   // kTernaryS8PartialMaxRows: splitting only happens while
                                   // gridX < 198, so n <= 12608, and slices * n <= 12672 + n. The
                                   // allocation asserts it, so a future shape that violates the
                                   // derivation fails loudly instead of overrunning the arena.
                                   ternary_s8_partial_budget_bytes(tokens) + 512u;
    return rotated + s8_scratch;
}

std::size_t ternary_projection_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens) {
    return ternary_rotation_workspace_bytes(output_rows, tokens) +
           ternary_rotation_workspace_bytes(input_rows, tokens);
}

Tensor folded_activation(const Tensor& x, const Weight& weight, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    if (!ternary_rotation_enabled()) { return x; }
    if (!ternary_weight_is_folded(weight)) {
        throw std::invalid_argument(
            "folded ternary weight has no sign block; the artifact must carry "
            "text/hadamard_signs and text/hadamard_widths");
    }
    // THE ROTATION BUFFER ALONE, not the whole capacity statement. This used to reserve
    // ternary_rotation_workspace_bytes() -- which also covers the int8 rung's codes, scales and
    // split-K budget -- and then hand back a tensor using only the first k*T*2 bytes of it, while
    // allocate_ternary_s8_scratch took those codes and scales from AFTER this reservation. The arena
    // therefore saw `rot + codes + scales` reserved against a declared `rot`, and the shortfall
    // stayed invisible wherever a stage happened to have spare room. causal_scoring has none, sized
    // to its logits buffer plus 8 KB, and it died with `bad_alloc bytes=5017600` -- which is exactly
    // 5120*980, the codes for one flush. Reserving what is used makes the declaration exact.
    const DeviceSpan span = workspace.alloc_bytes(rotation_buffer_bytes(weight.k, x.ne[1]));
    Tensor rotated(span.data, DType::BF16, {weight.k, x.ne[1]});
    launch_ternary_rotation(x, rotated, weight, stream);
    return rotated;
}

} // namespace ninfer::ops::detail
