// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/gguf/gguf_dispatch.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

// AllowA4 is permission to use Blackwell TMA, not a requirement of the v3
// checkpoint. Ada/Ampere kernels can consume the same NVFP4 codes with BF16
// activations; requesting unavailable W4A4 here fails during first warmup.
#if defined(NINFER_SM89) || defined(NINFER_SM86) || defined(NINFER_SM75)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
#endif
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

// The format of a v3 product belongs to each physical parent; Program's static planning
// sees only the Qwen3.8 weights profile. Reserve the maximum across the ggml block family.
std::size_t max_gguf_linear_workspace_bytes(std::int32_t output_rows,
                                             std::int32_t input_rows,
                                             std::int32_t first, std::int32_t last) {
    std::size_t peak = 0;
    for (unsigned code = static_cast<unsigned>(QType::GGUF_Q8_0);
         code <= static_cast<unsigned>(QType::GGUF_Q2_0); ++code) {
        const auto format = static_cast<QType>(code);
        const auto block = gguf_block_shape(format);
        if (input_rows % block.elements == 0) {
            peak = std::max(peak, ops::linear_workspace_capacity_bytes(
                                      format, output_rows, input_rows,
                                      ops::LinearPolicy::A16Only, first, last));
        }
    }
    return peak;
}

std::size_t max_gguf_add_workspace_bytes(std::int32_t output_rows,
                                         std::int32_t input_rows,
                                         std::int32_t first, std::int32_t last) {
    std::size_t peak = 0;
    for (unsigned code = static_cast<unsigned>(QType::GGUF_Q8_0);
         code <= static_cast<unsigned>(QType::GGUF_Q2_0); ++code) {
        const auto qtype = static_cast<QType>(code);
        if (input_rows % gguf_block_shape(qtype).elements == 0) {
            peak = std::max(peak, ops::linear_add_workspace_capacity_bytes(
                                      qtype, output_rows, input_rows,
                                      ops::LinearPolicy::A16Only, first, last));
        }
    }
    return peak;
}

std::size_t independent_post_mixer_workspace_bytes(std::int32_t first,
                                                    std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last}); // activation
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last}); // gate
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last}); // up
    const auto projection = max_gguf_linear_workspace_bytes(
        TextConfig::intermediate, TextConfig::hidden, first, last);
    const auto down = max_gguf_add_workspace_bytes(
        TextConfig::hidden, TextConfig::intermediate, first, last);
    (void)layout.alloc_bytes(std::max(projection, down));
    return layout.peak_bytes(1);
}

std::size_t individual_gguf_workspace_bytes(std::int32_t projection_rows,
                                             std::int32_t output_rows,
                                             std::int32_t token_columns,
                                             bool needs_projected) {
    WorkspaceLayoutBuilder layout;
    if (needs_projected) {
        (void)layout.alloc(DType::BF16, {TextConfig::convolution_dim, token_columns});
    }
    (void)layout.alloc(DType::BF16, {projection_rows, token_columns});
    (void)layout.alloc_bytes(max_gguf_linear_workspace_bytes(
        std::max(projection_rows, output_rows), TextConfig::hidden, 1, token_columns));
    return layout.peak_bytes(1);
}

void copy_projection_rows(const Tensor& source, Tensor& packed, std::int32_t row_begin,
                          cudaStream_t stream) {
    const std::size_t word = sizeof(std::uint16_t);
    const auto* from = static_cast<const std::byte*>(source.data);
    auto* to = static_cast<std::byte*>(packed.data) + std::size_t(row_begin) * word;
    CUDA_CHECK(cudaMemcpy2DAsync(to, std::size_t(packed.ne[0]) * word, from,
                                 std::size_t(source.ne[0]) * word,
                                 std::size_t(source.ne[0]) * word,
                                 static_cast<std::size_t>(source.ne[1]), cudaMemcpyDeviceToDevice,
                                 stream));
}

bool shared_vector_requested() {
#if defined(NINFER_SM89)
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM89_SHARED_VECTOR");
        if (!value || (value[0] == '0' && value[1] == '\0')) { return false; }
        if (value[0] == '1' && value[1] == '\0') { return true; }
        throw std::invalid_argument("NINFER_SM89_SHARED_VECTOR must be 0 or 1");
    }();
    return enabled;
#else
    return false;
#endif
}

void project_individual_gdn(const Tensor& hidden,
                             const IndividualGdnInputProjectionPayload& weights,
                             Tensor& projected, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const int tokens = hidden.ne[1] * hidden.ne[2];
    const Tensor input = hidden.view({TextConfig::hidden, tokens});
    Tensor packed = projected.view({TextConfig::convolution_dim, tokens});
    Tensor output_gate = z.view({TextConfig::value_dim, tokens});
    if (shared_vector_requested() && tokens <= ops::gguf::kMaxVectorColumns) {
        // The registered GSQ parent is [Q;K;V;Z]. Join only contiguous views
        // with identical encoding/gather. Each output still receives the same
        // FP32 vector reduction and the original BF16 rounding before conv.
        Weight qkv = weights.query;
        const auto block = gguf_block_shape(qkv.qtype);
        const bool contiguous = is_gguf(qkv.qtype) && block.elements &&
            qkv.k % block.elements == 0 &&
            weights.key.qtype == qkv.qtype && weights.value.qtype == qkv.qtype &&
            weights.key.k == qkv.k && weights.value.k == qkv.k &&
            weights.key.input_columns == qkv.input_columns &&
            weights.value.input_columns == qkv.input_columns &&
            static_cast<const std::byte*>(weights.key.qdata) ==
                static_cast<const std::byte*>(qkv.qdata) +
                static_cast<std::int64_t>(qkv.n) * (qkv.k / block.elements) * block.bytes &&
            static_cast<const std::byte*>(weights.value.qdata) ==
                static_cast<const std::byte*>(weights.key.qdata) +
                static_cast<std::int64_t>(weights.key.n) * (qkv.k / block.elements) * block.bytes;
        if (contiguous) {
            qkv.n += weights.key.n + weights.value.n;
            qkv.shape[0] = qkv.padded_shape[0] = qkv.n;
            qkv.payload_bytes = weights.query.payload_bytes + weights.key.payload_bytes +
                                weights.value.payload_bytes;
            const std::array<ops::detail::GgufProduct, 2> products{{
                {&qkv, &packed}, {&weights.z, &output_gate}}};
            if (ops::detail::try_gguf_shared_vector_dispatch(
                    input, products, workspace, stream)) { return; }
        } else {
            const std::array<ops::detail::GgufProduct, 4> products{{
                {&weights.query, &packed, 0},
                {&weights.key, &packed, TextConfig::key_dim},
                {&weights.value, &packed, 2 * TextConfig::key_dim},
                {&weights.z, &output_gate}}};
            if (ops::detail::try_gguf_shared_vector_dispatch(
                    input, products, workspace, stream)) { return; }
        }
    }
    const auto project = [&](const Weight& weight, std::int32_t offset) {
        static const bool packed_gguf = [] {
            const char* value = std::getenv("NINFER_R28_BUNDLE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (packed_gguf && is_gguf(weight.qtype) && weight.k == 5120 &&
            TextConfig::hidden == 5120 && TextConfig::key_dim == 2048 &&
            TextConfig::value_dim == 6144 && TextConfig::convolution_dim == 10240 &&
            ((weight.n == 2048 && (offset == 0 || offset == 2048)) ||
             (weight.n == 6144 && offset == 4096))) {
            ops::detail::gguf_linear_gdn_rows_bf16_dispatch(
                input, weight, packed, offset, workspace, stream);
            return;
        }
        auto scope = workspace.scope();
        Tensor temporary = workspace.alloc(DType::BF16, {weight.n, tokens});
        ops::linear(input, weight, temporary, text_policy(weight), workspace, stream);
        copy_projection_rows(temporary, packed, offset, stream);
    };
    project(weights.query, 0);
    project(weights.key, TextConfig::key_dim);
    project(weights.value, 2 * TextConfig::key_dim);
    ops::linear(input, weights.z, output_gate, text_policy(weights.z), workspace, stream);
}

// The folded ternary split parents reuse the Q4/Q5 row geometry but carry PTQ1_0/PQ2_0 at a 128-wide
// group, so the geometry-keyed capacity queries cannot tell the two encodings apart and take an
// explicit flag. The runtime answers them from the loaded weights and opts in here; the
// profile-keyed queries below reserve the rotation unconditionally (see folded_rotation_bytes).
bool folded_ternary_parent(const Weight& weight) {
    return weight.qtype == QType::PTQ1_0_G128 || weight.qtype == QType::PQ2_0_G128;
}

// Folded (rotated-basis) ternary activation scratch: a [input_width, last] BF16 buffer that
// ternary weights need before their matmul (P, then the signs, then the nominal Hadamard).
//
// The folded ternary port deliberately keeps the groupwise-int identity -- the binding layer
// resolves each weight's format from the artifact's own declaration, so the weights profile only
// decides which tensors exist, not how they are encoded -- which means this profile is shared by
// both artifacts and the reservation cannot be made conditional on it. The cost is one
// activation-sized buffer per ternary op, the same order as the activation roots the stage
// already reserves, and it is what lets the ternary artifact construct its CUDA graphs at all.
std::size_t folded_rotation_bytes(std::int32_t input_width, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(QType::PQ2_0_G128, input_width, input_width,
                                                ops::LinearPolicy::A16Only, 1, last);
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<IndividualGdnInputProjectionPayload>(weights.input_projection)) {
        return individual_gguf_workspace_bytes(TextConfig::value_dim,
                                                TextConfig::value_dim, width * batch, true);
    }
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width,
                            folded_ternary_parent(
                                std::get<SplitGdnInputProjectionPayload>(weights.input_projection)
                                    .query_key)));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<IndividualGdnInputProjectionPayload>(weights.input_projection)) {
        return individual_gguf_workspace_bytes(TextConfig::value_dim,
                                                TextConfig::value_dim, width * batch, false);
    }
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width,
                            folded_ternary_parent(
                                std::get<SplitGdnInputProjectionPayload>(weights.input_projection)
                                    .query_key)));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile>
Variant::dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t) {
    if (capacity == 0 || draft_window == 0 || draft_window > maximum_dflash_draft_tokens) {
        throw std::invalid_argument("invalid DFlash2 graph dimensions");
    }
    // Bounded attention envelopes; each tier owns its topology and can change kernel decomposition.
    auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        profiles[i].topology_class = static_cast<std::uint32_t>(i);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    if (const auto* individual =
            std::get_if<IndividualAttentionProjectionPayload>(&weights)) {
        const std::int32_t columns = hidden.ne[1];
        Tensor query_flat = query.view({TextConfig::query_size, columns});
        Tensor gate_flat = gate.view({TextConfig::query_size, columns});
        Tensor key_flat = key.view({TextConfig::kv_size, columns});
        Tensor value_flat = value.view({TextConfig::kv_size, columns});
        if (shared_vector_requested()) {
            const std::array<ops::detail::GgufProduct, 4> products{{
                {&individual->query, &query_flat}, {&individual->gate, &gate_flat},
                {&individual->key, &key_flat}, {&individual->value, &value_flat}}};
            if (ops::detail::try_gguf_shared_vector_dispatch(
                    hidden, products, workspace, stream)) { return; }
        }
        ops::linear(hidden, individual->query, query_flat, text_policy(individual->query),
                    workspace, stream);
        ops::linear(hidden, individual->gate, gate_flat, text_policy(individual->gate),
                    workspace, stream);
        ops::linear(hidden, individual->key, key_flat, text_policy(individual->key),
                    workspace, stream);
        ops::linear(hidden, individual->value, value_flat, text_policy(individual->value),
                    workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (weights.packed.qdata == nullptr) {
        const int cols = hidden.ne[1];
        Tensor q = query.view({TextConfig::query_size, cols});
        Tensor g = gate.view({TextConfig::query_size, cols});
        Tensor k = key.view({TextConfig::kv_size, cols});
        Tensor v = value.view({TextConfig::kv_size, cols});
        ops::linear(hidden, weights.query, q, text_policy(weights.query), workspace, stream);
        ops::linear(hidden, weights.output_gate, g, text_policy(weights.output_gate), workspace,
                    stream);
        ops::linear(hidden, weights.key, k, text_policy(weights.key), workspace, stream);
        ops::linear(hidden, weights.value, v, text_policy(weights.value), workspace, stream);
        return;
    }
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (weights.packed.qdata == nullptr) {
        ops::linear(hidden, weights.key, key, text_policy(weights.key), workspace, stream);
        ops::linear(hidden, weights.value, value, text_policy(weights.value), workspace, stream);
    } else {
        ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
    }
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    if (weights.packed.qdata == nullptr) {
        ops::linear(hidden, weights.query, query, text_policy(weights.query), workspace, stream);
        ops::linear(hidden, weights.output_gate, gate, text_policy(weights.output_gate), workspace,
                    stream);
    } else {
        ops::linear(hidden, weights.query, query, stream);
        ops::linear(hidden, weights.output_gate, gate, stream);
    }
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    if (const auto* individual =
            std::get_if<IndividualGdnInputProjectionPayload>(&weights.input_projection)) {
        auto scope = workspace.scope();
        project_individual_gdn(hidden, *individual, qkv, output_gate_flat, workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    if (const auto* individual =
            std::get_if<IndividualGdnInputProjectionPayload>(&weights.input_projection)) {
        auto scope = leaf_workspace.scope();
        Tensor projected = leaf_workspace.alloc(
            DType::BF16, {TextConfig::convolution_dim, hidden.ne[1], hidden.ne[2]});
        project_individual_gdn(hidden, *individual, projected, output_gate_view, leaf_workspace,
                               stream);
        ops::detail::gdn_projected_conv_snapshot_launch(
            projected, conv_weight, conv_states, valid_columns, initial_slot,
            snapshot_base_slot, query, key, value, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    if (const auto* individual =
            std::get_if<IndividualGdnInputProjectionPayload>(&weights.input_projection)) {
        auto scope = leaf_workspace.scope();
        project_individual_gdn(hidden, *individual, conv_record, output_gate_view,
                               leaf_workspace, stream);
        ops::detail::gdn_projected_conv_record_launch(
            conv_record, conv_weight, conv_states, valid_columns, initial_slots,
            query, key, value, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, execution);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, execution);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, const ::ninfer::ops::SparseMoeHints&,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    if (weights.gate && weights.up) {
        Tensor gate = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        Tensor up   = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        const std::array<ops::detail::GgufProduct, 2> products{{
            {&*weights.gate, &gate}, {&*weights.up, &up}}};
        if (!shared_vector_requested() ||
            !ops::detail::try_gguf_shared_vector_dispatch(hidden, products, workspace, stream)) {
            ops::linear(hidden, *weights.gate, gate, text_policy(*weights.gate), workspace, stream);
            ops::linear(hidden, *weights.up, up, text_policy(*weights.up), workspace, stream);
        }
#if defined(NINFER_SM89)
        static const bool fuse_down_quant = [] {
            const char* value = std::getenv("NINFER_R29_BUNDLE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        static const bool fuse_wide_down_quant = [] {
            const char* value = std::getenv("NINFER_GGUF_SILU_MMQ");
            if (!value || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_GGUF_SILU_MMQ must be 0 or 1");
        }();
        if (((fuse_down_quant && hidden.ne[1] == 512) || fuse_wide_down_quant) &&
            ops::detail::gguf_silu_mul_down_supported(weights.down, hidden.ne[1])) {
            // Reserve the original activation range above to keep workspace
            // addresses stable. Its BF16 values are now rounded in the quantizer.
            auto projected_scope = workspace.scope();
            Tensor projected = workspace.alloc(DType::BF16, {weights.down.n, hidden.ne[1]});
            ops::detail::gguf_linear_silu_mul_down_dispatch(
                gate, up, weights.down, projected, workspace, stream);
            ops::residual_add(projected, residual, stream);
            return;
        }
#endif
        ops::silu_mul(gate, up, activation, stream);
    } else {
        ops::linear_swiglu(hidden, weights.gate_up, activation,
                           text_policy(weights.gate_up), workspace, stream);
    }
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    if (weights.gate && weights.up) {
        Tensor gate = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
        Tensor up = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
        ops::linear(hidden, *weights.gate, gate, text_policy(*weights.gate), workspace, stream);
        ops::linear(hidden, *weights.up, up, text_policy(*weights.up), workspace, stream);
        ops::silu_mul(gate, up, activation, stream);
    } else {
        Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
        ops::linear(hidden, weights.gate_up, gate_up, stream);
        ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                      gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                      activation, stream);
    }
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, text_policy(weights.down), workspace, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return std::max(layout.peak_bytes(1),
                    max_gguf_linear_workspace_bytes(TextConfig::query_size,
                                                    TextConfig::hidden, first, last));
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return max_gguf_linear_workspace_bytes(TextConfig::kv_size,
                                            TextConfig::hidden, first, last);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return max_gguf_linear_workspace_bytes(TextConfig::query_size,
                                            TextConfig::hidden, first, last);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // Fused groupwise-int routes need no transient bytes; the folded ternary port shares this
        // profile and does (see folded_rotation_bytes). This projection consumes the raw hidden
        // state, so the activation is 5120 wide.
        return folded_rotation_bytes(TextConfig::hidden, last);
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(folded_rotation_bytes(TextConfig::hidden, last),
                        max_gguf_linear_workspace_bytes(TextConfig::query_size,
                                                         TextConfig::hidden, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::attention_input_rows, TextConfig::hidden,
            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, TextConfig::attention_input_rows, TextConfig::hidden,
            kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // The folded ternary port shares the groupwise-int profile: this projection folds into the
        // residual, and the ternary route needs a [hidden, T] projection scratch plus the
        // [query_size, T] activation rotation buffer. That is a superset of the zero bytes the
        // fused Q5 route needs, so one query serves both artifacts.
        return ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                      TextConfig::query_size,
                                                      ops::LinearPolicy::A16Only, first, last),
            max_gguf_add_workspace_bytes(TextConfig::hidden, TextConfig::query_size,
                                          first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // See attention_projection_workspace_capacity_bytes: the folded ternary port shares the
        // groupwise-int profile. This projection also consumes the raw hidden state.
        return std::max(folded_rotation_bytes(TextConfig::hidden, last),
                        individual_gguf_workspace_bytes(TextConfig::value_dim,
                                                         TextConfig::value_dim, last, false));
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(folded_rotation_bytes(TextConfig::hidden, last),
                        individual_gguf_workspace_bytes(TextConfig::value_dim,
                                                         TextConfig::value_dim, last, false));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::gdn_input_rows, TextConfig::hidden, kNvfp4TextPolicy,
            first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, TextConfig::gdn_input_rows, TextConfig::hidden,
            kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // The folded ternary artifact shares this profile (see folded_rotation_bytes): the
        // row geometry cannot say which artifact is loading, so the static layout reserves the
        // ternary rotation for both. The runtime query above opts out for the int artifact.
        return std::max(
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                batch_size, first, last, /*folded_ternary_rotation=*/true),
            individual_gguf_workspace_bytes(TextConfig::value_dim, TextConfig::value_dim,
                                             batch_size * last, true));
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                batch_size, first, last, /*folded_ternary_rotation=*/true),
            individual_gguf_workspace_bytes(TextConfig::value_dim, TextConfig::value_dim,
                                             batch_size * last, true));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, TextConfig::gdn_input_rows, TextConfig::hidden,
                            kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, TextConfig::gdn_input_rows,
                            TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // See the snapshot query above: the profile is shared, so the static layout reserves the
        // ternary rotation for both artifacts.
        return std::max(
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                batch_size, first, last, /*folded_ternary_rotation=*/true),
            individual_gguf_workspace_bytes(TextConfig::value_dim, TextConfig::value_dim,
                                             batch_size * last, false));
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                batch_size, first, last, /*folded_ternary_rotation=*/true),
            individual_gguf_workspace_bytes(TextConfig::value_dim, TextConfig::value_dim,
                                             batch_size * last, false));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, TextConfig::gdn_input_rows, TextConfig::hidden,
                            kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, TextConfig::gdn_input_rows,
                            TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // See attention_output_projection_workspace_capacity_bytes: the ternary route feeds the
        // 6144-wide GDN output into a [hidden, T] scratch and rotates a [value_dim, T] activation.
        return ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                      TextConfig::value_dim,
                                                      ops::LinearPolicy::A16Only, first, last),
            max_gguf_add_workspace_bytes(TextConfig::hidden, TextConfig::value_dim,
                                          first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::value_dim,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    // This query carries no weights profile, and the A/B control projections are among the folded
    // ternary weights, so the folded rotation scratch is reserved unconditionally. It is one
    // activation-sized buffer per token tile, matching what the stage already reserves.
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last) +
           folded_rotation_bytes(TextConfig::hidden, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        // The folded ternary port shares the groupwise-int profile: the SwiGLU gate/up parent is
        // projected whole (2 * intermediate rows) and the down projection folds into the residual,
        // so one query reserving both ternary routes serves both artifacts.
        return std::max(
            post_mixer_workspace_bytes(QType::PQ2_0_G128, QType::PQ2_0_G128,
                                       ops::LinearPolicy::A16Only, first, last),
            independent_post_mixer_workspace_bytes(first, last));
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            post_mixer_workspace_bytes(QType::PQ2_0_G128, QType::PQ2_0_G128,
                                       ops::LinearPolicy::A16Only, first, last),
            independent_post_mixer_workspace_bytes(first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last);
        return std::max(nvfp4, fp8);
    }
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    const std::size_t legacy = layout.peak_bytes(1);
    // v3 declares independent gate/up parents and the GGUF down projection. Keep their
    // activation planes, output delta, and one GGUF scratch simultaneously resident.
    return std::max(legacy, independent_post_mixer_workspace_bytes(first, last) +
                               std::size_t(TextConfig::hidden) * last * sizeof(std::uint16_t) +
                               256);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
