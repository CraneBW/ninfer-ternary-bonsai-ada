// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#include "ninfer/ops/linear_topk.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) {
    return TextConfig::is_full_attention(static_cast<int>(layer));
}

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen36Nvfp4:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::Qwen38Nvfp4:
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

// The grouped row-split family is interchangeable at the container level: every member shares
// the row-split-k128-v1 layout and differs only in group geometry and code packing, and the
// artifact declares which member it stores. So resolve the DECLARED format rather than trusting
// the plan's expectation -- that is what lets one runtime read both a groupwise-int conversion
// (Q4G64/Q5G64/Q6G64/W8G32) and the ternary port (PTQ1_0_G128/PQ2_0_G128).
//
// The loosening is deliberately narrow: the declared format must itself be a grouped row-split
// format, and bind_tensor() still validates layout AND shape, so a mismatch in either direction
// is still rejected. Everything downstream reads WeightPlan::format, so resolving it here
// propagates to all materialized_weight() call sites without touching them.
bool is_row_split_grouped(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
    case NumericFormat::PTQ1_0_G128:
    case NumericFormat::PQ2_0_G128:
        return true;
    default:
        return false;
    }
}

// Folded-basis feature permutation P. Only the GDN output projection stores grouped columns
// while the runtime produces tiled v-head geometry, so this is gated on BOTH the object's role
// and its input width rather than on the width alone. Mirrors llama-model.cpp:2080-2089.
void set_folded_perm(WeightPlan& plan, std::string_view name, std::uint64_t columns) {
    constexpr std::uint64_t kGdnOutputWidth = TextConfig::value_dim;
    constexpr std::int32_t kNv             = TextConfig::gdn_value_heads;
    constexpr std::int32_t kNk             = TextConfig::gdn_key_heads;
    if (name.ends_with("/gdn/output") && columns == kGdnOutputWidth) {
        plan.hadamard_perm_hd  = static_cast<std::int32_t>(columns) / kNv;
        plan.hadamard_perm_nk  = kNk;
        plan.hadamard_perm_rep = kNv / kNk;
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    const std::uint64_t columns = shape.size() > 1 ? shape.begin()[1] : 0;
    if (is_row_split_grouped(format)) {
        const artifact::ObjectDescriptor* descriptor = binder.find(name);
        if (descriptor == nullptr ||
            !std::holds_alternative<artifact::TensorDescriptor>(*descriptor)) {
            throw artifact::ArtifactError(std::string(name) + ": expected a tensor object");
        }
        const NumericFormat declared = std::get<artifact::TensorDescriptor>(*descriptor).format;
        if (!is_row_split_grouped(declared)) {
            throw artifact::ArtifactError(std::string(name) +
                                          ": artifact declares a non-grouped format for a "
                                          "row-split weight");
        }
        WeightPlan plan{.object = artifact::bind_tensor(binder, name, declared, shape, placement),
                        .format = declared};
        set_folded_perm(plan, name, columns);
        return plan;
    }
    WeightPlan plan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                    .format = format};
    set_folded_perm(plan, name, columns);
    return plan;
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

// --- folded (rotated-basis) sign table ---------------------------------------
//
// See bindings.h for what the table is. These constants describe the artifact contract the
// ternary port writes: 1024-wide normalized Sylvester-Hadamard blocks, 28672 explicit signs, and
// widths that partition the signs exactly.
inline constexpr std::int32_t kHadamardBlockSize  = 1024;
inline constexpr std::size_t kHadamardSignValues  =
    TextConfig::hidden + TextConfig::value_dim + TextConfig::intermediate;
inline constexpr std::size_t kHadamardWidthCount  = 3;

// Load-scoped: installed for exactly one materialization pass, then removed, so no global state
// outlives a load. Attaching the sign block here -- at the single funnel that builds every
// Weight -- is what keeps all thirty materialized_weight() call sites untouched.
struct FoldedSigns {
    const float* base = nullptr;
    std::vector<std::pair<std::int32_t, std::uint64_t>> width_offsets;

    [[nodiscard]] const float* for_width(std::int32_t width) const noexcept {
        for (const auto& entry : width_offsets) {
            if (entry.first == width) { return base + entry.second; }
        }
        return nullptr;
    }
};

FoldedSigns* g_folded_signs = nullptr;

class FoldedSignsScope {
public:
    FoldedSignsScope(const std::optional<HadamardSignsPlan>& plan,
                     const artifact::MaterializedArtifact& materialized) {
        if (!plan.has_value()) { return; }
        table_.base          = static_cast<const float*>(materialized.device_data(plan->values));
        table_.width_offsets = plan->width_offsets;
        if (table_.base != nullptr) { g_folded_signs = &table_; }
    }
    ~FoldedSignsScope() { g_folded_signs = nullptr; }

    FoldedSignsScope(const FoldedSignsScope&)            = delete;
    FoldedSignsScope& operator=(const FoldedSignsScope&) = delete;

private:
    FoldedSigns table_;
};

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    // Ternary weights are exactly the folded weights (the artifact's 402 ternary tensors are its
    // 401 folded weights plus the inverse-mapped token embedding), so the format alone decides
    // that a sign block is required.
    const auto attach_folded_signs = [&](Weight& w) {
        if (g_folded_signs == nullptr) { return; }
        if (plan.format != NumericFormat::PTQ1_0_G128 &&
            plan.format != NumericFormat::PQ2_0_G128) {
            return;
        }
        const float* signs = g_folded_signs->for_width(columns);
        if (signs == nullptr) {
            throw artifact::ArtifactError(
                "folded ternary weight has no sign block for input width " +
                std::to_string(columns));
        }
        w.hadamard_signs = signs;
        w.hadamard_n_blk = columns / kHadamardBlockSize;
        w.hadamard_perm_hd  = plan.hadamard_perm_hd;
        w.hadamard_perm_nk  = plan.hadamard_perm_nk;
        w.hadamard_perm_rep = plan.hadamard_perm_rep;
    };

    if (plan.format != NumericFormat::NVFP4) {
        Weight out =
            artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
        attach_folded_signs(out);
        return out;
    }

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));

    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    attach_folded_signs(out);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    // Row-split plane geometry is per-format: Q4/Q5/Q6/W8 carry 32 base bytes per group, while
    // PTQ1_0 carries 24 base + 2 high and PQ2_0 carries 32 base + no high plane. Using the
    // default 32/0 for PTQ1_0 would slice the wrong byte ranges out of the payload.
    const std::uint64_t low_group =
        block.qtype == QType::PTQ1_0_G128 ? 24 : 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S    ? 8
                                     : block.qtype == QType::Q6G64_F16S  ? 16
                                     : block.qtype == QType::PTQ1_0_G128 ? 2
                                                                         : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden);
    out.down    = materialized_weight(materialized, plan.down, TextConfig::hidden, TextConfig::intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, (TextConfig::query_size + TextConfig::kv_size), TextConfig::hidden),
            .gate_value = materialized_weight(materialized, split->gate_value, (TextConfig::query_size + TextConfig::kv_size), TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, TextConfig::attention_input_rows, TextConfig::hidden),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, (2 * TextConfig::key_dim), TextConfig::hidden),
            .value_z   = materialized_weight(materialized, split->value_z, (2 * TextConfig::value_dim), TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, TextConfig::gdn_input_rows, TextConfig::hidden),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, TextConfig::gdn_value_heads, TextConfig::hidden),
            .b_projection = materialized_weight(materialized, split->b_projection, TextConfig::gdn_value_heads, TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, TextConfig::gdn_control_rows, TextConfig::hidden),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {(TextConfig::query_size + TextConfig::kv_size), TextConfig::hidden}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {(TextConfig::query_size + TextConfig::kv_size), TextConfig::hidden}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {(2 * TextConfig::key_dim), TextConfig::hidden}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {(2 * TextConfig::value_dim), TextConfig::hidden}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {TextConfig::gdn_value_head_dim});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {TextConfig::hidden, TextConfig::intermediate});
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            WeightPlan input;
            if (is_early_attention_input(layer)) {
                input = bind_weight(binder, prefix + "attention/query_key_gate_value",
                                    NumericFormat::BF16, {TextConfig::attention_input_rows, TextConfig::hidden});
            } else {
                input = bind_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value", TextConfig::attention_input_rows, TextConfig::hidden,
                    prefix + "attention/input_projection/input_scale_divisor");
            }
            target.attention.projection =
                FusedAttentionProjectionPlan{.query_key_gate_value = input};
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            if (is_bf16_attention_output(layer)) {
                target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                      NumericFormat::BF16, {TextConfig::hidden, TextConfig::query_size});
            } else {
                target.attention.output =
                    bind_nvfp4_weight(binder, prefix + "attention/output", TextConfig::hidden, TextConfig::query_size,
                                      prefix + "attention/output_projection/input_scale_divisor");
            }
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", TextConfig::gdn_input_rows, TextConfig::hidden,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {TextConfig::gdn_value_head_dim});
            if (is_bf16_gdn_output(layer)) {
                target.gdn.output =
                    bind_weight(binder, prefix + "gdn/output", NumericFormat::BF16, {TextConfig::hidden, TextConfig::value_dim});
            } else {
                target.gdn.output =
                    bind_nvfp4_weight(binder, prefix + "gdn/output", TextConfig::hidden, TextConfig::value_dim,
                                      prefix + "gdn/output_projection/input_scale_divisor");
            }
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", TextConfig::hidden, TextConfig::intermediate,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

void bind_qwen38_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", kFp8, {TextConfig::attention_input_rows, TextConfig::hidden}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kFp8, {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {TextConfig::gdn_control_rows, TextConfig::hidden}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", kFp8, {TextConfig::gdn_input_rows, TextConfig::hidden}),
            };
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {TextConfig::gdn_value_head_dim});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kFp8, {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        if (layer < 56) {
            target.mlp.gate_up =
                bind_nvfp4_weight(binder, prefix + "mlp/gate_up", TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden,
                                  prefix + "mlp/gate_up_projection/input_scale_divisor");
            target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", TextConfig::hidden, TextConfig::intermediate,
                                                prefix + "mlp/down_projection/input_scale_divisor");
        } else {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kFp8, {TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden});
            target.mlp.down    = bind_weight(binder, prefix + "mlp/down", kFp8, {TextConfig::hidden, TextConfig::intermediate});
        }
    }
}

DFlash2Plan bind_dflash2(artifact::Binder& binder, artifact::TensorPlacement placement) {
    const auto bind_tensor = [&](std::string_view name, NumericFormat format,
                                 std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    DFlash2Plan out;
    out.feature_projection = bind_weight(binder, "dflash2/feature_projection",
                                         NumericFormat::W8G32_F16S, {DFlashConfig::hidden, DFlashConfig::feature_rows}, placement);
    out.context_norm       = bind_tensor("dflash2/context_norm", NumericFormat::BF16, {DFlashConfig::hidden});
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        DFlash2LayerPlan& target = out.layers[layer];
        const std::string prefix = "dflash2/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_tensor(prefix + "input_norm", NumericFormat::BF16, {DFlashConfig::hidden});
        target.attention_conv.base_kernel =
            bind_tensor(prefix + "attention_conv/base_kernel", NumericFormat::BF16, {2, 2, DFlashConfig::hidden});
        target.attention_conv.kernel_projection =
            bind_weight(binder, prefix + "attention_conv/kernel_projection", NumericFormat::BF16,
                        {DFlashConfig::dynamic_conv_rows, DFlashConfig::hidden}, placement);
        target.query_key_value = bind_weight(binder, prefix + "attention/query_key_value",
                                             NumericFormat::W8G32_F16S, {DFlashConfig::query_key_value_rows, DFlashConfig::hidden}, placement);
        target.query_norm =
            bind_tensor(prefix + "attention/query_norm", NumericFormat::BF16, {DFlashConfig::head_dim});
        target.key_norm = bind_tensor(prefix + "attention/key_norm", NumericFormat::BF16, {DFlashConfig::head_dim});
        target.attention_output = bind_weight(binder, prefix + "attention/output",
                                              NumericFormat::W8G32_F16S, {DFlashConfig::hidden, DFlashConfig::query_size}, placement);
        target.post_attention_norm =
            bind_tensor(prefix + "post_attention_norm", NumericFormat::BF16, {DFlashConfig::hidden});
        target.mlp_conv.base_kernel =
            bind_tensor(prefix + "mlp_conv/base_kernel", NumericFormat::BF16, {2, 2, DFlashConfig::hidden});
        target.mlp_conv.kernel_projection =
            bind_weight(binder, prefix + "mlp_conv/kernel_projection", NumericFormat::BF16,
                        {DFlashConfig::dynamic_conv_rows, DFlashConfig::hidden}, placement);
        target.gate_up = bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                     {DFlashConfig::mlp_gate_up_rows, DFlashConfig::hidden}, placement);
        target.down    = bind_weight(binder, prefix + "mlp/down", NumericFormat::W8G32_F16S,
                                     {DFlashConfig::hidden, DFlashConfig::intermediate}, placement);
    }
    out.final_norm = bind_tensor("dflash2/final_norm", NumericFormat::BF16, {DFlashConfig::hidden});
    out.candidate_selector.hidden_projection =
        bind_weight(binder, "dflash2/candidate_selector/hidden_projection", NumericFormat::BF16,
                    {DFlashConfig::selector_dim, DFlashConfig::hidden}, placement);
    out.candidate_selector.predecessor_codebook = bind_tensor(
        "dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16, {TextConfig::output_rows, DFlashConfig::selector_dim});
    out.candidate_selector.successor_codebook = bind_tensor(
        "dflash2/candidate_selector/successor_codebook", NumericFormat::BF16, {TextConfig::output_rows, DFlashConfig::selector_dim});
    return out;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = TextConfig::draft_head_rows;
    constexpr std::size_t kTokenizerVocab = TextConfig::token_domain;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

void validate_artifact_model_shape(const artifact::Binder& binder, const ModelShape& expected) {
    if (expected.hidden <= 0 || expected.layers <= 0 || expected.intermediate <= 0 ||
        expected.output_rows <= 0 || expected.full_attention_interval <= 0) {
        throw std::invalid_argument("qwen3_6_27b: invalid expected model shape");
    }

    const auto describe = [](const auto& dims) {
        std::string output = "[";
        for (const auto dim : dims) {
            if (output.size() != 1) { output += ", "; }
            output += std::to_string(dim);
        }
        return output + "]";
    };
    const auto require = [&](std::string_view name,
                             std::initializer_list<std::uint64_t> shape) {
        const auto* object = binder.find(name);
        const auto* tensor =
            object == nullptr ? nullptr : std::get_if<artifact::TensorDescriptor>(object);
        if (tensor == nullptr) {
            throw artifact::ArtifactError("qwen3_6_27b model shape: missing tensor '" +
                                          std::string(name) + "'");
        }
        if (tensor->shape.size() != shape.size() ||
            !std::equal(shape.begin(), shape.end(), tensor->shape.begin())) {
            throw artifact::ArtifactError("qwen3_6_27b model shape: '" + std::string(name) +
                                          "' expected " + describe(shape) + ", artifact declares " +
                                          describe(tensor->shape));
        }
    };

    // v2 artifacts advertise geometry through tensor descriptors; there is no model JSON
    // exposed by Binder. Check the backbone's independent dimensions and topology before any
    // object is consumed, so a 2B/4B/9B artifact cannot enter a 27B Program accidentally.
    require("text/token_embedding", {static_cast<std::uint64_t>(expected.output_rows),
                                      static_cast<std::uint64_t>(expected.hidden)});
    require("text/final_norm", {static_cast<std::uint64_t>(expected.hidden)});
    require("text/output_head", {static_cast<std::uint64_t>(expected.output_rows),
                                  static_cast<std::uint64_t>(expected.hidden)});

    for (int layer = 0; layer < expected.layers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        require(prefix + "input_norm", {static_cast<std::uint64_t>(expected.hidden)});
        require(prefix + "post_attention_norm", {static_cast<std::uint64_t>(expected.hidden)});
        require(prefix + "mlp/gate_up", {static_cast<std::uint64_t>(2 * expected.intermediate),
                                         static_cast<std::uint64_t>(expected.hidden)});
        require(prefix + "mlp/down", {static_cast<std::uint64_t>(expected.hidden),
                                      static_cast<std::uint64_t>(expected.intermediate)});
        if (expected.is_full_attention(layer)) {
            if (binder.contains(prefix + "attention/query_key_gate_value")) {
                require(prefix + "attention/query_key_gate_value",
                        {static_cast<std::uint64_t>(expected.attention_input_rows()),
                         static_cast<std::uint64_t>(expected.hidden)});
            } else {
                require(prefix + "attention/query_key",
                        {static_cast<std::uint64_t>(expected.query_size() + expected.kv_size()),
                         static_cast<std::uint64_t>(expected.hidden)});
                require(prefix + "attention/gate_value",
                        {static_cast<std::uint64_t>(expected.query_size() + expected.kv_size()),
                         static_cast<std::uint64_t>(expected.hidden)});
            }
            require(prefix + "attention/query_norm", {static_cast<std::uint64_t>(expected.head_dim)});
            require(prefix + "attention/key_norm", {static_cast<std::uint64_t>(expected.head_dim)});
            require(prefix + "attention/output", {static_cast<std::uint64_t>(expected.hidden),
                                                   static_cast<std::uint64_t>(expected.query_size())});
        } else {
            require(prefix + "gdn/a_log", {static_cast<std::uint64_t>(expected.gdn_value_heads)});
            require(prefix + "gdn/dt_bias", {static_cast<std::uint64_t>(expected.gdn_value_heads)});
            require(prefix + "gdn/convolution",
                    {static_cast<std::uint64_t>(expected.gdn_conv_kernel),
                     static_cast<std::uint64_t>(expected.convolution_dim())});
            if (binder.contains(prefix + "gdn/a_b_projection")) {
                require(prefix + "gdn/a_b_projection",
                        {static_cast<std::uint64_t>(2 * expected.gdn_value_heads),
                         static_cast<std::uint64_t>(expected.hidden)});
            } else {
                require(prefix + "gdn/a_projection", {static_cast<std::uint64_t>(expected.gdn_value_heads),
                                                       static_cast<std::uint64_t>(expected.hidden)});
                require(prefix + "gdn/b_projection", {static_cast<std::uint64_t>(expected.gdn_value_heads),
                                                       static_cast<std::uint64_t>(expected.hidden)});
            }
            if (binder.contains(prefix + "gdn/query_key_value_z")) {
                require(prefix + "gdn/query_key_value_z",
                        {static_cast<std::uint64_t>(expected.gdn_input_rows()),
                         static_cast<std::uint64_t>(expected.hidden)});
            } else {
                require(prefix + "gdn/query_key", {static_cast<std::uint64_t>(2 * expected.key_dim()),
                                                    static_cast<std::uint64_t>(expected.hidden)});
                require(prefix + "gdn/value_z", {static_cast<std::uint64_t>(2 * expected.value_dim()),
                                                  static_cast<std::uint64_t>(expected.hidden)});
            }
            require(prefix + "gdn/norm", {static_cast<std::uint64_t>(expected.gdn_value_head_dim)});
            require(prefix + "gdn/output", {static_cast<std::uint64_t>(expected.hidden),
                                             static_cast<std::uint64_t>(expected.value_dim())});
        }
    }

    const std::string extra = "text/layers/" + std::to_string(expected.layers) + "/input_norm";
    if (binder.contains(extra)) {
        throw artifact::ArtifactError("qwen3_6_27b model shape: artifact declares more than " +
                                      std::to_string(expected.layers) + " text layers");
    }
}

void validate_artifact_model_shape(const artifact::Binder& binder) {
    validate_artifact_model_shape(binder, kModelShape);
}

// Defined below, next to the materialization pass that consumes it.
std::optional<HadamardSignsPlan> bind_hadamard_signs(artifact::Binder& binder);

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    validate_artifact_model_shape(binder);
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);

    // A masked draft builds every proposal through the fused linear_topk, whose FULL-head arm
    // registers only W8 and FP8 (ops/linear_topk, where that list lives). When the artifact's
    // stored vocabulary head is some other member of the grouped row-split family -- a ternary
    // PQ2 head, in this port's case -- there is no profile to run and `--spec dflash2` on the
    // default full head cannot start. Today that fails LATE and obscurely: the 10 GB load, the
    // plan, the CUDA graphs, and then "linear_topk: unsupported head profile" from the first
    // proposal, naming neither the flag nor the model.
    //
    // The format has to be PEEKED, not taken from vocabulary_format. endpoint_format() names the
    // format the loader's plan asks for, and require_tensor() accepts any member of the family
    // (the members are container-compatible -- see its comment in artifact/binder.h); the
    // materialiser then stamps the Weight with what the artifact REALLY declares, not with what
    // the plan named. Those two disagreed by exactly this much here, which is why a check written
    // against vocabulary_format silently passed while the run went on to fail. binder.find() is
    // the documented way to learn the stored format before binding, and this is that check.
    //
    // The optimized draft head is the answer rather than a fallback: every weights profile below
    // requires text/draft_head of the artifact (it is bound unconditionally), so the flag is
    // always available to the user, and choosing it here rather than in the engine would change
    // the proposal distribution without saying so.
    if (features.masked_draft() && features.proposal_head == ProposalHead::Full) {
        const auto* descriptor = binder.find("text/output_head");
        const auto* stored =
            descriptor == nullptr ? nullptr : std::get_if<artifact::TensorDescriptor>(descriptor);
        // A missing or mis-shaped object is bind_weight()'s error to raise with its own message;
        // this only reads the format, and only when there is one to read.
        if (stored != nullptr &&
            !ops::linear_topk_serves_full_head(artifact::qtype_for(stored->format), TextConfig::output_rows, TextConfig::hidden)) {
            throw std::invalid_argument(
                "--spec dflash2 requires --lm-head-draft: this artifact stores its vocabulary "
                "head in a format the fused linear_topk has no profile for (it registers W8 and "
                "FP8 heads only), so the full head cannot serve as the proposal head. The flag "
                "selects the artifact's dedicated draft head instead.");
        }
    }

    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format, {TextConfig::output_rows, TextConfig::hidden});
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen36Nvfp4:
        bind_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4:
        bind_qwen38_nvfp4_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format, {TextConfig::output_rows, TextConfig::hidden});
    out.hadamard_signs = bind_hadamard_signs(binder);
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {TextConfig::draft_head_rows, TextConfig::hidden}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {TextConfig::draft_head_rows}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {TextConfig::hidden, TextConfig::mtp_input_rows});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {TextConfig::attention_input_rows, TextConfig::hidden});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {TextConfig::hidden, TextConfig::query_size});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {TextConfig::hidden, TextConfig::intermediate}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {TextConfig::hidden});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {TextConfig::hidden, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {TextConfig::hidden}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    const bool has_dflash2 = binder.contains("dflash2/feature_projection");
    if (features.dflash2() && !has_dflash2) {
        throw artifact::ArtifactError(
            "DFlash2 was selected but the artifact has no DFlash2 weight bundle");
    }
    if (has_dflash2) {
        const artifact::TensorPlacement placement = features.dflash2()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
        out.dflash2                               = bind_dflash2(binder, placement);
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

// Bind the folded sign table when the artifact carries one. Artifacts whose weights are not
// folded simply omit these objects, and every object must be bound or Binder::finish() rejects
// the artifact as unconsumed -- so this is not optional when the objects are present.
std::optional<HadamardSignsPlan> bind_hadamard_signs(artifact::Binder& binder) {
    if (!binder.contains("text/hadamard_signs")) { return std::nullopt; }

    HadamardSignsPlan plan;
    plan.values = artifact::bind_device_tensor(binder, "text/hadamard_signs", NumericFormat::FP32,
                                               {kHadamardSignValues});
    plan.widths = artifact::bind_tensor(binder, "text/hadamard_widths", NumericFormat::I32,
                                        {kHadamardWidthCount},
                                        artifact::TensorPlacement::ValidateOnly);

    // The widths only need to reach the host: they are read here to derive each width's element
    // offset, which is what lets a weight find its block from its input dimension alone.
    const artifact::PayloadSpan payload = binder.payload(plan.widths);
    if (payload.data.size() < kHadamardWidthCount * sizeof(std::uint32_t)) {
        throw artifact::ArtifactError("text/hadamard_widths is shorter than its declared count");
    }
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < kHadamardWidthCount; ++i) {
        const std::uint32_t width = read_u32_le(payload.data, i * sizeof(std::uint32_t),
                                                "text/hadamard_widths");
        if (width == 0) {
            throw artifact::ArtifactError("text/hadamard_widths entries must be positive");
        }
        plan.width_offsets.emplace_back(static_cast<std::int32_t>(width), offset);
        offset += width;
    }
    if (offset != kHadamardSignValues) {
        throw artifact::ArtifactError("text/hadamard_widths must sum to " +
                                      std::to_string(kHadamardSignValues) + ", got " +
                                      std::to_string(offset));
    }
    return plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);
    // Install the folded sign table for the whole materialization pass below.
    const FoldedSignsScope folded_signs_scope(plan.hadamard_signs, backing);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding, TextConfig::output_rows, TextConfig::hidden);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {TextConfig::hidden});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {TextConfig::head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {TextConfig::head_dim});
            target.output     = materialized_weight(backing, source.attention.output, TextConfig::hidden, TextConfig::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {TextConfig::hidden});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {TextConfig::gdn_value_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, TextConfig::hidden, TextConfig::value_dim);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows, TextConfig::hidden);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, TextConfig::draft_head_rows, TextConfig::hidden);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {TextConfig::draft_head_rows});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, TextConfig::hidden,
            TextConfig::mtp_input_rows);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, TextConfig::attention_input_rows, TextConfig::hidden);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, TextConfig::query_size);
        mtp.attention.key         = row_view(mtp.attention.packed, TextConfig::query_size, TextConfig::kv_size);
        mtp.attention.output_gate = row_view(mtp.attention.packed, (TextConfig::query_size + TextConfig::kv_size), TextConfig::query_size);
        mtp.attention.value       = row_view(mtp.attention.packed, (2 * TextConfig::query_size + TextConfig::kv_size), TextConfig::kv_size);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.output              = artifact::materialized_weight(backing, plan.mtp.output,
                                                                NumericFormat::W8G32_F16S, TextConfig::hidden, TextConfig::query_size);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {TextConfig::hidden});
    }

    if (plan.features.dflash2()) {
        if (!plan.dflash2) {
            throw std::logic_error("selected DFlash2 weights are absent from the binding plan");
        }
        const DFlash2Plan& source = *plan.dflash2;
        auto& dflash2             = runtime.dflash.emplace();
        dflash2.feature_projection =
            materialized_weight(backing, source.feature_projection, DFlashConfig::hidden, DFlashConfig::feature_rows);
        dflash2.context_norm = artifact::materialized_tensor(backing, source.context_norm,
                                                             NumericFormat::BF16, {DFlashConfig::hidden});
        for (std::size_t layer = 0; layer < dflash2.layers.size(); ++layer) {
            const DFlash2LayerPlan& layer_source = source.layers[layer];
            qwen3_6::DFlash2LayerWeights& target = dflash2.layers[layer];
            target.input_norm = artifact::materialized_tensor(backing, layer_source.input_norm,
                                                              NumericFormat::BF16, {DFlashConfig::hidden});
            target.attention_conv.base_kernel =
                artifact::materialized_tensor(backing, layer_source.attention_conv.base_kernel,
                                              NumericFormat::BF16, {DFlashConfig::hidden, 2, 2});
            target.attention_conv.kernel_projection = materialized_weight(
                backing, layer_source.attention_conv.kernel_projection, DFlashConfig::dynamic_conv_rows, DFlashConfig::hidden);
            target.query_key_value =
                materialized_weight(backing, layer_source.query_key_value, DFlashConfig::query_key_value_rows, DFlashConfig::hidden);
            target.context_key   = row_view(target.query_key_value, DFlashConfig::query_size, DFlashConfig::kv_size);
            target.context_value = row_view(target.query_key_value, DFlashConfig::query_size + DFlashConfig::kv_size,
                                            DFlashConfig::kv_size);
            target.query_norm    = artifact::materialized_tensor(backing, layer_source.query_norm,
                                                                 NumericFormat::BF16, {DFlashConfig::head_dim});
            target.key_norm      = artifact::materialized_tensor(backing, layer_source.key_norm,
                                                                 NumericFormat::BF16, {DFlashConfig::head_dim});
            target.attention_output =
                materialized_weight(backing, layer_source.attention_output, DFlashConfig::hidden, DFlashConfig::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, layer_source.post_attention_norm, NumericFormat::BF16, {DFlashConfig::hidden});
            target.mlp_conv.base_kernel = artifact::materialized_tensor(
                backing, layer_source.mlp_conv.base_kernel, NumericFormat::BF16, {DFlashConfig::hidden, 2, 2});
            target.mlp_conv.kernel_projection =
                materialized_weight(backing, layer_source.mlp_conv.kernel_projection, DFlashConfig::dynamic_conv_rows, DFlashConfig::hidden);
            target.gate_up = materialized_weight(backing, layer_source.gate_up, DFlashConfig::mlp_gate_up_rows, DFlashConfig::hidden);
            target.down    = materialized_weight(backing, layer_source.down, DFlashConfig::hidden, DFlashConfig::intermediate);
        }
        dflash2.final_norm =
            artifact::materialized_tensor(backing, source.final_norm, NumericFormat::BF16, {DFlashConfig::hidden});
        dflash2.candidate_selector.hidden_projection =
            materialized_weight(backing, source.candidate_selector.hidden_projection, DFlashConfig::selector_dim, DFlashConfig::hidden);
        dflash2.candidate_selector.predecessor_codebook =
            artifact::materialized_tensor(backing, source.candidate_selector.predecessor_codebook,
                                          NumericFormat::BF16, {DFlashConfig::selector_dim, TextConfig::output_rows});
        dflash2.candidate_selector.successor_codebook =
            artifact::materialized_tensor(backing, source.candidate_selector.successor_codebook,
                                          NumericFormat::BF16, {DFlashConfig::selector_dim, TextConfig::output_rows});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, TextConfig::hidden, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {TextConfig::hidden});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
