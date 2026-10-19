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
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count);

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    if (plan.parent_rows > 0) {
        if (rows != plan.selected_rows || columns != plan.parent_columns ||
            plan.first_row < 0 || plan.first_row + rows > plan.parent_rows) {
            throw artifact::ArtifactError("v3 logical weight range conflicts with parent shape");
        }
        if (plan.format == NumericFormat::NVFP4) {
            // A v3 parent packs 128-row-aligned semantic children into a single
            // blockscale-k16-m128x4 tensor. Its scale plane is NOT interleaved
            // with the code plane, so generic row slicing is incorrect.
            if (plan.first_row % 128 != 0 || rows % 128 != 0 || columns % 64 != 0) {
                throw artifact::ArtifactError("v3 NVFP4 logical view is not M128/K64 aligned");
            }
            require_positive_finite(plan.weight_scale_divisor_bits, "v3 NVFP4 weight divisor");
            require_positive_finite(plan.input_scale_divisor_bits, "v3 NVFP4 input divisor");
            const std::array<std::uint64_t, 2> parent_shape{
                static_cast<std::uint64_t>(plan.parent_rows),
                static_cast<std::uint64_t>(columns)};
            const artifact::BlockScaleGeometry g =
                artifact::block_scale_geometry(NumericFormat::NVFP4, parent_shape);
            const auto* base = static_cast<const std::byte*>(materialized.device_data(plan.object));
            if (base == nullptr) {
                throw artifact::ArtifactError("v3 NVFP4 parent was not materialized on CUDA");
            }
            const std::uint64_t code_offset =
                static_cast<std::uint64_t>(plan.first_row) * columns / 2;
            const std::uint64_t scale_offset = g.scale_plane_offset +
                static_cast<std::uint64_t>(plan.first_row) * columns / 16;
            if (code_offset + static_cast<std::uint64_t>(rows) * columns / 2 > g.code_plane_bytes ||
                scale_offset + static_cast<std::uint64_t>(rows) * columns / 16 >
                    g.weight_divisor_offset) {
                throw artifact::ArtifactError("v3 NVFP4 semantic view exceeds parent planes");
            }
            Weight view{};
            view.payload = base;
            view.payload_bytes = g.encoded_bytes;
            view.qtype = QType::NVFP4;
            view.group_size = 16;
            view.ndim = 2;
            view.qdata = base + code_offset;
            view.scales = base + scale_offset;
            view.n = rows;
            view.k = columns;
            view.group = 16;
            view.layout = QuantLayout::BlockScaleK16M128x4;
            view.scale_dtype = DType::FP8_E4M3FN;
            view.shape[0] = view.padded_shape[0] = rows;
            view.shape[1] = view.padded_shape[1] = columns;
            view.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
            view.input_scale_divisor = std::bit_cast<float>(plan.input_scale_divisor_bits);
            return view;
        }
        Weight weight = artifact::materialized_weight(materialized, plan.object, plan.format,
                                                      plan.parent_rows, plan.parent_columns);
        if (rows != plan.parent_rows || plan.first_row != 0) {
            if (weight.layout == QuantLayout::GgufBlocks) {
                const auto block = gguf_block_shape(weight.qtype);
                const std::uint64_t bytes_per_row =
                    static_cast<std::uint64_t>(columns / block.elements) * block.bytes;
                const auto* origin = static_cast<const std::byte*>(weight.qdata);
                weight.qdata = origin + static_cast<std::uint64_t>(plan.first_row) * bytes_per_row;
                weight.payload = weight.qdata;
                weight.payload_bytes = static_cast<std::uint64_t>(rows) * bytes_per_row;
                weight.n = rows;
                weight.shape[0] = rows;
                weight.padded_shape[0] = rows;
            } else if (weight.layout == QuantLayout::RowSplit) {
                weight = row_view(weight, plan.first_row, rows);
            } else if (weight.layout == QuantLayout::Contiguous) {
                const std::size_t word_bytes =
                    plan.format == NumericFormat::BF16 ? 2 : 4;
                const auto* first = static_cast<const std::byte*>(weight.qdata) +
                                    static_cast<std::size_t>(plan.first_row) * columns * word_bytes;
                weight.payload = weight.qdata = first;
                weight.payload_bytes = static_cast<std::uint64_t>(rows) * columns * word_bytes;
                weight.n = rows;
                weight.shape[0] = rows;
                weight.padded_shape[0] = rows;
            } else {
                throw artifact::ArtifactError("v3 partial tensor format is not row-sliceable");
            }
        }
        if (plan.input_columns) {
            weight.input_columns =
                static_cast<const std::int32_t*>(materialized.device_data(*plan.input_columns));
        }
        return weight;
    }
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
    if (plan.gate && plan.up) {
        out.gate = materialized_weight(materialized, *plan.gate, TextConfig::intermediate,
                                        TextConfig::hidden);
        out.up = materialized_weight(materialized, *plan.up, TextConfig::intermediate,
                                      TextConfig::hidden);
    } else {
        out.gate_up = materialized_weight(materialized, plan.gate_up,
                                         TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden);
    }
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
    if (const auto* individual =
            std::get_if<IndividualAttentionProjectionPlan>(&plan.projection)) {
        return IndividualAttentionProjectionPayload{
            .query = materialized_weight(materialized, individual->query,
                                          TextConfig::query_size, TextConfig::hidden),
            .key = materialized_weight(materialized, individual->key, TextConfig::kv_size,
                                        TextConfig::hidden),
            .gate = materialized_weight(materialized, individual->gate,
                                         TextConfig::query_size, TextConfig::hidden),
            .value = materialized_weight(materialized, individual->value,
                                          TextConfig::kv_size, TextConfig::hidden),
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
    if (const auto* individual =
            std::get_if<IndividualGdnInputProjectionPlan>(&plan.input_projection)) {
        return IndividualGdnInputProjectionPayload{
            .query = materialized_weight(materialized, individual->query,
                                          TextConfig::key_dim, TextConfig::hidden),
            .key = materialized_weight(materialized, individual->key,
                                        TextConfig::key_dim, TextConfig::hidden),
            .value = materialized_weight(materialized, individual->value,
                                          TextConfig::value_dim, TextConfig::hidden),
            .z = materialized_weight(materialized, individual->z,
                                      TextConfig::value_dim, TextConfig::hidden),
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

namespace {

using Json = nlohmann::json;

std::uint64_t v3_unsigned(const Json& value, std::string_view label) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw artifact::ArtifactError(std::string(label) + ": expected nonnegative integer");
    }
    return value.get<std::uint64_t>();
}

std::string v3_string(const Json& value, std::string_view label) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        throw artifact::ArtifactError(std::string(label) + ": expected nonempty string");
    }
    return value.get<std::string>();
}

NumericFormat v3_format(std::string_view spelling) {
    if (spelling == "bf16") { return NumericFormat::BF16; }
    if (spelling == "fp32") { return NumericFormat::FP32; }
    if (spelling == "int32") { return NumericFormat::I32; }
    if (spelling == "q2_g64_fp16") { return NumericFormat::Q2G64_F16S; }
    if (spelling == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (spelling == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (spelling == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (spelling == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (spelling == "t2_g128_fp16") { return NumericFormat::T2G128_F16S; }
    if (spelling == "nvfp4") { return NumericFormat::NVFP4; }
    if (spelling == "fp8_e4m3fn_row_bf16") {
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    }
    if (spelling == "fp8_e4m3fn_row_fp16") {
        return NumericFormat::FP8_E4M3FN_ROW_FP16S;
    }
    for (int code = static_cast<int>(NumericFormat::GGUF_Q8_0);
         code <= static_cast<int>(NumericFormat::GGUF_Q2_0); ++code) {
        const auto format = static_cast<NumericFormat>(code);
        if (artifact::format_name(format) == spelling) { return format; }
    }
    throw artifact::ArtifactError("qwen3_6_27b: unsupported v3 numeric format: " +
                                  std::string(spelling));
}

artifact::StorageLayout v3_layout(std::string_view spelling) {
    if (spelling == "contiguous_le_v1") {
        return artifact::StorageLayout::ContiguousLeV1;
    }
    if (spelling == "row_split_k128_v1") {
        return artifact::StorageLayout::RowSplitK128V1;
    }
    if (spelling == "block_scale_k16_m128x4_v1") {
        return artifact::StorageLayout::BlockScaleK16M128x4V1;
    }
    if (spelling == "row_scale_v1") {
        return artifact::StorageLayout::RowScaleV1;
    }
    if (spelling == "row_interleaved_v1") {
        return artifact::StorageLayout::RowInterleavedV1;
    }
    if (spelling == "gguf_blocks_v1") {
        return artifact::StorageLayout::GgufBlocksV1;
    }
    throw artifact::ArtifactError("qwen3_6_27b: unsupported v3 tensor layout: " +
                                  std::string(spelling));
}

void require_v3_shape(const Json& shape, std::initializer_list<std::uint64_t> expected,
                      std::string_view label) {
    if (!shape.is_array() || shape.size() != expected.size()) {
        throw artifact::ArtifactError(std::string(label) + ": wrong logical tensor rank");
    }
    std::size_t i = 0;
    for (auto dim : expected) {
        if (v3_unsigned(shape.at(i++), label) != dim) {
            throw artifact::ArtifactError(std::string(label) + ": declared tensor shape differs");
        }
    }
}

void validate_v3_text_config(const Json& root, const ModelShape& s) {
    const auto& components = root.at("components");
    const auto& text = components.at("text");
    const auto& config = text.at("config");
    const auto require_dim = [&](const char* field, int expected) {
        if (!config.contains(field) || v3_unsigned(config.at(field), field) !=
                                             static_cast<std::uint64_t>(expected)) {
            throw artifact::ArtifactError(std::string("components.text.config.") + field +
                                          ": incompatible 27B Program geometry");
        }
    };
    require_dim("hidden_size", s.hidden);
    require_dim("intermediate_size", s.intermediate);
    require_dim("num_hidden_layers", s.layers);
    require_dim("vocab_size", s.output_rows);
    require_dim("num_attention_heads", s.query_heads);
    require_dim("num_key_value_heads", s.kv_heads);
    require_dim("head_dim", s.head_dim);
    require_dim("linear_num_key_heads", s.gdn_key_heads);
    require_dim("linear_key_head_dim", s.gdn_key_head_dim);
    require_dim("linear_num_value_heads", s.gdn_value_heads);
    require_dim("linear_value_head_dim", s.gdn_value_head_dim);
    require_dim("linear_conv_kernel_dim", s.gdn_conv_kernel);
    require_dim("max_position_embeddings", static_cast<int>(kNativeContext));
    if (config.value("model_type", std::string{}) != "qwen3_5_text" ||
        config.value("tie_word_embeddings", true)) {
        throw artifact::ArtifactError("components.text.config: unsupported model family or tied head");
    }
    const auto& types = config.at("layer_types");
    if (!types.is_array() || types.size() != static_cast<std::size_t>(s.layers)) {
        throw artifact::ArtifactError("components.text.config.layer_types: wrong layer count");
    }
    for (int layer = 0; layer < s.layers; ++layer) {
        const std::string expected = s.is_full_attention(layer) ? "full_attention" :
                                                                "linear_attention";
        if (v3_string(types.at(layer), "layer_types") != expected) {
            throw artifact::ArtifactError("components.text.config.layer_types[" +
                                          std::to_string(layer) + "]: incompatible topology");
        }
    }
    const auto& rope = config.at("rope_parameters");
    if (!rope.is_object() || !rope.contains("rope_theta") ||
        !rope.at("rope_theta").is_number() ||
        std::abs(rope.at("rope_theta").get<double>() - TextConfig::rope_theta) > 1 ||
        !rope.contains("partial_rotary_factor") ||
        !rope.at("partial_rotary_factor").is_number() ||
        std::abs(rope.at("partial_rotary_factor").get<double>() -
                 double(s.rotary_dim) / s.head_dim) > 1e-6) {
        throw artifact::ArtifactError("components.text.config.rope_parameters: incompatible RoPE");
    }
    if (!config.contains("rms_norm_eps") || !config.at("rms_norm_eps").is_number() ||
        std::abs(config.at("rms_norm_eps").get<double>() - TextConfig::rms_epsilon) > 1e-10) {
        throw artifact::ArtifactError("components.text.config.rms_norm_eps: incompatible RMSNorm");
    }
}

} // namespace

V3TextMetadata inspect_v3_text_metadata(const nlohmann::json& root) {
    // The v3 container layer validates general schema, but model-specific geometry and semantic
    // role ownership belong here. This inspector uses JSON alone, so a partial 15GB download
    // can be checked from its 376800-byte directory without mmap'ing any weight.
    validate_v3_text_config(root, kModelShape);
    const Json& raw_objects = root.at("objects");
    const Json& bindings    = root.at("bindings");
    const Json& uses        = root.at("uses");
    if (!raw_objects.is_array() || !bindings.is_object() || !uses.is_array()) {
        throw artifact::ArtifactError("v3 text metadata requires objects, bindings and uses");
    }
    std::unordered_map<std::string, const Json*> objects;
    objects.reserve(raw_objects.size());
    for (const auto& object : raw_objects) {
        const auto& name = v3_string(object.at("id"), "v3 object id");
        if (!objects.emplace(name, &object).second) {
            throw artifact::ArtifactError("v3 text duplicate object: " + name);
        }
    }
    std::map<std::pair<std::string, std::string>, const Json*> use_index;
    for (const auto& use : uses) {
        const auto parameter = v3_string(use.at("parameter"), "Use parameter");
        const auto input     = v3_string(use.at("input"), "Use input");
        if (!use_index.emplace(std::make_pair(parameter, input), &use).second) {
            throw artifact::ArtifactError("v3 text duplicate Use: " + parameter + "@" + input);
        }
    }

    V3TextMetadata manifest;
    manifest.tensors.reserve(963);
    const auto add = [&](const std::string& parameter,
                         std::initializer_list<std::uint64_t> shape,
                         std::string_view expected_input = {}) {
        if (!bindings.contains(parameter)) {
            throw artifact::ArtifactError("v3 text missing binding " + parameter);
        }
        const auto& binding = bindings.at(parameter);
        if (!binding.is_object()) {
            throw artifact::ArtifactError(parameter + ": binding must be an object");
        }
        std::string id;
        std::uint64_t begin = 0;
        std::uint64_t end   = 0;
        bool partial = false;
        if (binding.size() == 1 && binding.contains("object")) {
            id = v3_string(binding.at("object"), parameter);
        } else if (binding.size() == 1 && binding.contains("parts") &&
                   binding.at("parts").is_array() && binding.at("parts").size() == 1) {
            const auto& piece = binding.at("parts").at(0);
            id = v3_string(piece.at("object"), parameter);
            const auto& range = piece.at("range");
            if (!range.is_array() || range.size() != 2) {
                throw artifact::ArtifactError(parameter + ": partial binding requires [begin,end]");
            }
            begin = v3_unsigned(range.at(0), parameter);
            end   = v3_unsigned(range.at(1), parameter);
            partial = true;
        } else {
            throw artifact::ArtifactError(parameter + ": expected whole or single contiguous part");
        }
        const auto found = objects.find(id);
        if (found == objects.end() || found->second->value("kind", std::string{}) != "tensor") {
            throw artifact::ArtifactError(parameter + ": missing tensor object " + id);
        }
        const auto& tensor = *found->second;
        const auto format = v3_format(v3_string(tensor.at("format"), id));
        const auto layout = v3_layout(v3_string(tensor.at("layout"), id));
        const auto& source_shape = tensor.at("shape");
        if (!source_shape.is_array() || source_shape.size() != shape.size()) {
            throw artifact::ArtifactError(parameter + ": wrong physical tensor rank");
        }
        V3LogicalTensor out;
        out.parameter = parameter;
        out.object_id = id;
        out.format    = format;
        out.layout    = layout;
        std::vector<std::uint64_t> dimensions(shape);
        const auto rows = dimensions.at(0);
        const auto columns = dimensions.size() == 2 ? dimensions[1] : 1;
        out.rows = rows;
        out.columns = columns;
        out.parent_rows = v3_unsigned(source_shape.at(0), id);
        const std::uint64_t source_columns =
            source_shape.size() == 2 ? v3_unsigned(source_shape.at(1), id) : 1;
        if (source_columns != columns) {
            throw artifact::ArtifactError(parameter + ": physical input width differs");
        }
        if (partial) {
            if (shape.size() != 2 || begin >= end ||
                begin % columns || end % columns || end - begin != rows * columns ||
                end / columns > out.parent_rows) {
                throw artifact::ArtifactError(parameter + ": invalid row-part geometry");
            }
            out.first_row = begin / columns;
        } else {
            require_v3_shape(source_shape, shape, parameter);
        }
        if (!expected_input.empty()) {
            const auto use = use_index.find({parameter, std::string(expected_input)});
            if (use == use_index.end()) {
                throw artifact::ArtifactError(parameter + ": missing Use " +
                                              std::string(expected_input));
            }
            const auto& declaration = *use->second;
            if (declaration.contains("activation_policy")) {
                out.activation_policy =
                    v3_string(declaration.at("activation_policy"), parameter + " Use");
                if (out.activation_policy != "A16Only" && out.activation_policy != "AllowA8" &&
                    out.activation_policy != "AllowA4") {
                    throw artifact::ArtifactError(parameter + ": unsupported activation policy");
                }
            }
            if (declaration.contains("auxiliaries")) {
                const auto& auxiliary = declaration.at("auxiliaries");
                if (!auxiliary.is_object() || auxiliary.size() != 1) {
                    throw artifact::ArtifactError(parameter + ": unexpected Use auxiliary");
                }
                if (parameter.find("/gdn/output") != std::string::npos &&
                    auxiliary.contains("input_columns")) {
                    const auto& ref = auxiliary.at("input_columns");
                    if (!ref.is_object() || ref.size() != 1 || !ref.contains("object")) {
                        throw artifact::ArtifactError(parameter + ": invalid input_columns binding");
                    }
                    out.input_columns_object =
                        v3_string(ref.at("object"), parameter + " input_columns");
                    const auto columns_object = objects.find(*out.input_columns_object);
                    if (columns_object == objects.end() ||
                        columns_object->second->value("kind", std::string{}) != "tensor") {
                        throw artifact::ArtifactError(parameter + ": missing input_columns tensor");
                    }
                    const auto& source = *columns_object->second;
                    if (v3_format(v3_string(source.at("format"), parameter)) != NumericFormat::I32 ||
                        v3_layout(v3_string(source.at("layout"), parameter)) !=
                            artifact::StorageLayout::ContiguousLeV1) {
                        throw artifact::ArtifactError(parameter + ": invalid input_columns format");
                    }
                    require_v3_shape(source.at("shape"), {columns}, parameter + " input_columns");
                } else if (auxiliary.contains("activation_input_divisor") &&
                           format == NumericFormat::NVFP4) {
                    const auto& ref = auxiliary.at("activation_input_divisor");
                    if (!ref.is_object() || ref.size() != 1 || !ref.contains("object")) {
                        throw artifact::ArtifactError(parameter + ": invalid divisor reference");
                    }
                    out.input_divisor_object =
                        v3_string(ref.at("object"), parameter + " activation_input_divisor");
                    const auto scalar = objects.find(*out.input_divisor_object);
                    if (scalar == objects.end() ||
                        scalar->second->value("kind", std::string{}) != "tensor" ||
                        v3_format(v3_string(scalar->second->at("format"), parameter)) !=
                            NumericFormat::FP32 ||
                        v3_layout(v3_string(scalar->second->at("layout"), parameter)) !=
                            artifact::StorageLayout::ContiguousLeV1) {
                        throw artifact::ArtifactError(parameter + ": invalid FP32 divisor");
                    }
                    require_v3_shape(scalar->second->at("shape"), {},
                                     parameter + " input_divisor");
                } else {
                    throw artifact::ArtifactError(parameter + ": unexpected Use auxiliary");
                }
            }
        }
        manifest.tensors.push_back(std::move(out));
    };

    const auto& s = kModelShape;
    add("text/token_embedding", {std::uint64_t(s.output_rows), std::uint64_t(s.hidden)});
    add("text/output_head", {std::uint64_t(s.output_rows), std::uint64_t(s.hidden)},
        "text/final_hidden");
    add("text/final_norm", {std::uint64_t(s.hidden)});
    for (int layer = 0; layer < s.layers; ++layer) {
        const std::string p = "text/layers/" + std::to_string(layer) + "/";
        const std::string mixer = p + "mixer_input";
        add(p + "input_norm", {std::uint64_t(s.hidden)});
        add(p + "post_attention_norm", {std::uint64_t(s.hidden)});
        if (s.is_full_attention(layer)) {
            add(p + "attention/query", {std::uint64_t(s.query_size()), std::uint64_t(s.hidden)},
                mixer);
            add(p + "attention/gate", {std::uint64_t(s.query_size()), std::uint64_t(s.hidden)},
                mixer);
            add(p + "attention/key", {std::uint64_t(s.kv_size()), std::uint64_t(s.hidden)},
                mixer);
            add(p + "attention/value", {std::uint64_t(s.kv_size()), std::uint64_t(s.hidden)},
                mixer);
            add(p + "attention/query_norm", {std::uint64_t(s.head_dim)});
            add(p + "attention/key_norm", {std::uint64_t(s.head_dim)});
            add(p + "attention/output", {std::uint64_t(s.hidden), std::uint64_t(s.query_size())},
                p + "attention/gated_output");
        } else {
            add(p + "gdn/a_log", {std::uint64_t(s.gdn_value_heads)});
            add(p + "gdn/dt_bias", {std::uint64_t(s.gdn_value_heads)});
            add(p + "gdn/convolution", {std::uint64_t(s.gdn_conv_kernel),
                                        std::uint64_t(s.convolution_dim())});
            add(p + "gdn/a_projection",
                {std::uint64_t(s.gdn_value_heads), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/b_projection",
                {std::uint64_t(s.gdn_value_heads), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/query", {std::uint64_t(s.key_dim()), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/key", {std::uint64_t(s.key_dim()), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/value", {std::uint64_t(s.value_dim()), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/z", {std::uint64_t(s.value_dim()), std::uint64_t(s.hidden)}, mixer);
            add(p + "gdn/norm", {std::uint64_t(s.gdn_value_head_dim)});
            add(p + "gdn/output", {std::uint64_t(s.hidden), std::uint64_t(s.value_dim())},
                p + "gdn/gated_output");
        }
        add(p + "mlp/gate", {std::uint64_t(s.intermediate), std::uint64_t(s.hidden)},
            p + "ffn_input");
        add(p + "mlp/up", {std::uint64_t(s.intermediate), std::uint64_t(s.hidden)},
            p + "ffn_input");
        add(p + "mlp/down", {std::uint64_t(s.hidden), std::uint64_t(s.intermediate)},
            p + "mlp/product");
    }
    return manifest;
}

std::vector<V3LogicalTensor> inspect_v3_feature_metadata(
    const nlohmann::json& root, qwen3_6::StartupFeatures features) {
    const auto& components = root.at("components");
    const auto& objects = root.at("objects");
    const auto& bindings = root.at("bindings");
    const auto& uses = root.at("uses");
    std::unordered_map<std::string, const Json*> object_index;
    object_index.reserve(objects.size());
    for (const auto& object : objects) {
        object_index.emplace(v3_string(object.at("id"), "v3 object"), &object);
    }
    std::map<std::pair<std::string, std::string>, const Json*> use_index;
    for (const auto& use : uses) {
        use_index.emplace(
            std::make_pair(v3_string(use.at("parameter"), "Use parameter"),
                           v3_string(use.at("input"), "Use input")), &use);
    }
    std::vector<V3LogicalTensor> out;
    out.reserve(features.vision ? 470 : 100);
    const auto add = [&](const std::string& name,
                         std::initializer_list<std::uint64_t> dimensions,
                         std::string_view activation = {},
                         std::optional<NumericFormat> required_format = std::nullopt)
        -> const V3LogicalTensor& {
        if (!bindings.contains(name)) {
            throw artifact::ArtifactError("v3 missing feature binding " + name);
        }
        const auto& declaration = bindings.at(name);
        std::string id;
        std::uint64_t start = 0;
        std::uint64_t stop = 0;
        bool part = false;
        if (declaration.is_object() && declaration.size() == 1 &&
            declaration.contains("object")) {
            id = v3_string(declaration.at("object"), name);
        } else if (declaration.is_object() && declaration.size() == 1 &&
                   declaration.contains("parts") && declaration.at("parts").is_array() &&
                   declaration.at("parts").size() == 1) {
            const auto& piece = declaration.at("parts").at(0);
            if (!piece.is_object() || piece.size() != 2 ||
                !piece.contains("object") || !piece.contains("range")) {
                throw artifact::ArtifactError(name + ": invalid single-parent row part");
            }
            id = v3_string(piece.at("object"), name);
            const auto& bounds = piece.at("range");
            if (!bounds.is_array() || bounds.size() != 2) {
                throw artifact::ArtifactError(name + ": part range must have two bounds");
            }
            start = v3_unsigned(bounds[0], name);
            stop  = v3_unsigned(bounds[1], name);
            part = true;
        } else {
            throw artifact::ArtifactError(name + ": requires whole or single-parent binding");
        }
        const auto found = object_index.find(id);
        if (found == object_index.end() ||
            found->second->value("kind", std::string{}) != "tensor") {
            throw artifact::ArtifactError(name + ": object missing or not tensor: " + id);
        }
        const auto& physical = *found->second;
        V3LogicalTensor logical;
        logical.parameter = name;
        logical.object_id = id;
        logical.format = v3_format(v3_string(physical.at("format"), name));
        logical.layout = v3_layout(v3_string(physical.at("layout"), name));
        if (required_format && logical.format != *required_format) {
            throw artifact::ArtifactError(name + ": unexpected physical encoding");
        }
        std::vector<std::uint64_t> shape(dimensions);
        if (shape.empty() || shape.size() > 3 || !physical.at("shape").is_array() ||
            physical.at("shape").size() != shape.size()) {
            throw artifact::ArtifactError(name + ": feature tensor rank differs");
        }
        logical.rows = shape.at(0);
        logical.columns = 1;
        for (std::size_t dim = 1; dim < shape.size(); ++dim) {
            logical.columns *= shape[dim];
        }
        logical.parent_rows = v3_unsigned(physical.at("shape")[0], name);
        std::uint64_t parent_columns = 1;
        for (std::size_t dim = 1; dim < shape.size(); ++dim) {
            parent_columns *= v3_unsigned(physical.at("shape")[dim], name);
        }
        if (parent_columns != logical.columns) {
            throw artifact::ArtifactError(name + ": physical matrix input width differs");
        }
        if (part) {
            if (shape.size() > 2) {
                throw artifact::ArtifactError(name + ": only vectors/matrices support row parts");
            }
            const std::uint64_t elements = logical.rows * logical.columns;
            if (start >= stop || start % logical.columns != 0 ||
                stop - start != elements || stop / logical.columns > logical.parent_rows) {
                throw artifact::ArtifactError(name + ": invalid row-aligned physical range");
            }
            logical.first_row = start / logical.columns;
        } else {
            require_v3_shape(physical.at("shape"), dimensions, name);
        }
        if (!activation.empty()) {
            const auto use_it = use_index.find({name, std::string(activation)});
            if (use_it == use_index.end()) {
                throw artifact::ArtifactError(name + ": missing Use " + std::string(activation));
            }
            const auto& use = *use_it->second;
            if (!use.contains("activation_policy")) {
                throw artifact::ArtifactError(name + ": missing activation policy");
            }
            logical.activation_policy = v3_string(use.at("activation_policy"), name);
            if (logical.activation_policy != "A16Only" &&
                logical.activation_policy != "AllowA8" &&
                logical.activation_policy != "AllowA4") {
                throw artifact::ArtifactError(name + ": unsupported activation policy");
            }
            if (use.contains("auxiliaries") && !use.at("auxiliaries").empty()) {
                throw artifact::ArtifactError(name + ": unsupported feature Use auxiliaries");
            }
        }
        out.push_back(std::move(logical));
        return out.back();
    };
    const auto dimension = [&](const Json& cfg, const char* field, int expected,
                               std::string_view component) {
        if (!cfg.contains(field) ||
            v3_unsigned(cfg.at(field), field) != static_cast<std::uint64_t>(expected)) {
            throw artifact::ArtifactError(std::string(component) + "." + field +
                                          ": incompatible 27B feature geometry");
        }
    };
    const auto target_component = [&](const char* name) -> const Json& {
        if (!components.contains(name) ||
            components.at(name).value("target", std::string{}) != "text") {
            throw artifact::ArtifactError(std::string("v3 ") + name +
                                          ": missing text target");
        }
        return components.at(name);
    };
    if (features.vision) {
        const auto& cfg = target_component("vision").at("config");
        if (cfg.value("model_type", std::string{}) != "qwen3_5_vision") {
            throw artifact::ArtifactError("v3 vision: unsupported model_type");
        }
        for (const auto& [name, value] :
             std::array<std::pair<const char*, int>, 8>{{
                 {"depth", qwen3_6::VisionBackboneConfig::layers},
                 {"hidden_size", qwen3_6::VisionBackboneConfig::hidden},
                 {"intermediate_size", qwen3_6::VisionBackboneConfig::intermediate},
                 {"num_heads", qwen3_6::VisionBackboneConfig::heads},
                 {"patch_size", 16}, {"temporal_patch_size", 2},
                 {"spatial_merge_size", qwen3_6::VisionBackboneConfig::merge},
                 {"num_position_embeddings", qwen3_6::VisionBackboneConfig::position_embeddings},
             }}) {
            dimension(cfg, name, value, "vision.config");
        }
        constexpr auto H = qwen3_6::VisionBackboneConfig::hidden;
        constexpr auto I = qwen3_6::VisionBackboneConfig::intermediate;
        constexpr auto M = qwen3_6::VisionBackboneConfig::merger_hidden;
        const auto vadd = [&](const std::string& n,
                              std::initializer_list<std::uint64_t> dims,
                              std::string_view input = {},
                              std::optional<NumericFormat> format = std::nullopt) {
            // The shared Vision ModelView materializes BF16 as contiguous tensors
            // and quantized projections as row-split planes. A matching codec with
            // another physical layout cannot be interpreted by its Vision kernels.
            if (!format) {
                throw std::logic_error("Vision bindings require an explicit encoding");
            }
            const V3LogicalTensor& logical = add(n, dims, input, format);
            const auto expected_layout =
                *format == NumericFormat::BF16
                    ? artifact::StorageLayout::ContiguousLeV1
                    : artifact::StorageLayout::RowSplitK128V1;
            if (logical.layout != expected_layout) {
                throw artifact::ArtifactError(n + ": unsupported Vision physical layout");
            }
            // Vision projections consume A16 activations; allowing a different
            // Use policy would silently promise a path that has no native kernel.
            if (!input.empty() && logical.activation_policy != "A16Only") {
                throw artifact::ArtifactError(n + ": unsupported Vision activation policy");
            }
            return logical;
        };
        vadd("vision/patch_embedding", {H, qwen3_6::VisionBackboneConfig::patch_dim},
             "vision/patch_input", NumericFormat::Q6G64_F16S);
        vadd("vision/patch_embedding_bias", {H}, {}, NumericFormat::BF16);
        vadd("vision/position_embedding",
             {qwen3_6::VisionBackboneConfig::position_embeddings, H}, {},
             NumericFormat::BF16);
        for (int i = 0; i < qwen3_6::VisionBackboneConfig::layers; ++i) {
            const std::string p = "vision/layers/" + std::to_string(i) + "/";
            for (const char* role : {"norm1_weight", "norm1_bias",
                                     "norm2_weight", "norm2_bias"}) {
                vadd(p + role, {H}, {}, NumericFormat::BF16);
            }
            const auto prefix = p + "attention/";
            const auto a = vadd(prefix + "query", {H, H}, p + "attention_input",
                                NumericFormat::Q4G64_F16S);
            const auto b = vadd(prefix + "key", {H, H}, p + "attention_input",
                                NumericFormat::Q4G64_F16S);
            const auto c = vadd(prefix + "value", {H, H}, p + "attention_input",
                                NumericFormat::Q4G64_F16S);
            if (a.object_id != b.object_id || b.object_id != c.object_id ||
                a.parent_rows != 3 * H || a.first_row != 0 ||
                b.first_row != H || c.first_row != 2 * H) {
                throw artifact::ArtifactError(p + ": Vision QKV parent or order differs");
            }
            const auto qb = vadd(prefix + "query_bias", {H}, {}, NumericFormat::BF16);
            const auto kb = vadd(prefix + "key_bias", {H}, {}, NumericFormat::BF16);
            const auto vb = vadd(prefix + "value_bias", {H}, {}, NumericFormat::BF16);
            if (qb.object_id != kb.object_id || kb.object_id != vb.object_id ||
                qb.parent_rows != 3 * H || qb.first_row != 0 ||
                kb.first_row != H || vb.first_row != 2 * H) {
                throw artifact::ArtifactError(p + ": Vision QKV bias parent or order differs");
            }
            vadd(prefix + "output", {H, H}, p + "attention_output",
                 NumericFormat::Q5G64_F16S);
            vadd(prefix + "output_bias", {H}, {}, NumericFormat::BF16);
            vadd(p + "mlp/fc1", {I, H}, p + "mlp_input", NumericFormat::Q4G64_F16S);
            vadd(p + "mlp/fc1_bias", {I}, {}, NumericFormat::BF16);
            vadd(p + "mlp/fc2", {H, I}, p + "mlp_activation", NumericFormat::Q5G64_F16S);
            vadd(p + "mlp/fc2_bias", {H}, {}, NumericFormat::BF16);
        }
        vadd("vision/merger/norm_weight", {H}, {}, NumericFormat::BF16);
        vadd("vision/merger/norm_bias", {H}, {}, NumericFormat::BF16);
        vadd("vision/merger/fc1", {M, M}, "vision/merger/input",
             NumericFormat::W8G32_F16S);
        vadd("vision/merger/fc1_bias", {M}, {}, NumericFormat::BF16);
        vadd("vision/merger/fc2", {TextConfig::hidden, M}, "vision/merger/activation",
             NumericFormat::W8G32_F16S);
        vadd("vision/merger/fc2_bias", {TextConfig::hidden}, {}, NumericFormat::BF16);
    }
    if (features.mtp()) {
        const auto& component = target_component("mtp");
        const auto& architectures = component.at("config").at("architectures");
        if (!architectures.is_array() || architectures.size() != 1 ||
            architectures.at(0) != "Qwen3_5MTP") {
            throw artifact::ArtifactError("v3 MTP: unsupported architecture");
        }
        // GSQ keeps MTP as GGUF Q6K; both Qwen3.6 v3 profiles preserve W8 row-split.
        const auto stem_binding = bindings.at("mtp/input_projection");
        const auto stem_id = v3_string(stem_binding.at("object"), "mtp input projection");
        const auto stem = object_index.find(stem_id);
        if (stem == object_index.end()) {
            throw artifact::ArtifactError("v3 MTP input projection parent missing");
        }
        const auto fmt = v3_format(v3_string(stem->second->at("format"), "mtp format"));
        if (fmt != NumericFormat::GGUF_Q6_K && fmt != NumericFormat::W8G32_F16S) {
            throw artifact::ArtifactError("v3 MTP requires GGUF Q6K or row-split W8");
        }
        const auto layout = fmt == NumericFormat::GGUF_Q6_K
                                ? artifact::StorageLayout::GgufBlocksV1
                                : artifact::StorageLayout::RowSplitK128V1;
        const auto add_projection = [&](const std::string& parameter,
                                        std::initializer_list<std::uint64_t> shape,
                                        std::string_view input) -> const V3LogicalTensor& {
            const auto& tensor = add(parameter, shape, input, fmt);
            if (tensor.layout != layout) {
                throw artifact::ArtifactError(parameter +
                                              ": v3 MTP projection format/layout mismatch");
            }
            return tensor;
        };
        const auto H = TextConfig::hidden;
        const auto Q = TextConfig::query_size;
        const auto K = TextConfig::kv_size;
        const auto I = TextConfig::intermediate;
        add_projection("mtp/input_projection", {H, 2 * H}, "mtp/stem_input");
        add("mtp/embedding_norm", {H}, {}, NumericFormat::BF16);
        add("mtp/hidden_norm", {H}, {}, NumericFormat::BF16);
        add("mtp/final_norm", {H}, {}, NumericFormat::BF16);
        const std::string p = "mtp/layers/0/";
        add(p + "input_norm", {H}, {}, NumericFormat::BF16);
        const auto prefix = p + "attention/";
        const auto q = add_projection(prefix + "query", {Q, H}, p + "mixer_input");
        const auto k = add_projection(prefix + "key", {K, H}, p + "mixer_input");
        const auto g = add_projection(prefix + "gate", {Q, H}, p + "mixer_input");
        const auto v = add_projection(prefix + "value", {K, H}, p + "mixer_input");
        if (q.object_id != k.object_id || q.object_id != g.object_id ||
            q.object_id != v.object_id || q.parent_rows != TextConfig::attention_input_rows ||
            q.first_row != 0 || k.first_row != Q ||
            g.first_row != Q + K || v.first_row != 2 * Q + K) {
            throw artifact::ArtifactError("v3 MTP attention packing differs");
        }
        add(prefix + "query_norm", {TextConfig::head_dim}, {}, NumericFormat::BF16);
        add(prefix + "key_norm", {TextConfig::head_dim}, {}, NumericFormat::BF16);
        add_projection(prefix + "output", {H, Q}, p + "attention/gated_output");
        add(p + "post_attention_norm", {H}, {}, NumericFormat::BF16);
        const auto gate = add_projection(p + "mlp/gate", {I, H}, p + "ffn_input");
        const auto up = add_projection(p + "mlp/up", {I, H}, p + "ffn_input");
        if (gate.object_id != up.object_id || gate.parent_rows != 2 * I ||
            gate.first_row != 0 || up.first_row != I) {
            throw artifact::ArtifactError("v3 MTP gate/up packing differs");
        }
        add_projection(p + "mlp/down", {H, I}, p + "mlp/product");
    }
    if (features.dflash2()) {
        const auto& cfg = target_component("dflash2").at("config");
        if (cfg.value("model_type", std::string{}) != "qwen3") {
            throw artifact::ArtifactError("v3 DFlash2: unsupported model_type");
        }
        for (const auto& [name, value] :
             std::array<std::pair<const char*, int>, 7>{{
                 {"num_hidden_layers", DFlashConfig::layers},
                 {"intermediate_size", DFlashConfig::intermediate},
                 {"num_attention_heads", DFlashConfig::query_heads},
                 {"num_key_value_heads", DFlashConfig::kv_heads},
                 {"head_dim", DFlashConfig::head_dim},
                 {"sliding_window", DFlashConfig::local_capacity},
                 {"max_position_embeddings", static_cast<int>(kNativeContext)},
             }}) {
            dimension(cfg, name, value, "dflash2.config");
        }
        const auto& architectures = cfg.at("architectures");
        if (!architectures.is_array() || architectures.size() != 1 ||
            architectures[0] != "DFlash2DraftModel") {
            throw artifact::ArtifactError("v3 DFlash2 architecture differs");
        }
        const auto& layer_types = cfg.at("layer_types");
        if (!layer_types.is_array() || layer_types.size() != DFlashConfig::layers) {
            throw artifact::ArtifactError("v3 DFlash2 layer type count differs");
        }
        for (const auto& type : layer_types) {
            if (type != "sliding_attention") {
                throw artifact::ArtifactError("v3 DFlash2 layer type differs");
            }
        }
        const auto& rope = cfg.at("rope_parameters").at("rope_theta");
        if (!rope.is_number() || std::abs(rope.get<double>() -
                                           static_cast<double>(DFlashConfig::rope_theta)) > 1) {
            throw artifact::ArtifactError("v3 DFlash2 RoPE theta differs");
        }
        const auto& eps = cfg.at("rms_norm_eps");
        if (!eps.is_number() ||
            std::abs(eps.get<double>() - DFlashConfig::rms_epsilon) > 1e-10) {
            throw artifact::ArtifactError("v3 DFlash2 norm epsilon differs");
        }
        const auto& special = cfg.at("dflash_config");
        dimension(special, "selector_rank", DFlashConfig::selector_dim, "dflash_config");
        dimension(special, "mask_token_id", DFlashConfig::mask_token, "dflash_config");
        dimension(special, "selector_top_k", 16, "dflash_config");
        dimension(special, "conv_kernel_size", 2, "dflash_config");
        dimension(special, "conv_group_size", 16, "dflash_config");
        const auto& ids = special.at("target_layer_ids");
        if (!ids.is_array() || ids.size() != DFlashConfig::target_feature_layers.size()) {
            throw artifact::ArtifactError("v3 DFlash2 target feature count differs");
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (v3_unsigned(ids[i], "DFlash target layer") !=
                static_cast<std::uint64_t>(DFlashConfig::target_feature_layers[i])) {
                throw artifact::ArtifactError("v3 DFlash2 target layers differ");
            }
        }
        constexpr int H = DFlashConfig::hidden;
        constexpr int Q = DFlashConfig::query_size;
        constexpr int K = DFlashConfig::kv_size;
        constexpr int I = DFlashConfig::intermediate;
        const auto dadd = [&](const std::string& name,
                              std::initializer_list<std::uint64_t> shape,
                              std::string_view input = {},
                              NumericFormat format = NumericFormat::BF16)
            -> const V3LogicalTensor& {
            const auto& logical = add(name, shape, input, format);
            // Native DFlash2 ModelView uses W8 row-split planes or contiguous BF16.
            // A matching dtype with another physical layout cannot be decoded.
            const auto layout = format == NumericFormat::BF16
                                    ? artifact::StorageLayout::ContiguousLeV1
                                    : artifact::StorageLayout::RowSplitK128V1;
            if (logical.layout != layout) {
                throw artifact::ArtifactError(name + ": unsupported DFlash2 physical layout");
            }
            if (!input.empty() && logical.activation_policy != "A16Only") {
                throw artifact::ArtifactError(name + ": unsupported DFlash2 activation policy");
            }
            return logical;
        };
        dadd("dflash2/feature_projection", {H, DFlashConfig::feature_rows},
            "dflash2/target_features", NumericFormat::W8G32_F16S);
        dadd("dflash2/context_norm", {H}, {}, NumericFormat::BF16);
        dadd("dflash2/final_norm", {H}, {}, NumericFormat::BF16);
        for (int i = 0; i < DFlashConfig::layers; ++i) {
            const std::string p = "dflash2/layers/" + std::to_string(i) + "/";
            dadd(p + "input_norm", {H}, {}, NumericFormat::BF16);
            dadd(p + "post_attention_norm", {H}, {}, NumericFormat::BF16);
            const auto q = dadd(p + "attention/query", {Q, H},
                               p + "query_projection_input", NumericFormat::W8G32_F16S);
            const auto k = dadd(p + "attention/key", {K, H},
                               p + "query_projection_input", NumericFormat::W8G32_F16S);
            const auto v = dadd(p + "attention/value", {K, H},
                               p + "query_projection_input", NumericFormat::W8G32_F16S);
            if (q.object_id != k.object_id || q.object_id != v.object_id ||
                q.parent_rows != Q + 2 * K || q.first_row != 0 ||
                k.first_row != Q || v.first_row != Q + K) {
                throw artifact::ArtifactError(p + ": DFlash attention packed parent differs");
            }
            const auto ck = dadd(p + "attention/context_key", {K, H},
                                "dflash2/context_input", NumericFormat::W8G32_F16S);
            const auto cv = dadd(p + "attention/context_value", {K, H},
                                "dflash2/context_input", NumericFormat::W8G32_F16S);
            if (ck.object_id != k.object_id || cv.object_id != v.object_id ||
                ck.first_row != k.first_row || cv.first_row != v.first_row) {
                throw artifact::ArtifactError(p + ": DFlash context KV differs");
            }
            dadd(p + "attention/query_norm", {DFlashConfig::head_dim}, {}, NumericFormat::BF16);
            dadd(p + "attention/key_norm", {DFlashConfig::head_dim}, {}, NumericFormat::BF16);
            dadd(p + "attention/output", {H, Q}, p + "attention_output",
                NumericFormat::W8G32_F16S);
            const auto gate = dadd(p + "mlp/gate", {I, H}, p + "mlp_input",
                                  NumericFormat::W8G32_F16S);
            const auto up = dadd(p + "mlp/up", {I, H}, p + "mlp_input",
                                NumericFormat::W8G32_F16S);
            if (gate.object_id != up.object_id || gate.parent_rows != 2 * I ||
                gate.first_row != 0 || up.first_row != I) {
                throw artifact::ArtifactError(p + ": DFlash packed gate/up differs");
            }
            dadd(p + "mlp/down", {H, I}, p + "mlp_product", NumericFormat::W8G32_F16S);
            dadd(p + "attention_conv/base_kernel", {2, 2, H}, {}, NumericFormat::BF16);
            dadd(p + "attention_conv/kernel_projection", {DFlashConfig::dynamic_conv_rows, H},
                p + "attention_conv_input", NumericFormat::BF16);
            dadd(p + "mlp_conv/base_kernel", {2, 2, H}, {}, NumericFormat::BF16);
            dadd(p + "mlp_conv/kernel_projection", {DFlashConfig::dynamic_conv_rows, H},
                p + "mlp_conv_input", NumericFormat::BF16);
        }
        dadd("dflash2/candidate_selector/hidden_projection",
            {DFlashConfig::selector_dim, H}, "dflash2/final_hidden",
            NumericFormat::BF16);
        dadd("dflash2/candidate_selector/predecessor_codebook",
            {TextConfig::output_rows, DFlashConfig::selector_dim}, {},
            NumericFormat::BF16);
        dadd("dflash2/candidate_selector/successor_codebook",
            {TextConfig::output_rows, DFlashConfig::selector_dim}, {},
            NumericFormat::BF16);
    }
    if (features.optimized_proposal()) {
        const auto& proposal = components.at("text").at("proposal");
        if (proposal.value("domain", std::string{}) != "indexed" ||
            v3_unsigned(proposal.at("rows"), "proposal rows") !=
                static_cast<std::uint64_t>(TextConfig::draft_head_rows)) {
            throw artifact::ArtifactError("v3 optimized proposal requires indexed 131072 rows");
        }
        const std::string name = features.mtp() ? "mtp/final_hidden" : "dflash2/final_hidden";
        const auto binding = bindings.at("proposal/head");
        const auto object = object_index.find(v3_string(binding.at("object"), "proposal/head"));
        if (object == object_index.end()) {
            throw artifact::ArtifactError("v3 optimized proposal parent missing");
        }
        const auto fmt = v3_format(v3_string(object->second->at("format"), "proposal format"));
        if (fmt != NumericFormat::GGUF_Q4_K && fmt != NumericFormat::Q4G64_F16S) {
            throw artifact::ArtifactError("v3 optimized proposal requires GGUF Q4K or Q4G64");
        }
        add("proposal/head", {TextConfig::draft_head_rows, TextConfig::hidden}, name, fmt);
        add("proposal/token_ids", {TextConfig::draft_head_rows}, {}, NumericFormat::I32);
    }
    return out;
}

ArtifactLoadPlan bind_v3_text_artifact(artifact::Binder& binder, const nlohmann::json& directory,
                                       qwen3_6::StartupFeatures features) {
    if (features.speculative != SpeculativeBackend::None &&
        features.speculative != SpeculativeBackend::Mtp &&
        features.speculative != SpeculativeBackend::DFlash2) {
        throw artifact::ArtifactError(
            "qwen3.8 v3: this target registers MTP and DFlash2 speculative backends");
    }
    const V3TextMetadata metadata = inspect_v3_text_metadata(directory);
    const auto feature_metadata = inspect_v3_feature_metadata(directory, features);
    const auto& root_objects = directory.at("objects");
    const auto& roles = directory.at("components").at("text").at("resources");
    const auto& vision_roles = directory.at("components").at("vision").at("resources");
    if (!roles.is_object() || !vision_roles.is_object()) {
        throw artifact::ArtifactError("v3 frontend resources must be role maps");
    }
    std::map<std::string, V3LogicalTensor, std::less<>> semantics;
    std::unordered_set<std::string> device_objects;
    for (const auto& tensor : metadata.tensors) {
        semantics.emplace(tensor.parameter, tensor);
        device_objects.insert(tensor.object_id);
        if (tensor.input_columns_object) { device_objects.insert(*tensor.input_columns_object); }
    }
    for (const auto& tensor : feature_metadata) {
        if (!semantics.emplace(tensor.parameter, tensor).second) {
            throw artifact::ArtifactError("v3 duplicate semantic weight " + tensor.parameter);
        }
        device_objects.insert(tensor.object_id);
        if (tensor.input_columns_object) { device_objects.insert(*tensor.input_columns_object); }
    }
    const auto frontend_id = [&](const nlohmann::json& component, std::string_view role) {
        const std::string key(role);
        if (!component.contains(key) || !component.at(key).is_string()) {
            throw artifact::ArtifactError("v3 frontend missing resource role " + key);
        }
        return component.at(key).get<std::string>();
    };
    std::unordered_set<std::string> host_objects{
        frontend_id(roles, "tokenizer.json"),
        frontend_id(roles, "tokenizer_config.json"),
        frontend_id(roles, "chat_template.jinja"),
        frontend_id(roles, "generation_config.json"),
        frontend_id(vision_roles, "preprocessor_config.json"),
        frontend_id(vision_roles, "video_preprocessor_config.json"),
    };

    std::unordered_map<std::string, artifact::ObjectHandle> handles;
    handles.reserve(root_objects.size());
    for (const auto& object : root_objects) {
        const std::string id = v3_string(object.at("id"), "v3 object id");
        const artifact::ObjectDescriptor* stored = binder.find(id);
        if (!stored) { throw artifact::ArtifactError("v3 object missing from Binder: " + id); }
        artifact::ObjectHandle handle;
        if (const auto* tensor = std::get_if<artifact::TensorDescriptor>(stored)) {
            if (host_objects.contains(id)) {
                throw artifact::ArtifactError("v3 frontend role points to tensor: " + id);
            }
            handle = binder.require_tensor(id, tensor->format, tensor->layout, tensor->shape);
            if (device_objects.contains(id)) {
                binder.materialize_on_device(handle);
            } else {
                binder.validate_only(handle);
            }
        } else {
            handle = binder.require_resource(id, artifact::ResourceEncoding::RawBytesV1);
            if (device_objects.contains(id)) {
                throw artifact::ArtifactError("v3 selected weight is a resource: " + id);
            }
            if (host_objects.contains(id)) {
                binder.retain_on_host(handle);
            } else {
                binder.validate_only(handle);
            }
        }
        handles.emplace(id, handle);
    }

    const auto selected = [&](std::string_view parameter) -> const V3LogicalTensor& {
        const auto it = semantics.find(std::string(parameter));
        if (it == semantics.end()) {
            throw artifact::ArtifactError("v3 parameter missing from selected semantic binding: " +
                                          std::string(parameter));
        }
        return it->second;
    };
    const auto handle_of = [&](std::string_view parameter) -> artifact::ObjectHandle {
        return handles.at(selected(parameter).object_id);
    };
    const auto weight_of = [&](std::string_view parameter) -> WeightPlan {
        const auto& source = selected(parameter);
        WeightPlan out{};
        out.object         = handles.at(source.object_id);
        out.format         = source.format;
        out.parent_rows    = static_cast<std::int32_t>(source.parent_rows);
        out.parent_columns = static_cast<std::int32_t>(source.columns);
        out.first_row      = static_cast<std::int32_t>(source.first_row);
        out.selected_rows  = static_cast<std::int32_t>(source.rows);
        if (source.input_columns_object) {
            out.input_columns = handles.at(*source.input_columns_object);
        }
        if (source.format == NumericFormat::NVFP4) {
            if (source.layout != artifact::StorageLayout::BlockScaleK16M128x4V1 ||
                !source.input_divisor_object || source.parent_rows % 128 != 0 ||
                source.first_row % 128 != 0 || source.rows % 128 != 0 ||
                source.columns % 64 != 0) {
                throw artifact::ArtifactError(
                    std::string(parameter) + ": invalid NVFP4 parent/Use divisor geometry");
            }
            const std::array<std::uint64_t, 2> parent_shape{source.parent_rows,
                                                             source.columns};
            const auto geometry =
                artifact::block_scale_geometry(NumericFormat::NVFP4, parent_shape);
            const auto weight_bits = read_u32_le(
                binder.payload(out.object).data, geometry.weight_divisor_offset,
                std::string(parameter) + " weight divisor");
            const auto divisor_bits = read_u32_le(
                binder.payload(handles.at(*source.input_divisor_object)).data, 0,
                std::string(parameter) + " activation_input_divisor");
            // Header-only plans are also bound from deliberately sparse v3 test
            // files whose payload bytes are zero. Keep the exact source words in
            // the plan and validate they are positive/finite when the real CUDA
            // ModelView is materialized; metadata geometry is checked above.
            out.weight_scale_divisor_bits = weight_bits;
            out.input_scale_divisor_bits = divisor_bits;
        }
        return out;
    };
    const auto host = [&](const nlohmann::json& resources, std::string_view role) {
        return handles.at(frontend_id(resources, role));
    };

    ArtifactLoadPlan load;
    BindingPlan& out = load.bindings;
    out.features = features;
    out.v3_text = true;
    out.frontend = {
        .tokenizer_json = host(roles, "tokenizer.json"),
        .tokenizer_config_json = host(roles, "tokenizer_config.json"),
        .chat_template_jinja = host(roles, "chat_template.jinja"),
        .generation_config_json = host(roles, "generation_config.json"),
        .preprocessor_config_json = host(vision_roles, "preprocessor_config.json"),
        .video_preprocessor_config_json = host(vision_roles, "video_preprocessor_config.json"),
    };
    out.token_embedding = weight_of("text/token_embedding");
    out.output_head = weight_of("text/output_head");
    out.final_norm = handle_of("text/final_norm");
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        auto& target = out.text_layers[layer];
        target.is_full_attention = is_full_layer(layer);
        target.input_norm = handle_of(prefix + "input_norm");
        target.post_attention_norm = handle_of(prefix + "post_attention_norm");
        if (target.is_full_attention) {
            target.attention.projection = IndividualAttentionProjectionPlan{
                .query = weight_of(prefix + "attention/query"),
                .key = weight_of(prefix + "attention/key"),
                .gate = weight_of(prefix + "attention/gate"),
                .value = weight_of(prefix + "attention/value"),
            };
            target.attention.query_norm = handle_of(prefix + "attention/query_norm");
            target.attention.key_norm = handle_of(prefix + "attention/key_norm");
            target.attention.output = weight_of(prefix + "attention/output");
        } else {
            target.gdn.a_log = handle_of(prefix + "gdn/a_log");
            target.gdn.dt_bias = handle_of(prefix + "gdn/dt_bias");
            target.gdn.convolution = handle_of(prefix + "gdn/convolution");
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = weight_of(prefix + "gdn/a_projection"),
                .b_projection = weight_of(prefix + "gdn/b_projection"),
            };
            target.gdn.input_projection = IndividualGdnInputProjectionPlan{
                .query = weight_of(prefix + "gdn/query"),
                .key = weight_of(prefix + "gdn/key"),
                .value = weight_of(prefix + "gdn/value"),
                .z = weight_of(prefix + "gdn/z"),
            };
            target.gdn.norm = handle_of(prefix + "gdn/norm");
            target.gdn.output = weight_of(prefix + "gdn/output");
        }
        target.mlp.gate = weight_of(prefix + "mlp/gate");
        target.mlp.up = weight_of(prefix + "mlp/up");
        target.mlp.down = weight_of(prefix + "mlp/down");
    }
    if (features.mtp()) {
        const std::string p = "mtp/layers/0/";
        MtpPlan::V3Projections semantic;
        semantic.input_projection = weight_of("mtp/input_projection");
        semantic.query = weight_of(p + "attention/query");
        semantic.gate = weight_of(p + "attention/gate");
        semantic.key = weight_of(p + "attention/key");
        semantic.value = weight_of(p + "attention/value");
        semantic.output = weight_of(p + "attention/output");
        out.mtp.v3 = std::move(semantic);
        out.mtp.embedding_norm = handle_of("mtp/embedding_norm");
        out.mtp.hidden_norm = handle_of("mtp/hidden_norm");
        out.mtp.final_norm = handle_of("mtp/final_norm");
        out.mtp.input_norm = handle_of(p + "input_norm");
        out.mtp.query_norm = handle_of(p + "attention/query_norm");
        out.mtp.key_norm = handle_of(p + "attention/key_norm");
        out.mtp.post_attention_norm = handle_of(p + "post_attention_norm");
        out.mtp.mlp.gate = weight_of(p + "mlp/gate");
        out.mtp.mlp.up = weight_of(p + "mlp/up");
        out.mtp.mlp.down = weight_of(p + "mlp/down");
    }
    if (features.vision) {
        const auto& h = qwen3_6::VisionBackboneConfig{};
        (void)h;
        auto& backbone = out.vision_backbone;
        backbone.patch_embedding = handle_of("vision/patch_embedding");
        backbone.patch_embedding_bias = handle_of("vision/patch_embedding_bias");
        backbone.position_embedding = handle_of("vision/position_embedding");
        for (std::size_t i = 0; i < backbone.layers.size(); ++i) {
            const std::string p = "vision/layers/" + std::to_string(i) + "/";
            auto& v = backbone.layers[i];
            // One packed [Q;K;V] parent (and corresponding packed bias) backs the individual
            // v3 semantic references. The shared Vision runtime consumes exactly this form.
            v.qkv = handle_of(p + "attention/query");
            v.qkv_bias = handle_of(p + "attention/query_bias");
            v.output = handle_of(p + "attention/output");
            v.output_bias = handle_of(p + "attention/output_bias");
            v.fc1 = handle_of(p + "mlp/fc1");
            v.fc1_bias = handle_of(p + "mlp/fc1_bias");
            v.fc2 = handle_of(p + "mlp/fc2");
            v.fc2_bias = handle_of(p + "mlp/fc2_bias");
            v.norm1_weight = handle_of(p + "norm1_weight");
            v.norm1_bias = handle_of(p + "norm1_bias");
            v.norm2_weight = handle_of(p + "norm2_weight");
            v.norm2_bias = handle_of(p + "norm2_bias");
        }
        out.vision_merger_input.fc1 = handle_of("vision/merger/fc1");
        out.vision_merger_input.fc1_bias = handle_of("vision/merger/fc1_bias");
        out.vision_merger_norm.weight = handle_of("vision/merger/norm_weight");
        out.vision_merger_norm.bias = handle_of("vision/merger/norm_bias");
        out.vision_merger_fc2 = handle_of("vision/merger/fc2");
        out.vision_merger_fc2_bias = handle_of("vision/merger/fc2_bias");
    }
    if (features.dflash2()) {
        out.dflash2.emplace();
        auto& dflash = *out.dflash2;
        dflash.feature_projection = weight_of("dflash2/feature_projection");
        dflash.context_norm = handle_of("dflash2/context_norm");
        dflash.final_norm = handle_of("dflash2/final_norm");
        for (std::size_t i = 0; i < dflash.layers.size(); ++i) {
            const std::string p = "dflash2/layers/" + std::to_string(i) + "/";
            auto& layer = dflash.layers[i];
            layer.input_norm = handle_of(p + "input_norm");
            layer.post_attention_norm = handle_of(p + "post_attention_norm");
            // Q/K/V share a single W8 [6144,5120] source; preserve it as the parent,
            // because the existing DFlash2 runtime has a registered fused W8 projection.
            layer.query_key_value = weight_of(p + "attention/query");
            layer.query_key_value.first_row = 0;
            layer.query_key_value.selected_rows = DFlashConfig::query_key_value_rows;
            layer.query_norm = handle_of(p + "attention/query_norm");
            layer.key_norm = handle_of(p + "attention/key_norm");
            layer.attention_output = weight_of(p + "attention/output");
            layer.gate_up = weight_of(p + "mlp/gate");
            layer.gate_up.first_row = 0;
            layer.gate_up.selected_rows = DFlashConfig::mlp_gate_up_rows;
            layer.down = weight_of(p + "mlp/down");
            layer.attention_conv.base_kernel = handle_of(p + "attention_conv/base_kernel");
            layer.attention_conv.kernel_projection =
                weight_of(p + "attention_conv/kernel_projection");
            layer.mlp_conv.base_kernel = handle_of(p + "mlp_conv/base_kernel");
            layer.mlp_conv.kernel_projection = weight_of(p + "mlp_conv/kernel_projection");
        }
        dflash.candidate_selector.hidden_projection =
            weight_of("dflash2/candidate_selector/hidden_projection");
        dflash.candidate_selector.predecessor_codebook =
            handle_of("dflash2/candidate_selector/predecessor_codebook");
        dflash.candidate_selector.successor_codebook =
            handle_of("dflash2/candidate_selector/successor_codebook");
    }
    if (features.optimized_proposal()) {
        out.v3_proposal_head = weight_of("proposal/head");
        out.draft_head_token_ids = handle_of("proposal/token_ids");
    }
    load.materialization = binder.finish();
    return load;
}

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

void validate_v3_input_columns(std::span<const std::int32_t> columns,
                               std::size_t expected_width) {
    if (columns.size() != expected_width || expected_width == 0) {
        throw artifact::ArtifactError("v3 GDN input_columns has an invalid width");
    }
    std::vector<std::uint8_t> seen(expected_width, 0);
    for (const std::int32_t column : columns) {
        if (column < 0 || static_cast<std::size_t>(column) >= expected_width ||
            seen[static_cast<std::size_t>(column)] != 0) {
            throw artifact::ArtifactError(
                "v3 GDN input_columns must be a permutation of the activation width");
        }
        seen[static_cast<std::size_t>(column)] = 1;
    }
}

MtpWeights materialize_mtp_weights(const MtpPlan& plan,
                                    const artifact::MaterializedArtifact& materialized) {
    MtpWeights mtp{};
    if (plan.v3) {
        mtp.input_projection = materialized_weight(materialized, plan.v3->input_projection,
                                                    TextConfig::hidden, TextConfig::mtp_input_rows);
    } else {
        mtp.input_projection = artifact::materialized_weight(
            materialized, plan.input_projection, NumericFormat::W8G32_F16S,
            TextConfig::hidden, TextConfig::mtp_input_rows);
    }
    mtp.embedding_norm = artifact::materialized_tensor(
        materialized, plan.embedding_norm, NumericFormat::BF16, {TextConfig::hidden});
    mtp.hidden_norm = artifact::materialized_tensor(
        materialized, plan.hidden_norm, NumericFormat::BF16, {TextConfig::hidden});
    mtp.input_norm = artifact::materialized_tensor(
        materialized, plan.input_norm, NumericFormat::BF16, {TextConfig::hidden});
    if (plan.v3) {
        const auto& projections = *plan.v3;
        mtp.attention.query = materialized_weight(
            materialized, projections.query, TextConfig::query_size, TextConfig::hidden);
        mtp.attention.key = materialized_weight(
            materialized, projections.key, TextConfig::kv_size, TextConfig::hidden);
        mtp.attention.output_gate = materialized_weight(
            materialized, projections.gate, TextConfig::query_size, TextConfig::hidden);
        mtp.attention.value = materialized_weight(
            materialized, projections.value, TextConfig::kv_size, TextConfig::hidden);
    } else {
        mtp.attention.packed = artifact::materialized_weight(
            materialized, plan.query_key_gate_value, NumericFormat::W8G32_F16S,
            TextConfig::attention_input_rows, TextConfig::hidden);
        mtp.attention.query = row_view(mtp.attention.packed, 0, TextConfig::query_size);
        mtp.attention.key = row_view(mtp.attention.packed, TextConfig::query_size, TextConfig::kv_size);
        mtp.attention.output_gate = row_view(
            mtp.attention.packed, TextConfig::query_size + TextConfig::kv_size,
            TextConfig::query_size);
        mtp.attention.value = row_view(
            mtp.attention.packed, 2 * TextConfig::query_size + TextConfig::kv_size,
            TextConfig::kv_size);
    }
    mtp.query_norm = artifact::materialized_tensor(
        materialized, plan.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
    mtp.key_norm = artifact::materialized_tensor(
        materialized, plan.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
    if (plan.v3) {
        mtp.output = materialized_weight(
            materialized, plan.v3->output, TextConfig::hidden, TextConfig::query_size);
    } else {
        mtp.output = artifact::materialized_weight(
            materialized, plan.output, NumericFormat::W8G32_F16S,
            TextConfig::hidden, TextConfig::query_size);
    }
    mtp.post_attention_norm = artifact::materialized_tensor(
        materialized, plan.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
    mtp.post_mixer = load_mlp(plan.mlp, materialized);
    mtp.final_norm = artifact::materialized_tensor(
        materialized, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    return mtp;
}

qwen3_6::DFlash2Weights materialize_dflash2_weights(
    const DFlash2Plan& source, const artifact::MaterializedArtifact& backing) {
    qwen3_6::DFlash2Weights dflash2{};
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
    return dflash2;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);
    // Install the folded sign table for the whole materialization pass below.
    const FoldedSignsScope folded_signs_scope(plan.hadamard_signs, backing);

    // Directory metadata validates the auxiliary's dtype and shape, but its
    // *values* are stored at the end of the real GSQ v3 payload. Verify once
    // per physical map after materialization and before using it as a GPU
    // gather index; invalid values would otherwise access unrelated VRAM.
    if (plan.v3_text) {
        std::unordered_set<std::uint32_t> checked_maps;
        std::vector<std::int32_t> host_columns(TextConfig::value_dim);
        for (const TextLayerPlan& layer : plan.text_layers) {
            if (layer.is_full_attention || !layer.gdn.output.input_columns) { continue; }
            const auto handle = *layer.gdn.output.input_columns;
            if (!checked_maps.insert(handle.index).second) { continue; }
            CUDA_CHECK(cudaMemcpy(
                host_columns.data(), backing.device_data(handle),
                host_columns.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            validate_v3_input_columns(host_columns, TextConfig::value_dim);
        }
    }

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
        if (plan.v3_proposal_head) {
            proposal.head = materialized_weight(backing, *plan.v3_proposal_head,
                                                TextConfig::draft_head_rows,
                                                TextConfig::hidden);
        } else {
            proposal.head = artifact::materialized_weight(
                backing, plan.draft_head, NumericFormat::Q4G64_F16S,
                TextConfig::draft_head_rows, TextConfig::hidden);
        }
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {TextConfig::draft_head_rows});
    }

    if (plan.features.mtp()) {
        runtime.mtp.emplace(materialize_mtp_weights(plan.mtp, backing));
    }

    if (plan.features.dflash2()) {
        if (!plan.dflash2) {
            throw std::logic_error("selected DFlash2 weights are absent from the binding plan");
        }
        runtime.dflash.emplace(materialize_dflash2_weights(*plan.dflash2, backing));
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
