// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "targets/qwen3_6_27b/impl/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = TextConfig::layers;
inline constexpr std::size_t kFullAttentionLayers = TextConfig::full_attention_layers();
inline constexpr std::size_t kGdnLayers           = TextConfig::gdn_layers();

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
    // Folded-basis feature permutation for this weight; perm_rep == 1 means "none".
    // Set at bind time from the weight's identity (only /gdn/output carries one).
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;
    // v3 semantic binding may select consecutive rows from a shared GGUF parent.
    std::int32_t parent_rows = 0;
    std::int32_t parent_columns = 0;
    std::int32_t first_row = 0;
    std::int32_t selected_rows = 0;
    std::optional<artifact::ObjectHandle> input_columns;
};

// The v3 GSQ GDN output physically stores permuted input columns. Validate
// the device-side map before any GGUF kernel indexes activations through it.
void validate_v3_input_columns(std::span<const std::int32_t> columns,
                               std::size_t expected_width);

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
    std::optional<WeightPlan> gate;
    std::optional<WeightPlan> up;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

struct IndividualAttentionProjectionPlan {
    WeightPlan query;
    WeightPlan key;
    WeightPlan gate;
    WeightPlan value;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan,
                 IndividualAttentionProjectionPlan> projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct IndividualGdnInputProjectionPlan {
    WeightPlan query;
    WeightPlan key;
    WeightPlan value;
    WeightPlan z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan,
                 IndividualGdnInputProjectionPlan> input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    struct V3Projections {
        WeightPlan input_projection;
        WeightPlan query;
        WeightPlan key;
        WeightPlan gate;
        WeightPlan value;
        WeightPlan output;
    };
    std::optional<V3Projections> v3;
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct DFlash2DynamicConvPlan {
    artifact::ObjectHandle base_kernel;
    WeightPlan kernel_projection;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_norm;
    DFlash2DynamicConvPlan attention_conv;
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    artifact::ObjectHandle post_attention_norm;
    DFlash2DynamicConvPlan mlp_conv;
    WeightPlan gate_up;
    WeightPlan down;
};

struct DFlash2CandidateSelectorPlan {
    WeightPlan hidden_projection;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct DFlash2Plan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlash2LayerPlan, qwen3_6::DFlash2Weights::layer_count> layers;
    artifact::ObjectHandle final_norm;
    DFlash2CandidateSelectorPlan candidate_selector;
};

// Sign table of the rotated (folded) weight basis.
//
// The ternary port stores linear weights folded into a rotated basis, so the runtime must map an
// activation as y = H * (s . x) before each folded matmul and as h = s . (H * z) after the
// token-embedding lookup. `s` lives in one artifact object: `values` holds every sign, and
// `widths` partitions it, because a weight whose input dimension is W owns exactly W/block_size
// consecutive sign rows. `width_offsets` is the prefix sum of `widths`, so a weight can find its
// own block without knowing anything but its input dimension.
//
// Absent on artifacts whose weights are not folded (the groupwise-int conversion), which is why
// the whole thing is optional.
struct HadamardSignsPlan {
    artifact::ObjectHandle values;
    artifact::ObjectHandle widths;
    std::vector<std::pair<std::int32_t, std::uint64_t>> width_offsets;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    bool v3_text = false;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    std::optional<WeightPlan> v3_proposal_head;
    // Absent unless the artifact stores folded (rotated-basis) weights; the ternary port does.
    std::optional<HadamardSignsPlan> hadamard_signs;
    MtpPlan mtp;
    std::optional<DFlash2Plan> dflash2;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// Validate the declared tensor geometry without consuming objects or materializing weights.
// The overload permits a small synthetic shape to be exercised without a multi-GB fixture.
void validate_artifact_model_shape(const artifact::Binder& binder, const ModelShape& expected);
void validate_artifact_model_shape(const artifact::Binder& binder);

// Inspect the complete v3 *directory* before the large weight payload is available.
// Every semantic text parameter must have its expected shape, source parent/range, and Use.
// This is also the read-only preflight for the actual v3 binding route.
struct V3LogicalTensor {
    std::string parameter;
    std::string object_id;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
    artifact::StorageLayout layout = artifact::StorageLayout::ContiguousLeV1;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t parent_rows = 0;
    std::uint64_t first_row = 0;
    std::string activation_policy;
    std::optional<std::string> input_columns_object;
    std::optional<std::string> input_divisor_object;
};
struct V3TextMetadata {
    ModelShape shape = kModelShape;
    std::vector<V3LogicalTensor> tensors;
};
[[nodiscard]] V3TextMetadata inspect_v3_text_metadata(const nlohmann::json& directory);
// Feature-specific semantic records use the same physical-parent row coordinates as text.
// Optional proposal records are returned when the selected backend uses the indexed head.
[[nodiscard]] std::vector<V3LogicalTensor>
inspect_v3_feature_metadata(const nlohmann::json& directory,
                            qwen3_6::StartupFeatures features);
ArtifactLoadPlan bind_v3_text_artifact(artifact::Binder& binder,
                                       const nlohmann::json& directory,
                                       qwen3_6::StartupFeatures features);

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
    std::optional<Weight> gate;
    std::optional<Weight> up;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

struct IndividualAttentionProjectionPayload {
    Weight query;
    Weight key;
    Weight gate;
    Weight value;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload,
                 IndividualAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

struct IndividualGdnInputProjectionPayload {
    Weight query;
    Weight key;
    Weight value;
    Weight z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload,
                 IndividualGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlash2Weights,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

// Materialize one MTP layer through the same native ModelView path as the
// loaded model. Exposing the narrow binding boundary lets the v3 target test
// materialize just the actual MTP physical parents (without loading ~15 GiB
// of unrelated text/vision weights) and validate every GGUF/row-split slice.
[[nodiscard]] MtpWeights materialize_mtp_weights(
    const MtpPlan& plan, const artifact::MaterializedArtifact& materialized);

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
