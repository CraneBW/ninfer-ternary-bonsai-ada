#include <ninfer/targets/qwen3_6_27b_tp2/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_27b_tp2/impl/load/bindings.h"
#include "targets/qwen3_6_27b_tp2/impl/tp2_comm.h"
#include "targets/qwen3_6_27b_tp2/impl/variant.h"
#include "runtime/tp2/nccl_collectives.h"
#include "runtime/tp2/v3_rank_manifest.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6_27b_tp2::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail

namespace ninfer::targets::qwen3_6_27b_tp2 {
namespace {

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6Defaults; }
    if (model == qwen3_8_model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if ((identity.model_id == model_id || identity.model_id == qwen3_8_model_id) &&
        identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen38GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const auto* v3 = binder.reader().v3_directory();
    if (v3 == nullptr || weights_profile != WeightsProfile::Qwen38GroupwiseInt) {
        throw artifact::ArtifactError("TP2 target requires rank-sharded Qwen3.8 v3 groupwise weights");
    }
    const auto rank_manifest = runtime::tp2::inspect_v3_rank_manifest(*v3);
    const char* env = std::getenv("NINFER_TP_RANK");
    if (env == nullptr || std::string_view(env) != std::to_string(rank_manifest.rank)) {
        throw artifact::ArtifactError("TP2 rank artifact disagrees with NINFER_TP_RANK");
    }
    if (!runtime::tp2::NcclCollectives::compiled_with_nccl()) {
        throw artifact::ArtifactError("TP2 native execution requires NCCL; rebuild with NCCL");
    }
    // The v3 rank sharder retains the complete Vision and DFlash2 parents on
    // both ranks, and independently shards MTP attention/MLP roles. The v3
    // feature binder validates their physical formats, row-span boundaries,
    // and native ModelView layout before selecting CUDA allocations. MTP's
    // partial attention/MLP outputs use Variant::tp_combine_partial and the
    // rank-sharded MLP down-path's AllReduce; the output head is replicated.
    const auto startup = qwen3_6::startup_features(options);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_v3_text_artifact(binder, *v3, startup)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    detail::tp2::init();
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                      .vision_max_tokens =
                                          model.impl_->data.runtime.features.vision_max_tokens,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver& startup_observer) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(model.impl_->data.runtime,
                                                    model.impl_->weights_profile, std::move(plan),
                                                    device, startup_observer);
}

} // namespace ninfer::targets::qwen3_6_27b_tp2
