#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/startup.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
    case KvCapacityMode::DeviceBudget:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.managed_runtime_capacity_bytes != 0 && !options.managed_weights) {
        throw std::invalid_argument("Engine managed runtime capacity requires managed_weights");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes,
                                                bool wddm_evictable_budget = false,
                                                bool managed_weights = false,
                                                std::size_t managed_runtime_bytes = 0) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (managed_weights) {
        // CUDA-managed model storage may exceed physical VRAM. Reserve a bounded
        // resident runtime budget before uploading weights; CUDA is responsible
        // for migrating model pages as kernels access them. Avoid reporting this
        // as free VRAM or claiming the whole model will be resident.
        constexpr std::size_t kRuntimeBudget = 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t kMinimumFree = 512ULL * 1024ULL * 1024ULL;
        if (free_bytes < kMinimumFree) {
            throw std::invalid_argument(
                "managed weights require at least 512 MiB initially free device memory");
        }
        // Adapt the previously isolated full-KV diagnostic's explicit ceiling,
        // keeping device/display headroom rather than subtracting managed weights.
        if (managed_runtime_bytes != 0) {
            return std::min(managed_runtime_bytes, free_bytes - kMinimumFree);
        }
        return std::min(kRuntimeBudget, free_bytes - kMinimumFree);
    }
#if defined(_WIN32)
    // Allow WDDM to evict background apps down to a 512 MiB non-evictable DWM display floor.
    // Opt-in only: on a card that is also driving the desktop the background allocations are
    // frequently not evictable, so budgeting against total VRAM oversubscribes the device and
    // pushes the runtime into WDDM paging.
    constexpr std::size_t kMinDwmHeadroom = 512ULL * 1024ULL * 1024ULL;
    if (wddm_evictable_budget && total_bytes > weight_bytes + kMinDwmHeadroom) {
        const std::size_t evictable_capacity =
            total_bytes - static_cast<std::size_t>(weight_bytes) - kMinDwmHeadroom;
        const std::size_t free_after_weights =
            free_bytes > weight_bytes ? free_bytes - static_cast<std::size_t>(weight_bytes) : 0;
        return std::max(free_after_weights, evictable_capacity);
    }
#endif
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes(std::size_t weights_bytes = 0,
                                      bool wddm_evictable_budget = false,
                                      std::size_t planned_runtime_bytes = 0) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
#if defined(_WIN32)
    if (wddm_evictable_budget && weights_bytes > 0) {
        // Allow WDDM to evict background apps down to 512 MiB non-evictable DWM display floor
        constexpr std::size_t kMinDwmHeadroom = 512ULL * 1024ULL * 1024ULL;
        if (total_bytes > weights_bytes + kMinDwmHeadroom) {
            const std::size_t evictable_free = total_bytes - weights_bytes - kMinDwmHeadroom;
            return std::max(free_bytes, evictable_free);
        }
    }
    // The arena already owns the pre-flight budget, so free_bytes understates what the runtime
    // may still place. Report the budget the arena was actually sized from.
    if (weights_bytes > 0 && planned_runtime_bytes > 0) {
        return std::max(free_bytes, planned_runtime_bytes);
    }
#endif
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader,
                                       const artifact::ArtifactIdentity& identity,
                                       Clock::time_point load_start,
                                       std::string_view target_key) {
#if defined(_WIN32)
    core::set_wddm_residency_lock_enabled(options.wddm_evictable_budget);
#endif
    StartupPhaseScope target_plan_phase(options.startup_observer, StartupPhase::TargetPlan);
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes = runtime_bytes_after_planned_weights(
        load_plan.materialization().device_capacity_bytes, options.wddm_evictable_budget,
        options.managed_weights, options.managed_runtime_capacity_bytes);
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);
    target_plan_phase.complete();

    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              &options.startup_observer, options.managed_weights);
    const artifact::MaterializationStats stats = materialized.stats();

    StartupPhaseScope target_finalize_phase(options.startup_observer, StartupPhase::TargetFinalize);
    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    const std::size_t weights_capacity_bytes = stats.h2d_bytes;
    runtime::KvCapacityResolution capacity_resolution = runtime::resolve_kv_capacity(
        options.kv_capacity, curve,
        options.managed_weights
            ? preflight_runtime_bytes
            : current_free_device_bytes(weights_capacity_bytes, options.wddm_evictable_budget,
                                        preflight_runtime_bytes));
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    target_finalize_phase.complete();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto loaded = std::make_unique<Loaded>(std::move(model), options);
    frontend_phase.complete();

    StartupPhaseScope program_phase(options.startup_observer, StartupPhase::ProgramInitialize);
    auto instance =
        std::make_unique<Instance>(std::move(loaded), capacity_resolution, std::move(sequence_plan),
                                   device, options.startup_observer);
    device.synchronize();
    program_phase.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model)};
}

artifact::ArtifactIdentity registered_identity(const artifact::Reader& reader) {
    if (const auto* directory = reader.v3_directory()) {
        // v3 carries model identity in the validated component config, rather
        // than the v2 top-level identity. A new architecture must register
        // its own target instead of accidentally receiving the 27B program.
        const auto& components = directory->at("components");
        if (!components.contains("text")) {
            throw artifact::ArtifactError("v3 artifact has no text component");
        }
        const auto& config = components.at("text").at("config");
        const auto& metadata = directory->at("metadata");
        const std::string name = metadata.value("name", "");
        if (name == "qwen3.6-27b") {
            throw artifact::ArtifactError("Qwen3.6-27B support has been retired; use Qwen3.8-27B");
        }
        if ((name == "qwen3.8-27b" ||
             name == "qwen3.8-27b-tp2-rank0" || name == "qwen3.8-27b-tp2-rank1") &&
            config.value("model_type", "") == "qwen3_5_text" &&
            config.value("hidden_size", 0) == 5120 &&
            config.value("num_hidden_layers", 0) == 64) {
            if (name == "qwen3.8-27b-tp2-rank0" || name == "qwen3.8-27b-tp2-rank1") {
                return {name, "groupwise-int"};
            }
            return {name, "groupwise-int"};
        }
        throw artifact::ArtifactError(
            "v3 component geometry is not registered as an executable target");
    }
    return reader.identity();
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device,
                                         const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device,
                                          startup_observer)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_27BTp2::LoadedQwen3_6_27BTp2(
    std::unique_ptr<Qwen3_6_27BTp2::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27BTp2::make_frontend(*model, options)) {}

LoadedQwen3_6_27BTp2::~LoadedQwen3_6_27BTp2() = default;

Qwen3_6_27BTp2Instance::Qwen3_6_27BTp2Instance(
    std::unique_ptr<LoadedQwen3_6_27BTp2> stable_loaded,
    runtime::KvCapacityResolution resolution, Qwen3_6_27BTp2::SequencePlan sequence_plan,
    DeviceContext& device, const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27BTp2::create_program(*loaded->model, std::move(sequence_plan), device,
                                             startup_observer)) {}

Qwen3_6_27BTp2Instance::~Qwen3_6_27BTp2Instance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device,
                                               const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device,
                                             startup_observer)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    StartupPhaseScope inspect_phase(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect_phase.complete();
    const auto identity = registered_identity(reader);
    if (identity.model_id == "qwen3.6-27b") {
        throw artifact::ArtifactError("Qwen3.6-27B support has been retired; use Qwen3.8-27B");
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, identity, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_27BTp2::model_id ||
        identity.model_id == Qwen3_6_27BTp2::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27BTp2, LoadedQwen3_6_27BTp2,
                                    Qwen3_6_27BTp2Instance>(
            options, device, reader, identity, load_start, Qwen3_6_27BTp2::target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, identity, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
