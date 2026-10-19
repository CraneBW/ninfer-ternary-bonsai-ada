#include "serve/serve_metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

void metric(std::ostringstream& out, std::string_view name, std::uint64_t value) {
    out << name << ' ' << value << '\n';
}

void metric(std::ostringstream& out, std::string_view name, std::int64_t value) {
    out << name << ' ' << value << '\n';
}

void metric(std::ostringstream& out, std::string_view name, double value) {
    out << name << ' ' << value << '\n';
}

} // namespace

void ServeMetrics::begin_request(std::uint64_t id, const RequestLogContext& context) {
    std::lock_guard lock(mutex_);
    ++requests_total_;
    ActiveRequest request;
    request.id            = id;
    request.protocol      = context.protocol;
    request.model         = context.model;
    request.stream        = context.stream;
    request.prompt_tokens = context.prompt_tokens;
    request.started       = std::chrono::steady_clock::now();
    active_[id]           = std::move(request);
}

void ServeMetrics::end_request(std::uint64_t id) noexcept {
    std::lock_guard lock(mutex_);
    active_.erase(id);
}

void ServeMetrics::record(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    ++requests_completed_total_;
    const std::uint64_t prompt =
        outcome.prompt_tokens > 0 ? static_cast<std::uint64_t>(outcome.prompt_tokens) : 0;
    const std::uint64_t hit = outcome.metrics.prefix_cache_hit_tokens;
    prompt_tokens_total_ += prompt > hit ? prompt - hit : 0;
    prompt_seconds_total_ += outcome.metrics.prefill_seconds;
    ttft_seconds_total_ += outcome.metrics.ttft_seconds;
    tokens_predicted_total_ +=
        outcome.completion_tokens > 0 ? static_cast<std::uint64_t>(outcome.completion_tokens) : 0;
    predicted_seconds_total_ += outcome.metrics.decode_seconds;
    prefix_cache_hit_tokens_total_ += hit;
    draft_tokens_total_ += outcome.metrics.speculative_draft_tokens;
    draft_accepted_tokens_total_ += outcome.metrics.speculative_accepted_tokens;
}

std::string ServeMetrics::render(const ninfer::RuntimeStats& stats,
                                 const ninfer::EngineOptions& options) const {
    std::size_t active = 0;
    std::uint64_t requests_total = 0;
    std::uint64_t requests_completed = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t predicted_tokens = 0;
    double prompt_seconds = 0.0;
    double predicted_seconds = 0.0;
    double ttft_seconds = 0.0;
    std::uint64_t prefix_hits = 0;
    std::uint64_t draft = 0;
    std::uint64_t accepted = 0;
    {
        std::lock_guard lock(mutex_);
        active              = active_.size();
        requests_total      = requests_total_;
        requests_completed  = requests_completed_total_;
        prompt_tokens       = prompt_tokens_total_;
        predicted_tokens    = tokens_predicted_total_;
        prompt_seconds      = prompt_seconds_total_;
        predicted_seconds   = predicted_seconds_total_;
        ttft_seconds        = ttft_seconds_total_;
        prefix_hits         = prefix_cache_hit_tokens_total_;
        draft               = draft_tokens_total_;
        accepted            = draft_accepted_tokens_total_;
    }

    const std::size_t concurrency =
        std::max<std::size_t>(1, static_cast<std::size_t>(options.max_concurrency));
    const std::size_t processing = std::min(active, concurrency);
    const std::size_t deferred   = active - processing;

    std::ostringstream out;
    // llama.cpp-compatible names first so existing dashboards keep working.
    metric(out, "llamacpp:prompt_tokens_total", prompt_tokens);
    metric(out, "llamacpp:prompt_seconds_total", prompt_seconds);
    metric(out, "llamacpp:tokens_predicted_total", predicted_tokens);
    metric(out, "llamacpp:tokens_predicted_seconds_total", predicted_seconds);
    metric(out, "llamacpp:requests_processing", static_cast<std::uint64_t>(processing));
    metric(out, "llamacpp:requests_deferred", static_cast<std::uint64_t>(deferred));

    metric(out, "ninfer:requests_total", requests_total);
    metric(out, "ninfer:requests_completed_total", requests_completed);
    metric(out, "ninfer:requests_active", static_cast<std::uint64_t>(active));
    metric(out, "ninfer:ttft_seconds_total", ttft_seconds);
    metric(out, "ninfer:prefix_cache_hit_tokens_total", prefix_hits);
    metric(out, "ninfer:draft_tokens_total", draft);
    metric(out, "ninfer:draft_accepted_tokens_total", accepted);

    metric(out, "ninfer:computed_prefill_tokens_total", stats.computed_prefill_tokens);
    metric(out, "ninfer:committed_decode_tokens_total", stats.committed_decode_tokens);
    metric(out, "ninfer:reused_prompt_tokens_total", stats.reused_prompt_tokens);
    metric(out, "ninfer:decode_rounds_total", stats.decode_rounds);
    metric(out, "ninfer:running_requests", static_cast<std::uint64_t>(stats.running_requests));
    metric(out, "ninfer:prefilling_requests",
           static_cast<std::uint64_t>(stats.prefilling_requests));
    metric(out, "ninfer:waiting_requests", static_cast<std::uint64_t>(stats.waiting_requests));
    metric(out, "ninfer:materializing_requests",
           static_cast<std::uint64_t>(stats.materializing_requests));
    metric(out, "ninfer:capture_pending_requests",
           static_cast<std::uint64_t>(stats.capture_pending_requests));
    metric(out, "ninfer:device_state_occupied_slots",
           static_cast<std::uint64_t>(stats.device_state_occupied_slots));
    metric(out, "ninfer:host_state_occupied_slots",
           static_cast<std::uint64_t>(stats.host_state_occupied_slots));
    metric(out, "ninfer:device_main_kv_occupied_pages",
           static_cast<std::uint64_t>(stats.device_main_kv_occupied_pages));
    metric(out, "ninfer:device_backend_kv_occupied_pages",
           static_cast<std::uint64_t>(stats.device_backend_kv_occupied_pages));
    metric(out, "ninfer:host_kv_occupied_bytes",
           static_cast<std::uint64_t>(stats.host_kv_occupied_bytes));

    metric(out, "ninfer:kvmem_enabled",
           static_cast<std::uint64_t>(options.kvmem.enabled ? 1 : 0));
    metric(out, "ninfer:kvmem_budget_tokens", static_cast<std::uint64_t>(options.kvmem.budget_tokens));
    metric(out, "ninfer:kvmem_gen_reserve_tokens",
           static_cast<std::uint64_t>(options.kvmem.gen_reserve_tokens));
    metric(out, "ninfer:kvmem_plans_total", stats.kvmem.plans);
    metric(out, "ninfer:kvmem_compactions_total", stats.kvmem.compactions);
    metric(out, "ninfer:kvmem_compact_skips_total", stats.kvmem.compact_skips);
    metric(out, "ninfer:kvmem_stage_in_blocks_total", stats.kvmem.stage_in_blocks);
    metric(out, "ninfer:kvmem_stage_out_blocks_total", stats.kvmem.stage_out_blocks);
    metric(out, "ninfer:kvmem_transfer_runs_total", stats.kvmem.transfer_runs);
    metric(out, "ninfer:kvmem_transfer_us_total", stats.kvmem.transfer_us);
    metric(out, "ninfer:kvmem_scoreable_blocks_total", stats.kvmem.scoreable_blocks);
    metric(out, "ninfer:kvmem_zero_score_blocks_total", stats.kvmem.zero_score_blocks);
    metric(out, "ninfer:kvmem_window_tokens", static_cast<std::uint64_t>(stats.kvmem.window_tokens));
    metric(out, "ninfer:kvmem_host_bytes", static_cast<std::uint64_t>(stats.kvmem.host_bytes));

    metric(out, "ninfer:capture_skips_total", stats.capture_skips.total);
    for (std::size_t index = 0; index < stats.capture_skips.by_reason.size(); ++index) {
        out << "ninfer:capture_skips{reason=\""
            << capture_skip_reason_name(static_cast<CaptureSkipReason>(index)) << "\"} "
            << stats.capture_skips.by_reason[index] << '\n';
    }
    return out.str();
}

std::string ServeMetrics::render_slots_json() const {
    nlohmann::json slots = nlohmann::json::array();
    const auto now       = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    for (const auto& [id, request] : active_) {
        const double elapsed =
            std::chrono::duration<double>(now - request.started).count();
        slots.push_back(nlohmann::json{
            {"id", id},
            {"protocol", request.protocol},
            {"model", request.model},
            {"stream", request.stream},
            {"prompt_tokens", request.prompt_tokens},
            {"elapsed_seconds", elapsed},
            {"state", "active"},
        });
    }
    return slots.dump();
}

} // namespace ninfer::serve
