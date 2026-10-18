#include "serve/serve_metrics.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

RequestLogContext context(std::uint64_t id, int prompt_tokens, std::string protocol,
                          std::string model, bool stream) {
    RequestLogContext context;
    context.id            = id;
    context.prompt_tokens = prompt_tokens;
    context.protocol      = std::move(protocol);
    context.model         = std::move(model);
    context.stream        = stream;
    return context;
}

} // namespace

int main() {
    int failures = 0;

    ServeMetrics metrics;
    metrics.begin_request(7, context(7, 100, "openai_chat_completions", "ternary-bonsai", true));
    metrics.begin_request(8, context(8, 0, "anthropic_messages", "ternary-bonsai", false));

    GenerationOutcome outcome;
    outcome.prompt_tokens               = 100;
    outcome.completion_tokens           = 7;
    outcome.metrics.prefill_seconds     = 1.0;
    outcome.metrics.decode_seconds      = 0.5;
    outcome.metrics.ttft_seconds        = 1.25;
    outcome.metrics.prefix_cache_hit_tokens    = 40;
    outcome.metrics.speculative_draft_tokens   = 5;
    outcome.metrics.speculative_accepted_tokens = 3;
    metrics.record(outcome);

    ninfer::RuntimeStats stats;
    stats.computed_prefill_tokens        = 1234;
    stats.committed_decode_tokens        = 567;
    stats.reused_prompt_tokens           = 89;
    stats.running_requests               = 1;
    stats.prefilling_requests            = 1;
    stats.waiting_requests               = 2;
    stats.materializing_requests         = 1;
    stats.capture_pending_requests       = 1;
    stats.device_state_occupied_slots    = 16;
    stats.host_state_occupied_slots      = 16;
    stats.device_main_kv_occupied_pages  = 1024;
    stats.device_backend_kv_occupied_pages = 32;
    stats.host_kv_occupied_bytes         = (std::size_t{8} << 30);
    stats.kvmem.plans                    = 9;
    stats.kvmem.compactions              = 4;
    stats.kvmem.compact_skips            = 2;
    stats.kvmem.stage_in_blocks          = 11;
    stats.kvmem.stage_out_blocks         = 12;
    stats.kvmem.scoreable_blocks         = 13;
    stats.kvmem.zero_score_blocks        = 14;
    stats.kvmem.window_tokens            = 98304;
    stats.kvmem.host_bytes               = 4096;
    stats.capture_skips.note(ninfer::CaptureSkipReason::Terminal);
    stats.capture_skips.note(ninfer::CaptureSkipReason::PhysicalPeak);

    ninfer::EngineOptions options;
    options.max_concurrency = 1;
    options.kvmem.enabled   = true;
    options.kvmem.budget_tokens     = 98304;
    options.kvmem.gen_reserve_tokens = 16384;

    const std::string text = metrics.render(stats, options);
    failures += check(contains(text, "llamacpp:prompt_tokens_total 60"),
                      "computed prompt tokens are wrong");
    failures += check(contains(text, "llamacpp:tokens_predicted_total 7"),
                      "predicted tokens are wrong");
    failures += check(contains(text, "llamacpp:requests_processing 1"),
                      "processing slots are wrong");
    failures += check(contains(text, "llamacpp:requests_deferred 1"), "deferred slots are wrong");
    failures += check(contains(text, "ninfer:requests_total 2"), "request count is wrong");
    failures += check(contains(text, "ninfer:requests_completed_total 1"),
                      "completed request count is wrong");
    failures += check(contains(text, "ninfer:prefix_cache_hit_tokens_total 40"),
                      "prefix hit tokens are wrong");
    failures += check(contains(text, "ninfer:draft_accepted_tokens_total 3"),
                      "accepted draft tokens are wrong");
    failures += check(contains(text, "ninfer:computed_prefill_tokens_total 1234"),
                      "engine prefill tokens are missing");
    failures += check(contains(text, "ninfer:kvmem_window_tokens 98304"),
                      "kvmem window is missing");
    failures += check(contains(text, "ninfer:capture_skips_total 2"),
                      "capture skip total is wrong");
    failures += check(contains(text, "ninfer:capture_skips{reason=\"terminal\"} 1"),
                      "terminal skip reason is missing");
    failures += check(contains(text, "ninfer:capture_skips{reason=\"physical_peak\"} 1"),
                      "physical_peak skip reason is missing");

    const Json slots = Json::parse(metrics.render_slots_json());
    failures += check(slots.is_array() && slots.size() == 2, "active slots are wrong");
    if (slots.is_array() && slots.size() == 2) {
        bool found = false;
        for (const Json& slot : slots) {
            if (slot.at("id") == 7) {
                found = true;
                failures += check(slot.at("prompt_tokens") == 100,
                                  "slot prompt tokens are wrong");
                failures += check(slot.at("protocol") == "openai_chat_completions",
                                  "slot protocol is wrong");
                failures += check(slot.at("elapsed_seconds").get<double>() >= 0.0,
                                  "slot elapsed time is negative");
            }
        }
        failures += check(found, "request 7 is missing from the slot snapshot");
    }

    metrics.end_request(7);
    const Json after = Json::parse(metrics.render_slots_json());
    failures += check(after.is_array() && after.size() == 1, "finished request stayed active");

    if (failures == 0) { std::cout << "serve metrics tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
