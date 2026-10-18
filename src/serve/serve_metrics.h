#pragma once

// Product-side request accounting for the two monitoring endpoints:
//   GET /metrics  Prometheus text, llama.cpp-compatible names first, then NInfer and KVMem gauges
//   GET /slots    JSON snapshot of the requests currently inside GenerationService
// This is diagnostics only: nothing here changes scheduling, capture admission, or request logs.

#include "serve/request_events.h"

#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    void begin_request(std::uint64_t id, const RequestLogContext& context);
    void end_request(std::uint64_t id) noexcept;
    void record(const GenerationOutcome& outcome);

    [[nodiscard]] std::string render(const ninfer::RuntimeStats& stats,
                                     const ninfer::EngineOptions& options) const;
    [[nodiscard]] std::string render_slots_json() const;

private:
    struct ActiveRequest {
        std::uint64_t id = 0;
        std::string protocol;
        std::string model;
        bool stream   = false;
        int prompt_tokens = 0;
        std::chrono::steady_clock::time_point started;
    };

    mutable std::mutex mutex_;
    std::map<std::uint64_t, ActiveRequest> active_;
    std::uint64_t requests_total_          = 0;
    std::uint64_t requests_completed_total_ = 0;
    std::uint64_t prompt_tokens_total_     = 0;
    std::uint64_t tokens_predicted_total_  = 0;
    double prompt_seconds_total_           = 0.0;
    double predicted_seconds_total_        = 0.0;
    double ttft_seconds_total_             = 0.0;
    std::uint64_t prefix_cache_hit_tokens_total_ = 0;
    std::uint64_t draft_tokens_total_      = 0;
    std::uint64_t draft_accepted_tokens_total_ = 0;
};

} // namespace ninfer::serve
