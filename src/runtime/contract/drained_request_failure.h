#pragma once

#include <exception>
#include <utility>

namespace ninfer::runtime {

// A Program may propagate a request-local execution failure through the
// Engine without poisoning the entire worker ONLY after it has synchronized
// queued device work, retired the failing lane and verified all exclusive
// native owners are gone. Preserve the original exception for the HTTP/API
// response. Never use this signal for an unverified CUDA/Cold I/O failure.
class DrainedRequestFailure final : public std::exception {
public:
    explicit DrainedRequestFailure(std::exception_ptr cause) noexcept
        : cause_(std::move(cause)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return "request execution failed after verified physical cleanup";
    }

    [[nodiscard]] std::exception_ptr cause() const noexcept { return cause_; }

private:
    std::exception_ptr cause_;
};

} // namespace ninfer::runtime
