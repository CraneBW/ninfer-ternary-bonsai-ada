#pragma once

// Scoped failure boundary for the original-position q0 Main/MTP + StateImage
// rewind. Adapted to NInfer's existing native KVAddressSpaceStore invalidator
// and Program lane epoch from:
//   Strata 9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4
//     src/core/conversation_state.cpp (validate every tier before restore);
//   KVMem faf3c22582adf9d99ff072c543ea30ca03a8e2ca
//     src/adapter/llama-memory-kvmem.cpp (drain writers before old store discard).
// Strata licensing: third_party/strata/LICENSE.
//
// This guard NEVER frees native pages, StateImage slots, pinned source buffers
// or streams, nor invalidate the active lane's capability: the existing
// Program::advance_prefill failure path owns those operations and must drain
// CUDA/Cold work before releasing the request. Revocation only makes the
// request-local q0 proof and native recovery tuple unpublishable. It calls each
// KV store's *existing* invalidate_original_prefix_branch(handle) instead of
// reimplementing its address lookup or original-epoch rules.

#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::kvmem {

template<class KVStore, class KVHandle, class InvalidateRequest>
class OriginalRecoveryGuard final {
    static_assert(std::is_nothrow_copy_constructible_v<KVHandle>,
                  "original KV handles must be safe to pass by value in noexcept cleanup");
    static_assert(noexcept(std::declval<KVStore&>().invalidate_original_prefix_branch(
                      std::declval<KVHandle>())),
                  "use the existing noexcept original-prefix invalidator");
    static_assert(std::is_nothrow_invocable_v<InvalidateRequest&>,
                  "request q0/sequence capability invalidation must be noexcept");

public:
    // Both stores own their respective KV address generations independently.
    // nullptr + nullopt is the only permitted form for a model without MTP.
    // The request callback must revoke ONLY the request-local original q0
    // checkpoint eligibility (flag and captured prefix optionals). Do not
    // invalidate_lane(), clear a RequestControl or release its physical owners
    // here: Program::clear_execution_failure_lanes does all of that after this
    // call stack has unwound and its borrowed references are no longer live.
    OriginalRecoveryGuard(KVStore& main, KVHandle main_handle,
                          KVStore* backend, std::optional<KVHandle> backend_handle,
                          InvalidateRequest invalidate_request)
        : main_(&main), main_handle_(main_handle),
          backend_(backend), backend_handle_(backend_handle),
          invalidate_request_(std::move(invalidate_request)) {
        if ((backend_ == nullptr) != !backend_handle_.has_value()) {
            throw std::invalid_argument("original recovery Main/MTP owner pair is incomplete");
        }
    }

    OriginalRecoveryGuard(const OriginalRecoveryGuard&) = delete;
    OriginalRecoveryGuard& operator=(const OriginalRecoveryGuard&) = delete;
    OriginalRecoveryGuard(OriginalRecoveryGuard&&) = delete;
    OriginalRecoveryGuard& operator=(OriginalRecoveryGuard&&) = delete;

    ~OriginalRecoveryGuard() noexcept {
        // A return/exception after the first potentially destructive native
        // call must never leave a usable half-restored Main/MTP/StateImage view.
        if (phase_ == Phase::Mutating) { cancel(); }
    }

    // Call only AFTER the joint q0 + Main + MTP + StateImage preflight, directly
    // before Main's first rewind. Preflight itself is read-only and any failure
    // there leaves both native branches intact (Program still unwinds request).
    void begin_mutation() {
        if (phase_ != Phase::BeforeMutation) {
            throw std::logic_error("original recovery mutation phase is not available");
        }
        phase_ = Phase::Mutating;
    }

    // Call only AFTER both rewinds, StateImage restore, suffix admission and
    // required device-stream synchronization/validation are fully successful.
    // A premature commit would allow the next consumer to read partial state.
    void commit() {
        if (phase_ != Phase::Mutating) {
            throw std::logic_error("original recovery cannot publish this branch");
        }
        phase_ = Phase::Committed;
    }

    // Explicit cancellation is valid *both* before Main starts mutating and
    // during a partially submitted Main/MTP/StateImage restore. It revokes
    // only the current private q0 branch, not other continuation owners.
    // Idempotent across cancel(), exception unwinding and the destructor.
    // An outer Program cleanup is still required to drain and release owners.
    void cancel() noexcept {
        if (phase_ == Phase::Committed || phase_ == Phase::Revoked) { return; }
        phase_ = Phase::Revoked;
        // Revoke the private q0 proof first, preventing this same request
        // from treating a half-restored Main/MTP/StateImage as restartable.
        // The outer Program catch owns the lane capability invalidation.
        std::invoke(invalidate_request_);
        main_->invalidate_original_prefix_branch(main_handle_);
        if (backend_) {
            backend_->invalidate_original_prefix_branch(*backend_handle_);
        }
    }

    [[nodiscard]] bool revoked() const noexcept { return phase_ == Phase::Revoked; }
    [[nodiscard]] bool committed() const noexcept { return phase_ == Phase::Committed; }
    [[nodiscard]] bool mutation_started() const noexcept { return phase_ == Phase::Mutating; }

private:
    enum class Phase : unsigned char {
        BeforeMutation,
        Mutating,
        Committed,
        Revoked,
    };

    KVStore* main_ = nullptr;
    KVHandle main_handle_;
    KVStore* backend_ = nullptr;
    std::optional<KVHandle> backend_handle_;
    InvalidateRequest invalidate_request_;
    Phase phase_ = Phase::BeforeMutation;
};

template<class KVStore, class KVHandle, class InvalidateRequest>
[[nodiscard]] auto make_original_recovery_guard(
    KVStore& main, KVHandle main_handle,
    KVStore* backend, std::optional<KVHandle> backend_handle,
    InvalidateRequest&& invalidate_request) {
    return OriginalRecoveryGuard<KVStore, KVHandle, std::decay_t<InvalidateRequest>>{
        main, main_handle, backend, backend_handle,
        std::forward<InvalidateRequest>(invalidate_request)};
}

} // namespace ninfer::kvmem
