// Targeted CPU test of NInfer original q0 paired failure/cancellation ownership.
// The mock's invalidate_original_prefix_branch method matches the real
// KVAddressSpaceStore contract: revoke an active address epoch, leave memory
// and stream teardown to the existing Program failure path.
#include "kvmem/original_recovery_guard.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

using ninfer::kvmem::make_original_recovery_guard;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "original recovery guard failed: %s\n", message);
        std::exit(1);
    }
}

template<class Action>
void throws(Action&& action, const char* message) {
    bool rejected = false;
    try { action(); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, message);
}

struct Address {
    std::uint32_t index;
    std::uint64_t generation;
};

struct NativeTier {
    Address active{0, 1};
    std::uint64_t original_epoch = 7;
    std::uint32_t payload = 10;
    std::uint32_t invalidations = 0;
    std::uint32_t releases = 0;

    void invalidate_original_prefix_branch(Address h) noexcept {
        if (h.index == active.index && h.generation == active.generation) {
            original_epoch = 0;
            ++invalidations;
        }
    }

    void rewind(Address h, std::uint32_t replacement) {
        if (h.index != active.index || h.generation != active.generation ||
            original_epoch == 0) {
            throw std::logic_error("stale native address or failed original rewind");
        }
        payload = replacement;
    }
};

struct Request {
    bool q0_ready = true;
    std::optional<std::uint64_t> main_capture = 11;
    std::optional<std::uint64_t> mtp_capture = 13;
    std::uint64_t lane_epoch = 44;
    bool resources_owned = true;
    bool terminal = false;
    std::uint32_t request_invalidations = 0;
    std::uint32_t resource_releases = 0;

    void invalidate_local_q0() noexcept {
        q0_ready = false;
        main_capture.reset();
        mtp_capture.reset();
        ++request_invalidations;
        // Deliberately do NOT modify lane_epoch or physical ownership.
    }

    // The existing outer advance_prefill catch owns the real equivalent:
    // clear_execution_failure_lanes, after it has drained outstanding GPU IO.
    void outer_cleanup_after_drain() noexcept {
        ++lane_epoch;
        resources_owned = false;
        ++resource_releases;
    }
};

void check_preflight_and_success() {
    NativeTier main, mtp;
    main.active = {3, 71};
    mtp.active = {6, 85};
    Request request;
    const Address main_handle = main.active, mtp_handle = mtp.active;

    // Owner/lineage validation is still done by the real store, before the
    // first mutation and before the guard becomes armed.
    throws([&] {
        if (mtp.original_epoch != 9) {
            throw std::logic_error("draft epoch preflight refused");
        }
        auto unused = make_original_recovery_guard(
            main, main_handle, &mtp, std::optional<Address>{mtp_handle},
            [&]() noexcept { request.invalidate_local_q0(); });
        unused.begin_mutation();
    }, "invalid MTP preflight did not reject");
    require(main.original_epoch == 7 && mtp.original_epoch == 7 &&
            main.payload == 10 && mtp.payload == 10 &&
            request.q0_ready && request.request_invalidations == 0,
            "failed preflight mutated a Main/MTP/request owner");

    {
        auto guard = make_original_recovery_guard(
            main, main_handle, &mtp, std::optional<Address>{mtp_handle},
            [&]() noexcept { request.invalidate_local_q0(); });
        // Creating a guard without starting mutation cannot revoke valid
        // checkpoints, even on an early no-op return from preflight code.
        require(!guard.mutation_started(), "guard armed before Main rewind");
    }
    require(main.invalidations == 0 && mtp.invalidations == 0 &&
            request.request_invalidations == 0,
            "read-only q0 check incorrectly revoked existing owners");

    {
        auto guard = make_original_recovery_guard(
            main, main_handle, &mtp, std::optional<Address>{mtp_handle},
            [&]() noexcept { request.invalidate_local_q0(); });
        guard.begin_mutation();
        throws([&] { guard.begin_mutation(); }, "double begin accepted");
        main.rewind(main_handle, 101);
        mtp.rewind(mtp_handle, 103);
        const bool stream_synchronized = true;
        require(stream_synchronized, "fixture device completion barrier absent");
        guard.commit();
        require(guard.committed(), "fully restored q0 did not become publishable");
        guard.cancel();  // Later ordinary request cancellation belongs to Program.
    }
    require(main.payload == 101 && mtp.payload == 103 &&
            main.original_epoch != 0 && mtp.original_epoch != 0 &&
            main.invalidations == 0 && mtp.invalidations == 0 &&
            request.q0_ready && request.lane_epoch == 44 && request.resources_owned,
            "successful restore was revoked by guard teardown");
}

void check_partial_mutation_error() {
    NativeTier main, mtp;
    main.active = {3, 71};
    mtp.active = {4, 72};
    Request request;
    const Address main_handle = main.active, mtp_handle = mtp.active;

    try {
        auto guard = make_original_recovery_guard(
            main, main_handle, &mtp, std::optional<Address>{mtp_handle},
            [&]() noexcept { request.invalidate_local_q0(); });
        guard.begin_mutation();
        main.rewind(main_handle, 21);
        // MTP may fail after native Main has already truncated future pages.
        throw std::runtime_error("injected MTP first-write I/O failure");
    } catch (const std::runtime_error&) {}
    require(main.payload == 21 && mtp.payload == 10,
            "fixture did not exercise a partially mutated restore");
    require(main.original_epoch == 0 && mtp.original_epoch == 0 &&
            main.invalidations == 1 && mtp.invalidations == 1 &&
            !request.q0_ready && !request.main_capture && !request.mtp_capture &&
            request.request_invalidations == 1,
            "unwind did not revoke both native epochs and original q0 proof");
    require(request.lane_epoch == 44 && request.resources_owned &&
            request.resource_releases == 0 &&
            main.releases == 0 && mtp.releases == 0,
            "guard duplicated Program cleanup before GPU readers could drain");
    throws([&] { main.rewind(main_handle, 31); },
           "poisoned Main original checkpoint was still reusable");
    throws([&] { mtp.rewind(mtp_handle, 33); },
           "poisoned MTP original checkpoint was still reusable");
    request.outer_cleanup_after_drain();
    require(request.lane_epoch == 45 && !request.resources_owned &&
            request.resource_releases == 1,
            "outer Program failure cleanup was not independent of guard");
}

void check_failure_after_state_mutation() {
    NativeTier main, mtp;
    Request request;
    std::uint32_t state_image = 9;
    try {
        auto guard = make_original_recovery_guard(
            main, main.active, &mtp, std::optional<Address>{mtp.active},
            [&]() noexcept { request.invalidate_local_q0(); });
        guard.begin_mutation();
        main.rewind(main.active, 111);
        mtp.rewind(mtp.active, 115);
        state_image = 17;  // Host GDN/conv/hidden copy has been submitted.
        throw std::runtime_error("device stream failed before StateImage sync");
    } catch (const std::runtime_error&) {}
    require(state_image == 17 && main.payload == 111 && mtp.payload == 115 &&
            main.original_epoch == 0 && mtp.original_epoch == 0 && !request.q0_ready,
            "partially completed StateImage was published after stream failure");
    require(request.resources_owned && request.lane_epoch == 44,
            "guard released request objects before outer stream drain");
}

void check_cancel_boundaries() {
    for (const bool began_mutation : {false, true}) {
        NativeTier main, mtp;
        Request request;
        {
            auto guard = make_original_recovery_guard(
                main, main.active, &mtp, std::optional<Address>{mtp.active},
                [&]() noexcept { request.invalidate_local_q0(); });
            if (began_mutation) {
                guard.begin_mutation();
                main.rewind(main.active, 27);
            }
            guard.cancel();
            guard.cancel();
            require(guard.revoked(), "cancel did not close restoration lifetime");
            throws([&] { guard.commit(); }, "cancelled q0 restoration committed");
            throws([&] { guard.begin_mutation(); }, "cancelled q0 restoration restarted");
            require(main.original_epoch == 0 && mtp.original_epoch == 0 &&
                    main.invalidations == 1 && mtp.invalidations == 1 &&
                    request.request_invalidations == 1 && !request.q0_ready,
                    "explicit cancellation did not revoke the complete q0 pair");
            require(request.resources_owned && request.lane_epoch == 44 &&
                    !request.terminal, "cancel modified owner or external request lifecycle");
        }
        require(main.invalidations == 1 && mtp.invalidations == 1 &&
                request.request_invalidations == 1,
                "destructor repeated explicit cancellation");
    }
}

void check_owner_generation_and_optional_draft() {
    NativeTier main;
    Request request;
    const Address retired{main.active.index, main.active.generation};
    {
        auto guard = make_original_recovery_guard(
            main, retired, static_cast<NativeTier*>(nullptr),
            std::optional<Address>{},
            [&]() noexcept { request.invalidate_local_q0(); });
        guard.begin_mutation();
        // A slot is reissued before a stale guard retires. Real owner handles
        // are generation-checked; abort must not poison a newer incarnation.
        ++main.active.generation;
    }
    require(main.original_epoch == 7 && main.invalidations == 0 &&
            !request.q0_ready && request.request_invalidations == 1,
            "stale handle guard invalidated another owner's generation");

    Request another;
    throws([&] {
        auto invalid = make_original_recovery_guard(
            main, main.active, static_cast<NativeTier*>(nullptr),
            std::optional<Address>{main.active},
            [&]() noexcept { another.invalidate_local_q0(); });
        (void)invalid;
    }, "unpaired backend address admitted");
    require(another.request_invalidations == 0 && main.invalidations == 0,
            "invalid constructor made changes before any restoration");
    throws([&] {
        auto invalid = make_original_recovery_guard(
            main, main.active, &main, std::optional<Address>{},
            [&]() noexcept { another.invalidate_local_q0(); });
        (void)invalid;
    }, "unpaired backend store admitted");
    require(another.q0_ready && another.request_invalidations == 0,
            "invalid backend pairing changed request q0");
}

} // namespace

int main() {
    check_preflight_and_success();
    check_partial_mutation_error();
    check_failure_after_state_mutation();
    check_cancel_boundaries();
    check_owner_generation_and_optional_draft();
    std::puts("ORIGINAL_RECOVERY_GUARD CPU PASS");
}
