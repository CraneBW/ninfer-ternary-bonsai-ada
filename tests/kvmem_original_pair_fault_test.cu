// Native CUDA/KVMem S1 q0 failure boundary, not a second preflight regression.
//
// Reuse the proven physical PageMajor / original-position / Cold Host1 rig,
// WITHOUT invoking any of its cases or its main(). All assertions below run
// real KVAddressSpaceStore::rewind_original_prefix on independent Main/MTP
// GPU pools, and exercise OriginalRecoveryGuard's failure/cancellation contract.
//
// Sources: Strata 9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4,
// KVMem faf3c22582adf9d99ff072c543ea30ca03a8e2ca. Strata MIT license
// is preserved in third_party/strata/LICENSE.
#define main kvmem_original_pair_preflight_unused_entry
#include "kvmem_original_pair_preflight_test.cu"
#undef main

#include "kvmem/original_recovery_guard.h"
#include "targets/qwen3_6/state_image.h"

#include <exception>
#include <optional>
#include <string_view>

namespace {

namespace state = ninfer::targets::qwen3_6;

enum class Fault : std::uint8_t {
    MtpRewind,
    StateImageRejected,
    StateImageAfterEnqueue,
};

[[nodiscard]] const char* fault_name(Fault fault) noexcept {
    switch (fault) {
    case Fault::MtpRewind: return "mtp_rewind_after_main";
    case Fault::StateImageRejected: return "state_image_invalid_layout";
    case Fault::StateImageAfterEnqueue: return "state_image_after_h2d_enqueue";
    }
    return "invalid_fault";
}

// The Program's request-local q0 proof, separate from both actual KV owners.
// The guard may retire this proof; only the enclosing Program failure handler
// may invalidate a lane capability or release a live KV/StateImage.
struct Q0Request {
    std::optional<KVAddressSpaceStore::OriginalPrefixCheckpoint> main_prefix;
    std::optional<KVAddressSpaceStore::OriginalPrefixCheckpoint> mtp_prefix;
    bool checkpoint_ready = false;
    bool resource_owned = true;
    std::uint32_t lane_epoch = 7;
    std::uint32_t proof_revocations = 0;

    void revoke_q0_proof() noexcept {
        checkpoint_ready = false;
        main_prefix.reset();
        mtp_prefix.reset();
        ++proof_revocations;
    }
};

// True StateImage device + pinned host payload, with GDN conv/recurrent and
// continuation hidden components. Its physical storage is independent of
// Main/MTP, just as in Program's real original-position q0 restore.
struct StateRig {
    DeviceContext& device;
    DeviceBuffer allocation;
    std::unique_ptr<state::StateImageDevicePool> gpu;
    std::unique_ptr<state::HostStatePool> host;
    state::HostStateSlotHandle checkpoint{};
    bool occupied = false;

    explicit StateRig(DeviceContext& device_) : device(device_) {
        state::StateImageSpec spec;
        spec.linear = LinearAttentionStatePoolSpec{
            .layers = 2,
            .conv_channels = 8,
            .conv_width = 3,
            .value_heads = 2,
            .value_head_dim = 8,
            .key_head_dim = 8,
            .slot_count = 1,
            .conv_dtype = DType::BF16,
        };
        spec.hidden = 16;
        LayoutBuilder builder;
        const auto layout = state::plan_state_image_device_pool(builder, spec);
        allocation = DeviceBuffer(builder.finish(256));
        gpu = std::make_unique<state::StateImageDevicePool>(
            DeviceSpan{allocation.p, allocation.bytes}, layout);
        host = std::make_unique<state::HostStatePool>(gpu->host_layout(), 1);
        const auto reserved = host->allocate();
        require(reserved.has_value(), "reserve complete pinned StateImage");
        checkpoint = *reserved;
        occupied = true;

        // A valid saved GDN/conv/hidden q0 payload. Byte-for-byte contents are
        // not a fake scalar flag: copy_from_host will enqueue the real device
        // component copies for all StateImage regions.
        const auto writable = host->writable_view(checkpoint);
        std::memset(writable.data, 0x37, writable.layout->image_bytes);
        gpu->zero_slot(0, device.stream);
        cuda_ok(cudaStreamSynchronize(device.stream), "initialize StateImage");
    }

    void restore_real() {
        gpu->copy_from_host(host->view(checkpoint), 0, device.stream);
    }

    void restore_rejected() {
        // Deliberate *actual* StateImage API failure (not a mock callback).
        // The layout differs from the captured complete StateImage. Store
        // restore must throw after both native KV rewinds have already changed.
        auto invalid = gpu->host_layout();
        ++invalid.image_bytes;
        const auto source = host->view(checkpoint);
        gpu->copy_from_host({source.data, &invalid}, 0, device.stream);
    }

    void drain_and_close() {
        cuda_ok(cudaStreamSynchronize(device.stream), "drain StateImage restoration");
        if (occupied) {
            require(host->release(checkpoint), "release pinned StateImage checkpoint");
            occupied = false;
        }
        require(host->occupied() == 0, "StateImage pinned q0 slot leaked");
    }

    ~StateRig() {
        if (occupied) { (void)host->release(checkpoint); }
    }
};

using NativeCheckpoint = KVAddressSpaceStore::OriginalPrefixCheckpoint;

// Use the same q0 partial append layout as the original native rig (337
// original tokens, 17 committed in its last first-write page), but do not
// re-run any old preflight suite. Main and draft have wholly distinct pools.
NativeCheckpoint prepare_native_probe(Rig& rig) {
    constexpr std::uint32_t q0 = 5 * kPage + 17;
    constexpr std::uint32_t after_probe = 7 * kPage;
    rig.store->ensure_mapped_to_tokens(rig.handle, q0);
    for (int page = 0; page < 6; ++page) {
        rig.write(page, page, 0, page == 5 ? 17 : kPage, 0);
    }
    rig.store->commit_frontier(rig.handle, q0);
    auto checkpoint = rig.store->capture_original_prefix(rig.handle, q0, rig.device.stream);
    rig.store->ensure_mapped_to_tokens(rig.handle, after_probe);
    rig.write(5, 5, 17, kPage, 1);
    rig.write(6, 6, 0, kPage, 1);
    rig.store->commit_frontier(rig.handle, after_probe);
    require(rig.store->committed_frontier(rig.handle) == after_probe,
            "fixture did not establish an actual post-q0 native probe");
    return checkpoint;
}

template<class F>
void expect_poison_rejected(F&& action, const char* message) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

void verify_new_generation(Rig& main, Rig& mtp, const NativeCheckpoint& old_main,
                           const NativeCheckpoint& old_mtp) {
    // This new address incarnation must be able to make a wholly fresh native
    // first-write checkpoint. A poisoned *old* branch may not be resurrected.
    main.open();
    mtp.open();
    constexpr std::uint32_t fresh_q0 = kPage + 1;
    constexpr std::uint32_t fresh_probe = 2 * kPage;
    const auto fresh = [&](Rig& rig) {
        rig.store->ensure_mapped_to_tokens(rig.handle, fresh_q0);
        rig.write(0, 0, 0, kPage, 2);
        rig.write(1, 1, 0, 1, 2);
        rig.store->commit_frontier(rig.handle, fresh_q0);
        const auto checkpoint =
            rig.store->capture_original_prefix(rig.handle, fresh_q0, rig.device.stream);
        rig.store->ensure_mapped_to_tokens(rig.handle, fresh_probe);
        rig.write(1, 1, 1, kPage, 3);
        rig.store->commit_frontier(rig.handle, fresh_probe);
        return checkpoint;
    };
    const auto fresh_main = fresh(main);
    const auto fresh_mtp = fresh(mtp);
    expect_poison_rejected(
        [&] { main.store->validate_original_prefix(main.handle, old_main); },
        "old Main checkpoint escaped its old address generation");
    expect_poison_rejected(
        [&] { mtp.store->validate_original_prefix(mtp.handle, old_mtp); },
        "old MTP checkpoint escaped its old address generation");
    main.store->validate_original_prefix(main.handle, fresh_main);
    mtp.store->validate_original_prefix(mtp.handle, fresh_mtp);
    const auto a =
        main.store->rewind_original_prefix(main.handle, fresh_main, main.device.stream);
    const auto b =
        mtp.store->rewind_original_prefix(mtp.handle, fresh_mtp, mtp.device.stream);
    require(a == fresh_q0 && b == fresh_q0,
            "new request could not recover a valid independent native checkpoint");
    cuda_ok(cudaStreamSynchronize(main.device.stream), "drain fresh native request");
    main.close();
    mtp.close();
    require(main.store->kvmem_inflight_pages() == 0 &&
            mtp.store->kvmem_inflight_pages() == 0,
            "new generation retained inflight source/Cold IO");
}

void run_fault(DeviceContext& device, Fault fault) {
    Rig main(device), mtp(device);
    StateRig state_image(device);
    Q0Request request;
    request.main_prefix = prepare_native_probe(main);
    request.mtp_prefix = prepare_native_probe(mtp);
    request.checkpoint_ready = true;
    const auto stale_main = *request.main_prefix;
    const auto stale_mtp = *request.mtp_prefix;

    // Legitimate production order: ALL metadata + native owner validators
    // pass before guard creation and before any destructive Main mutation.
    main.store->validate_original_prefix(main.handle, *request.main_prefix);
    mtp.store->validate_original_prefix(mtp.handle, *request.mtp_prefix);
    require(state_image.host->view(state_image.checkpoint).layout->image_bytes ==
                state_image.gpu->host_layout().image_bytes,
            "complete q0 StateImage shape preflight");

    bool caught = false;
    std::uint32_t main_rewind = 0;
    std::uint32_t mtp_rewind = 0;
    try {
        auto guard = ninfer::kvmem::make_original_recovery_guard(
            *main.store, main.handle, mtp.store.get(),
            std::optional<KVAddressSpaceHandle>{mtp.handle},
            [&]() noexcept { request.revoke_q0_proof(); });
        guard.begin_mutation();
        main_rewind = main.store->rewind_original_prefix(
            main.handle, *request.main_prefix, device.stream);
        require(main_rewind == 5 * kPage + 17 &&
                main.store->committed_frontier(main.handle) == main_rewind,
                "native Main rewind did not really change the probe KV");

        if (fault == Fault::MtpRewind) {
            // *After* both tiers successfully preflighted, inject a stale
            // first-write revision at MTP's actual restore entrypoint. The
            // physical store itself throws; Main has already destructively
            // rewound its actual KV mapping.
            auto stale_at_rewind = *request.mtp_prefix;
            stale_at_rewind.prefix_revisions[0] ^= 1U;
            (void)mtp.store->rewind_original_prefix(
                mtp.handle, stale_at_rewind, device.stream);
            throw std::logic_error("actual MTP rewind accepted an injected bad revision");
        }

        mtp_rewind = mtp.store->rewind_original_prefix(
            mtp.handle, *request.mtp_prefix, device.stream);
        require(mtp_rewind == main_rewind,
                "Main/MTP native rewinds returned different windows");
        if (fault == Fault::StateImageRejected) {
            state_image.restore_rejected();
            throw std::logic_error("actual StateImage accepted invalid source layout");
        }

        // The entire StateImage H2D transfer was genuinely enqueued, including
        // recurrent GDN/conv and hidden state. Inject the failure BEFORE its
        // completion boundary; the outer Program must drain the stream.
        state_image.restore_real();
        throw std::runtime_error("injected StateImage error after async H2D enqueue");
    } catch (const std::exception& error) {
        caught = true;
        if (fault == Fault::MtpRewind) {
            require(std::string_view(error.what()).find("revision") != std::string_view::npos ||
                    std::string_view(error.what()).find("first-write") != std::string_view::npos,
                    "MTP fault came from an unrelated failure");
        } else if (fault == Fault::StateImageRejected) {
            require(std::string_view(error.what()).find("StateImage") != std::string_view::npos ||
                    std::string_view(error.what()).find("layouts") != std::string_view::npos,
                    "StateImage layout fault came from an unrelated failure");
        } else {
            require(std::string_view(error.what()).find("after async H2D") != std::string_view::npos,
                    "post-H2D injection did not reach StateImage copy");
        }
    }
    require(caught && main_rewind == 5 * kPage + 17,
            "fault never crossed the actual destructive Main rewind");
    if (fault != Fault::MtpRewind) {
        require(mtp_rewind == main_rewind,
                "StateImage error occurred before successful MTP rewind");
    } else {
        require(mtp.store->committed_frontier(mtp.handle) == 7 * kPage,
                "stale MTP restore mutated its own native frontier");
    }

    // The actual original-epoch invalidators are invoked by RAII, not manually
    // called by the fixture. They must poison BOTH stores, even if MTP itself
    // never started a write, and revoke q0 without clearing a live lane.
    require(!request.checkpoint_ready && !request.main_prefix &&
                !request.mtp_prefix && request.proof_revocations == 1 &&
                request.lane_epoch == 7 && request.resource_owned,
            "guard did not revoke exactly the q0 proof without early lane cleanup");
    expect_poison_rejected(
        [&] { main.store->validate_original_prefix(main.handle, stale_main); },
        "poisoned Main branch remained a valid checkpoint");
    expect_poison_rejected(
        [&] { mtp.store->validate_original_prefix(mtp.handle, stale_mtp); },
        "poisoned MTP branch remained a valid checkpoint");
    expect_poison_rejected(
        [&] { (void)main.store->rewind_original_prefix(main.handle, stale_main, device.stream); },
        "poisoned Main branch still accepts a destructive rewind");
    expect_poison_rejected(
        [&] { (void)mtp.store->rewind_original_prefix(mtp.handle, stale_mtp, device.stream); },
        "poisoned MTP branch still accepts a destructive rewind");

    // Existing Program outer failure path owns this order, not the guard:
    // drain real stream and pending IO before releasing either native owner.
    cuda_ok(cudaStreamSynchronize(device.stream), "drain post-fault CUDA reader/writer stream");
    state_image.drain_and_close();
    (void)main.store->retire_kvmem_transfers(true);
    (void)mtp.store->retire_kvmem_transfers(true);
    require(main.store->kvmem_inflight_pages() == 0 &&
            mtp.store->kvmem_inflight_pages() == 0 &&
            main.store->kvmem_layer_pending_pages() == 0 &&
            mtp.store->kvmem_layer_pending_pages() == 0,
            "fault left a native transfer or pinned source/Cold page in flight");
    main.close();
    mtp.close();
    request.resource_owned = false;
    ++request.lane_epoch; // Outer Program cleanup, not the guard.
    require(!request.resource_owned && request.lane_epoch == 8,
            "outer request cleanup did not retire request generation");

    verify_new_generation(main, mtp, stale_main, stale_mtp);
    std::printf("{\"case\":\"%s\",\"verdict\":\"PASS\","
                "\"main_destructive_rewind\":true,\"mtpfault_after_preflight\":%s,"
                "\"state_image_real_h2d\":%s,\"two_tiers_poisoned\":true,"
                "\"request_q0_revoked\":true,\"full_release\":true,"
                "\"new_request_isolated\":true}\n",
                fault_name(fault),
                fault == Fault::MtpRewind ? "true" : "false",
                fault == Fault::StateImageAfterEnqueue ? "true" : "false");
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        require(argc == 1,
                "usage: ninfer_kvmem_original_pair_fault_test (self-owned cold directory)");
        const auto directory = std::filesystem::temp_directory_path() /
            ("ninfer-kvmem-pair-fault-" + std::to_string(::getpid()));
        require(std::filesystem::create_directory(directory),
                "fault test Cold directory already exists");
        require(::setenv("NINFER_KVMEM_ORIGINAL_POSITIONS", "1", 1) == 0,
                "select native original-position mode");
        require(::setenv("NINFER_KVMEM_HOST_PAGES", "1", 1) == 0,
                "limit Host tier to one page");
        require(::setenv("NINFER_KVMEM_COLD_DIR", directory.c_str(), 1) == 0,
                "select isolated Cold directory");
        ::unsetenv("NINFER_KVMEM_TRACE");
        DeviceContext device;
        run_fault(device, Fault::MtpRewind);
        run_fault(device, Fault::StateImageRejected);
        run_fault(device, Fault::StateImageAfterEnqueue);
        cuda_ok(cudaStreamSynchronize(device.stream), "final fault fixture stream drain");
        std::filesystem::remove_all(directory);
        std::puts("{\"summary\":true,\"cases\":3,\"verdict\":\"PASS\","
                  "\"native_main_mtp\":true,\"real_state_image\":true,"
                  "\"generation_isolated\":true,\"models_loaded\":0}");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_ORIGINAL_PAIR_FAULT FAIL: %s\n", error.what());
        return 1;
    }
}
