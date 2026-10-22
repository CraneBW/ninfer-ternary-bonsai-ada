#pragma once

#include "core/device.h"
#include "kvmem/kvmem_bridge.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace ninfer::kvmem {

// One bounded, opt-in producer/consumer handoff for a PageMajor Main16 window.
// The ordinary KVMem page reservation and address generation remain the owners
// of the bytes. This object only owns stream ordering; it never owns a page.
// The producer records ready[i] *after* all rephase writes to layer i; the
// attention stream explicitly waits before its first cached-KV access and
// records released[i] after enqueueing its final read of that layer.
class LayerReadyFence final : public LayerReadyConsumer {
public:
    static constexpr std::uint32_t kLayers = 16;

    LayerReadyFence() {
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&producer_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&source_released_, cudaEventDisableTiming));
            for (auto& event : ready_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            for (auto& event : released_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~LayerReadyFence() override {
        if (producer_ && cudaStreamSynchronize(producer_) != cudaSuccess) { std::terminate(); }
        if (consumer_ && cudaStreamSynchronize(consumer_) != cudaSuccess) { std::terminate(); }
        destroy();
    }

    LayerReadyFence(const LayerReadyFence&) = delete;
    LayerReadyFence& operator=(const LayerReadyFence&) = delete;

    // Serializes reuse of the one pinned/device rephase metadata slot. The new
    // producer also waits for all earlier attention/Graph readers already
    // enqueued on the same compute stream. Never infer this from CPU return.
    void begin(cudaStream_t compute) {
        if (compute == nullptr || compute == cudaStreamLegacy) {
            throw std::invalid_argument("layer-ready requires a nonlegacy consumer stream");
        }
        if (producer_ == compute) { throw std::logic_error("layer-ready streams must differ"); }
        CUDA_CHECK(cudaStreamSynchronize(producer_));
        consumer_ = compute;
        CUDA_CHECK(cudaEventRecord(source_released_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(producer_, source_released_, 0));
        ready_recorded_.fill(false);
        consumer_waited_.fill(false);
        reader_released_.fill(false);
        active_ = true;
        ++epoch_;
    }

    [[nodiscard]] cudaStream_t producer_stream() const noexcept { return producer_; }
    [[nodiscard]] cudaEvent_t ready_event(std::uint32_t layer) const {
        if (layer >= kLayers) { throw std::out_of_range("layer-ready layer index"); }
        return ready_[layer];
    }

    void mark_ready(std::uint32_t layer) {
        if (!active_ || layer >= kLayers || ready_recorded_[layer]) {
            throw std::logic_error("layer-ready event is not publishable");
        }
        // The caller already recorded the event on producer_ directly.
        ready_recorded_[layer] = true;
    }

    void wait_before_layer(std::uint32_t layer, cudaStream_t compute) override {
        if (!active_ || layer >= kLayers || !ready_recorded_[layer] || compute == nullptr ||
            compute != consumer_) {
            // A different stream could wait for ready but would not be ordered
            // with the release event queued on the stream bound by begin().
            throw std::logic_error("layer-ready wait needs the bound consumer stream");
        }
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
        consumer_waited_[layer] = true;
    }

    void release_after_layer(std::uint32_t layer, cudaStream_t compute) override {
        if (!active_ || layer >= kLayers || !consumer_waited_[layer] || compute == nullptr ||
            compute != consumer_) {
            // Recording released[] on an unrelated stream can retire a physical
            // page while its actual attention reader is still in flight.
            throw std::logic_error("layer-ready release needs the bound consumer stream");
        }
        CUDA_CHECK(cudaEventRecord(released_[layer], compute));
        reader_released_[layer] = true;
    }

    // Called at the next compaction/error boundary before any physical targets,
    // metadata buffer or event epoch can be recycled. A subsequent producer
    // waits for the compute stream's last queued reader in begin().
    void retire() {
        if (!active_) { return; }
        CUDA_CHECK(cudaStreamSynchronize(producer_));
        active_ = false;
    }

    // An address release, truncate or cancelled prefill may retire its page
    // reservation instead of scheduling another kernel on the same stream.
    // Unlike ordinary epoch reuse, that requires draining actual readers too.
    void drain_consumers() {
        retire();
        if (consumer_) { CUDA_CHECK(cudaStreamSynchronize(consumer_)); }
    }

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

private:
    void destroy() noexcept {
        for (auto& event : released_) {
            if (event) { (void)cudaEventDestroy(event); event = nullptr; }
        }
        for (auto& event : ready_) {
            if (event) { (void)cudaEventDestroy(event); event = nullptr; }
        }
        if (source_released_) { (void)cudaEventDestroy(source_released_); source_released_ = nullptr; }
        if (producer_) { (void)cudaStreamDestroy(producer_); producer_ = nullptr; }
    }

    cudaStream_t producer_ = nullptr;
    cudaStream_t consumer_ = nullptr; // non-owning; Program/DeviceContext outlives the store
    cudaEvent_t source_released_ = nullptr;
    std::array<cudaEvent_t, kLayers> ready_{};
    std::array<cudaEvent_t, kLayers> released_{};
    std::array<bool, kLayers> ready_recorded_{};
    std::array<bool, kLayers> consumer_waited_{};
    std::array<bool, kLayers> reader_released_{};
    bool active_ = false;
    std::uint64_t epoch_ = 0;
};

} // namespace ninfer::kvmem
