#pragma once

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/host_worker_pool.h"
#include "kvmem/kvmem_bridge.h"
#include "kvmem/kvmem_cold_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <span>

namespace ninfer::kvmem {

// Immutable-source policy adapted from KVMem/ActKV kvmem_store.cpp and
// raw_kv_store.cpp. Preserve the FIRST represented K codes/scales and each
// column's birth position. Decode/un-RoPE this source when rebuilding K, rather
// than decoding a repeatedly quantized working page. This is equivalent to a
// lazy raw-K mirror at the initial INT8 boundary; it is not pre-quantization K.
// Only K is backed here. V and the ordinary Host/Cold payload keep their owner.
class ImmutableKStore {
public:
    struct Record {
        ColdKVStore::Handle backing;
        std::array<std::int32_t, kPagedKVPageSize> positions{};
        std::uint32_t columns = 0;
    };

    struct RestoreRequest {
        const Record* record;
        std::int32_t physical;
        std::int32_t destination;
    };

    explicit ImmutableKStore(const KVPageGeometry& geometry)
        : layout_(plan_host_kv_page_layout(geometry)) {
        if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
            throw std::invalid_argument("immutable K requires INT8 layer plane groups");
        }
        // This bounded experiment targets the existing Qwen3.8 GSQ Main16/H4
        // INT8 layout. Other geometries retain their existing pitched copies.
        copy_batch_geometry_ = geometry.page_tokens == kPagedKVPageSize &&
            geometry.device_plane_order == PagedKVPlaneOrder::PageMajor &&
            geometry.planes.size() == 64;
        // Reuse the native packed Host offsets for K-code and K-scale planes.
        offsets_.resize(layout_.planes.size());
        for (std::size_t i = 0; i < offsets_.size(); i += 2) {
            offsets_[i] = bytes_;
            bytes_ += layout_.planes[i].page_payload_bytes;
            const auto& plane = geometry.planes[i];
            copy_batch_geometry_ = copy_batch_geometry_ && plane.head_extent == 4 &&
                (i % 4 == 0 ? plane.dtype == DType::I8 && plane.leading_extent == 256
                            : plane.dtype == DType::FP16 && plane.leading_extent == 4);
        }
        staging_ = std::make_unique<PinnedHostBuffer>(bytes_);
        backing_ = std::make_unique<ColdKVStore>(bytes_);
        if (source_read_ahead_enabled() && event_pipeline_enabled()) {
            // cudaHostAlloc may wait on outstanding transfers. Allocate before
            // the first async demotion, never in its intended overlap interval.
            read_ahead_staging_ = std::make_unique<PinnedHostBuffer>(bytes_);
            event_pipeline_ = std::make_unique<EventPipeline>();
        }
    }

    ~ImmutableKStore() {
        // The second pinned slot and the stable source backing must outlive
        // every previously submitted per-layer H2D. In particular, an
        // exception before the final ready event cannot abandon a DMA reader.
        if (layer_source_stream_ && cudaStreamSynchronize(layer_source_stream_) != cudaSuccess) {
            std::terminate();
        }
        (void)drain_pipeline_reads();
        source_workers_.reset(); // Join workers while backing and all pinned slots still live.
        if (stage_trace_enabled() && stage_timing_.restore_pages != 0) {
            // GPU event durations cover just the copies and the re-RoPE kernel;
            // overlapping prefetch waits are host time and must not be added to them.
            std::fprintf(stderr,
                "KVMEM_STAGE_SOURCE pages=%llu moved=%llu read_sync_us=%llu "
                "read_wait_us=%llu h2d_gpu_ms=%.3f rebuild_gpu_ms=%.3f "
                "gpu_timed_pages=%llu\n",
                static_cast<unsigned long long>(stage_timing_.restore_pages),
                static_cast<unsigned long long>(stage_timing_.moved_pages),
                static_cast<unsigned long long>(stage_timing_.read_sync_us),
                static_cast<unsigned long long>(stage_timing_.read_wait_us),
                stage_timing_.h2d_gpu_ms, stage_timing_.rebuild_gpu_ms,
                static_cast<unsigned long long>(stage_timing_.gpu_timed_pages));
        }
        if (event_pipeline_pages_ && std::getenv("NINFER_KVMEM_PREFIX_TRACE")) {
            std::fprintf(stderr, "SOURCE_EVENT_PIPELINE pages=%llu staging_bytes=%zu slots=2\n",
                         static_cast<unsigned long long>(event_pipeline_pages_), staging_bytes());
        }
        if (pipeline_trace_emitted_ && r26_bundle_trace()) {
            std::fprintf(stderr,
                "R26_SOURCE_PIPELINE_FINAL read_submits=%llu groups_started=%llu "
                "groups_completed=%llu pages_completed=%llu moved_pages=%llu "
                "rebuild_calls=%llu stream_sync_calls=%llu error_drains=%llu staging_bytes=%zu\n",
                static_cast<unsigned long long>(pipeline_stats_.read_submits),
                static_cast<unsigned long long>(pipeline_stats_.groups_started),
                static_cast<unsigned long long>(pipeline_stats_.groups_completed),
                static_cast<unsigned long long>(pipeline_stats_.pages_completed),
                static_cast<unsigned long long>(pipeline_stats_.moved_pages),
                static_cast<unsigned long long>(pipeline_stats_.rebuild_calls),
                static_cast<unsigned long long>(pipeline_stats_.stream_sync_calls),
                static_cast<unsigned long long>(pipeline_stats_.error_drains), staging_bytes());
        }
    }

    template<class Writer> void save_record(const Record& record, Writer& out) {
        drain_layer_source();
        out.pod(record.columns); out.pod(record.positions);
        out.pod(static_cast<std::uint8_t>(record.backing.valid()));
        if (record.backing.valid()) {
            backing_->load(record.backing,{static_cast<std::byte*>(staging_->data()),bytes_});
            out.write(staging_->data(),bytes_);
        }
    }
    template<class Reader> void load_record(Record& record, Reader& in) {
        drain_layer_source();
        record.columns = in.template pod<std::uint32_t>();
        record.positions = in.template pod<decltype(record.positions)>();
        const auto present = in.template pod<std::uint8_t>();
        if (record.columns > kPagedKVPageSize || present > 1 ||
            (record.columns != 0 && !present)) { throw std::invalid_argument("snapshot raw-K coverage"); }
        if (present) {
            const auto snapshot=in.stable_backing();
            const auto offset=in.position();
            in.read(staging_->data(),bytes_); // full per-consume integrity still applies
            record.backing = snapshot ? backing_->adopt_snapshot_extent(snapshot,offset)
                : backing_->store({static_cast<const std::byte*>(staging_->data()),bytes_});
        }
    }

    void release(Record& record) noexcept {
        if (layer_source_stream_ && cudaStreamSynchronize(layer_source_stream_) != cudaSuccess) {
            std::terminate();
        }
        layer_source_stream_ = nullptr;
        layer_source_count_ = 0;
        if (record.backing.valid()) { (void)backing_->release(record.backing); }
        record = {};
    }

    void capture(Record& record, DeviceKVPagePool& pool, std::int32_t physical,
                 std::int32_t baked, std::uint32_t columns, cudaStream_t stream) {
        drain_layer_source();
        if (columns > kPagedKVPageSize || columns < record.columns) {
            throw std::logic_error("immutable K capture coverage is invalid");
        }
        if (columns == record.columns) { return; }
        auto bytes = std::span(static_cast<std::byte*>(staging_->data()), bytes_);
        if (record.backing.valid()) { backing_->load(record.backing, bytes); }
        else { std::memset(bytes.data(), 0, bytes.size()); }
        copy_columns(pool, physical, record.columns, columns, false, stream, staging_->data());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Never replace committed columns with the working page's rotated K.
        const auto replacement = backing_->store(bytes);
        if (record.backing.valid()) { (void)backing_->release(record.backing); }
        for (auto t = record.columns; t < columns; ++t) { record.positions[t] = baked + t; }
        record.backing = replacement;
        record.columns = columns;
        ++captures_;
    }

    void restore(const Record& record, DeviceKVPagePool& pool, std::int32_t physical,
                 std::int32_t destination, cudaStream_t stream) {
        drain_layer_source();
        if (!record.backing.valid() || record.columns == 0) {
            throw std::logic_error("immutable K restore has no committed source");
        }
        const auto read_start = std::chrono::steady_clock::now();
        backing_->load(record.backing,
                       {static_cast<std::byte*>(staging_->data()), bytes_});
        if (stage_trace_enabled()) {
            stage_timing_.read_sync_us += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - read_start).count());
        }
        restore_loaded(record, pool, physical, destination, stream, staging_->data());
    }

    // Adapt the existing bounded Cold read-ahead pattern and KVMem/ActKV owner contract:
    // two file reads at most, no GPU work or publication in the worker. Every page still
    // completes its original copy/rebuild/sync before that pinned slot may be refilled.
    void restore_many(std::span<const RestoreRequest> requests, DeviceKVPagePool& pool,
                       cudaStream_t stream) {
        drain_layer_source();
        if (!source_read_ahead_enabled() || requests.size() < 2) {
            for (const auto& request : requests) {
                if (request.record == nullptr) {
                    throw std::logic_error("immutable K restore request has no owner");
                }
                restore(*request.record, pool, request.physical, request.destination, stream);
            }
            return;
        }
        if (event_pipeline_enabled() && stream != nullptr && stream != cudaStreamLegacy) {
            restore_event_pipeline(requests, pool, stream);
            return;
        }
        if (r26_bundle_enabled() && source_pipeline_geometry(pool)) {
            restore_pipeline_batch(requests, pool, stream);
            return;
        }
        if (!read_ahead_staging_) {
            read_ahead_staging_ = std::make_unique<PinnedHostBuffer>(bytes_);
        }
        struct ReadSlot {
            const Record* owner = nullptr;
            Record captured;
            void* data = nullptr;
            // Destroy future first on unwind; the store retains both pinned buffers.
            std::future<void> ready;
        };
        std::array<ReadSlot, 2> slots;
        const auto start_read = [&](std::size_t i) {
            auto& slot = slots[i % slots.size()];
            if (slot.ready.valid() || requests[i].record == nullptr) {
                throw std::logic_error("immutable K read-ahead slot has no free owner");
            }
            slot.owner = requests[i].record;
            slot.captured = *slot.owner;
            if (!slot.captured.backing.valid() || slot.captured.columns == 0) {
                throw std::logic_error("immutable K read-ahead has no committed source");
            }
            slot.data = i % slots.size() == 0 ? staging_->data() : read_ahead_staging_->data();
            slot.ready = backing_->prefetch(slot.captured.backing,
                {static_cast<std::byte*>(slot.data), bytes_});
        };
        start_read(0);
        start_read(1);
        bool gpu_pending = false;
        try {
            for (std::size_t i = 0; i < requests.size(); ++i) {
                auto& slot = slots[i % slots.size()];
                const auto wait_start = std::chrono::steady_clock::now();
                slot.ready.get();
                if (stage_trace_enabled()) {
                    stage_timing_.read_wait_us += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - wait_start).count());
                }
                const auto& source = *requests[i].record;
                if (slot.owner != requests[i].record ||
                    source.backing != slot.captured.backing ||
                    source.columns != slot.captured.columns ||
                    source.positions != slot.captured.positions ||
                    !backing_->valid(slot.captured.backing)) {
                    throw std::logic_error("immutable K read-ahead source identity changed");
                }
                gpu_pending = true;
                restore_loaded(slot.captured, pool, requests[i].physical,
                               requests[i].destination, stream, slot.data);
                gpu_pending = false;
                if (i + slots.size() < requests.size()) { start_read(i + slots.size()); }
            }
        } catch (...) {
            // A partially submitted copy may not have reached restore_loaded's sync.
            // Drain before either pinned slot may be reused or its owner destroyed.
            if (gpu_pending && cudaStreamSynchronize(stream) != cudaSuccess) { std::terminate(); }
            throw;
        }
    }

    // Restricted Main16 producer: at most two independent immutable K pages
    // occupy the store's *existing* pinned slots. The disk reads and identity
    // validation finish on the owner thread. H2D and original K rephase are
    // then enqueued for one attention-plane group at a time by enqueue_layer.
    // Unlike restore_many, no producer stream sync is performed on success.
    void prepare_layer_sources(std::span<const RestoreRequest> requests,
                               DeviceKVPagePool& pool, cudaStream_t producer) {
        drain_layer_source();
        if (requests.empty() || requests.size() > layer_source_slots_.size() ||
            producer == nullptr || producer == cudaStreamLegacy ||
            !source_pipeline_geometry(pool) || pool.geometry().planes.size() != 64) {
            throw std::invalid_argument("source layer pipeline needs one/two Main16 pages");
        }
        if (!read_ahead_staging_) {
            read_ahead_staging_ = std::make_unique<PinnedHostBuffer>(bytes_);
        }
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            if (!request.record || request.physical < 0 ||
                static_cast<std::uint32_t>(request.physical) >= pool.capacity_pages() ||
                request.destination < 0 || !request.record->backing.valid() ||
                request.record->columns == 0 || request.record->columns > kPagedKVPageSize ||
                request.destination > std::numeric_limits<std::int32_t>::max() -
                    static_cast<std::int32_t>(request.record->columns - 1) ||
                !backing_->valid(request.record->backing) ||
                (i == 1 && request.physical == requests[0].physical)) {
                throw std::logic_error("source layer pipeline stale/overlapping target");
            }
        }
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            auto& slot = layer_source_slots_[i];
            slot.owner = request.record;
            slot.captured = *request.record;
            slot.physical = request.physical;
            slot.destination = request.destination;
            slot.data = i == 0 ? staging_->data() : read_ahead_staging_->data();
            backing_->load(slot.captured.backing,
                           {static_cast<std::byte*>(slot.data), bytes_});
            if (slot.owner->backing != slot.captured.backing ||
                slot.owner->columns != slot.captured.columns ||
                slot.owner->positions != slot.captured.positions ||
                !backing_->valid(slot.captured.backing)) {
                throw std::logic_error("source layer pipeline owner changed after read");
            }
            slot.moved = false;
            for (std::uint32_t token = 0; token < slot.captured.columns; ++token) {
                slot.moved |= slot.captured.positions[token] !=
                    slot.destination + static_cast<std::int32_t>(token);
            }
        }
        if (stage_trace_enabled()) {
            stage_timing_.read_sync_us += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count());
        }
        layer_source_stream_ = producer;
        layer_source_count_ = requests.size();
        next_layer_source_ = 0;
    }

    void enqueue_layer_source(std::uint32_t layer, DeviceKVPagePool& pool,
                              cudaStream_t producer) {
        if (!layer_source_stream_ || producer != layer_source_stream_ ||
            layer >= 16 || layer != next_layer_source_ || layer_source_count_ == 0) {
            throw std::logic_error("source layer producer order/owner is invalid");
        }
        std::array<SourcePageRestore, 2> moved{};
        std::size_t moved_count = 0;
        for (std::size_t i = 0; i < layer_source_count_; ++i) {
            auto& slot = layer_source_slots_[i];
            // Exactly the captured column extent, not unwritten tail cells:
            // a later append may write the remainder of a partial page.
            for (const std::size_t plane : {static_cast<std::size_t>(layer) * 4,
                                            static_cast<std::size_t>(layer) * 4 + 2}) {
                const auto& dst = pool.plane(plane);
                const auto& packed = layout_.planes[plane];
                const auto heads = pool.geometry().planes[plane].head_extent;
                const auto row_bytes = packed.head_payload_bytes / kPagedKVPageSize;
                auto* device = static_cast<std::byte*>(dst.data) +
                    static_cast<std::int64_t>(slot.physical) * dst.nb[3];
                auto* source = static_cast<std::byte*>(slot.data) + offsets_[plane];
                CUDA_CHECK(cudaMemcpy2DAsync(
                    device, static_cast<std::size_t>(dst.nb[2]), source,
                    packed.head_payload_bytes,
                    slot.captured.columns * row_bytes, heads,
                    cudaMemcpyHostToDevice, producer));
            }
            if (slot.moved) {
                moved[moved_count++] = SourcePageRestore{
                    slot.physical, slot.captured.positions,
                    slot.destination, slot.captured.columns};
            }
        }
        if (moved_count != 0) {
            kvmem_rebuild_source_k_batch_layer(pool,
                std::span<const SourcePageRestore>(moved.data(), moved_count), layer, producer);
        }
        ++next_layer_source_;
        if (next_layer_source_ == 16) {
            restores_ += layer_source_count_;
            stage_timing_.restore_pages += layer_source_count_;
            for (std::size_t i = 0; i < layer_source_count_; ++i) {
                stage_timing_.moved_pages += static_cast<std::uint64_t>(layer_source_slots_[i].moved);
            }
        }
    }

    void drain_layer_source() {
        if (!layer_source_stream_) { return; }
        CUDA_CHECK(cudaStreamSynchronize(layer_source_stream_));
        layer_source_stream_ = nullptr;
        layer_source_count_ = 0;
        next_layer_source_ = 0;
    }

    [[nodiscard]] std::size_t pending_layer_source_pages() const noexcept {
        return layer_source_count_;
    }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return backing_->occupied_bytes(); }
    [[nodiscard]] std::size_t file_capacity_bytes() const noexcept { return backing_->allocated_bytes(); }
    [[nodiscard]] std::size_t staging_bytes() const noexcept {
        return bytes_ * (1 + static_cast<std::size_t>(read_ahead_staging_ != nullptr) +
                          static_cast<std::size_t>(pipeline_staging_[0] != nullptr) +
                          static_cast<std::size_t>(pipeline_staging_[1] != nullptr));
    }
    [[nodiscard]] std::uint64_t captures() const noexcept { return captures_; }
    [[nodiscard]] std::uint64_t restores() const noexcept { return restores_; }

private:
    struct LayerSourceSlot {
        const Record* owner = nullptr;
        Record captured;
        std::int32_t physical = -1;
        std::int32_t destination = 0;
        void* data = nullptr;
        bool moved = false;
    };
    struct StageTiming {
        std::uint64_t restore_pages = 0;
        std::uint64_t moved_pages = 0;
        std::uint64_t read_sync_us = 0;
        std::uint64_t read_wait_us = 0;
        std::uint64_t gpu_timed_pages = 0;
        double h2d_gpu_ms = 0;
        double rebuild_gpu_ms = 0;
    };
    struct TimedEvent {
        cudaEvent_t value = nullptr;
        TimedEvent() { CUDA_CHECK(cudaEventCreate(&value)); }
        ~TimedEvent() { if (value) { (void)cudaEventDestroy(value); } }
        TimedEvent(const TimedEvent&) = delete;
        TimedEvent& operator=(const TimedEvent&) = delete;
    };
    struct TimedStages {
        TimedEvent before_copy;
        TimedEvent after_copy;
        TimedEvent after_rebuild;
    };
    static bool stage_trace_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_KVMEM_STAGE_TRACE");
            if (!value || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_KVMEM_STAGE_TRACE must be 0 or 1");
        }();
        return enabled;
    }
    struct CopyBatchDescriptors {
        std::array<void*, 32> destinations{};
        std::array<const void*, 32> sources{};
        std::array<std::size_t, 32> sizes{};
    };
    struct PipelineEvent {
        cudaEvent_t value = nullptr;
        PipelineEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
        ~PipelineEvent() { if (value) { (void)cudaEventDestroy(value); } }
        PipelineEvent(const PipelineEvent&) = delete;
        PipelineEvent& operator=(const PipelineEvent&) = delete;
    };
    struct EventPipeline {
        cudaStream_t copy = nullptr;
        PipelineEvent start;
        std::array<PipelineEvent, 2> ready;
        EventPipeline() { CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking)); }
        ~EventPipeline() { if (copy) { (void)cudaStreamDestroy(copy); } }
    };
    static bool event_pipeline_enabled() {
        static const bool value = [] {
            const char* flag = std::getenv("NINFER_KVMEM_SOURCE_EVENT_PIPELINE");
            if (!flag || std::strcmp(flag, "0") == 0) { return false; }
            if (std::strcmp(flag, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_KVMEM_SOURCE_EVENT_PIPELINE must be 0 or 1");
        }();
        return value;
    }

    // Sync/Async ledger46 uses distinct submitted/retired ownership. Adapted here
    // to two source-K slots: file ready -> copy ready -> compute wait. KVMem/ActKV
    // immutable owner checks remain authoritative. No requantization changes.
    void restore_event_pipeline(std::span<const RestoreRequest> requests, DeviceKVPagePool& pool,
                                cudaStream_t stream) {
        std::vector<std::int32_t> targets;
        targets.reserve(requests.size());
        for (const auto& request : requests) {
            if (!request.record || request.physical < 0 ||
                static_cast<std::uint32_t>(request.physical) >= pool.capacity_pages() ||
                !request.record->backing.valid() || request.record->columns == 0 ||
                request.record->columns > kPagedKVPageSize || request.destination < 0 ||
                request.destination > std::numeric_limits<std::int32_t>::max() -
                    static_cast<std::int32_t>(request.record->columns - 1) ||
                !backing_->valid(request.record->backing)) {
                throw std::logic_error("source-K event pipeline has invalid owner or extent");
            }
            targets.push_back(request.physical);
        }
        std::sort(targets.begin(), targets.end());
        if (std::adjacent_find(targets.begin(), targets.end()) != targets.end()) {
            throw std::logic_error("source-K event pipeline requires disjoint device targets");
        }
        if (!event_pipeline_) { event_pipeline_ = std::make_unique<EventPipeline>(); }
        if (!read_ahead_staging_) { read_ahead_staging_ = std::make_unique<PinnedHostBuffer>(bytes_); }
        auto& events = *event_pipeline_;
        struct Slot {
            Record captured;
            const Record* owner = nullptr;
            void* data = nullptr;
            CopyBatchDescriptors descriptors;
            bool copied = false;
            std::future<void> read;
        };
        std::array<Slot, 2> slots;
        slots[0].data = staging_->data();
        slots[1].data = read_ahead_staging_->data();
        // Retire every earlier compute access before any transfer overwrites a target.
        CUDA_CHECK(cudaEventRecord(events.start.value, stream));
        CUDA_CHECK(cudaStreamWaitEvent(events.copy, events.start.value, 0));
        const auto start = [&](std::size_t i) {
            auto& slot = slots[i % 2];
            if (slot.copied) {
                // Source bytes/descriptors can be reused after H2D, without waiting
                // for the independent rebuild kernel on the compute stream.
                CUDA_CHECK(cudaEventSynchronize(events.ready[i % 2].value));
                slot.copied = false;
            }
            slot.owner = requests[i].record;
            slot.captured = *slot.owner;
            slot.read = backing_->prefetch(slot.captured.backing,
                {static_cast<std::byte*>(slot.data), bytes_});
        };
        std::uint64_t completed = 0;
        try {
            start(0); start(1);
            for (std::size_t i = 0; i < requests.size(); ++i) {
                auto& slot = slots[i % 2];
                slot.read.get();
                if (slot.owner != requests[i].record ||
                    slot.owner->backing != slot.captured.backing ||
                    slot.owner->columns != slot.captured.columns ||
                    slot.owner->positions != slot.captured.positions ||
                    !backing_->valid(slot.captured.backing)) {
                    throw std::logic_error("source-K event pipeline owner changed");
                }
                copy_columns(pool, requests[i].physical, 0, slot.captured.columns, true,
                             events.copy, slot.data, &slot.descriptors);
                CUDA_CHECK(cudaEventRecord(events.ready[i % 2].value, events.copy));
                slot.copied = true;
                // Each wait binds the already recorded event, even if that slot's
                // event is re-recorded for a later page after its source retires.
                CUDA_CHECK(cudaStreamWaitEvent(stream, events.ready[i % 2].value, 0));
                bool moved = false;
                for (std::uint32_t t = 0; t < slot.captured.columns; ++t) {
                    moved |= slot.captured.positions[t] != requests[i].destination + static_cast<std::int32_t>(t);
                }
                if (moved) {
                    kvmem_rebuild_source_k(pool, requests[i].physical, slot.captured.positions,
                                          requests[i].destination, slot.captured.columns, stream);
                }
                ++completed;
                if (i + 2 < requests.size()) { start(i + 2); }
            }
            // Completion/publication stays at the existing caller boundary. This
            // does not claim per-layer forward consumption, which needs its own hook.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            restores_ += completed;
            event_pipeline_pages_ += completed;
        } catch (...) {
            // A failed later enqueue cannot be covered by an earlier ready event.
            // Drain both GPU streams before local slot futures or buffers retire.
            if (cudaStreamSynchronize(events.copy) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess) { std::terminate(); }
            throw;
        }
    }

    struct PipelineStats {
        std::uint64_t read_submits = 0, groups_started = 0, groups_completed = 0;
        std::uint64_t pages_completed = 0, moved_pages = 0, rebuild_calls = 0;
        std::uint64_t stream_sync_calls = 0, error_drains = 0;
    };
    struct PipelineReadSlot {
        const Record* owner = nullptr;
        Record captured;
        void* data = nullptr;
        CopyBatchDescriptors copies;
        std::future<void> ready;
    };

    static bool r26_bundle_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_R26_BUNDLE");
            if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_R26_BUNDLE must be 0 or 1");
        }();
        return enabled;
    }

    static bool r26_bundle_trace() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_R26_BUNDLE_TRACE");
            if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_R26_BUNDLE_TRACE must be 0 or 1");
        }();
        return enabled;
    }

    static bool source_pipeline_geometry(const DeviceKVPagePool& pool) {
        const auto& geometry = pool.geometry();
        if (geometry.page_tokens != kPagedKVPageSize ||
            geometry.device_plane_order != PagedKVPlaneOrder::PageMajor ||
            (geometry.planes.size() != 64 && geometry.planes.size() != 4)) { return false; }
#if !defined(NINFER_SM89)
        return false;
#endif
        for (std::size_t i = 0; i < geometry.planes.size(); ++i) {
            const auto& plane = geometry.planes[i];
            const bool scale = i % 4 >= 2;
            if (plane.head_extent != 4 || plane.leading_extent != (scale ? 4 : 256) ||
                plane.dtype != (scale ? DType::FP16 : DType::I8)) { return false; }
        }
        return true;
    }

    // packaged_task futures do NOT join on destruction. Explicitly consume every
    // valid future on success, initial-submit failure, I/O failure and GPU failure.
    [[nodiscard]] std::exception_ptr drain_pipeline_reads() noexcept {
        std::exception_ptr first;
        for (auto& slot : pipeline_slots_) {
            if (slot.ready.valid()) {
                try { slot.ready.get(); }
                catch (...) { if (!first) { first = std::current_exception(); } }
            }
            slot.owner = nullptr;
            slot.captured = {};
        }
        return first;
    }

    void restore_pipeline_batch(std::span<const RestoreRequest> requests, DeviceKVPagePool& pool,
                                 cudaStream_t stream) {
        // Disjoint physical targets are required both within and across groups:
        // reordering independent reads cannot be used to define aliasing writes.
        std::vector<std::int32_t> targets;
        targets.reserve(requests.size());
        for (const auto& request : requests) {
            if (request.record == nullptr || request.physical < 0 ||
                static_cast<std::uint32_t>(request.physical) >= pool.capacity_pages() ||
                !request.record->backing.valid() || request.record->columns == 0 ||
                request.record->columns > kPagedKVPageSize || request.destination < 0 ||
                request.destination > std::numeric_limits<std::int32_t>::max() -
                    static_cast<std::int32_t>(request.record->columns - 1) ||
                !backing_->valid(request.record->backing)) {
                throw std::logic_error("immutable K pipeline request identity/extent is invalid");
            }
            targets.push_back(request.physical);
        }
        std::sort(targets.begin(), targets.end());
        if (std::adjacent_find(targets.begin(), targets.end()) != targets.end()) {
            throw std::logic_error("immutable K pipeline physical destinations overlap");
        }
        if (!read_ahead_staging_) {
            read_ahead_staging_ = std::make_unique<PinnedHostBuffer>(bytes_);
        }
        for (auto& buffer : pipeline_staging_) {
            if (!buffer) { buffer = std::make_unique<PinnedHostBuffer>(bytes_); }
        }
        if (!source_workers_) { source_workers_ = std::make_unique<HostWorkerPool>(2, 2); }
        const std::array<void*, 4> buffers{staging_->data(), read_ahead_staging_->data(),
                                         pipeline_staging_[0]->data(), pipeline_staging_[1]->data()};
        for (std::size_t i = 0; i < pipeline_slots_.size(); ++i) {
            pipeline_slots_[i].data = buffers[i];
        }
        if (r26_bundle_trace() && !pipeline_trace_emitted_) {
            const auto workers = source_workers_->snapshot();
            std::fprintf(stderr,
                "R26_SOURCE_PIPELINE threads=%u queue_capacity=2 slots=4 staging_bytes=%zu "
                "group_pages=2 requests=%zu\n", workers.threads, staging_bytes(), requests.size());
            pipeline_trace_emitted_ = true;
        }
        const auto start_read = [&](std::size_t i) {
            auto& slot = pipeline_slots_[i % pipeline_slots_.size()];
            if (slot.ready.valid()) {
                throw std::logic_error("immutable K pipeline slot is still pending");
            }
            slot.owner = requests[i].record;
            slot.captured = *slot.owner;
            if (!backing_->valid(slot.captured.backing)) {
                throw std::logic_error("immutable K pipeline source became stale before submission");
            }
            // Reuse native HostWorkerPool; like ActKV's bounded I/O queue, jobs
            // hold immutable extents/buffers and never invoke CUDA or publish KV.
            slot.ready = source_workers_->submit(
                [backing = backing_.get(), handle = slot.captured.backing,
                 data = slot.data, bytes = bytes_] {
                    backing->load(handle, {static_cast<std::byte*>(data), bytes});
                });
            ++pipeline_stats_.read_submits;
        };
        bool gpu_pending = false;
        try {
            // Two two-page groups: active<=2, queued<=2, at most four futures.
            for (std::size_t i = 0; i < std::min(requests.size(), pipeline_slots_.size()); ++i) {
                start_read(i);
            }
            for (std::size_t begin = 0; begin < requests.size(); begin += 2) {
                const auto count = std::min<std::size_t>(2, requests.size() - begin);
                for (std::size_t j = 0; j < count; ++j) {
                    auto& slot = pipeline_slots_[(begin + j) % pipeline_slots_.size()];
                    slot.ready.get();
                    const auto& source = *requests[begin + j].record;
                    if (slot.owner != requests[begin + j].record ||
                        source.backing != slot.captured.backing ||
                        source.columns != slot.captured.columns ||
                        source.positions != slot.captured.positions ||
                        !backing_->valid(slot.captured.backing)) {
                        throw std::logic_error("immutable K read-ahead source identity changed");
                    }
                }
                std::array<SourcePageRestore, 2> rebuilds;
                std::size_t rebuild_count = 0;
                gpu_pending = true;
                ++pipeline_stats_.groups_started;
                for (std::size_t j = 0; j < count; ++j) {
                    auto& slot = pipeline_slots_[(begin + j) % pipeline_slots_.size()];
                    const auto& request = requests[begin + j];
                    copy_columns(pool, request.physical, 0, slot.captured.columns, true,
                                 stream, slot.data, &slot.copies);
                    bool moved = false;
                    for (std::uint32_t t = 0; t < slot.captured.columns; ++t) {
                        moved = moved || slot.captured.positions[t] !=
                            request.destination + static_cast<std::int32_t>(t);
                    }
                    if (moved) {
                        rebuilds[rebuild_count++] = SourcePageRestore{
                            request.physical, slot.captured.positions,
                            request.destination, slot.captured.columns};
                    }
                }
                if (rebuild_count != 0) {
                    pipeline_stats_.moved_pages += rebuild_count;
                    ++pipeline_stats_.rebuild_calls;
                    kvmem_rebuild_source_k_batch(pool,
                        std::span<const SourcePageRestore>(rebuilds.data(), rebuild_count), stream);
                }
                ++pipeline_stats_.stream_sync_calls;
                CUDA_CHECK(cudaStreamSynchronize(stream));
                gpu_pending = false;
                ++pipeline_stats_.groups_completed;
                pipeline_stats_.pages_completed += count;
                restores_ += count;
                // The other group is reading while this one copies/rebuilds.
                // Refill this group's pin only after all GPU readers completed.
                for (std::size_t j = 0; j < count; ++j) {
                    const auto next = begin + j + pipeline_slots_.size();
                    if (next < requests.size()) { start_read(next); }
                }
            }
            if (const auto error = drain_pipeline_reads()) { std::rethrow_exception(error); }
        } catch (...) {
            const auto original = std::current_exception();
            ++pipeline_stats_.error_drains;
            const auto gpu_error = gpu_pending ? cudaStreamSynchronize(stream) : cudaSuccess;
            (void)drain_pipeline_reads();
            if (gpu_error != cudaSuccess) { std::terminate(); }
            std::rethrow_exception(original);
        }
    }

    static bool source_read_ahead_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_KVMEM_SOURCE_READ_AHEAD");
            if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_KVMEM_SOURCE_READ_AHEAD must be 0 or 1");
        }();
        return enabled;
    }

    void restore_loaded(const Record& record, DeviceKVPagePool& pool, std::int32_t physical,
                         std::int32_t destination, cudaStream_t stream, void* source) {
        TimedStages* events = nullptr;
        if (stage_trace_enabled()) {
            if (!stage_events_) { stage_events_ = std::make_unique<TimedStages>(); }
            events = stage_events_.get();
            CUDA_CHECK(cudaEventRecord(events->before_copy.value, stream));
        }
        copy_columns(pool, physical, 0, record.columns, true, stream, source);
        if (events) { CUDA_CHECK(cudaEventRecord(events->after_copy.value, stream)); }
        bool moved = false;
        for (std::uint32_t t = 0; t < record.columns; ++t) {
            moved = moved || record.positions[t] != destination + static_cast<std::int32_t>(t);
        }
        if (moved) {
            kvmem_rebuild_source_k(pool, physical, record.positions, destination,
                                   record.columns, stream);
        }
        if (events) { CUDA_CHECK(cudaEventRecord(events->after_rebuild.value, stream)); }
        // The selected pinned staging page is reusable only after this sync.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (events) {
            float copy_ms = 0, rebuild_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&copy_ms, events->before_copy.value,
                                            events->after_copy.value));
            CUDA_CHECK(cudaEventElapsedTime(&rebuild_ms, events->after_copy.value,
                                            events->after_rebuild.value));
            stage_timing_.h2d_gpu_ms += copy_ms;
            if (moved) { stage_timing_.rebuild_gpu_ms += rebuild_ms; }
            ++stage_timing_.gpu_timed_pages;
        }
        ++stage_timing_.restore_pages;
        stage_timing_.moved_pages += static_cast<std::uint64_t>(moved);
        ++restores_;
    }

    static bool source_copy_batch_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_KVMEM_SOURCE_COPY_BATCH");
            if (value == nullptr || std::strcmp(value, "1") == 0) { return true; }
            if (std::strcmp(value, "0") == 0) { return false; }
            throw std::invalid_argument("NINFER_KVMEM_SOURCE_COPY_BATCH must be 0 or 1");
        }();
        return enabled;
    }

    bool try_copy_columns_batch(DeviceKVPagePool& pool, std::int32_t physical,
                                 std::uint32_t begin, std::uint32_t end,
                                 bool to_device, cudaStream_t stream, void* host_data,
                                 CopyBatchDescriptors* batch = nullptr) {
#if CUDART_VERSION >= 13000 && defined(NINFER_SM89)
        if (!copy_batch_geometry_ || !to_device || begin != 0 || end != kPagedKVPageSize ||
            stream == nullptr || stream == cudaStreamLegacy ||
            pool.geometry().device_plane_order != PagedKVPlaneOrder::PageMajor ||
            pool.plane_count() != 64 || !source_copy_batch_enabled()) {
            return false;
        }
        auto& descriptors = batch != nullptr ? *batch : copy_batch_;
        for (std::size_t j = 0; j < descriptors.sizes.size(); ++j) {
            const std::size_t i = 2 * j;
            const auto& plane = pool.plane(i);
            const auto& packed = layout_.planes[i];
            // A partial head or padded head cannot be flattened into one copy.
            if (plane.nb[2] != static_cast<std::int64_t>(packed.head_payload_bytes) ||
                plane.nb[3] < static_cast<std::int64_t>(packed.page_payload_bytes)) {
                return false;
            }
            descriptors.destinations[j] = static_cast<std::byte*>(plane.data) +
                                    static_cast<std::int64_t>(physical) * plane.nb[3];
            descriptors.sources[j] = static_cast<std::byte*>(host_data) + offsets_[i];
            descriptors.sizes[j] = packed.page_payload_bytes;
        }
        // Adapted from Infernix a3edb450 src/core/copy_batch.cpp (Apache-2.0),
        // via Don-Chad/ninfer-3090 d8c29253f2c0845611a794d87057950107a4956b.
        // Modified: fixed 32 independent H2D ranges, CUDA13 signature, no queue
        // or silent error fallback. Any CUDA error propagates through CUDA_CHECK.
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        std::size_t first = 0;
        CUDA_CHECK(cudaMemcpyBatchAsync(descriptors.destinations.data(), descriptors.sources.data(),
                                        descriptors.sizes.data(), descriptors.sizes.size(), &attributes,
                                        &first, 1, stream));
        return true;
#else
        return false;
#endif
    }

    void copy_columns(DeviceKVPagePool& pool, std::int32_t physical, std::uint32_t begin,
                      std::uint32_t end, bool to_device, cudaStream_t stream, void* host_data,
                      CopyBatchDescriptors* batch = nullptr) {
        if (try_copy_columns_batch(pool, physical, begin, end, to_device, stream, host_data, batch)) {
            return;
        }
        const bool page_major = pool.geometry().device_plane_order == PagedKVPlaneOrder::PageMajor;
        for (std::size_t i = 0; i < offsets_.size(); i += 2) {
            const auto& plane = pool.plane(i);
            const auto& packed = layout_.planes[i];
            const auto heads = pool.geometry().planes[i].head_extent;
            const auto row_bytes = packed.head_payload_bytes / kPagedKVPageSize;
            auto* device = static_cast<std::byte*>(plane.data) +
                           physical * plane.nb[page_major ? 3 : 2] + begin * row_bytes;
            auto* host = static_cast<std::byte*>(host_data) + offsets_[i] + begin * row_bytes;
            const auto head_stride = static_cast<std::size_t>(plane.nb[page_major ? 2 : 3]);
            if (to_device) {
                CUDA_CHECK(cudaMemcpy2DAsync(device, head_stride, host, packed.head_payload_bytes,
                                             (end - begin) * row_bytes, heads,
                                             cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(host, packed.head_payload_bytes, device, head_stride,
                                             (end - begin) * row_bytes, heads,
                                             cudaMemcpyDeviceToHost, stream));
            }
        }
    }

    HostKVPageLayout layout_;
    std::vector<std::size_t> offsets_;
    std::size_t bytes_ = 0;
    std::unique_ptr<PinnedHostBuffer> staging_;
    std::unique_ptr<EventPipeline> event_pipeline_;
    std::uint64_t event_pipeline_pages_ = 0;
    // Optional second page is fixed-size, store-owned and included in staging_bytes().
    std::unique_ptr<PinnedHostBuffer> read_ahead_staging_;
    std::unique_ptr<ColdKVStore> backing_;
    // Two reused pinned buffers are held until the producer stream finishes.
    // Their record identities are captured before scheduling any DMA and no
    // buffer is refilled by capture/restore until drain_layer_source().
    std::array<LayerSourceSlot, 2> layer_source_slots_{};
    cudaStream_t layer_source_stream_ = nullptr;
    std::size_t layer_source_count_ = 0;
    std::uint32_t next_layer_source_ = 0;
    // Host descriptors live with the original single staging owner, including
    // through restore's existing stream synchronization. Payload size is unchanged.
    bool copy_batch_geometry_ = false;
    CopyBatchDescriptors copy_batch_;
    std::uint64_t captures_ = 0, restores_ = 0;
    StageTiming stage_timing_;
    std::unique_ptr<TimedStages> stage_events_;
    // Four total pinned pages: two groups of two. No pin is refilled before group sync.
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> pipeline_staging_;
    std::array<PipelineReadSlot, 4> pipeline_slots_;
    bool pipeline_trace_emitted_ = false;
    PipelineStats pipeline_stats_;
    // Declared last: workers also join before backing/pin if construction unwinds.
    std::unique_ptr<HostWorkerPool> source_workers_;
};
} // namespace ninfer::kvmem
