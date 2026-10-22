#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_blocks.h"
#include "kvmem/kvmem_bridge.h"
#include "kvmem/kvmem_cold_store.h"
#include "kvmem/kvmem_layer_ready.h"
#include "kvmem/kvmem_source_k.h"
#include "kvmem/kvmem_retrieval.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cerrno>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class LogicalKVPageStore;
class KVAddressSpaceStore;
class HostKVExtentStore;
class KVPrefixForkReservation;
class KVActiveSnapshotReservation;

class HostKVExtentCapability {
public:
    HostKVExtentCapability() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVExtentCapability,
                                         HostKVExtentCapability) noexcept = default;

private:
    HostKVExtentCapability(const HostKVExtentStore* owner, std::uint32_t index,
                           std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const HostKVExtentStore* owner_ = nullptr;
    std::uint32_t index_            = 0;
    std::uint32_t generation_       = 0;

    friend class HostKVExtentStore;
};

struct HostKVPageReplica {
    HostKVExtentCapability extent;
    std::uint32_t page_offset       = 0;
    std::uint32_t membership_node   = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
};

class LogicalKVPageHandle {
public:
    LogicalKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(LogicalKVPageHandle,
                                         LogicalKVPageHandle) noexcept = default;

private:
    LogicalKVPageHandle(const LogicalKVPageStore* owner, std::uint32_t index,
                        std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const LogicalKVPageStore* owner_ = nullptr;
    std::uint32_t index_             = 0;
    std::uint32_t generation_        = 0;

    friend class LogicalKVPageStore;
    friend class HostKVExtentStore;
};

class KVAddressSpaceHandle {
public:
    KVAddressSpaceHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(KVAddressSpaceHandle,
                                         KVAddressSpaceHandle) noexcept = default;

private:
    KVAddressSpaceHandle(const KVAddressSpaceStore* owner, std::uint32_t index,
                         std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const KVAddressSpaceStore* owner_ = nullptr;
    std::uint32_t index_              = 0;
    std::uint32_t generation_         = 0;

    friend class KVAddressSpaceStore;
};

class KVActivationReservation {
public:
    KVActivationReservation() noexcept = default;

    KVActivationReservation(KVActivationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), address_(other.address_),
          requested_entitlement_(other.requested_entitlement_),
          device_claim_pages_(other.device_claim_pages_),
          activation_frontier_(other.activation_frontier_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t requested_entitlement, std::uint32_t device_claim_pages,
                            std::optional<std::uint32_t> activation_frontier,
                            DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), requested_entitlement_(requested_entitlement),
          device_claim_pages_(device_claim_pages), activation_frontier_(activation_frontier),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t requested_entitlement_ = 0;
    // Device pages this activation may reserve from the physical pool. Equal to the
    // requested entitlement unless the pool sits below it, where the host tier carries
    // the difference.
    std::uint32_t device_claim_pages_ = 0;
    std::optional<std::uint32_t> activation_frontier_;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;

    friend class KVAddressSpaceStore;
};

class KVPrefixForkReservation {
public:
    KVPrefixForkReservation() noexcept = default;
    ~KVPrefixForkReservation();

    KVPrefixForkReservation(KVPrefixForkReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_),
          full_pages_(other.full_pages_), tail_columns_(other.tail_columns_),
          requested_entitlement_(other.requested_entitlement_),
          device_claim_(other.device_claim_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)),
          tail_destination_(other.tail_destination_),
          staged_tail_release_(other.staged_tail_release_),
          tail_source_settled_(other.tail_source_settled_) {
        other.tail_destination_.reset();
    }

    KVPrefixForkReservation& operator=(KVPrefixForkReservation&&)      = delete;
    KVPrefixForkReservation(const KVPrefixForkReservation&)            = delete;
    KVPrefixForkReservation& operator=(const KVPrefixForkReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_columns_ != 0; }

private:
    KVPrefixForkReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                            KVAddressSpaceHandle destination, std::uint32_t frontier,
                            std::uint32_t full_pages, std::uint32_t tail_columns,
                            std::uint32_t requested_entitlement, std::uint32_t device_claim,
                            DeviceKVPageReservation&& page_reservation, KVExecutionRowLease&& row,
                            std::optional<LogicalKVPageHandle> tail_destination,
                            bool staged_tail_release) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          full_pages_(full_pages), tail_columns_(tail_columns),
          requested_entitlement_(requested_entitlement), device_claim_(device_claim),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)),
          tail_destination_(tail_destination), staged_tail_release_(staged_tail_release) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_              = 0;
    std::uint32_t full_pages_            = 0;
    std::uint32_t tail_columns_          = 0;
    std::uint32_t requested_entitlement_ = 0;
    // A host-backed address space may hold a logical entitlement larger than the device page pool.
    // The fork only reserves pages inside that device claim; the rest of the prefix stays Host-backed.
    std::uint32_t device_claim_        = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;
    std::optional<LogicalKVPageHandle> tail_destination_;
    bool staged_tail_release_ = false;
    bool tail_source_settled_ = false;

    friend class KVAddressSpaceStore;
};

struct KVActiveSnapshotShape {
    std::uint32_t full_pages           = 0;
    std::uint32_t immutable_full_pages = 0;
    std::uint32_t unique_full_pages    = 0;
    std::uint32_t tail_columns         = 0;

    [[nodiscard]] std::uint32_t copied_pages() const noexcept {
        return tail_columns == 0 ? 0U : 1U;
    }

    [[nodiscard]] friend bool operator==(KVActiveSnapshotShape,
                                         KVActiveSnapshotShape) noexcept = default;
};

// Transaction-exclusive conversion of one active address into an immutable checkpoint while
// installing a new active writer address. An active address may begin with aliased immutable full
// pages and end with unique mutable pages. Full pages are referenced in place; only a non-aligned
// mutable tail receives a new physical page.
class KVActiveSnapshotReservation {
public:
    KVActiveSnapshotReservation() noexcept = default;
    ~KVActiveSnapshotReservation();

    KVActiveSnapshotReservation(KVActiveSnapshotReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_), shape_(other.shape_),
          entitlement_(other.entitlement_), tail_reservation_(std::move(other.tail_reservation_)),
          tail_destination_(other.tail_destination_) {
        other.tail_destination_.reset();
    }

    KVActiveSnapshotReservation& operator=(KVActiveSnapshotReservation&&)      = delete;
    KVActiveSnapshotReservation(const KVActiveSnapshotReservation&)            = delete;
    KVActiveSnapshotReservation& operator=(const KVActiveSnapshotReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return shape_.copied_pages() != 0; }

private:
    KVActiveSnapshotReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                                KVAddressSpaceHandle destination, std::uint32_t frontier,
                                KVActiveSnapshotShape shape, std::uint32_t entitlement,
                                DeviceKVPageReservation&& tail_reservation,
                                std::optional<LogicalKVPageHandle> tail_destination) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          shape_(shape), entitlement_(entitlement), tail_reservation_(std::move(tail_reservation)),
          tail_destination_(tail_destination) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_ = 0;
    KVActiveSnapshotShape shape_;
    std::uint32_t entitlement_ = 0;
    DeviceKVPageReservation tail_reservation_;
    std::optional<LogicalKVPageHandle> tail_destination_;

    friend class KVAddressSpaceStore;
};

class LogicalKVPageStore {
public:
    LogicalKVPageStore(DeviceKVPagePool& physical, std::uint32_t logical_capacity)
        : physical_(&physical), pages_(logical_capacity), free_(pages_.size()),
          free_count_(static_cast<std::uint32_t>(pages_.size())) {
        if (pages_.empty() || logical_capacity < physical.capacity_pages()) {
            throw std::invalid_argument("Logical KV page capacity is inconsistent");
        }
        for (std::uint32_t index = 0; index < pages_.size(); ++index) {
            free_[index] = static_cast<std::uint32_t>(pages_.size()) - 1U - index;
        }
        materialization_scratch_.reserve(physical.capacity_pages());
    }

    LogicalKVPageStore(const LogicalKVPageStore&)            = delete;
    LogicalKVPageStore& operator=(const LogicalKVPageStore&) = delete;
    LogicalKVPageStore(LogicalKVPageStore&&)                 = delete;
    LogicalKVPageStore& operator=(LogicalKVPageStore&&)      = delete;

    [[nodiscard]] DeviceKVPagePool& physical_pool() noexcept { return *physical_; }

    [[nodiscard]] const DeviceKVPagePool& physical_pool() const noexcept { return *physical_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(pages_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] LogicalKVPageHandle materialize(DeviceKVPageReservation& reservation) {
        if (free_count_ == 0) {
            throw std::logic_error("logical KV descriptors exhausted before physical capacity");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch     = next_epoch(page.content_epoch);
        page.committed_columns = 0;
        page.references        = 1;
        page.active_references = 0;
        page.writer_references = 1;
        page.protected_columns = 0;
        page.occupied          = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void materialize(DeviceKVPageReservation& reservation,
                     std::span<LogicalKVPageHandle> destinations,
                     std::optional<LogicalKVPageHandle> preferred_predecessor = std::nullopt) {
        if (destinations.empty()) { return; }
        if (destinations.size() > free_count_ ||
            destinations.size() > materialization_scratch_.capacity()) {
            throw std::logic_error("logical KV batch materialization exceeds descriptor capacity");
        }
        for (const LogicalKVPageHandle destination : destinations) {
            if (destination.valid()) {
                throw std::logic_error("logical KV batch destination is already populated");
            }
        }
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            if (pages_[index].occupied) {
                throw std::logic_error("logical KV free descriptor is occupied");
            }
        }

        std::optional<DeviceKVPageHandle> physical_predecessor;
        if (preferred_predecessor) { physical_predecessor = physical(*preferred_predecessor); }
        materialization_scratch_.clear();
        physical_->materialize(reservation, static_cast<std::uint32_t>(destinations.size()),
                               materialization_scratch_, physical_predecessor);
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            Page& page                = pages_[index];
            page.device_replica.emplace(std::move(materialization_scratch_[offset]));
            page.content_epoch     = next_epoch(page.content_epoch);
            page.committed_columns = 0;
            page.references        = 1;
            page.active_references = 0;
            page.writer_references = 1;
            page.protected_columns = 0;
            page.occupied          = true;
            destinations[offset]   = LogicalKVPageHandle(this, index, page.generation);
        }
        free_count_ -= static_cast<std::uint32_t>(destinations.size());
        materialization_scratch_.clear();
    }

    [[nodiscard]] LogicalKVPageHandle
    materialize_transfer_destination(DeviceKVPageReservation& reservation,
                                     std::uint32_t committed_columns) {
        if (committed_columns == 0 ||
            committed_columns > static_cast<std::uint32_t>(kPagedKVPageSize) || free_count_ == 0) {
            throw std::invalid_argument("logical KV transfer destination is invalid");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch      = next_epoch(page.content_epoch);
        page.committed_columns  = committed_columns;
        page.references         = 0;
        page.active_references  = 0;
        page.writer_references  = 0;
        page.protected_columns  = 0;
        page.destination_pinned = true;
        page.occupied           = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void publish_transfer_destination(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            std::terminate();
        }
        page.references         = 1;
        page.writer_references  = writer ? 1U : 0U;
        page.destination_pinned = false;
    }

    void abort_transfer_destination(LogicalKVPageHandle handle,
                                    DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            return;
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool valid(LogicalKVPageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < pages_.size() &&
               pages_[handle.index_].occupied &&
               pages_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(LogicalKVPageHandle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] DeviceKVPageHandle physical(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.device_replica) {
            throw std::logic_error("logical KV page has no Device replica");
        }
        return page.device_replica->handle();
    }

    [[nodiscard]] bool device_resident(LogicalKVPageHandle handle) const {
        return require(handle).device_replica.has_value();
    }

    [[nodiscard]] bool host_resident(LogicalKVPageHandle handle) const {
        return require(handle).host_replica.has_value();
    }

    [[nodiscard]] const HostKVPageReplica& host_replica(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.host_replica) { throw std::logic_error("logical KV page has no Host replica"); }
        return *page.host_replica;
    }

    [[nodiscard]] std::uint64_t content_epoch(LogicalKVPageHandle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t committed_columns(LogicalKVPageHandle handle) const {
        return require(handle).committed_columns;
    }

    [[nodiscard]] std::uint32_t address_references(LogicalKVPageHandle handle) const {
        return require(handle).references;
    }

    [[nodiscard]] std::uint32_t active_address_references(LogicalKVPageHandle handle) const {
        return require(handle).active_references;
    }

    [[nodiscard]] std::uint8_t writer_references(LogicalKVPageHandle handle) const {
        return require(handle).writer_references;
    }

    [[nodiscard]] std::uint32_t protected_columns(LogicalKVPageHandle handle) const {
        return require(handle).protected_columns;
    }

    [[nodiscard]] std::uint32_t source_pins(LogicalKVPageHandle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] bool can_pin_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 0 &&
               !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] bool can_pin_active_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 1 &&
               page.references == 1 && !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] DeviceKVPageHandle reserve_device_replica(LogicalKVPageHandle handle,
                                                            DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.device_replica || page.pending_device_replica || !page.host_replica ||
            page.destination_pinned || page.source_pins != 0 ||
            page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV Device restore is not reservable");
        }
        page.pending_device_replica.emplace(physical_->materialize_one(reservation));
        page.destination_pinned = true;
        return page.pending_device_replica->handle();
    }

    void publish_device_replica(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.device_replica || !page.pending_device_replica || !page.destination_pinned ||
            !page.host_replica || page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV Device restore is not publishable");
        }
        page.device_replica = std::move(page.pending_device_replica);
        page.pending_device_replica.reset();
        page.destination_pinned = false;
    }

    void abort_device_replica(LogicalKVPageHandle handle,
                              DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.pending_device_replica || !page.destination_pinned) { return; }
        physical_->dematerialize_one(reservation, std::move(*page.pending_device_replica));
        page.pending_device_replica.reset();
        page.destination_pinned = false;
    }

    [[nodiscard]] bool drop_device_replica(LogicalKVPageHandle handle) noexcept {
        if (!can_drop_device_replica(handle)) { return false; }
        Page& page = pages_[handle.index_];
        page.device_replica.reset();
        return true;
    }

    [[nodiscard]] bool host_replica_current(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
               page.host_replica->committed_columns == page.committed_columns;
    }

    [[nodiscard]] bool can_drop_device_replica(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica && host_replica_current(handle) &&
               !page.pending_device_replica && page.writer_references == 0 &&
               page.active_references == 0 && page.source_pins == 0 && !page.destination_pinned;
    }

    void retain_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == std::numeric_limits<std::uint32_t>::max() ||
            page.active_references >= page.references) {
            throw std::logic_error("logical KV active reference is not retainable");
        }
        ++page.active_references;
    }

    void release_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == 0) {
            throw std::logic_error("logical KV page has no active reference");
        }
        --page.active_references;
    }

    void retain_reference(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (page.references == std::numeric_limits<std::uint32_t>::max() ||
            (writer && page.writer_references != 0)) {
            throw std::logic_error("logical KV page reference is not retainable");
        }
        ++page.references;
        if (writer) { page.writer_references = 1; }
    }

    [[nodiscard]] bool can_retain_reference(LogicalKVPageHandle handle,
                                            bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != std::numeric_limits<std::uint32_t>::max() &&
               (!writer || (page.writer_references == 0 && page.references == 0)) &&
               !page.destination_pinned;
    }

    void set_writer(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (writer && page.writer_references != 0) {
            throw std::logic_error("logical KV page already has a writer");
        }
        page.writer_references = writer ? 1U : 0U;
    }

    [[nodiscard]] bool can_set_writer(LogicalKVPageHandle handle, bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return writer ? page.writer_references == 0 && page.references == 1 &&
                            page.source_pins == 0 && page.device_replica.has_value()
                      : page.writer_references == 1;
    }

    void protect_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = std::max(page.protected_columns, columns);
    }

    void set_protected_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = columns;
    }

    void clear_protection(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.references != 1 || page.source_pins != 0) {
            throw std::logic_error("shared logical KV protection cannot be cleared in place");
        }
        page.protected_columns = 0;
    }

    void pin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("logical KV source pin count overflow");
        }
        ++page.source_pins;
    }

    void unpin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == 0) { throw std::logic_error("logical KV source is not pinned"); }
        --page.source_pins;
    }

    void commit_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!page.device_replica || page.writer_references != 1 ||
            columns < page.committed_columns ||
            columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
            throw std::invalid_argument("logical KV committed coverage is not monotonic");
        }
        page.committed_columns = columns;
    }

    void destructive_truncate(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns || columns < page.protected_columns ||
            page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica) {
            throw std::invalid_argument("logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
        }
    }

    [[nodiscard]] bool can_destructive_truncate(LogicalKVPageHandle handle,
                                                std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && columns >= page.protected_columns &&
               page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool
    can_destructive_truncate_inactive_after_host_release(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               page.device_replica.has_value();
    }

    void destructive_truncate_inactive(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!can_destructive_truncate_inactive(handle, columns)) {
            throw std::logic_error("inactive logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
        }
        page.protected_columns = std::min(page.protected_columns, columns);
    }

    [[nodiscard]] bool can_dematerialize(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_release_reference(LogicalKVPageHandle handle,
                                             bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        if (page.references == 0 || (writer && page.writer_references != 1) ||
            page.active_references >= page.references || page.source_pins != 0 ||
            page.destination_pinned) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_reference_after_active_reference(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != 0 && page.active_references != 0 &&
               page.active_references <= page.references && page.writer_references <= 1 &&
               page.source_pins == 0 && !page.destination_pinned;
    }

    [[nodiscard]] bool release_reference(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!can_release_reference(handle, writer)) { return false; }
        Page& page = pages_[handle.index_];
        if (writer) { page.writer_references = 0; }
        if (--page.references != 0) { return true; }
        page.device_replica.reset();
        // A Host extent owns its page capability until the extent is released as one atomic
        // allocation. Keep the zero-reference logical descriptor alive long enough for
        // HostKVExtentStore to detach that replica; without a Host replica the descriptor can be
        // reclaimed immediately.
        if (!page.host_replica) { release_descriptor(handle, page); }
        return true;
    }

    [[nodiscard]] bool can_attach_host_replica(LogicalKVPageHandle handle, std::uint64_t epoch,
                                               std::uint32_t coverage) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return !page.host_replica && epoch == page.content_epoch &&
               coverage == page.committed_columns;
    }

    void attach_host_replica(LogicalKVPageHandle handle, HostKVPageReplica replica) {
        Page& page = require(handle);
        if (!replica.extent.valid() ||
            replica.membership_node == std::numeric_limits<std::uint32_t>::max() ||
            !can_attach_host_replica(handle, replica.content_epoch, replica.committed_columns)) {
            throw std::logic_error("logical KV Host replica is not publishable");
        }
        page.host_replica = replica;
    }

    void rebind_host_replica(LogicalKVPageHandle handle, HostKVPageReplica expected,
                             HostKVPageReplica replacement) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != expected.extent ||
            page.host_replica->page_offset != expected.page_offset ||
            page.host_replica->membership_node != expected.membership_node ||
            page.host_replica->content_epoch != expected.content_epoch ||
            page.host_replica->committed_columns != expected.committed_columns ||
            expected.content_epoch != page.content_epoch || !replacement.extent.valid() ||
            replacement.membership_node != expected.membership_node ||
            replacement.content_epoch != expected.content_epoch ||
            replacement.committed_columns != expected.committed_columns) {
            std::terminate();
        }
        page.host_replica = replacement;
    }

    [[nodiscard]] bool detach_host_replica(LogicalKVPageHandle handle,
                                           HostKVExtentCapability extent) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != extent ||
            (!page.device_replica && page.references != 0)) {
            return false;
        }
        page.host_replica.reset();
        if (page.references == 0 && page.source_pins == 0 && !page.device_replica) {
            release_descriptor(handle, page);
        }
        return true;
    }

    void dematerialize(LogicalKVPageHandle handle, DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica || !page.device_replica) {
            throw std::logic_error("logical KV page is shared or has no Device replica");
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool release(LogicalKVPageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        return release_reference(handle, page.writer_references != 0);
    }

private:
    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t source_pins       = 0;
        std::uint8_t writer_references  = 0;
        bool destination_pinned         = false;
        bool occupied                   = false;
        std::optional<DeviceKVPageLease> device_replica;
        std::optional<DeviceKVPageLease> pending_device_replica;
        std::optional<HostKVPageReplica> host_replica;
    };

    [[nodiscard]] static std::uint64_t next_epoch(std::uint64_t epoch) noexcept {
        ++epoch;
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Page& require(LogicalKVPageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    [[nodiscard]] const Page& require(LogicalKVPageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    void release_descriptor(LogicalKVPageHandle handle, Page& page) noexcept {
        if (page.active_references != 0) { std::terminate(); }
        page.committed_columns  = 0;
        page.references         = 0;
        page.active_references  = 0;
        page.protected_columns  = 0;
        page.source_pins        = 0;
        page.writer_references  = 0;
        page.destination_pinned = false;
        page.occupied           = false;
        page.pending_device_replica.reset();
        page.host_replica.reset();
        if (++page.generation == 0) { ++page.generation; }
        free_[free_count_++] = handle.index_;
    }

    DeviceKVPagePool* physical_ = nullptr;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
    std::vector<DeviceKVPageLease> materialization_scratch_;
    std::uint32_t free_count_ = 0;
};

class KVAddressSpaceStore {
    static constexpr std::size_t kAsyncBatchBytes = 16ULL << 20;
    static bool layer_ready_requested() {
        static const bool enabled = [] {
            const char* flag = std::getenv("NINFER_KVMEM_LAYER_READY");
            if (!flag || std::strcmp(flag, "0") == 0) { return false; }
            if (std::strcmp(flag, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_KVMEM_LAYER_READY must be 0 or 1");
        }();
        return enabled;
    }
    static bool transfer_pipeline_enabled() {
        static const bool value = [] {
            const char* flag = std::getenv("NINFER_KVMEM_TRANSFER_PIPELINE");
            if (!flag || std::strcmp(flag, "0") == 0) { return false; }
            if (std::strcmp(flag, "1") == 0) { return true; }
            throw std::invalid_argument("NINFER_KVMEM_TRANSFER_PIPELINE must be 0 or 1");
        }();
        return value;
    }
    struct PendingDemote {
        struct Page {
            std::uint32_t block = 0;
            LogicalKVPageHandle logical;
            std::uint64_t epoch = 0;
            std::uint32_t columns = 0;
            HostKVAllocation destination;
            bool pinned = false;
        };

        PendingDemote(LogicalKVPageStore& owner, const DeviceContext& device,
                       KVAddressSpaceHandle handle)
            : store(&owner), address(handle), stream(device.transfer_stream),
              source_ready(device), completion(device) {}

        ~PendingDemote() {
            // The copy may have been partially queued before a submission
            // error. Its storage remains owned until every queued access lands.
            if (submitted) {
                if (recorded) {
                    completion.synchronize();
                } else if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    std::terminate();
                }
            }
            for (auto& page : pages) {
                if (page.pinned) { store->unpin_source(page.logical); }
                if (detached && !store->release_reference(page.logical, false)) {
                    std::terminate();
                }
            }
        }

        LogicalKVPageStore* store;
        KVAddressSpaceHandle address;
        cudaStream_t stream;
        CudaCompletionEvent source_ready;
        CudaCompletionEvent completion;
        std::vector<Page> pages;
        bool submitted = false;
        bool recorded = false;
        bool detached = false;
    };

    // Restores stay private to this scheduling boundary. Their Device targets
    // are destination-pinned, and their Host sources remain owned until the
    // copy completes. Only the final rephase/publication may expose them.
    struct PendingRestore {
        struct Page {
            std::uint32_t block;
            LogicalKVPageHandle logical;
            HostKVAllocation source;
        };
        PendingRestore(LogicalKVPageStore& owner, DeviceKVPageReservation& claim,
                       std::vector<HostKVAllocation>& hosts, const DeviceContext& device)
            : store(&owner), reservation(&claim), sources(&hosts),
              stream(device.transfer_stream), source_ready(device), completion(device) {}
        ~PendingRestore() {
            if (submitted) {
                if (recorded) { completion.synchronize(); }
                else if (cudaStreamSynchronize(stream) != cudaSuccess) { std::terminate(); }
            }
            for (auto& page : pages) {
                store->abort_transfer_destination(page.logical, *reservation);
                if (page.source.valid()) {
                    (*sources)[page.block] = std::move(page.source);
                }
            }
        }
        LogicalKVPageStore* store;
        DeviceKVPageReservation* reservation;
        std::vector<HostKVAllocation>* sources;
        cudaStream_t stream;
        CudaCompletionEvent source_ready;
        CudaCompletionEvent completion;
        std::vector<Page> pages;
        bool submitted = false;
        bool recorded = false;
    };

public:
    // kvmem_compact keeps a sink, a recent tail and a middle quota, so its window cannot be
    // narrower than one page per band.
    static constexpr std::uint32_t kMinimumWindowPages = 4;
    // Floor of the automatic sink/recent clamps (1024 + 4096 tokens). A budget below it cannot
    // hold the automatic bands, so the planner is handed explicit proportional ones instead --
    // otherwise a budget of 4K..5K tokens would refuse to plan at all.
    static constexpr std::uint32_t kAutoBandFloorTokens = 1024 + 4096;

    // Native original-position prefix identity, not a StateImage. The caller
    // captures/restores the matching GDN/hidden state separately at logical q0.
    struct OriginalPrefixCheckpoint {
        KVAddressSpaceHandle address;
        std::uint32_t logical_frontier = 0;
        std::uint64_t lineage_epoch = 0;
        std::vector<std::uint64_t> prefix_revisions;
        std::vector<float> partial_mean;
        std::uint32_t partial_mean_tokens = 0;
    };

    [[nodiscard]] bool kvmem_original_positions() const noexcept {
        return kvmem_original_positions_;
    }

    // Read-only diagnostic of the bytes actually visible to attention. Reuse
    // the native KVMem Host codec, bounded to one page; omit unwritten columns
    // and layout padding. The visitor owns hashing/reporting, not storage.
    template<class Visitor>
    void visit_original_window(KVAddressSpaceHandle handle, Visitor&& visitor,
                               cudaStream_t stream) {
        (void)retire_kvmem_transfers(true);
        auto& address = require_active(handle);
        if (!kvmem_original_positions_ || address.original_epoch == 0) {
            throw std::logic_error("native window inspection needs a live original epoch");
        }
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        auto& pool = pages_->physical_pool();
        ensure_kvmem_host_arena(pool);
        auto staging = kvmem_cold_buffer();
        auto view = kvmem_cold_staging_->writable_view(staging);
        for (const auto& block : kvmem_repos_[index].blocks()) {
            if (block.tier != ninfer::kvmem::KvTier::Gpu) { continue; }
            const auto columns = kvmem_native_records_[index].at(block.id).columns;
            require_original_native_owner(index, block.id, columns);
            const std::array<DeviceKVPageHandle, 1> source{
                pages_->physical(kvmem_block_pages_[index].at(block.id))};
            const std::array<HostKVAllocationView, 1> destination{view};
            (void)pool.copy_to_host_pages(source, destination, stream);
            kvmem_cold_sync(stream);
            visitor(&block.id, sizeof(block.id));
            visitor(&columns, sizeof(columns));
            for (std::size_t p = 0; p < kvmem_host_layout_.planes.size(); ++p) {
                const auto& plane = kvmem_host_layout_.planes[p];
                const auto heads = kvmem_host_layout_.geometry.planes[p].head_extent;
                for (std::int32_t h = 0; h < heads; ++h) {
                    visitor(view.data() + plane.offset + h * plane.head_payload_bytes,
                            plane.head_payload_bytes / kPagedKVPageSize * columns);
                }
            }
        }
    }

    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity,
                        KvMemOptions kvmem = {}, DeviceSpan rerope_workspace = {},
                        std::uint32_t extra_device_pages = 0)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity),
          memberships_(checked_membership_cells(address_capacity, page_capacity)),
          free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        // KVMem K1a: an enabled window mirrors the page lifecycle into a K0 block
        // repository per address and re-plans the (passthrough) window on every
        // frontier commit. Purely observational -- no engine state is redirected.
        if (kvmem.enabled) {
            kvmem_enabled_ = true;
            if (const char* value = std::getenv("NINFER_KVMEM_ORIGINAL_POSITIONS")) {
                if (std::strcmp(value, "1") == 0) { kvmem_original_positions_ = true; }
                else if (std::strcmp(value, "0") != 0) {
                    throw std::invalid_argument("NINFER_KVMEM_ORIGINAL_POSITIONS must be 0 or 1");
                }
            }
            kvmem_extra_device_pages_ = std::min(extra_device_pages,
                                               pages.physical_pool().capacity_pages());
            kvmem_host_layout_ = plan_host_kv_page_layout(pages.physical_pool().geometry());
            if (!kvmem_original_positions_) {
                kvmem_source_k_ = std::make_unique<ninfer::kvmem::ImmutableKStore>(
                    pages.physical_pool().geometry());
            }
            kvmem_action_lrfu_ = kvmem.action_lrfu;
            kvmem_lrfu_lambda_ = kvmem.lrfu_lambda;
            kvmem_lrfu_top_fraction_ = kvmem.lrfu_top_fraction;
            if (kvmem_action_lrfu_ &&
                (!(kvmem_lrfu_lambda_ >= 0.0 && kvmem_lrfu_lambda_ <= 1.0) ||
                 !(kvmem_lrfu_top_fraction_ > 0.0 && kvmem_lrfu_top_fraction_ <= 1.0))) {
                throw std::invalid_argument("KVMem LRFU settings are out of range");
            }
            if (kvmem.budget_tokens != 0) {
                kvmem_budget_ = std::min<std::uint32_t>(
                    kvmem.budget_tokens,
                    static_cast<std::uint32_t>(page_capacity) * kPagedKVPageSize);
                // Page-aligned once here so the completion-time window and the
                // window-edge anchor frontier are the same number.
                kvmem_budget_ -= kvmem_budget_ % kPagedKVPageSize;
            }
            // Generation headroom. The pool has to keep this much room free while a request
            // decodes, so kvmem_compact_generation reclaims the window-external blocks once the
            // resident window would leave less than it. The request is the configured value,
            // clamped to the room this pool actually has beyond the window: a pool that only
            // spans the window reserves nothing, and the mechanism stays inert exactly where the
            // pool covers the context.
            constexpr std::uint32_t kPageTokens =
                static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t pool_pages = pages.physical_pool().capacity_pages();
            const std::uint32_t window_pages =
                kvmem.budget_tokens == 0 ? page_capacity : kvmem_budget_ / kPageTokens;
            const std::uint32_t spare_pages =
                pool_pages > window_pages ? pool_pages - window_pages : 0U;
            const std::uint32_t room_pages =
                pool_pages > kMinimumWindowPages ? pool_pages - kMinimumWindowPages : 0U;
            std::uint32_t reserve =
                std::min(kvmem.gen_reserve_tokens, room_pages * kPageTokens);
            reserve = std::min(reserve, spare_pages * kPageTokens);
            kvmem_gen_reserve_ = reserve - reserve % kPageTokens;
            kvmem_repos_.resize(address_capacity);
            kvmem_block_pages_.resize(address_capacity);
            kvmem_block_hosts_.resize(address_capacity);
            kvmem_block_cold_.resize(address_capacity);
            kvmem_source_records_.resize(address_capacity);
            kvmem_native_records_.resize(address_capacity);
            kvmem_reuse_floors_.resize(address_capacity);
            kvmem_mean_k_.resize(address_capacity);
            kvmem_mean_k_tokens_.resize(address_capacity);
            kvmem_lexical_pages_.resize(address_capacity);
            kvmem_older_matches_.resize(address_capacity);
            const std::size_t metadata_bytes =
                static_cast<std::size_t>(page_capacity_) * 3 * sizeof(std::int32_t);
            if (rerope_workspace.data == nullptr || rerope_workspace.bytes < metadata_bytes) {
                throw std::invalid_argument("KVMem needs its planned rephase metadata workspace");
            }
            kvmem_rerope_host_ = std::make_unique<PinnedHostBuffer>(metadata_bytes);
            kvmem_rerope_device_ = rerope_workspace;
        }
        for (std::uint32_t index = 0; index < address_capacity; ++index) {
            free_[index] = address_capacity - 1U - index;
        }
        publish_scratch_.reserve(page_capacity_);
    }

    KVAddressSpaceStore(const KVAddressSpaceStore&)            = delete;
    KVAddressSpaceStore& operator=(const KVAddressSpaceStore&) = delete;
    KVAddressSpaceStore(KVAddressSpaceStore&&)                 = delete;
    KVAddressSpaceStore& operator=(KVAddressSpaceStore&&)      = delete;

    ~KVAddressSpaceStore() {
        // Cold staging is owned by this store and may still be read by an
        // independent per-layer copy stream when a request is cancelled.
        try {
            if (kvmem_layer_ready_) { kvmem_layer_ready_->drain_consumers(); }
            if (kvmem_source_k_) { kvmem_source_k_->drain_layer_source(); }
            drain_cold_layer_buffers();
        } catch (...) { std::terminate(); }
        kvmem_pending_demote_.reset();
        if (kvmem_enabled_) {
            const char* trace = std::getenv("NINFER_KVMEM_LAYER_READY_TRACE");
            if (trace && std::strcmp(trace, "1") == 0) {
                std::fprintf(stderr,
                    "KVMEM_LAYER_FINAL pending_pinned_pages=%zu inflight_pages=%u "
                    "producer_active=%d failure_drains=%llu\n",
                    kvmem_layer_pending_pages(), kvmem_inflight_pages(),
                    kvmem_layer_ready_ && kvmem_layer_ready_->active() ? 1 : 0,
                    static_cast<unsigned long long>(kvmem_layer_failure_drains_));
            }
        }
    }

    // The Program owns the stream and outlives this store. One bounded batch
    // per typed pool is enabled first; the synchronous path remains available.
    void enable_kvmem_async_d2h(const DeviceContext& device) {
        if (!kvmem_enabled_ || device.transfer_stream == nullptr) {
            throw std::invalid_argument("async KVMem needs an enabled store and transfer stream");
        }
        ensure_kvmem_host_arena(pages_->physical_pool());
        kvmem_async_device_ = &device;
    }

    [[nodiscard]] std::uint32_t kvmem_inflight_pages() const noexcept {
        return kvmem_pending_demote_
                   ? static_cast<std::uint32_t>(kvmem_pending_demote_->pages.size()) : 0U;
    }

    bool retire_kvmem_transfers(bool wait = false) {
        if (!kvmem_pending_demote_) { return true; }
        auto& pending = *kvmem_pending_demote_;
        const bool ready = pending.completion.ready();
        if (!wait && !ready) { return false; }
        const auto wait_start = std::chrono::steady_clock::now();
        pending.completion.synchronize();
        if (!ready) {
            kvmem_async_wait_us_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wait_start).count());
        }
        const Address& address = require(pending.address);
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        auto& hosts = kvmem_block_hosts_[index];
        // Validate the complete batch before publishing any destination.
        for (const auto& page : pending.pages) {
            if (!pages_->valid(page.logical) ||
                pages_->content_epoch(page.logical) != page.epoch ||
                pages_->committed_columns(page.logical) != page.columns ||
                page.block >= hosts.size() || hosts[page.block].valid()) {
                throw std::logic_error("KVMem async demote is no longer publishable");
            }
        }
        for (auto& page : pending.pages) {
            hosts[page.block] = std::move(page.destination);
        }
        ++kvmem_async_completed_;
        kvmem_pending_demote_.reset();
        return true;
    }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(addresses_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_active(std::uint32_t entitlement,
                                                                    std::int32_t execution_row) {
        if (entitlement == 0 || entitlement > page_capacity_) { return std::nullopt; }
        std::optional<KVAddressSpaceHandle> handle = create_inactive();
        if (!handle) { return std::nullopt; }
        try {
            activate(*handle, entitlement, execution_row);
            return handle;
        } catch (...) {
            (void)release(*handle);
            throw;
        }
    }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_inactive() noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_[--free_count_];
        Address& address          = addresses_[index];
        if (address.occupied) {
            free_[free_count_++] = index;
            return std::nullopt;
        }
        address.occupied = true;
        return KVAddressSpaceHandle(this, index, address.generation);
    }

    [[nodiscard]] bool valid(KVAddressSpaceHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < addresses_.size() &&
               addresses_[handle.index_].occupied &&
               addresses_[handle.index_].generation == handle.generation_;
    }

    void activate(KVAddressSpaceHandle handle, std::uint32_t entitlement,
                  std::int32_t execution_row) {
        Address& address = require(handle);
        if (entitlement < address.page_count) {
            throw std::logic_error("KV address space is not activatable");
        }
        auto reservation = prepare_activation(handle, entitlement, execution_row);
        commit_activation(std::move(reservation));
    }

    [[nodiscard]] KVActivationReservation
    prepare_activation(KVAddressSpaceHandle handle, std::uint32_t entitlement,
                       std::int32_t execution_row,
                       std::optional<std::uint32_t> activation_frontier = std::nullopt) {
        Address& address = require(handle);
        if (address.failed_compaction || address.active || address.row ||
            address.reservation.valid() || entitlement == 0 ||
            entitlement > page_capacity_ ||
            (activation_frontier && *activation_frontier > address.committed_frontier)) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        const std::uint32_t required_pages =
            activation_frontier ? pages_for_tokens(*activation_frontier) : address.page_count;
        // Only the device claim is reserved. A host-backed address space may name more
        // logical pages than the pool holds, and the excess lives in the host tier.
        const std::uint32_t device_claim = device_claim_pages(entitlement);
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        std::uint32_t missing_replicas      = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (!pages_->device_resident(membership(address, page))) { ++missing_replicas; }
        }
        const std::uint32_t growth =
            device_claim > required_pages ? device_claim - required_pages : 0U;
        const std::uint64_t required = static_cast<std::uint64_t>(missing_replicas) + growth;
        if (required > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("KV activation reservation overflow");
        }
        pages_->physical_pool().resize_reservation(reservation,
                                                   static_cast<std::uint32_t>(required));
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, entitlement, device_claim,
                                       activation_frontier, std::move(reservation), std::move(row));
    }

    // Logical page handles of the mapped window, in slot order. Orthogonal to
    // missing_device_pages (which filters by residency).
    [[nodiscard]] std::vector<LogicalKVPageHandle>
    window_page_handles(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        std::vector<LogicalKVPageHandle> out;
        out.reserve(address.page_count);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            out.push_back(membership(address, page));
        }
        return out;
    }

    [[nodiscard]] std::vector<LogicalKVPageHandle>
    missing_device_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        std::vector<LogicalKVPageHandle> missing;
        missing.reserve(address.page_count);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical)) { missing.push_back(logical); }
        }
        return missing;
    }

    [[nodiscard]] DeviceKVPageReservation& page_reservation(KVActivationReservation& activation) {
        if (activation.owner_ != this || !valid(activation.address_)) {
            throw std::logic_error("KV activation reservation is stale");
        }
        return activation.page_reservation_;
    }

    void commit_activation(KVActivationReservation&& activation, cudaStream_t stream = nullptr) {
        if (activation.owner_ != this) {
            throw std::logic_error("KV activation reservation belongs to another store");
        }
        if (!activation.row_) {
            throw std::logic_error("KV activation reservation has no execution row");
        }
        if (!valid(activation.address_)) {
            throw std::logic_error("KV activation reservation address is stale");
        }
        Address& address = require(activation.address_);
        if (address.failed_compaction || address.active || address.row || address.reservation.valid() ||
            activation.requested_entitlement_ == 0 ||
            activation.requested_entitlement_ > page_capacity_ ||
            (activation.activation_frontier_ &&
             address.committed_frontier != *activation.activation_frontier_)) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.device_claim_pages_ > address.page_count
                                           ? activation.device_claim_pages_ - address.page_count
                                           : 0U;
        if (!activation.page_reservation_.belongs_to(pages_->physical_pool()) ||
            activation.page_reservation_.pages() != expected) {
            throw std::logic_error("KV activation capacity reservation changed");
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical) || (pages_->address_references(logical) == 1 &&
                                                      !pages_->can_set_writer(logical, true))) {
                throw std::logic_error("KV activation has an unavailable Device page");
            }
        }
        address.reservation = std::move(activation.page_reservation_);
        address.row.emplace(std::move(*activation.row_));
        activation.row_.reset();
        address.active              = true;
        address.checkpoint_frontier = 0;
        // Keep the logical bound above the device claim where the host tier carries the
        // difference; everywhere else the derived value already is the entitlement.
        address.logical_entitlement =
            activation.requested_entitlement_ > activation.device_claim_pages_
                ? activation.requested_entitlement_
                : 0U;
        activation.owner_           = nullptr;
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->address_references(logical) == 1) {
                pages_->clear_protection(logical);
                pages_->set_writer(logical, true);
            }
            pages_->retain_active_reference(logical);
        }
        publish_membership(address, stream);
    }

    void deactivate(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        drain_kvmem_layer_reader(handle);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0 &&
                !pages_->can_set_writer(logical, false)) {
                throw std::logic_error("KV deactivation has an invalid writer reference");
            }
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->release_active_reference(logical);
        }
        address.row.reset();
        address.reservation.release();
        address.active = false;
    }

    [[nodiscard]] KVPrefixForkReservation
    prepare_prefix_fork(KVAddressSpaceHandle source_handle, KVAddressSpaceHandle destination_handle,
                        std::uint32_t frontier, std::uint32_t entitlement,
                        std::int32_t execution_row, bool staged_tail_release = false) {
        Address& source      = require(source_handle);
        Address& destination = require(destination_handle);
        if (source.active || source.row || source.reservation.valid()) {
            throw std::logic_error("KV retained-prefix source is still active");
        }
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KV retained-prefix destination is not empty");
        }
        if (frontier == 0 || frontier > source.committed_frontier) {
            throw std::logic_error("KV retained-prefix frontier is unavailable");
        }
        const std::uint32_t required_pages = pages_for_tokens(frontier);
        if (entitlement < required_pages || entitlement > page_capacity_) {
            throw std::invalid_argument("KV retained-prefix entitlement is invalid");
        }
        const std::uint32_t page_size    = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t full_pages   = frontier / page_size;
        const std::uint32_t tail_columns = frontier % page_size;
        // The fork shares the leading full pages and only reserves its own tail window. A host-backed
        // source can carry a logical entitlement larger than the device pool, so the reservation has
        // to stop at the device claim or it asks the pool for pages that live on the Host tier.
        const std::uint32_t device_claim = device_claim_pages(entitlement);
        if (device_claim < required_pages) {
            throw std::invalid_argument("KV retained-prefix device claim cannot hold the prefix");
        }
        if (staged_tail_release && device_claim <= full_pages) {
            throw std::invalid_argument("KV retained-prefix device claim cannot hold the staged tail");
        }
        if (staged_tail_release && tail_columns == 0) {
            throw std::invalid_argument("staged KV prefix fork has no partial tail");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const std::uint32_t required      = page < full_pages ? page_size : tail_columns;
            if (!pages_->device_resident(logical) ||
                pages_->committed_columns(logical) < required || !pages_->can_pin_source(logical) ||
                (page < full_pages && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV retained-prefix source is not stable");
            }
        }

        const std::uint32_t reserved_pages =
            staged_tail_release ? 1U : device_claim > full_pages ? device_claim - full_pages : 0U;
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(reservation, reserved_pages);
        KVExecutionRowLease row = tables_->acquire(execution_row);
        std::optional<LogicalKVPageHandle> tail_destination;
        if (tail_columns != 0) {
            tail_destination = pages_->materialize_transfer_destination(reservation, tail_columns);
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->pin_source(membership(source, page));
        }
        return KVPrefixForkReservation(*this, source_handle, destination_handle, frontier,
                                       full_pages, tail_columns, entitlement, device_claim,
                                       std::move(reservation), std::move(row), tail_destination,
                                       staged_tail_release);
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_source(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (fork.tail_columns_ == 0) {
            throw std::logic_error("page-aligned KV prefix fork has no tail source");
        }
        const Address& source = require(fork.source_);
        return pages_->physical(membership(source, fork.full_pages_));
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_destination(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (!fork.tail_destination_) {
            throw std::logic_error("page-aligned KV prefix fork has no tail destination");
        }
        return pages_->physical(*fork.tail_destination_);
    }

    [[nodiscard]] LogicalKVPageHandle
    prefix_fork_tail_logical_source(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (fork.tail_columns_ == 0) {
            throw std::logic_error("page-aligned KV prefix fork has no logical tail source");
        }
        const Address& source = require(fork.source_);
        return membership(source, fork.full_pages_);
    }

    void settle_prefix_fork_tail_source(KVPrefixForkReservation& fork) {
        require_prefix_fork(fork);
        if (!fork.staged_tail_release_ || fork.tail_source_settled_ || fork.tail_columns_ == 0) {
            throw std::logic_error("KV prefix-fork tail is not stage-releasable");
        }
        Address& source                   = require(fork.source_);
        const LogicalKVPageHandle logical = membership(source, fork.full_pages_);
        if (!pages_->valid(logical) || pages_->source_pins(logical) == 0 ||
            !pages_->device_resident(logical)) {
            throw std::logic_error("KV prefix-fork tail source changed before settlement");
        }
        pages_->unpin_source(logical);
        fork.tail_source_settled_ = true;
    }

    void complete_prefix_fork_after_tail_release(KVPrefixForkReservation& fork) {
        require_prefix_fork(fork);
        if (!fork.staged_tail_release_ || !fork.tail_source_settled_ || fork.tail_columns_ == 0) {
            throw std::logic_error("KV prefix-fork staged release is incomplete");
        }
        const Address& source             = require(fork.source_);
        const LogicalKVPageHandle logical = membership(source, fork.full_pages_);
        if (!pages_->valid(logical) || pages_->device_resident(logical) ||
            !pages_->host_resident(logical)) {
            throw std::logic_error("KV prefix-fork retained tail was not moved to Host");
        }
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth =
            fork.device_claim_ > required_pages ? fork.device_claim_ - required_pages : 0U;
        pages_->physical_pool().resize_reservation(fork.page_reservation_, growth);
    }

    void commit_prefix_fork(KVPrefixForkReservation&& fork, cudaStream_t stream = nullptr) {
        require_prefix_fork(fork);
        Address& source                    = require(fork.source_);
        Address& destination               = require(fork.destination_);
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth =
            fork.device_claim_ > required_pages ? fork.device_claim_ - required_pages : 0U;
        if (!fork.row_ || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            fork.page_reservation_.pages() != growth) {
            throw std::logic_error("KV prefix-fork destination changed before publication");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const bool settled_tail = page == fork.full_pages_ && fork.tail_source_settled_;
            if ((!settled_tail &&
                 (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical))) ||
                (settled_tail && (!fork.staged_tail_release_ || pages_->device_resident(logical) ||
                                  !pages_->host_resident(logical))) ||
                (page < fork.full_pages_ && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV prefix-fork source changed before publication");
            }
        }
        if (fork.tail_columns_ != 0 &&
            (!fork.tail_destination_ || !pages_->valid(*fork.tail_destination_))) {
            throw std::logic_error("KV prefix-fork tail destination is stale");
        }

        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(source, page)));
        }
        if (fork.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*fork.tail_destination_));
        }
        tables_->publish(fork.row_->handle(), 0, publish_scratch_, stream);

        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
            membership(destination, page) = logical;
        }
        if (fork.tail_destination_) {
            pages_->publish_transfer_destination(*fork.tail_destination_, true);
            membership(destination, fork.full_pages_) = *fork.tail_destination_;
            fork.tail_destination_.reset();
        }
        destination.page_count         = required_pages;
        destination.committed_frontier = fork.frontier_;
        // Mirror commit_activation: the logical bound survives even when the device claim is smaller.
        destination.logical_entitlement =
            fork.requested_entitlement_ > fork.device_claim_ ? fork.requested_entitlement_ : 0U;
        destination.reservation = std::move(fork.page_reservation_);
        destination.row.emplace(std::move(*fork.row_));
        fork.row_.reset();
        destination.active = true;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->retain_active_reference(membership(destination, page));
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (page == fork.full_pages_ && fork.tail_source_settled_) { continue; }
            pages_->unpin_source(membership(source, page));
        }
        fork.owner_ = nullptr;
    }

    void abort_prefix_fork(KVPrefixForkReservation& fork) noexcept {
        if (fork.owner_ != this) { return; }
        if (valid(fork.source_)) {
            Address& source                    = addresses_[fork.source_.index_];
            const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                if (page == fork.full_pages_ && fork.tail_source_settled_) { continue; }
                const LogicalKVPageHandle logical = membership(source, page);
                if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                    pages_->unpin_source(logical);
                }
            }
        }
        if (fork.tail_destination_) {
            pages_->abort_transfer_destination(*fork.tail_destination_, fork.page_reservation_);
            fork.tail_destination_.reset();
        }
        fork.owner_ = nullptr;
    }

    [[nodiscard]] KVActiveSnapshotShape active_snapshot_shape(KVAddressSpaceHandle source_handle,
                                                              std::uint32_t frontier) const {
        const Address& source = require_active(source_handle);
        if (frontier == 0 || frontier != source.committed_frontier) {
            throw std::logic_error("active KV snapshot frontier is invalid");
        }
        const std::uint32_t required_pages    = pages_for_tokens(frontier);
        const std::uint32_t entitlement_pages = entitlement(source);
        if (source.page_count < required_pages || entitlement_pages < required_pages) {
            throw std::logic_error("active KV snapshot frontier is not fully materialized");
        }

        const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        KVActiveSnapshotShape shape{
            .full_pages   = frontier / page_size,
            .tail_columns = frontier % page_size,
        };
        bool mutable_full_seen = false;
        for (std::uint32_t page = 0; page < shape.full_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const std::uint32_t references    = pages_->address_references(logical);
            const std::uint8_t writers        = pages_->writer_references(logical);
            if (!pages_->device_resident(logical) || references == 0 ||
                pages_->committed_columns(logical) < page_size ||
                !pages_->can_retain_reference(logical, false)) {
                throw std::logic_error("active KV snapshot full page is not stable");
            }
            if (writers == 0) {
                if (mutable_full_seen || !pages_->can_pin_source(logical)) {
                    throw std::logic_error("active KV snapshot page ownership is not ordered");
                }
                ++shape.immutable_full_pages;
            } else if (writers == 1 && references == 1 && pages_->can_pin_active_source(logical)) {
                mutable_full_seen = true;
            } else {
                throw std::logic_error("active KV snapshot full page ownership is invalid");
            }
            if (references == 1) { ++shape.unique_full_pages; }
        }

        if (shape.tail_columns != 0) {
            const LogicalKVPageHandle tail = membership(source, shape.full_pages);
            if (!pages_->device_resident(tail) || pages_->writer_references(tail) != 1 ||
                pages_->address_references(tail) != 1 ||
                pages_->committed_columns(tail) < shape.tail_columns ||
                !pages_->can_pin_active_source(tail)) {
                throw std::logic_error("active KV snapshot tail is not a unique writer");
            }
        }
        return shape;
    }

    [[nodiscard]] KVActiveSnapshotReservation
    prepare_active_snapshot(KVAddressSpaceHandle source_handle,
                            KVAddressSpaceHandle destination_handle, std::uint32_t frontier) {
        Address& source      = require_active(source_handle);
        Address& destination = require(destination_handle);
        if (!source.row || !source.reservation.valid() || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            destination.committed_frontier != 0 || frontier == 0 ||
            frontier != source.committed_frontier) {
            throw std::logic_error("active KV snapshot endpoints are invalid");
        }
        const KVActiveSnapshotShape shape     = active_snapshot_shape(source_handle, frontier);
        const std::uint32_t required_pages    = pages_for_tokens(frontier);
        const std::uint32_t entitlement_pages = entitlement(source);
        if (source.page_count != required_pages) {
            throw std::logic_error("active KV snapshot source has untrimmed suffix pages");
        }

        DeviceKVPageReservation tail_reservation = pages_->physical_pool().make_empty_reservation();
        if (shape.copied_pages() != 0) {
            pages_->physical_pool().resize_reservation(tail_reservation, shape.copied_pages());
        }
        std::optional<LogicalKVPageHandle> tail_destination;
        try {
            if (shape.tail_columns != 0) {
                tail_destination =
                    pages_->materialize_transfer_destination(tail_reservation, shape.tail_columns);
            }
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                pages_->pin_source(membership(source, page));
            }
        } catch (...) {
            if (tail_destination) {
                pages_->abort_transfer_destination(*tail_destination, tail_reservation);
            }
            throw;
        }
        return KVActiveSnapshotReservation(*this, source_handle, destination_handle, frontier,
                                           shape, entitlement_pages, std::move(tail_reservation),
                                           tail_destination);
    }

    [[nodiscard]] DeviceKVPageHandle
    active_snapshot_tail_source(const KVActiveSnapshotReservation& snapshot) const {
        require_active_snapshot(snapshot);
        if (snapshot.shape_.tail_columns == 0) {
            throw std::logic_error("page-aligned active KV snapshot has no tail source");
        }
        const Address& source = require(snapshot.source_);
        return pages_->physical(membership(source, snapshot.shape_.full_pages));
    }

    [[nodiscard]] DeviceKVPageHandle
    active_snapshot_tail_destination(const KVActiveSnapshotReservation& snapshot) const {
        require_active_snapshot(snapshot);
        if (!snapshot.tail_destination_) {
            throw std::logic_error("page-aligned active KV snapshot has no tail destination");
        }
        return pages_->physical(*snapshot.tail_destination_);
    }

    void commit_active_snapshot(KVActiveSnapshotReservation&& snapshot,
                                cudaStream_t stream = nullptr) {
        require_active_snapshot(snapshot);
        Address& source      = require_active(snapshot.source_);
        Address& destination = require(snapshot.destination_);
        const KVActiveSnapshotShape shape =
            active_snapshot_shape(snapshot.source_, snapshot.frontier_);
        const std::uint32_t required_pages  = pages_for_tokens(snapshot.frontier_);
        const std::uint32_t expected_growth = snapshot.entitlement_ - required_pages;
        if (shape != snapshot.shape_ || !source.row || !source.reservation.valid() ||
            source.page_count != required_pages ||
            source.committed_frontier != snapshot.frontier_ ||
            source.reservation.pages() != expected_growth || destination.active ||
            destination.row || destination.reservation.valid() || destination.page_count != 0 ||
            destination.committed_frontier != 0 || snapshot.tail_reservation_.pages() != 0) {
            throw std::logic_error("active KV snapshot changed before publication");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical)) {
                throw std::logic_error("active KV snapshot source changed before publication");
            }
        }
        if (snapshot.shape_.tail_columns != 0 &&
            (!snapshot.tail_destination_ || !pages_->valid(*snapshot.tail_destination_))) {
            throw std::logic_error("active KV snapshot tail destination is stale");
        }

        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < snapshot.shape_.full_pages; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(source, page)));
        }
        if (snapshot.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*snapshot.tail_destination_));
        }
        tables_->publish(source.row->handle(), 0, publish_scratch_, stream);

        for (std::uint32_t page = 0; page < snapshot.shape_.full_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
            membership(destination, page) = logical;
        }
        if (snapshot.tail_destination_) {
            const LogicalKVPageHandle tail = membership(source, snapshot.shape_.full_pages);
            pages_->set_writer(tail, false);
            pages_->protect_coverage(tail, snapshot.shape_.tail_columns);
            pages_->publish_transfer_destination(*snapshot.tail_destination_, true);
            membership(destination, snapshot.shape_.full_pages) = *snapshot.tail_destination_;
            snapshot.tail_destination_.reset();
        }
        destination.page_count         = required_pages;
        destination.committed_frontier = snapshot.frontier_;
        destination.reservation        = std::move(source.reservation);
        destination.row                = std::move(source.row);
        destination.active             = true;

        if (snapshot.shape_.tail_columns != 0) {
            pages_->release_active_reference(membership(source, snapshot.shape_.full_pages));
            pages_->retain_active_reference(membership(destination, snapshot.shape_.full_pages));
        }

        source.row.reset();
        source.active              = false;
        source.checkpoint_frontier = snapshot.frontier_;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->unpin_source(membership(source, page));
        }
        snapshot.owner_ = nullptr;
    }

    void abort_active_snapshot(KVActiveSnapshotReservation& snapshot) noexcept {
        if (snapshot.owner_ != this) { return; }
        if (valid(snapshot.source_)) {
            Address& source                    = addresses_[snapshot.source_.index_];
            const std::uint32_t required_pages = pages_for_tokens(snapshot.frontier_);
            for (std::uint32_t page = 0; page < required_pages && page < source.page_count;
                 ++page) {
                const LogicalKVPageHandle logical = membership(source, page);
                if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                    pages_->unpin_source(logical);
                }
            }
        }
        if (snapshot.tail_destination_) {
            pages_->abort_transfer_destination(*snapshot.tail_destination_,
                                               snapshot.tail_reservation_);
            snapshot.tail_destination_.reset();
        }
        snapshot.owner_ = nullptr;
    }

    void resize_entitlement(KVAddressSpaceHandle handle, std::uint32_t entitlement) {
        (void)retire_kvmem_transfers();
        Address& address = require_active(handle);
        if (entitlement < address.page_count || entitlement > page_capacity_) {
            throw std::invalid_argument("KV entitlement is smaller than mapped pages");
        }
        const std::uint32_t device_claim = device_claim_pages(entitlement);
        const std::uint32_t desired =
            device_claim > address.page_count ? device_claim - address.page_count : 0U;
        if (desired > address.reservation.pages() &&
            desired - address.reservation.pages() > pages_->physical_pool().available_pages() &&
            desired - address.reservation.pages() <=
                pages_->physical_pool().available_pages() + kvmem_inflight_pages()) {
            (void)retire_kvmem_transfers(true);
        }
        address.logical_entitlement = entitlement > device_claim ? entitlement : 0U;
        pages_->physical_pool().resize_reservation(
            address.reservation,
            device_claim > address.page_count ? device_claim - address.page_count : 0U);
    }

    void release_growth_entitlement(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        pages_->physical_pool().resize_reservation(address.reservation, 0);
    }

    // Coverage is a lower bound. A speculative mapping may already extend beyond this stage's
    // needs; only an explicit truncate releases it, and commit_frontier publishes valid tokens.
    void ensure_mapped_to_tokens(KVAddressSpaceHandle handle, std::uint32_t tokens,
                                 cudaStream_t stream = nullptr) {
        Address& address           = require_active(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        // Coverage is bounded by the logical entitlement, never by the device claim: a
        // host-backed address space still maps every logical page even where the pool is
        // smaller.
        const std::uint32_t limit = std::max(entitlement(address), address.logical_entitlement);
        if (target > limit) {
            throw std::invalid_argument(
                "KV coverage exceeds active entitlement: tokens=" + std::to_string(tokens) +
                " required_pages=" + std::to_string(target) +
                " mapped_pages=" + std::to_string(address.page_count) +
                " reserved_pages=" + std::to_string(address.reservation.pages()) +
                " entitlement=" + std::to_string(limit));
        }
        if (target <= address.page_count) { return; }
        const std::uint32_t begin       = address.page_count;
        const std::uint32_t count       = target - begin;
        const std::size_t address_index = static_cast<std::size_t>(&address - addresses_.data());
        std::span<LogicalKVPageHandle> added(
            memberships_.data() + address_index * page_capacity_ + begin, count);
        const std::optional<LogicalKVPageHandle> predecessor =
            begin == 0 ? std::nullopt
                       : std::optional<LogicalKVPageHandle>(membership(address, begin - 1U));
        pages_->materialize(address.reservation, added, predecessor);
        try {
            publish_scratch_.clear();
            for (const LogicalKVPageHandle page : added) {
                publish_scratch_.push_back(pages_->physical(page));
            }
            tables_->publish(address.row->handle(), begin, publish_scratch_, stream);
        } catch (...) {
            for (LogicalKVPageHandle& page : added) {
                pages_->dematerialize(page, address.reservation);
                page = {};
            }
            throw;
        }
        for (const LogicalKVPageHandle page : added) { pages_->retain_active_reference(page); }
        address.page_count = target;
        if (kvmem_enabled_) {
            auto& repo          = kvmem_repos_[address_index];
            // A resumed window -- or any fork/snapshot membership -- hands this
            // address pages that no append registered. Register them born-placed at
            // the slot their writer already baked, so the ledger stays aligned with
            // the address before the new tail is appended.
            while (repo.block_count() < begin) {
                const std::uint32_t id = repo.block_count();
                ninfer::kvmem::KvBlockMeta m;
                m.id             = id;
                m.token_begin    = repo.total_tokens();
                m.n_tokens       = static_cast<std::uint32_t>(kPagedKVPageSize);
                m.baked_pos      = static_cast<std::int64_t>(id) * kPagedKVPageSize;
                m.in_working_set = true;
                (void)repo.append(m);
                repo.set_tier(id, ninfer::kvmem::KvTier::Gpu);
                if (kvmem_block_pages_.size() > static_cast<std::size_t>(address_index)) {
                    auto& block_pages = kvmem_block_pages_[static_cast<std::size_t>(address_index)];
                    if (block_pages.size() <= id) { block_pages.resize(id + 1); }
                    block_pages[id] = membership(address, id);
                }
            }
            std::uint32_t added_i = 0;
            for (const LogicalKVPageHandle page : added) {
                ninfer::kvmem::KvBlockMeta m;
                m.id          = repo.block_count();
                // Ledger order is the tracker's order, so the new tail block keys off the tokens
                // already registered while its window slot is the page being appended. On a
                // KVMem-compacted window the two differ: slots were compacted, ledgers were not.
                m.token_begin = repo.total_tokens();
                m.n_tokens    = static_cast<std::uint32_t>(kPagedKVPageSize);
                // Appended blocks are BORN PLACED: the append kernel already baked
                // their K at the current end of window (= this natural slot), so
                // they register as resident with valid phases, exactly like
                // laamaafung's register_append.
                m.baked_pos = static_cast<std::int64_t>(begin + added_i) * kPagedKVPageSize;
                m.in_working_set = true;
                (void)repo.append(m);
                repo.set_tier(m.id, ninfer::kvmem::KvTier::Gpu);
                if (kvmem_block_pages_.size() > static_cast<std::size_t>(address_index)) {
                    auto& block_pages = kvmem_block_pages_[static_cast<std::size_t>(address_index)];
                    if (block_pages.size() <= m.id) { block_pages.resize(m.id + 1); }
                    block_pages[m.id] = membership(address, begin + added_i);
                }
                ++added_i;
            }
        }
    }

    void commit_frontier(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier < address.committed_frontier ||
            pages_for_tokens(frontier) > address.page_count) {
            throw std::invalid_argument("KV committed frontier is invalid");
        }
        if (frontier == address.committed_frontier) { return; }
        const std::uint32_t page_size          = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t first_changed_page = address.committed_frontier / page_size;
        const std::uint32_t final_changed_page = (frontier - 1U) / page_size;
        for (std::uint32_t page = first_changed_page; page <= final_changed_page; ++page) {
            const std::uint32_t begin   = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t columns = std::min(page_size, frontier - begin);
            if (columns > pages_->committed_columns(membership(address, page))) {
                pages_->commit_coverage(membership(address, page), columns);
            }
        }
        address.committed_frontier = frontier;
        if (kvmem_enabled_) {
            const auto index = static_cast<std::size_t>(&address - addresses_.data());
            kvmem_plan_passthrough(index, address);
            if (kvmem_original_positions_) { commit_original_native_columns(index, address); }
        }
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        if (frontier == address.committed_frontier && target == address.page_count) { return; }
        drain_kvmem_layer_reader(handle);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_dematerialize(membership(address, page))) {
                throw std::logic_error("KV truncate would partially release a protected page");
            }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (!pages_->can_destructive_truncate(membership(address, target - 1U), columns)) {
                throw std::logic_error("KV truncate would overwrite protected coverage");
            }
        }
        while (address.page_count > target) {
            const std::uint32_t index  = --address.page_count;
            LogicalKVPageHandle page   = membership(address, index);
            membership(address, index) = {};
            pages_->release_active_reference(page);
            pages_->dematerialize(page, address.reservation);
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            pages_->destructive_truncate(membership(address, target - 1U), columns);
        }
        address.committed_frontier = frontier;
        if (kvmem_enabled_) {
            kvmem_forget_window_tail(static_cast<std::size_t>(&address - addresses_.data()), frontier);
        }
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(
        KVAddressSpaceHandle handle, std::uint32_t frontier,
        bool tail_host_replica_will_be_released = false) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return false;
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->can_release_reference(logical, false)) { return false; }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail) &&
                !(tail_host_replica_will_be_released
                      ? pages_->can_destructive_truncate_inactive_after_host_release(tail, columns)
                      : pages_->can_destructive_truncate_inactive(tail, columns))) {
                return false;
            }
        }
        return true;
    }

    void destructive_truncate_inactive(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_destructive_truncate_inactive(handle, frontier)) {
            throw std::logic_error("inactive KV address space is not destructively truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        // Full pages before the selected frontier can remain shared. Only removed suffix
        // references and a changed private partial tail need mutation; protection is rebuilt from
        // the surviving address-space requirements after publication.
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            membership(address, index)        = {};
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail)) {
                pages_->destructive_truncate_inactive(tail, columns);
            }
        }
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = 0;
        rebuild_checkpoint_protection();
        if (kvmem_enabled_) {
            kvmem_forget_window_tail(static_cast<std::size_t>(&address - addresses_.data()), frontier);
        }
    }

    void set_checkpoint_requirement(KVAddressSpaceHandle handle, std::uint32_t protected_frontier) {
        Address& address = require(handle);
        if (protected_frontier > address.committed_frontier) {
            throw std::invalid_argument("KV checkpoint requirement exceeds committed frontier");
        }
        address.checkpoint_frontier = protected_frontier;
        rebuild_checkpoint_protection();
    }

    [[nodiscard]] bool can_truncate_inactive_prefix(KVAddressSpaceHandle handle,
                                                    std::uint32_t frontier) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return false;
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    void truncate_inactive_prefix(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_truncate_inactive_prefix(handle, frontier)) {
            throw std::logic_error("inactive KV checkpoint prefix is not truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            membership(address, index)        = {};
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = std::min(address.checkpoint_frontier, frontier);
        rebuild_checkpoint_protection();
        if (kvmem_enabled_) {
            kvmem_forget_window_tail(static_cast<std::size_t>(&address - addresses_.data()), frontier);
        }
    }

    [[nodiscard]] std::uint32_t mapped_pages(KVAddressSpaceHandle handle) const {
        return require(handle).page_count;
    }

    [[nodiscard]] std::uint32_t entitlement(KVAddressSpaceHandle handle) const {
        return entitlement(require(handle));
    }

    // Highest logical page count this active address may map. It stays above the device claim
    // wherever the host tier carries the difference, so a caller that reclaims pages (a mid
    // prefill offload) can re-establish the growth entitlement from it.
    [[nodiscard]] std::uint32_t mapping_limit(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        return std::max(entitlement(address), address.logical_entitlement);
    }

    // The physical pool may be larger than the KVMem window granted to this
    // address. Generation may materialize only up to its device claim; pages
    // beyond that bound must first be recovered through window compaction.
    // Use the same device_claim_pages() policy as activation and resize_entitlement
    // so decode reclaim cannot accidentally use spare pages claimed by other lanes.
    [[nodiscard]] std::uint32_t kvmem_device_claim_pages(KVAddressSpaceHandle handle) const {
        return device_claim_pages(mapping_limit(handle));
    }

    [[nodiscard]] std::uint32_t kvmem_device_claim_limit() const noexcept {
        return device_claim_pages(page_capacity_);
    }

    // Start with some existing growth entitlement still unused. The lead is
    // shared by sync/async, so switching transfer mode cannot change selection.
    // It is bounded in typed payload bytes, never borrowed from protected bands.
    [[nodiscard]] std::uint32_t kvmem_reclaim_watermark_pages(KVAddressSpaceHandle handle) const {
        const std::uint32_t claim = kvmem_device_claim_pages(handle);
        const std::uint32_t reserve = kvmem_gen_reserve_ / kPagedKVPageSize;
        if (reserve + kMinimumWindowPages > claim) { return claim; }
        const auto stride = kvmem_host_layout_.page_stride;
        const std::uint32_t lead = static_cast<std::uint32_t>(
            std::min<std::size_t>(reserve / 2, kAsyncBatchBytes / stride));
        return claim - lead;
    }

    // Allocation still uses the actual claim and real free pages. A promise
    // merely avoids duplicate demotion; resize_entitlement harvests it first.
    [[nodiscard]] bool kvmem_generation_reclaim_due(KVAddressSpaceHandle handle,
                                                     std::uint32_t next_frontier_tokens) const {
        return kvmem_enabled_ && kvmem_gen_reserve_ != 0 &&
               pages_for_tokens(next_frontier_tokens) > kvmem_reclaim_watermark_pages(handle);
    }

    // Leading pages this address shares with another view (a retained shared prefix or a
    // snapshot continuation). KVMem has to pin them: re-phasing a page moves it for every view.
    // The run starts at page zero because prefix forks publish the shared prefix first.
    [[nodiscard]] std::uint32_t kvmem_shared_prefix_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        std::uint32_t pages    = 0;
        while (pages < address.page_count &&
               pages_->address_references(membership(address, pages)) > 1) {
            ++pages;
        }
        return pages;
    }

    [[nodiscard]] bool kvmem_enabled() const noexcept { return kvmem_enabled_; }

    // Only an explicit prefill compaction may publish a layer-ready ticket;
    // decode/Graph and other addresses never inherit a producer by accident.
    [[nodiscard]] ninfer::kvmem::LayerReadyConsumer*
    kvmem_layer_ready_for(KVAddressSpaceHandle handle) noexcept {
        return handle.valid() && handle == kvmem_layer_ready_address_ &&
                       kvmem_layer_ready_ && kvmem_layer_ready_->active()
            ? kvmem_layer_ready_.get() : nullptr;
    }

    [[nodiscard]] std::size_t kvmem_layer_pending_pages() const noexcept {
        return cold_layer_count_ +
            (kvmem_source_k_ ? kvmem_source_k_->pending_layer_source_pages() : 0);
    }

    [[nodiscard]] std::uint64_t kvmem_layer_failure_drains() const noexcept {
        return kvmem_layer_failure_drains_;
    }

    void drain_kvmem_layer_reader(KVAddressSpaceHandle handle) {
        if (kvmem_layer_ready_ && handle.valid() && handle == kvmem_layer_ready_address_) {
            // Cancellation / address invalidation is not a same-stream
            // remap. Drain *both* the producer and final queued consumer reads
            // before giving its physical pages or execution row back.
            kvmem_layer_ready_->drain_consumers();
            drain_cold_layer_buffers();
            kvmem_layer_ready_address_ = {};
        }
    }

    [[nodiscard]] std::uint32_t committed_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).committed_frontier;
    }

    [[nodiscard]] std::int32_t bound_row(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle) || !addresses_[handle.index_].row) { return -1; }
        return addresses_[handle.index_].row->row_index();
    }

    [[nodiscard]] bool active(KVAddressSpaceHandle handle) const noexcept {
        return valid(handle) && addresses_[handle.index_].active;
    }

    [[nodiscard]] const KVExecutionRowLease& execution_row(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row) {
            throw std::logic_error("KV address space has no execution row");
        }
        return *address.row;
    }

    [[nodiscard]] DeviceKVPageHandle physical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->physical(membership(address, logical_page));
    }

    [[nodiscard]] std::uint64_t content_epoch(KVAddressSpaceHandle handle,
                                              std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->content_epoch(membership(address, logical_page));
    }

    [[nodiscard]] LogicalKVPageHandle logical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return membership(address, logical_page);
    }

    [[nodiscard]] bool has_active_reference(LogicalKVPageHandle page) const noexcept {
        return pages_->valid(page) && pages_->active_address_references(page) != 0;
    }

    [[nodiscard]] bool can_release(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid()) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool can_release_after_deactivate(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (!address.active) { return can_release(handle); }
        if (!address.row) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference_after_active_reference(membership(address, page))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool release_after_deactivate(KVAddressSpaceHandle handle) noexcept {
        if (!can_release_after_deactivate(handle)) { return false; }
        try {
            if (addresses_[handle.index_].active) { deactivate(handle); }
        } catch (...) { std::terminate(); }
        if (!release(handle)) { std::terminate(); }
        return true;
    }

    [[nodiscard]] bool release(KVAddressSpaceHandle handle) noexcept {
        if (!can_release(handle)) { return false; }
        try { drain_kvmem_layer_reader(handle); }
        catch (...) { std::terminate(); }
        if (kvmem_pending_demote_ && kvmem_pending_demote_->address == handle) {
            // The address is being discarded: drain without publishing an
            // otherwise unnecessary Host replica, then release its held refs.
            kvmem_pending_demote_.reset();
            ++kvmem_async_discarded_;
        }
        Address& address = addresses_[handle.index_];
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->release_reference(membership(address, page), false)) { std::terminate(); }
            membership(address, page) = {};
        }
        const std::uint32_t index      = handle.index_;
        const std::uint32_t generation = next_generation(address.generation);
        address                        = Address{};
        address.generation             = generation;
        free_[free_count_++]           = index;
        if (kvmem_enabled_) {
            kvmem_repos_[index] = {};
            kvmem_block_pages_[index].clear();
            for (HostKVAllocation& host : kvmem_block_hosts_[index]) { host.release(); }
            kvmem_block_hosts_[index].clear();
            for (const auto cold : kvmem_block_cold_[index]) {
                if (cold.valid()) { (void)kvmem_cold_file_->release(cold); }
            }
            kvmem_block_cold_[index].clear();
            for (auto& source : kvmem_source_records_[index]) {
                if (kvmem_source_k_) { kvmem_source_k_->release(source); }
            }
            kvmem_source_records_[index].clear();
            kvmem_native_records_[index].clear();
            kvmem_reuse_floors_[index] = 0;
            kvmem_mean_k_[index].clear();
            kvmem_mean_k_tokens_[index].clear();
            kvmem_lexical_pages_[index].clear();
            kvmem_older_matches_[index].clear();
        }
        rebuild_checkpoint_protection();
        return true;
    }

private:
    struct Address {
        std::uint32_t generation          = 1;
        std::uint32_t page_count          = 0;
        std::uint32_t committed_frontier  = 0;
        std::uint32_t checkpoint_frontier = 0;
        // Logical mapping bound. Zero means "derive it from the reservation". A device
        // pool below the context records it above the device claim, so coverage may
        // still name every logical page the host tier backs.
        std::uint32_t logical_entitlement = 0;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
        // The only legal follow-up after a failed page remap is address
        // disposal. Partially restored contents must never be activated.
        bool failed_compaction = false;
        // An epoch covers one original execution branch. Ordinary append and
        // native owner migration preserve it; a destructive rewind revokes it.
        std::uint64_t original_epoch = 1;
        std::uint64_t next_native_revision = 1;
    };

    [[nodiscard]] static std::size_t checked_membership_cells(std::uint32_t addresses,
                                                              std::uint32_t pages) {
        if (pages != 0 && addresses > std::numeric_limits<std::size_t>::max() / pages) {
            throw std::overflow_error("KV ordered-membership capacity overflow");
        }
        return static_cast<std::size_t>(addresses) * pages;
    }

    [[nodiscard]] static std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
        return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    [[nodiscard]] static std::uint32_t next_generation(std::uint32_t generation) noexcept {
        ++generation;
        return generation == 0 ? 1 : generation;
    }

    void rebuild_checkpoint_protection() noexcept {
        try {
            for (const Address& address : addresses_) {
                if (!address.occupied) { continue; }
                for (std::uint32_t page = 0; page < address.page_count; ++page) {
                    pages_->set_protected_coverage(membership(address, page), 0);
                }
            }
            for (const Address& address : addresses_) {
                if (!address.occupied || address.checkpoint_frontier == 0) { continue; }
                if (address.checkpoint_frontier > address.committed_frontier ||
                    pages_for_tokens(address.checkpoint_frontier) > address.page_count) {
                    std::terminate();
                }
                for (std::uint32_t page = 0; page < pages_for_tokens(address.checkpoint_frontier);
                     ++page) {
                    const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
                    pages_->protect_coverage(membership(address, page),
                                             std::min(static_cast<std::uint32_t>(kPagedKVPageSize),
                                                      address.checkpoint_frontier - begin));
                }
            }
        } catch (...) { std::terminate(); }
    }

    [[nodiscard]] Address& require(KVAddressSpaceHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] const Address& require(KVAddressSpaceHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    void require_prefix_fork(const KVPrefixForkReservation& fork) const {
        if (fork.owner_ != this || !valid(fork.source_) || !valid(fork.destination_)) {
            throw std::logic_error("KV prefix-fork reservation is stale");
        }
    }

    void require_active_snapshot(const KVActiveSnapshotReservation& snapshot) const {
        if (snapshot.owner_ != this || !valid(snapshot.source_) || !valid(snapshot.destination_)) {
            throw std::logic_error("active KV snapshot reservation is stale");
        }
    }

    [[nodiscard]] Address& require_active(KVAddressSpaceHandle handle) {
        Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] const Address& require_active(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] std::uint32_t entitlement(const Address& address) const noexcept {
        return address.page_count +
               (address.reservation.valid() ? address.reservation.pages() : 0U);
    }

    // Device pages an activation may claim from the physical pool. The pool can sit below
    // the logical entitlement when the host tier backs the rest of the address space, and
    // only the claim is reserved; wherever the pool already covers the context the clamp is
    // inert and the claim is the entitlement itself.
    //
    // KVMem caps the claim at the retained window budget, not the pool. The window is what
    // the address actually keeps resident, and admission sums claims across lanes: claiming
    // the whole pool would let one long request reserve every page and keep every other
    // KVMem lane out even though the compaction window leaves room for it.
    [[nodiscard]] std::uint32_t device_claim_pages(std::uint32_t entitlement) const noexcept {
        std::uint32_t claim = std::min(entitlement, pages_->physical_pool().capacity_pages());
        if (kvmem_enabled_ && kvmem_budget_ != 0) {
            claim = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                claim, kvmem_budget_ / static_cast<std::uint32_t>(kPagedKVPageSize) +
                           static_cast<std::uint64_t>(kvmem_extra_device_pages_)));
        }
        return claim;
    }

    [[nodiscard]] LogicalKVPageHandle& membership(Address& address, std::uint32_t page) noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    [[nodiscard]] const LogicalKVPageHandle& membership(const Address& address,
                                                        std::uint32_t page) const noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    void publish_membership(const Address& address, cudaStream_t stream = nullptr) {
        if (!address.row) { throw std::logic_error("KV address space has no execution row"); }
        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(address, page)));
        }
        tables_->publish(address.row->handle(), 0, publish_scratch_, stream);
    }

    [[nodiscard]] ninfer::kvmem::ColdKVStore& kvmem_cold_file() {
        if (!kvmem_cold_file_) {
            kvmem_cold_file_ =
                std::make_unique<ninfer::kvmem::ColdKVStore>(kvmem_host_layout_.page_stride);
        }
        return *kvmem_cold_file_;
    }

    // Two read-ahead pages plus one outgoing staging page remain independently
    // bounded even when the ordinary Host cache has no vacant allocation.
    [[nodiscard]] HostKVAllocation kvmem_cold_buffer() {
        if (!kvmem_cold_staging_) {
            kvmem_cold_staging_ = std::make_unique<HostKVArena>(
                3 * kvmem_host_layout_.page_stride,
                std::vector<HostKVPageLayout>{kvmem_host_layout_});
        }
        auto buffer = kvmem_cold_staging_->allocate(kvmem_host_layout_, 1);
        if (!buffer) { throw std::runtime_error("KVMem cold staging page unavailable"); }
        return std::move(*buffer);
    }

    static void kvmem_cold_sync(cudaStream_t stream) {
        const cudaError_t error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("KVMem cold transfer: ") +
                                     cudaGetErrorString(error));
        }
    }

    struct ColdLayerSlot {
        HostKVAllocation staging;
        std::int32_t physical = -1;
        std::uint32_t block_id = 0;
    };

    void drain_cold_layer_buffers() {
        if (cold_layer_producer_stream_) {
            CUDA_CHECK(cudaStreamSynchronize(cold_layer_producer_stream_));
            cold_layer_producer_stream_ = nullptr;
        }
        for (std::size_t i = 0; i < cold_layer_count_; ++i) {
            cold_layer_slots_[i].staging.release();
            cold_layer_slots_[i].physical = -1;
        }
        cold_layer_count_ = 0;
    }

    // Cold futures have completed and their source handle generation was
    // checked before arriving here. These one/two HostKVAllocation instances
    // stay owned until producer GPU DMA retires, not until memcpy is enqueued.
    void enqueue_cold_layer(std::uint32_t layer, DeviceKVPagePool& pool,
                            cudaStream_t producer) {
        if (!cold_layer_producer_stream_ || producer != cold_layer_producer_stream_ ||
            cold_layer_count_ == 0 || layer >= 16 ||
            pool.geometry().page_tokens != kPagedKVPageSize ||
            pool.geometry().device_plane_order != PagedKVPlaneOrder::PageMajor ||
            pool.geometry().planes.size() != 64) {
            throw std::logic_error("Cold layer copy has no compatible pending owner");
        }
        for (std::size_t i = 0; i < cold_layer_count_; ++i) {
            const auto& entry = cold_layer_slots_[i];
            const auto view = kvmem_cold_staging_->view(entry.staging);
            if (!view.valid() || view.page_count() != 1 ||
                view.layout() != kvmem_host_layout_) {
                throw std::logic_error("Cold layer source staging is stale");
            }
            for (std::uint32_t group = 0; group < 4; ++group) {
                const std::size_t plane_index = layer * 4 + group;
                const auto& plane = pool.plane(plane_index);
                const auto& packed = kvmem_host_layout_.planes[plane_index];
                const auto heads = pool.geometry().planes[plane_index].head_extent;
                if (plane.nb[2] < static_cast<std::int64_t>(packed.head_payload_bytes)) {
                    throw std::logic_error("Cold layer destination head pitch is invalid");
                }
                auto* device = static_cast<std::byte*>(plane.data) +
                    static_cast<std::int64_t>(entry.physical) * plane.nb[3];
                const auto* source = view.data() + packed.offset;
                CUDA_CHECK(cudaMemcpy2DAsync(device, static_cast<std::size_t>(plane.nb[2]),
                    source, packed.head_payload_bytes, packed.head_payload_bytes, heads,
                    cudaMemcpyHostToDevice, producer));
                // OpenSpec 3.4 fault receipt: one real H2D has been submitted
                // but no layer_ready has been recorded. Error unwinding must
                // retain this Host source and physical target until DMA ends.
                if (layer == 0 && i == 0 && group == 0) {
                    const char* fault = std::getenv("NINFER_KVMEM_TEST_PARTIAL_LAYER_H2D_FAILURE");
                    if (fault && std::strcmp(fault, "1") == 0) {
                        throw std::runtime_error("KVMem injected partial Cold layer H2D failure");
                    }
                }
            }
        }
    }

    // Reclaim one Host slot before allocating an evicted page. The closed think
    // tier is evicted first, followed by low-LRFU old context. System and live
    // action pages are protected from this background Host->disk demotion.
    [[nodiscard]] bool kvmem_spill_one_host(
        std::size_t current_address, std::span<const std::uint32_t> restoring) {
        struct Candidate {
            std::size_t address;
            std::uint32_t id;
            const ninfer::kvmem::KvBlockMeta* meta;
        };
        std::optional<Candidate> candidate;
        for (std::size_t address = 0; address < kvmem_repos_.size(); ++address) {
            const auto blocks = kvmem_repos_[address].blocks();
            const auto& hosts = kvmem_block_hosts_[address];
            for (std::uint32_t id = 0; id < hosts.size() && id < blocks.size(); ++id) {
                if (!hosts[id].valid() ||
                    kvmem_repos_[address].tier_of(id) != ninfer::kvmem::KvTier::Host ||
                    !ninfer::kvmem::kvmem_cold_spillable(blocks[id]) ||
                    (address == current_address &&
                     std::find(restoring.begin(), restoring.end(), id) != restoring.end())) {
                    continue;
                }
                if (!candidate || ninfer::kvmem::kvmem_colder_than(blocks[id], *candidate->meta)) {
                    candidate = Candidate{address, id, &blocks[id]};
                }
            }
        }
        if (!candidate) { return false; }
        const std::size_t address = candidate->address;
        const std::uint32_t id = candidate->id;
        auto& records = kvmem_block_cold_[address];
        if (records.size() <= id) { records.resize(id + 1); }
        if (records[id].valid()) {
            throw std::logic_error("KVMem Host and NVMe both own the same block");
        }
        HostKVAllocation& host = kvmem_block_hosts_[address][id];
        const HostKVAllocationConstView view = kvmem_host_arena_->view(host);
        const auto record = kvmem_cold_file().store(
            std::span<const std::byte>(view.data(), kvmem_host_layout_.page_stride));
        records[id] = record;
        host.release();
        kvmem_repos_[address].set_tier(id, ninfer::kvmem::KvTier::Nvme);
        ++kvmem_host_to_cold_spills_;
        return true;
    }

    [[nodiscard]] ninfer::kvmem::ColdKVStore::Handle kvmem_archive_device(
        DeviceKVPagePool& pool, DeviceKVPageHandle page, cudaStream_t stream,
        std::uint64_t& transfer_runs) {
        HostKVAllocation buffer = kvmem_cold_buffer();
        const auto view = kvmem_cold_staging_->writable_view(buffer);
        const std::array<DeviceKVPageHandle, 1> devices{page};
        const std::array<HostKVAllocationView, 1> hosts{view};
        transfer_runs += pool.copy_to_host_pages(devices, hosts, stream);
        kvmem_cold_sync(stream);
        return kvmem_cold_file().store(
            std::span<const std::byte>(view.data(), kvmem_host_layout_.page_stride));
    }

    // ---- KVMem K1a (observational passthrough) ----
    // Mirrors the page lifecycle into K0 block repositories and re-plans the
    // window on every frontier commit. In passthrough the selection is "every
    // mapped page in order", so the plan must always cover exactly the mapped
    // tokens and every remap must be skip-or-first-placement. The executor
    // contract is applied right after planning: staged-in ids become Gpu.
    // Passthrough re-plan after every frontier commit: the window is exactly
    // the GPU-resident blocks in baked order, and every block must already sit
    // at its window slot. Blocks evicted by kvmem_compact (tier Host, bytes
    // parked in its host arena) stay in the ledger and are NOT part of the
    // window; the address's page_count only ever counts the window.

    // KVMem K1b: the address was destructively truncated to `slot_frontier` cache slots, so the
    // resident blocks past that frontier leave the window for good: leftovers of a compaction
    // window are never re-selected into a slot that no longer exists. Those blocks are the
    // newest resident ones, which occupy the tracker's end in ledger order, so the tracker drops
    // a suffix and hands their parked host copies back to the arena. Blocks below the frontier
    // keep their own window slots, and evicted (Host) blocks stay registered -- parked bytes are
    // not the window's business.
    void kvmem_forget_window_tail(std::size_t index, std::uint32_t slot_frontier) {
        auto& repo        = kvmem_repos_[index];
        auto& block_pages = kvmem_block_pages_[index];
        auto& block_hosts = kvmem_block_hosts_[index];
        // Rollback can replace columns of the retained partial tail. Preserve
        // only its surviving immutable prefix; the next append captures the
        // new columns with their own birth positions.
        auto& sources = kvmem_source_records_[index];
        bool original_changed = false;
        if (kvmem_original_positions_) {
            auto& native = kvmem_native_records_[index];
            for (const auto& block : repo.blocks()) {
                if (block.tier != ninfer::kvmem::KvTier::Gpu || block.id >= native.size()) {
                    continue;
                }
                const auto columns = block.baked_pos >= 0 && block.baked_pos < slot_frontier
                    ? std::min<std::uint32_t>(kPagedKVPageSize, slot_frontier - block.baked_pos)
                    : 0U;
                if (columns < native[block.id].columns) {
                    native[block.id].columns = columns;
                    original_changed = true;
                    // A clipped mean is not the mean of its surviving prefix.
                    if (block.id < kvmem_mean_k_tokens_[index].size()) {
                        kvmem_mean_k_tokens_[index][block.id] = 0;
                    }
                }
            }
        }
        for (std::uint32_t id = 0; id < std::min<std::size_t>(sources.size(), repo.block_count()); ++id) {
            if (repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu) { continue; }
            const auto baked = repo.baked_pos_of(id);
            const auto columns = baked >= 0 && baked < slot_frontier
                ? std::min<std::uint32_t>(kPagedKVPageSize, slot_frontier - baked) : 0U;
            sources[id].columns = std::min(sources[id].columns, columns);
        }
        std::uint32_t keep = repo.block_count();
        for (std::uint32_t id = keep; id != 0; --id) {
            const std::uint32_t candidate = id - 1;
            if (repo.tier_of(candidate) != ninfer::kvmem::KvTier::Gpu) { break; }
            const std::int64_t baked = repo.baked_pos_of(candidate);
            if (baked < 0 || static_cast<std::uint32_t>(baked) < slot_frontier) { break; }
            keep = candidate;
        }
        if (original_changed) { ++addresses_[index].original_epoch; }
        if (keep == repo.block_count()) { return; }
        for (std::uint32_t id = keep; id < repo.block_count(); ++id) {
            if (block_pages.size() > id) { block_pages[id] = {}; }
            if (block_hosts.size() > id && block_hosts[id].valid()) { block_hosts[id].release(); }
            if (kvmem_block_cold_[index].size() > id &&
                kvmem_block_cold_[index][id].valid()) {
                (void)kvmem_cold_file_->release(kvmem_block_cold_[index][id]);
            }
            if (sources.size() > id && kvmem_source_k_) { kvmem_source_k_->release(sources[id]); }
        }
        block_pages.resize(keep);
        block_hosts.resize(keep);
        kvmem_block_cold_[index].resize(keep);
        sources.resize(std::min<std::size_t>(sources.size(), keep));
        if (kvmem_original_positions_) { kvmem_native_records_[index].resize(keep); }
        if (kvmem_mean_k_.size() > index) {
            kvmem_mean_k_[index].resize(static_cast<std::size_t>(keep) * kvmem_mean_k_stride_);
            kvmem_mean_k_tokens_[index].resize(keep);
        }
        repo.truncate(keep);
    }

    void kvmem_plan_passthrough(std::size_t index, const Address& address) {
        auto& repo        = kvmem_repos_[index];
        auto& block_pages = kvmem_block_pages_[index];
        const std::uint32_t want = address.page_count;
        std::vector<std::uint32_t> ids;
        for (std::uint32_t id = 0; id < repo.block_count(); ++id) {
            if (repo.tier_of(id) == ninfer::kvmem::KvTier::Gpu) { ids.push_back(id); }
        }
        if (ids.size() > want) {
            throw std::logic_error("KVMem tracker: resident blocks exceed mapped pages");
        }
        // Resync: fork/snapshot memberships skip ensure_mapped_to_tokens, so
        // slots beyond the resident set are born-placed here -- the fork source
        // baked them at the very same slots.
        for (std::uint32_t id = static_cast<std::uint32_t>(ids.size()); id < want; ++id) {
            ninfer::kvmem::KvBlockMeta m;
            m.id             = id;
            m.token_begin    = static_cast<std::uint64_t>(id) * kPagedKVPageSize;
            m.n_tokens       = kPagedKVPageSize;
            m.baked_pos      = m.token_begin;
            m.in_working_set = true;
            (void)repo.append(m);
            repo.set_tier(id, ninfer::kvmem::KvTier::Gpu);
            if (block_pages.size() <= id) { block_pages.resize(id + 1); }
            block_pages[id] = membership(address, id);
            ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end(), [&](std::uint32_t a, std::uint32_t b) {
            return repo.baked_pos_of(a) < repo.baked_pos_of(b);
        });
        for (std::uint32_t slot = 0; slot < ids.size(); ++slot) {
            if (repo.baked_pos_of(ids[slot]) !=
                static_cast<std::int64_t>(slot) * static_cast<std::int64_t>(kPagedKVPageSize)) {
                throw std::logic_error("KVMem tracker: resident block is not window-placed");
            }
        }
        ninfer::kvmem::KvSelection sel;
        sel.block_ids = std::move(ids);
        ninfer::kvmem::KvSelectConfig cfg;
        cfg.budget_tokens = want * kPagedKVPageSize;
        ninfer::kvmem::KvWindowPlan plan = repo.set_selection(sel, cfg);
        if (plan.total_window_tokens !=
            static_cast<std::uint64_t>(want) * static_cast<std::uint64_t>(kPagedKVPageSize)) {
            throw std::logic_error("KVMem passthrough window accounting diverged");
        }
        for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
            if (!rm.skip) {
                throw std::logic_error(
                    "KVMem passthrough remap must be a no-op: id=" +
                    std::to_string(rm.block_id) + " baked=" + std::to_string(rm.from_base) +
                    " to=" + std::to_string(rm.to_base) + " plans=" +
                    std::to_string(kvmem_plans_));
            }
        }
        for (const std::uint32_t id : plan.stage_in) { repo.set_tier(id, ninfer::kvmem::KvTier::Gpu); }
        ++kvmem_plans_;
    }

    // KVMem K3: refresh the host mean-K of every resident block, then rank the
    // whole ledger -- resident and evicted -- against the request query. A block
    // keeps its host mean-K after it is parked, so a middle block evicted by an
    // earlier chunk can still be selected by the final question; the reference
    // scorer's evidence never depended on the block staying on the device.
    void kvmem_score_window(std::size_t index, DeviceKVPagePool& pool, cudaStream_t stream,
                            bool commit_query = false) {
        const ninfer::kvmem::KvmemQueryCapture* query = kvmem_query_capture();
        if (query == nullptr || query->layers == 0 || query->kv_heads == 0 ||
            kvmem_mean_k_stride_ == 0 || index >= kvmem_mean_k_.size()) {
            return;
        }
        auto& repo            = kvmem_repos_[index];
        const std::uint32_t n = repo.block_count();
        if (n == 0) { return; }
        const std::size_t stride = kvmem_mean_k_stride_;
        kvmem_mean_k_[index].resize(static_cast<std::size_t>(n) * stride, 0.0f);
        kvmem_mean_k_tokens_[index].resize(n, 0);

        const std::uint32_t frontier = addresses_[index].committed_frontier;
        constexpr std::uint32_t page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
        std::vector<std::uint32_t> export_ids;
        std::vector<std::int32_t> export_pages;
        std::vector<std::int32_t> export_baked;
        std::vector<std::uint32_t> export_tokens;
        for (std::uint32_t id = 0; id < n; ++id) {
            if (repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu) { continue; }
            if (kvmem_block_pages_[index].size() <= id ||
                !kvmem_block_pages_[index][id].valid()) {
                continue;
            }
            const std::int64_t baked = repo.baked_pos_of(id);
            if (baked < 0 || static_cast<std::uint64_t>(baked) >= frontier) {
                kvmem_mean_k_tokens_[index][id] = 0;
                continue;
            }
            const std::uint32_t written = kvmem_original_positions_
                ? kvmem_native_records_[index].at(id).columns
                : std::min(page_tokens, frontier - static_cast<std::uint32_t>(baked));
            if (written == 0) {
                kvmem_mean_k_tokens_[index][id] = 0;
                continue;
            }
            if (kvmem_mean_k_tokens_[index][id] == written) { continue; }
            export_ids.push_back(id);
            export_pages.push_back(
                pool.physical_index_of(pages_->physical(kvmem_block_pages_[index][id])));
            // In original mode baked_pos is only the packed cache placement.
            // Native K retains the logical RoPE it had at its first write.
            export_baked.push_back(static_cast<std::int32_t>(kvmem_original_positions_
                ? repo.blocks()[id].token_begin : baked));
            export_tokens.push_back(written);
        }
        if (!export_ids.empty()) {
            std::vector<float> exported(export_ids.size() * stride, 0.0f);
            try {
                ninfer::kvmem::kvmem_export_mean_k(
                    pool, export_pages.data(), export_baked.data(), export_tokens.data(),
                    static_cast<std::uint32_t>(export_ids.size()), exported.data(), stream);
            } catch (const std::exception& error) {
                // Retrieval only ranks the middle quota: a geometry it cannot read
                // must not make the compaction fail, it must leave it unranked.
                std::fprintf(stderr, "KVMEM_TRACE mean_k_failed %s\n", error.what());
                return;
            }
            for (std::size_t i = 0; i < export_ids.size(); ++i) {
                const std::uint32_t id = export_ids[i];
                std::copy_n(exported.data() + i * stride, stride,
                            kvmem_mean_k_[index].data() + static_cast<std::size_t>(id) * stride);
                kvmem_mean_k_tokens_[index][id] = export_tokens[i];
            }
        }

        const std::size_t query_elems = static_cast<std::size_t>(query->layers) *
                                        query->q_heads * query->head_dim;
        if (kvmem_query_host_.size() != query_elems) { kvmem_query_host_.resize(query_elems); }
        try {
            KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
            kvmem_query_buffer_.copy_to_host(kvmem_query_host_.data(),
                                             query_elems * sizeof(float));
        } catch (const std::exception& error) {
            std::fprintf(stderr, "KVMEM_TRACE query_copy_failed %s\n", error.what());
            return;
        }

        std::vector<float> scores;
        try {
            const ninfer::kvmem::MeanKeyQueryView view{kvmem_query_host_.data(), query->layers,
                                                       query->head_dim, query->q_heads,
                                                       query->kv_heads, false};
            scores = ninfer::kvmem::mean_key_scores(
                n, kvmem_mean_k_[index].data(), kvmem_mean_k_tokens_[index].data(), view);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "KVMEM_TRACE score_failed %s\n", error.what());
            return;
        }
        if (kvmem_action_lrfu_) {
            repo.protect_system_prefix(kvmem_system_prefix_end_);
            repo.protect_action_span(kvmem_action_begin_, kvmem_action_end_);
            repo.set_closed_thinking_spans(kvmem_closed_thinking_spans_);
            if (commit_query) {
                repo.record_lrfu_scores(scores, kvmem_lrfu_lambda_, kvmem_lrfu_top_fraction_);
            }
        }
        repo.set_retrieval_scores(scores);
        for (std::uint32_t id = 0; id < n; ++id) {
            if (kvmem_mean_k_tokens_[index][id] == 0) { continue; }
            ++kvmem_scoreable_blocks_;
            if (scores[id] == 0.0F) { ++kvmem_zero_score_blocks_; }
        }
        if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            for (std::uint32_t id = 0; id < n; ++id) {
                std::fprintf(stderr,
                             "KVMEM_TRACE score id=%u tier=%d tokens=%u retrieval=%.5f\n", id,
                             static_cast<int>(repo.tier_of(id)),
                             kvmem_mean_k_tokens_[index][id], scores[id]);
            }
        }
    }

public:
    // Benchmark prefix archive: quiescent canonical q0 only. The caller binds
    // model/prompt/config and validates the entire file before invoking load.
    // Adapt KVMem snapshot allocation ownership; runtime handles are recreated.
    template<class Writer>
    void save_benchmark_prefix(KVAddressSpaceHandle handle, Writer& out, cudaStream_t stream) {
        if (!kvmem_enabled_ || kvmem_original_positions_) {
            throw std::invalid_argument("prefix archive requires canonical KVMem");
        }
        (void)retire_kvmem_transfers(true);
        auto& address = require_active(handle);
        const auto index = static_cast<std::size_t>(&address-addresses_.data());
        if (address.checkpoint_frontier || address.original_epoch == 0) {
            throw std::logic_error("prefix archive has protected or invalid state");
        }
        ensure_kvmem_host_arena(pages_->physical_pool());
        out.pod(kvmem_host_layout_.page_stride);
        out.pod(kvmem_budget_); out.pod(kvmem_gen_reserve_);
        out.pod(kvmem_action_lrfu_); out.pod(kvmem_lrfu_lambda_); out.pod(kvmem_lrfu_top_fraction_);
        out.pod(kvmem_host_arena_->capacity_bytes());
        out.pod(address.page_count); out.pod(address.committed_frontier);
        out.pod(kvmem_reuse_floors_[index]);
        const auto& repo=kvmem_repos_[index];
        out.pod(repo.checkpoint_scoring_step()); out.pod(repo.block_count());
        for(const auto& block:repo.blocks()) out.pod(block);
        out.pod_vector(kvmem_mean_k_[index]); out.pod_vector(kvmem_mean_k_tokens_[index]);
        auto staging=kvmem_cold_buffer();
        auto view=kvmem_cold_staging_->writable_view(staging);
        auto& pool=pages_->physical_pool();
        for(const auto& block:repo.blocks()) {
            const auto id=block.id;
            const ninfer::kvmem::ImmutableKStore::Record empty{};
            kvmem_source_k_->save_record(id<kvmem_source_records_[index].size()
                ? kvmem_source_records_[index][id] : empty,out);
            if(block.tier==ninfer::kvmem::KvTier::Gpu) {
                const auto logical=kvmem_block_pages_[index].at(id);
                if(!logical.valid() || block.baked_pos<0 ||
                   membership(address,block.baked_pos/kPagedKVPageSize)!=logical) {
                    throw std::logic_error("prefix native resident identity");
                }
                const std::array<DeviceKVPageHandle,1> src{pages_->physical(logical)};
                const std::array<HostKVAllocationView,1> dst{view};
                (void)pool.copy_to_host_pages(src,dst,stream);kvmem_cold_sync(stream);
            } else if(block.tier==ninfer::kvmem::KvTier::Host) {
                const auto src=kvmem_host_arena_->view(kvmem_block_hosts_[index].at(id));
                out.write(src.data(),kvmem_host_layout_.page_stride);continue;
            } else {
                kvmem_cold_file().load(kvmem_block_cold_[index].at(id),
                    {view.data(),kvmem_host_layout_.page_stride});
            }
            out.write(view.data(),kvmem_host_layout_.page_stride);
        }
    }

    template<class Reader>
    void load_benchmark_prefix(KVAddressSpaceHandle handle, Reader& in, cudaStream_t stream) {
        if(!kvmem_enabled_ || kvmem_original_positions_) {
            throw std::invalid_argument("prefix archive requires canonical KVMem");
        }
        auto& address=require_active(handle);
        const auto index=static_cast<std::size_t>(&address-addresses_.data());
        if(address.page_count || address.committed_frontier || kvmem_repos_[index].block_count()) {
            throw std::logic_error("prefix load requires an empty exclusive address");
        }
        ensure_kvmem_host_arena(pages_->physical_pool());
        in.template expect<std::size_t>(kvmem_host_layout_.page_stride);
        in.template expect<std::uint32_t>(kvmem_budget_); in.template expect<std::uint32_t>(kvmem_gen_reserve_);
        in.template expect<bool>(kvmem_action_lrfu_); in.template expect<double>(kvmem_lrfu_lambda_);
        in.template expect<double>(kvmem_lrfu_top_fraction_);
        in.template expect<std::size_t>(kvmem_host_arena_->capacity_bytes());
        const auto count=in.template pod<std::uint32_t>();
        const auto frontier=in.template pod<std::uint32_t>();
        const auto floor=in.template pod<std::uint32_t>();
        const auto step=in.template pod<std::uint32_t>();
        const auto n=in.template pod<std::uint32_t>();
        if(count==0 || count>page_capacity_ || n<count || n>page_capacity_ ||
           pages_for_tokens(frontier)!=count || floor>frontier) {
            throw std::invalid_argument("prefix archive directory extent");
        }
        std::vector<ninfer::kvmem::KvBlockMeta> blocks(n);
        std::vector<std::uint8_t> seen(count,0);
        for(std::uint32_t i=0;i<n;++i) {
            blocks[i]=in.template pod<ninfer::kvmem::KvBlockMeta>();const auto& b=blocks[i];
            if(b.id!=i || b.token_begin!=std::uint64_t(i)*kPagedKVPageSize ||
               b.n_tokens!=kPagedKVPageSize || b.tier>ninfer::kvmem::KvTier::Nvme ||
               (b.in_working_set!=(b.tier==ninfer::kvmem::KvTier::Gpu))) {
                throw std::invalid_argument("prefix archive block identity");
            }
            if(b.in_working_set) {
                if(b.baked_pos<0 || b.baked_pos%kPagedKVPageSize ||
                   std::uint64_t(b.baked_pos/kPagedKVPageSize)>=count ||
                   seen[b.baked_pos/kPagedKVPageSize]++) {
                    throw std::invalid_argument("prefix archive resident permutation");
                }
            }
        }
        if(std::find(seen.begin(),seen.end(),0)!=seen.end()) {
            throw std::invalid_argument("prefix archive incomplete resident table");
        }
        auto means=in.template pod_vector<float>(std::size_t(n)*kvmem_mean_k_stride_);
        auto columns=in.template pod_vector<std::uint32_t>(n);
        ensure_mapped_to_tokens(handle,count*kPagedKVPageSize,stream);
        commit_frontier(handle,frontier);
        auto& repo=kvmem_repos_[index];repo={};
        for(const auto& block:blocks) (void)repo.append(block);
        repo.restore_scoring_step(step);
        kvmem_block_pages_[index].assign(n,{});
        kvmem_block_hosts_[index].resize(n);kvmem_block_cold_[index].resize(n);
        kvmem_source_records_[index].resize(n);
        kvmem_mean_k_[index]=std::move(means);kvmem_mean_k_tokens_[index]=std::move(columns);
        kvmem_reuse_floors_[index]=floor;
        const bool pipeline=ninfer::kvmem::checkpoint_load_pipeline_enabled();
        const auto load_began=std::chrono::steady_clock::now();
        std::uint64_t source_us=0,read_us=0,host_copy_us=0,gpu_submit_us=0,wait_us=0;
        std::uint32_t gpu_pages=0,host_pages=0,cold_pages=0,batch_calls=0;
        const auto add_time=[](auto began,std::uint64_t& total) {
            total+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-began).count();
        };
        // Adapt Strata released/ready to the existing bounded Cold staging arena.
        // Last H2D reader releases a slot; the next CPU read may use the other.
        // The whole native address remains unpublished until both have retired.
        struct LoadSlot {
            HostKVAllocation owner;
            cudaStream_t stream=nullptr;
            cudaEvent_t ready=nullptr;
            bool submitted=false;
            std::array<void*,64> destinations{};
            std::array<const void*,64> sources{};
            std::array<std::size_t,64> sizes{};
            void wait() {
                if(submitted) { KVMEM_CUDA_CHECK(cudaEventSynchronize(ready));submitted=false; }
            }
            ~LoadSlot() {
                // A failed partial enqueue may not be covered by its last event.
                if(submitted && cudaStreamSynchronize(stream)!=cudaSuccess) std::terminate();
                if(ready) (void)cudaEventDestroy(ready);
            }
        };
        std::array<LoadSlot,2> slots;
        for(std::size_t i=0;i<(pipeline?2U:1U);++i) {
            slots[i].owner=kvmem_cold_buffer();slots[i].stream=stream;
            if(pipeline) KVMEM_CUDA_CHECK(cudaEventCreateWithFlags(&slots[i].ready,cudaEventDisableTiming));
        }
        auto& pool=pages_->physical_pool();
        for(const auto& b:blocks) {
            auto began=std::chrono::steady_clock::now();
            kvmem_source_k_->load_record(kvmem_source_records_[index][b.id],in);
            add_time(began,source_us);
            if(pipeline && b.tier==ninfer::kvmem::KvTier::Host) {
                auto allocation=kvmem_host_arena_->allocate(kvmem_host_layout_,1);
                if(!allocation) throw std::runtime_error("prefix archive Host quota");
                auto dst=kvmem_host_arena_->writable_view(*allocation);
                began=std::chrono::steady_clock::now();
                in.read(dst.data(),kvmem_host_layout_.page_stride);add_time(began,read_us);
                kvmem_block_hosts_[index][b.id]=std::move(*allocation);++host_pages;continue;
            }
            auto& slot=slots[pipeline ? gpu_pages%2 : 0];
            began=std::chrono::steady_clock::now();slot.wait();add_time(began,wait_us);
            auto view=kvmem_cold_staging_->writable_view(slot.owner);
            began=std::chrono::steady_clock::now();
            in.read(view.data(),kvmem_host_layout_.page_stride);add_time(began,read_us);
            if(b.tier==ninfer::kvmem::KvTier::Gpu) {
                const auto logical=membership(address,b.baked_pos/kPagedKVPageSize);
                const std::array<HostKVAllocationConstView,1> src{view};
                const std::array<DeviceKVPageHandle,1> dst{pages_->physical(logical)};
                began=std::chrono::steady_clock::now();
                bool batch=false;
#if CUDART_VERSION >= 13000 && defined(NINFER_SM89)
                if(pipeline && stream && pool.plane_count()<=slot.sizes.size() &&
                   pool.geometry().device_plane_order==PagedKVPlaneOrder::PageMajor) {
                    batch=true;
                    const auto physical=pool.physical_index_of(dst[0]);
                    for(std::size_t i=0;i<pool.plane_count();++i) {
                        const auto& plane=pool.plane(i);const auto& packed=kvmem_host_layout_.planes[i];
                        if(plane.nb[2]!=static_cast<std::int64_t>(packed.head_payload_bytes) ||
                           plane.nb[3]<static_cast<std::int64_t>(packed.page_payload_bytes)) { batch=false;break; }
                        slot.destinations[i]=static_cast<std::byte*>(plane.data)+physical*plane.nb[3];
                        slot.sources[i]=view.data()+packed.offset;slot.sizes[i]=packed.page_payload_bytes;
                    }
                    if(batch) {
                        // Existing Infernix/ninfer-3090 CUDA13 batch-copy contract:
                        // independent ranges, stream-ordered source, owned descriptors.
                        cudaMemcpyAttributes attrs{};attrs.srcAccessOrder=cudaMemcpySrcAccessOrderStream;
                        std::size_t first=0;slot.submitted=true;
                        KVMEM_CUDA_CHECK(cudaMemcpyBatchAsync(slot.destinations.data(),slot.sources.data(),
                            slot.sizes.data(),pool.plane_count(),&attrs,&first,1,stream));++batch_calls;
                    }
                }
#endif
                if(!batch) {
                    slot.submitted=true;
                    (void)pool.copy_from_host_pages(src,dst,stream);
                }
                if(pipeline) KVMEM_CUDA_CHECK(cudaEventRecord(slot.ready,stream));
                add_time(began,gpu_submit_us);
                if(!pipeline) { began=std::chrono::steady_clock::now();kvmem_cold_sync(stream);slot.submitted=false;add_time(began,wait_us); }
                kvmem_block_pages_[index][b.id]=logical;++gpu_pages;
            } else if(b.tier==ninfer::kvmem::KvTier::Host) {
                began=std::chrono::steady_clock::now();
                auto allocation=kvmem_host_arena_->allocate(kvmem_host_layout_,1);
                if(!allocation) throw std::runtime_error("prefix archive Host quota");
                auto dst=kvmem_host_arena_->writable_view(*allocation);
                std::memcpy(dst.data(),view.data(),kvmem_host_layout_.page_stride);
                kvmem_block_hosts_[index][b.id]=std::move(*allocation);++host_pages;add_time(began,host_copy_us);
            } else {
                kvmem_block_cold_[index][b.id]=kvmem_cold_file().store(
                    {view.data(),kvmem_host_layout_.page_stride});++cold_pages;
            }
        }
        auto began=std::chrono::steady_clock::now();
        for(auto& slot:slots) slot.wait();add_time(began,wait_us);
        publish_membership(address,stream);kvmem_cold_sync(stream);
        std::fprintf(stderr,"KVMEM_CHECKPOINT_PAGES pipeline=%d private=%d gpu=%u host=%u cold=%u batch=%u source_us=%llu read_us=%llu host_copy_us=%llu submit_us=%llu wait_us=%llu total_us=%llu staging_slots=%u\n",
            pipeline,in.stable_backing()!=nullptr,gpu_pages,host_pages,cold_pages,batch_calls,
            static_cast<unsigned long long>(source_us),static_cast<unsigned long long>(read_us),
            static_cast<unsigned long long>(host_copy_us),static_cast<unsigned long long>(gpu_submit_us),
            static_cast<unsigned long long>(wait_us),static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-load_began).count()),pipeline?2:1);
    }

    // Commit one completed-prompt history round without moving pages. Compaction previews
    // (including complete-query reselection before the assistant opener) update current
    // retrieval only; they must not repeatedly count the same query in the LRFU profile.
    void kvmem_refresh_scores(KVAddressSpaceHandle handle, cudaStream_t stream = nullptr) {
        if (!kvmem_enabled_ || !kvmem_query_.valid()) { return; }
        Address& address        = require(handle);
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        kvmem_score_window(index, pages_->physical_pool(), stream, true);
    }

    // KVMem K1b: compact one address's KV into a budget-driven working window.
    //
    // Requires an INACTIVE address whose pages are solely owned (no fork or
    // snapshot sharing -- re-phasing a shared page would corrupt the other
    // views). Returns the new window token count, or 0 when nothing needed
    // compacting. Evicted blocks keep their bytes in this store's KVMem host
    // arena or the anonymous NVMe backing; a re-selected block is restored from there
    // and re-phased to its new window slot. Cold/threshold rebuilds decode the
    // immutable first-write K, never a repeatedly quantized working copy.
    std::uint32_t kvmem_compact(KVAddressSpaceHandle handle, std::uint32_t budget_tokens,
                                cudaStream_t stream = nullptr,
                                std::vector<std::uint32_t> mandatory_ids = {},
                                bool sink_only = false, std::uint32_t pinned_prefix_pages = 0,
                                std::span<const std::uint32_t> coupled_selection = {},
                                std::optional<std::uint32_t> replay_from = std::nullopt,
                                bool prefill_layer_consumer = false) {
        if (!kvmem_enabled_) { return 0; }
        // Finish the prior producer before reusing its pinned/device metadata;
        // reader writes on the compute stream remain ordered with new work.
        if (kvmem_layer_ready_) { kvmem_layer_ready_->retire(); }
        drain_cold_layer_buffers();
        kvmem_layer_ready_address_ = {};
        // Caller must immediately begin canonical replay before any consumer.
        // The reusable prefix retains native V; begin_kvmem_window_replay
        // recovers its immutable K. Later columns will be computed afresh.
        // Adapt KVMem's prepare/restore separation, retaining local ownership.
        if (replay_from && (kvmem_original_positions_ || pinned_prefix_pages != 0 ||
                            *replay_from % kPagedKVPageSize != 0)) {
            throw std::invalid_argument("deferred KV preparation needs an exclusive canonical replay");
        }
        (void)retire_kvmem_transfers(true);
        Address& address = require(handle);
        if (address.failed_compaction) {
            throw std::logic_error("failed KVMem compaction address must be released");
        }
        if (kvmem_original_positions_ && address.original_epoch == 0) {
            throw std::logic_error("original KV compaction requires fresh prefill after a failed rewind");
        }
        // A reused prefill can map growth ahead of its committed frontier.
        // Those empty pages are reservations, not historical evidence. Reuse
        // the normal rollback path before selecting recent pages, otherwise an
        // empty tail can be packed as a full interior page on the next append.
        if (address.active && address.page_count > pages_for_tokens(address.committed_frontier)) {
            if (std::getenv("NINFER_KVMEM_TRACE")) {
                std::fprintf(stderr,
                             "KVMEM_TRACE trim_unwritten mapped=%u committed=%u\n",
                             address.page_count, address.committed_frontier);
            }
            destructive_truncate(handle, address.committed_frontier);
        }
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        // Active addresses are allowed: the serve path compacts at request
        // completion while the execution row is still bound (before the
        // continuation checkpoint pins the pages). The table is republished at
        // the end for exactly that case.
        auto& repo        = kvmem_repos_[index];
        auto& block_pages = kvmem_block_pages_[index];
        auto& block_hosts = kvmem_block_hosts_[index];
        budget_tokens -= budget_tokens % kPagedKVPageSize;
        const std::uint64_t ledger_tokens =
            static_cast<std::uint64_t>(repo.block_count()) * kPagedKVPageSize;
        if (budget_tokens == 0 || (ledger_tokens <= budget_tokens &&
            (!kvmem_original_positions_ || coupled_selection.empty()))) { return 0; }
        if (pinned_prefix_pages > address.page_count) {
            throw std::invalid_argument("KVMem pinned prefix exceeds the mapped KV pages");
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const std::uint32_t references = pages_->address_references(membership(address, page));
            // The shared prefix stays at fixed slots and may be referenced by other views. The
            // remainder of the address must be privately owned, otherwise its phases cannot move.
            if (references == 0 || (page >= pinned_prefix_pages && references != 1)) {
                // A page this address shares with another view cannot move: its phases are
                // baked for the other view's slots too. The trace names the page that blocked
                // the compaction.
                if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
                    std::fprintf(stderr,
                                 "KVMEM_TRACE compact_skip shared page=%u references=%u "
                                 "pinned=%u budget=%u ledger=%llu\n",
                                 page, references, pinned_prefix_pages, budget_tokens,
                                 static_cast<unsigned long long>(ledger_tokens));
                }
                ++kvmem_compact_skips_;
                return 0;
            }
        }
        const std::uint32_t pinned_blocks = pinned_prefix_pages;
        const std::uint32_t budget_blocks = budget_tokens / kPagedKVPageSize;
        DeviceKVPagePool& pool = pages_->physical_pool();
        const auto& pipeline_geometry = pool.geometry();
        const bool layer_ready = prefill_layer_consumer && layer_ready_requested() &&
            !kvmem_original_positions_ && !replay_from && stream != nullptr &&
            stream != cudaStreamLegacy && pipeline_geometry.page_tokens == kPagedKVPageSize &&
            pipeline_geometry.device_plane_order == PagedKVPlaneOrder::PageMajor &&
            pipeline_geometry.planes.size() == 16 * 4;
        // A future.get(), partial CUDA enqueue or metadata publication may
        // throw after some work has already been submitted. Keep this scope
        // alive until every local ColdRead future has been joined during stack
        // unwinding; then drain both streams before the store can release or
        // reallocate any producer-owned Host source / physical KV target.
        bool compaction_complete = false;
        const auto abort_layer_attempt = [&]() noexcept {
            if (!layer_ready || compaction_complete) { return; }
            try {
                if (kvmem_layer_ready_) { kvmem_layer_ready_->drain_consumers(); }
                if (kvmem_source_k_) { kvmem_source_k_->drain_layer_source(); }
                drain_cold_layer_buffers();
                kvmem_layer_ready_address_ = {};
                // set_selection and stage-out can invalidate entries of the
                // old membership before a file read or a CUDA enqueue fails.
                // Rebuild only the address's release directory from valid
                // references already held by this address and its block
                // ledger. No partially restored data becomes consumable.
                const std::uint32_t old_count = address.page_count;
                std::uint32_t owned = 0;
                const auto retain_for_disposal = [&](LogicalKVPageHandle page) {
                    if (!pages_->valid(page)) { return; }
                    for (std::uint32_t i = 0; i < owned; ++i) {
                        if (membership(address, i) == page) { return; }
                    }
                    if (owned >= page_capacity_) { std::terminate(); }
                    membership(address, owned++) = page;
                };
                // This forward compaction never overwrites an unread member.
                for (std::uint32_t i = 0; i < old_count; ++i) {
                    retain_for_disposal(membership(address, i));
                }
                for (const auto page : kvmem_block_pages_[index]) {
                    retain_for_disposal(page);
                }
                for (std::uint32_t i = owned; i < old_count; ++i) {
                    membership(address, i) = {};
                }
                address.page_count = owned;
                address.committed_frontier = std::min(address.committed_frontier,
                    owned * static_cast<std::uint32_t>(kPagedKVPageSize));
                address.checkpoint_frontier = 0;
                address.failed_compaction = true;
                ++kvmem_layer_failure_drains_;
            } catch (...) { std::terminate(); }
        };
        struct LayerAttemptGuard {
            decltype(abort_layer_attempt)& action;
            ~LayerAttemptGuard() noexcept { action(); }
        } layer_attempt_guard{abort_layer_attempt};
        // Capture before set_selection mutates baked positions, and before any
        // working K can be re-phased or evicted. Only committed new columns
        // enter the immutable source. One pinned K page bounds staging RAM.
        if (!kvmem_original_positions_ && !kvmem_source_k_) {
            kvmem_source_k_ = std::make_unique<ninfer::kvmem::ImmutableKStore>(pool.geometry());
        }
        auto& sources = kvmem_source_records_[index];
        if (!kvmem_original_positions_) { sources.resize(repo.block_count()); }
        for (std::uint32_t id = 0; !kvmem_original_positions_ && id < repo.block_count(); ++id) {
            if (repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu ||
                !block_pages[id].valid()) { continue; }
            const auto columns = pages_->committed_columns(block_pages[id]);
            kvmem_source_k_->capture(sources[id], pool,
                pool.physical_index_of(pages_->physical(block_pages[id])),
                static_cast<std::int32_t>(repo.baked_pos_of(id)), columns, stream);
        }
        if (pinned_blocks > repo.block_count()) {
            throw std::logic_error("KVMem pinned prefix exceeds the block ledger");
        }
        if (pinned_blocks > budget_blocks) {
            throw std::invalid_argument(
                "KVMem compaction budget cannot hold the pinned shared prefix");
        }
        const std::uint32_t private_blocks = budget_blocks - pinned_blocks;
        if (!sink_only && private_blocks < kMinimumWindowPages) {
            throw std::invalid_argument(
                "KVMem window budget cannot retain the shared prefix and a private window: "
                "pinned_pages=" + std::to_string(pinned_blocks) +
                " budget_pages=" + std::to_string(budget_blocks) +
                " minimum_private_pages=" + std::to_string(kMinimumWindowPages));
        }

        // ---- plan through K0 ----
        ninfer::kvmem::KvSelectConfig cfg;
        cfg.block_tokens  = kPagedKVPageSize;
        cfg.budget_tokens = budget_tokens;
        cfg.action_lrfu   = kvmem_action_lrfu_;
        // Match the reference KVMem/ActKV immutable-source refresh bounds.
        cfg.raw_refresh_remaps = 8;
        cfg.raw_refresh_abs_delta = 262144;
        ninfer::kvmem::KvSelection sel;
        if (!coupled_selection.empty()) {
            // Main owns retrieval. MTP consumes the exact same logical history
            // at the same packed slots, while retaining its own typed storage.
            if (coupled_selection.size() > budget_blocks ||
                !std::is_sorted(coupled_selection.begin(), coupled_selection.end()) ||
                std::adjacent_find(coupled_selection.begin(), coupled_selection.end()) !=
                    coupled_selection.end() || coupled_selection.back() >= repo.block_count()) {
                throw std::invalid_argument("KVMem coupled selection has invalid geometry");
            }
            sel.block_ids.assign(coupled_selection.begin(), coupled_selection.end());
        } else if (sink_only) {
            // Round-boundary mode: keep the oldest head, evict the tail. The
            // selection is direct (bands cannot express "no recent"); the next
            // round re-prefills everything after the head, and K3 will replace
            // the lost ranges with query-driven retention.
            sel.block_ids.resize(budget_blocks);
            for (std::uint32_t i = 0; i < budget_blocks; ++i) { sel.block_ids[i] = i; }
        } else {
            if (budget_blocks < kMinimumWindowPages) {
                throw std::invalid_argument(
                    "KVMem compaction budget must span at least four pages "
                    "(sink/recent/middle)");
            }
            if (pinned_blocks != 0) {
                // The shared prefix already is the sink: it cannot move, so the private bands
                // only need one sink page and a recent tail. The old bands are computed from
                // the whole budget and would leave too little room for retrieval otherwise.
                cfg.sink_tokens   = kPagedKVPageSize;
                cfg.recent_tokens = std::max<std::uint32_t>(1, private_blocks / 4) *
                                    kPagedKVPageSize;
                std::vector<std::uint32_t> selection_mandatory;
                selection_mandatory.reserve(mandatory_ids.size() + pinned_blocks);
                for (const std::uint32_t id : mandatory_ids) {
                    if (id < repo.block_count()) { selection_mandatory.push_back(id); }
                }
                for (std::uint32_t id = 0; id < pinned_blocks; ++id) {
                    selection_mandatory.push_back(id);
                }
                mandatory_ids = std::move(selection_mandatory);
            } else if (budget_tokens < kAutoBandFloorTokens) {
                // Explicit bands: the auto clamps (sink 1024..2048, recent
                // 4096..16384) do not fit a budget below their floor, and the
                // selection refuses to plan rather than shrink them. Keep one
                // sink page, one recent page and let the quota fill the middle.
                cfg.sink_tokens  = std::clamp(budget_blocks / 8, 1u, budget_blocks / 4) *
                                  kPagedKVPageSize;
                cfg.recent_tokens =
                    std::max<std::uint32_t>(1, budget_blocks / 4) * kPagedKVPageSize;
            }
            kvmem_score_window(index, pool, stream);
            sel = repo.preview_select(cfg, mandatory_ids,
                                      kvmem_lexical_pages_[index],
                                      kvmem_older_matches_[index]); // K3 ranks the middle
        }
        if (pinned_blocks != 0) {
            for (std::uint32_t id = 0; id < pinned_blocks; ++id) {
                if (std::find(sel.block_ids.begin(), sel.block_ids.end(), id) ==
                    sel.block_ids.end()) {
                    throw std::logic_error("KVMem compaction dropped a pinned shared prefix block");
                }
            }
        }
        ninfer::kvmem::KvWindowPlan plan = repo.set_selection(sel, cfg);
        const auto replay_overwrites = [&](std::uint32_t id) {
            return replay_from && repo.baked_pos_of(id) >= static_cast<std::int64_t>(*replay_from);
        };
        const std::uint32_t window_tokens = static_cast<std::uint32_t>(plan.total_window_tokens);
        const std::uint32_t window_pages  = window_tokens / kPagedKVPageSize;
        if (plan.remaps.size() != window_pages) {
            throw std::logic_error("KVMem compaction plan geometry diverged");
        }
        if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            std::fprintf(stderr, "KVMEM_TRACE select window_tokens=%u ids=", window_tokens);
            for (const std::uint32_t id : sel.block_ids) { std::fprintf(stderr, "%u,", id); }
            std::fprintf(stderr, "\n");
        }
        // The packed window is a whole number of pages, but the newest retained page holds the most
        // recent tokens and only the columns up to the previous KV frontier are written -- unless
        // the selection stops at the head, where the last page is full. The window's KV frontier,
        // the position new tokens continue from, is that last valid column: the next write lands
        // in the same page, in the columns the page has not filled yet.
        const std::uint32_t issue_frontier = address.committed_frontier;
        std::uint32_t kv_window           = window_tokens;
        if (!plan.remaps.empty()) {
            const ninfer::kvmem::KvRemap& tail = plan.remaps.back();
            const auto logical = block_pages[tail.block_id];
            // Tracker placement can be older than a resumed address's tail
            // coverage. The logical page is authoritative for written columns;
            // declaring its unwritten tail committed makes checkpoint protection
            // fail after the answer has already been streamed.
            const std::uint32_t valid = logical.valid()
                ? pages_->committed_columns(logical)
                : (kvmem_original_positions_ ? kvmem_native_records_[index].at(tail.block_id).columns
                : (sources[tail.block_id].columns != 0
                       ? sources[tail.block_id].columns
                : (tail.from_base >= 0 &&
                           issue_frontier > static_cast<std::uint32_t>(tail.from_base)
                       ? std::min(issue_frontier - static_cast<std::uint32_t>(tail.from_base),
                                  tail.n_tokens)
                       : tail.n_tokens)));
            if (valid != 0) {
                const std::uint32_t packed = static_cast<std::uint32_t>(
                    tail.to_base + static_cast<std::int64_t>(valid));
                if (packed < window_tokens) { kv_window = packed; }
            }
            if (std::getenv("NINFER_KVMEM_TRACE")) {
                std::fprintf(stderr,
                             "KVMEM_TRACE tail_coverage id=%u from=%lld issue=%u valid=%u "
                             "window=%u\n", tail.block_id, static_cast<long long>(tail.from_base),
                             issue_frontier, valid, kv_window);
            }
        }
        if (pages_for_tokens(kv_window) != window_pages) {
            throw std::logic_error("KVMem compaction window frontier diverged from its geometry");
        }
        // The window is packed, not prefix-contiguous, so it holds a head run followed (after a
        // possible gap) by a tail run. Every retained token lies below the tail run's first
        // ledger, so a resume point at or above its first slot still names only tokens the state
        // already saw. Below that the window would expose the resumed request to its own future,
        // and readers use this floor to reject such a checkpoint instead of silently mis-reading.
        std::uint32_t reuse_floor_slots = window_tokens;
        if (!plan.remaps.empty()) {
            std::size_t run = plan.remaps.size();
            while (run > 1 &&
                   plan.remaps[run - 1].from_base ==
                       plan.remaps[run - 2].from_base +
                           static_cast<std::int64_t>(plan.remaps[run - 2].n_tokens)) {
                --run;
            }
            if (run != 0) {
                reuse_floor_slots = static_cast<std::uint32_t>(plan.remaps[run].to_base);
            }
        }

        ensure_kvmem_host_arena(pool);

        // ---- 1) stage out: park evicted pages in the host arena ----
        // One copy per evicted block used to cost one cudaMemcpy2DAsync per plane per page
        // (64 planes on the 27B text cache), the same per-page fragmentation melis2023 measured
        // at 26% link utilisation (ninfer-kvmem-actkv b6c8337, docs/kvmem-recall-opt-20261019.md
        // section 8). Instead, order the evicted pages by physical index, park them in ONE
        // contiguous host extent split back into per-block slots (each block still releases on
        // its own), and issue a single gather whose device-consecutive stretches move as one
        // strided copy per plane. A fragmented arena falls back to per-page slots, which still
        // merge wherever they happen to sit back to back.
        const auto transfer_start = std::chrono::steady_clock::now();
        std::uint64_t transfer_runs = 0;
        std::unique_ptr<PendingDemote> async_demote;
        struct ColdRead {
            std::uint32_t id;
            ninfer::kvmem::ColdKVStore::Handle source;
            HostKVAllocation buffer;
            // Destroy future first, including unwinding: never free bytes still used by I/O.
            std::future<void> ready;
        };
        std::vector<std::uint32_t> cold_read_ids;
        std::vector<ColdRead> cold_reads;
        std::size_t next_cold_read = 0;
        const auto start_cold_read = [&] {
            if (next_cold_read == cold_read_ids.size()) { return; }
            // Two read owners, each with a distinct pinned allocation. Leave
            // the third Cold staging page available for synchronous spill.
            if (cold_reads.size() >= 2) {
                throw std::logic_error("KVMem Cold read-ahead slots are still owned");
            }
            const auto id = cold_read_ids[next_cold_read++];
            const auto source = kvmem_block_cold_[index][id];
            auto buffer = kvmem_cold_buffer();
            auto view = kvmem_cold_staging_->writable_view(buffer);
            auto future = kvmem_cold_file_->prefetch(source,
                std::span<std::byte>(view.data(), kvmem_host_layout_.page_stride));
            cold_reads.push_back({id, source, std::move(buffer), std::move(future)});
            ++kvmem_nvme_prefetch_submitted_;
        };
        if (kvmem_async_device_) {
            for (auto id : plan.stage_in) {
                if (!replay_overwrites(id) && id < kvmem_block_cold_[index].size() &&
                    kvmem_block_cold_[index][id].valid()) {
                    cold_read_ids.push_back(id);
                }
            }
            cold_reads.reserve(2);
            start_cold_read(); start_cold_read();
        }

        if (!plan.stage_out.empty()) {
            struct Evicted {
                std::int32_t physical = 0;
                std::uint32_t id      = 0;
                DeviceKVPageHandle handle;
            };
            std::vector<Evicted> evicted;
            evicted.reserve(plan.stage_out.size());
            for (const std::uint32_t id : plan.stage_out) {
                if (block_pages.size() <= id) { block_pages.resize(id + 1); }
                if (block_hosts.size() <= id) { block_hosts.resize(id + 1); }
                const DeviceKVPageHandle handle = pages_->physical(block_pages[id]);
                evicted.push_back(Evicted{.physical = pool.physical_index_of(handle),
                                          .id       = id,
                                          .handle   = handle});
            }
            std::sort(evicted.begin(), evicted.end(),
                      [](const Evicted& a, const Evicted& b) { return a.physical < b.physical; });
            const auto count = static_cast<std::uint32_t>(evicted.size());
            std::vector<HostKVAllocation> slots(count);
            std::optional<HostKVAllocation> extent;
            if (static_cast<std::size_t>(count) <=
                kvmem_host_arena_->capacity_bytes() / kvmem_host_layout_.page_stride) {
                extent = kvmem_host_arena_->allocate(kvmem_host_layout_, count);
                while (!extent && kvmem_spill_one_host(index, plan.stage_in)) {
                    extent = kvmem_host_arena_->allocate(kvmem_host_layout_, count);
                }
            }
            if (extent) {
                HostKVAllocation rest = std::move(*extent);
                for (std::uint32_t i = 0; i + 1 < count; ++i) {
                    auto [head, tail] = kvmem_host_arena_->split(std::move(rest), 1);
                    slots[i]          = std::move(head);
                    rest              = std::move(tail);
                }
                slots[count - 1] = std::move(rest);
            } else {
                // When all new pages cannot fit, keep the protected/newer
                // context in Host first, and send the remainder directly to
                // cold storage through the one-page staging buffer.
                const auto host_priority = [&](std::uint32_t id) {
                    switch (repo.blocks()[id].stratum) {
                    case ninfer::kvmem::BlockStratum::PinnedSink: return 0;
                    case ninfer::kvmem::BlockStratum::ActionProtected: return 1;
                    case ninfer::kvmem::BlockStratum::StandardContext: return 2;
                    case ninfer::kvmem::BlockStratum::EphemeralThinking: return 3;
                    }
                    return 3;
                };
                std::stable_sort(evicted.begin(), evicted.end(), [&](const Evicted& a,
                                                                     const Evicted& b) {
                    return host_priority(a.id) < host_priority(b.id);
                });
                for (std::uint32_t i = 0; i < count; ++i) {
                    std::optional<HostKVAllocation> slot =
                        kvmem_host_arena_->allocate(kvmem_host_layout_, 1);
                    while (!slot && kvmem_spill_one_host(index, plan.stage_in)) {
                        slot = kvmem_host_arena_->allocate(kvmem_host_layout_, 1);
                    }
                    if (slot) { slots[i] = std::move(*slot); }
                }
            }
            std::vector<DeviceKVPageHandle> handles;
            std::vector<HostKVAllocationView> views;
            handles.reserve(count);
            views.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                if (!slots[i].valid()) { continue; }
                handles.push_back(evicted[i].handle);
                views.push_back(kvmem_host_arena_->writable_view(slots[i]));
            }
            bool async_eligible = kvmem_async_device_ != nullptr && stream != nullptr &&
                                  plan.stage_in.empty() &&
                                  handles.size() == evicted.size() &&
                                  kvmem_host_layout_.page_stride <= kAsyncBatchBytes;
            for (const auto& item : evicted) {
                const auto logical = block_pages[item.id];
                async_eligible = async_eligible &&
                    pages_->committed_columns(logical) == kPagedKVPageSize &&
                    pages_->source_pins(logical) == 0;
            }
            if (!async_eligible && kvmem_async_device_ != nullptr) {
                ++kvmem_async_fallbacks_;
            }
            if (async_eligible) {
                const auto batch_pages = kAsyncBatchBytes / kvmem_host_layout_.page_stride;
                for (std::size_t begin = 0; begin < handles.size(); begin += batch_pages) {
                    const auto size = std::min(batch_pages, handles.size() - begin);
                    auto batch = std::make_unique<PendingDemote>(
                        *pages_, *kvmem_async_device_, handle);
                    batch->pages.reserve(size);
                    for (std::size_t j = begin; j < begin + size; ++j) {
                        const auto logical = block_pages[evicted[j].id];
                        batch->pages.push_back({evicted[j].id, logical,
                            pages_->content_epoch(logical), pages_->committed_columns(logical),
                            std::move(slots[j]), false});
                        pages_->pin_source(logical);
                        batch->pages.back().pinned = true;
                    }
                    batch->source_ready.record(stream);
                    batch->source_ready.wait(batch->stream);
                    batch->submitted = true;
                    transfer_runs += pool.copy_to_host_pages(
                        std::span(handles).subspan(begin, size),
                        std::span(views).subspan(begin, size), batch->stream);
                    batch->completion.record(batch->stream);
                    batch->recorded = true;
                    ++kvmem_async_submitted_;
                    kvmem_async_bytes_ += size * kvmem_host_layout_.page_stride;
                    if (transfer_pipeline_enabled()) {
                        // Adapt Sync/Async ledger46 multibatch: submit the next
                        // bounded batch BEFORE waiting on the previous one. At most
                        // two 16MiB batches are in flight. Claimed pages remain pinned
                        // and never count as free until their completion retires.
                        if (async_demote) {
                            const auto wait_start = std::chrono::steady_clock::now();
                            async_demote->completion.synchronize();
                            kvmem_async_wait_us_ += static_cast<std::uint64_t>(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - wait_start).count());
                            for (auto& page : async_demote->pages) {
                                block_hosts[page.block] = std::move(page.destination);
                            }
                            ++kvmem_async_completed_;
                            ++kvmem_async_split_batches_;
                        }
                        async_demote = std::move(batch);
                    } else if (begin + size == handles.size()) {
                        async_demote = std::move(batch);
                    } else {
                        // At most 16MiB remains queued/pinned at any time. Complete
                        // earlier sub-batches; only the tail overlaps subsequent compute.
                        const auto wait_start = std::chrono::steady_clock::now();
                        batch->completion.synchronize();
                        kvmem_async_wait_us_ += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - wait_start).count());
                        for (auto& page : batch->pages) {
                            block_hosts[page.block] = std::move(page.destination);
                        }
                        ++kvmem_async_completed_;
                        ++kvmem_async_split_batches_;
                    }
                }
            } else if (!handles.empty()) {
                transfer_runs += pool.copy_to_host_pages(handles, views, stream);
            }
            auto& cold_records = kvmem_block_cold_[index];
            if (cold_records.size() < repo.block_count()) {
                cold_records.resize(repo.block_count());
            }
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint32_t id = evicted[i].id;
                if (cold_records[id].valid()) {
                    throw std::logic_error("KVMem evicted block already has NVMe backing");
                }
                if (async_demote && std::any_of(async_demote->pages.begin(), async_demote->pages.end(),
                        [id](const auto& page) { return page.block == id; })) {
                    continue; // This tail sub-batch remains unpublished until retirement.
                } else if (block_hosts[id].valid()) {
                    continue; // An earlier sub-batch is already complete.
                } else if (slots[i].valid()) {
                    block_hosts[id] = std::move(slots[i]);
                } else {
                    cold_records[id] =
                        kvmem_archive_device(pool, evicted[i].handle, stream, transfer_runs);
                }
            }
        }

        // ---- 1b) give the evicted pages' device storage back BEFORE anything reserves ----
        // The window's pages are all device-resident when compaction is called on a full pool, so
        // a stage-in reservation taken first would find nothing free. The parked Host copy owns
        // the block from here on, and a selected block never appears in stage_out.
        for (const std::uint32_t id : plan.stage_out) {
            if (async_demote && std::any_of(async_demote->pages.begin(), async_demote->pages.end(),
                    [id](const auto& page) { return page.block == id; })) { continue; }
            const LogicalKVPageHandle logical = block_pages[id];
            // The block leaves the address's window on this step, so its active reference goes
            // with it; without that the page keeps a reference and its device storage is never
            // returned to the pool.
            if (pages_->active_address_references(logical) != 0) {
                pages_->release_active_reference(logical);
            }
            (void)pages_->release_reference(logical, false);
            block_pages[id] = LogicalKVPageHandle{};
        }

        // pass 1 is independent resident work; pass 2 consumes completed restores.
        const auto rephase = [&](int pass) {
            if (kvmem_original_positions_ || replay_from) { return; }
            // Source and resident relocations must enter the same layer gate;
            // defer the resident half until async Host stage-in has retired.
            if (layer_ready && pass == 1) { return; }
            if (!plan.remaps.empty()) {
                std::vector<std::int32_t> rerope_pages;
                std::vector<std::int32_t> rerope_from;
                std::vector<std::int32_t> rerope_to;
                std::vector<ninfer::kvmem::ImmutableKStore::RestoreRequest> source_restores;
                source_restores.reserve(plan.remaps.size());
                rerope_pages.reserve(plan.remaps.size());
                rerope_from.reserve(plan.remaps.size());
                rerope_to.reserve(plan.remaps.size());
                for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
                    const bool restored = std::find(plan.stage_in.begin(), plan.stage_in.end(),
                                                    rm.block_id) != plan.stage_in.end();
                    if (rm.skip || (!layer_ready &&
                        ((pass == 1 && restored) || (pass == 2 && !restored)))) { continue; }
                    if (rm.raw_refresh) {
                        source_restores.push_back({&sources[rm.block_id],
                            pool.physical_index_of(pages_->physical(block_pages[rm.block_id])),
                            static_cast<std::int32_t>(rm.to_base)});
                        continue;
                    }
                    if (block_pages.size() <= rm.block_id) { block_pages.resize(rm.block_id + 1); }
                    rerope_pages.push_back(
                        pool.physical_index_of(pages_->physical(block_pages[rm.block_id])));
                    rerope_from.push_back(static_cast<std::int32_t>(rm.from_base));
                    rerope_to.push_back(static_cast<std::int32_t>(rm.to_base));
                }
                // Two existing pinned source-K read slots are sufficient for a
                // truly async per-layer H2D+K rebuild. Larger batches retain
                // the proven sync source path, never an unbounded staging pool.
                const bool layer_sources = layer_ready && !source_restores.empty() &&
                    source_restores.size() <= 2;
                if (!source_restores.empty() && !layer_sources) {
                    kvmem_source_k_->restore_many(source_restores, pool, stream);
                }
                const bool layer_cold = layer_ready && cold_layer_count_ != 0;
                if (!rerope_pages.empty() || layer_sources || layer_cold) {
                    const ninfer::kvmem::KvmemReropeWorkspace workspace{
                        {static_cast<std::int32_t*>(kvmem_rerope_host_->data()),
                         static_cast<std::size_t>(page_capacity_) * 3},
                        {static_cast<std::int32_t*>(kvmem_rerope_device_.data),
                         static_cast<std::size_t>(page_capacity_) * 3}};
                    if (layer_ready) {
                        if (!kvmem_layer_ready_) {
                            kvmem_layer_ready_ = std::make_unique<ninfer::kvmem::LayerReadyFence>();
                        }
                        auto& ticket = *kvmem_layer_ready_;
                        ticket.begin(stream);
                        std::array<cudaEvent_t, ninfer::kvmem::LayerReadyFence::kLayers> events;
                        for (std::uint32_t layer = 0; layer < events.size(); ++layer) {
                            events[layer] = ticket.ready_event(layer);
                        }
                        try {
                            struct SourceCallback {
                                KVAddressSpaceStore* owner;
                                ninfer::kvmem::ImmutableKStore* source;
                                DeviceKVPagePool* pool;
                                bool layer_sources;
                                bool layer_cold;
                            } callback{this, kvmem_source_k_.get(), &pool,
                                       layer_sources, layer_cold};
                            if (layer_sources) {
                                callback.source->prepare_layer_sources(
                                    source_restores, pool, ticket.producer_stream());
                            }
                            if (layer_cold) {
                                cold_layer_producer_stream_ = ticket.producer_stream();
                            }
                            const ninfer::kvmem::KvmemLayerEnqueue enqueue_source =
                                (layer_sources || layer_cold) ? +[](void* owner, std::uint32_t layer,
                                                     cudaStream_t producer) {
                                    auto& item = *static_cast<SourceCallback*>(owner);
                                    // Full Host/Cold K/V arrives before raw immutable
                                    // K overwrites the selected layer's K codes/scales.
                                    if (item.layer_cold) {
                                        item.owner->enqueue_cold_layer(layer, *item.pool, producer);
                                    }
                                    if (item.layer_sources) {
                                        item.source->enqueue_layer_source(layer, *item.pool, producer);
                                    }
                                } : nullptr;
                            ninfer::kvmem::kvmem_rerope_pages_layer_ready(
                                pool, rerope_pages.data(), rerope_from.data(), rerope_to.data(),
                                static_cast<std::uint32_t>(rerope_pages.size()), workspace,
                                events, ticket.producer_stream(), enqueue_source,
                                (layer_sources || layer_cold) ? &callback : nullptr);
                            for (std::uint32_t layer = 0; layer < events.size(); ++layer) {
                                ticket.mark_ready(layer);
                            }
                            kvmem_layer_ready_address_ = handle;
                            if (const char* trace = std::getenv("NINFER_KVMEM_LAYER_READY_TRACE");
                                trace && std::strcmp(trace, "1") == 0) {
                                std::fprintf(stderr,
                                             "KVMEM_LAYER_READY epoch=%llu layers=%zu remap_pages=%zu "
                                             "source_restores=%zu source_layer_async_pages=%zu "
                                             "source_sync_pages=%zu cold_layer_async_pages=%zu "
                                             "producer=%s\n",
                                             static_cast<unsigned long long>(ticket.epoch()),
                                             events.size(), rerope_pages.size(), source_restores.size(),
                                             layer_sources ? source_restores.size() : 0UL,
                                             layer_sources ? 0UL : source_restores.size(),
                                             layer_cold ? cold_layer_count_ : 0UL,
                                             layer_cold ? "cold_h2d+source+resident_rephase"
                                                 : layer_sources ? "source_h2d_rebuild+resident_rephase"
                                                                 : "resident_rephase");
                            }
                        } catch (...) {
                            ticket.retire();
                            if (layer_sources) { kvmem_source_k_->drain_layer_source(); }
                            if (layer_cold) { drain_cold_layer_buffers(); }
                            throw;
                        }
                    } else {
                        ninfer::kvmem::kvmem_rerope_pages(
                            pool, rerope_pages.data(), rerope_from.data(), rerope_to.data(),
                            static_cast<std::uint32_t>(rerope_pages.size()), workspace, stream);
                    }
                }
            }

        };

        bool restore_was_async = false;
        std::uint64_t restore_wait_us = 0;
        // ---- 2) stage in: fresh pages for cold selected blocks ----
        // Materializing in ascending parked-host-address order pairs adjacent host slots (such as
        // blocks evicted together in an earlier compaction) with consecutive physical device pages
        // from free_page_runs_, so plan_kv_transfer_runs merges them into single H2D runs.
        if (!plan.stage_in.empty()) {
            const std::uint32_t required = static_cast<std::uint32_t>(plan.stage_in.size());
            std::optional<DeviceKVPageReservation> temporary_reservation;
            DeviceKVPageReservation* stage_in_reservation = nullptr;
            if (address.active) {
                // The active writer has already claimed its growth headroom in
                // address.reservation. Physical-pool available_pages() excludes
                // those pages, so taking an independent stage-in reservation can
                // fail even when the required capacity belongs to this address.
                // Consume its own entitlement; any future growth re-establishes
                // the remaining budget through resize_entitlement().
                if (address.reservation.pages() < required) {
                    pool.resize_reservation(address.reservation, required);
                }
                stage_in_reservation = &address.reservation;
            } else {
                temporary_reservation = pool.reserve(required);
                if (!temporary_reservation) {
                    throw std::runtime_error("KVMem stage-in found no free device page");
                }
                stage_in_reservation = &*temporary_reservation;
            }
            std::vector<std::uint32_t> ordered = plan.stage_in;
            for (const std::uint32_t id : ordered) {
                if (block_pages.size() <= id) { block_pages.resize(id + 1); }
                if (block_hosts.size() <= id) { block_hosts.resize(id + 1); }
            }
            // Reserve the usual bounded physical pages, but leave overwritten
            // destinations uncommitted. Host/Cold/source owners stay alive
            // until begin_kvmem_window_replay revokes them. No H2D/pread here.
            std::erase_if(ordered, [&](std::uint32_t id) {
                if (!replay_overwrites(id)) { return false; }
                const auto logical = pages_->materialize(*stage_in_reservation);
                block_pages[id] = logical;
                if (address.active) { pages_->retain_active_reference(logical); }
                return true;
            });
            std::stable_sort(ordered.begin(), ordered.end(),
                             [&](std::uint32_t a, std::uint32_t b) {
                                 const bool va = block_hosts[a].valid();
                                 const bool vb = block_hosts[b].valid();
                                 if (va != vb) { return va > vb; }
                                 if (!va) { return false; }
                                 return kvmem_host_arena_->view(block_hosts[a]).data() <
                                        kvmem_host_arena_->view(block_hosts[b]).data();
                             });
            std::vector<HostKVAllocationConstView> views;
            std::vector<DeviceKVPageHandle> handles;
            std::vector<std::uint32_t> restored_ids;
            std::vector<std::pair<std::uint32_t, DeviceKVPageHandle>> cold_restores;
            views.reserve(ordered.size());
            handles.reserve(ordered.size());
            restored_ids.reserve(ordered.size());
            const bool async_restore = !ordered.empty() && kvmem_async_device_ != nullptr && stream != nullptr &&
                kvmem_host_layout_.page_stride <= kAsyncBatchBytes &&
                std::all_of(ordered.begin(), ordered.end(), [&](std::uint32_t id) {
                    return block_hosts[id].valid() && (kvmem_original_positions_
                        ? kvmem_native_records_[index].at(id).columns : sources[id].columns)
                            == kPagedKVPageSize;
                });
            std::unique_ptr<PendingRestore> restore;
            if (async_restore) {
                restore = std::make_unique<PendingRestore>(
                    *pages_, *stage_in_reservation, block_hosts, *kvmem_async_device_);
                restore->pages.reserve(required);
            }
            for (const std::uint32_t id : ordered) {
                LogicalKVPageHandle logical = async_restore
                    ? pages_->materialize_transfer_destination(*stage_in_reservation, kPagedKVPageSize)
                    : pages_->materialize(*stage_in_reservation);
                if (restore) {
                    restore->pages.push_back({id, logical, std::move(block_hosts[id])});
                    views.push_back(kvmem_host_arena_->view(restore->pages.back().source));
                    handles.push_back(pages_->physical(logical));
                } else {
                    block_pages[id] = logical;
                    if (block_hosts[id].valid()) {
                        views.push_back(kvmem_host_arena_->view(block_hosts[id]));
                        handles.push_back(pages_->physical(logical));
                        restored_ids.push_back(id);
                    } else if (kvmem_block_cold_[index].size() > id &&
                               kvmem_block_cold_[index][id].valid()) {
                        cold_restores.emplace_back(id, pages_->physical(logical));
                    } else {
                        throw std::logic_error("KVMem selected block has no valid Host/Cold source");
                    }
                    pages_->commit_coverage(logical, kvmem_original_positions_
                        ? kvmem_native_records_[index].at(id).columns : sources[id].columns);
                    if (address.active) { pages_->retain_active_reference(logical); }
                }
            }
            if (restore) {
                const auto submit_start = std::chrono::steady_clock::now();
                // Some targets may reuse pages whose D2H was queued above on
                // compute. Do not overwrite them until those reads have ended.
                restore->source_ready.record(stream);
                restore->source_ready.wait(restore->stream);
                restore->submitted = true;
                const auto batch_pages = kAsyncBatchBytes / kvmem_host_layout_.page_stride;
                for (std::size_t begin = 0; begin < handles.size(); begin += batch_pages) {
                    const auto size = std::min(batch_pages, handles.size() - begin);
                    // A partial later submission must drain the stream, not
                    // mistake the previous sub-batch's event for its completion.
                    restore->recorded = false;
                    transfer_runs += pool.copy_from_host_pages(
                        std::span(views).subspan(begin, size),
                        std::span(handles).subspan(begin, size), restore->stream);
                    restore->completion.record(restore->stream);
                    restore->recorded = true;
                    if (begin + size < handles.size()) {
                        const auto wait_start = std::chrono::steady_clock::now();
                        restore->completion.synchronize();
                        kvmem_async_wait_us_ += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - wait_start).count());
                        ++kvmem_async_submitted_;
                        ++kvmem_async_completed_;
                        ++kvmem_async_split_batches_;
                    }
                }
                kvmem_async_submit_us_ += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - submit_start).count());
                ++kvmem_async_submitted_;
                kvmem_async_bytes_ += required * kvmem_host_layout_.page_stride;
                ++kvmem_async_restore_submitted_;
                // These pages are disjoint from every H2D target. Their rephase
                // can run while the copy stream restores the selected history.
                rephase(1);
                const bool ready = restore->completion.ready();
                const auto wait_start = std::chrono::steady_clock::now();
                restore->completion.synchronize();
                if (!ready) {
                    restore_wait_us = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - wait_start).count());
                    kvmem_async_wait_us_ += restore_wait_us;
                }
                for (auto& page : restore->pages) {
                    pages_->publish_transfer_destination(page.logical, true);
                    if (address.active) { pages_->retain_active_reference(page.logical); }
                    block_pages[page.block] = page.logical;
                    page.source.release();
                }
                restore->pages.clear();
                ++kvmem_async_completed_;
                restore_was_async = true;
            } else {
                if (kvmem_async_device_ != nullptr) { ++kvmem_async_fallbacks_; }
                if (!views.empty()) {
                    transfer_runs += pool.copy_from_host_pages(views, handles, stream);
                }
                if (transfer_runs != 0) { KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream)); }
                for (const std::uint32_t id : restored_ids) { block_hosts[id].release(); }
            }
            // The existing three-page Cold arena has two read-ahead slots and
            // one spill/restore reserve. Only one/two Cold restore pages can
            // stay pinned through the forward producer; larger batches retain
            // the original, already-drained per-page H2D path.
            const bool layer_cold_stage = layer_ready && !cold_restores.empty() &&
                cold_restores.size() <= cold_layer_slots_.size();
            for (const auto [id, device] : cold_restores) {
                HostKVAllocation buffer;
                const auto record = kvmem_block_cold_[index][id];
                auto read = std::find_if(cold_reads.begin(), cold_reads.end(),
                                        [id](const auto& item) { return item.id == id; });
                if (read != cold_reads.end()) {
                    if (read->source != record || !kvmem_cold_file_->valid(read->source)) {
                        throw std::logic_error("KVMem Cold read-ahead source generation changed");
                    }
                    const auto wait_start = std::chrono::steady_clock::now();
                    read->ready.get();
                    // Never publish bytes from a worker whose source was
                    // released/replaced before the owner regained control.
                    if (read->source != kvmem_block_cold_[index][id] ||
                        !kvmem_cold_file_->valid(read->source)) {
                        throw std::logic_error("KVMem Cold read-ahead owner changed");
                    }
                    kvmem_nvme_prefetch_wait_us_ += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - wait_start).count());
                    buffer = std::move(read->buffer);
                    cold_reads.erase(read);
                    ++kvmem_nvme_prefetch_completed_;
                } else {
                    buffer = kvmem_cold_buffer();
                    const auto writable = kvmem_cold_staging_->writable_view(buffer);
                    kvmem_cold_file_->load(record,
                        std::span<std::byte>(writable.data(), kvmem_host_layout_.page_stride));
                }
                if (layer_cold_stage) {
                    auto& slot = cold_layer_slots_.at(cold_layer_count_++);
                    slot.staging = std::move(buffer);
                    slot.block_id = id;
                    slot.physical = pool.physical_index_of(device);
                } else {
                    const std::array<HostKVAllocationConstView, 1> source{
                        kvmem_cold_staging_->view(buffer)};
                    const std::array<DeviceKVPageHandle, 1> destination{device};
                    transfer_runs += pool.copy_from_host_pages(source, destination, stream);
                    kvmem_cold_sync(stream);
                }
                (void)kvmem_cold_file_->release(record);
                kvmem_block_cold_[index][id] = {};
                buffer.release();
                if (!layer_cold_stage) { start_cold_read(); }
            }
        } else if (transfer_runs != 0 && !async_demote) {
            KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        const auto transfer_us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - transfer_start)
                .count());

        // ---- 3) finish relocation before publishing any restored page ----
        rephase(restore_was_async ? 2 : 0);
        if (replay_from && std::getenv("NINFER_KVMEM_PREFIX_TRACE")) {
            const auto skipped = std::count_if(plan.stage_in.begin(), plan.stage_in.end(), replay_overwrites);
            std::fprintf(stderr, "KVMEM_REPLAY_PREPARE from=%u stage_in_skipped=%zu rephase_skipped=%zu\n",
                         *replay_from, static_cast<std::size_t>(skipped), plan.remaps.size());
        }

        // ---- 4) rewire the window and shrink the address ----
        const std::uint32_t old_page_count = address.page_count;
        std::uint32_t slot = 0;
        for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
            if (block_pages.size() <= rm.block_id) { block_pages.resize(rm.block_id + 1); }
            membership(address, slot) = block_pages[rm.block_id];
            ++slot;
        }
        // Clear the membership slots past the window: they leave the address for good, and a
        // stale handle would fail the next growth batch's already-populated guard.
        for (std::uint32_t page = window_pages; page < old_page_count; ++page) {
            membership(address, page) = {};
        }
        address.page_count         = window_pages;
        address.committed_frontier = kv_window;
        if (async_demote) {
            // The membership rewrite transfers these refs to the batch. Mark
            // that ownership before the potentially failing table upload so
            // error cleanup cannot strand a removed source reference.
            for (const auto& page : async_demote->pages) {
                if (pages_->active_address_references(page.logical) != 0) {
                    pages_->release_active_reference(page.logical);
                }
                block_pages[page.block] = {};
            }
            async_demote->detached = true;
        }
        if (address.active) { publish_membership(address, stream); }
        if (async_demote) {
            kvmem_pending_demote_ = std::move(async_demote);
        }

        // ---- 5) ledger tiers ----
        // The tracker keeps its ledger order: evicted blocks retain Host or NVMe copies,
        // so a later round can select one again and stage it back in. Only their
        // window placement goes stale, and a destructive truncate drops the ones that leave the
        // window for good.
        for (const std::uint32_t id : plan.stage_out) {
            const bool archived = kvmem_block_cold_[index].size() > id &&
                                  kvmem_block_cold_[index][id].valid();
            repo.set_tier(id, archived ? ninfer::kvmem::KvTier::Nvme
                                      : ninfer::kvmem::KvTier::Host);
        }
        for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
            repo.set_tier(rm.block_id, ninfer::kvmem::KvTier::Gpu);
        }
        kvmem_reuse_floors_[index] = reuse_floor_slots;
        kvmem_stage_out_blocks_ += plan.stage_out.size();
        kvmem_stage_in_blocks_ += plan.stage_in.size();
        kvmem_transfer_runs_ += transfer_runs;
        if (kvmem_pending_demote_) {
            kvmem_async_submit_us_ += transfer_us;
        } else if (restore_was_async) {
            kvmem_async_restore_us_ += transfer_us;
        } else {
            kvmem_transfer_us_ += transfer_us;
        }
        kvmem_window_tokens_ = kv_window;
        ++kvmem_compactions_;
        if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            std::size_t cold_blocks = 0;
            std::size_t cold_thinking = 0;
            for (const auto& block : repo.blocks()) {
                if (block.tier != ninfer::kvmem::KvTier::Nvme) { continue; }
                ++cold_blocks;
                if (block.stratum == ninfer::kvmem::BlockStratum::EphemeralThinking) {
                    ++cold_thinking;
                }
            }
            std::fprintf(stderr,
                         "KVMEM_TRACE cold bytes=%zu blocks=%zu closed_thinking=%zu "
                         "host_capacity=%zu spills=%llu\n",
                         kvmem_cold_bytes(), cold_blocks, cold_thinking,
                         kvmem_host_capacity_bytes(),
                         static_cast<unsigned long long>(kvmem_host_to_cold_spills_));
            std::fprintf(stderr, "KVMEM_TRACE source_k bytes=%zu captures=%llu rebuilds=%llu\n",
                         kvmem_source_k_ ? kvmem_source_k_->occupied_bytes() : std::size_t{0},
                         static_cast<unsigned long long>(kvmem_source_k_ ? kvmem_source_k_->captures() : 0),
                         static_cast<unsigned long long>(kvmem_source_k_ ? kvmem_source_k_->restores() : 0));
        }
        if (const char* env = std::getenv("KVMEM_RESELECT"); env != nullptr && env[0] != '\0' &&
            env[0] != '0') {
            std::fprintf(stderr,
                         "KVMEM_RESELECT blocks=%u stage_out=%zu stage_in=%zu runs=%llu "
                         "transfer_us=%llu window=%u\n",
                         repo.block_count(), plan.stage_out.size(), plan.stage_in.size(),
                         static_cast<unsigned long long>(transfer_runs),
                         static_cast<unsigned long long>(transfer_us), kv_window);
        }
        compaction_complete = true;
        return kv_window;
    }


    // Completion-time convenience: sink + recent compaction under the configured
    // budget (NINFER_KVMEM_BUDGET tokens). The kernel keeps the oldest sink pages
    // and the newest pages and drops the middle, refolding the retained tail onto
    // compact slots. Returns the new window token count, or 0 when compaction did
    // not run (disabled, fits, active or shared). The caller keeps its token
    // ledger and re-bases the KV slot arithmetic on the returned window.
    std::uint32_t kvmem_compact_round_boundary(KVAddressSpaceHandle handle,
                                               cudaStream_t stream = nullptr) {

        Address& address = require(handle);
        auto& repo = kvmem_repos_[static_cast<std::size_t>(&address - addresses_.data())];
        const std::uint64_t ledger_tokens =
            static_cast<std::uint64_t>(repo.block_count()) * kPagedKVPageSize;
        if (ledger_tokens <= kvmem_budget_) { return 0; }
        const std::uint32_t pinned_prefix_pages = kvmem_shared_prefix_pages(handle);
        return kvmem_compact(handle, kvmem_budget_, stream, {}, false, pinned_prefix_pages);
    }

    [[nodiscard]] std::uint32_t kvmem_budget_tokens() const noexcept { return kvmem_budget_; }

    [[nodiscard]] std::size_t kvmem_cold_bytes() const noexcept {
        return kvmem_cold_file_ ? kvmem_cold_file_->occupied_bytes() : 0;
    }

    [[nodiscard]] std::size_t kvmem_host_capacity_bytes() const noexcept {
        return kvmem_host_arena_ ? kvmem_host_arena_->capacity_bytes() : 0;
    }

    [[nodiscard]] std::uint64_t kvmem_host_to_cold_spills() const noexcept {
        return kvmem_host_to_cold_spills_;
    }

    // KVMem: the pool room generation may not consume, in tokens. Zero means a decode round has
    // no dedicated headroom, so a window that reaches the pool simply stops there.
    [[nodiscard]] std::uint32_t kvmem_gen_reserve_tokens() const noexcept {
        return kvmem_gen_reserve_;
    }

    // KVMem: make room for the tokens a decode round keeps appending. The window is compacted
    // back to `pool - generation headroom`, which parks the blocks that leave the working set in
    // the host tier and hands their device pages back to the pool -- the same window compaction
    // the prefill side runs, under the budget generation reserves for itself. Returns the new
    // window token count, or 0 when nothing needed compacting (disabled, already inside the
    // budget, or a page of the window is shared).
    std::uint32_t kvmem_compact_generation(KVAddressSpaceHandle handle,
                                           cudaStream_t stream = nullptr) {
        if (!kvmem_enabled_ || kvmem_gen_reserve_ == 0) { return 0; }
        const std::uint32_t pool_pages   = pages_->physical_pool().capacity_pages();
        // The window budget, not the pool, bounds generation: a lane is charged its window
        // at admission, so letting decode grow back to the pool would overrun that claim.
        const std::uint32_t window_pages = std::min(
            pool_pages, kvmem_budget_ / static_cast<std::uint32_t>(kPagedKVPageSize));
        const std::uint32_t reserve_pages =
            kvmem_gen_reserve_ / static_cast<std::uint32_t>(kPagedKVPageSize);
        if (reserve_pages + kMinimumWindowPages > window_pages) { return 0; }
        const std::uint32_t budget_tokens =
            (window_pages - reserve_pages) * static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t pinned_prefix_pages = kvmem_shared_prefix_pages(handle);
        return kvmem_compact(handle, budget_tokens, stream, {}, false, pinned_prefix_pages);
    }

    // KVMem K1b: the lowest window slot a compacted address may be resumed from, in the slots of
    // the current window. Zero means the address was never compacted. A resume point below it
    // would let the retained tail expose the request to tokens its state has not seen.
    [[nodiscard]] std::uint32_t kvmem_reuse_floor(KVAddressSpaceHandle handle) const noexcept {
        if (!kvmem_enabled_ || !valid(handle)) { return 0; }
        const std::size_t index = static_cast<std::size_t>(&addresses_[handle.index_] -
                                                          addresses_.data());
        return index < kvmem_reuse_floors_.size() ? kvmem_reuse_floors_[index] : 0U;
    }

    // KVMem K1b: the window a completion-time compaction would retain, or 0 when
    // this request needs no compaction (disabled, no budget, or the KV is already
    // inside the budget).
    [[nodiscard]] std::uint32_t kvmem_window_edge(std::uint64_t kv_tokens) const noexcept {
        if (!kvmem_enabled_ || kvmem_budget_ == 0) { return 0; }
        return kvmem_budget_ < kv_tokens ? kvmem_budget_ : 0U;
    }

    // Per-address, short-query lexical fallback. Never shared across request
    // addresses or retained after address release. The selector admits these
    // only after sink, mandatory, action and recent pages have been reserved.
    void set_kvmem_lexical_pages(KVAddressSpaceHandle handle,
                                 std::span<const std::uint32_t> pages,
                                 std::span<const std::uint32_t> older_matches = {}) {
        if (!kvmem_enabled_) { return; }
        Address& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        auto& stored = kvmem_lexical_pages_[index];
        if (kvmem_action_lrfu_) {
            stored.assign(pages.begin(), pages.end());
            kvmem_older_matches_[index].assign(older_matches.begin(), older_matches.end());
        } else {
            stored.clear();
            kvmem_older_matches_[index].clear();
        }
    }

    [[nodiscard]] std::vector<std::uint32_t>
    kvmem_selected_blocks(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        std::vector<std::uint32_t> selected;
        if (kvmem_enabled_) {
            for (const auto& block : kvmem_repos_[index].blocks()) {
                if (block.in_working_set) { selected.push_back(block.id); }
            }
        }
        return selected;
    }

    // Canonical reconstruction reads exactly the selected original token ranges,
    // in their packed execution order. Never treat a physical slot as a ledger id.
    [[nodiscard]] std::vector<std::uint32_t>
    kvmem_window_token_positions(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        std::vector<std::uint32_t> positions(
            address.committed_frontier, std::numeric_limits<std::uint32_t>::max());
        for (const auto& block : kvmem_repos_[index].blocks()) {
            if (block.tier != ninfer::kvmem::KvTier::Gpu || block.baked_pos < 0 ||
                static_cast<std::uint64_t>(block.baked_pos) >= positions.size()) { continue; }
            const auto begin = static_cast<std::uint32_t>(block.baked_pos);
            const auto count = std::min<std::uint32_t>(kPagedKVPageSize, positions.size() - begin);
            for (std::uint32_t col = 0; col < count; ++col) {
                positions[begin + col] = static_cast<std::uint32_t>(block.token_begin + col);
            }
        }
        return positions;
    }

    // Adapted from KVMem faf3c22582adf9d99ff072c543ea30ca03a8e2ca,
    // backends/patches/ninfer-kvmem.patch: capture_memory_checkpoint and
    // restore_memory_prefix. Native K/V codes AND scales remain authoritative;
    // unlike the donor's durable duplicate archive, this bounded port keeps
    // exactly one native owner per original page across Device/Host/Cold.
    [[nodiscard]] OriginalPrefixCheckpoint capture_original_prefix(
        KVAddressSpaceHandle handle, std::uint32_t logical_frontier,
        cudaStream_t stream = nullptr) {
        (void)retire_kvmem_transfers(true);
        Address& address = require_active(handle);
        if (!kvmem_original_positions_ || address.original_epoch == 0) {
            throw std::logic_error("original KV prefix capture requires a live original-position mode");
        }
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        const auto& repo = kvmem_repos_[index];
        const auto& native = kvmem_native_records_[index];
        const auto count = pages_for_tokens(logical_frontier);
        if (count > repo.block_count() || count > native.size()) {
            throw std::logic_error("original KV prefix is not fully archived");
        }
        std::uint64_t written_frontier = 0;
        for (const auto& block : repo.blocks()) {
            if (block.id < native.size() && native[block.id].columns != 0) {
                written_frontier = std::max(written_frontier,
                    block.token_begin + native[block.id].columns);
            }
        }
        if (written_frontier != logical_frontier) {
            throw std::logic_error("original KV checkpoint must be captured at its execution frontier");
        }
        OriginalPrefixCheckpoint checkpoint;
        checkpoint.address = handle;
        checkpoint.logical_frontier = logical_frontier;
        checkpoint.lineage_epoch = address.original_epoch;
        checkpoint.prefix_revisions.reserve(count);
        for (std::uint32_t id = 0; id < count; ++id) {
            const auto columns = std::min<std::uint32_t>(kPagedKVPageSize,
                logical_frontier - id * kPagedKVPageSize);
            require_original_native_owner(index, id, columns);
            checkpoint.prefix_revisions.push_back(native[id].revision);
        }
        const auto tail_columns = logical_frontier % kPagedKVPageSize;
        if (tail_columns != 0 && kvmem_mean_k_stride_ != 0) {
            const auto id = count - 1U;
            auto& means = kvmem_mean_k_[index];
            auto& tokens = kvmem_mean_k_tokens_[index];
            means.resize(std::max(means.size(), static_cast<std::size_t>(count) * kvmem_mean_k_stride_));
            tokens.resize(std::max(tokens.size(), static_cast<std::size_t>(count)), 0);
            if (tokens[id] != tail_columns && repo.tier_of(id) == ninfer::kvmem::KvTier::Gpu) {
                const std::int32_t page = pages_->physical_pool().physical_index_of(
                    pages_->physical(kvmem_block_pages_[index][id]));
                const auto position = static_cast<std::int32_t>(repo.blocks()[id].token_begin);
                ninfer::kvmem::kvmem_export_mean_k(pages_->physical_pool(), &page, &position,
                    &tail_columns, 1, means.data() + static_cast<std::size_t>(id) * kvmem_mean_k_stride_, stream);
                tokens[id] = tail_columns;
            }
            // An invalid/nonresident tail mean stays explicitly invalid. Once
            // materialized, the ordinary scorer recomputes it from native K.
            if (tokens[id] == tail_columns) {
                checkpoint.partial_mean.assign(
                    means.begin() + static_cast<std::size_t>(id) * kvmem_mean_k_stride_,
                    means.begin() + static_cast<std::size_t>(id + 1U) * kvmem_mean_k_stride_);
                checkpoint.partial_mean_tokens = tail_columns;
            }
        }
        return checkpoint;
    }

    [[nodiscard]] std::uint32_t rewind_original_prefix(
        KVAddressSpaceHandle handle, const OriginalPrefixCheckpoint& checkpoint,
        cudaStream_t stream = nullptr) {
        (void)retire_kvmem_transfers(true);
        Address& address = require_active(handle);
        if (!kvmem_original_positions_ || address.original_epoch == 0 ||
            checkpoint.address != handle || checkpoint.lineage_epoch != address.original_epoch) {
            throw std::logic_error("original KV rewind rejected a stale address or execution branch");
        }
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        auto& repo = kvmem_repos_[index];
        auto& native = kvmem_native_records_[index];
        const auto frontier = checkpoint.logical_frontier;
        const auto count = pages_for_tokens(frontier);
        if (checkpoint.prefix_revisions.size() != count || count > native.size() ||
            count > repo.block_count() ||
            (checkpoint.partial_mean.empty() && checkpoint.partial_mean_tokens != 0) ||
            (!checkpoint.partial_mean.empty() &&
             (checkpoint.partial_mean.size() != kvmem_mean_k_stride_ ||
              checkpoint.partial_mean_tokens != frontier % kPagedKVPageSize))) {
            throw std::logic_error("original KV checkpoint has invalid prefix coverage");
        }
        // Check every original page, including parked Host/NVMe pages, before
        // any destructive mapping change. Ownership may migrate after capture;
        // page identity, first-write revision and surviving columns may not.
        for (std::uint32_t id = 0; id < count; ++id) {
            const auto columns = std::min<std::uint32_t>(kPagedKVPageSize,
                frontier - id * kPagedKVPageSize);
            require_original_native_owner(index, id, columns);
            if (native[id].revision != checkpoint.prefix_revisions[id]) {
                throw std::logic_error("original KV prefix belongs to a replaced first-write page");
            }
        }
        try {
            // A probe can compact again. Derive the cut from the CURRENT map,
            // never from q0 plus an offset saved before that compaction. Keep
            // the partial original append page as the final packed slot.
            std::vector<std::uint32_t> selected;
            for (const auto& block : repo.blocks()) {
                if (block.id < count && block.tier == ninfer::kvmem::KvTier::Gpu) {
                    selected.push_back(block.id);
                }
            }
            const auto tail_columns = frontier % kPagedKVPageSize;
            if (tail_columns != 0 && (selected.empty() || selected.back() != count - 1U)) {
                selected.push_back(count - 1U);
                // Reuse the existing native Host/Cold -> Device transfer and
                // ownership publication. No source-K reconstruction is needed.
                const auto restored = kvmem_compact(handle,
                    static_cast<std::uint32_t>(selected.size()) * kPagedKVPageSize,
                    stream, {}, true, 0, selected);
                if (restored == 0) {
                    throw std::logic_error("original KV rewind could not restore its partial append page");
                }
            }
            std::uint32_t packed_frontier = 0;
            for (const auto& block : repo.blocks()) {
                if (block.id >= count || block.tier != ninfer::kvmem::KvTier::Gpu) { continue; }
                const auto columns = std::min<std::uint32_t>(kPagedKVPageSize,
                    frontier - static_cast<std::uint32_t>(block.token_begin));
                if (block.baked_pos != static_cast<std::int64_t>(packed_frontier)) {
                    throw std::logic_error("original KV rewind found a non-prefix packed selection");
                }
                packed_frontier += columns;
            }
            destructive_truncate(handle, packed_frontier);
            // The generic packed truncate only drops a trailing resident run.
            // Revoke ALL original future owners, even ones the probe offloaded.
            for (std::uint32_t id = count; id < repo.block_count(); ++id) {
                if (id < kvmem_block_hosts_[index].size()) {
                    kvmem_block_hosts_[index][id].release();
                }
                if (id < kvmem_block_cold_[index].size() && kvmem_block_cold_[index][id].valid()) {
                    if (!kvmem_cold_file_->release(kvmem_block_cold_[index][id])) {
                        throw std::logic_error("original KV future Cold owner became stale");
                    }
                }
            }
            kvmem_block_pages_[index].resize(count);
            kvmem_block_hosts_[index].resize(count);
            kvmem_block_cold_[index].resize(count);
            native.resize(count);
            if (tail_columns != 0) { native.back().columns = tail_columns; }
            repo.truncate(count);
            kvmem_mean_k_[index].resize(static_cast<std::size_t>(count) * kvmem_mean_k_stride_);
            kvmem_mean_k_tokens_[index].resize(count, 0);
            if (tail_columns != 0) {
                kvmem_mean_k_tokens_[index].back() = checkpoint.partial_mean_tokens;
                if (!checkpoint.partial_mean.empty()) {
                    std::copy(checkpoint.partial_mean.begin(), checkpoint.partial_mean.end(),
                        kvmem_mean_k_[index].begin() +
                            static_cast<std::size_t>(count - 1U) * kvmem_mean_k_stride_);
                }
            }
            const auto clip_candidates = [count](auto& ids) {
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [count](std::uint32_t id) { return id >= count; }), ids.end());
            };
            clip_candidates(kvmem_lexical_pages_[index]);
            clip_candidates(kvmem_older_matches_[index]);
            kvmem_reuse_floors_[index] = 0;
            ++address.original_epoch; // Revoke checkpoints above the restored q0.
            if (address.original_epoch == 0) { ++address.original_epoch; }
            return packed_frontier;
        } catch (...) {
            // Native copies and directory changes are not a transactional COW
            // archive. A partially destructive failure requires a fresh session.
            address.original_epoch = 0;
            throw;
        }
    }

    // Reuse the normal per-page rollback contract without dropping selected page
    // identities or their device reservation. Rebuilt K/V belongs to a new
    // conditioning epoch, so its old means/source replicas must not resurrect.
    void begin_kvmem_window_replay(KVAddressSpaceHandle handle, std::uint32_t prefix = 0,
                                   cudaStream_t stream = nullptr) {
        if (kvmem_original_positions_) {
            throw std::logic_error("canonical window replay cannot overwrite original-position KV");
        }
        drain_kvmem_layer_reader(handle);
        (void)retire_kvmem_transfers(true);
        Address& address = require_active(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        if (prefix % kPagedKVPageSize != 0 || prefix > address.committed_frontier) {
            throw std::logic_error("KVMem replay prefix has invalid coverage");
        }
        for (std::uint32_t slot = prefix / kPagedKVPageSize; slot < address.page_count; ++slot) {
            if (!pages_->can_destructive_truncate(membership(address, slot), 0)) {
                throw std::logic_error("KVMem window replay cannot overwrite protected KV");
            }
        }
        for (std::uint32_t slot = prefix / kPagedKVPageSize; slot < address.page_count; ++slot) {
            pages_->destructive_truncate(membership(address, slot), 0);
        }
        std::vector<ninfer::kvmem::ImmutableKStore::RestoreRequest> source_restores;
        source_restores.reserve(prefix / kPagedKVPageSize);
        for (const auto& block : kvmem_repos_[index].blocks()) {
            if (block.tier != ninfer::kvmem::KvTier::Gpu) { continue; }
            const auto id = block.id;
            if (block.baked_pos >= 0 && static_cast<std::uint64_t>(block.baked_pos) < prefix) {
                // V still has the same canonical prefix conditioning. Recover K
                // from that epoch's first-write codes to remove intermediate rephase drift.
                const auto logical = kvmem_block_pages_[index][id];
                source_restores.push_back({&kvmem_source_records_[index][id],
                    pages_->physical_pool().physical_index_of(pages_->physical(logical)),
                    static_cast<std::int32_t>(block.baked_pos)});
                continue;
            }
            if (id < kvmem_source_records_[index].size()) {
                kvmem_source_k_->release(kvmem_source_records_[index][id]);
            }
            if (id < kvmem_mean_k_tokens_[index].size()) {
                kvmem_mean_k_tokens_[index][id] = 0;
            }
            if (id < kvmem_block_hosts_[index].size()) { kvmem_block_hosts_[index][id].release(); }
            if (id < kvmem_block_cold_[index].size() && kvmem_block_cold_[index][id].valid()) {
                (void)kvmem_cold_file_->release(kvmem_block_cold_[index][id]);
                kvmem_block_cold_[index][id] = {};
            }
        }
        if (!source_restores.empty()) {
            kvmem_source_k_->restore_many(source_restores, pages_->physical_pool(), stream);
        }
        address.committed_frontier = prefix;
    }

    // A prefix fork copies native mapped pages, not this address's archive and
    // source-K directory. Only a full, unrebased history is self-contained.
    [[nodiscard]] bool kvmem_can_fork_prefix(KVAddressSpaceHandle handle) const {
        if (!kvmem_enabled_) { return true; }
        // A native prefix fork does not clone the single-owner original archive
        // or its branch identity. Start this mode with its own empty directory.
        if (kvmem_original_positions_) { return false; }
        const auto& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        const auto& repo = kvmem_repos_[index];
        if (repo.block_count() != address.page_count) { return false; }
        for (const auto& block : repo.blocks()) {
            if (block.tier != ninfer::kvmem::KvTier::Gpu ||
                block.baked_pos != static_cast<std::int64_t>(block.token_begin)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool kvmem_has_lexical_candidates(KVAddressSpaceHandle handle) const {
        if (!kvmem_enabled_) { return false; }
        const Address& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        return !kvmem_lexical_pages_[index].empty();
    }

    // Stronger than !restore_needed: that helper can decline optional recall
    // for shared or not-yet-written candidates. Direct continuation needs every
    // candidate to have real committed columns in this exact working view.
    [[nodiscard]] bool kvmem_lexical_candidates_resident(KVAddressSpaceHandle handle) const {
        if (!kvmem_enabled_ || !kvmem_action_lrfu_) { return false; }
        const auto& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        const auto& repo = kvmem_repos_[index];
        const auto& ids = kvmem_lexical_pages_[index];
        if (ids.empty()) { return false; }
        for (const auto id : ids) {
            if (id >= repo.block_count() || repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu ||
                id >= kvmem_block_pages_[index].size()) { return false; }
            const auto page = kvmem_block_pages_[index][id];
            if (!page.valid() || !pages_->valid(page) || pages_->committed_columns(page) == 0) {
                return false;
            }
        }
        return true;
    }

    // A reused window can fit the incoming question while its lexical evidence
    // is still on Host/Cold. Waiting for capacity pressure (or round completion)
    // would let that question run without the evidence. Only request reselection
    // once every candidate has actually been written into this lineage.
    [[nodiscard]] bool kvmem_lexical_restore_needed(KVAddressSpaceHandle handle) const {
        if (!kvmem_enabled_ || !kvmem_action_lrfu_) { return false; }
        const Address& address = require(handle);
        const auto index = static_cast<std::size_t>(&address - addresses_.data());
        const auto& repo = kvmem_repos_[index];
        bool missing = false;
        for (const std::uint32_t id : kvmem_lexical_pages_[index]) {
            if (id >= repo.block_count()) { return false; }
            missing = missing || repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu;
        }
        if (!missing) { return false; }
        // Optional recall must not turn an otherwise fitting shared-prefix
        // request into a geometry error. Leave shared views to the existing
        // pressure/protection policy; only force an exclusive window to move.
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (pages_->address_references(membership(address, page)) != 1) { return false; }
        }
        return true;
    }

    // KVMem K3: install the device query-sum buffer (f32 [layers][q_heads][head_dim])
    // and the host mean-K geometry. Called once at start-up so the addresses stay
    // stable across CUDA-graph capture; the prefill schedule accumulates the
    // request's query sums there and kvmem_compact ranks its ledger with them.
    void install_kvmem_query(std::uint32_t layers, std::uint32_t q_heads, std::uint32_t head_dim,
                             std::uint32_t kv_heads) {
        if (!kvmem_enabled_ || layers == 0 || q_heads == 0 || head_dim == 0 || kv_heads == 0) {
            return;
        }
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(layers) * q_heads * head_dim * sizeof(float);
        kvmem_query_buffer_ = DeviceBuffer(static_cast<std::size_t>(bytes));
        kvmem_query_buffer_.fill(0);
        kvmem_query_ = ninfer::kvmem::KvmemQueryCapture{
            kvmem_query_buffer_.p, layers, q_heads, head_dim, kv_heads, 0, 0};
        kvmem_mean_k_stride_ =
            static_cast<std::size_t>(layers) * kv_heads *
            ninfer::kvmem::kMeanKeySubblocks * head_dim;
        kvmem_query_host_.assign(static_cast<std::size_t>(layers) * q_heads * head_dim, 0.0f);
    }

    // Starts a request's retrieval capture: clears the summed query and records
    // which prompt positions the latest user turn covers. Chunked prefill then
    // accumulates the Q columns that fall inside that span.
    void reset_kvmem_query(
        std::uint32_t query_begin, std::uint32_t query_end,
        std::uint32_t action_begin, std::uint32_t action_end,
        std::uint32_t system_prefix_end,
        std::span<const std::pair<std::uint32_t, std::uint32_t>> closed_thinking_spans) {
        if (!kvmem_query_.valid()) { return; }
        kvmem_query_buffer_.fill(0);
        kvmem_query_.query_begin = query_begin;
        kvmem_query_.query_end   = query_end;
        kvmem_action_begin_ = action_begin;
        kvmem_action_end_ = action_end;
        kvmem_system_prefix_end_ = system_prefix_end;
        kvmem_closed_thinking_spans_.assign(closed_thinking_spans.begin(),
                                            closed_thinking_spans.end());
    }

    // Keep the completed probe Q for retrieval/profile while replaying the query.
    void freeze_kvmem_query() noexcept {
        kvmem_query_.query_begin = 0;
        kvmem_query_.query_end = 0;
    }

    // Benchmark snapshot of the completed probe. Shape is bound by the archive
    // header; action/lexical metadata is reconstructed from its exact prompt.
    [[nodiscard]] std::vector<float> benchmark_query_sums(cudaStream_t stream) const {
        if (!kvmem_query_.valid() || kvmem_query_.query_begin || kvmem_query_.query_end) {
            throw std::logic_error("canonical checkpoint needs a frozen query");
        }
        std::vector<float> sums(kvmem_query_host_.size());
        KVMEM_CUDA_CHECK(cudaMemcpyAsync(sums.data(), kvmem_query_.sums,
            sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        return sums;
    }

    void restore_benchmark_query(std::span<const float> sums, cudaStream_t stream) {
        if (!kvmem_query_.valid() || sums.size() != kvmem_query_host_.size()) {
            throw std::invalid_argument("canonical checkpoint query shape mismatch");
        }
        KVMEM_CUDA_CHECK(cudaMemcpyAsync(kvmem_query_.sums, sums.data(), sums.size_bytes(),
                                   cudaMemcpyHostToDevice, stream));
        KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        freeze_kvmem_query();
    }

    // Null until install_kvmem_query ran, i.e. unless KVMem is enabled.
    [[nodiscard]] const ninfer::kvmem::KvmemQueryCapture* kvmem_query_capture() const noexcept {
        return kvmem_query_.valid() ? &kvmem_query_ : nullptr;
    }

    // Read-only diagnostics for GET /metrics: lifetime compaction/retrieval counters and the last
    // window's residency. A store that was never enabled reports zeroes.
    [[nodiscard]] KvMemUsage kvmem_usage() const noexcept {
        KvMemUsage usage;
        usage.plans             = kvmem_plans_;
        usage.compactions       = kvmem_compactions_;
        usage.compact_skips     = kvmem_compact_skips_;
        usage.stage_in_blocks   = kvmem_stage_in_blocks_;
        usage.stage_out_blocks  = kvmem_stage_out_blocks_;
        usage.transfer_runs     = kvmem_transfer_runs_;
        usage.transfer_us       = kvmem_transfer_us_;
        usage.async_submitted   = kvmem_async_submitted_;
        usage.async_completed   = kvmem_async_completed_;
        usage.async_discarded   = kvmem_async_discarded_;
        usage.async_submit_us   = kvmem_async_submit_us_;
        usage.async_wait_us     = kvmem_async_wait_us_;
        usage.async_bytes       = kvmem_async_bytes_;
        usage.async_fallbacks   = kvmem_async_fallbacks_;
        usage.async_restore_submitted = kvmem_async_restore_submitted_;
        usage.async_restore_us = kvmem_async_restore_us_;
        usage.async_split_batches = kvmem_async_split_batches_;
        usage.nvme_prefetch_submitted = kvmem_nvme_prefetch_submitted_;
        usage.nvme_prefetch_completed = kvmem_nvme_prefetch_completed_;
        usage.nvme_prefetch_wait_us = kvmem_nvme_prefetch_wait_us_;
        usage.inflight_pages   = kvmem_inflight_pages();
        usage.inflight_bytes   = usage.inflight_pages * kvmem_host_layout_.page_stride;
        usage.scoreable_blocks  = kvmem_scoreable_blocks_;
        usage.zero_score_blocks = kvmem_zero_score_blocks_;
        usage.window_tokens     = kvmem_window_tokens_;
        usage.host_bytes        = kvmem_host_arena_ ? kvmem_host_arena_->occupied_bytes() : 0;
        usage.host_capacity_bytes = kvmem_host_capacity_bytes();
        usage.cold_bytes          = kvmem_cold_bytes();
        usage.host_to_cold_spills = kvmem_host_to_cold_spills();
        if (kvmem_source_k_) {
            usage.source_k_bytes = kvmem_source_k_->occupied_bytes();
            usage.source_k_file_capacity_bytes = kvmem_source_k_->file_capacity_bytes();
            usage.source_k_staging_bytes = kvmem_source_k_->staging_bytes();
            usage.source_k_captures = kvmem_source_k_->captures();
            usage.source_k_rebuilds = kvmem_source_k_->restores();
        }
        return usage;
    }

private:
    struct OriginalNativeRecord {
        std::uint32_t columns = 0;
        std::uint64_t revision = 0;
    };

    void commit_original_native_columns(std::size_t index, Address& address) {
        if (address.original_epoch == 0) {
            throw std::logic_error("original KV branch requires fresh prefill");
        }
        const auto& repo = kvmem_repos_[index];
        auto& native = kvmem_native_records_[index];
        native.resize(repo.block_count());
        for (const auto& block : repo.blocks()) {
            if (block.tier != ninfer::kvmem::KvTier::Gpu) { continue; }
            const auto logical = kvmem_block_pages_[index].at(block.id);
            const auto columns = pages_->committed_columns(logical);
            auto& record = native[block.id];
            if (columns < record.columns) {
                throw std::logic_error("original KV native coverage shrank without a rewind");
            }
            if (record.revision == 0 && columns != 0) {
                record.revision = address.next_native_revision++;
                if (address.next_native_revision == 0) { ++address.next_native_revision; }
            }
            record.columns = columns;
        }
    }

    void require_original_native_owner(std::size_t index, std::uint32_t id,
                                       std::uint32_t columns) const {
        const auto& address = addresses_[index];
        const auto& repo = kvmem_repos_[index];
        const auto& record = kvmem_native_records_[index].at(id);
        const auto& block = repo.blocks()[id];
        const auto logical = id < kvmem_block_pages_[index].size()
            ? kvmem_block_pages_[index][id] : LogicalKVPageHandle{};
        const bool device = logical.valid() && pages_->valid(logical);
        const bool host = id < kvmem_block_hosts_[index].size() &&
            kvmem_block_hosts_[index][id].valid();
        const bool cold = id < kvmem_block_cold_[index].size() &&
            kvmem_block_cold_[index][id].valid();
        if (block.id != id || block.token_begin != static_cast<std::uint64_t>(id) * kPagedKVPageSize ||
            columns == 0 || record.revision == 0 || columns > record.columns ||
            record.columns > kPagedKVPageSize || static_cast<int>(device) + host + cold != 1) {
            throw std::logic_error("original KV prefix has no unique complete native owner");
        }
        if (device) {
            const auto slot = block.baked_pos / kPagedKVPageSize;
            if (block.tier != ninfer::kvmem::KvTier::Gpu || block.baked_pos < 0 ||
                block.baked_pos % kPagedKVPageSize != 0 ||
                static_cast<std::uint64_t>(slot) >= address.page_count ||
                membership(address, static_cast<std::uint32_t>(slot)) != logical ||
                pages_->committed_columns(logical) != record.columns ||
                pages_->address_references(logical) != 1) {
                throw std::logic_error("original KV native Device identity is stale or shared");
            }
        } else if (host) {
            if (block.tier != ninfer::kvmem::KvTier::Host || !kvmem_host_arena_ ||
                !kvmem_host_arena_->view(kvmem_block_hosts_[index][id]).valid()) {
                throw std::logic_error("original KV native Host identity is stale");
            }
        } else if (block.tier != ninfer::kvmem::KvTier::Nvme || !kvmem_cold_file_ ||
                   !kvmem_cold_file_->valid(kvmem_block_cold_[index][id])) {
            throw std::logic_error("original KV native Cold identity is stale");
        }
    }

    void ensure_kvmem_host_arena(DeviceKVPagePool& pool) {
        if (kvmem_host_arena_) { return; }
        kvmem_host_layout_ = plan_host_kv_page_layout(pool.geometry());
        const std::size_t stride = kvmem_host_layout_.page_stride;
        std::size_t host_pages = page_capacity_;
        bool limited_host_tier = false;
        if (const char* setting = std::getenv("NINFER_KVMEM_HOST_PAGES")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long requested = std::strtoull(setting, &end, 10);
            if (errno != 0 || end == setting || *end != '\0' || requested == 0) {
                throw std::invalid_argument("NINFER_KVMEM_HOST_PAGES must be positive");
            }
            host_pages = std::min<std::size_t>(page_capacity_, requested);
            limited_host_tier = true;
        }
        const std::size_t capacity =
            stride * host_pages + (limited_host_tier ? 0 : (std::size_t{1} << 20));
        kvmem_host_arena_ = std::make_unique<HostKVArena>(
            capacity, std::vector<HostKVPageLayout>{kvmem_host_layout_});
    }

    const DeviceContext* kvmem_async_device_ = nullptr;
    std::unique_ptr<PinnedHostBuffer> kvmem_rerope_host_;
    DeviceSpan kvmem_rerope_device_;
    std::unique_ptr<PendingDemote> kvmem_pending_demote_;
    std::uint64_t kvmem_async_completed_ = 0;
    std::uint64_t kvmem_async_split_batches_ = 0;
    std::uint64_t kvmem_nvme_prefetch_submitted_ = 0;
    std::uint64_t kvmem_nvme_prefetch_completed_ = 0;
    std::uint64_t kvmem_nvme_prefetch_wait_us_ = 0;
    std::uint64_t kvmem_async_submit_us_ = 0;
    std::uint64_t kvmem_async_submitted_ = 0;
    std::uint64_t kvmem_async_discarded_ = 0;
    std::uint64_t kvmem_async_wait_us_ = 0;
    std::uint64_t kvmem_async_bytes_ = 0;
    std::uint64_t kvmem_async_fallbacks_ = 0;
    std::uint64_t kvmem_async_restore_submitted_ = 0;
    std::uint64_t kvmem_async_restore_us_ = 0;
    bool kvmem_enabled_ = false;
    bool kvmem_original_positions_ = false;
    bool kvmem_action_lrfu_ = false;
    double kvmem_lrfu_lambda_ = 0.5;
    double kvmem_lrfu_top_fraction_ = 0.9;
    std::uint32_t kvmem_budget_ = 0;  // NINFER_KVMEM_BUDGET tokens; 0 = unbounded
    std::uint32_t kvmem_extra_device_pages_ = 0;
    // Generation headroom in tokens (page-aligned, already clamped to the pool's spare room).
    // Zero disables generation-time reclaim, which is the identity configuration.
    std::uint32_t kvmem_gen_reserve_ = 0;
    std::vector<ninfer::kvmem::KvBlockRepository> kvmem_repos_;
    std::vector<std::vector<std::uint32_t>> kvmem_lexical_pages_;
    std::vector<std::vector<std::uint32_t>> kvmem_older_matches_;
    // Per address: block id -> logical page currently holding it (invalid while
    // the block is evicted), and the parked host slot while evicted.
    std::vector<std::vector<LogicalKVPageHandle>> kvmem_block_pages_;
    std::vector<std::vector<HostKVAllocation>> kvmem_block_hosts_;
    std::vector<std::vector<ninfer::kvmem::ColdKVStore::Handle>> kvmem_block_cold_;
    std::vector<std::vector<ninfer::kvmem::ImmutableKStore::Record>> kvmem_source_records_;
    // Valid first-write coverage lives independently of the legacy K-only
    // mirror. Tier migration transfers one complete native K/V+scale owner.
    std::vector<std::vector<OriginalNativeRecord>> kvmem_native_records_;
    std::unique_ptr<ninfer::kvmem::ImmutableKStore> kvmem_source_k_;
    // Main16 prefill-only producer/consumer events; address is generation-bound.
    std::unique_ptr<ninfer::kvmem::LayerReadyFence> kvmem_layer_ready_;
    KVAddressSpaceHandle kvmem_layer_ready_address_;
    std::unique_ptr<HostKVArena> kvmem_host_arena_;
    std::unique_ptr<HostKVArena> kvmem_cold_staging_;
    std::array<ColdLayerSlot, 2> cold_layer_slots_{};
    std::size_t cold_layer_count_ = 0;
    cudaStream_t cold_layer_producer_stream_ = nullptr;
    std::unique_ptr<ninfer::kvmem::ColdKVStore> kvmem_cold_file_;
    HostKVPageLayout kvmem_host_layout_{};
    // KVMem K3: the request query the prefill schedule captured, and the device
    // buffer behind it.
    DeviceBuffer kvmem_query_buffer_;
    ninfer::kvmem::KvmemQueryCapture kvmem_query_{};
    // Per address: the lowest window slot a resumed request may continue from.
    std::vector<std::uint32_t> kvmem_reuse_floors_;
    // Per address: F32 sum of raw K, [block][layer][kv_heads][head_dim], plus the
    // number of written tokens the row was exported with (0 = no evidence yet).
    std::vector<std::vector<float>> kvmem_mean_k_;
    std::vector<std::vector<std::uint32_t>> kvmem_mean_k_tokens_;
    std::size_t kvmem_mean_k_stride_ = 0;
    std::vector<float> kvmem_query_host_;
    std::uint32_t kvmem_action_begin_ = 0;
    std::uint32_t kvmem_action_end_ = 0;
    std::uint32_t kvmem_system_prefix_end_ = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kvmem_closed_thinking_spans_;
    std::uint64_t kvmem_plans_             = 0;
    std::uint64_t kvmem_compactions_       = 0;
    std::uint64_t kvmem_compact_skips_     = 0;
    std::uint64_t kvmem_stage_in_blocks_   = 0;
    std::uint64_t kvmem_stage_out_blocks_  = 0;
    std::uint64_t kvmem_host_to_cold_spills_ = 0;
    std::uint64_t kvmem_transfer_runs_     = 0;
    std::uint64_t kvmem_transfer_us_       = 0;
    std::uint64_t kvmem_layer_failure_drains_ = 0;
    std::uint64_t kvmem_scoreable_blocks_  = 0;
    std::uint64_t kvmem_zero_score_blocks_ = 0;
    std::uint32_t kvmem_window_tokens_     = 0;

    LogicalKVPageStore* pages_    = nullptr;
    KVExecutionTablePool* tables_ = nullptr;
    std::uint32_t page_capacity_  = 0;
    std::vector<Address> addresses_;
    std::vector<std::uint32_t> free_;
    std::vector<LogicalKVPageHandle> memberships_;
    std::vector<DeviceKVPageHandle> publish_scratch_;
    std::uint32_t free_count_ = 0;
};

inline KVPrefixForkReservation::~KVPrefixForkReservation() {
    if (owner_ != nullptr) { owner_->abort_prefix_fork(*this); }
}

inline KVActiveSnapshotReservation::~KVActiveSnapshotReservation() {
    if (owner_ != nullptr) { owner_->abort_active_snapshot(*this); }
}

} // namespace ninfer::targets::qwen3_6::detail
