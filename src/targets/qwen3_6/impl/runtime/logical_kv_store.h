#pragma once

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_blocks.h"
#include "kvmem/kvmem_bridge.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstddef>
#include <cstdint>
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
          activation_frontier_(other.activation_frontier_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t requested_entitlement,
                            std::optional<std::uint32_t> activation_frontier,
                            DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), requested_entitlement_(requested_entitlement),
          activation_frontier_(activation_frontier), page_reservation_(std::move(page_reservation)),
          row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t requested_entitlement_ = 0;
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
                            std::uint32_t requested_entitlement,
                            DeviceKVPageReservation&& page_reservation, KVExecutionRowLease&& row,
                            std::optional<LogicalKVPageHandle> tail_destination,
                            bool staged_tail_release) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          full_pages_(full_pages), tail_columns_(tail_columns),
          requested_entitlement_(requested_entitlement),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)),
          tail_destination_(tail_destination), staged_tail_release_(staged_tail_release) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_              = 0;
    std::uint32_t full_pages_            = 0;
    std::uint32_t tail_columns_          = 0;
    std::uint32_t requested_entitlement_ = 0;
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
public:
    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity),
          memberships_(checked_membership_cells(address_capacity, page_capacity)),
          free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        // KVMem K1a: NINFER_KVMEM=1 mirrors the page lifecycle into a K0 block
        // repository per address and re-plans the (passthrough) window on every
        // frontier commit. Purely observational -- no engine state is redirected.
        if (const char* env = std::getenv("NINFER_KVMEM"); env != nullptr && env[0] == '1') {
            kvmem_enabled_ = true;
            if (const char* budget = std::getenv("NINFER_KVMEM_BUDGET");
                budget != nullptr && *budget != '\0') {
                const long long parsed = std::atoll(budget);
                if (parsed > 0) {
                    kvmem_budget_ = static_cast<std::uint32_t>(
                        std::min<long long>(parsed, static_cast<long long>(page_capacity) *
                                                            kPagedKVPageSize));
                    // Page-aligned once here so the completion-time window and the
                    // window-edge anchor frontier are the same number.
                    kvmem_budget_ -= kvmem_budget_ % kPagedKVPageSize;
                }
            }
            kvmem_repos_.resize(address_capacity);
            kvmem_block_pages_.resize(address_capacity);
            kvmem_block_hosts_.resize(address_capacity);
            kvmem_reuse_floors_.resize(address_capacity);
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
        if (address.active || address.row || address.reservation.valid() || entitlement == 0 ||
            entitlement > page_capacity_ ||
            (activation_frontier && *activation_frontier > address.committed_frontier)) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        const std::uint32_t required_pages =
            activation_frontier ? pages_for_tokens(*activation_frontier) : address.page_count;
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        std::uint32_t missing_replicas      = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (!pages_->device_resident(membership(address, page))) { ++missing_replicas; }
        }
        const std::uint32_t growth =
            entitlement > required_pages ? entitlement - required_pages : 0U;
        const std::uint64_t required = static_cast<std::uint64_t>(missing_replicas) + growth;
        if (required > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("KV activation reservation overflow");
        }
        pages_->physical_pool().resize_reservation(reservation,
                                                   static_cast<std::uint32_t>(required));
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, entitlement, activation_frontier,
                                       std::move(reservation), std::move(row));
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
        if (address.active || address.row || address.reservation.valid() ||
            activation.requested_entitlement_ == 0 ||
            activation.requested_entitlement_ > page_capacity_ ||
            (activation.activation_frontier_ &&
             address.committed_frontier != *activation.activation_frontier_)) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.requested_entitlement_ > address.page_count
                                           ? activation.requested_entitlement_ - address.page_count
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

        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(
            reservation, staged_tail_release ? 1U : entitlement - full_pages);
        KVExecutionRowLease row = tables_->acquire(execution_row);
        std::optional<LogicalKVPageHandle> tail_destination;
        if (tail_columns != 0) {
            tail_destination = pages_->materialize_transfer_destination(reservation, tail_columns);
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->pin_source(membership(source, page));
        }
        return KVPrefixForkReservation(*this, source_handle, destination_handle, frontier,
                                       full_pages, tail_columns, entitlement,
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
        const std::uint32_t growth         = fork.requested_entitlement_ - required_pages;
        pages_->physical_pool().resize_reservation(fork.page_reservation_, growth);
    }

    void commit_prefix_fork(KVPrefixForkReservation&& fork, cudaStream_t stream = nullptr) {
        require_prefix_fork(fork);
        Address& source                    = require(fork.source_);
        Address& destination               = require(fork.destination_);
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth         = fork.requested_entitlement_ - required_pages;
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
        destination.reservation        = std::move(fork.page_reservation_);
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
        Address& address = require_active(handle);
        if (entitlement < address.page_count || entitlement > page_capacity_) {
            throw std::invalid_argument("KV entitlement is smaller than mapped pages");
        }
        pages_->physical_pool().resize_reservation(address.reservation,
                                                   entitlement - address.page_count);
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
        if (target > entitlement(address)) {
            throw std::invalid_argument(
                "KV coverage exceeds active entitlement: tokens=" + std::to_string(tokens) +
                " required_pages=" + std::to_string(target) +
                " mapped_pages=" + std::to_string(address.page_count) +
                " reserved_pages=" + std::to_string(address.reservation.pages()) +
                " entitlement=" + std::to_string(entitlement(address)));
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
                repo.append(m);
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
                repo.append(m);
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
            kvmem_plan_passthrough(static_cast<std::size_t>(&address - addresses_.data()),
                                   address);
        }
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        if (frontier == address.committed_frontier && target == address.page_count) { return; }
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
            kvmem_reuse_floors_[index] = 0;
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
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
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
        std::uint32_t keep = repo.block_count();
        for (std::uint32_t id = keep; id != 0; --id) {
            const std::uint32_t candidate = id - 1;
            if (repo.tier_of(candidate) != ninfer::kvmem::KvTier::Gpu) { break; }
            const std::int64_t baked = repo.baked_pos_of(candidate);
            if (baked < 0 || static_cast<std::uint32_t>(baked) < slot_frontier) { break; }
            keep = candidate;
        }
        if (keep == repo.block_count()) { return; }
        for (std::uint32_t id = keep; id < repo.block_count(); ++id) {
            if (block_pages.size() > id) { block_pages[id] = {}; }
            if (block_hosts.size() > id && block_hosts[id].valid()) { block_hosts[id].release(); }
        }
        block_pages.resize(keep);
        block_hosts.resize(keep);
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
            repo.append(m);
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

    // KVMem K3: fill the ledger's retrieval scores from the captured query. A
    // block whose K is not device-resident (evicted to the host arena) keeps
    // score 0 -- "no evidence". With no capture installed, or with nothing
    // scoreable, every score stays 0 and the window selection is the unranked
    // one, so a KVMem-disabled run is bit-identical to the K2 path.
    void kvmem_score_window(std::size_t index, DeviceKVPagePool& pool, cudaStream_t stream) {
        const ninfer::kvmem::KvmemQueryCapture* query = kvmem_query_capture();
        if (query == nullptr) { return; }
        auto& repo            = kvmem_repos_[index];
        const std::uint32_t n = repo.block_count();
        if (n == 0) { return; }
        std::vector<std::int32_t> pages(n, -1);
        std::vector<std::int32_t> baked(n, 0);
        bool scoreable = false;
        for (std::uint32_t id = 0; id < n; ++id) {
            if (repo.tier_of(id) != ninfer::kvmem::KvTier::Gpu) { continue; }
            if (kvmem_block_pages_[index].size() <= id ||
                !kvmem_block_pages_[index][id].valid()) {
                continue;
            }
            pages[id] = pool.physical_index_of(pages_->physical(kvmem_block_pages_[index][id]));
            baked[id] = static_cast<std::int32_t>(repo.baked_pos_of(id));
            scoreable = true;
        }
        if (!scoreable) { return; }
        std::vector<float> scores(n, 0.0f);
        try {
            ninfer::kvmem::kvmem_score_blocks(pool, *query, pages.data(), baked.data(), n,
                                              scores.data(), stream);
        } catch (const std::exception& error) {
            // Retrieval only ranks the middle quota: a geometry it cannot read
            // must not make the compaction fail, it must leave it unranked.
            std::fprintf(stderr, "KVMEM_TRACE score_failed %s\n", error.what());
            return;
        }
        for (std::uint32_t id = 0; id < n; ++id) { repo.add_scores(id, scores[id], 0.0); }
        if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            for (std::uint32_t id = 0; id < n; ++id) {
                std::fprintf(stderr, "KVMEM_TRACE score id=%u page=%d pos=%d retrieval=%.5f\n", id,
                             pages[id], baked[id], scores[id]);
            }
        }
    }

public:
    // KVMem K1b: compact one address's KV into a budget-driven working window.
    //
    // Requires an INACTIVE address whose pages are solely owned (no fork or
    // snapshot sharing -- re-phasing a shared page would corrupt the other
    // views). Returns the new window token count, or 0 when nothing needed
    // compacting. Evicted blocks keep their bytes in this store's KVMem host
    // arena (K4 adds an SSD tier); a re-selected block is restored from there
    // and re-phased to its new window slot. NOTE: a restored block carries one
    // extra int8 quantization (the host copy holds the old-phased page, not a
    // raw-K mirror -- removing that extra step is K2-full work).
    std::uint32_t kvmem_compact(KVAddressSpaceHandle handle, std::uint32_t budget_tokens,
                                cudaStream_t stream = nullptr,
                                std::vector<std::uint32_t> mandatory_ids = {},
                                bool sink_only = false) {
        if (!kvmem_enabled_) { return 0; }
        Address& address = require(handle);
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
        if (budget_tokens == 0 || ledger_tokens <= budget_tokens) { return 0; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (pages_->address_references(membership(address, page)) != 1) {
                        return 0;
            }
        }
        DeviceKVPagePool& pool = pages_->physical_pool();

        // ---- plan through K0 ----
        ninfer::kvmem::KvSelectConfig cfg;
        cfg.block_tokens  = kPagedKVPageSize;
        cfg.budget_tokens = budget_tokens;
        const std::uint32_t budget_blocks = budget_tokens / kPagedKVPageSize;
        ninfer::kvmem::KvSelection sel;
        if (sink_only) {
            // Round-boundary mode: keep the oldest head, evict the tail. The
            // selection is direct (bands cannot express "no recent"); the next
            // round re-prefills everything after the head, and K3 will replace
            // the lost ranges with query-driven retention.
            sel.block_ids.resize(budget_blocks);
            for (std::uint32_t i = 0; i < budget_blocks; ++i) { sel.block_ids[i] = i; }
        } else {
            if (budget_blocks < 4) {
                throw std::invalid_argument(
                    "KVMem compaction budget must span at least four pages "
                    "(sink/recent/middle)");
            }
            if (budget_tokens < 4096) {
                // Explicit bands: the auto clamps (sink 1024..2048, recent
                // 4096..16384) exceed a small budget. Keep one sink page, one
                // recent page and let the quota fill the middle.
                cfg.sink_tokens  = std::clamp(budget_blocks / 8, 1u, budget_blocks / 4) *
                                  kPagedKVPageSize;
                cfg.recent_tokens =
                    std::max<std::uint32_t>(1, budget_blocks / 4) * kPagedKVPageSize;
            }
            kvmem_score_window(index, pool, stream);
            sel = repo.preview_select(cfg, std::move(mandatory_ids));  // K3 ranks the middle
        }
        ninfer::kvmem::KvWindowPlan plan = repo.set_selection(sel, cfg);
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
            if (issue_frontier > static_cast<std::uint32_t>(tail.from_base)) {
                const std::uint32_t valid =
                    std::min(issue_frontier - static_cast<std::uint32_t>(tail.from_base),
                             tail.n_tokens);
                const std::uint32_t packed = static_cast<std::uint32_t>(
                    tail.to_base + static_cast<std::int64_t>(valid));
                if (packed < window_tokens) { kv_window = packed; }
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

        // ---- host arena (lazy) ----
        if (!kvmem_host_arena_) {
            kvmem_host_layout_ = plan_host_kv_page_layout(pool.geometry());
            // Size for the WORST case -- every page of this address evicted.
            // A budget-sized arena exhausts exactly when the ledger has grown
            // past the budget the arena was sized for.
            const std::size_t capacity =
                kvmem_host_layout_.page_stride * static_cast<std::size_t>(page_capacity_) +
                (std::size_t{1} << 20);
            kvmem_host_arena_ = std::make_unique<HostKVArena>(
                capacity, std::vector<HostKVPageLayout>{kvmem_host_layout_});
        }

        // ---- 1) stage out: park evicted pages in the host arena ----
        for (const std::uint32_t id : plan.stage_out) {
            if (block_pages.size() <= id) { block_pages.resize(id + 1); }
            if (block_hosts.size() <= id) { block_hosts.resize(id + 1); }
            std::optional<HostKVAllocation> slot =
                kvmem_host_arena_->allocate(kvmem_host_layout_, 1);
            if (!slot) {
                throw std::runtime_error("KVMem host arena exhausted (SSD tier is K4)");
            }
            block_hosts[id] = std::move(*slot);
            DeviceKVPageHandle ph = pages_->physical(block_pages[id]);
            pool.copy_to_host(std::span<const DeviceKVPageHandle>(&ph, 1),
                              kvmem_host_arena_->writable_view(block_hosts[id]));
        }

        // ---- 2) stage in: fresh pages for cold selected blocks ----
        if (!plan.stage_in.empty()) {
            std::optional<DeviceKVPageReservation> reservation =
                pool.reserve(static_cast<std::uint32_t>(plan.stage_in.size()));
            if (!reservation) {
                throw std::runtime_error("KVMem stage-in found no free device page");
            }
            for (const std::uint32_t id : plan.stage_in) {
                if (block_pages.size() <= id) { block_pages.resize(id + 1); }
                if (block_hosts.size() <= id) { block_hosts.resize(id + 1); }
                LogicalKVPageHandle logical = pages_->materialize(*reservation);
                block_pages[id]             = logical;
                if (block_hosts[id].valid()) {
                    DeviceKVPageHandle ph = pages_->physical(logical);
                    pool.copy_from_host(kvmem_host_arena_->writable_view(block_hosts[id]),
                                        std::span<const DeviceKVPageHandle>(&ph, 1));
                    block_hosts[id].release();
                }
                pages_->commit_coverage(logical, kPagedKVPageSize);
            }
        }

        // ---- 3) re-phase every moved block from its baked slot to the new one ----
        if (!plan.remaps.empty()) {
            std::int32_t *d_src = nullptr, *d_dst = nullptr;
            const std::size_t pos_bytes = 3 * kPagedKVPageSize * sizeof(std::int32_t);
            KVMEM_CUDA_CHECK(cudaMalloc(&d_src, pos_bytes));
            KVMEM_CUDA_CHECK(cudaMalloc(&d_dst, pos_bytes));
            std::vector<std::int32_t> src(3 * kPagedKVPageSize), dst(3 * kPagedKVPageSize);
            for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
                if (rm.skip) { continue; }
                if (rm.raw_refresh) {
                    // K1b: the host copy holds the old-phased page (not raw K),
                    // so a restored block re-phases exactly like a resident one;
                    // the drift thresholds then only reset their counters. The
                    // raw-K mirror that makes this a true rebuild is K2-full.
                    repo.reset_drift(rm.block_id);
                }
                for (int a = 0; a < 3; ++a) {
                    for (std::uint32_t t = 0; t < kPagedKVPageSize; ++t) {
                        src[static_cast<std::size_t>(a) * kPagedKVPageSize + t] =
                            static_cast<std::int32_t>(rm.from_base) + static_cast<std::int32_t>(t);
                        dst[static_cast<std::size_t>(a) * kPagedKVPageSize + t] =
                            static_cast<std::int32_t>(rm.to_base) + static_cast<std::int32_t>(t);
                    }
                }
                KVMEM_CUDA_CHECK(cudaMemcpyAsync(d_src, src.data(), pos_bytes,
                                                 cudaMemcpyHostToDevice, stream));
                KVMEM_CUDA_CHECK(cudaMemcpyAsync(d_dst, dst.data(), pos_bytes,
                                                 cudaMemcpyHostToDevice, stream));
                if (block_pages.size() <= rm.block_id) { block_pages.resize(rm.block_id + 1); }
                ninfer::kvmem::kvmem_rerope_page(pool,
                                  pool.physical_index_of(pages_->physical(block_pages[rm.block_id])),
                                  d_src, d_dst, stream);
            }
            KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
            KVMEM_CUDA_CHECK(cudaFree(d_src));
            KVMEM_CUDA_CHECK(cudaFree(d_dst));
        }

        // ---- 4) release the evicted pages' device storage ----
        for (const std::uint32_t id : plan.stage_out) {
            pages_->release_reference(block_pages[id], false);
            block_pages[id] = LogicalKVPageHandle{};
        }

        // ---- 5) rewire the window and shrink the address ----
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
        if (address.active) { publish_membership(address, stream); }

        // ---- 6) ledger tiers ----
        // The tracker keeps its ledger order: evicted blocks stay registered as Host with their
        // parked bytes, so a later round can select one again and stage it back in. Only their
        // window placement goes stale, and a destructive truncate drops the ones that leave the
        // window for good.
        for (const std::uint32_t id : plan.stage_out) { repo.set_tier(id, ninfer::kvmem::KvTier::Host); }
        for (const ninfer::kvmem::KvRemap& rm : plan.remaps) {
            repo.set_tier(rm.block_id, ninfer::kvmem::KvTier::Gpu);
        }
        kvmem_reuse_floors_[index] = reuse_floor_slots;
        ++kvmem_compactions_;
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
        return kvmem_compact(handle, kvmem_budget_, stream, {}, false);
    }

    [[nodiscard]] std::uint32_t kvmem_budget_tokens() const noexcept { return kvmem_budget_; }

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

    // KVMem K3: install the device query-capture buffer
    // (bf16 [layers][q_heads][head_dim]). Called once at start-up so the address
    // stays stable across CUDA-graph capture; the prefill schedule writes the
    // request's query there and kvmem_compact ranks its ledger with it.
    void install_kvmem_query(std::uint32_t layers, std::uint32_t q_heads, std::uint32_t head_dim) {
        if (!kvmem_enabled_ || layers == 0 || q_heads == 0 || head_dim == 0) { return; }
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(layers) * q_heads * head_dim * sizeof(std::uint16_t);
        kvmem_query_buffer_ = DeviceBuffer(static_cast<std::size_t>(bytes));
        kvmem_query_buffer_.fill(0);
        kvmem_query_ =
            ninfer::kvmem::KvmemQueryCapture{kvmem_query_buffer_.p, layers, q_heads, head_dim};
    }

    // Null until install_kvmem_query ran, i.e. unless KVMem is enabled.
    [[nodiscard]] const ninfer::kvmem::KvmemQueryCapture* kvmem_query_capture() const noexcept {
        return kvmem_query_.valid() ? &kvmem_query_ : nullptr;
    }

private:
    bool kvmem_enabled_ = false;
    std::uint32_t kvmem_budget_ = 0;  // NINFER_KVMEM_BUDGET tokens; 0 = unbounded
    std::vector<ninfer::kvmem::KvBlockRepository> kvmem_repos_;
    // Per address: block id -> logical page currently holding it (invalid while
    // the block is evicted), and the parked host slot while evicted.
    std::vector<std::vector<LogicalKVPageHandle>> kvmem_block_pages_;
    std::vector<std::vector<HostKVAllocation>> kvmem_block_hosts_;
    std::unique_ptr<HostKVArena> kvmem_host_arena_;
    HostKVPageLayout kvmem_host_layout_{};
    // KVMem K3: the request query the prefill schedule captured, and the device
    // buffer behind it.
    DeviceBuffer kvmem_query_buffer_;
    ninfer::kvmem::KvmemQueryCapture kvmem_query_{};
    // Per address: the lowest window slot a resumed request may continue from.
    std::vector<std::uint32_t> kvmem_reuse_floors_;
    std::uint64_t kvmem_plans_       = 0;
    std::uint64_t kvmem_compactions_ = 0;

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
