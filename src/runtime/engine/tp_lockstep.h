#pragma once
// Two-process tensor-parallel lockstep (TP2 bring-up).
//
// Each rank runs its own Engine; the model's collectives only line up if both engines issue the
// same GPU units in the same order and make the same host decisions. Wall-clock driven decisions
// (client cancellation) are therefore agreed here once per GPU unit, and each unit's committed
// result (token count + token hash) is compared; any disagreement kills the process loudly.
//
// Transport: a shared-memory file (NINFER_TP_LOCKSTEP_FILE), not the NCCL communicator, so control
// traffic never interleaves with the model's in-graph all-reduces. Rank 0 creates it; the
// supervisor deletes it before (re)starting both ranks. Enabled only when NINFER_TP_RANK and
// NINFER_TP_LOCKSTEP_FILE are both set.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::runtime::tp {

enum class UnitKind : std::uint64_t { Prefill = 1, Decode = 2, Admission = 3 };

struct alignas(64) Record {
    std::uint64_t seq;
    std::uint64_t kind;
    std::uint64_t cancel_mask;
    std::uint64_t count;
    std::uint64_t hash;
    std::uint64_t position;
    std::uint64_t pending;   // requests waiting in this rank's FIFO when the unit was exchanged
    std::uint64_t pad[1];
};

struct alignas(64) SharedBlock {
    std::uint64_t magic;
    std::uint64_t pad0[7];
    std::atomic<std::uint64_t> published[2];  // last exchanged seq per rank
    std::uint64_t pad1[6];
    Record records[2][2];                     // [rank][seq & 1]
};

inline constexpr std::uint64_t kLockstepMagic = 0x4e494e4650325450ULL;  // "NINFP2TP"

[[noreturn]] inline void lockstep_fatal(const std::string& what) {
    std::fprintf(stderr, "[ninfer] TP2 LOCKSTEP FAILURE: %s\n", what.c_str());
    std::fflush(stderr);
    std::abort();
}

class Lockstep {
  public:
    static Lockstep* instance() {
        static Lockstep* self = [] () -> Lockstep* {
            const char* rank = std::getenv("NINFER_TP_RANK");
            const char* path = std::getenv("NINFER_TP_LOCKSTEP_FILE");
            if (rank == nullptr || path == nullptr) { return nullptr; }
            return new Lockstep(std::atoi(rank), path);
        }();
        return self;
    }

    [[nodiscard]] int rank() const noexcept { return rank_; }

    // Publishes this rank's view of the unit, waits for the peer's, verifies both describe the same
    // unit and result, and returns the OR of both cancellation masks.
    // `pending` is this rank's FIFO length; the peer's is readable afterwards through
    // peer_pending(), and the smaller of the two is what both ranks may admit next.
    std::uint64_t exchange(UnitKind kind, std::uint64_t cancel_mask, std::uint64_t count,
                           std::uint64_t hash, std::uint64_t position, std::uint64_t pending = 0) {
        const std::uint64_t seq = ++seq_;
        if (fault_unit_ != 0 && seq == fault_unit_) { hash ^= 0x5a5a5a5aULL; }  // test hook
        const Record mine{seq, static_cast<std::uint64_t>(kind), cancel_mask, count, hash, position,
                          pending, {}};
        const Record theirs = publish_and_wait(mine);
        if (theirs.seq != seq || theirs.kind != mine.kind || theirs.count != mine.count ||
            theirs.hash != mine.hash || theirs.position != mine.position) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "unit %llu diverged: kind %llu/%llu count %llu/%llu hash %016llx/%016llx "
                          "position %llu/%llu (rank%d/peer)",
                          static_cast<unsigned long long>(seq),
                          static_cast<unsigned long long>(mine.kind),
                          static_cast<unsigned long long>(theirs.kind),
                          static_cast<unsigned long long>(mine.count),
                          static_cast<unsigned long long>(theirs.count),
                          static_cast<unsigned long long>(mine.hash),
                          static_cast<unsigned long long>(theirs.hash),
                          static_cast<unsigned long long>(mine.position),
                          static_cast<unsigned long long>(theirs.position), rank_);
            lockstep_fatal(buf);
        }
        peer_pending_ = theirs.pending;
        return cancel_mask | theirs.cancel_mask;
    }

    // Admission agreement while no GPU unit is flowing (Engine::worker_loop): both ranks publish
    // how many requests wait in their FIFO and may admit the smaller number. The counts may differ
    // (a request reaches the two processes at slightly different times); only the kind must match.
    std::uint64_t agree_admissions(std::uint64_t pending) {
        const std::uint64_t seq = ++seq_;
        const Record mine{seq, static_cast<std::uint64_t>(UnitKind::Admission), 0, pending, 0, 0,
                          pending, {}};
        const Record theirs = publish_and_wait(mine);
        if (theirs.seq != seq || theirs.kind != mine.kind) {
            lockstep_fatal("admission exchange " + std::to_string(seq) + " met unit kind " +
                           std::to_string(theirs.kind) + " on the peer");
        }
        peer_pending_ = theirs.pending;
        return std::min(pending, theirs.pending);
    }

    [[nodiscard]] std::uint64_t peer_pending() const noexcept { return peer_pending_; }

    template <class T>
    static std::uint64_t hash_tokens(std::span<const T> tokens,
                                     std::uint64_t h = 1469598103934665603ULL) noexcept {
        for (const T t : tokens) {
            h ^= static_cast<std::uint32_t>(t);
            h *= 1099511628211ULL;
        }
        return h;
    }

  private:
    // Publishes this rank's record for its seq, waits until the peer published the same seq (or the
    // next one), and returns the peer's record for that seq.
    Record publish_and_wait(const Record& mine) {
        const std::uint64_t seq = mine.seq;
        block_->records[rank_][seq & 1] = mine;
        block_->published[rank_].store(seq, std::memory_order_release);

        const int peer = 1 - rank_;
        const auto start = std::chrono::steady_clock::now();
        std::uint32_t spins = 0;
        for (;;) {
            const std::uint64_t seen = block_->published[peer].load(std::memory_order_acquire);
            if (seen == seq || seen == seq + 1) { break; }
            if (seen > seq + 1) {
                lockstep_fatal("peer is " + std::to_string(seen - seq) + " units ahead (seq " +
                               std::to_string(seq) + ")");
            }
            if (++spins < 2048) { continue; }
            if (std::chrono::steady_clock::now() - start > timeout_) {
                lockstep_fatal("peer did not reach unit " + std::to_string(seq) + " within " +
                               std::to_string(timeout_.count()) + " s");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        return block_->records[peer][seq & 1];
    }

    Lockstep(int rank, const char* path) : rank_(rank) {
        if (rank_ != 0 && rank_ != 1) { lockstep_fatal("NINFER_TP_RANK must be 0 or 1"); }
        // Test hook: corrupt this rank's hash at the given unit to prove divergence is fatal.
        if (const char* f = std::getenv("NINFER_TP_LOCKSTEP_FAULT_UNIT")) {
            fault_unit_ = std::strtoull(f, nullptr, 10);
        }
        if (const char* t = std::getenv("NINFER_TP_LOCKSTEP_TIMEOUT_S")) {
            timeout_ = std::chrono::seconds(std::atoi(t));
        }
        const std::size_t bytes = sizeof(SharedBlock);
        int fd = -1;
        if (rank_ == 0) {
            fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
                lockstep_fatal(std::string("cannot create ") + path);
            }
        } else {
            const auto start = std::chrono::steady_clock::now();
            for (;;) {
                fd = ::open(path, O_RDWR);
                struct stat st {};
                if (fd >= 0 && ::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) >= bytes) {
                    break;
                }
                if (fd >= 0) { ::close(fd); fd = -1; }
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(600)) {
                    lockstep_fatal(std::string("timed out waiting for ") + path);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) { lockstep_fatal("mmap failed"); }
        block_ = static_cast<SharedBlock*>(p);
        if (rank_ == 0) {
            std::memset(static_cast<void*>(block_), 0, bytes);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            reinterpret_cast<std::atomic<std::uint64_t>*>(&block_->magic)
                ->store(kLockstepMagic, std::memory_order_release);
        } else {
            const auto start = std::chrono::steady_clock::now();
            while (reinterpret_cast<std::atomic<std::uint64_t>*>(&block_->magic)
                       ->load(std::memory_order_acquire) != kLockstepMagic) {
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(600)) {
                    lockstep_fatal("rank 0 never initialised the lockstep block");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        std::fprintf(stderr, "[ninfer] TP2 lockstep rank %d on %s (timeout %lld s)\n", rank_, path,
                     static_cast<long long>(timeout_.count()));
    }

    int rank_;
    std::uint64_t seq_ = 0;
    std::uint64_t peer_pending_ = 0;
    std::uint64_t fault_unit_ = 0;
    std::chrono::seconds timeout_{300};
    SharedBlock* block_ = nullptr;
};

}  // namespace ninfer::runtime::tp
