#include "runtime/tp2/nccl_collectives.h"

#ifdef NDEBUG
#undef NDEBUG // Assertions in these focused tests must execute in Release builds.
#endif
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>

template <typename F>
void expect_failure(F&& fn, const char* expected) {
    try { fn(); } catch (const std::exception& error) {
        assert(std::string(error.what()).find(expected) != std::string::npos);
        return;
    }
    assert(false && "expected an exception");
}

int main() {
    using ninfer::runtime::tp2::NcclCollectives;
    expect_failure([] { NcclCollectives c(-1); }, "rank");
    expect_failure([] { NcclCollectives c(2); }, "rank");
    expect_failure([] { NcclCollectives c(0, 1); }, "world_size");
    NcclCollectives rank0(0);
    NcclCollectives rank1(1);
    assert(rank0.world_size() == 2 && rank1.rank() == 1);
    assert(!rank0.initialized());
    expect_failure([&] { rank0.initialize(""); }, "ID file");
    expect_failure([&] { rank0.initialize("/tmp/unused", std::chrono::milliseconds(0)); },
                   "timeout");
    expect_failure([&] { rank0.sum_bf16(nullptr, 1, nullptr); }, "not initialized");
    expect_failure([&] { rank0.allgather_bf16(nullptr, nullptr, 1, nullptr); },
                   "not initialized");
    if (!NcclCollectives::compiled_with_nccl()) {
        expect_failure([&] { rank0.initialize("/tmp/unused"); }, "requires NCCL");
    }
}
