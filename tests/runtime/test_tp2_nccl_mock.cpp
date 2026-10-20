#include "runtime/tp2/nccl_collectives.h"
#include "runtime/mock_nccl/nccl.h"

#ifdef NDEBUG
#undef NDEBUG // Assertions in these focused tests must execute in Release builds.
#endif
#include <cassert>
#include <cstdint>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
int created = 0;
int destroyed = 0;
int reductions = 0;
int gathers = 0;
bool fail_next_reduce = false;
} // namespace

extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
    for (char& c : id->internal) { c = '\x42'; }
    return ncclSuccess;
}
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int world, ncclUniqueId id, int rank) {
    assert(world == 2 && (rank == 0 || rank == 1) && id.internal[0] == '\x42');
    *comm = reinterpret_cast<ncclComm_t>(static_cast<std::uintptr_t>(rank + 1));
    ++created;
    return ncclSuccess;
}
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    assert(comm != nullptr);
    ++destroyed;
    return ncclSuccess;
}
extern "C" ncclResult_t ncclAllReduce(const void* src, void* dst, size_t n,
                                        ncclDataType_t dtype, ncclRedOp_t op,
                                        ncclComm_t comm, cudaStream_t) {
    assert(src == dst && src != nullptr && n == 512 && dtype == ncclBfloat16 &&
           op == ncclSum && comm != nullptr);
    ++reductions;
    if (fail_next_reduce) { fail_next_reduce = false; return ncclInvalidArgument; }
    return ncclSuccess;
}
extern "C" ncclResult_t ncclAllGather(const void* src, void* dst, size_t n,
                                        ncclDataType_t dtype, ncclComm_t comm, cudaStream_t) {
    assert(src != dst && src != nullptr && dst != nullptr && n == 512 &&
           dtype == ncclBfloat16 && comm != nullptr);
    ++gathers;
    return ncclSuccess;
}
extern "C" const char* ncclGetErrorString(ncclResult_t result) {
    return result == ncclInvalidArgument ? "mock invalid argument" : "success";
}

int main() {
    using ninfer::runtime::tp2::NcclCollectives;
    assert(NcclCollectives::compiled_with_nccl());
    const auto file = std::filesystem::temp_directory_path() /
                      ("ninfer_tp2_mock_" + std::to_string(::getpid()));
    std::filesystem::remove(file);
    {
        NcclCollectives rank0(0), rank1(1);
        rank0.initialize(file);
        rank1.initialize(file, std::chrono::milliseconds(10));
        assert(rank0.initialized() && rank1.initialized() && created == 2);
        try { rank0.initialize(file); assert(false); }
        catch (const std::logic_error&) {}
        auto* input = reinterpret_cast<void*>(0x1000);
        auto* output = reinterpret_cast<void*>(0x2000);
        rank0.sum_bf16(input, 512, nullptr);
        rank1.allgather_bf16(input, output, 512, nullptr);
        assert(reductions == 1 && gathers == 1);
        fail_next_reduce = true;
        try { rank0.sum_bf16(input, 512, nullptr); assert(false); }
        catch (const std::runtime_error& ex) {
            assert(std::string(ex.what()).find("mock invalid argument") != std::string::npos);
        }
        try { rank0.allgather_bf16(nullptr, output, 512, nullptr); assert(false); }
        catch (const std::invalid_argument&) {}
    }
    assert(destroyed == 2);
    std::filesystem::remove(file);
}
