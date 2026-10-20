#include "runtime/tp2/nccl_collectives.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if NINFER_HAVE_NCCL
#include <nccl.h>
#include <cstdio>
#include <unistd.h>
#endif

namespace ninfer::runtime::tp2 {
namespace {

void require_collective_args(const void* input, const void* output, std::size_t count) {
    if (input == nullptr || output == nullptr || count == 0) {
        throw std::invalid_argument("TP2 collective requires nonempty device buffers");
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("TP2 collective element count exceeds int64 range");
    }
}

#if NINFER_HAVE_NCCL

void check(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("TP2 ") + operation + ": " + ncclGetErrorString(result));
    }
}

ncclUniqueId exchange_id(int rank, const std::filesystem::path& id_file,
                         std::chrono::milliseconds timeout) {
    ncclUniqueId id{};
    if (rank == 0) {
        check(ncclGetUniqueId(&id), "ncclGetUniqueId");
        const auto temporary = id_file.string() + ".rank0." + std::to_string(::getpid());
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            if (!stream.write(reinterpret_cast<const char*>(&id), sizeof(id))) {
                throw std::runtime_error("TP2 cannot publish NCCL ID at " + temporary);
            }
        }
        if (std::rename(temporary.c_str(), id_file.c_str()) != 0) {
            std::filesystem::remove(temporary);
            throw std::runtime_error("TP2 cannot publish NCCL ID at " + id_file.string());
        }
    } else {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        do {
            std::error_code ec;
            if (std::filesystem::file_size(id_file, ec) == sizeof(id) && !ec) {
                std::ifstream stream(id_file, std::ios::binary);
                if (stream.read(reinterpret_cast<char*>(&id), sizeof(id))) { return id; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        } while (std::chrono::steady_clock::now() < deadline);
        throw std::runtime_error("TP2 timed out reading NCCL rank0 ID: " + id_file.string());
    }
    return id;
}

#endif

} // namespace

NcclCollectives::NcclCollectives(int rank, int world_size)
    : rank_(rank), world_size_(world_size) {
    if (world_size_ != 2 || rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument("TP2 requires world_size=2 and rank in {0,1}");
    }
}

NcclCollectives::~NcclCollectives() {
#if NINFER_HAVE_NCCL
    if (comm_ != nullptr) { (void)ncclCommDestroy(static_cast<ncclComm_t>(comm_)); }
#endif
}

bool NcclCollectives::compiled_with_nccl() noexcept {
#if NINFER_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

void NcclCollectives::initialize(const std::filesystem::path& id_file,
                                 std::chrono::milliseconds timeout) {
    if (comm_ != nullptr) { throw std::logic_error("TP2 NCCL communicator already initialized"); }
    if (id_file.empty() || timeout <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("TP2 requires an ID file and positive publication timeout");
    }
#if NINFER_HAVE_NCCL
    const ncclUniqueId id = exchange_id(rank_, id_file, timeout);
    ncclComm_t comm = nullptr;
    check(ncclCommInitRank(&comm, world_size_, id, rank_), "ncclCommInitRank");
    comm_ = comm;
#else
    throw std::runtime_error("native TP2 requires NCCL; rebuild with NCCL headers and libnccl");
#endif
}

void NcclCollectives::sum_bf16(void* inout, std::size_t elements, cudaStream_t stream) {
    if (comm_ == nullptr) { throw std::logic_error("TP2 NCCL communicator is not initialized"); }
    require_collective_args(inout, inout, elements);
#if NINFER_HAVE_NCCL
    check(ncclAllReduce(inout, inout, elements, ncclBfloat16, ncclSum,
                        static_cast<ncclComm_t>(comm_), stream), "ncclAllReduce");
#else
    (void)stream;
#endif
}

void NcclCollectives::allgather_bf16(const void* local, void* gathered,
                                    std::size_t elements, cudaStream_t stream) {
    if (comm_ == nullptr) { throw std::logic_error("TP2 NCCL communicator is not initialized"); }
    require_collective_args(local, gathered, elements);
#if NINFER_HAVE_NCCL
    check(ncclAllGather(local, gathered, elements, ncclBfloat16,
                        static_cast<ncclComm_t>(comm_), stream), "ncclAllGather");
#else
    (void)stream;
#endif
}

} // namespace ninfer::runtime::tp2
