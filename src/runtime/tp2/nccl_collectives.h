#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <filesystem>

namespace ninfer::runtime::tp2 {

// Native two-process CUDA/NCCL data plane. Each rank owns one CUDA device and
// joins the same communicator before launching any graph-captured collectives.
// This is deliberately distinct from the HTTP proxy and v3 shard writer.
class NcclCollectives {
public:
    explicit NcclCollectives(int rank, int world_size = 2);
    ~NcclCollectives();

    NcclCollectives(const NcclCollectives&) = delete;
    NcclCollectives& operator=(const NcclCollectives&) = delete;

    [[nodiscard]] static bool compiled_with_nccl() noexcept;
    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int world_size() const noexcept { return world_size_; }
    [[nodiscard]] bool initialized() const noexcept { return comm_ != nullptr; }

    // Rank 0 atomically publishes a NCCL unique ID; rank 1 reads it. The
    // supervisor must remove the prior ID file before starting a new session.
    // Both ranks must join simultaneously. The timeout bounds ID publication,
    // not NCCL's own two-rank communicator establishment.
    void initialize(const std::filesystem::path& id_file,
                    std::chrono::milliseconds id_timeout = std::chrono::seconds(60));

    // Inputs and outputs are device-resident, contiguous BF16 buffers. Counts
    // are per rank; all-gather writes world_size * elements in rank order.
    // Call collectives in an identical order on both rank streams.
    void sum_bf16(void* inout, std::size_t elements, cudaStream_t stream);
    void allgather_bf16(const void* local, void* gathered, std::size_t elements,
                        cudaStream_t stream);

private:
    int rank_;
    int world_size_;
    void* comm_ = nullptr;
};

} // namespace ninfer::runtime::tp2
