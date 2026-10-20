#include "targets/qwen3_6_27b_tp2/impl/tp2_comm.h"

#include "core/device.h"
#include "ninfer/ops/residual_add.h"
#include "runtime/engine/tp_lockstep.h"
#include "runtime/tp2/nccl_collectives.h"

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

std::unique_ptr<runtime::tp2::NcclCollectives> g_collectives;

int required_rank() {
    const char* raw = std::getenv("NINFER_TP_RANK");
    if (raw == nullptr || (std::string_view(raw) != "0" && std::string_view(raw) != "1")) {
        throw std::invalid_argument("native TP2 requires NINFER_TP_RANK=0 or 1");
    }
    return raw[0] - '0';
}

} // namespace

void init() {
    if (g_collectives) { return; }
    if (!runtime::tp2::NcclCollectives::compiled_with_nccl()) {
        throw std::runtime_error("native TP2 requires a rebuild with NCCL development libraries");
    }
    const char* file = std::getenv("NINFER_TP_ID_FILE");
    if (file == nullptr || file[0] == '\0') {
        throw std::invalid_argument("native TP2 requires NINFER_TP_ID_FILE");
    }
    const char* lockstep_file = std::getenv("NINFER_TP_LOCKSTEP_FILE");
    if (lockstep_file == nullptr || lockstep_file[0] == '\0') {
        throw std::invalid_argument("native TP2 requires NINFER_TP_LOCKSTEP_FILE");
    }
    auto communicator = std::make_unique<runtime::tp2::NcclCollectives>(required_rank());
    communicator->initialize(file);
    // Establish NCCL's device connection before the Qwen runtime captures its
    // decode/prefill graphs. Both ranks warm the same sequence of BF16 sizes.
    constexpr std::size_t kWarmElements = 5120 * 128;
    void* scratch = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, kWarmElements * sizeof(std::uint16_t)));
    try {
        CUDA_CHECK(cudaMemset(scratch, 0, kWarmElements * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (const std::size_t elements : {std::size_t{5120}, std::size_t{5120 * 4},
                                           std::size_t{5120 * 16}, kWarmElements}) {
            communicator->sum_bf16(scratch, elements, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
        (void)cudaFree(scratch);
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(scratch));
    g_collectives = std::move(communicator);
    (void)::ninfer::runtime::tp::Lockstep::instance();
}

int rank() {
    if (!g_collectives) { throw std::logic_error("TP2 communicator has not initialized"); }
    return g_collectives->rank();
}

void allreduce(Tensor& residual, cudaStream_t stream) {
    if (!g_collectives) { throw std::logic_error("TP2 allreduce before communicator init"); }
    if (residual.dtype != DType::BF16 || !residual.is_contiguous() || residual.data == nullptr) {
        throw std::invalid_argument("TP2 allreduce requires contiguous device BF16");
    }
    g_collectives->sum_bf16(residual.data, static_cast<std::size_t>(residual.numel()), stream);
}

void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream) {
    if (partial.dtype != DType::BF16 || residual.dtype != DType::BF16 ||
        !partial.is_contiguous() || !residual.is_contiguous() ||
        partial.numel() != residual.numel() || partial.data == nullptr || residual.data == nullptr) {
        throw std::invalid_argument("TP2 partial and residual must match BF16 geometry");
    }
    if (rank() == 0) {
        ops::residual_add(partial, residual, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(residual.data, partial.data, residual.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    allreduce(residual, stream);
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
