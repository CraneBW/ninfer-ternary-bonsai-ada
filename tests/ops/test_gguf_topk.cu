#include "ninfer/ops/linear_topk.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void check(bool result, const char* why) {
    if (!result) { throw std::runtime_error(why); }
}

void verify(bool optimized, int tokens) {
    using namespace ninfer;
    constexpr int hidden = 5120;
    constexpr int candidates = 16;
    const int rows = optimized ? 131072 : 248320;
    const std::size_t row_bytes = (hidden / 256) * 144; // GGUF Q4_K

    DeviceBuffer weight_data(row_bytes * rows);
    weight_data.fill(0); // Q4_K d/dmin zero; every projected logit is exactly zero
    Weight weight{};
    weight.qtype = QType::GGUF_Q4_K;
    weight.layout = QuantLayout::GgufBlocks;
    weight.payload = weight_data.p;
    weight.payload_bytes = weight_data.bytes;
    weight.qdata = weight_data.p;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = hidden;
    weight.shape[0] = rows;
    weight.shape[1] = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;

    std::vector<__nv_bfloat16> x_host(hidden * tokens, __float2bfloat16(1.0f));
    DeviceBuffer x_dev(x_host.size() * sizeof(__nv_bfloat16));
    DeviceBuffer ids_dev(candidates * tokens * sizeof(std::int32_t));
    DeviceBuffer scores_dev(candidates * tokens * sizeof(float));
    x_dev.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));
    Tensor x(x_dev.p, DType::BF16, {hidden, tokens});
    Tensor ids(ids_dev.p, DType::I32, {candidates, tokens});
    Tensor scores(scores_dev.p, DType::FP32, {candidates, tokens});
    WorkspaceArena workspace(ops::linear_topk_workspace_capacity_bytes(
        QType::GGUF_Q4_K, rows, hidden, 1, tokens));
    if (optimized) {
        std::vector<std::int32_t> map(rows);
        for (int row = 0; row < rows; ++row) { map[row] = rows - 1 - row; }
        DeviceBuffer map_dev(map.size() * sizeof(std::int32_t));
        map_dev.copy_from_host(map.data(), map.size() * sizeof(std::int32_t));
        Tensor row_map(map_dev.p, DType::I32, {rows, 1});
        ops::linear_topk(x, weight, row_map, ids, scores, workspace, nullptr);
        check(cudaDeviceSynchronize() == cudaSuccess, "optimized GGUF topk CUDA failure");
    } else {
        ops::linear_topk(x, weight, 248077, ids, scores, workspace, nullptr);
        check(cudaDeviceSynchronize() == cudaSuccess, "full GGUF topk CUDA failure");
    }
    std::vector<std::int32_t> id_result(candidates * tokens);
    std::vector<float> score_result(candidates * tokens);
    ids_dev.copy_to_host(id_result.data(), id_result.size() * sizeof(std::int32_t));
    scores_dev.copy_to_host(score_result.data(), score_result.size() * sizeof(float));
    for (int column = 0; column < tokens; ++column) {
        for (int rank = 0; rank < candidates; ++rank) {
            const auto offset = column * candidates + rank;
            check(id_result[offset] == rank, "GGUF topk tie-breaking or row-map mismatch");
            check(score_result[offset] == 0.0f, "GGUF zero-weight score mismatch");
        }
    }
    std::printf("GGUF Q4_K %s T=%d sorted ties PASS\n",
                optimized ? "optimized" : "full", tokens);
}

} // namespace

int main() {
    try {
        verify(false, 1);
        verify(false, 16);
        verify(true, 1);
        verify(true, 16);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF TopK error: %s\n", e.what());
        return 1;
    }
    return 0;
}
