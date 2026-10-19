#include "ninfer/ops/linear_topk.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
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

void verify_nonzero(bool optimized) {
    using namespace ninfer;
    constexpr int hidden = 5120;
    constexpr int candidates = 16;
    constexpr int tokens = 16;
    constexpr std::size_t block_bytes = 144; // One 256-element GGUF Q4_K block.
    const int rows = optimized ? 131072 : 248320;
    const std::size_t row_bytes = (hidden / 256) * block_bytes;

    DeviceBuffer weight_data(row_bytes * rows);
    weight_data.fill(0);
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

    // Independent analytic reference: each marked row has the given constant
    // integer weight at input coordinates [0,32), and zero everywhere else.
    // Q4_K encodes it in one block with d=dmin=1 and either q or a min scale.
    const auto set_row = [&](int row, int coefficient) {
        std::array<std::uint8_t, block_bytes> block{};
        block[1] = 0x3c; // FP16 1.0, Q4_K superblock d.
        block[3] = 0x3c; // FP16 1.0, Q4_K superblock dmin.
        if (coefficient > 15) {
            check(coefficient == 60, "invalid oracle coefficient");
            block[4] = 4;
        } else if (coefficient > 0) {
            block[4] = 1;
        } else {
            block[8] = static_cast<std::uint8_t>(-coefficient);
        }
        if (coefficient > 0) {
            std::fill(block.begin() + 16, block.begin() + 48,
                      static_cast<std::uint8_t>(coefficient > 15 ? 15 : coefficient));
        }
        auto* destination = static_cast<std::uint8_t*>(weight_data.p) + row * row_bytes;
        check(cudaMemcpy(destination, block.data(), block.size(), cudaMemcpyHostToDevice) ==
                  cudaSuccess, "nonzero Q4_K block upload failed");
    };

    std::vector<std::pair<int, int>> marked_rows;
    for (int index = 0; index < 40; ++index) {
        const int row = 1000 + index * 3000;
        const int coefficient = index < 20 ? 1 + index / 2 : -(1 + (index - 20) / 2);
        marked_rows.emplace_back(row, coefficient);
        set_row(row, coefficient);
    }
    // Rows 248077..248319 are physically present but excluded from the full head.
    if (!optimized) { set_row(248120, 60); }

    std::vector<__nv_bfloat16> x_host(hidden * tokens, __float2bfloat16(0.0f));
    std::array<float, 4> input_values{1.0f, -1.0f, 0.5f, -0.5f};
    for (int column = 0; column < tokens; ++column) {
        for (int input = 0; input < 32; ++input) {
            x_host[column * hidden + input] = __float2bfloat16(input_values[column % 4]);
        }
    }
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
    } else {
        ops::linear_topk(x, weight, 248077, ids, scores, workspace, nullptr);
    }
    check(cudaDeviceSynchronize() == cudaSuccess, "nonzero GGUF topk CUDA failure");

    std::vector<std::int32_t> id_result(candidates * tokens);
    std::vector<float> score_result(candidates * tokens);
    ids_dev.copy_to_host(id_result.data(), id_result.size() * sizeof(std::int32_t));
    scores_dev.copy_to_host(score_result.data(), score_result.size() * sizeof(float));
    for (int column = 0; column < tokens; ++column) {
        std::vector<std::pair<float, int>> expected;
        for (const auto [row, coefficient] : marked_rows) {
            const int token_id = optimized ? rows - 1 - row : row;
            expected.emplace_back(32.0f * coefficient * input_values[column % 4], token_id);
        }
        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        for (int rank = 0; rank < candidates; ++rank) {
            const auto offset = column * candidates + rank;
            if (id_result[offset] != expected[rank].second ||
                std::fabs(score_result[offset] - expected[rank].first) > 0.1f) {
                std::fprintf(stderr, "nonzero topk %s col=%d rank=%d expected=(%d, %.3f) got=(%d, %.3f)\n",
                             optimized ? "optimized" : "full", column, rank,
                             expected[rank].second, expected[rank].first,
                             id_result[offset], score_result[offset]);
                throw std::runtime_error("nonzero GGUF topk differs from FP32 oracle");
            }
        }
    }
    std::printf("GGUF Q4_K %s T=16 nonzero FP32 oracle PASS\n",
                optimized ? "optimized" : "full");
}

} // namespace

int main() {
    try {
        verify(false, 1);
        verify(false, 16);
        verify(true, 1);
        verify(true, 16);
        verify_nonzero(false);
        verify_nonzero(true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF TopK error: %s\n", e.what());
        return 1;
    }
    return 0;
}
