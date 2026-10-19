#include "core/arena.h"
#include "ops/linear/gguf/gguf_dispatch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool good, const char* name) {
    if (!good) { throw std::runtime_error(name); }
}

void verify(int tokens) {
    constexpr int n = 128, k = 512;
    constexpr int block_bytes = 34, block_values = 32;
    const std::size_t row_bytes = (k / block_values) * block_bytes;
    std::vector<std::uint8_t> encoded(n * row_bytes, 0);
    std::vector<float> expected(n, 0.0f);

    // The ggml Q8_0 block is FP16 scale 1.0 + 32 signed bytes. The CPU
    // oracle decodes the mathematical weight value directly from the bytes.
    for (int row = 0; row < n; ++row) {
        for (int block = 0; block < k / block_values; ++block) {
            auto* ptr = encoded.data() + row * row_bytes + block * block_bytes;
            ptr[0] = 0;
            ptr[1] = 0x3c; // exact FP16 1.0
            for (int d = 0; d < block_values; ++d) {
                const std::int8_t q = static_cast<std::int8_t>((d + row * 3) % 9 - 4);
                ptr[2 + d] = static_cast<std::uint8_t>(q);
                expected[row] += static_cast<float>(q);
            }
        }
    }
    std::vector<__nv_bfloat16> source(k * tokens, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> result(n * tokens);
    ninfer::DeviceBuffer weights(encoded.size()), input(source.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer output(result.size() * sizeof(__nv_bfloat16));
    weights.copy_from_host(encoded.data(), encoded.size());
    input.copy_from_host(source.data(), source.size() * sizeof(__nv_bfloat16));
    ninfer::WorkspaceArena workspace(
        ninfer::ops::detail::gguf_linear_workspace_capacity_bytes(
            ninfer::QType::GGUF_Q8_0, n, k, 1, tokens));
    ninfer::Weight w{};
    w.qtype = ninfer::QType::GGUF_Q8_0;
    w.layout = ninfer::QuantLayout::GgufBlocks;
    w.qdata = weights.p;
    w.n = n;
    w.k = k;
    ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
    ninfer::Tensor y(output.p, ninfer::DType::BF16, {n, tokens});
    ninfer::ops::detail::gguf_linear_dispatch(x, w, y, workspace, nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess, "GGUF CUDA execution");
    output.copy_to_host(result.data(), result.size() * sizeof(__nv_bfloat16));
    float worst = 0.0f;
    for (int column = 0; column < tokens; ++column) {
        for (int row = 0; row < n; ++row) {
            const float got = __bfloat162float(result[column * n + row]);
            const float error = std::fabs(got - expected[row]);
            worst = std::max(worst, error);
            require(std::isfinite(got) && error < 0.75f, "GGUF Q8_0 numerical mismatch");
        }
    }
    std::printf("GGUF Q8_0 T=%d max_abs_error=%.6f PASS\n", tokens, worst);
}

void verify_real_mixed_formats(const char* artifact_path) {
    // Test actual mixed-format GSQ-RCO checkpoint bytes from the already
    // downloaded prefix; the same CUDA bridge supplies GEMV and MMQ.
    std::ifstream file(artifact_path, std::ios::binary);
    require(bool(file), "GSQ model file is unavailable");
    std::array<unsigned char, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    require(file.gcount() == 32 &&
                std::string(reinterpret_cast<const char*>(header.data()), 6) == "NINFER" &&
                header[6] == 0,
            "invalid GSQ v3 artifact header");
    require(header[7] == 3, "GSQ test expects a v3 artifact");
    std::uint64_t json_length = 0;
    for (int i = 0; i < 8; ++i) {
        json_length |= static_cast<std::uint64_t>(header[8 + i]) << (8 * i);
    }
    require(json_length > 0 && json_length < 4'000'000, "GSQ directory invalid size");
    std::string json_text(json_length, '\0');
    file.read(json_text.data(), static_cast<std::streamsize>(json_length));
    require(static_cast<std::uint64_t>(file.gcount()) == json_length,
            "GSQ directory is incomplete");
    const nlohmann::json directory = nlohmann::json::parse(json_text);

    struct FormatSample {
        const char* name;
        ninfer::QType qtype;
    };
    constexpr std::array<FormatSample, 6> samples{{
        {"gguf_iq4_xs", ninfer::QType::GGUF_IQ4_XS},
        {"gguf_iq3_s", ninfer::QType::GGUF_IQ3_S},
        {"gguf_iq3_xxs", ninfer::QType::GGUF_IQ3_XXS},
        {"gguf_iq2_xs", ninfer::QType::GGUF_IQ2_XS},
        {"gguf_q2_k", ninfer::QType::GGUF_Q2_K},
        {"gguf_q4_k", ninfer::QType::GGUF_Q4_K},
    }};
    for (const auto& sample : samples) {
    const nlohmann::json* object = nullptr;
    for (const auto& item : directory.at("objects")) {
        if (item.value("format", "") == sample.name && item.value("kind", "") == "tensor" &&
            item.at("shape").size() == 2 &&
            item.at("shape")[0].get<int>() >= 128 &&
            item.at("shape")[1].get<int>() % 256 == 0) {
            object = &item;
            break;
        }
    }
    require(object != nullptr, "GSQ mixed-format tensor missing");
    constexpr int rows = 128;
    const int k = object->at("shape")[1].get<int>();
    require(k > 0 && k % 256 == 0 && object->at("shape")[0].get<int>() >= rows,
            "GSQ checkpoint geometry invalid");
    const auto block = ninfer::gguf_block_shape(sample.qtype);
    require(block.elements > 0 && k % block.elements == 0,
            "GSQ block geometry invalid");
    const std::uint64_t bytes_per_row =
        static_cast<std::uint64_t>(k / block.elements) * block.bytes;
    const std::uint64_t start =
        ((32 + json_length + 4095) / 4096) * 4096 + object->at("offset").get<std::uint64_t>();
    require(object->at("bytes").get<std::uint64_t>() >= rows * bytes_per_row,
            "GSQ tensor payload too small");
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows * bytes_per_row));
    file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    file.read(reinterpret_cast<char*>(codes.data()),
              static_cast<std::streamsize>(codes.size()));
    require(static_cast<std::size_t>(file.gcount()) == codes.size(),
            "GSQ selected tensor data not yet downloaded");
    ninfer::DeviceBuffer weights(codes.size()), decoded(rows * k * sizeof(float));
    weights.copy_from_host(codes.data(), codes.size());
    ninfer::ops::gguf::dequantize_rows(
        ninfer::ops::detail::gguf_type(sample.qtype), weights.p, bytes_per_row, k, nullptr,
        rows, static_cast<float*>(decoded.p), k, nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess, "GGUF CUDA dequantization");
    std::vector<float> weights_f32(rows * k);
    decoded.copy_to_host(weights_f32.data(), weights_f32.size() * sizeof(float));
    std::vector<float> expected(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < k; ++c) { expected[r] += weights_f32[r * k + c]; }
    }

    ninfer::Weight w{};
    w.qtype = sample.qtype;
    w.layout = ninfer::QuantLayout::GgufBlocks;
    w.qdata = weights.p;
    w.n = rows;
    w.k = k;
    for (int tokens : {1, 16}) {
        std::vector<__nv_bfloat16> ones(k * tokens, __float2bfloat16(1.0f));
        std::vector<__nv_bfloat16> result(rows * tokens);
        ninfer::DeviceBuffer input(ones.size() * sizeof(__nv_bfloat16));
        ninfer::DeviceBuffer output(result.size() * sizeof(__nv_bfloat16));
        input.copy_from_host(ones.data(), ones.size() * sizeof(__nv_bfloat16));
        ninfer::WorkspaceArena ws(
            ninfer::ops::detail::gguf_linear_workspace_capacity_bytes(
                w.qtype, rows, k, 1, tokens));
        ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
        ninfer::Tensor y(output.p, ninfer::DType::BF16, {rows, tokens});
        ninfer::ops::detail::gguf_linear_dispatch(x, w, y, ws, nullptr);
        require(cudaDeviceSynchronize() == cudaSuccess, "GSQ CUDA product");
        output.copy_to_host(result.data(), result.size() * sizeof(__nv_bfloat16));
        float worst = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < rows; ++row) {
                const float value = __bfloat162float(result[token * rows + row]);
                const float error = std::fabs(value - expected[row]);
                worst = std::max(worst, error);
                const float tolerance = std::max(0.7f, std::fabs(expected[row]) * 0.012f);
                require(std::isfinite(value) && error < tolerance,
                        "GSQ mixed-format tensor numerical mismatch");
            }
        }
        std::printf("real GSQ %s T=%d rows=%d K=%d max_abs_error=%.6f PASS\n",
                    sample.name, tokens, rows, k, worst);
    }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        verify(1);   // GGUF vector decode
        verify(16);  // GGUF integer tensor-core prefill
        if (argc == 2) { verify_real_mixed_formats(argv[1]); }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF test failed: %s\n", e.what());
        return 1;
    }
}
