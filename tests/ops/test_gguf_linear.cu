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

void verify_gathered_columns(int tokens) {
    // Real GSQ GDN output uses a permutation of the 6144 physical input
    // columns. Test that both the vector decoder and MMQ prefill obey the
    // *same* non-identity gather instead of silently using x[i].
    constexpr int n = 128, k = 512, block_values = 32, block_bytes = 34;
    constexpr std::size_t row_bytes = (k / block_values) * block_bytes;
    std::vector<std::uint8_t> encoded(n * row_bytes, 0);
    std::vector<std::int32_t> columns(k);
    std::vector<__nv_bfloat16> activations(k * tokens);
    std::vector<float> oracle(n * tokens, 0.0f);
    for (int i = 0; i < k; ++i) {
        columns[i] = (i * 197 + 23) % k; // permutation; 197 is coprime to 512
    }
    for (int token = 0; token < tokens; ++token) {
        for (int i = 0; i < k; ++i) {
            const int value = (i * 7 + token * 11) % 17 - 8;
            activations[token * k + i] = __float2bfloat16(static_cast<float>(value));
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int block = 0; block < k / block_values; ++block) {
            auto* code = encoded.data() + row * row_bytes + block * block_bytes;
            code[0] = 0;
            code[1] = 0x3c; // GGUF Q8_0 FP16 scale = exactly 1.0
            for (int i = 0; i < block_values; ++i) {
                const int physical = block * block_values + i;
                const int weight = (physical * 3 + row * 5) % 17 - 8;
                code[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(weight));
                for (int token = 0; token < tokens; ++token) {
                    oracle[token * n + row] +=
                        weight * __bfloat162float(activations[token * k + columns[physical]]);
                }
            }
        }
    }
    ninfer::DeviceBuffer weight_gpu(encoded.size());
    ninfer::DeviceBuffer input_gpu(activations.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer map_gpu(columns.size() * sizeof(std::int32_t));
    ninfer::DeviceBuffer output_gpu(oracle.size() * sizeof(__nv_bfloat16));
    weight_gpu.copy_from_host(encoded.data(), encoded.size());
    input_gpu.copy_from_host(activations.data(), activations.size() * sizeof(__nv_bfloat16));
    map_gpu.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    ninfer::Weight weight{};
    weight.qtype = ninfer::QType::GGUF_Q8_0;
    weight.layout = ninfer::QuantLayout::GgufBlocks;
    weight.qdata = weight_gpu.p;
    weight.n = n;
    weight.k = k;
    weight.input_columns = static_cast<const std::int32_t*>(map_gpu.p);
    ninfer::Tensor x(input_gpu.p, ninfer::DType::BF16, {k, tokens});
    ninfer::Tensor y(output_gpu.p, ninfer::DType::BF16, {n, tokens});
    ninfer::WorkspaceArena workspace(
        ninfer::ops::detail::gguf_linear_workspace_capacity_bytes(
            weight.qtype, n, k, 1, tokens));
    ninfer::ops::detail::gguf_linear_dispatch(x, weight, y, workspace, nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess, "gathered GGUF CUDA execution");
    std::vector<__nv_bfloat16> actual(oracle.size());
    output_gpu.copy_to_host(actual.data(), actual.size() * sizeof(__nv_bfloat16));
    float worst = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float got = __bfloat162float(actual[i]);
        const float error = std::fabs(got - oracle[i]);
        worst = std::max(worst, error);
        require(std::isfinite(got) && error <= 6.0f + 0.015f * std::fabs(oracle[i]),
                "GGUF input_columns gather numerical mismatch");
    }
    std::printf("GGUF Q8_0 gathered T=%d max_abs_error=%.6f PASS\n", tokens, worst);
}

struct RealOptions {
    bool timing = false;
    bool shared_vector = false;
    bool shared_matrix = false;
    std::string format;
    std::string entry;
    std::string dump_output;
    std::string pattern = "constant";
    int tokens = 0;

    bool filtered() const { return shared_vector || shared_matrix || !format.empty() || !entry.empty() || tokens != 0; }
};

void verify_real_mixed_formats(const char* artifact_path, const char* oracle_path,
                              const RealOptions& options) {
    std::ifstream oracle_file(oracle_path);
    require(bool(oracle_file), "CPU GGUF oracle file is unavailable");
    const auto cpu_oracle = nlohmann::json::parse(oracle_file);
    // Test actual mixed-format GSQ-RCO checkpoint bytes from the existing
    // artifact. GGML's independent NumPy CPU decoder supplies represented
    // weights, with FP64 host accumulation; no CUDA dequantizer is the oracle.
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
    constexpr std::array<FormatSample, 10> samples{{
        {"gguf_iq4_xs", ninfer::QType::GGUF_IQ4_XS},
        {"gguf_iq3_s", ninfer::QType::GGUF_IQ3_S},
        {"gguf_iq3_xxs", ninfer::QType::GGUF_IQ3_XXS},
        {"gguf_iq2_xs", ninfer::QType::GGUF_IQ2_XS},
        {"gguf_iq2_s", ninfer::QType::GGUF_IQ2_S},
        {"gguf_iq2_xxs", ninfer::QType::GGUF_IQ2_XXS},
        {"gguf_q6_k", ninfer::QType::GGUF_Q6_K},
        {"gguf_iq1_m", ninfer::QType::GGUF_IQ1_M},
        {"gguf_q2_k", ninfer::QType::GGUF_Q2_K},
        {"gguf_q4_k", ninfer::QType::GGUF_Q4_K},
    }};
    auto token_counts = cpu_oracle.value(
        "token_counts", std::vector<int>{1, 16, 128, 129, 256, 512, 513});
    if (options.tokens != 0) { token_counts = {options.tokens}; }
    require(!token_counts.empty() &&
                std::all_of(token_counts.begin(), token_counts.end(), [](int t) { return t > 0; }),
            "CPU oracle token counts must be positive");
    const auto timing_tokens = cpu_oracle.value("timing_tokens", std::vector<int>{512});
    int checked_entries = 0;
    bool dumped_output = false;
    for (const auto& entry : cpu_oracle.at("formats").items()) {
        if (!options.entry.empty() && options.entry != entry.key()) { continue; }
        const auto& oracle = entry.value();
        const auto format = oracle.value("format", entry.key());
        if (!options.format.empty() && options.format != format) { continue; }
        const auto sample = std::find_if(samples.begin(), samples.end(),
                                        [&](const auto& s) { return format == s.name; });
        require(sample != samples.end(), "CPU oracle contains an unsupported GGUF format");
        const auto offset = oracle.at("offset").get<std::uint64_t>();
        const nlohmann::json* object = nullptr;
        for (const auto& item : directory.at("objects")) {
            if (item.value("format", "") == format && item.value("kind", "") == "tensor" &&
                item.at("offset").get<std::uint64_t>() == offset) {
                object = &item;
                break;
            }
        }
        require(object != nullptr, "CPU oracle tensor offset and format are absent from artifact");
        const int rows = oracle.at("rows").get<int>();
        const int k = oracle.at("k").get<int>();
        require(object->at("shape").size() == 2 && rows > 0 && k > 0 && k % 256 == 0 &&
                    object->at("shape")[0].get<int>() >= rows && object->at("shape")[1] == k,
                "CPU oracle tensor geometry differs from artifact");
        if (oracle.contains("n")) {
            require(oracle.at("n") == object->at("shape")[0], "CPU oracle N differs from artifact");
        }
        if (oracle.contains("object_id")) {
            require(oracle.at("object_id") == object->at("id"), "CPU oracle tensor ID differs from artifact");
        }
        const auto block = ninfer::gguf_block_shape(sample->qtype);
        require(block.elements > 0 && k % block.elements == 0,
                "GSQ block geometry invalid");
        const std::uint64_t bytes_per_row =
            static_cast<std::uint64_t>(k / block.elements) * block.bytes;
        const std::uint64_t start = ((32 + json_length + 4095) / 4096) * 4096 + offset;
        require(object->at("bytes").get<std::uint64_t>() >= rows * bytes_per_row,
                "GSQ tensor payload too small");
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows * bytes_per_row));
        file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
        file.read(reinterpret_cast<char*>(codes.data()),
                  static_cast<std::streamsize>(codes.size()));
        require(static_cast<std::size_t>(file.gcount()) == codes.size(),
                "GSQ selected tensor payload is incomplete");
        ninfer::DeviceBuffer weights(codes.size());
        weights.copy_from_host(codes.data(), codes.size());
        const auto expected = oracle.at("row_sums").get<std::vector<double>>();
        require(expected.size() == static_cast<std::size_t>(rows), "CPU oracle has incorrect row count");

        ninfer::Weight w{};
        w.qtype = sample->qtype;
        w.layout = ninfer::QuantLayout::GgufBlocks;
        w.qdata = weights.p;
        w.n = rows;
        w.k = k;
        for (int tokens : token_counts) {
            std::vector<__nv_bfloat16> ones(static_cast<std::size_t>(k) * tokens);
            const auto column_scale = [](int token) { return (token % 7 - 3) * 0.25f; };
            if (options.pattern == "random") {
                // Fixed xorshift32 stream; every integer/16 in [-8, 8) is
                // exactly representable in BF16, independent of host libraries.
                std::uint32_t state = 0x6d2b79f5u;
                for (auto& value : ones) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    const int numerator = static_cast<int>(state & 255u) - 128;
                    value = __float2bfloat16(static_cast<float>(numerator) / 16.0f);
                }
            } else {
                for (int token = 0; token < tokens; ++token) {
                    std::fill_n(ones.begin() + static_cast<std::size_t>(token) * k, k,
                                __float2bfloat16(column_scale(token)));
                }
            }
            std::vector<__nv_bfloat16> result(static_cast<std::size_t>(rows) * tokens);
            ninfer::DeviceBuffer input(ones.size() * sizeof(__nv_bfloat16));
            constexpr std::size_t kOutputGuardBytes = 256;
            const std::size_t output_bytes = result.size() * sizeof(__nv_bfloat16);
            ninfer::DeviceBuffer output(output_bytes + 2 * kOutputGuardBytes);
            output.fill(0xa5);
            auto* output_data = static_cast<std::uint8_t*>(output.p) + kOutputGuardBytes;
            input.copy_from_host(ones.data(), ones.size() * sizeof(__nv_bfloat16));
            require(cudaMemset(output_data, 0xff, output_bytes) == cudaSuccess,
                    "GSQ output NaN poison");
            ninfer::WorkspaceArena ws(
                ninfer::ops::detail::gguf_linear_workspace_capacity_bytes(
                    w.qtype, rows, k, 1, tokens));
            ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
            ninfer::Tensor y(output_data, ninfer::DType::BF16, {rows, tokens});
            if (options.shared_vector) {
                require(tokens <= 8, "shared-vector qualification requires T<=8");
                ninfer::DeviceBuffer second(output_bytes);
                ninfer::Tensor duplicate(second.p, ninfer::DType::BF16, {rows, tokens});
                const std::array<ninfer::ops::detail::GgufProduct, 2> products{{
                    {&w, &y}, {&w, &duplicate}}};
                require(ninfer::ops::detail::try_gguf_shared_vector_dispatch(
                            x, products, ws, nullptr), "shared-vector route did not execute");
                require(cudaDeviceSynchronize() == cudaSuccess, "shared-vector completion");
                std::vector<__nv_bfloat16> a(result.size()), b(result.size());
                output.copy_to_host(a.data(), output_bytes, kOutputGuardBytes);
                second.copy_to_host(b.data(), output_bytes);
                require(std::memcmp(a.data(), b.data(), output_bytes) == 0,
                        "shared-vector second destination differs");
                // Exact route equivalence qualifies inherited rounding/state
                // evidence; it is ADDITIONAL to the independent FP64 oracle below.
                ninfer::ops::detail::gguf_linear_dispatch(x, w, duplicate, ws, nullptr);
                require(cudaDeviceSynchronize() == cudaSuccess, "ordinary reference completion");
                second.copy_to_host(b.data(), output_bytes);
                require(std::memcmp(a.data(), b.data(), output_bytes) == 0,
                        "shared-vector changed represented product bytes");
                std::printf("GGUF_SHARED_VECTOR entry=%s T=%d N=%d K=%d two_outputs_bitwise=1 PASS\n",
                            entry.key().c_str(), tokens, rows, k);
            } else if (options.shared_matrix) {
                require(tokens == 512 || tokens == 1024, "shared-matrix requires T512/1024");
                const int stride = rows + 16;
                const std::size_t packed_bytes = static_cast<std::size_t>(stride) * tokens * 2;
                ninfer::DeviceBuffer packed_buffer(packed_bytes), reference(output_bytes);
                require(cudaMemset(packed_buffer.p, 0xa5, packed_bytes) == cudaSuccess,
                        "shared-matrix packed guard initialization");
                ninfer::Tensor packed(packed_buffer.p, ninfer::DType::BF16, {stride, tokens});
                const std::array<ninfer::ops::detail::GgufProduct, 2> products{{
                    {&w, &y}, {&w, &packed, 8}}};
                require(ninfer::ops::detail::try_gguf_shared_matrix_dispatch(
                            x, products, ws, nullptr), "shared-matrix route did not execute");
                require(cudaDeviceSynchronize() == cudaSuccess, "shared-matrix completion");
                std::vector<__nv_bfloat16> first(result.size());
                std::vector<std::uint8_t> packed_host(packed_bytes);
                output.copy_to_host(first.data(), output_bytes, kOutputGuardBytes);
                packed_buffer.copy_to_host(packed_host.data(), packed_bytes);
                for (int column = 0; column < tokens; ++column) {
                    const auto* at = packed_host.data() + static_cast<std::size_t>(column) * stride * 2;
                    require(std::memcmp(at + 16, first.data() + static_cast<std::size_t>(column) * rows,
                                        static_cast<std::size_t>(rows) * 2) == 0,
                            "shared-matrix strided destination differs");
                    for (int byte = 0; byte < 16; ++byte) {
                        require(at[byte] == 0xa5 && at[16 + rows * 2 + byte] == 0xa5,
                                "shared-matrix inter-column guard changed");
                    }
                }
                ninfer::Tensor reference_tensor(reference.p, ninfer::DType::BF16, {rows, tokens});
                ninfer::ops::detail::gguf_linear_dispatch(x, w, reference_tensor, ws, nullptr);
                require(cudaDeviceSynchronize() == cudaSuccess, "shared-matrix ordinary completion");
                std::vector<__nv_bfloat16> ordinary(result.size());
                reference.copy_to_host(ordinary.data(), output_bytes);
                require(std::memcmp(first.data(), ordinary.data(), output_bytes) == 0,
                        "shared-matrix changed original product bytes");
                std::printf("GGUF_SHARED_MATRIX entry=%s T=%d N=%d K=%d stride=%d full_bytes=1 guards=1 PASS\n",
                            entry.key().c_str(), tokens, rows, k, stride);
            } else {
                ninfer::ops::detail::gguf_linear_dispatch(x, w, y, ws, nullptr);
            }
            require(cudaDeviceSynchronize() == cudaSuccess, "GSQ CUDA product");
            output.copy_to_host(result.data(), output_bytes, kOutputGuardBytes);
            std::array<std::uint8_t, kOutputGuardBytes> prefix, suffix;
            output.copy_to_host(prefix.data(), prefix.size());
            output.copy_to_host(suffix.data(), suffix.size(), kOutputGuardBytes + output_bytes);
            const auto canary = [](std::uint8_t value) { return value == 0xa5; };
            require(std::all_of(prefix.begin(), prefix.end(), canary) &&
                        std::all_of(suffix.begin(), suffix.end(), canary),
                    "GSQ public output guard changed");
            std::printf("GGUF_OUTPUT_GUARDS entry=%s tokens=%d prefix_bytes=%zu suffix_bytes=%zu PASS\n",
                        entry.key().c_str(), tokens, prefix.size(), suffix.size());
            float worst = 0.0f;
            if (options.pattern == "constant") {
                for (int token = 0; token < tokens; ++token) {
                    for (int row = 0; row < rows; ++row) {
                        const float value = __bfloat162float(result[static_cast<std::size_t>(token) * rows + row]);
                        const double ref = expected[row] * column_scale(token);
                        const float error = static_cast<float>(std::fabs(value - ref));
                        worst = std::max(worst, error);
                        const double tolerance = std::max(0.7, std::fabs(ref) * 0.012);
                        if (!std::isfinite(value) || error >= tolerance) {
                            std::fprintf(stderr,
                                         "GSQ mismatch entry=%s T=%d token=%d row=%d got=%.9g ref=%.9g tolerance=%.9g\n",
                                         entry.key().c_str(), tokens, token, row, value, ref, tolerance);
                        }
                        require(std::isfinite(value) && error < tolerance,
                                "GSQ mixed-format tensor numerical mismatch");
                    }
                }
                std::printf("real GSQ %s T=%d rows=%d K=%d max_abs_error=%.6f entry=%s object=%s PASS\n",
                            sample->name, tokens, rows, k, worst, entry.key().c_str(),
                            object->at("id").get_ref<const std::string&>().c_str());
            } else {
                require(std::all_of(result.begin(), result.end(), [](const auto& value) {
                            return std::isfinite(__bfloat162float(value));
                        }), "random GGUF capture contains a non-finite output");
                std::printf("real GSQ %s T=%d rows=%d K=%d entry=%s object=%s pattern=random seed=0x6d2b79f5 BITWISE_CAPTURE\n",
                            sample->name, tokens, rows, k, entry.key().c_str(),
                            object->at("id").get_ref<const std::string&>().c_str());
            }
            if (!options.dump_output.empty()) {
                require(!dumped_output && checked_entries == 0 && token_counts.size() == 1,
                        "--dump-output requires exactly one selected entry and token count");
                static_assert(sizeof(__nv_bfloat16) == 2, "BF16 capture requires two-byte elements");
                std::ofstream dump(options.dump_output, std::ios::binary | std::ios::trunc);
                require(bool(dump), "cannot open BF16 output capture");
                dump.write(reinterpret_cast<const char*>(result.data()),
                           static_cast<std::streamsize>(result.size() * sizeof(__nv_bfloat16)));
                dump.close();
                require(bool(dump), "cannot write complete BF16 output capture");
                dumped_output = true;
                std::printf("GGUF_OUTPUT_DUMP entry=%s tokens=%d rows=%d k=%d pattern=%s bytes=%zu path=%s\n",
                            entry.key().c_str(), tokens, rows, k, options.pattern.c_str(),
                            result.size() * sizeof(__nv_bfloat16), options.dump_output.c_str());
            }
            if (options.timing && (options.tokens != 0 ||
                std::find(timing_tokens.begin(), timing_tokens.end(), tokens) != timing_tokens.end())) {
                cudaEvent_t start_event, stop_event;
                require(cudaEventCreate(&start_event) == cudaSuccess, "timing start event");
                require(cudaEventCreate(&stop_event) == cudaSuccess, "timing stop event");
                std::vector<float> times;
                for (int rep = 0; rep < 9; ++rep) {
                    require(cudaEventRecord(start_event) == cudaSuccess, "timing start");
                    ninfer::ops::detail::gguf_linear_dispatch(x, w, y, ws, nullptr);
                    require(cudaEventRecord(stop_event) == cudaSuccess, "timing stop");
                    require(cudaEventSynchronize(stop_event) == cudaSuccess, "timing wait");
                    float ms = 0;
                    require(cudaEventElapsedTime(&ms, start_event, stop_event) == cudaSuccess,
                            "timing elapsed");
                    if (rep >= 2) { times.push_back(ms * 1000.0f); }
                }
                cudaEventDestroy(start_event);
                cudaEventDestroy(stop_event);
                std::sort(times.begin(), times.end());
                std::printf("GGUF_NATIVE_TIMING format=%s tokens=%d rows=%d k=%d median_us=%.3f min_us=%.3f max_us=%.3f entry=%s pattern=%s\n",
                            sample->name, tokens, rows, k, times[3], times.front(), times.back(),
                            entry.key().c_str(), options.pattern.c_str());
            }
        }
        ++checked_entries;
    }
    require(checked_entries > 0, "CPU oracle filters selected no tensor entries");
    require(options.dump_output.empty() || dumped_output, "BF16 capture selected no output");
}

} // namespace

int main(int argc, char** argv) {
    try {
        RealOptions options;
        if (argc == 2) {
            throw std::invalid_argument("real GSQ check requires artifact and CPU oracle JSON");
        }
        for (int arg = 3; arg < argc; ++arg) {
            const std::string name = argv[arg];
            if (name == "--time") {
                options.timing = true;
            } else if (name == "--shared-vector") {
                options.shared_vector = true;
            } else if (name == "--shared-matrix") {
                options.shared_matrix = true;
            } else if (name == "--format" || name == "--entry" || name == "--tokens" ||
                       name == "--dump-output" || name == "--pattern") {
                require(arg + 1 < argc, "GGUF filter option is missing a value");
                const std::string value = argv[++arg];
                if (name == "--format") { options.format = value; }
                if (name == "--entry") { options.entry = value; }
                if (name == "--dump-output") {
                    require(!value.empty(), "--dump-output requires a path");
                    options.dump_output = value;
                }
                if (name == "--pattern") { options.pattern = value; }
                if (name == "--tokens") {
                    std::size_t end = 0;
                    options.tokens = std::stoi(value, &end);
                    require(end == value.size() && options.tokens > 0,
                            "--tokens requires a positive integer");
                }
            } else {
                throw std::invalid_argument("unknown GGUF test option: " + name);
            }
        }
        require(options.pattern == "constant" || options.pattern == "random",
                "--pattern must be constant or random");
        if (!options.dump_output.empty()) {
            require(!options.entry.empty() && options.tokens > 0,
                    "--dump-output requires explicit --entry and --tokens");
            require(options.dump_output != argv[1] && options.dump_output != argv[2],
                    "BF16 output capture must differ from the artifact and CPU oracle paths");
        }
        require(options.pattern != "random" || options.timing || !options.dump_output.empty() || options.shared_matrix,
                "--pattern random is only available for --dump-output or --time");
        // Filters and random captures skip the unrelated constant synthetic
        // cases so their output cannot be mistaken for the captured evidence.
        if (!options.filtered() && options.pattern == "constant") {
            verify(1);   // GGUF vector decode
            verify(16);  // GGUF integer tensor-core prefill
            verify_gathered_columns(1);
            verify_gathered_columns(16);
            verify(512);
            verify_gathered_columns(512);
        }
        if (argc >= 3) {
            verify_real_mixed_formats(argv[1], argv[2], options);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF test failed: %s\n", e.what());
        return 1;
    }
}
