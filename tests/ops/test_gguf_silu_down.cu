// Reuses the R29 real-artifact fixture, extending its full-byte and FP64
// qualification to the wide-prefill activation route. No model inference.
#include "core/arena.h"
#include "ops/linear/gguf/gguf_dispatch.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/residual_add.h"
#include <cstdlib>
#include <cstring>

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

namespace ninfer::ops::gguf {
// Internal bridge API also used by production dispatch, no public ABI added.
void quantize_matrix_silu_mul_activation(GgmlType, const __nv_bfloat16*,
    const __nv_bfloat16*, int, int, void*, cudaStream_t);
}
namespace {

void require(bool good, const char* name) {
    if (!good) { throw std::runtime_error(name); }
}

struct RealOptions {
    bool timing = false;
    std::string format;
    std::string entry;
    std::string dump_output;
    std::string pattern = "constant";
    int tokens = 0;

    bool filtered() const { return !format.empty() || !entry.empty() || tokens != 0; }
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
    constexpr std::array<FormatSample, 7> samples{{
        {"gguf_iq4_xs", ninfer::QType::GGUF_IQ4_XS},
        {"gguf_iq3_s", ninfer::QType::GGUF_IQ3_S},
        {"gguf_iq3_xxs", ninfer::QType::GGUF_IQ3_XXS},
        {"gguf_iq2_xs", ninfer::QType::GGUF_IQ2_XS},
        {"gguf_iq2_s", ninfer::QType::GGUF_IQ2_S},
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
            constexpr std::size_t output_guard = 256;
            ninfer::DeviceBuffer output(result.size() * sizeof(__nv_bfloat16) + 2 * output_guard);
            output.fill(0xa5);
            auto* output_data = static_cast<std::uint8_t*>(output.p) + output_guard;
            require(cudaMemset(output_data, 0xff, result.size()*2) == cudaSuccess, "output initialize");
            input.copy_from_host(ones.data(), ones.size() * sizeof(__nv_bfloat16));
            ninfer::WorkspaceArena ws(
                ninfer::ops::detail::gguf_linear_workspace_capacity_bytes(
                    w.qtype, rows, k, 1, tokens));
            ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
            ninfer::Tensor y(output_data, ninfer::DType::BF16, {rows, tokens});
            require(rows == 5120 && k == 17408 && (tokens == 256 || tokens == 512 || tokens == 1024),
                    "R29 fusion fixture is only actual main down N5120 K17408 T256/512/1024");
            const char* policy_raw = std::getenv("NINFER_GGUF_SILU_MMQ");
            const bool fused = policy_raw && policy_raw[0] == '1' && policy_raw[1] == '\0';
            std::vector<__nv_bfloat16> ups(ones.size(), __float2bfloat16(1.0f));
            if (options.pattern == "random") {
                std::uint32_t state=0x9e3779b9u;
                for (auto& value:ups) { state^=state<<13;state^=state>>17;state^=state<<5;
                    value=__float2bfloat16(float(int(state&255u)-128)/16.0f); }
            }
            ninfer::DeviceBuffer up_gpu(ups.size()*2), activation_gpu(ones.size()*2);
            up_gpu.copy_from_host(ups.data(),ups.size()*2);
            ninfer::Tensor up(up_gpu.p,ninfer::DType::BF16,{k,tokens});
            ninfer::Tensor activation(activation_gpu.p,ninfer::DType::BF16,{k,tokens});
            // The old producer runs in both fixture arms only as a mathematical
            // reference. Production candidate does not launch or store this.
            ninfer::ops::silu_mul(x,up,activation,nullptr);
            const std::size_t qbytes=ninfer::ops::gguf::matrix_activation_bytes(k,tokens);
            constexpr std::size_t qguard=256;
            ninfer::DeviceBuffer quant_gpu(qbytes+2*qguard);
            quant_gpu.fill(0xa5);
            auto* quant_codes=static_cast<std::uint8_t*>(quant_gpu.p)+qguard;
            const auto type=ninfer::ops::detail::gguf_type(w.qtype);
            if (fused) {
                ninfer::ops::gguf::quantize_matrix_silu_mul_activation(type,
                    static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(up.data),
                    k,tokens,quant_codes,nullptr);
                ninfer::ops::detail::gguf_linear_silu_mul_down_dispatch(x,up,w,y,ws,nullptr);
            } else {
                ninfer::ops::gguf::quantize_matrix_activation(type,
                    static_cast<const __nv_bfloat16*>(activation.data),k,tokens,nullptr,quant_codes,nullptr);
                ninfer::ops::detail::gguf_linear_dispatch(activation,w,y,ws,nullptr);
            }
            require(cudaDeviceSynchronize() == cudaSuccess, "GSQ CUDA product");
            std::vector<std::uint8_t> quant_host(qbytes+2*qguard);
            quant_gpu.copy_to_host(quant_host.data(),quant_host.size());
            require(std::all_of(quant_host.begin(),quant_host.begin()+qguard,[](auto v){return v==0xa5;}) &&
                    std::all_of(quant_host.end()-qguard,quant_host.end(),[](auto v){return v==0xa5;}),
                    "R29 complete quantizer guards");
            std::vector<__nv_bfloat16> activation_host(ones.size());
            activation_gpu.copy_to_host(activation_host.data(),activation_host.size()*2);
            const auto cpu_activation = [&](int token) {
                const float gate=column_scale(token);
                return __bfloat162float(__float2bfloat16(gate/(1.0f+std::exp(-gate))));
            };
            if (options.pattern == "constant") {
                for (int token=0;token<tokens;++token) for(int i=0;i<k;++i)
                    require(__bfloat162float(activation_host[std::size_t(token)*k+i])==cpu_activation(token),
                            "R29 independent host BF16 SiLU activation");
            }
            std::vector<__nv_bfloat16> residual_host(result.size());
            for(std::size_t i=0;i<residual_host.size();++i)
                residual_host[i]=__float2bfloat16(float(int(i%17)-8)*.25f);
            ninfer::DeviceBuffer residual_gpu(residual_host.size()*2);
            residual_gpu.copy_from_host(residual_host.data(),residual_host.size()*2);
            ninfer::Tensor residual(residual_gpu.p,ninfer::DType::BF16,{rows,tokens});
            ninfer::ops::residual_add(y,residual,nullptr);
            require(cudaDeviceSynchronize()==cudaSuccess,"original residual completion");
            residual_gpu.copy_to_host(residual_host.data(),residual_host.size()*2);
            const auto capture_extra = [&](const std::string& suffix,const void* data,std::size_t count) {
                require(!options.dump_output.empty(),"R29 fixture requires full capture path");
                std::ofstream extra(options.dump_output+suffix,std::ios::binary|std::ios::trunc);
                extra.write(static_cast<const char*>(data),static_cast<std::streamsize>(count));
                extra.close();require(bool(extra),"R29 complete extra capture");
            };
            capture_extra(".q8",quant_host.data(),quant_host.size());
            capture_extra(".residual.bf16",residual_host.data(),residual_host.size()*2);
            std::printf("R29_FUSION_BYTES policy=%d type=%d N=%d K=%d T=%d quant_with_guards=%zu projected=%zu residual=%zu BF16_BOUNDARIES=PASS\n",
                int(fused),int(type),rows,k,tokens,quant_host.size(),result.size()*2,residual_host.size()*2);
            std::vector<std::uint8_t> guarded_output(result.size()*2 + 2*output_guard);
            output.copy_to_host(guarded_output.data(), guarded_output.size());
            require(std::all_of(guarded_output.begin(), guarded_output.begin()+output_guard, [](auto v){return v==0xa5;}) &&
                    std::all_of(guarded_output.end()-output_guard, guarded_output.end(), [](auto v){return v==0xa5;}),
                    "complete projected output guards");
            std::memcpy(result.data(), guarded_output.data()+output_guard, result.size()*2);
            float worst = 0.0f;
            if (options.pattern == "constant") {
                for (int token = 0; token < tokens; ++token) {
                    for (int row = 0; row < rows; ++row) {
                        const float value = __bfloat162float(result[static_cast<std::size_t>(token) * rows + row]);
                        const double ref = expected[row] * cpu_activation(token);
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

        }
        ++checked_entries;
    }
    require(checked_entries > 0, "CPU oracle filters selected no tensor entries");
    require(options.dump_output.empty() || dumped_output, "BF16 capture selected no output");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::fprintf(stderr,
                     "SKIP: real GSQ artifact and independent CPU oracle required; "
                     "run tools.bench.gguf_silu_qualification with explicit paths\n");
        return 77;
    }
    try {
        RealOptions options;
        if (argc == 2) {
            throw std::invalid_argument("real GSQ check requires artifact and CPU oracle JSON");
        }
        for (int arg = 3; arg < argc; ++arg) {
            const std::string name = argv[arg];
            if (name == "--time") {
                options.timing = true;
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
        require(!options.timing && (options.tokens == 256 || options.tokens == 512 || options.tokens == 1024) && !options.entry.empty() && !options.dump_output.empty(),
                "R29 fixture requires --entry --tokens 256|512|1024 --dump-output; timing disabled");
        require(options.pattern == "constant" || options.pattern == "random",
                "--pattern must be constant or random");
        if (!options.dump_output.empty()) {
            require(!options.entry.empty() && options.tokens > 0,
                    "--dump-output requires explicit --entry and --tokens");
            require(options.dump_output != argv[1] && options.dump_output != argv[2],
                    "BF16 output capture must differ from the artifact and CPU oracle paths");
        }
        require(options.pattern != "random" || options.timing || !options.dump_output.empty(),
                "--pattern random is only available for --dump-output or --time");
        if (argc >= 3) {
            verify_real_mixed_formats(argv[1], argv[2], options);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF test failed: %s\n", e.what());
        return 1;
    }
}
