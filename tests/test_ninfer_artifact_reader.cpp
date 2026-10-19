#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;
using ninfer::test::artifact_fixture::write_v3_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_BF16S"},
                         {"layout", "row-scale-v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                    })},
    };
}

Json v3_directory(bool split) {
    return {
        {"components", {
            {"text", {{"config", {{"hidden_size", 5120}}},
                      {"resources", {{"tokenizer.json", "asset"}}},
                      {"proposal", {{"domain", "full"}}}}},
            {"vision", {{"config", Json::object()}, {"target", "text"}}},
        }},
        {"objects", Json::array({
            {{"id", "asset"}, {"kind", "resource"}, {"encoding", "raw_bytes_v1"},
             {"offset", 0}, {"bytes", 5}},
            {{"id", "q5"}, {"kind", "tensor"}, {"shape", {2, 130}},
             {"format", "q5_g64_fp16"}, {"layout", "row_split_k128_v1"},
             {"offset", 256}, {"bytes", 528}},
            {{"id", "divisor"}, {"kind", "tensor"}, {"shape", {2}},
             {"format", "fp32"}, {"layout", "contiguous_le_v1"},
             {"offset", 1024}, {"bytes", 8}},
        })},
        {"bindings", {
            {"matrix", {{"object", "q5"}}},
            {"row", {{"parts", Json::array({{{"object", "q5"}, {"range", {130, 260}}}})}}},
        }},
        {"uses", Json::array({
            {{"parameter", "row"}, {"input", "query"}, {"activation_policy", "AllowA4"},
             {"auxiliaries", {{"input_divisor", {{"object", "divisor"}}}}}},
        })},
        {"files", split
            ? Json::array({{{"path", nullptr}, {"payload_bytes", 500}},
                           {{"path", "continuation.bin"}, {"payload_bytes", 844}}})
            : Json::array({{{"path", nullptr}, {"payload_bytes", 1344}}})},
        {"metadata", {{"name", "v3-reader-fixture"}}},
        {"provenance", Json::object()},
    };
}

std::vector<std::byte> v3_payload() {
    std::vector<std::byte> bytes(1344);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = std::byte(i % 251);
    }
    return bytes;
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct   = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows     = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout fp8_rows = StorageLayout::RowScaleV1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};
    const std::array<std::uint64_t, 2> fp8_shape = {2, 4};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape) != 260) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
    expect_artifact_error([&] { tensor_encoded_size(fp8_rows, NumericFormat::NVFP4, fp8_shape); },
                          "row-scale format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape_2); },
        "row-scale rank mismatch");
}

void test_gguf_layouts() {
    using ninfer::artifact::GgufBlockShape;
    using ninfer::artifact::gguf_block_shape;
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout blocks = StorageLayout::GgufBlocksV1;
    constexpr std::array formats = {
        std::pair{NumericFormat::GGUF_Q8_0, GgufBlockShape{32, 34}},
        std::pair{NumericFormat::GGUF_Q2_K, GgufBlockShape{256, 84}},
        std::pair{NumericFormat::GGUF_Q3_K, GgufBlockShape{256, 110}},
        std::pair{NumericFormat::GGUF_Q4_K, GgufBlockShape{256, 144}},
        std::pair{NumericFormat::GGUF_Q5_K, GgufBlockShape{256, 176}},
        std::pair{NumericFormat::GGUF_Q6_K, GgufBlockShape{256, 210}},
        std::pair{NumericFormat::GGUF_IQ2_XXS, GgufBlockShape{256, 66}},
        std::pair{NumericFormat::GGUF_IQ2_XS, GgufBlockShape{256, 74}},
        std::pair{NumericFormat::GGUF_IQ2_S, GgufBlockShape{256, 82}},
        std::pair{NumericFormat::GGUF_IQ3_XXS, GgufBlockShape{256, 98}},
        std::pair{NumericFormat::GGUF_IQ3_S, GgufBlockShape{256, 110}},
        std::pair{NumericFormat::GGUF_IQ1_S, GgufBlockShape{256, 50}},
        std::pair{NumericFormat::GGUF_IQ1_M, GgufBlockShape{256, 56}},
        std::pair{NumericFormat::GGUF_IQ4_NL, GgufBlockShape{32, 18}},
        std::pair{NumericFormat::GGUF_IQ4_XS, GgufBlockShape{256, 136}},
        std::pair{NumericFormat::GGUF_Q4_0, GgufBlockShape{32, 18}},
        std::pair{NumericFormat::GGUF_Q5_0, GgufBlockShape{32, 22}},
        std::pair{NumericFormat::GGUF_Q2_0, GgufBlockShape{64, 18}},
    };
    for (const auto& [format, block] : formats) {
        if (gguf_block_shape(format).elements != block.elements ||
            gguf_block_shape(format).bytes != block.bytes) {
            throw std::runtime_error("GGUF block table mismatches upstream ggml sizes");
        }
        const std::array<std::uint64_t, 2> dims{2, 3 * block.elements};
        if (tensor_encoded_size(blocks, format, dims) != 6 * block.bytes) {
            throw std::runtime_error("GGUF encoded size incorrect");
        }
        const std::array<std::uint64_t, 2> incomplete{2, 3 * block.elements - 1};
        expect_artifact_error(
            [&] { (void)tensor_encoded_size(blocks, format, incomplete); },
            "GGUF partial block");
        const auto size = 2 * block.bytes;
        Json directory = {
            {"components", {{"text", {{"config", Json::object()}}}}},
            {"objects", Json::array({
                {{"id", "quantized"}, {"kind", "tensor"},
                 {"format", std::string(ninfer::artifact::format_name(format))},
                 {"layout", "gguf_blocks_v1"},
                 {"shape", {2, block.elements}}, {"offset", 0}, {"bytes", size}},
            })},
            {"bindings", {{"projection", {{"object", "quantized"}}}}},
            {"uses", Json::array()},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", size}}})},
        };
        const std::vector<std::byte> exact_block_bytes(size, std::byte{0x37});
        auto fixture = write_v3_fixture(directory, ninfer::artifact::format_name(format),
                                        exact_block_bytes);
        Reader reader(fixture.path);
        const auto* tensor = std::get_if<TensorDescriptor>(reader.find("quantized"));
        if (tensor == nullptr || tensor->format != format || tensor->layout != blocks ||
            reader.read_object(*reader.find("quantized")) != exact_block_bytes) {
            throw std::runtime_error("v3 GGUF format/bytes were not preserved");
        }
    }
    const std::array<std::uint64_t, 2> sample{2, 256};
    expect_artifact_error(
        [&] { (void)tensor_encoded_size(blocks, NumericFormat::BF16, sample); },
        "GGUF wrong format");
    if (tensor_encoded_size(StorageLayout::RowInterleavedV1,
                            NumericFormat::FP8_E4M3FN_ROW_FP16S,
                            std::array<std::uint64_t, 2>{3, 160}) != 486 ||
        tensor_encoded_size(StorageLayout::RowSplitK128V1,
                            NumericFormat::Q2G64_F16S,
                            std::array<std::uint64_t, 2>{2, 65}) != 264 ||
        tensor_encoded_size(StorageLayout::BlockScaleK16M128x4V1,
                            NumericFormat::NVFP4,
                            std::array<std::uint64_t, 2>{256, 64}, 2) != 9224) {
        throw std::runtime_error("v3 legacy-compatible geometry changed");
    }
    expect_artifact_error(
        [&] {
            (void)tensor_encoded_size(StorageLayout::BlockScaleK16M128x4V1,
                                      NumericFormat::NVFP4,
                                      std::array<std::uint64_t, 2>{128, 64}, 2);
        }, "NVFP4 divisor splitting scale tile");
}

void test_v3_reader(bool split) {
    const auto root = v3_directory(split);
    const auto expected = v3_payload();
    auto fixture = write_v3_fixture(root, split ? "split" : "single", expected);
    Reader reader(fixture.path);
    if (reader.container_version() != 3 || reader.artifact_id()[0] != std::byte{0x71} ||
        reader.v3_directory() == nullptr || *reader.v3_directory() != root ||
        reader.identity().model_id != "" || reader.objects().size() != 3 ||
        reader.file_bytes() != reader.payload_offset() + expected.size()) {
        throw std::runtime_error("v3 metadata was not preserved");
    }
    const auto* q5 = std::get_if<TensorDescriptor>(reader.find("q5"));
    if (q5 == nullptr || q5->format != NumericFormat::Q5G64_F16S ||
        q5->layout != StorageLayout::RowSplitK128V1 ||
        q5->shape != std::vector<std::uint64_t>{2, 130} ||
        reader.find("missing") != nullptr) {
        throw std::runtime_error("v3 object index does not match normalized tensor");
    }
    const auto result = reader.read_object(*reader.find("q5"));
    if (result.size() != 528 ||
        !std::equal(result.begin(), result.end(), expected.begin() + 256)) {
        throw std::runtime_error("v3 payload differs from original GGUF/quant data");
    }
    const auto direct = reader.read_range(0, expected.size());
    if (direct != expected || !reader.read_range(expected.size(), 0).empty()) {
        throw std::runtime_error("v3 logical payload boundary mismatch");
    }
    expect_artifact_error([&] { (void)reader.read_range(expected.size(), 1); },
                          "v3 out-of-bounds read");
    if (split) {
        const auto spans = reader.segments(256, 528);
        if (spans.size() != 2 || spans[0].file_index != 0 || spans[0].file_offset != 4352 ||
            spans[0].bytes != 244 || spans[1].file_index != 1 ||
            spans[1].file_offset != 4096 || spans[1].destination_offset != 244 ||
            spans[1].bytes != 284) {
            throw std::runtime_error("v3 nonaligned continuation mapping is wrong");
        }
        expect_artifact_error([&] { (void)reader.payload(*reader.find("q5")); },
                              "cross-part object wrongly returned contiguous span");
    } else {
        if (reader.payload(*reader.find("q5")).data.size() != 528) {
            throw std::runtime_error("v3 single-file mmap view missing");
        }
    }
    alignas(4096) std::array<std::byte, 4096> scratch{};
    if (reader.read_direct(reader.payload_offset(), scratch) != expected.size() ||
        !std::equal(expected.begin(), expected.end(), scratch.begin())) {
        throw std::runtime_error("v3 direct-read facade changed logical payload bytes");
    }
}

void test_v3_rejections() {
    const auto payload = v3_payload();
    const auto reject = [&](Json root, std::string_view label) {
        auto fixture = write_v3_fixture(root, label, payload);
        expect_artifact_error([&] { Reader reader(fixture.path); }, label);
    };
    {
        auto invalid = v3_directory(false);
        invalid["bindings"]["row"]["parts"][0]["range"] = {250, 300};
        reject(std::move(invalid), "v3_invalid_binding");
    }
    {
        auto invalid = v3_directory(false);
        invalid["components"]["text"]["resources"]["tokenizer.json"] = "q5";
        reject(std::move(invalid), "v3_wrong_resource_ref");
    }
    {
        auto invalid = v3_directory(false);
        invalid["uses"].push_back(invalid["uses"][0]);
        reject(std::move(invalid), "v3_duplicate_use");
    }
    {
        auto invalid = v3_directory(true);
        invalid["files"][1]["path"] = "../sibling";
        reject(std::move(invalid), "v3_path_traversal");
    }
    {
        auto invalid = v3_directory(true);
        invalid["files"][1]["path"] = "entry.ninfer";
        reject(std::move(invalid), "v3_duplicate_file");
    }
    {
        auto invalid = v3_directory(false);
        invalid["objects"][1]["bytes"] = 527;
        reject(std::move(invalid), "v3_wrong_encoded_size");
    }
    {
        auto fixture = write_v3_fixture(v3_directory(true), "uuid_mismatch", payload);
        std::fstream part(fixture.folder / "continuation.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        part.seekp(16);
        part.put('\0');
        part.close();
        Reader reader(fixture.path);
        expect_artifact_error([&] { (void)reader.read_object(*reader.find("q5")); },
                              "v3 continuation from another artifact");
    }
    {
        auto root = v3_directory(false);
        std::string raw = root.dump();
        const auto position = raw.find("\"components\":");
        if (position == std::string::npos) {
            throw std::runtime_error("v3 fixture lost components key");
        }
        raw.insert(position, "\"components\":{},");
        auto fixture = write_v3_fixture(root, "duplicate_json", payload, raw);
        expect_artifact_error([&] { Reader reader(fixture.path); },
                              "duplicate v3 JSON member");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 9 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 9> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8", "fp8_row",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* fp8      = std::get_if<TensorDescriptor>(reader.find("fp8_row"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1 ||
        fp8 == nullptr || fp8->shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        fp8->layout != StorageLayout::RowScaleV1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        constexpr std::array<std::uint8_t, 8> invalid_magic = {
            'I', 'N', 'V', 'A', 'L', 'I', 'D', '!',
        };
        auto fixture = write_fixture(normative_directory(), "invalid_magic", invalid_magic);
        expect_artifact_error([&] { Reader reader(fixture.path); }, "invalid magic");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_gguf_layouts();
        test_normative_fixture();
        test_common_validation();
        test_v3_reader(false);
        test_v3_reader(true);
        test_v3_rejections();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
