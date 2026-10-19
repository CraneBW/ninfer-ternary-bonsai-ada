#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

ninfer::targets::qwen3_6::StartupFeatures
features(ninfer::SpeculativeBackend backend,
         ninfer::ProposalHead proposal_head = ninfer::ProposalHead::Full) {
    return {
        .vision        = false,
        .speculative   = backend,
        .proposal_head = proposal_head,
    };
}

bool is_device_object(const ninfer::artifact::MaterializationPlan& plan,
                      ninfer::artifact::ObjectHandle handle) {
    return std::ranges::any_of(plan.device_objects, [handle](const auto& object) {
        return object.object.index == handle.index;
    });
}

std::size_t dflash2_device_objects(const ninfer::artifact::Reader& reader,
                                   const ninfer::artifact::MaterializationPlan& plan) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plan.device_objects, [&](const auto& item) {
            return ninfer::artifact::object_name(reader.objects().at(item.object.index))
                .starts_with("dflash2/");
        }));
}

bool valid_divisors(const WeightPlan& weight) {
    if (weight.format != NumericFormat::NVFP4) { return false; }
    const float weight_divisor = std::bit_cast<float>(weight.weight_scale_divisor_bits);
    const float input_divisor  = std::bit_cast<float>(weight.input_scale_divisor_bits);
    return std::isfinite(weight_divisor) && weight_divisor > 0.0F && std::isfinite(input_divisor) &&
           input_divisor > 0.0F;
}

using Json = ninfer::test::artifact_fixture::Json;

Json small_shape_directory(const ModelShape& shape, bool fused) {
    Json directory = {
        {"identity", {{"model_id", "shape-fixture"}, {"weights_id", "fixture"}}},
        {"objects", Json::array()},
    };
    std::uint64_t offset = 0;
    const auto tensor = [&](std::string name, std::initializer_list<std::uint64_t> dims) {
        std::uint64_t elements = 1;
        for (const std::uint64_t dim : dims) { elements *= dim; }
        offset                    = ninfer::test::artifact_fixture::align_up(offset, 256);
        const std::uint64_t bytes = elements * sizeof(std::uint16_t);
        directory["objects"].push_back({{"name", std::move(name)},
                                        {"kind", "tensor"},
                                        {"shape", dims},
                                        {"format", "BF16"},
                                        {"layout", "contiguous-le-v1"},
                                        {"offset", offset},
                                        {"bytes", bytes}});
        offset += bytes;
    };

    tensor("text/token_embedding", {static_cast<std::uint64_t>(shape.output_rows),
                                     static_cast<std::uint64_t>(shape.hidden)});
    tensor("text/final_norm", {static_cast<std::uint64_t>(shape.hidden)});
    tensor("text/output_head", {static_cast<std::uint64_t>(shape.output_rows),
                                 static_cast<std::uint64_t>(shape.hidden)});
    for (int layer = 0; layer < shape.layers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        tensor(prefix + "input_norm", {static_cast<std::uint64_t>(shape.hidden)});
        tensor(prefix + "post_attention_norm", {static_cast<std::uint64_t>(shape.hidden)});
        tensor(prefix + "mlp/gate_up", {static_cast<std::uint64_t>(2 * shape.intermediate),
                                        static_cast<std::uint64_t>(shape.hidden)});
        tensor(prefix + "mlp/down", {static_cast<std::uint64_t>(shape.hidden),
                                     static_cast<std::uint64_t>(shape.intermediate)});
        if (shape.is_full_attention(layer)) {
            if (fused) {
                tensor(prefix + "attention/query_key_gate_value",
                       {static_cast<std::uint64_t>(shape.attention_input_rows()),
                        static_cast<std::uint64_t>(shape.hidden)});
            } else {
                tensor(prefix + "attention/query_key",
                       {static_cast<std::uint64_t>(shape.query_size() + shape.kv_size()),
                        static_cast<std::uint64_t>(shape.hidden)});
                tensor(prefix + "attention/gate_value",
                       {static_cast<std::uint64_t>(shape.query_size() + shape.kv_size()),
                        static_cast<std::uint64_t>(shape.hidden)});
            }
            tensor(prefix + "attention/query_norm", {static_cast<std::uint64_t>(shape.head_dim)});
            tensor(prefix + "attention/key_norm", {static_cast<std::uint64_t>(shape.head_dim)});
            tensor(prefix + "attention/output", {static_cast<std::uint64_t>(shape.hidden),
                                                  static_cast<std::uint64_t>(shape.query_size())});
        } else {
            tensor(prefix + "gdn/a_log", {static_cast<std::uint64_t>(shape.gdn_value_heads)});
            tensor(prefix + "gdn/dt_bias", {static_cast<std::uint64_t>(shape.gdn_value_heads)});
            tensor(prefix + "gdn/convolution",
                   {static_cast<std::uint64_t>(shape.gdn_conv_kernel),
                    static_cast<std::uint64_t>(shape.convolution_dim())});
            if (fused) {
                tensor(prefix + "gdn/a_b_projection",
                       {static_cast<std::uint64_t>(2 * shape.gdn_value_heads),
                        static_cast<std::uint64_t>(shape.hidden)});
                tensor(prefix + "gdn/query_key_value_z",
                       {static_cast<std::uint64_t>(shape.gdn_input_rows()),
                        static_cast<std::uint64_t>(shape.hidden)});
            } else {
                tensor(prefix + "gdn/a_projection",
                       {static_cast<std::uint64_t>(shape.gdn_value_heads),
                        static_cast<std::uint64_t>(shape.hidden)});
                tensor(prefix + "gdn/b_projection",
                       {static_cast<std::uint64_t>(shape.gdn_value_heads),
                        static_cast<std::uint64_t>(shape.hidden)});
                tensor(prefix + "gdn/query_key",
                       {static_cast<std::uint64_t>(2 * shape.key_dim()),
                        static_cast<std::uint64_t>(shape.hidden)});
                tensor(prefix + "gdn/value_z",
                       {static_cast<std::uint64_t>(2 * shape.value_dim()),
                        static_cast<std::uint64_t>(shape.hidden)});
            }
            tensor(prefix + "gdn/norm", {static_cast<std::uint64_t>(shape.gdn_value_head_dim)});
            tensor(prefix + "gdn/output", {static_cast<std::uint64_t>(shape.hidden),
                                            static_cast<std::uint64_t>(shape.value_dim())});
        }
    }
    return directory;
}

int verify_self_describing_shape() {
    ModelShape small           = kModelShape;
    small.hidden               = 4;
    small.layers               = 4;
    small.intermediate         = 8;
    small.output_rows          = 16;
    small.gdn_conv_kernel      = 2;
    small.gdn_key_heads        = 1;
    small.gdn_key_head_dim     = 2;
    small.gdn_value_heads      = 2;
    small.gdn_value_head_dim   = 2;
    small.query_heads          = 2;
    small.kv_heads             = 1;
    small.head_dim             = 2;

    const auto check_fixture = [&](const Json& directory, std::string_view suffix,
                                   std::string_view expected_error) {
        auto file = ninfer::test::artifact_fixture::write_fixture(directory, suffix);
        ninfer::artifact::Reader reader(file.path);
        ninfer::artifact::Binder binder(reader);
        try {
            validate_artifact_model_shape(binder, small);
            if (!expected_error.empty()) {
                std::cerr << "shape fixture accepted " << suffix << '\n';
                return false;
            }
            // Validation is read-only: the second invocation must remain valid.
            validate_artifact_model_shape(binder, small);
            return true;
        } catch (const ninfer::artifact::ArtifactError& error) {
            if (!expected_error.empty() &&
                std::string_view(error.what()).find(expected_error) != std::string_view::npos) {
                return true;
            }
            std::cerr << "unexpected shape error on " << suffix << ": " << error.what() << '\n';
            return false;
        }
    };

    const Json split = small_shape_directory(small, false);
    const Json fused = small_shape_directory(small, true);
    if (!check_fixture(split, "model_shape_split", "") ||
        !check_fixture(fused, "model_shape_fused", "")) {
        return 1;
    }

    // The production load route must reject a smaller model before requesting even its
    // frontend resources, and the diagnostic must identify the incompatible tensor.
    {
        auto file = ninfer::test::artifact_fixture::write_fixture(split, "model_shape_load_reject");
        ninfer::artifact::Reader reader(file.path);
        ninfer::artifact::Binder binder(reader);
        bool rejected = false;
        try {
            (void)bind_artifact(binder, WeightsProfile::Qwen36GroupwiseInt, all_features());
        } catch (const ninfer::artifact::ArtifactError& error) {
            rejected = std::string_view(error.what()).find("text/token_embedding") !=
                       std::string_view::npos;
        }
        if (!rejected) {
            std::cerr << "27B load did not reject the smaller model shape early\n";
            return 1;
        }
    }

    auto wrong_width = split;
    for (auto& object : wrong_width["objects"]) {
        if (object.at("name") == "text/layers/2/mlp/down") {
            object["shape"] = {2, 16}; // Same encoded byte count, different geometry.
        }
    }
    if (!check_fixture(wrong_width, "model_shape_wrong_width", "text/layers/2/mlp/down")) {
        return 1;
    }

    auto extra_layer = split;
    const auto& last = extra_layer["objects"].back();
    const std::uint64_t next_offset = ninfer::test::artifact_fixture::align_up(
        last.at("offset").get<std::uint64_t>() + last.at("bytes").get<std::uint64_t>(), 256);
    extra_layer["objects"].push_back({{"name", "text/layers/4/input_norm"},
                                      {"kind", "tensor"},
                                      {"shape", {4}},
                                      {"format", "BF16"},
                                      {"layout", "contiguous-le-v1"},
                                      {"offset", next_offset},
                                      {"bytes", 8}});
    if (!check_fixture(extra_layer, "model_shape_extra_layer", "more than 4 text layers")) {
        return 1;
    }

    auto wrong_topology = fused;
    for (auto it = wrong_topology["objects"].begin(); it != wrong_topology["objects"].end();
         ++it) {
        if (it->at("name") == "text/layers/3/attention/query_norm") {
            wrong_topology["objects"].erase(it);
            break;
        }
    }
    if (!check_fixture(wrong_topology, "model_shape_wrong_topology",
                       "text/layers/3/attention/query_norm")) {
        return 1;
    }
    return 0;
}

int verify_real_v3_gsq_header() {
    // Verify the published 27B v3 semantic directory without opening its incomplete 15GB
    // payload. The 376800-byte JSON header contains the whole text tensor inventory.
    const auto path = std::filesystem::path(NINFER_SOURCE_DIR) / ".." / ".." /
                      "real-models" / "qwen3_8_gsq_rco_iq3_s_v3.ninfer";
    if (!std::filesystem::is_regular_file(path)) { return 0; }
    std::ifstream file(path, std::ios::binary);
    std::array<unsigned char, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    if (!file || !std::equal(magic.begin(), magic.end(), header.begin())) {
        std::cerr << "real GSQ v3 header has invalid magic\n";
        return 1;
    }
    std::uint64_t bytes = 0;
    for (int i = 0; i < 8; ++i) { bytes |= std::uint64_t(header[8 + i]) << (8 * i); }
    if (bytes == 0 || bytes > 1024 * 1024) {
        std::cerr << "real GSQ v3 JSON header has invalid length\n";
        return 1;
    }
    std::string json(static_cast<std::size_t>(bytes), '\0');
    file.read(json.data(), static_cast<std::streamsize>(json.size()));
    if (!file) {
        std::cerr << "real GSQ JSON directory not fully downloaded\n";
        return 1;
    }
    const auto root = nlohmann::json::parse(json);
    const V3TextMetadata metadata = inspect_v3_text_metadata(root);
    if (metadata.tensors.size() != 963 || metadata.shape.hidden != TextConfig::hidden ||
        root.at("bindings").size() != 1513) {
        std::cerr << "GSQ v3 text semantic inventory size differs\n";
        return 1;
    }
    const auto find = [&](std::string_view name) -> const V3LogicalTensor& {
        const auto it = std::find_if(metadata.tensors.begin(), metadata.tensors.end(),
                                     [&](const auto& t) { return t.parameter == name; });
        if (it == metadata.tensors.end()) {
            throw std::runtime_error("real GSQ text missing " + std::string(name));
        }
        return *it;
    };
    const auto& shared_query = find("text/layers/0/gdn/query");
    const auto& shared_z = find("text/layers/0/gdn/z");
    const auto& mixed_gate = find("text/layers/0/mlp/gate");
    const auto& mixed_up = find("text/layers/0/mlp/up");
    const auto& reordered = find("text/layers/0/gdn/output");
    if (shared_query.object_id != shared_z.object_id ||
        shared_query.parent_rows != 16384 || shared_query.first_row != 0 ||
        shared_z.first_row != 10240 || shared_z.rows != 6144 ||
        shared_query.format != NumericFormat::GGUF_IQ4_XS ||
        mixed_gate.format != NumericFormat::GGUF_IQ2_XS ||
        mixed_up.format != NumericFormat::GGUF_IQ2_XXS ||
        reordered.input_columns_object != "auxiliary/000000") {
        std::cerr << "GSQ semantic parent slices, mixed formats or Use were lost\n";
        return 1;
    }
    const auto reject_v3_metadata = [&](nlohmann::json invalid, std::string_view label) {
        try {
            (void)inspect_v3_text_metadata(invalid);
        } catch (const ninfer::artifact::ArtifactError&) {
            return true;
        }
        std::cerr << "GSQ v3 metadata accepted invalid " << label << '\n';
        return false;
    };
    {
        auto invalid = root;
        invalid["components"]["text"]["config"]["hidden_size"] = 4096;
        if (!reject_v3_metadata(std::move(invalid), "hidden_size")) { return 1; }
    }
    {
        auto invalid = root;
        auto& end = invalid["bindings"]["text/layers/0/gdn/query"]["parts"][0]["range"][1];
        end = end.get<std::uint64_t>() + TextConfig::hidden;
        if (!reject_v3_metadata(std::move(invalid), "shared parent slice")) { return 1; }
    }
    {
        auto invalid = root;
        invalid["bindings"].erase("text/layers/3/attention/key");
        if (!reject_v3_metadata(std::move(invalid), "missing attention/key")) { return 1; }
    }

    // A sparse file supplies the declared 15GB *virtual* payload length with only the
    // downloaded header+directory occupying disk blocks. This exercises the production
    // v3 Reader -> Binder -> target27B LoadPlan path, including parent de-duplication,
    // without loading a single model weight or allocating GPU memory.
    const auto sparse_path = std::filesystem::temp_directory_path() /
        ("ninfer_gsq_v3_sparse_" + std::to_string(std::random_device{}()) + ".ninfer");
    auto sparse = ninfer::test::artifact_fixture::TemporaryArtifact{sparse_path};
    {
        std::ofstream output(sparse.path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(header.data()), header.size());
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!output) { throw std::runtime_error("cannot write v3 metadata fixture"); }
    }
    const auto payload_offset =
        ninfer::test::artifact_fixture::align_up(32 + bytes, 4096);
    const auto payload_bytes = root.at("files").at(0).at("payload_bytes").get<std::uint64_t>();
    std::filesystem::resize_file(sparse.path, payload_offset + payload_bytes);
    ninfer::artifact::Reader reader(sparse.path);
    ninfer::artifact::Binder binder(reader);
    const auto plan = bind_v3_text_artifact(
        binder, root, features(ninfer::SpeculativeBackend::None));
    if (!plan.bindings.v3_text || plan.bindings.dflash2 ||
        !plan.bindings.text_layers[3].is_full_attention ||
        plan.bindings.text_layers[0].is_full_attention ||
        !std::holds_alternative<IndividualGdnInputProjectionPlan>(
            plan.bindings.text_layers[0].gdn.input_projection) ||
        !std::holds_alternative<IndividualAttentionProjectionPlan>(
            plan.bindings.text_layers[3].attention.projection) ||
        !plan.bindings.text_layers[0].mlp.gate ||
        !plan.bindings.text_layers[0].mlp.up ||
        plan.materialization.object_count != root.at("objects").size() ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_objects.empty() ||
        plan.materialization.device_capacity_bytes < (std::uint64_t{1} << 30)) {
        std::cerr << "real GSQ v3 text-only materialization plan is incomplete\n";
        return 1;
    }
    const auto& gdn = std::get<IndividualGdnInputProjectionPlan>(
        plan.bindings.text_layers[0].gdn.input_projection);
    if (gdn.query.object.index != gdn.key.object.index ||
        gdn.query.object.index != gdn.value.object.index ||
        gdn.query.object.index != gdn.z.object.index ||
        gdn.query.first_row != 0 || gdn.key.first_row != 2048 ||
        gdn.value.first_row != 4096 || gdn.z.first_row != 10240 ||
        !plan.bindings.text_layers[0].gdn.output.input_columns) {
        std::cerr << "real GSQ v3 plan lost shared parents or activation gather\n";
        return 1;
    }
    {
        ninfer::artifact::Binder package_binder(reader);
        const ninfer::EngineOptions options{};
        const auto packaged =
            Package::plan_load(package_binder, options, WeightsProfile::Qwen38GroupwiseInt);
        if (packaged.materialization().device_objects.size() !=
                plan.materialization.device_objects.size()) {
            std::cerr << "Package did not select the real GSQ v3 binder\n";
            return 1;
        }
    }
    struct FeatureCase {
        ninfer::targets::qwen3_6::StartupFeatures choice;
        const char* name;
        std::size_t logical_bindings;
    };
    auto vision_only = features(ninfer::SpeculativeBackend::None);
    vision_only.vision = true;
    auto mtp_with_vision = features(ninfer::SpeculativeBackend::Mtp);
    mtp_with_vision.vision = true;
    auto dflash_with_vision = features(ninfer::SpeculativeBackend::DFlash2);
    dflash_with_vision.vision = true;
    constexpr std::size_t kVisionBindings = 441;
    constexpr std::size_t kMtpBindings = 16;
    constexpr std::size_t kDFlashBindings = 91;
    for (const FeatureCase& test : {
             FeatureCase{vision_only, "Vision", kVisionBindings},
             FeatureCase{features(ninfer::SpeculativeBackend::Mtp), "MTP", kMtpBindings},
             FeatureCase{features(ninfer::SpeculativeBackend::DFlash2), "DFlash2", kDFlashBindings},
             FeatureCase{mtp_with_vision, "Vision+MTP", kVisionBindings + kMtpBindings},
             FeatureCase{dflash_with_vision, "Vision+DFlash2",
                         kVisionBindings + kDFlashBindings},
             FeatureCase{features(ninfer::SpeculativeBackend::Mtp,
                                  ninfer::ProposalHead::Optimized), "MTP+indexed proposal",
                         kMtpBindings + 2},
             FeatureCase{features(ninfer::SpeculativeBackend::DFlash2,
                                  ninfer::ProposalHead::Optimized), "DFlash2+indexed proposal",
                         kDFlashBindings + 2},
         }) {
        const auto metadata = inspect_v3_feature_metadata(root, test.choice);
        if (metadata.size() != test.logical_bindings) {
            std::cerr << "GSQ " << test.name << " feature semantic inventory incomplete\n";
            return 1;
        }
        ninfer::artifact::Binder selected_binder(reader);
        const auto selected_plan = bind_v3_text_artifact(selected_binder, root, test.choice);
        if (!selected_plan.bindings.v3_text ||
            selected_plan.materialization.object_count != root.at("objects").size() ||
            selected_plan.materialization.host_objects.size() != 6 ||
            selected_plan.materialization.device_objects.size() <=
                plan.materialization.device_objects.size()) {
            std::cerr << "GSQ " << test.name << " physical placement incomplete\n";
            return 1;
        }
        if (test.choice.mtp() &&
            (!selected_plan.bindings.mtp.v3 ||
             !selected_plan.bindings.mtp.mlp.gate ||
             !selected_plan.bindings.mtp.mlp.up ||
             selected_plan.bindings.mtp.v3->query.object.index !=
                 selected_plan.bindings.mtp.v3->key.object.index)) {
            std::cerr << "GSQ MTP ModelView parent references incomplete\n";
            return 1;
        }
        if (test.choice.vision &&
            (selected_plan.bindings.vision_backbone.layers[0].qkv.index ==
                 selected_plan.bindings.vision_backbone.layers[1].qkv.index ||
             selected_plan.bindings.vision_backbone.layers[0].qkv.index ==
                 selected_plan.bindings.vision_backbone.layers[0].qkv_bias.index)) {
            std::cerr << "GSQ Vision physical QKV/bias parent identity incorrect\n";
            return 1;
        }
        if (test.choice.dflash2() &&
            (!selected_plan.bindings.dflash2 ||
             selected_plan.bindings.dflash2->layers[0].query_key_value.selected_rows !=
                 DFlashConfig::query_key_value_rows ||
             selected_plan.bindings.dflash2->layers[0].gate_up.selected_rows !=
                 DFlashConfig::mlp_gate_up_rows)) {
            std::cerr << "GSQ DFlash2 packed runtime parent geometry incomplete\n";
            return 1;
        }
        if (test.choice.optimized_proposal() &&
            (!selected_plan.bindings.v3_proposal_head ||
             selected_plan.bindings.v3_proposal_head->selected_rows !=
                 TextConfig::draft_head_rows)) {
            std::cerr << "GSQ optimized proposal binding incomplete\n";
            return 1;
        }
    }
    const auto feature_rejected = [&](nlohmann::json invalid,
                                      ninfer::targets::qwen3_6::StartupFeatures selected,
                                      const char* label) {
        try {
            (void)inspect_v3_feature_metadata(invalid, selected);
        } catch (const ninfer::artifact::ArtifactError&) {
            return true;
        }
        std::cerr << "GSQ v3 accepted broken feature " << label << '\n';
        return false;
    };
    {
        auto invalid = root;
        invalid["bindings"].erase("mtp/layers/0/attention/key");
        if (!feature_rejected(std::move(invalid), features(ninfer::SpeculativeBackend::Mtp),
                              "MTP key")) { return 1; }
    }
    {
        auto invalid = root;
        auto& start = invalid["bindings"]["vision/layers/0/attention/key"]["parts"][0]
                              ["range"][0];
        start = start.get<std::uint64_t>() + 1152;
        if (!feature_rejected(std::move(invalid), vision_only, "Vision QKV order")) { return 1; }
    }
    {
        auto invalid = root;
        invalid["components"]["dflash2"]["config"]["dflash_config"]["selector_rank"] = 128;
        if (!feature_rejected(std::move(invalid),
                              features(ninfer::SpeculativeBackend::DFlash2),
                              "DFlash2 selector config")) { return 1; }
    }
    return 0;
}

int verify_real_qwen36_v3_headers() {
    // The two official Qwen3.6 artifacts also use v3 directories. A sparse header-derived
    // fixture exercises their exact physical layouts without requiring 17+GiB on disk or GPU.
    const auto root_path = std::filesystem::path(NINFER_SOURCE_DIR) / ".." / ".." / "real-models";
    for (const auto& [filename, expected_mlp_format] : {
             std::pair{"qwen3_6_27b.ninfer", NumericFormat::Q4G64_F16S},
             std::pair{"qwen3_6_27b_nvfp4.ninfer", NumericFormat::NVFP4},
         }) {
        const auto file_path = root_path / filename;
        if (!std::filesystem::is_regular_file(file_path)) { continue; }
        std::ifstream file(file_path, std::ios::binary);
        std::array<unsigned char, 32> header{};
        file.read(reinterpret_cast<char*>(header.data()), header.size());
        const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
        if (!file || !std::equal(magic.begin(), magic.end(), header.begin())) {
            throw std::runtime_error(std::string(filename) + " is not v3");
        }
        std::uint64_t bytes = 0;
        for (int i = 0; i < 8; ++i) {
            bytes |= std::uint64_t(header[8 + i]) << (8 * i);
        }
        if (bytes == 0 || bytes > (1 << 20)) {
            throw std::runtime_error(std::string(filename) + " has invalid JSON length");
        }
        std::string content(bytes, '\0');
        file.read(content.data(), static_cast<std::streamsize>(content.size()));
        if (!file) { throw std::runtime_error("incomplete Qwen3.6 v3 header"); }
        const auto directory = nlohmann::json::parse(content);
        const auto metadata = inspect_v3_text_metadata(directory);
        if (metadata.tensors.size() != 963 ||
            directory.at("metadata").at("name") != "qwen3.6-27b") {
            throw std::runtime_error("Qwen3.6 v3 semantic text inventory differs");
        }
        const auto gate = std::ranges::find_if(metadata.tensors, [](const V3LogicalTensor& item) {
            return item.parameter == "text/layers/0/mlp/gate";
        });
        if (gate == metadata.tensors.end() || gate->format != expected_mlp_format) {
            throw std::runtime_error("Qwen3.6 v3 physical MLP format differs");
        }
        auto sparse_file = ninfer::test::artifact_fixture::TemporaryArtifact{
            std::filesystem::temp_directory_path() /
            ("ninfer_v3_qwen36_" + std::to_string(std::random_device{}()) + ".ninfer")};
        {
            std::ofstream output(sparse_file.path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(header.data()), header.size());
            output.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!output) { throw std::runtime_error("Qwen3.6 metadata fixture failed"); }
        }
        const auto start = ninfer::test::artifact_fixture::align_up(32 + bytes, 4096);
        const auto payload = directory.at("files").at(0).at("payload_bytes")
                                 .get<std::uint64_t>();
        std::filesystem::resize_file(sparse_file.path, start + payload);
        ninfer::artifact::Reader reader(sparse_file.path);
        ninfer::artifact::Binder plain_binder(reader);
        const auto plain = bind_v3_text_artifact(
            plain_binder, directory, features(ninfer::SpeculativeBackend::None));
        if (!plain.bindings.v3_text || plain.materialization.host_objects.size() != 6 ||
            plain.materialization.object_count != directory.at("objects").size()) {
            throw std::runtime_error("Qwen3.6 v3 base plan differs");
        }
        {
            ninfer::artifact::Binder package_binder(reader);
            const ninfer::EngineOptions options{};
            const auto profile = expected_mlp_format == NumericFormat::NVFP4
                                     ? WeightsProfile::Qwen36Nvfp4
                                     : WeightsProfile::Qwen36GroupwiseInt;
            const auto packaged = Package::plan_load(package_binder, options, profile);
            if (packaged.materialization().device_objects.size() !=
                    plain.materialization.device_objects.size()) {
                throw std::runtime_error("Package rejected Qwen3.6 native v3 profile");
            }
        }
        for (const auto& selected : {
                 features(ninfer::SpeculativeBackend::Mtp),
                 [] {
                     auto feature = features(ninfer::SpeculativeBackend::None);
                     feature.vision = true;
                     return feature;
                 }(),
             }) {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_v3_text_artifact(binder, directory, selected);
            if (plan.materialization.device_objects.size() <=
                    plain.materialization.device_objects.size() ||
                plan.materialization.object_count != directory.at("objects").size()) {
                throw std::runtime_error("Qwen3.6 v3 feature plan incomplete");
            }
        }
    }
    return 0;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
    }
    return 0;
}

int verify_nvfp4(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36Nvfp4) {
        std::cerr << "NVFP4 identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36Nvfp4, all_features());
    if (plan.materialization.object_count != 1307 ||
        plan.materialization.device_objects.size() != 1054 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.object_count - plan.materialization.device_objects.size() -
                plan.materialization.host_objects.size() !=
            247 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "NVFP4 materialization plan is incomplete: objects="
                  << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size() << '\n';
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::W8G32_F16S ||
        plan.bindings.output_head.format != NumericFormat::W8G32_F16S) {
        std::cerr << "NVFP4 vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }

    std::size_t nvfp4_weights          = 0;
    std::size_t bf16_attention_inputs  = 0;
    std::size_t bf16_attention_outputs = 0;
    std::size_t bf16_gdn_outputs       = 0;
    const auto count_weight            = [&](const WeightPlan& weight) {
        if (weight.format == NumericFormat::NVFP4) {
            ++nvfp4_weights;
            return valid_divisors(weight);
        }
        return true;
    };
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (!count_weight(layer.mlp.gate_up) || !count_weight(layer.mlp.down)) {
            std::cerr << "NVFP4 MLP divisor is invalid\n";
            return 1;
        }
        if (layer.is_full_attention) {
            const auto* fused =
                std::get_if<FusedAttentionProjectionPlan>(&layer.attention.projection);
            if (fused == nullptr || !count_weight(fused->query_key_gate_value) ||
                !count_weight(layer.attention.output)) {
                std::cerr << "NVFP4 attention binding is invalid\n";
                return 1;
            }
            bf16_attention_inputs +=
                fused->query_key_gate_value.format == NumericFormat::BF16 ? 1 : 0;
            bf16_attention_outputs += layer.attention.output.format == NumericFormat::BF16 ? 1 : 0;
        } else {
            const auto* fused =
                std::get_if<FusedGdnInputProjectionPlan>(&layer.gdn.input_projection);
            if (fused == nullptr || !count_weight(fused->query_key_value_z) ||
                !count_weight(layer.gdn.output)) {
                std::cerr << "NVFP4 GDN binding is invalid\n";
                return 1;
            }
            bf16_gdn_outputs += layer.gdn.output.format == NumericFormat::BF16 ? 1 : 0;
        }
    }
    if (nvfp4_weights != 247 || bf16_attention_inputs != 6 || bf16_attention_outputs != 2 ||
        bf16_gdn_outputs != 1) {
        std::cerr << "NVFP4 Text inventory has the wrong storage profile: nvfp4=" << nvfp4_weights
                  << " bf16_attention_input=" << bf16_attention_inputs
                  << " bf16_attention_output=" << bf16_attention_outputs
                  << " bf16_gdn_output=" << bf16_gdn_outputs << '\n';
        return 1;
    }
    return 0;
}

int verify_legacy_dflash2_compatibility(const std::filesystem::path& path, WeightsProfile profile) {
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::None));
        if (plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "legacy artifact unexpectedly bound DFlash2: " << path << '\n';
            return 1;
        }
    }
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::Mtp));
        if (plan.bindings.dflash2 ||
            !is_device_object(plan.materialization, plan.bindings.mtp.input_projection)) {
            std::cerr << "legacy artifact did not preserve MTP-only binding: " << path << '\n';
            return 1;
        }
    }
    try {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        (void)bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::DFlash2));
    } catch (const ninfer::artifact::ArtifactError& error) {
        if (std::string(error.what()).find("no DFlash2 weight bundle") != std::string::npos) {
            return 0;
        }
    }
    std::cerr << "legacy artifact did not reject selected DFlash2: " << path << '\n';
    return 1;
}

int verify_dflash2_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    for (const ninfer::SpeculativeBackend backend :
         {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan = bind_artifact(binder, profile, features(backend));
        if (!plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "inactive DFlash2 bundle was not validate-only: " << path << '\n';
            return 1;
        }
        const bool mtp_is_device =
            is_device_object(plan.materialization, plan.bindings.mtp.input_projection);
        if (mtp_is_device != (backend == ninfer::SpeculativeBackend::Mtp)) {
            std::cerr << "MTP placement does not match backend selection: " << path << '\n';
            return 1;
        }
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(
        binder, profile,
        features(ninfer::SpeculativeBackend::DFlash2, ninfer::ProposalHead::Optimized));
    if (!plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 66 ||
        is_device_object(plan.materialization, plan.bindings.mtp.input_projection) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head_token_ids)) {
        std::cerr << "selected DFlash2 bundle has the wrong placement: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_rejection() {
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context                      = 128;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk                    = 128;
    options.use_cuda_graph                   = false;
    options.context_cache.device_state_slots = options.max_concurrency;
    auto planner =
        Package::make_sequence_planner(device, options, WeightsProfile::Qwen36GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence             = std::move(planner).finalize(pages);
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(
            empty_model, WeightsProfile::Qwen36Nvfp4, std::move(sequence), device,
            ninfer::StartupObserver{});
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

int verify_vision_workspace_planning() {
    static_assert(ninfer::targets::qwen3_6::kMaximumPromptVisionTokens == 32768);
    static_assert(ninfer::targets::qwen3_6::kMaximumVisionItemTokens == 16384);
    constexpr std::size_t kExpectedMaximumItemWorkspace = 866'648'064;

    ninfer::DeviceContext device(0);
    const auto workspace_capacity = [&](std::uint32_t max_context) {
        ninfer::EngineOptions options;
        options.max_context              = max_context;
        options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
        options.prefill_chunk            = 1024;
        options.kv_storage =
            ninfer::KvStoragePolicy::explicit_storage(ninfer::KvCacheStorage::Fp8E4M3Row256);
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
        options.enable_vision                    = true;
        options.use_cuda_graph                   = false;
        options.context_cache.device_state_slots = 1;
        auto planner = Package::make_sequence_planner(device, options, WeightsProfile::Qwen36Nvfp4);
        const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
        return std::move(planner).finalize(pages).workspace_capacity_bytes();
    };

    const std::size_t at_item_limit    = workspace_capacity(16384);
    const std::size_t above_item_limit = workspace_capacity(131072);
    if (at_item_limit != kExpectedMaximumItemWorkspace ||
        above_item_limit != kExpectedMaximumItemWorkspace) {
        std::cerr << "Vision workspace does not clamp Device execution at the 16K item bound: "
                  << "at_limit=" << at_item_limit << " above_limit=" << above_item_limit
                  << " expected=" << kExpectedMaximumItemWorkspace << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // Validate real v3 input_columns payload semantics independently of
    // weight availability. Shape-only metadata validation cannot detect a
    // duplicate or out-of-range activation gather index.
    {
        std::vector<std::int32_t> columns(TextConfig::value_dim);
        std::iota(columns.begin(), columns.end(), 0);
        std::reverse(columns.begin(), columns.end());
        validate_v3_input_columns(columns, columns.size());
        const auto rejects = [&](std::span<const std::int32_t> map) {
            try {
                validate_v3_input_columns(map, TextConfig::value_dim);
            } catch (const ninfer::artifact::ArtifactError&) {
                return true;
            }
            return false;
        };
        if (!rejects(std::span<const std::int32_t>(columns).first(columns.size() - 1))) {
            std::cerr << "v3 input_columns accepted a short mapping\n";
            return 1;
        }
        columns[0] = columns[1];
        if (!rejects(columns)) {
            std::cerr << "v3 input_columns accepted duplicate entries\n";
            return 1;
        }
        columns[0] = -1;
        if (!rejects(columns)) {
            std::cerr << "v3 input_columns accepted negative indices\n";
            return 1;
        }
        columns[0] = TextConfig::value_dim;
        if (!rejects(columns)) {
            std::cerr << "v3 input_columns accepted out-of-range indices\n";
            return 1;
        }
    }
    if (const int result = verify_self_describing_shape(); result != 0) { return result; }
    if (const int result = verify_real_v3_gsq_header(); result != 0) { return result; }
    if (const int result = verify_real_qwen36_v3_headers(); result != 0) { return result; }
    if (argc == 2 && std::string_view(argv[1]) == "--v3-header-only") {
        std::cout << "real v3 GSQ text metadata validated\n";
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--shape-only") {
        std::cout << "self-describing model shape fixtures passed\n";
        return 0;
    }
    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path nvfp4 =
        artifact_path("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "qwen3_6_27b_nvfp4.ninfer");
    const std::filesystem::path qwen38_groupwise =
        artifact_path("NINFER_QWEN3_8_27B_OLD_WEIGHTS", "qwen3_8_27b_old.ninfer");
    const std::filesystem::path qwen38_nvfp4 =
        artifact_path("NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS", "qwen3_8_27b_nvfp4_old.ninfer");
    const std::filesystem::path qwen38_groupwise_dflash2 =
        artifact_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path qwen38_nvfp4_dflash2 = artifact_path(
        "NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS", "qwen3_8_27b_nvfp4.ninfer");
    if (!std::filesystem::is_regular_file(groupwise) || !std::filesystem::is_regular_file(nvfp4)) {
        std::cerr << "skip: both real 27B artifacts are required: groupwise=" << groupwise
                  << " nvfp4=" << nvfp4 << '\n';
        return 77;
    }
    if (const int result = verify_vision_workspace_planning(); result != 0) { return result; }
    if (const int result = verify_rejection(); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }
    if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
    if (const int result = verify_nvfp4(nvfp4); result != 0) { return result; }
    if (const int result =
            verify_legacy_dflash2_compatibility(groupwise, WeightsProfile::Qwen36GroupwiseInt);
        result != 0) {
        return result;
    }
    const std::array dflash2_artifacts{qwen38_groupwise, qwen38_nvfp4, qwen38_groupwise_dflash2,
                                       qwen38_nvfp4_dflash2};
    if (!std::ranges::all_of(dflash2_artifacts, [](const auto& path) {
            return std::filesystem::is_regular_file(path);
        })) {
        std::cerr << "skip DFlash2 binding matrix: old and new Qwen3.8 artifacts are required\n";
        return 0;
    }
    if (const int result = verify_legacy_dflash2_compatibility(qwen38_groupwise,
                                                               WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_legacy_dflash2_compatibility(qwen38_nvfp4, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_dflash2_bundle(qwen38_groupwise_dflash2, WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result = verify_dflash2_bundle(qwen38_nvfp4_dflash2, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    return 0;
}
