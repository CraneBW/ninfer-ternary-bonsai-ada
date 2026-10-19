#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
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
#include <unordered_set>
#include <vector>

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
        if (test.choice.mtp()) {
            const auto& mtp = *selected_plan.bindings.mtp.v3;
            const auto& gate = *selected_plan.bindings.mtp.mlp.gate;
            const auto& up = *selected_plan.bindings.mtp.mlp.up;
            if (mtp.query.format != NumericFormat::GGUF_Q6_K ||
                mtp.key.format != NumericFormat::GGUF_Q6_K ||
                mtp.query.first_row != 0 ||
                mtp.key.first_row != TextConfig::query_size ||
                mtp.gate.first_row != TextConfig::query_size + TextConfig::kv_size ||
                mtp.value.first_row != 2 * TextConfig::query_size + TextConfig::kv_size ||
                mtp.query.object.index != mtp.gate.object.index ||
                mtp.query.object.index != mtp.value.object.index ||
                mtp.query.parent_rows != TextConfig::attention_input_rows ||
                gate.format != NumericFormat::GGUF_Q6_K ||
                gate.first_row != 0 || up.first_row != TextConfig::intermediate ||
                gate.object.index != up.object.index ||
                gate.parent_rows != TextConfig::mtp_mlp_gate_up_rows ||
                mtp.input_projection.parent_columns != TextConfig::mtp_input_rows ||
                mtp.output.parent_columns != TextConfig::query_size) {
                std::cerr << "GSQ MTP physical parents / logical-element slices do not match "
                             "native ModelView\n";
                return 1;
            }
            // Package startup uses the same MTP feature selection as the
            // direct Binder path. In particular the shared GGUF parent is
            // materialized once, while inactive DFlash2 is validate-only.
            ninfer::artifact::Binder package_binder(reader);
            ninfer::EngineOptions options;
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.proposal_head = test.choice.proposal_head;
            options.enable_vision = test.choice.vision;
            const auto packaged = Package::plan_load(
                package_binder, options, WeightsProfile::Qwen38GroupwiseInt);
            if (packaged.materialization().device_objects.size() !=
                selected_plan.materialization.device_objects.size()) {
                std::cerr << "GSQ MTP Package startup did not materialize selected MTP parents\n";
                return 1;
            }
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
        auto& mtp_gate = invalid["bindings"]["mtp/layers/0/attention/gate"]["parts"][0];
        mtp_gate["range"][0] = mtp_gate["range"][0].get<std::uint64_t>() + TextConfig::hidden;
        if (!feature_rejected(std::move(invalid), features(ninfer::SpeculativeBackend::Mtp),
                              "MTP fused attention gate row range")) { return 1; }
    }
    {
        auto invalid = root;
        const auto physical_id = invalid["bindings"]["mtp/layers/0/attention/gate"]["parts"][0]
                                         ["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == physical_id) {
                object["layout"] = "row_split_k128_v1";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), features(ninfer::SpeculativeBackend::Mtp),
                              "MTP GGUF physical layout")) { return 1; }
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
        const auto id = invalid["bindings"]["vision/layers/0/attention/query"]["parts"][0]
                            ["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["format"] = "q5_g64_fp16";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), vision_only, "Vision QKV codec")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["vision/layers/0/attention/output"]["object"]
                            .get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["layout"] = "gguf_blocks_v1";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), vision_only, "Vision quantized layout")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["vision/layers/0/norm1_weight"]["object"]
                            .get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["layout"] = "row_split_k128_v1";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), vision_only, "Vision BF16 layout")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        for (auto& use : invalid["uses"]) {
            if (use["parameter"] == "vision/patch_embedding" &&
                use["input"] == "vision/patch_input") {
                use["activation_policy"] = "AllowA8";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), vision_only, "Vision activation policy")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["vision/merger/fc2"]["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["shape"][1] = 4096;
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), vision_only, "Vision merger input width")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        invalid["components"]["dflash2"]["config"]["dflash_config"]["selector_rank"] = 128;
        if (!feature_rejected(std::move(invalid),
                              features(ninfer::SpeculativeBackend::DFlash2),
                              "DFlash2 selector config")) { return 1; }
    }
    const auto dflash_only = features(ninfer::SpeculativeBackend::DFlash2);
    {
        auto invalid = root;
        auto& end = invalid["bindings"]["dflash2/layers/0/mlp/up"]["parts"][0]["range"][1];
        end = end.get<std::uint64_t>() - DFlashConfig::hidden;
        if (!feature_rejected(std::move(invalid), dflash_only,
                              "DFlash2 fused gate/up row range")) { return 1; }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["dflash2/layers/0/attention/query"]
                              ["parts"][0]["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["format"] = "q5_g64_fp16";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), dflash_only, "DFlash2 W8 codec")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["dflash2/layers/0/attention/output"]
                              ["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["layout"] = "gguf_blocks_v1";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), dflash_only, "DFlash2 W8 physical layout")) {
            return 1;
        }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["dflash2/candidate_selector/hidden_projection"]
                              ["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["layout"] = "row_split_k128_v1";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), dflash_only,
                              "DFlash2 BF16 physical layout")) { return 1; }
    }
    {
        auto invalid = root;
        for (auto& use : invalid["uses"]) {
            if (use["parameter"] == "dflash2/feature_projection" &&
                use["input"] == "dflash2/target_features") {
                use["activation_policy"] = "AllowA8";
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), dflash_only,
                              "DFlash2 activation policy")) { return 1; }
    }
    {
        auto invalid = root;
        const auto id = invalid["bindings"]["dflash2/candidate_selector/predecessor_codebook"]
                              ["object"].get<std::string>();
        for (auto& object : invalid["objects"]) {
            if (object["id"] == id) {
                object["shape"][1] = 128;
                break;
            }
        }
        if (!feature_rejected(std::move(invalid), dflash_only,
                              "DFlash2 selector codebook width")) { return 1; }
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

int verify_real_v3_mtp_materialization(std::string_view filename,
                                       NumericFormat expected_format) {
    // Materialize only MTP's *actual physical parents* into a native CUDA
    // arena. All of Text/Vision/DFlash2 stay validate-only, so the check runs
    // without needing another >15 GiB GPU for the unrelated weights. This is
    // the same materialize_mtp_weights() called by LoadedModelData's ModelView.
    const auto root_path = std::filesystem::path(NINFER_SOURCE_DIR) / ".." / ".." / "real-models";
    const auto path = root_path / std::string(filename);
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: no real v3 artifact to materialize: " << path << '\n';
        return 77;
    }
    std::ifstream prefix(path, std::ios::binary);
    std::array<unsigned char, 32> header{};
    prefix.read(reinterpret_cast<char*>(header.data()), header.size());
    const std::array<unsigned char, 8> expected_magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    if (!prefix || !std::equal(expected_magic.begin(), expected_magic.end(), header.begin())) {
        throw std::runtime_error("real MTP source is not v3");
    }
    std::uint64_t directory_bytes = 0;
    for (unsigned i = 0; i < 8; ++i) {
        directory_bytes |= std::uint64_t(header[8 + i]) << (i * 8);
    }
    if (directory_bytes == 0 || directory_bytes > 1024 * 1024) {
        throw std::runtime_error("real MTP directory size is invalid");
    }
    std::string raw_directory(static_cast<std::size_t>(directory_bytes), '\0');
    prefix.read(raw_directory.data(), static_cast<std::streamsize>(raw_directory.size()));
    if (!prefix) { throw std::runtime_error("real MTP directory is incomplete"); }
    const auto directory = nlohmann::json::parse(raw_directory);
    const auto expected_size =
        ninfer::test::artifact_fixture::align_up(32 + directory_bytes, 4096) +
        directory.at("files").at(0).at("payload_bytes").get<std::uint64_t>();
    if (std::filesystem::file_size(path) != expected_size) {
        std::cerr << "skip: real v3 weights are incomplete; MTP physical parents not yet "
                     "available for materialization\n";
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder complete_binder(reader);
    const ArtifactLoadPlan complete = bind_v3_text_artifact(
        complete_binder, directory, features(ninfer::SpeculativeBackend::Mtp));
    const MtpPlan& plan = complete.bindings.mtp;
    if (!plan.v3 || !plan.mlp.gate || !plan.mlp.up) {
        throw std::runtime_error("real MTP is missing independent semantic projections");
    }

    std::unordered_set<std::size_t> selected;
    const auto add = [&](ninfer::artifact::ObjectHandle handle) {
        selected.insert(handle.index);
    };
    const auto add_weight = [&](const WeightPlan& weight) { add(weight.object); };
    const auto& projection = *plan.v3;
    for (const auto* weight : {&projection.input_projection, &projection.query,
                               &projection.key, &projection.gate, &projection.value,
                               &projection.output, &*plan.mlp.gate, &*plan.mlp.up,
                               &plan.mlp.down}) {
        add_weight(*weight);
    }
    for (const auto handle : {plan.embedding_norm, plan.hidden_norm, plan.input_norm,
                              plan.query_norm, plan.key_norm, plan.post_attention_norm,
                              plan.final_norm}) {
        add(handle);
    }
    if (selected.size() != 12) {
        throw std::runtime_error("real MTP selected physical parent count changed");
    }

    ninfer::artifact::Binder only_mtp(reader);
    for (std::size_t index = 0; index < reader.objects().size(); ++index) {
        const auto& descriptor = reader.objects()[index];
        if (const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&descriptor)) {
            const auto handle = only_mtp.require_tensor(
                tensor->name, tensor->format, tensor->layout, tensor->shape);
            if (selected.contains(index)) {
                only_mtp.materialize_on_device(handle);
            } else {
                only_mtp.validate_only(handle);
            }
        } else {
            const auto& resource = std::get<ninfer::artifact::ResourceDescriptor>(descriptor);
            const auto handle = only_mtp.require_resource(resource.name, resource.encoding);
            only_mtp.validate_only(handle);
        }
    }
    const auto materialization = only_mtp.finish();
    if (materialization.device_objects.size() != selected.size() ||
        materialization.host_objects.size() != 0 ||
        materialization.device_capacity_bytes > (std::uint64_t{512} << 20)) {
        throw std::runtime_error("real MTP physical-only materialization plan changed");
    }
    ninfer::DeviceContext device(0);
    const auto backing = ninfer::artifact::materialize(reader, materialization, device);
    const MtpWeights native = materialize_mtp_weights(plan, backing);
    if (native.attention.packed.qdata != nullptr || !native.post_mixer.gate ||
        !native.post_mixer.up || native.post_mixer.gate_up.qdata != nullptr) {
        throw std::runtime_error("v3 MTP ModelView routed fused or shared projections incorrectly");
    }

    // Independent oracle: convert each logical-element range to an encoded
    // GGUF Q6_K or row-split W8 byte offset and verify FIRST and LAST blocks of the
    // CUDA view against the source file itself. This catches a mistaken
    // range-in-bytes interpretation, row ordering and an incorrect parent
    // materialization even though no model inference is run.
    constexpr std::uint64_t kGgufElements = 256;
    constexpr std::size_t kQ6BlockBytes = 210;
    const bool gguf = expected_format == NumericFormat::GGUF_Q6_K;
    const auto verify = [&](const char* name, const WeightPlan& source, const ninfer::Weight& actual,
                            int rows, int columns) {
        const auto& physical =
            std::get<ninfer::artifact::TensorDescriptor>(reader.objects().at(source.object.index));
        if (source.format != expected_format ||
            physical.format != expected_format ||
            physical.layout != (gguf ? ninfer::artifact::StorageLayout::GgufBlocksV1
                                    : ninfer::artifact::StorageLayout::RowSplitK128V1) ||
            actual.qtype != (gguf ? ninfer::QType::GGUF_Q6_K
                                  : ninfer::QType::W8G32_F16S) ||
            actual.layout != (gguf ? ninfer::QuantLayout::GgufBlocks
                                   : ninfer::QuantLayout::RowSplit) ||
            actual.n != rows || actual.k != columns ||
            source.selected_rows != rows || source.parent_columns != columns ||
            physical.shape[0] != static_cast<std::uint64_t>(source.parent_rows) ||
            physical.shape[1] != static_cast<std::uint64_t>(columns)) {
            throw std::runtime_error(std::string("MTP materialized format/shape differs: ") + name);
        }
        const std::uint64_t row_bytes = gguf
            ? static_cast<std::uint64_t>(columns) / kGgufElements * kQ6BlockBytes
            : static_cast<std::uint64_t>(columns);
        const auto* base = static_cast<const std::byte*>(backing.device_data(source.object));
        if (actual.qdata != base + static_cast<std::uint64_t>(source.first_row) * row_bytes) {
            throw std::runtime_error(std::string("MTP parent row offset differs: ") + name);
        }
        const auto check_block = [&](std::uint64_t row, std::uint64_t block) {
            const std::size_t bytes = gguf ? kQ6BlockBytes : 32;
            const std::uint64_t byte_offset = row * row_bytes + block * bytes;
            std::array<std::byte, kQ6BlockBytes> device_block{};
            std::array<std::byte, kQ6BlockBytes> source_block{};
            CUDA_CHECK(cudaMemcpy(device_block.data(),
                                  static_cast<const std::byte*>(actual.qdata) + byte_offset,
                                  bytes, cudaMemcpyDeviceToHost));
            std::ifstream input(path, std::ios::binary);
            input.seekg(static_cast<std::streamoff>(
                reader.payload_offset() + physical.offset +
                static_cast<std::uint64_t>(source.first_row) * row_bytes + byte_offset));
            input.read(reinterpret_cast<char*>(source_block.data()),
                       static_cast<std::streamsize>(bytes));
            if (!input || device_block != source_block) {
                throw std::runtime_error(std::string("MTP physical block differs: ") + name);
            }
            if (!gguf) {
                const auto geometry = ninfer::artifact::row_split_geometry(
                    expected_format, physical.shape);
                if (geometry.padded_columns != static_cast<std::uint64_t>(columns)) {
                    throw std::runtime_error("MTP W8 physical row padding changed");
                }
                const std::uint64_t scale_stride =
                    (static_cast<std::uint64_t>(columns) / 32) * 2;
                const auto scale_offset = row * scale_stride + block * 2;
                if (actual.scales != static_cast<const std::byte*>(backing.device_data(source.object)) +
                                        geometry.scale_plane_offset +
                                        static_cast<std::uint64_t>(source.first_row) * scale_stride) {
                    throw std::runtime_error(std::string("MTP W8 scale row offset differs: ") + name);
                }
                std::array<std::byte, 2> device_scale{};
                std::array<std::byte, 2> source_scale{};
                CUDA_CHECK(cudaMemcpy(device_scale.data(),
                                      static_cast<const std::byte*>(actual.scales) + scale_offset,
                                      2, cudaMemcpyDeviceToHost));
                input.seekg(static_cast<std::streamoff>(
                    reader.payload_offset() + physical.offset +
                    geometry.scale_plane_offset +
                    static_cast<std::uint64_t>(source.first_row) * scale_stride + scale_offset));
                input.read(reinterpret_cast<char*>(source_scale.data()), 2);
                if (!input || device_scale != source_scale) {
                    throw std::runtime_error(std::string("MTP W8 scale block differs: ") + name);
                }
            }
        };
        check_block(0, 0);
        check_block(rows - 1, columns / (gguf ? kGgufElements : 32) - 1);
    };

    verify("stem", projection.input_projection, native.input_projection,
           TextConfig::hidden, TextConfig::mtp_input_rows);
    verify("query", projection.query, native.attention.query,
           TextConfig::query_size, TextConfig::hidden);
    verify("key", projection.key, native.attention.key, TextConfig::kv_size, TextConfig::hidden);
    verify("gate", projection.gate, native.attention.output_gate,
           TextConfig::query_size, TextConfig::hidden);
    verify("value", projection.value, native.attention.value,
           TextConfig::kv_size, TextConfig::hidden);
    verify("attention output", projection.output, native.output,
           TextConfig::hidden, TextConfig::query_size);
    verify("MLP gate", *plan.mlp.gate, *native.post_mixer.gate,
           TextConfig::intermediate, TextConfig::hidden);
    verify("MLP up", *plan.mlp.up, *native.post_mixer.up,
           TextConfig::intermediate, TextConfig::hidden);
    verify("MLP down", plan.mlp.down, native.post_mixer.down,
           TextConfig::hidden, TextConfig::intermediate);
    if (native.embedding_norm.ne[0] != TextConfig::hidden ||
        native.hidden_norm.ne[0] != TextConfig::hidden ||
        native.final_norm.ne[0] != TextConfig::hidden ||
        native.query_norm.ne[0] != TextConfig::head_dim ||
        native.key_norm.ne[0] != TextConfig::head_dim) {
        throw std::runtime_error("real v3 MTP BF16 norms were not materialized to ModelView");
    }
    std::cout << "real v3 " << filename << " MTP physical parent blocks and native ModelView "
                 "verified\n";
    return 0;
}

int verify_real_v3_dflash2_materialization() {
    // Upload the real GSQ v3 DFlash2 bundle alone. The native ModelView must
    // interpret 91 semantic uses through 66 physical parents without loading
    // text, Vision or MTP weights (the entire model is >14 GiB).
    const auto path = std::filesystem::path(NINFER_SOURCE_DIR) / ".." / ".." /
                      "real-models" / "qwen3_8_gsq_rco_iq3_s_v3.ninfer";
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real GSQ v3 DFlash2 weights are unavailable\n";
        return 77;
    }
    ninfer::artifact::Reader reader(path);
    const auto* directory = reader.v3_directory();
    if (!directory) { throw std::runtime_error("DFlash2 source is not v3"); }
    const auto expected_size = reader.payload_offset() +
        directory->at("files").at(0).at("payload_bytes").get<std::uint64_t>();
    if (std::filesystem::file_size(path) != expected_size) {
        std::cerr << "skip: real GSQ v3 DFlash2 source payload is incomplete\n";
        return 77;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "skip: no CUDA GPU for real DFlash2 materialization\n";
        return 77;
    }
    const auto choice = features(ninfer::SpeculativeBackend::DFlash2);
    ninfer::artifact::Binder complete_binder(reader);
    const auto complete = bind_v3_text_artifact(complete_binder, *directory, choice);
    if (!complete.bindings.dflash2) {
        throw std::runtime_error("real DFlash2 semantic binding was not selected");
    }
    const auto& plan = *complete.bindings.dflash2;
    {
        ninfer::artifact::Binder package_binder(reader);
        ninfer::EngineOptions options;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        const auto package = Package::plan_load(
            package_binder, options, WeightsProfile::Qwen38GroupwiseInt);
        if (package.materialization().device_objects.size() !=
            complete.materialization.device_objects.size()) {
            throw std::runtime_error("Package DFlash2 v3 device selection differs");
        }
    }

    std::unordered_set<std::size_t> selected;
    const auto add = [&](ninfer::artifact::ObjectHandle object) {
        selected.insert(object.index);
    };
    const auto add_weight = [&](const WeightPlan& weight) { add(weight.object); };
    add_weight(plan.feature_projection);
    add(plan.context_norm);
    add(plan.final_norm);
    for (const auto& layer : plan.layers) {
        for (const auto* weight : {&layer.query_key_value, &layer.attention_output,
                                   &layer.gate_up, &layer.down,
                                   &layer.attention_conv.kernel_projection,
                                   &layer.mlp_conv.kernel_projection}) {
            add_weight(*weight);
        }
        for (const auto object : {layer.input_norm, layer.post_attention_norm,
                                  layer.query_norm, layer.key_norm,
                                  layer.attention_conv.base_kernel, layer.mlp_conv.base_kernel}) {
            add(object);
        }
    }
    add_weight(plan.candidate_selector.hidden_projection);
    add(plan.candidate_selector.predecessor_codebook);
    add(plan.candidate_selector.successor_codebook);
    if (selected.size() != 66) {
        throw std::runtime_error("real GSQ DFlash2 physical parent inventory differs");
    }
    for (const auto index : selected) {
        if (!is_device_object(complete.materialization,
                              ninfer::artifact::ObjectHandle{index})) {
            throw std::runtime_error("real DFlash2 Package binding missed a GPU parent");
        }
    }

    ninfer::artifact::Binder only_dflash(reader);
    for (std::size_t index = 0; index < reader.objects().size(); ++index) {
        const auto& descriptor = reader.objects()[index];
        if (const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&descriptor)) {
            const auto handle = only_dflash.require_tensor(
                tensor->name, tensor->format, tensor->layout, tensor->shape);
            if (selected.contains(index)) {
                only_dflash.materialize_on_device(handle);
            } else {
                only_dflash.validate_only(handle);
            }
        } else {
            const auto& resource = std::get<ninfer::artifact::ResourceDescriptor>(descriptor);
            const auto handle = only_dflash.require_resource(resource.name, resource.encoding);
            only_dflash.validate_only(handle);
        }
    }
    const auto materialization = only_dflash.finish();
    if (materialization.device_objects.size() != selected.size() ||
        !materialization.host_objects.empty() ||
        materialization.device_capacity_bytes > (std::uint64_t{2304} << 20)) {
        throw std::runtime_error("DFlash2-only materialization exceeds physical bundle size");
    }

    ninfer::DeviceContext device(0);
    const auto backing = ninfer::artifact::materialize(reader, materialization, device);
    const auto native = materialize_dflash2_weights(plan, backing);
    std::unordered_set<std::size_t> viewed;
    std::ifstream source(path, std::ios::binary);
    // Exact oracle compares actual CUDA views to source-file bytes. For W8
    // split planes sample the first and last encoded values and FP16 scales
    // independently so wrong row offsets or plane offsets are observable.
    const auto verify_bytes = [&](ninfer::artifact::ObjectHandle object,
                                  std::uint64_t offset, const std::byte* gpu,
                                  std::uint64_t count, std::string_view name) {
        if (gpu == nullptr || count == 0) {
            throw std::runtime_error(std::string(name) + ": no DFlash2 device data");
        }
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(object.index));
        if (offset > physical.bytes || count > physical.bytes - offset) {
            throw std::runtime_error(std::string(name) + ": data exceeds physical object");
        }
        const auto compare = [&](std::uint64_t at, std::size_t bytes) {
            std::array<std::byte, 32> actual{}, expected{};
            CUDA_CHECK(cudaMemcpy(actual.data(), gpu + at, bytes, cudaMemcpyDeviceToHost));
            source.clear();
            source.seekg(static_cast<std::streamoff>(
                reader.payload_offset() + physical.offset + offset + at));
            source.read(reinterpret_cast<char*>(expected.data()),
                        static_cast<std::streamsize>(bytes));
            if (!source || !std::equal(actual.begin(), actual.begin() + bytes,
                                       expected.begin())) {
                throw std::runtime_error(std::string(name) + ": GPU bytes differ from GSQ v3");
            }
        };
        const auto sample = static_cast<std::size_t>(std::min<std::uint64_t>(count, 32));
        compare(0, sample);
        if (count > sample) { compare(count - sample, sample); }
        viewed.insert(object.index);
    };
    const auto verify_weight = [&](std::string_view name, const WeightPlan& weight,
                                   const ninfer::Weight& actual, int rows, int columns) {
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(weight.object.index));
        const auto* base = static_cast<const std::byte*>(backing.device_data(weight.object));
        if (weight.parent_rows <= 0 || weight.parent_columns != columns ||
            weight.selected_rows != rows || weight.first_row < 0 ||
            weight.first_row + rows > weight.parent_rows ||
            physical.shape != std::vector<std::uint64_t>{
                static_cast<std::uint64_t>(weight.parent_rows),
                static_cast<std::uint64_t>(columns)} ||
            physical.format != weight.format ||
            actual.qtype != ninfer::artifact::qtype_for(weight.format) ||
            actual.n != rows || actual.k != columns) {
            throw std::runtime_error(std::string(name) + ": DFlash2 native shape/codec differs");
        }
        const auto first = static_cast<std::uint64_t>(weight.first_row);
        if (weight.format == NumericFormat::BF16) {
            const auto stride = static_cast<std::uint64_t>(columns) * sizeof(std::uint16_t);
            const auto offset = first * stride;
            if (physical.layout != ninfer::artifact::StorageLayout::ContiguousLeV1 ||
                actual.layout != ninfer::QuantLayout::Contiguous ||
                actual.qdata != base + offset || actual.scales != nullptr) {
                throw std::runtime_error(std::string(name) + ": BF16 ModelView offset differs");
            }
            verify_bytes(weight.object, offset,
                         static_cast<const std::byte*>(actual.qdata),
                         static_cast<std::uint64_t>(rows) * stride, name);
        } else if (weight.format == NumericFormat::W8G32_F16S) {
            const std::array<std::uint64_t, 2> shape{
                static_cast<std::uint64_t>(weight.parent_rows),
                static_cast<std::uint64_t>(columns)};
            const auto geometry = ninfer::artifact::row_split_geometry(weight.format, shape);
            const auto low_stride = geometry.groups_per_row * geometry.low_bytes_per_group;
            const auto scale_stride = geometry.groups_per_row * sizeof(std::uint16_t);
            const auto low_offset = first * low_stride;
            const auto scale_offset = geometry.scale_plane_offset + first * scale_stride;
            if (physical.layout != ninfer::artifact::StorageLayout::RowSplitK128V1 ||
                actual.layout != ninfer::QuantLayout::RowSplit ||
                actual.qdata != base + low_offset || actual.qhigh != nullptr ||
                actual.scales != base + scale_offset ||
                actual.group != 32) {
                throw std::runtime_error(std::string(name) + ": W8 ModelView planes differ");
            }
            verify_bytes(weight.object, low_offset,
                         static_cast<const std::byte*>(actual.qdata),
                         static_cast<std::uint64_t>(rows) * low_stride, name);
            verify_bytes(weight.object, scale_offset,
                         static_cast<const std::byte*>(actual.scales),
                         static_cast<std::uint64_t>(rows) * scale_stride, name);
        } else {
            throw std::runtime_error(std::string(name) + ": unexpected DFlash2 codec");
        }
    };
    const auto verify_tensor = [&](std::string_view name,
                                   ninfer::artifact::ObjectHandle object,
                                   const ninfer::Tensor& actual, std::uint64_t elements) {
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(object.index));
        if (physical.layout != ninfer::artifact::StorageLayout::ContiguousLeV1 ||
            physical.format != NumericFormat::BF16 ||
            actual.dtype != ninfer::DType::BF16 ||
            actual.data != backing.device_data(object) ||
            static_cast<std::uint64_t>(actual.numel()) != elements) {
            throw std::runtime_error(std::string(name) + ": BF16 tensor ModelView differs");
        }
        verify_bytes(object, 0, static_cast<const std::byte*>(actual.data),
                     elements * sizeof(std::uint16_t), name);
    };
    constexpr int H = DFlashConfig::hidden;
    constexpr int K = DFlashConfig::kv_size;
    constexpr int Q = DFlashConfig::query_size;
    constexpr int I = DFlashConfig::intermediate;
    constexpr int C = DFlashConfig::dynamic_conv_rows;
    verify_weight("DFlash2 feature projection", plan.feature_projection,
                  native.feature_projection, H, DFlashConfig::feature_rows);
    verify_tensor("DFlash2 context norm", plan.context_norm, native.context_norm, H);
    verify_tensor("DFlash2 final norm", plan.final_norm, native.final_norm, H);
    for (std::size_t index = 0; index < plan.layers.size(); ++index) {
        const auto& l = plan.layers[index];
        const auto& w = native.layers[index];
        verify_tensor("DFlash2 input norm", l.input_norm, w.input_norm, H);
        verify_tensor("DFlash2 post norm", l.post_attention_norm, w.post_attention_norm, H);
        verify_weight("DFlash2 fused QKV", l.query_key_value, w.query_key_value,
                      Q + 2 * K, H);
        auto context = l.query_key_value;
        context.first_row = Q;
        context.selected_rows = K;
        verify_weight("DFlash2 context K", context, w.context_key, K, H);
        context.first_row = Q + K;
        verify_weight("DFlash2 context V", context, w.context_value, K, H);
        verify_tensor("DFlash2 Q norm", l.query_norm, w.query_norm, DFlashConfig::head_dim);
        verify_tensor("DFlash2 K norm", l.key_norm, w.key_norm, DFlashConfig::head_dim);
        verify_weight("DFlash2 attention output", l.attention_output,
                      w.attention_output, H, Q);
        verify_weight("DFlash2 gate/up", l.gate_up, w.gate_up, 2 * I, H);
        verify_weight("DFlash2 down", l.down, w.down, H, I);
        verify_tensor("DFlash2 attention conv", l.attention_conv.base_kernel,
                      w.attention_conv.base_kernel, 4 * H);
        verify_weight("DFlash2 attention conv projection",
                      l.attention_conv.kernel_projection,
                      w.attention_conv.kernel_projection, C, H);
        verify_tensor("DFlash2 MLP conv", l.mlp_conv.base_kernel,
                      w.mlp_conv.base_kernel, 4 * H);
        verify_weight("DFlash2 MLP conv projection", l.mlp_conv.kernel_projection,
                      w.mlp_conv.kernel_projection, C, H);
        if (w.attention_conv.base_kernel.ne[0] != H ||
            w.attention_conv.base_kernel.ne[1] != 2 ||
            w.attention_conv.base_kernel.ne[2] != 2 ||
            w.mlp_conv.base_kernel.ne[0] != H ||
            w.mlp_conv.base_kernel.ne[1] != 2 ||
            w.mlp_conv.base_kernel.ne[2] != 2) {
            throw std::runtime_error("DFlash2 dynamic convolution tensor geometry differs");
        }
    }
    const auto& selector = plan.candidate_selector;
    const auto& selector_view = native.candidate_selector;
    verify_weight("DFlash2 selector hidden projection", selector.hidden_projection,
                  selector_view.hidden_projection, DFlashConfig::selector_dim, H);
    verify_tensor("DFlash2 predecessor codebook", selector.predecessor_codebook,
                  selector_view.predecessor_codebook,
                  static_cast<std::uint64_t>(DFlashConfig::selector_dim) * TextConfig::output_rows);
    verify_tensor("DFlash2 successor codebook", selector.successor_codebook,
                  selector_view.successor_codebook,
                  static_cast<std::uint64_t>(DFlashConfig::selector_dim) * TextConfig::output_rows);
    if (selector_view.predecessor_codebook.ne[0] != DFlashConfig::selector_dim ||
        selector_view.predecessor_codebook.ne[1] != TextConfig::output_rows ||
        selector_view.successor_codebook.ne[0] != DFlashConfig::selector_dim ||
        selector_view.successor_codebook.ne[1] != TextConfig::output_rows ||
        viewed != selected) {
        throw std::runtime_error("DFlash2 native ModelView did not consume all 66 parents");
    }
    std::cout << "real GSQ v3 DFlash2: " << selected.size()
              << " physical parents, " << materialization.device_capacity_bytes
              << " device bytes; native ModelView and exact source-file plane oracle passed\n";
    return 0;
}

int verify_real_v3_vision_materialization() {
    // The real Vision tower occupies 282 MiB in the GSQ v3 artifact. Select only
    // its 333 physical parents, then materialize them through the same common
    // ModelView builder used during an actual Vision-enabled model startup.
    const auto path = std::filesystem::path(NINFER_SOURCE_DIR) / ".." / ".." /
                      "real-models" / "qwen3_8_gsq_rco_iq3_s_v3.ninfer";
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real GSQ v3 Vision weights are not available\n";
        return 77;
    }
    ninfer::artifact::Reader reader(path);
    const auto* directory = reader.v3_directory();
    if (!directory) {
        throw std::runtime_error("real Vision source has no v3 directory");
    }
    const auto expected_bytes = reader.payload_offset() +
        directory->at("files").at(0).at("payload_bytes").get<std::uint64_t>();
    if (std::filesystem::file_size(path) != expected_bytes) {
        std::cerr << "skip: real GSQ v3 Vision payload is incomplete\n";
        return 77;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "skip: no CUDA device for real Vision materialization\n";
        return 77;
    }
    auto choice = features(ninfer::SpeculativeBackend::None);
    choice.vision = true;
    ninfer::artifact::Binder complete_binder(reader);
    const auto complete = bind_v3_text_artifact(complete_binder, *directory, choice);
    const auto& binding = complete.bindings;
    {
        ninfer::artifact::Binder package_binder(reader);
        ninfer::EngineOptions options;
        options.enable_vision = true;
        const auto package = Package::plan_load(
            package_binder, options, WeightsProfile::Qwen38GroupwiseInt);
        if (package.materialization().device_objects.size() !=
                complete.materialization.device_objects.size()) {
            throw std::runtime_error("Vision package route lost v3 physical parents");
        }
    }

    std::unordered_set<std::size_t> selected;
    const auto add = [&](ninfer::artifact::ObjectHandle object) {
        selected.insert(object.index);
    };
    const auto& backbone = binding.vision_backbone;
    add(backbone.patch_embedding);
    add(backbone.patch_embedding_bias);
    add(backbone.position_embedding);
    for (const auto& layer : backbone.layers) {
        for (const auto object : {
                 layer.qkv, layer.qkv_bias, layer.output, layer.output_bias,
                 layer.fc1, layer.fc1_bias, layer.fc2, layer.fc2_bias,
                 layer.norm1_weight, layer.norm1_bias, layer.norm2_weight, layer.norm2_bias,
             }) {
            add(object);
        }
    }
    for (const auto object : {
             binding.vision_merger_input.fc1, binding.vision_merger_input.fc1_bias,
             binding.vision_merger_norm.weight, binding.vision_merger_norm.bias,
             binding.vision_merger_fc2, binding.vision_merger_fc2_bias,
         }) {
        add(object);
    }
    if (selected.size() != 333) {
        throw std::runtime_error("real GSQ Vision physical parent count changed");
    }

    ninfer::artifact::Binder only_vision(reader);
    for (std::size_t index = 0; index < reader.objects().size(); ++index) {
        const auto& descriptor = reader.objects()[index];
        if (const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&descriptor)) {
            const auto object = only_vision.require_tensor(
                tensor->name, tensor->format, tensor->layout, tensor->shape);
            if (selected.contains(index)) {
                only_vision.materialize_on_device(object);
            } else {
                only_vision.validate_only(object);
            }
        } else {
            const auto& resource = std::get<ninfer::artifact::ResourceDescriptor>(descriptor);
            const auto object = only_vision.require_resource(resource.name, resource.encoding);
            only_vision.validate_only(object);
        }
    }
    const auto materialization = only_vision.finish();
    if (materialization.device_objects.size() != selected.size() ||
        !materialization.host_objects.empty() ||
        materialization.device_capacity_bytes > (std::uint64_t{384} << 20)) {
        throw std::runtime_error("Vision-only device upload exceeds expected memory");
    }

    ninfer::DeviceContext device(0);
    const auto backing = ninfer::artifact::materialize(reader, materialization, device);
    ninfer::targets::qwen3_6::VisionWeights native{};
    native.common = ninfer::targets::qwen3_6::materialize_vision_common(
        backing, backbone, binding.vision_merger_input, binding.vision_merger_norm);
    native.merger_fc2 = ninfer::artifact::materialized_weight(
        backing, binding.vision_merger_fc2, NumericFormat::W8G32_F16S,
        TextConfig::hidden, ninfer::targets::qwen3_6::VisionBackboneConfig::merger_hidden);
    native.merger_fc2_bias = ninfer::artifact::materialized_tensor(
        backing, binding.vision_merger_fc2_bias, NumericFormat::BF16,
        {TextConfig::hidden});

    // Exact independent oracle: compare the first and last encoded block of
    // each sampled GPU plane against the corresponding bytes in the source file.
    // Checking qhigh and scales separately catches shifted plane pointers.
    std::ifstream source(path, std::ios::binary);
    const auto verify_bytes = [&](ninfer::artifact::ObjectHandle object,
                                  std::uint64_t offset, const std::byte* gpu,
                                  std::uint64_t count, std::string_view name) {
        if (gpu == nullptr || count == 0) {
            throw std::runtime_error(std::string(name) + ": empty Vision device plane");
        }
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(object.index));
        const auto compare = [&](std::uint64_t at, std::size_t size) {
            std::array<std::byte, 32> actual{}, expected{};
            CUDA_CHECK(cudaMemcpy(actual.data(), gpu + at, size, cudaMemcpyDeviceToHost));
            source.clear();
            source.seekg(static_cast<std::streamoff>(
                reader.payload_offset() + physical.offset + offset + at));
            source.read(reinterpret_cast<char*>(expected.data()),
                        static_cast<std::streamsize>(size));
            if (!source || !std::equal(actual.begin(), actual.begin() + size, expected.begin())) {
                throw std::runtime_error(std::string(name) + ": Vision GPU payload differs");
            }
        };
        const auto sample = static_cast<std::size_t>(std::min<std::uint64_t>(count, 32));
        compare(0, sample);
        if (count > sample) { compare(count - sample, sample); }
    };
    const auto verify_weight = [&](std::string_view name,
                                   ninfer::artifact::ObjectHandle object,
                                   const ninfer::Weight& actual, NumericFormat format,
                                   int rows, int columns) {
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(object.index));
        const auto shape = std::array<std::uint64_t, 2>{
            static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
        const auto geometry = ninfer::artifact::row_split_geometry(format, shape);
        const auto* base = static_cast<const std::byte*>(backing.device_data(object));
        if (physical.format != format ||
            physical.layout != ninfer::artifact::StorageLayout::RowSplitK128V1 ||
            physical.shape != std::vector<std::uint64_t>(shape.begin(), shape.end()) ||
            actual.qtype != ninfer::artifact::qtype_for(format) ||
            actual.layout != ninfer::QuantLayout::RowSplit ||
            actual.n != rows || actual.k != columns ||
            actual.qdata != base || actual.payload != base ||
            actual.payload_bytes != geometry.encoded_bytes ||
            actual.scales != base + geometry.scale_plane_offset ||
            actual.qhigh != (geometry.high_plane_bytes
                                 ? base + geometry.high_plane_offset : nullptr)) {
            throw std::runtime_error(std::string(name) + ": native Vision weight view differs");
        }
        verify_bytes(object, 0, static_cast<const std::byte*>(actual.qdata),
                     geometry.low_plane_bytes, name);
        if (geometry.high_plane_bytes) {
            verify_bytes(object, geometry.high_plane_offset,
                         static_cast<const std::byte*>(actual.qhigh),
                         geometry.high_plane_bytes, name);
        }
        verify_bytes(object, geometry.scale_plane_offset,
                     static_cast<const std::byte*>(actual.scales),
                     geometry.scale_plane_bytes, name);
    };
    const auto verify_bf16 = [&](std::string_view name,
                                 ninfer::artifact::ObjectHandle object,
                                 const ninfer::Tensor& actual, std::uint64_t elements) {
        const auto& physical = std::get<ninfer::artifact::TensorDescriptor>(
            reader.objects().at(object.index));
        if (physical.format != NumericFormat::BF16 ||
            physical.layout != ninfer::artifact::StorageLayout::ContiguousLeV1 ||
            actual.data != backing.device_data(object) ||
            static_cast<std::uint64_t>(actual.numel()) != elements) {
            throw std::runtime_error(std::string(name) + ": native Vision BF16 view differs");
        }
        verify_bytes(object, 0, static_cast<const std::byte*>(actual.data),
                     elements * sizeof(std::uint16_t), name);
    };
    using V = ninfer::targets::qwen3_6::VisionBackboneConfig;
    verify_weight("patch embedding", backbone.patch_embedding,
                  native.common.patch_embedding, NumericFormat::Q6G64_F16S,
                  V::hidden, V::patch_dim);
    verify_bf16("position embedding", backbone.position_embedding,
                native.common.position_embedding, V::position_embeddings * V::hidden);
    for (const auto layer : {0, V::layers / 2, V::layers - 1}) {
        const auto& plan = backbone.layers[layer];
        const auto& view = native.common.layers[layer];
        verify_weight("Vision QKV", plan.qkv, view.qkv, NumericFormat::Q4G64_F16S,
                      3 * V::hidden, V::hidden);
        verify_bf16("Vision QKV bias", plan.qkv_bias, view.qkv_bias, 3 * V::hidden);
        verify_weight("Vision attention output", plan.output, view.output,
                      NumericFormat::Q5G64_F16S, V::hidden, V::hidden);
        verify_weight("Vision MLP fc1", plan.fc1, view.fc1,
                      NumericFormat::Q4G64_F16S, V::intermediate, V::hidden);
        verify_weight("Vision MLP fc2", plan.fc2, view.fc2,
                      NumericFormat::Q5G64_F16S, V::hidden, V::intermediate);
        verify_bf16("Vision norm1", plan.norm1_weight, view.norm1_weight, V::hidden);
    }
    verify_weight("Vision merger fc1", binding.vision_merger_input.fc1,
                  native.common.merger_fc1, NumericFormat::W8G32_F16S,
                  V::merger_hidden, V::merger_hidden);
    verify_weight("Vision merger fc2", binding.vision_merger_fc2,
                  native.merger_fc2, NumericFormat::W8G32_F16S,
                  TextConfig::hidden, V::merger_hidden);
    verify_bf16("Vision merger bias", binding.vision_merger_fc2_bias,
                native.merger_fc2_bias, TextConfig::hidden);
    std::cout << "real GSQ v3 Vision: " << selected.size()
              << " physical parents, " << materialization.device_capacity_bytes
              << " device bytes; native ModelView and source-file byte oracle passed\n";
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
    if (argc == 2 && std::string_view(argv[1]) == "--v3-dflash2-device") {
        return verify_real_v3_dflash2_materialization();
    }
    if (argc == 2 && std::string_view(argv[1]) == "--v3-vision-device") {
        return verify_real_v3_vision_materialization();
    }
    if (argc == 2 && std::string_view(argv[1]) == "--v3-mtp-materialize") {
        for (const auto& [filename, format] : {
                 std::pair{"qwen3_8_gsq_rco_iq3_s_v3.ninfer", NumericFormat::GGUF_Q6_K},
                 std::pair{"qwen3_6_27b.ninfer", NumericFormat::W8G32_F16S},
             }) {
            if (const int result = verify_real_v3_mtp_materialization(filename, format);
                result != 0) {
                return result;
            }
        }
        return 0;
    }
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
