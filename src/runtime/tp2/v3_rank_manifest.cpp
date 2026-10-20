#include "runtime/tp2/v3_rank_manifest.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace ninfer::runtime::tp2 {
namespace {

using Json = nlohmann::json;

std::uint64_t u64(const Json& value, std::string_view label) {
    if (!value.is_number_unsigned()) {
        throw std::invalid_argument("TP2 " + std::string(label) + " must be unsigned");
    }
    return value.get<std::uint64_t>();
}

struct Dims { std::uint64_t rows, columns; };

void add_view(V3RankManifest& result, const Json& bindings,
              const std::unordered_map<std::string, const Json*>& objects,
              const std::string& name, Dims expected) {
    if (!bindings.contains(name)) {
        throw std::invalid_argument("TP2 missing rank-local semantic role: " + name);
    }
    const auto& binding = bindings.at(name);
    if (!binding.is_object()) {
        throw std::invalid_argument("TP2 invalid rank-local binding: " + name);
    }
    bool partial = false;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::string id;
    if (binding.size() == 1 && binding.contains("object") &&
        binding.at("object").is_string()) {
        id = binding.at("object").get<std::string>();
    } else if (binding.size() == 1 && binding.contains("parts") &&
               binding.at("parts").is_array() && binding.at("parts").size() == 1) {
        const auto& part = binding.at("parts").at(0);
        if (!part.is_object() || part.size() != 2 || !part.contains("object") ||
            !part.at("object").is_string() || !part.contains("range") ||
            !part.at("range").is_array() || part.at("range").size() != 2) {
            throw std::invalid_argument("TP2 invalid logical range: " + name);
        }
        id = part.at("object").get<std::string>();
        begin = u64(part.at("range").at(0), name);
        end = u64(part.at("range").at(1), name);
        partial = true;
    } else {
        throw std::invalid_argument("TP2 expects whole or one contiguous logical part: " + name);
    }
    const auto found = objects.find(id);
    if (found == objects.end()) { throw std::invalid_argument("TP2 missing parent: " + id); }
    const auto& object = *found->second;
    if (object.value("kind", std::string{}) != "tensor" || !object.contains("shape") ||
        !object.at("shape").is_array() || object.at("shape").size() != 2) {
        throw std::invalid_argument("TP2 expected rank-local matrix parent: " + id);
    }
    const std::uint64_t parent_rows = u64(object.at("shape").at(0), id);
    const std::uint64_t parent_columns = u64(object.at("shape").at(1), id);
    if (parent_columns != expected.columns || expected.rows == 0 ||
        expected.columns == 0 || parent_rows == 0) {
        throw std::invalid_argument("TP2 physical matrix shape differs: " + name);
    }
    if (partial) {
        if (begin >= end || begin % expected.columns != 0 ||
            end - begin != expected.rows * expected.columns ||
            end / expected.columns > parent_rows) {
            throw std::invalid_argument("TP2 invalid physical row span: " + name);
        }
    } else if (parent_rows != expected.rows) {
        throw std::invalid_argument("TP2 whole physical parent rows differ: " + name);
    }
    result.projections.push_back({name, id, begin, expected.rows, expected.columns});
}

void check_projection(V3RankManifest& result, const Json& bindings,
                      const std::unordered_map<std::string, const Json*>& objects,
                      const std::string& layer, const char* suffix, Dims dims) {
    add_view(result, bindings, objects, layer + suffix, dims);
}

} // namespace

V3RankManifest inspect_v3_rank_manifest(const Json& directory) {
    if (!directory.is_object() || !directory.contains("metadata") ||
        !directory.at("metadata").is_object()) {
        throw std::invalid_argument("TP2 v3 directory requires metadata");
    }
    const auto& metadata = directory.at("metadata");
    if (!metadata.contains("tensor_parallel") ||
        !metadata.at("tensor_parallel").is_object()) {
        throw std::invalid_argument("TP2 v3 requires tensor_parallel metadata");
    }
    const auto& tp = metadata.at("tensor_parallel");
    if (!tp.contains("world_size") || !tp.at("world_size").is_number_integer() ||
        tp.at("world_size").get<int>() != 2 || !tp.contains("rank") ||
        !tp.at("rank").is_number_integer()) {
        throw std::invalid_argument("TP2 rank/world_size must describe exactly two ranks");
    }
    const int rank = tp.at("rank").get<int>();
    if ((rank != 0 && rank != 1) ||
        metadata.value("name", std::string{}) !=
            "qwen3.8-27b-tp2-rank" + std::to_string(rank) ||
        tp.value("source_model", std::string{}) != "qwen3.8-27b" ||
        tp.value("partition", std::string{}) != "exact-physical-v3-v1" ||
        !tp.value("requires_native_tp2", false)) {
        throw std::invalid_argument("TP2 mismatched v3 rank identity/partition");
    }
    if (!directory.contains("objects") || !directory.at("objects").is_array() ||
        !directory.contains("bindings") || !directory.at("bindings").is_object()) {
        throw std::invalid_argument("TP2 rank missing v3 objects or bindings");
    }
    std::unordered_map<std::string, const Json*> objects;
    for (const auto& object : directory.at("objects")) {
        if (!object.is_object() || !object.contains("id") || !object.at("id").is_string()) {
            throw std::invalid_argument("TP2 invalid physical object entry");
        }
        const auto id = object.at("id").get<std::string>();
        if (!objects.emplace(id, &object).second) {
            throw std::invalid_argument("TP2 duplicate physical parent: " + id);
        }
    }
    const auto& bindings = directory.at("bindings");
    V3RankManifest result;
    result.rank = rank;
    result.projections.reserve(64 * 15);
    for (int i = 0; i < 64; ++i) {
        const std::string layer = "text/layers/" + std::to_string(i) + "/";
        const auto check = [&](const char* suffix, Dims shape) {
            check_projection(result, bindings, objects, layer, suffix, shape);
        };
        check("mlp/gate", {8704, 5120});
        check("mlp/up", {8704, 5120});
        check("mlp/down", {5120, 8704});
        if ((i + 1) % 4 == 0) {
            check("attention/query", {3072, 5120});
            check("attention/gate", {3072, 5120});
            check("attention/key", {512, 5120});
            check("attention/value", {512, 5120});
            check("attention/output", {5120, 3072});
        } else {
            check("gdn/query", {1024, 5120});
            check("gdn/key", {1024, 5120});
            check("gdn/value", {3072, 5120});
            check("gdn/z", {3072, 5120});
            check("gdn/a_projection", {24, 5120});
            check("gdn/b_projection", {24, 5120});
            check("gdn/output", {5120, 3072});
            check("gdn/convolution", {4, 5120});
        }
    }
    return result;
}

} // namespace ninfer::runtime::tp2
