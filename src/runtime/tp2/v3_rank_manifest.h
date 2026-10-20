#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::runtime::tp2 {

// One rank-local logical weight, resolved to its real encoded v3 physical
// parent. Element offsets are logical (not encoded GGUF byte offsets).
struct RankWeightView {
    std::string role;
    std::string parent;
    std::uint64_t element_begin = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
};

struct V3RankManifest {
    int rank = -1;
    std::vector<RankWeightView> projections;
};

// Consumes the actual v3 rank directory produced by shard_v3_qwen38.py.
// Validates all 64 text-layer TP projection contracts against v3 physical
// objects and exact semantic parts, before any upload or ModelView usage.
// This is an input to a future rank-local executor, not a working executor.
[[nodiscard]] V3RankManifest inspect_v3_rank_manifest(const nlohmann::json& directory);

} // namespace ninfer::runtime::tp2
