#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "runtime/tp2/v3_rank_manifest.h"

#include <string>
#include <unordered_map>

namespace ninfer::runtime::tp2 {

struct RankMaterializationPlan {
    struct Parent {
        artifact::ObjectHandle handle;
        artifact::NumericFormat format;
        artifact::StorageLayout layout;
        std::uint64_t rows;
        std::uint64_t columns;
    };
    V3RankManifest manifest;
    artifact::MaterializationPlan materialization;
    std::unordered_map<std::string, Parent> parents;
    std::unordered_map<std::string, std::string> input_column_parents;
};

// Materialize each physical v3 parent exactly once, even when a GGUF parent
// backs multiple logical projection roles. Unselected objects are validated,
// not silently ignored or uploaded. Must be called on a rank-local Reader.
[[nodiscard]] RankMaterializationPlan plan_rank_projection_materialization(
    artifact::Binder& binder, const nlohmann::json& directory);

// Build a CUDA-backed logical Weight view for one rank-local semantic role.
// The owning MaterializedArtifact must outlive this non-owning Weight.
[[nodiscard]] Weight rank_projection_view(const RankMaterializationPlan& plan,
                                          const artifact::MaterializedArtifact& materialized,
                                          const std::string& role);

} // namespace ninfer::runtime::tp2
