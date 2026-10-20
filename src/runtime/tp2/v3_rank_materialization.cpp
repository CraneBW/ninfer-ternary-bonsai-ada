#include "runtime/tp2/v3_rank_materialization.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>

namespace ninfer::runtime::tp2 {

RankMaterializationPlan plan_rank_projection_materialization(
    artifact::Binder& binder, const nlohmann::json& directory) {
    RankMaterializationPlan plan;
    plan.manifest = inspect_v3_rank_manifest(directory);
    std::unordered_set<std::string> selected;
    for (const auto& view : plan.manifest.projections) { selected.insert(view.parent); }
    // The GDN output physical columns are permuted by an explicit v3 Use.
    // The mapping is a device tensor, not metadata to be discarded.
    if (!directory.contains("uses") || !directory.at("uses").is_array()) {
        throw artifact::ArtifactError("TP2 v3 rank missing semantic Uses");
    }
    for (const auto& use : directory.at("uses")) {
        if (!use.is_object() || !use.contains("parameter") ||
            !use.at("parameter").is_string() || !use.contains("auxiliaries") ||
            !use.at("auxiliaries").is_object()) { continue; }
        const auto& parameter = use.at("parameter").get_ref<const std::string&>();
        const auto& auxiliaries = use.at("auxiliaries");
        if (!parameter.ends_with("/gdn/output") || !auxiliaries.contains("input_columns")) {
            continue;
        }
        const auto& ref = auxiliaries.at("input_columns");
        if (!ref.is_object() || ref.size() != 1 || !ref.contains("object") ||
            !ref.at("object").is_string()) {
            throw artifact::ArtifactError("TP2 invalid GDN output input_columns Use");
        }
        const auto id = ref.at("object").get<std::string>();
        if (!plan.input_column_parents.emplace(parameter, id).second) {
            throw artifact::ArtifactError("TP2 duplicate GDN input_columns Use: " + parameter);
        }
        selected.insert(id);
    }
    for (const auto& object : directory.at("objects")) {
        const std::string id = object.at("id").get<std::string>();
        const auto* stored = binder.find(id);
        if (stored == nullptr) { throw artifact::ArtifactError("TP2 missing stored parent " + id); }
        artifact::ObjectHandle handle;
        if (const auto* tensor = std::get_if<artifact::TensorDescriptor>(stored)) {
            handle = binder.require_tensor(id, tensor->format, tensor->layout, tensor->shape);
            if (selected.contains(id)) {
                binder.materialize_on_device(handle);
            } else {
                binder.validate_only(handle);
            }
            const auto rows = tensor->shape.empty() ? 1ULL : tensor->shape[0];
            const auto columns = tensor->shape.size() == 2 ? tensor->shape[1] : 1ULL;
            plan.parents.emplace(id, RankMaterializationPlan::Parent{
                                      handle, tensor->format, tensor->layout, rows, columns});
        } else {
            handle = binder.require_resource(id, artifact::ResourceEncoding::RawBytesV1);
            if (selected.contains(id)) {
                throw artifact::ArtifactError("TP2 selected projection is not a tensor: " + id);
            }
            binder.validate_only(handle);
        }
    }
    for (const auto& [parameter, id] : plan.input_column_parents) {
        const auto found = plan.parents.find(id);
        if (found == plan.parents.end() || found->second.format != artifact::NumericFormat::I32 ||
            found->second.layout != artifact::StorageLayout::ContiguousLeV1 ||
            found->second.rows != 3072 || found->second.columns != 1) {
            throw artifact::ArtifactError("TP2 invalid rank-local GDN columns map: " + parameter);
        }
    }
    plan.materialization = binder.finish();
    return plan;
}

Weight rank_projection_view(const RankMaterializationPlan& plan,
                            const artifact::MaterializedArtifact& materialized,
                            const std::string& role) {
    const auto view = std::find_if(plan.manifest.projections.begin(),
                                   plan.manifest.projections.end(),
                                   [&](const RankWeightView& item) { return item.role == role; });
    if (view == plan.manifest.projections.end()) {
        throw std::invalid_argument("TP2 rank projection role not bound: " + role);
    }
    const auto& parent = plan.parents.at(view->parent);
    if (parent.rows > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        parent.columns > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        view->element_begin % view->columns != 0 ||
        view->element_begin / view->columns + view->rows > parent.rows) {
        throw artifact::ArtifactError("TP2 rank logical view exceeds its physical parent");
    }
    Weight weight = artifact::materialized_weight(
        materialized, parent.handle, parent.format, static_cast<std::int32_t>(parent.rows),
        static_cast<std::int32_t>(parent.columns));
    if (weight.qdata == nullptr) {
        throw artifact::ArtifactError("TP2 physical CUDA parent not materialized: " + view->parent);
    }
    const auto first_row = view->element_begin / view->columns;
    if (first_row != 0 || view->rows != parent.rows) {
        std::uint64_t bytes_per_row = 0;
        if (weight.layout == QuantLayout::GgufBlocks) {
            const auto block = gguf_block_shape(weight.qtype);
            if (block.elements == 0 || view->columns % block.elements != 0) {
                throw artifact::ArtifactError("TP2 GGUF logical row is not block-aligned");
            }
            bytes_per_row = view->columns / block.elements * block.bytes;
        } else if (weight.layout == QuantLayout::Contiguous &&
                   (parent.format == artifact::NumericFormat::BF16 ||
                    parent.format == artifact::NumericFormat::FP32)) {
            bytes_per_row = view->columns *
                            (parent.format == artifact::NumericFormat::BF16 ? 2 : 4);
        } else {
            throw artifact::ArtifactError("TP2 rank sliced parent has unsupported encoding");
        }
        const auto* first = static_cast<const std::byte*>(weight.qdata) + first_row * bytes_per_row;
        weight.qdata = weight.payload = first;
        weight.payload_bytes = view->rows * bytes_per_row;
        weight.n = static_cast<std::int32_t>(view->rows);
        weight.shape[0] = weight.padded_shape[0] = static_cast<std::int32_t>(view->rows);
    }
    const auto columns = plan.input_column_parents.find(role);
    if (columns != plan.input_column_parents.end()) {
        weight.input_columns = static_cast<const std::int32_t*>(
            materialized.device_data(plan.parents.at(columns->second).handle));
        if (weight.input_columns == nullptr) {
            throw artifact::ArtifactError("TP2 GDN physical column map is not on CUDA");
        }
    }
    return weight;
}

} // namespace ninfer::runtime::tp2
