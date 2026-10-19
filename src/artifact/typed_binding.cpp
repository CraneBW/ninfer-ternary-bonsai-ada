// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "artifact/typed_binding.h"

#include "artifact/materializer.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace ninfer::artifact {
namespace {

StorageLayout storage_layout_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return StorageLayout::ContiguousLeV1;
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
    case NumericFormat::Q2G64_F16S:
    case NumericFormat::T2G128_F16S:
    case NumericFormat::PTQ1_0_G128:
    case NumericFormat::PQ2_0_G128:
        return StorageLayout::RowSplitK128V1;
    case NumericFormat::NVFP4:
        return StorageLayout::BlockScaleK16M128x4V1;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return StorageLayout::RowScaleV1;
    case NumericFormat::FP8_E4M3FN_ROW_FP16S:
        return StorageLayout::RowInterleavedV1;
    case NumericFormat::GGUF_Q8_0:
    case NumericFormat::GGUF_Q2_K:
    case NumericFormat::GGUF_Q3_K:
    case NumericFormat::GGUF_Q4_K:
    case NumericFormat::GGUF_Q5_K:
    case NumericFormat::GGUF_Q6_K:
    case NumericFormat::GGUF_IQ2_XXS:
    case NumericFormat::GGUF_IQ2_XS:
    case NumericFormat::GGUF_IQ2_S:
    case NumericFormat::GGUF_IQ3_XXS:
    case NumericFormat::GGUF_IQ3_S:
    case NumericFormat::GGUF_IQ1_S:
    case NumericFormat::GGUF_IQ1_M:
    case NumericFormat::GGUF_IQ4_NL:
    case NumericFormat::GGUF_IQ4_XS:
    case NumericFormat::GGUF_Q4_0:
    case NumericFormat::GGUF_Q5_0:
    case NumericFormat::GGUF_Q2_0:
        return StorageLayout::GgufBlocksV1;
    }
    throw std::logic_error("unhandled numeric format");
}

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return DType::BF16;
    case NumericFormat::FP32:
        return DType::FP32;
    case NumericFormat::I32:
        return DType::I32;
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

Weight contiguous_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload       = materialized.device_data(handle);
    out.qdata         = out.payload;
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

Weight row_split_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry          = row_split_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh       = geometry.high_plane_bytes == 0 ? nullptr : bytes + geometry.high_plane_offset;
    out.scales      = bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

Weight row_scale_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowScaleGeometry geometry          = row_scale_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = qtype_for(format);
    out.layout          = QuantLayout::RowScale;
    out.group_size      = static_cast<std::uint32_t>(geometry.columns);
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = columns;
    out.scale_dtype     = DType::BF16;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 2;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 2;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

Weight gguf_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                   NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const QType qtype = qtype_for(format);
    if (!is_gguf(qtype)) {
        throw std::invalid_argument("GGUF materialization requires ggml block format");
    }
    const auto block = gguf_block_shape(qtype);
    if (rows <= 0 || columns <= 0 || columns % block.elements != 0) {
        throw std::invalid_argument("GGUF materialization requires whole blocks per row");
    }
    const std::array<std::uint64_t, 2> shape = {
        static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    const auto bytes = tensor_encoded_size(StorageLayout::GgufBlocksV1, format, shape);
    Weight out{};
    out.payload        = materialized.device_data(handle);
    out.payload_bytes  = bytes;
    out.qdata          = out.payload;
    out.qtype          = qtype;
    out.layout         = QuantLayout::GgufBlocks;
    out.group          = block.elements;
    out.group_size     = static_cast<std::uint32_t>(block.elements);
    out.n              = rows;
    out.k              = columns;
    out.ndim           = 2;
    out.shape[0]       = rows;
    out.shape[1]       = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    // ggml scales are part of each block; there is no separate scale plane.
    return out;
}

Weight row_interleaved_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                              std::int32_t rows, std::int32_t columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("invalid FP8 row-interleaved dimensions");
    }
    const auto* data = static_cast<const std::byte*>(materialized.device_data(handle));
    const std::array<std::uint64_t, 2> shape = {
        static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    Weight out{};
    out.payload         = data;
    out.payload_bytes   = tensor_encoded_size(StorageLayout::RowInterleavedV1,
                                              NumericFormat::FP8_E4M3FN_ROW_FP16S, shape);
    out.qdata           = data;
    out.scales          = data + columns;
    out.qtype           = QType::FP8_E4M3FN_ROW_FP16;
    out.layout          = QuantLayout::RowInterleaved;
    out.scale_dtype     = DType::FP16;
    out.group           = columns;
    out.group_size      = static_cast<std::uint32_t>(columns);
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = static_cast<std::int64_t>(columns) + 2;
    out.scale_nb[1]     = out.scale_nb[0] * rows;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

} // namespace

// Public because the correspondence is not the binder's private business: a caller that has to
// reason about a tensor's format before it is materialised (choices that depend on what a kernel
// can accept, made while the artifact is still a plan) needs the same qtype the materialiser will
// stamp on the Weight. Declared in the header, defined here, so there is one table.
QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return QType::BF16_CTRL;
    case NumericFormat::FP32:
        return QType::FP32_CTRL;
    case NumericFormat::I32:
        return QType::I32_CTRL;
    case NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    case NumericFormat::NVFP4:
        return QType::NVFP4;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return QType::FP8_E4M3FN_ROW_BF16S;
    case NumericFormat::PTQ1_0_G128:
        return QType::PTQ1_0_G128;
    case NumericFormat::PQ2_0_G128:
        return QType::PQ2_0_G128;
    case NumericFormat::Q2G64_F16S: return QType::Q2_G64_FP16;
    case NumericFormat::T2G128_F16S: return QType::T2_G128_FP16;
    case NumericFormat::FP8_E4M3FN_ROW_FP16S: return QType::FP8_E4M3FN_ROW_FP16;
    case NumericFormat::GGUF_Q8_0: return QType::GGUF_Q8_0;
    case NumericFormat::GGUF_Q2_K: return QType::GGUF_Q2_K;
    case NumericFormat::GGUF_Q3_K: return QType::GGUF_Q3_K;
    case NumericFormat::GGUF_Q4_K: return QType::GGUF_Q4_K;
    case NumericFormat::GGUF_Q5_K: return QType::GGUF_Q5_K;
    case NumericFormat::GGUF_Q6_K: return QType::GGUF_Q6_K;
    case NumericFormat::GGUF_IQ2_XXS: return QType::GGUF_IQ2_XXS;
    case NumericFormat::GGUF_IQ2_XS: return QType::GGUF_IQ2_XS;
    case NumericFormat::GGUF_IQ2_S: return QType::GGUF_IQ2_S;
    case NumericFormat::GGUF_IQ3_XXS: return QType::GGUF_IQ3_XXS;
    case NumericFormat::GGUF_IQ3_S: return QType::GGUF_IQ3_S;
    case NumericFormat::GGUF_IQ1_S: return QType::GGUF_IQ1_S;
    case NumericFormat::GGUF_IQ1_M: return QType::GGUF_IQ1_M;
    case NumericFormat::GGUF_IQ4_NL: return QType::GGUF_IQ4_NL;
    case NumericFormat::GGUF_IQ4_XS: return QType::GGUF_IQ4_XS;
    case NumericFormat::GGUF_Q4_0: return QType::GGUF_Q4_0;
    case NumericFormat::GGUF_Q5_0: return QType::GGUF_Q5_0;
    case NumericFormat::GGUF_Q2_0: return QType::GGUF_Q2_0;
    }
    throw std::logic_error("unhandled numeric format");
}

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    const ObjectHandle handle =
        binder.require_tensor(name, format, storage_layout_for(format),
                              std::span<const std::uint64_t>(shape.begin(), shape.size()));
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_declared_tensor(Binder& binder, std::string_view name,
                                  std::initializer_list<std::uint64_t> shape,
                                  TensorPlacement placement) {
    const auto* descriptor = binder.find(name);
    const auto* tensor = descriptor == nullptr ? nullptr :
                         std::get_if<TensorDescriptor>(descriptor);
    if (!tensor) {
        throw ArtifactError("required declared tensor is missing: " + std::string(name));
    }
    // require_tensor() performs the actual format, layout, and shape validation.
    const auto handle = binder.require_tensor(
        name, tensor->format, tensor->layout,
        std::span<const std::uint64_t>(shape.begin(), shape.size()));
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    const ObjectHandle handle = binder.require_resource(name, ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format,
                           std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), dtype_for(format), internal_shape);
}

Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns) {
    if (format == NumericFormat::NVFP4) {
        throw std::invalid_argument(
            "materialized_weight: NVFP4 requires target-validated weight and input divisors");
    }
    if (storage_layout_for(format) == StorageLayout::ContiguousLeV1) {
        return contiguous_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::RowScaleV1) {
        return row_scale_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::GgufBlocksV1) {
        return gguf_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::RowInterleavedV1) {
        return row_interleaved_weight(materialized, handle, rows, columns);
    }
    return row_split_weight(materialized, handle, format, rows, columns);
}

} // namespace ninfer::artifact
