// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "artifact/reader.h"

#include <limits>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q2G64_F16S:
        return {64, 16, 0};
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    // Prism ternary: 128-weight group -> base plane {qs} + high plane {qh} + fp16 scale.
    // PTQ1_0 = 24 B base-3 trits (5/byte) + 2 B extra trits (4/byte) + 2 B scale = 28 B/128.
    case NumericFormat::PTQ1_0_G128:
        return {128, 24, 2};
    // PQ2_0 = 32 B of 2-bit codes (no high plane) + 2 B scale = 34 B/128.
    case NumericFormat::PQ2_0_G128:
    case NumericFormat::T2G128_F16S:
        return {128, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, or I32");
    }
}

} // namespace

GgufBlockShape gguf_block_shape(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::GGUF_Q8_0: return {32, 34};
    case NumericFormat::GGUF_Q2_K: return {256, 84};
    case NumericFormat::GGUF_Q3_K: return {256, 110};
    case NumericFormat::GGUF_Q4_K: return {256, 144};
    case NumericFormat::GGUF_Q5_K: return {256, 176};
    case NumericFormat::GGUF_Q6_K: return {256, 210};
    case NumericFormat::GGUF_IQ2_XXS: return {256, 66};
    case NumericFormat::GGUF_IQ2_XS: return {256, 74};
    case NumericFormat::GGUF_IQ2_S: return {256, 82};
    case NumericFormat::GGUF_IQ3_XXS: return {256, 98};
    case NumericFormat::GGUF_IQ3_S: return {256, 110};
    case NumericFormat::GGUF_IQ1_S: return {256, 50};
    case NumericFormat::GGUF_IQ1_M: return {256, 56};
    case NumericFormat::GGUF_IQ4_NL: return {32, 18};
    case NumericFormat::GGUF_IQ4_XS: return {256, 136};
    case NumericFormat::GGUF_Q4_0: return {32, 18};
    case NumericFormat::GGUF_Q5_0: return {32, 22};
    case NumericFormat::GGUF_Q2_0: return {64, 18};
    default: return {};
    }
}

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::PTQ1_0_G128:
        return "PTQ1_0_G128";
    case NumericFormat::PQ2_0_G128:
        return "PQ2_0_G128";
    case NumericFormat::Q2G64_F16S: return "q2_g64_fp16";
    case NumericFormat::T2G128_F16S: return "t2_g128_fp16";
    case NumericFormat::FP8_E4M3FN_ROW_FP16S: return "fp8_e4m3fn_row_fp16";
    case NumericFormat::GGUF_Q8_0: return "gguf_q8_0";
    case NumericFormat::GGUF_Q2_K: return "gguf_q2_k";
    case NumericFormat::GGUF_Q3_K: return "gguf_q3_k";
    case NumericFormat::GGUF_Q4_K: return "gguf_q4_k";
    case NumericFormat::GGUF_Q5_K: return "gguf_q5_k";
    case NumericFormat::GGUF_Q6_K: return "gguf_q6_k";
    case NumericFormat::GGUF_IQ2_XXS: return "gguf_iq2_xxs";
    case NumericFormat::GGUF_IQ2_XS: return "gguf_iq2_xs";
    case NumericFormat::GGUF_IQ2_S: return "gguf_iq2_s";
    case NumericFormat::GGUF_IQ3_XXS: return "gguf_iq3_xxs";
    case NumericFormat::GGUF_IQ3_S: return "gguf_iq3_s";
    case NumericFormat::GGUF_IQ1_S: return "gguf_iq1_s";
    case NumericFormat::GGUF_IQ1_M: return "gguf_iq1_m";
    case NumericFormat::GGUF_IQ4_NL: return "gguf_iq4_nl";
    case NumericFormat::GGUF_IQ4_XS: return "gguf_iq4_xs";
    case NumericFormat::GGUF_Q4_0: return "gguf_q4_0";
    case NumericFormat::GGUF_Q5_0: return "gguf_q5_0";
    case NumericFormat::GGUF_Q2_0: return "gguf_q2_0";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1:
        return "row-scale-v1";
    case StorageLayout::RowInterleavedV1: return "row_interleaved_v1";
    case StorageLayout::GgufBlocksV1: return "gguf_blocks_v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t divisors) {
    if (!divisors || (divisors != 1 && format != NumericFormat::NVFP4)) {
        throw ArtifactError("only NVFP4 tensors can declare multiple divisors");
    }
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        const auto geometry = block_scale_geometry(format, shape);
        if (geometry.rows % divisors || (divisors > 1 && (geometry.rows / divisors) % 128)) {
            throw ArtifactError("NVFP4 divisors must cover complete 128-row tiles");
        }
        return checked_add(geometry.weight_divisor_offset,
                           checked_mul(divisors, 4, "NVFP4 divisors"), "NVFP4 tensor encoded size");
    }
    if (layout == StorageLayout::RowScaleV1) {
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowInterleavedV1) {
        if (format != NumericFormat::FP8_E4M3FN_ROW_FP16S || shape.size() != 2 ||
            shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row_interleaved_v1 requires FP8 row scales and rank two");
        }
        return checked_mul(shape[0], checked_add(shape[1], 2, "interleaved row stride"),
                           "interleaved tensor encoded size");
    }
    if (layout == StorageLayout::GgufBlocksV1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("gguf_blocks_v1 requires a positive rank-two shape");
        }
        const auto block = gguf_block_shape(format);
        if (!block.elements || shape[1] % block.elements != 0) {
            throw ArtifactError("gguf_blocks_v1 requires whole ggml blocks per row");
        }
        return checked_mul(shape[0], checked_mul(shape[1] / block.elements, block.bytes,
                           "GGUF row bytes"), "GGUF tensor encoded size");
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        throw ArtifactError("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-scale-v1 requires a positive rank-two shape");
    }

    RowScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, 2, "FP8 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
