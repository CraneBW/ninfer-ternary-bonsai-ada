#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + ": NVFP4 geometry overflows");
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(operation) + ": NVFP4 geometry overflows");
    }
    return left + right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* operation) {
    return checked_mul(checked_add(value, alignment - 1, operation) / alignment, alignment,
                       operation);
}

} // namespace

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation) {
    if (weight.n <= 0 || weight.k <= 0 || (weight.n % 128) != 0 || (weight.k % 64) != 0) {
        throw std::invalid_argument(std::string(operation) + ": NVFP4 requires N%128=0 and K%64=0");
    }

    const std::uint64_t elements = checked_mul(static_cast<std::uint64_t>(weight.n),
                                               static_cast<std::uint64_t>(weight.k), operation);
    Nvfp4WeightGeometry geometry{};
    geometry.code_plane_bytes   = elements / 2;
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    geometry.scale_plane_bytes  = elements / 16;
    geometry.required_payload_bytes =
        checked_add(checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation),
                    sizeof(float), operation);

    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.scale_dtype != DType::FP8_E4M3FN || weight.group_size != 16 || weight.group != 16 ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.qdata == nullptr || weight.scales == nullptr ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < geometry.required_payload_bytes ||
        !std::isfinite(weight.weight_scale_divisor) || weight.weight_scale_divisor <= 0.0F ||
        !std::isfinite(weight.input_scale_divisor) || weight.input_scale_divisor <= 0.0F ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata == payload && weight.scales == payload + geometry.scale_plane_offset) {
        return geometry;
    }
    // V3 may bind an M128-aligned logical row interval from a shared physical
    // NVFP4 parent. Its code and scale intervals have different base offsets:
    // [all parent codes][all parent scales][parent divisor]. Validate the
    // *entire* parent geometry and slice coordinates, not only alignment,
    // before allowing the kernels to use the two independently sliced planes.
    const std::uint64_t total_bytes = weight.payload_bytes;
    const std::uint64_t divisor_bytes = sizeof(float);
    const std::uint64_t divisor_denominator = checked_mul(9, static_cast<std::uint64_t>(weight.k), operation);
    if (total_bytes < divisor_bytes ||
        checked_mul(total_bytes - divisor_bytes, 16, operation) % divisor_denominator != 0) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    const std::uint64_t parent_rows =
        checked_mul(total_bytes - divisor_bytes, 16, operation) / divisor_denominator;
    if (parent_rows < static_cast<std::uint64_t>(weight.n) || parent_rows % 128 != 0) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 parent geometry");
    }
    const std::uint64_t parent_elements =
        checked_mul(parent_rows, static_cast<std::uint64_t>(weight.k), operation);
    const std::uint64_t parent_codes = parent_elements / 2;
    const std::uint64_t parent_scale_offset = align_up(parent_codes, 256, operation);
    if (checked_add(checked_add(parent_scale_offset, parent_elements / 16, operation),
                    divisor_bytes, operation) != total_bytes) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 parent payload");
    }
    const auto base_address = reinterpret_cast<std::uintptr_t>(payload);
    const auto code_address = reinterpret_cast<std::uintptr_t>(weight.qdata);
    const auto scale_address = reinterpret_cast<std::uintptr_t>(weight.scales);
    if (code_address < base_address || scale_address < base_address ||
        code_address - base_address > parent_codes) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    const std::uint64_t code_offset = code_address - base_address;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(weight.k) / 2;
    if (code_offset % row_bytes != 0) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 row offset");
    }
    const std::uint64_t first_row = code_offset / row_bytes;
    const std::uint64_t expected_scale_offset = parent_scale_offset +
        first_row * static_cast<std::uint64_t>(weight.k) / 16;
    if (first_row % 128 != 0 || first_row + weight.n > parent_rows ||
        scale_address - base_address != expected_scale_offset ||
        code_offset + geometry.code_plane_bytes > parent_codes ||
        expected_scale_offset + geometry.scale_plane_bytes > total_bytes - divisor_bytes) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    return geometry;
}

} // namespace ninfer::ops::detail
