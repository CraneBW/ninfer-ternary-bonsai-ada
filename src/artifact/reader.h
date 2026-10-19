// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    PTQ1_0_G128,
    PQ2_0_G128,
    Q2G64_F16S,
    T2G128_F16S,
    FP8_E4M3FN_ROW_FP16S,
    GGUF_Q8_0,
    GGUF_Q2_K,
    GGUF_Q3_K,
    GGUF_Q4_K,
    GGUF_Q5_K,
    GGUF_Q6_K,
    GGUF_IQ2_XXS,
    GGUF_IQ2_XS,
    GGUF_IQ2_S,
    GGUF_IQ3_XXS,
    GGUF_IQ3_S,
    GGUF_IQ1_S,
    GGUF_IQ1_M,
    GGUF_IQ4_NL,
    GGUF_IQ4_XS,
    GGUF_Q4_0,
    GGUF_Q5_0,
    GGUF_Q2_0,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    RowInterleavedV1,
    GgufBlocksV1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t divisors = 1);

struct GgufBlockShape {
    std::uint64_t elements = 0;
    std::uint64_t bytes    = 0;
};

[[nodiscard]] GgufBlockShape gguf_block_shape(NumericFormat format) noexcept;

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint64_t divisors = 1;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    // v3 preserves the complete validated component/binding/use directory. Null for v2.
    [[nodiscard]] const nlohmann::json* v3_directory() const noexcept;
    [[nodiscard]] std::uint8_t container_version() const noexcept;
    [[nodiscard]] const std::array<std::byte, 16>& artifact_id() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    // Offsets here refer to the unified logical payload, for both v2 and v3.
    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    [[nodiscard]] std::vector<std::byte> read_object(const ObjectDescriptor& object) const;
    // For v3, absolute offsets are virtual coordinates starting at payload_offset();
    // a read can traverse multiple continuation files without copying/rewriting the artifact.
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
