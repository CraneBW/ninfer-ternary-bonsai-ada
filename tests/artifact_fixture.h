#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline constexpr std::array<std::uint8_t, 8> kMagic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 2,
};

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { output[i] = std::byte((value >> (i * 8)) & 0xff); }
}

struct TemporaryArtifact {
    std::filesystem::path path;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

inline TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix,
                                       const std::array<std::uint8_t, 8>& magic = kMagic) {
    const std::string json         = directory.dump();
    const auto payload_offset      = align_up(16 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            payload_bytes = std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                        object.at("bytes").get<std::uint64_t>());
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < magic.size(); ++i) { file[i] = std::byte{magic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    std::uint8_t marker = 1;
    for (const auto& object : directory.at("objects")) {
        if (object.contains("offset") && object.contains("bytes") &&
            nonnegative_integer(object.at("offset")) && nonnegative_integer(object.at("bytes"))) {
            const auto offset = object.at("offset").get<std::uint64_t>();
            const auto bytes  = object.at("bytes").get<std::uint64_t>();
            std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
        }
    }

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write artifact fixture"); }
    return {std::move(path)};
}

struct TemporaryV3Artifact {
    std::filesystem::path folder;
    std::filesystem::path path;
    TemporaryV3Artifact(std::filesystem::path directory, std::filesystem::path entry)
        : folder(std::move(directory)), path(std::move(entry)) {}
    ~TemporaryV3Artifact() {
        std::error_code ignored;
        std::filesystem::remove_all(folder, ignored);
    }
    TemporaryV3Artifact(const TemporaryV3Artifact&)            = delete;
    TemporaryV3Artifact& operator=(const TemporaryV3Artifact&) = delete;
    TemporaryV3Artifact(TemporaryV3Artifact&&)                 = default;
};

// The v3 header is 32 bytes and the payload of every continuation starts at byte 4096.
// The supplied payload is one logical stream; continuation boundaries deliberately may split
// tensor blocks or rows.
inline TemporaryV3Artifact write_v3_fixture(
    const Json& directory, std::string_view suffix,
    std::span<const std::byte> logical_payload, std::string_view raw_json = {}) {
    std::random_device random;
    std::filesystem::path folder;
    for (int attempt = 0; attempt < 16; ++attempt) {
        folder = std::filesystem::temp_directory_path() /
                 ("ninfer_v3_" + std::string(suffix) + "_" + std::to_string(random()));
        if (std::filesystem::create_directory(folder)) { break; }
        if (attempt == 15) { throw std::runtime_error("could not create v3 fixture directory"); }
    }
    auto entry = folder / "entry.ninfer";
    const std::string json = raw_json.empty() ? directory.dump() : std::string(raw_json);
    std::uint64_t copied = 0;
    for (std::size_t i = 0; i < directory.at("files").size(); ++i) {
        const auto& record  = directory.at("files").at(i);
        const auto file_size = record.at("payload_bytes").get<std::uint64_t>();
        if (file_size > logical_payload.size() - copied) {
            throw std::runtime_error("v3 fixture payload shorter than file records");
        }
        const auto named_path = i == 0 ? std::filesystem::path("entry.ninfer")
                                     : std::filesystem::path(record.at("path").get<std::string>());
        // Malformed path fixtures must never escape their private temp directory.
        const auto safe_name = named_path == named_path.filename() &&
                                       named_path != "." && named_path != ".." &&
                                       (i == 0 || named_path != entry.filename())
                                   ? named_path
                                   : std::filesystem::path("invalid_part.bin");
        const auto file_path = folder / safe_name;
        std::ofstream output(file_path, std::ios::binary | std::ios::trunc);
        std::array<std::byte, 32> header{};
        const auto magic = i == 0
            ? std::array<std::uint8_t, 8>{'N', 'I', 'N', 'F', 'E', 'R', 0, 3}
            : std::array<std::uint8_t, 8>{'N', 'I', 'N', 'P', 'R', 'T', 0, 3};
        for (std::size_t j = 0; j < magic.size(); ++j) { header[j] = std::byte{magic[j]}; }
        write_u64_le(header.data() + 8, i == 0 ? json.size() : i);
        header[16] = std::byte{0x71}; // same artifact UUID for entry and every part
        output.write(reinterpret_cast<const char*>(header.data()), header.size());
        if (i == 0) { output.write(json.data(), static_cast<std::streamsize>(json.size())); }
        const auto metadata_end = 32 + (i == 0 ? json.size() : 0);
        const auto payload_start = align_up(metadata_end, 4096);
        const std::vector<char> padding(payload_start - metadata_end, '\0');
        output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        output.write(reinterpret_cast<const char*>(logical_payload.data() + copied),
                     static_cast<std::streamsize>(file_size));
        if (!output) { throw std::runtime_error("could not write v3 fixture"); }
        copied += file_size;
    }
    if (copied != logical_payload.size()) {
        throw std::runtime_error("v3 fixture has unassigned payload bytes");
    }
    return {std::move(folder), std::move(entry)};
}

} // namespace ninfer::test::artifact_fixture
