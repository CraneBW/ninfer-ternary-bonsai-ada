#pragma once

#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Encoding provenance only, never a KV eligibility shortcut. Adapted from this family's
// EncodedHistoryCache bounded storage. Generated BPE can differ from encode(decode(ids)); unlike
// the host-encode cache, these entries intentionally retain the committed generated ids.
class GeneratedHistoryCache {
public:
    static constexpr std::size_t max_entries = 8;
    static constexpr std::size_t max_ids = 262144;
    static constexpr std::size_t max_bytes = 2 * 1024 * 1024;
    static constexpr std::size_t max_markers = 4096;

    struct Hit {
        std::size_t canonical_tokens;
        std::size_t original_tokens;
    };
    void remember(const Tokenizer& tokenizer, std::vector<int> ids);
    [[nodiscard]] std::optional<Hit> apply(const Tokenizer& tokenizer, EncodedChat& chat) const;

private:
    struct Marker {
        std::size_t offset;
        int id;
        bool operator==(const Marker&) const = default;
    };
    struct Entry {
        std::string bytes;
        std::vector<int> ids;
        std::vector<Marker> markers;
    };
    mutable std::mutex mutex_;
    std::vector<Entry> entries_; // oldest first; completed publication replaces an identical entry
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
