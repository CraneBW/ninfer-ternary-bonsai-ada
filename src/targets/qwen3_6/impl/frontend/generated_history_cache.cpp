#include "targets/qwen3_6/impl/frontend/generated_history_cache.h"

#include <algorithm>
#include <limits>

namespace ninfer::targets::qwen3_6::frontend_internal {

void GeneratedHistoryCache::remember(const Tokenizer& tokenizer, std::vector<int> ids) {
    if (ids.empty() || ids.size() > max_ids) { return; }
    const auto end = tokenizer.decoded_token(ids.back());
    // Only a complete assistant turn provides a deterministic structural splice boundary.
    if (!end.added || end.bytes != "<|im_end|>") { return; }
    Entry entry;
    for (const int id : ids) {
        const auto token = tokenizer.decoded_token(id);
        if (token.bytes.empty() || token.bytes.size() > max_bytes - entry.bytes.size()) { return; }
        if (token.added) {
            if (entry.markers.size() == max_markers) { return; }
            entry.markers.push_back({entry.bytes.size(), id});
        }
        entry.bytes += token.bytes;
    }
    entry.ids = std::move(ids);
    entry.bytes.shrink_to_fit();
    entry.ids.shrink_to_fit();
    entry.markers.shrink_to_fit();
    std::lock_guard lock(mutex_);
    std::erase_if(entries_, [&](const Entry& old) { return old.bytes == entry.bytes; });
    if (entries_.size() == max_entries) { entries_.erase(entries_.begin()); }
    entries_.push_back(std::move(entry));
}

std::optional<GeneratedHistoryCache::Hit>
GeneratedHistoryCache::apply(const Tokenizer& tokenizer, EncodedChat& chat) const {
    if (!chat.media_token_runs.empty() || chat.input_ids.empty()) { return std::nullopt; }
    std::lock_guard lock(mutex_);
    if (entries_.empty()) { return std::nullopt; }
    std::string bytes;
    std::vector<Marker> markers;
    std::vector<std::size_t> offsets{0};
    offsets.reserve(chat.input_ids.size() + 1);
    for (const int id : chat.input_ids) {
        const auto token = tokenizer.decoded_token(id);
        if (token.bytes.empty()) { return std::nullopt; }
        if (token.added) { markers.push_back({bytes.size(), id}); }
        bytes += token.bytes;
        offsets.push_back(bytes.size());
    }
    const Entry* selected = nullptr;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (selected && it->bytes.size() <= selected->bytes.size()) { continue; }
        if (!bytes.starts_with(it->bytes)) { continue; }
        const auto marker_end = std::lower_bound(markers.begin(), markers.end(), it->bytes.size(),
            [](const Marker& marker, std::size_t offset) { return marker.offset < offset; });
        if (!std::equal(markers.begin(), marker_end, it->markers.begin(), it->markers.end())) {
            continue; // literal special-looking text is not a structural token
        }
        selected = &*it;
    }
    if (!selected) { return std::nullopt; }
    const Entry& entry = *selected;
    const auto end = std::lower_bound(offsets.begin(), offsets.end(), entry.bytes.size());
    if (end == offsets.end() || *end != entry.bytes.size()) { return std::nullopt; }
    const std::size_t canonical_count = end - offsets.begin();
    if (std::equal(entry.ids.begin(), entry.ids.end(), chat.input_ids.begin(),
                   chat.input_ids.begin() + canonical_count)) { return std::nullopt; }

    std::vector<std::size_t> original_offsets{0};
    original_offsets.reserve(entry.ids.size() + 1);
    for (const int id : entry.ids) {
        original_offsets.push_back(original_offsets.back() + tokenizer.decoded_token(id).bytes.size());
    }
    const auto remap = [&](std::uint32_t frontier) -> std::optional<std::uint32_t> {
        if (frontier > chat.input_ids.size()) { return std::nullopt; }
        std::size_t mapped;
        if (frontier >= canonical_count) {
            mapped = entry.ids.size() + frontier - canonical_count;
        } else {
            const auto found = std::lower_bound(original_offsets.begin(), original_offsets.end(),
                                                offsets[frontier]);
            if (found == original_offsets.end() || *found != offsets[frontier]) { return std::nullopt; }
            mapped = found - original_offsets.begin();
        }
        if (mapped > std::numeric_limits<std::uint32_t>::max()) { return std::nullopt; }
        return static_cast<std::uint32_t>(mapped);
    };
    // Transactional: a boundary inside an original BPE token declines the whole substitution.
    // No partly remapped prompt can escape to Program's identity/state checks.
    EncodedChat replacement = chat;
    if (replacement.rewrite_checkpoint) {
        const auto mapped = remap(replacement.rewrite_checkpoint->frontier);
        if (!mapped) { return std::nullopt; }
        replacement.rewrite_checkpoint->frontier = *mapped;
    }
    for (auto& frontier : replacement.rewrite_execution_frontiers) {
        const auto mapped = remap(frontier);
        if (!mapped) { return std::nullopt; }
        frontier = *mapped;
    }
    for (auto* boundaries : {&replacement.message_boundaries, &replacement.cache_boundaries}) {
        for (auto& frontier : *boundaries) {
            if (!frontier) { continue; }
            const auto mapped = remap(*frontier);
            if (!mapped) { return std::nullopt; }
            frontier = *mapped;
        }
    }
    replacement.input_ids = entry.ids;
    replacement.input_ids.insert(replacement.input_ids.end(), chat.input_ids.begin() + canonical_count,
                                 chat.input_ids.end());
    chat = std::move(replacement);
    return Hit{canonical_count, entry.ids.size()};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
