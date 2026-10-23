#pragma once

// Conservative lexical rescue for query-conditioned KVMem compaction.
// The semantic mean-K scorer can miss an exact historical name/label even
// when that name appears verbatim in the user's short question. Match only
// rare prompt token IDs (never arbitrary substrings), and leave sink/action/
// recent bands ahead of the candidate pages in the window selector.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ninfer::kvmem {

// Query-conditioned selection changes the attention inputs of later history
// tokens, and therefore their native KV and recurrent StateImage. Equal text
// prefixes alone do not certify reuse of a compressed original-position run.
struct KvmemSelectionCondition {
    std::uint32_t prefill_policy = 0;
    std::vector<std::uint32_t> lexical;
    std::vector<std::uint32_t> older;
    bool operator==(const KvmemSelectionCondition&) const = default;
};

// Query-independent extension of the donor sink/recent prefill policy. Keep
// a bounded set of rare-content pages outside the recent band, so a repeated
// boilerplate history need not erase its distinct records before retrieval.
// Only history tokens are inputs; no question tokens, expected labels or
// model outputs enter this score. The exact resulting IDs bind cache reuse.
[[nodiscard]] inline std::vector<std::uint32_t> kvmem_content_anchor_pages(
    std::span<const std::int32_t> tokens, std::uint32_t history_end,
    std::uint32_t recent_tokens, std::uint32_t page_tokens = 64) {
    if (page_tokens == 0 || history_end > tokens.size() || history_end <= recent_tokens) {
        return {};
    }
    const auto pages = (history_end - recent_tokens) / page_tokens;
    std::unordered_map<std::int32_t, std::uint32_t> df;
    for (std::uint32_t page = 0; page < pages; ++page) {
        std::unordered_set<std::int32_t> seen;
        for (const auto token : tokens.subspan(page * page_tokens, page_tokens)) {
            if (token >= 0 && seen.insert(token).second) { ++df[token]; }
        }
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranked;
    for (std::uint32_t page = 0; page < pages; ++page) {
        std::unordered_set<std::int32_t> seen;
        std::uint32_t score = 0, rare = 0;
        for (const auto token : tokens.subspan(page * page_tokens, page_tokens)) {
            const auto it = df.find(token);
            if (it != df.end() && it->second <= 4 && seen.insert(token).second) {
                score += 5 - it->second;
                ++rare;
            }
        }
        if (rare >= 2) { ranked.emplace_back(score, page); }
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    std::vector<std::uint32_t> result;
    for (const auto [score, page] : ranked) {
        (void)score;
        // Include the following page for a record crossing the page boundary.
        for (auto id = page; id < std::min(page + 2, pages); ++id) {
            if (result.size() < 32 && std::find(result.begin(), result.end(), id) == result.end()) {
                result.push_back(id);
            }
        }
        if (result.size() == 32) { break; }
    }
    return result;
}

[[nodiscard]] inline std::vector<std::uint32_t> kvmem_lexical_recall_pages(
    std::span<const std::int32_t> tokens, std::uint32_t query_begin,
    std::span<const std::int32_t> query_tokens, std::uint32_t page_tokens = 64,
    std::vector<std::uint32_t>* older_matches = nullptr,
    std::uint32_t candidate_groups = 1) {
    if (older_matches) { older_matches->clear(); }
    if (page_tokens == 0 || query_begin > tokens.size() || query_tokens.empty() ||
        query_tokens.size() > 128) {
        return {};
    }
    // A page overlapping the current question is not historical evidence.
    const std::uint32_t pages = query_begin / page_tokens;
    if (pages < 4) { return {}; }

    std::unordered_map<std::int32_t, std::uint32_t> frequency;
    for (const std::int32_t token : query_tokens) {
        if (token >= 0) { frequency.try_emplace(token, 0); }
    }
    if (frequency.size() < 2) { return {}; }

    for (std::uint32_t page = 0; page < pages; ++page) {
        std::unordered_set<std::int32_t> seen;
        for (std::uint32_t col = 0; col < page_tokens; ++col) {
            const auto id = tokens[static_cast<std::size_t>(page) * page_tokens + col];
            if (frequency.contains(id) && seen.insert(id).second) { ++frequency[id]; }
        }
    }

    // Need at least two *different* rare query tokens in one history page;
    // common whitespace, punctuation, or repeated boilerplate cannot qualify.
    const std::uint32_t max_df = std::max<std::uint32_t>(2, pages / 12);
    std::vector<std::uint32_t> weights(pages, 0), hit_counts(pages, 0);
    std::vector<std::uint32_t> rarest_df(pages, max_df + 1);
    std::vector<std::vector<std::int32_t>> anchors(pages);
    std::vector<std::uint32_t> ranked;
    for (std::uint32_t page = 0; page < pages; ++page) {
        std::unordered_set<std::int32_t> seen;
        std::uint32_t hits = 0;
        std::uint32_t weight = 0;
        for (std::uint32_t col = 0; col < page_tokens; ++col) {
            const auto id = tokens[static_cast<std::size_t>(page) * page_tokens + col];
            const auto it = frequency.find(id);
            if (it != frequency.end() && it->second > 0 && it->second <= max_df &&
                seen.insert(id).second) {
                ++hits;
                anchors[page].push_back(id);
                weight += (max_df + 1 - it->second);
                rarest_df[page] = std::min(rarest_df[page], it->second);
            }
        }
        weights[page] = weight;
        hit_counts[page] = hits;
        std::sort(anchors[page].begin(), anchors[page].end());
        if (hits >= 2) { ranked.push_back(page); }
    }
    // Preserve the original priority, but keep a bounded set of distinct anchor
    // signatures. A previous question and a real name can have identical DF,
    // weight and hit counts; their scalar tie is not evidence of the same fact.
    std::sort(ranked.begin(), ranked.end(), [&](auto a, auto b) {
        if (rarest_df[a] != rarest_df[b]) { return rarest_df[a] < rarest_df[b]; }
        if (weights[a] != weights[b]) { return weights[a] > weights[b]; }
        if (hit_counts[a] != hit_counts[b]) { return hit_counts[a] > hit_counts[b]; }
        return a > b;
    });
    std::vector<std::uint32_t> representatives;
    std::vector<std::uint32_t> result;
    const auto limit = std::clamp(candidate_groups, 1U, 4U);
    for (const auto page : ranked) {
        const auto duplicate = std::find_if(representatives.begin(), representatives.end(),
            [&](auto selected) { return anchors[selected] == anchors[page]; });
        if (duplicate != representatives.end()) {
            if (older_matches) {
                for (auto id = page; id < std::min(page + 2, pages); ++id) {
                    older_matches->push_back(id);
                }
            }
            continue;
        }
        if (representatives.size() == limit) { continue; }
        representatives.push_back(page);
        result.push_back(page);
        if (page + 1 < pages) { result.push_back(page + 1); }
    }
    if (older_matches) {
        std::sort(older_matches->begin(), older_matches->end());
        older_matches->erase(std::unique(older_matches->begin(), older_matches->end()),
                              older_matches->end());
        std::erase_if(*older_matches, [&](auto id) {
            return std::find(result.begin(), result.end(), id) != result.end();
        });
    }
    return result;
}

// Unframed token fixtures and callers can use an in-prompt query span. Chat
// callers supply the content-only query above so role/control tokens cannot
// become rare historical evidence.
[[nodiscard]] inline std::vector<std::uint32_t> kvmem_lexical_recall_pages(
    std::span<const std::int32_t> tokens, std::uint32_t query_begin,
    std::uint32_t query_end, std::uint32_t page_tokens = 64) {
    if (query_begin >= query_end || query_end > tokens.size()) { return {}; }
    return kvmem_lexical_recall_pages(tokens, query_begin,
                                     tokens.subspan(query_begin, query_end - query_begin),
                                     page_tokens);
}

} // namespace ninfer::kvmem
