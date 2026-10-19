#pragma once

// Host-side n-gram draft source for MTP rounds (--lookup-ngram). MTP's own proposals come from the
// draft head, which cannot copy a span it has read: a continuation of the request's own tokens is
// exactly what a lookup finds, and copying text (a quoted passage, a rewritten file, a repeated
// list) is where fixed draft widths leave the most acceptance on the table.
//
// The index maps a run of kContextTokens tokens to the positions where it occurred, most recent
// first. A lookup on the sequence's newest context therefore returns the continuation of an earlier
// occurrence of the same run. Only runs whose continuation is already known are indexed, so the
// newest context itself is never a hit: a hit always points at an occurrence that happened strictly
// earlier and has at least one following token to copy.
//
// A run that repeats but does not continue the same way is not a copy. A repeated phrase, a JSON
// key before a varied value, an identifier that differs in later digits: predicting such a reuse
// would displace drafts the draft head is getting right, so the index compares the newest
// occurrence's continuation with the previous occurrence's and marks the run conflicted as soon as
// the two diverge within kAgreementTokens tokens. Conflicted runs are never predicted from, which
// keeps formatted output on the draft head's drafts. For a real copy the earlier occurrence
// continues identically, so the run stays trusted.
//
// Agreement on the right is not enough on its own. A run that occurs once in the prompt has no
// second occurrence to conflict with, and generated JSON walks back onto the runs of an example it
// read: the eight tokens match, one or two tokens before them may match by coincidence, and the
// example's continuation is another example. A copy corroborates on both sides, because the tokens
// it has already reproduced stand before the matched run exactly as the earlier occurrence's own
// tokens do, so a hit is offered only when kCorroboratingTokens tokens before the match agree too.
// The first tokens of a copy are therefore drafted by the draft head, and the lookup takes over once
// the copy has enough of itself behind the run.
//
// A hit still has to be verified by the target model like any other draft, so a wrong prediction
// costs verify columns, never output tokens. Predictions are only offered when the run is long
// enough to be worth trusting and continues for at least kMinimumPrediction tokens; otherwise the
// request keeps MTP's own drafts.

#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class NgramLookup final {
public:
    // Long enough that prose rarely matches by accident, short enough that a copied passage locks
    // on within the first few generated tokens.
    static constexpr std::uint32_t kContextTokens = 8;
    // How many tokens after the run the two newest occurrences must agree on for the run to count as
    // a copy. A template that repeats a run but continues differently is caught inside this window;
    // at eight, JSON objects with identical structure still agreed past it often enough to spend a
    // round on a wrong prediction.
    static constexpr std::uint32_t kAgreementTokens = 16;
    // How many tokens before the matched occurrence must also agree for the run to count as a copy
    // in progress. A local pattern (a JSON example in the prompt, a template line that recurs)
    // matches the run itself while the tokens on its left diverge, and its continuation is the
    // varied part; a copy is reproducing a span, so it matches on both sides of the run.
    static constexpr std::uint32_t kCorroboratingTokens = 16;
    // A one-token prediction would displace a wider MTP proposal with a coin flip.
    static constexpr std::uint32_t kMinimumPrediction = 3;
    // Upper bound on what one lookup returns; the round's width bounds it further.
    static constexpr std::uint32_t kMaximumPrediction = 7;

    NgramLookup() = default;
    NgramLookup(const NgramLookup&) = delete;
    NgramLookup& operator=(const NgramLookup&) = delete;
    NgramLookup(NgramLookup&&) noexcept = default;
    NgramLookup& operator=(NgramLookup&&) noexcept = default;

    void reset() {
        history_.clear();
        slots_.clear();
        previous_.clear();
        conflicted_.clear();
    }

    [[nodiscard]] std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(history_.size());
    }

    // Appends tokens the sequence has committed. The token being appended completes the run before
    // it, so that run gains a known continuation and becomes predictable from here on.
    void append(std::span<const TokenId> tokens) {
        // The first append after a prefill carries the whole prompt; sizing the table for it once
        // costs one allocation instead of a rebuild per doubling on the round's critical path.
        if (!tokens.empty()) {
            const std::size_t needed = (history_.size() + tokens.size() + 1U) * 2U;
            if (slots_.size() < needed) { rebuild(needed); }
        }
        for (const TokenId token : tokens) {
            history_.push_back(token);
            const std::size_t last = history_.size() - 1U;
            if (last < kContextTokens) { continue; }
            const std::size_t context_begin = last - kContextTokens;
            place(context_at(context_begin), context_begin);
            // The newest token is also the d-th continuation of the runs d tokens behind it, so the
            // agreement window keeps closing as more of a repeated span is committed.
            for (std::size_t distance = 1; distance < kAgreementTokens; ++distance) {
                if (last < kContextTokens + distance) { break; }
                note_continuation(last - kContextTokens - distance, distance);
            }
        }
    }

    // Copies up to "out.size()" predicted tokens into "out" and returns how many. A prediction is a
    // continuation of an earlier occurrence of the last kContextTokens committed tokens.
    [[nodiscard]] std::uint32_t predict(std::span<TokenId> out) const {
        const std::size_t size = history_.size();
        if (size < kContextTokens + kCorroboratingTokens || out.empty()) { return 0; }
        const std::size_t here       = size - kContextTokens;
        const std::uint32_t position = find(here);
        if (position == kNoPosition || static_cast<std::size_t>(position) >= here) { return 0; }
        // A run that repeats but whose earlier occurrence is not preceded by the same tokens is a
        // local pattern, not a copy in progress: the match alone says nothing about what comes next
        // (a JSON example in the prompt, a template line that recurs). A copy corroborates itself on
        // both sides of the match, because the tokens the request has already reproduced stand
        // before the matched run exactly as the earlier occurrence's own tokens do.
        std::size_t corroborating = 0;
        while (corroborating < kCorroboratingTokens && position > corroborating && here > corroborating) {
            if (history_[position - 1U - corroborating] != history_[here - 1U - corroborating]) { break; }
            ++corroborating;
        }
        if (corroborating < kCorroboratingTokens) { return 0; }
        std::uint32_t predicted = 0;
        for (; predicted < out.size(); ++predicted) {
            const std::size_t source =
                static_cast<std::size_t>(position) + kContextTokens + predicted;
            if (source >= size) { break; }
            out[predicted] = history_[source];
        }
        return predicted;
    }

private:
    static constexpr std::uint32_t kNoPosition = 0xFFFF'FFFFU;
    static constexpr std::size_t kNoSlot       = static_cast<std::size_t>(-1);
    static constexpr std::size_t kMaximumProbes = 64;
    static constexpr std::size_t kMinimumSlots  = 1024;

    [[nodiscard]] static std::uint64_t hash(std::span<const TokenId> run) noexcept {
        std::uint64_t value = 14695981039346656037ULL;
        for (const TokenId token : run) {
            value ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(token)) + 0x9E3779B97F4A7C15ULL;
            value *= 1099511628211ULL;
            value ^= value >> 29U;
        }
        return value;
    }

    [[nodiscard]] std::span<const TokenId> context_at(std::size_t begin) const noexcept {
        return std::span<const TokenId>(history_.data() + begin, kContextTokens);
    }

    [[nodiscard]] std::size_t find_slot(std::size_t context_begin) const noexcept {
        if (slots_.empty()) { return kNoSlot; }
        const std::size_t mask = slots_.size() - 1U;
        std::size_t slot       = static_cast<std::size_t>(hash(context_at(context_begin))) & mask;
        const std::size_t limit = std::min(kMaximumProbes, slots_.size());
        for (std::size_t probe = 0; probe < limit; ++probe) {
            if (slots_[slot] == kNoPosition) { return kNoSlot; }
            if (conflicted_[slot] == 0 && context_matches(slots_[slot], context_at(context_begin))) {
                return slot;
            }
            slot = (slot + 1U) & mask;
        }
        return kNoSlot;
    }

    [[nodiscard]] std::uint32_t find(std::size_t context_begin) const noexcept {
        const std::size_t slot = find_slot(context_begin);
        return slot == kNoSlot ? kNoPosition : slots_[slot];
    }

    void place(std::span<const TokenId> context, std::size_t context_begin) {
        if (slots_.empty()) { return; }
        const std::size_t mask = slots_.size() - 1U;
        std::size_t slot       = static_cast<std::size_t>(hash(context)) & mask;
        const auto position    = static_cast<std::uint32_t>(context_begin);
        for (;;) {
            if (slots_[slot] == kNoPosition) {
                slots_[slot]      = position;
                previous_[slot]   = kNoPosition;
                conflicted_[slot] = 0;
                return;
            }
            if (context_matches(slots_[slot], context)) {
                if (conflicted_[slot] != 0) { return; }
                if (history_[slots_[slot] + kContextTokens] == history_[context_begin + kContextTokens]) {
                    // The run continues the same way, so the earlier occurrence is a copy to
                    // follow. Keep the newest occurrence current: repeated text tends to recur
                    // nearby, and the newest copy has the most continuation left.
                    previous_[slot] = slots_[slot];
                    slots_[slot]    = position;
                } else {
                    // The same run continues differently: a repeated phrase or template, not a
                    // copy. Predicting the earlier continuation would displace good drafts.
                    conflicted_[slot] = 1;
                }
                return;
            }
            slot = (slot + 1U) & mask;
        }
    }

    // The newest occurrence of the run at "context_begin" just gained its "distance"-th continuation
    // token. If the occurrence before it continued differently, the run is a template, not a copy.
    void note_continuation(std::size_t context_begin, std::size_t distance) {
        const std::size_t slot = find_slot(context_begin);
        if (slot == kNoSlot || conflicted_[slot] != 0 || previous_[slot] == kNoPosition) { return; }
        const std::size_t current  = context_begin + kContextTokens + distance;
        const std::size_t earliest = std::size_t{previous_[slot]} + kContextTokens + distance;
        if (current >= history_.size() || earliest >= history_.size()) { return; }
        if (history_[earliest] != history_[current]) { conflicted_[slot] = 1; }
    }

    // Rebuilds the table at least "minimum_slots" wide, replaying the appends so that the agreement
    // flags are recomputed exactly as they were set incrementally.
    void rebuild(std::size_t minimum_slots) {
        std::size_t target = std::max(kMinimumSlots, slots_.size() * 2U);
        while (target < minimum_slots) { target *= 2U; }
        slots_.assign(target, kNoPosition);
        previous_.assign(target, kNoPosition);
        conflicted_.assign(target, 0);
        for (std::size_t last = kContextTokens; last < history_.size(); ++last) {
            const std::size_t context_begin = last - kContextTokens;
            place(context_at(context_begin), context_begin);
            for (std::size_t distance = 1; distance < kAgreementTokens; ++distance) {
                if (last < kContextTokens + distance) { break; }
                note_continuation(last - kContextTokens - distance, distance);
            }
        }
    }

    [[nodiscard]] bool context_matches(std::uint32_t position,
                                       std::span<const TokenId> context) const noexcept {
        if (position + kContextTokens > history_.size()) { return false; }
        for (std::uint32_t index = 0; index < kContextTokens; ++index) {
            if (history_[position + index] != context[index]) { return false; }
        }
        return true;
    }

    std::vector<TokenId> history_;
    // Open addressing with linear probing: the slot holds the newest occurrence's first offset and
    // equality is settled against the history itself, so a hit cannot be a hash collision.
    std::vector<std::uint32_t> slots_;
    // The occurrence before the newest one: the agreement window compares the newest continuation
    // with this one's.
    std::vector<std::uint32_t> previous_;
    // The run repeated and continued differently, so it is a repeated phrase rather than a copy and
    // must not be predicted from.
    std::vector<std::uint8_t> conflicted_;
};

} // namespace ninfer::targets::qwen3_6::detail
