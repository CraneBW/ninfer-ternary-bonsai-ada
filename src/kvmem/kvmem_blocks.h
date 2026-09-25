// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// KVMem K0: the pure-host decision layer of the block-sparse KV memory -- block
// metadata, working-set selection, and the window diff/remap plan.
//
// Ported from laamaafung's kvmem_store (llama.cpp fork line v26), reduced to the
// semantics zatfung needs in K1/K2: sink/recent/mandatory bands, retrieval+profile
// quota top-k, and the position-diff remap plan with the fp16-drift double-threshold
// counters. Media-group constraints (vision block affinity) are NOT ported yet --
// zatfung runs vision opt-in and the window planner has no media ranges in K1.
//
// This header is self-contained (no engine dependencies) so the unit test can build
// it standalone; it drops into the engine unchanged.

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::kvmem {

// Where a block's KV payload currently lives. Gpu means "resident in the device
// page pool at window slot baked_pos"; Host means "archived in host memory".
enum class KvTier : std::uint8_t { Gpu, Host };

struct KvBlockMeta {
    std::uint32_t id            = 0;
    std::uint64_t token_begin   = 0;  // absolute position of the first token
    std::uint32_t n_tokens      = 0;  // one 64-token page in K1
    KvTier tier                 = KvTier::Host;
    bool in_working_set         = false;
    // Window slot this block's K phases are currently baked to (-1 = never placed).
    std::int64_t baked_pos      = -1;
    // Selection signals. retrieval: query-vs-block score from the host scorer;
    // profile: window-local cumulative attention heat. attn_score is the merged
    // value the generic fill quota ranks by.
    double attn_score           = 0.0;
    double retrieval_score      = 0.0;
    double profile_score        = 0.0;
    // fp16-drift ledger: how many in-place re-rotations this block survived and
    // the accumulated |position delta| they moved it by. Crossing either
    // threshold forces a rebuild from the raw-K mirror instead of another
    // in-place rotation.
    std::uint64_t remap_count     = 0;
    std::uint64_t remap_abs_delta = 0;
};

struct KvSelectConfig {
    std::uint32_t block_tokens   = 64;
    std::uint32_t budget_tokens  = 0;  // required; multiple of block_tokens
    std::uint32_t sink_tokens    = 0;  // 0 -> auto (1% of budget, clamped 1024..2048)
    std::uint32_t recent_tokens  = 0;  // 0 -> auto (8% of budget, clamped 4096..16384)
    // Quota split for the middle (non-sink/recent) blocks; 0/0 -> 2/3 retrieval,
    // remainder profile. Exactly one of them may be 0 to give the other the rest.
    std::uint32_t retrieval_blocks = 0;
    std::uint32_t profile_blocks   = 0;
    // fp16-drift thresholds (0 = unlimited), mirrored from laamaafung
    // immutable_refresh_*.
    std::uint64_t raw_refresh_remaps      = 0;
    std::uint64_t raw_refresh_abs_delta   = 0;
};

struct KvSelection {
    // Ascending block ids forming the working set, in window order.
    std::vector<std::uint32_t> block_ids;
};

struct KvRemap {
    std::uint32_t block_id = 0;
    std::uint32_t n_tokens = 0;
    std::int64_t from_base = 0;  // baked_pos at planning time (de-rotate source)
    std::int64_t to_base   = 0;  // new window slot (re-rotate target)
    bool skip        = false;    // K phases already correct: no re-rotation needed
    bool raw_refresh = false;    // rebuild from the raw-K mirror instead
};

struct KvWindowPlan {
    std::vector<std::uint32_t> stage_in;   // blocks to upload from host
    std::vector<std::uint32_t> stage_out;  // resident blocks no longer selected
    std::vector<KvRemap> remaps;           // full window, in window order
    std::uint64_t total_window_tokens      = 0;  // query positions start here
    std::uint32_t retained_position_stable = 0;
    std::uint32_t retained_position_moved  = 0;
};

[[nodiscard]] inline std::uint32_t kvmem_ceil_div(std::uint32_t value, std::uint32_t by) {
    return (value + by - 1) / by;
}

[[nodiscard]] inline std::uint32_t kvmem_clamp_u32(std::uint64_t value, std::uint64_t lo,
                                                   std::uint64_t hi) {
    return static_cast<std::uint32_t>(std::min(std::max(value, lo), hi));
}

// Resolves the sink/recent token bands the same way laamaafung's
// resolve_band does: explicit tokens win, otherwise the auto percentage
// clamped to a sane absolute range. Returns block counts.
[[nodiscard]] inline std::pair<std::uint32_t, std::uint32_t> kvmem_resolve_bands(
    const KvSelectConfig& cfg) {
    if (cfg.block_tokens == 0) {
        throw std::invalid_argument("kvmem: block_tokens must be positive");
    }
    if (cfg.budget_tokens < cfg.block_tokens || cfg.budget_tokens % cfg.block_tokens != 0) {
        throw std::invalid_argument("kvmem: budget must be a positive multiple of block_tokens");
    }
    const std::uint32_t budget_blocks = cfg.budget_tokens / cfg.block_tokens;
    const auto auto_tokens            = [&](std::uint64_t percent, std::uint64_t lo,
                                 std::uint64_t hi) {
        const std::uint64_t value = (static_cast<std::uint64_t>(cfg.budget_tokens) * percent +
                                     99) /
                                    100;
        return kvmem_clamp_u32(value, lo, hi);
    };
    const std::uint64_t sink   = cfg.sink_tokens != 0 ? cfg.sink_tokens : auto_tokens(1, 1024, 2048);
    const std::uint64_t recent = cfg.recent_tokens != 0 ? cfg.recent_tokens : auto_tokens(8, 4096, 16384);
    std::uint32_t sink_blocks   = kvmem_ceil_div(static_cast<std::uint32_t>(sink), cfg.block_tokens);
    std::uint32_t recent_blocks = kvmem_ceil_div(static_cast<std::uint32_t>(recent), cfg.block_tokens);
    if (sink_blocks > budget_blocks || recent_blocks > budget_blocks ||
        sink_blocks + recent_blocks > budget_blocks) {
        throw std::invalid_argument("kvmem: sink + recent allocation exceeds the selection budget");
    }
    return {sink_blocks, recent_blocks};
}

// Working-set selection: sink + mandatory + recent bands, then the middle filled
// by the retrieval/profile quota top-k (nth_element), leftover by merged score.
[[nodiscard]] inline KvSelection kvmem_select_blocks(std::span<const KvBlockMeta> blocks,
                                                     const KvSelectConfig& cfg,
                                                     std::span<const std::uint32_t> mandatory) {
    KvSelection out;
    const std::uint32_t n = static_cast<std::uint32_t>(blocks.size());
    if (n == 0) { return out; }
    const std::uint32_t budget = cfg.budget_tokens / cfg.block_tokens;
    if (budget == 0 || n <= budget) {
        // Everything fits: passthrough. Band resolution is skipped on purpose --
        // tiny test/degenerate budgets must not trip the sink+recent sanity checks.
        out.block_ids.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) { out.block_ids[i] = blocks[i].id; }
        return out;
    }
    const auto [sink_blocks, recent_blocks] = kvmem_resolve_bands(cfg);

    std::vector<std::uint8_t> kept(n, 0);
    std::uint32_t kept_count = 0;
    const auto keep          = [&](std::uint32_t i) {
        if (i < n && kept[i] == 0) {
            kept[i] = 1;
            ++kept_count;
        }
    };

    for (std::uint32_t i = 0; i < sink_blocks && kept_count < budget; ++i) { keep(i); }

    // Mandatory: newest first so a huge query suffix cannot evict the tail; the
    // oldest overflow is trimmed (counted, mirroring laamaafung's KVMEM_TRACE).
    std::uint32_t mand_unique = 0;
    {
        std::vector<std::uint8_t> seen(n, 0);
        for (const std::uint32_t id : mandatory) {
            if (id < n && seen[id] == 0) {
                seen[id] = 1;
                ++mand_unique;
            }
        }
    }
    std::uint32_t mand_kept = 0;
    for (std::size_t i = mandatory.size(); i > 0 && kept_count < budget; --i) {
        const std::uint32_t before = kept_count;
        // Map id -> index by position: ids are assigned densely in K0/K1.
        if (mandatory[i - 1] < n) { keep(mandatory[i - 1]); }
        if (kept_count > before) { ++mand_kept; }
    }
    if (mand_kept < mand_unique) {
        std::fprintf(stderr, "KVMEM_TRACE mandatory_trim kept=%u dropped=%u budget=%u\n",
                     mand_kept, mand_unique - mand_kept, budget);
    }

    for (std::uint32_t i = 0; i < recent_blocks && kept_count < budget; ++i) { keep(n - 1 - i); }

    const auto take_top = [&](std::uint32_t quota, auto score_fn) {
        if (kept_count >= budget || quota == 0) { return; }
        std::vector<std::uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (std::uint32_t i = 0; i < n; ++i) {
            if (kept[i] == 0) { candidates.push_back(i); }
        }
        if (candidates.empty()) { return; }
        const std::uint32_t need =
            std::min({quota, budget - kept_count, static_cast<std::uint32_t>(candidates.size())});
        const auto better = [&](std::uint32_t a, std::uint32_t b) {
            const double sa = score_fn(blocks[a]);
            const double sb = score_fn(blocks[b]);
            if (sa != sb) { return sa > sb; }
            return a > b;  // stable: newer block wins ties
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need, candidates.end(),
                             better);
            candidates.resize(need);
        }
        for (const std::uint32_t i : candidates) { keep(i); }
    };

    if (kept_count < budget) {
        std::uint32_t remaining        = budget - kept_count;
        std::uint32_t retrieval_quota  = cfg.retrieval_blocks;
        std::uint32_t profile_quota    = cfg.profile_blocks;
        if (retrieval_quota == 0 && profile_quota == 0) {
            retrieval_quota = (remaining * 2) / 3;
            profile_quota   = remaining - retrieval_quota;
        } else if (retrieval_quota == 0) {
            retrieval_quota = remaining > profile_quota ? remaining - profile_quota : 0;
        } else if (profile_quota == 0) {
            profile_quota = remaining > retrieval_quota ? remaining - retrieval_quota : 0;
        }
        take_top(retrieval_quota,
                 [](const KvBlockMeta& b) { return b.retrieval_score; });
        take_top(profile_quota, [](const KvBlockMeta& b) { return b.profile_score; });
        // Rounding / overlap / zero explicit quotas: fill the rest by merged score.
        if (kept_count < budget) {
            take_top(budget - kept_count, [](const KvBlockMeta& b) { return b.attn_score; });
        }
    }

    out.block_ids.reserve(kept_count);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (kept[i] != 0) { out.block_ids.push_back(blocks[i].id); }
    }
    return out;
}

// The block repository owns the mutable metadata (tier, working-set membership,
// baked_pos, drift ledger). Selection reads; set_selection commits the diff and
// updates the ledger exactly like laamaafung's KvMemStore::set_selection.
class KvBlockRepository {
public:
    // Registers a block; ids must arrive in ascending token order.
    [[nodiscard]] std::uint32_t append(KvBlockMeta meta) {
        if (!blocks_.empty()) {
            if (meta.token_begin < blocks_.back().token_begin + blocks_.back().n_tokens) {
                throw std::invalid_argument("kvmem: blocks must arrive in ascending token order");
            }
            if (meta.id != blocks_.back().id + 1) {
                throw std::invalid_argument("kvmem: block ids must be dense and ascending");
            }
        } else if (meta.id != 0) {
            throw std::invalid_argument("kvmem: the first block must have id 0");
        }
        if (meta.n_tokens == 0) {
            throw std::invalid_argument("kvmem: block n_tokens must be positive");
        }
        blocks_.push_back(meta);
        return meta.id;
    }

    [[nodiscard]] std::span<const KvBlockMeta> blocks() const noexcept { return blocks_; }

    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }

    [[nodiscard]] std::uint64_t total_tokens() const noexcept {
        return blocks_.empty() ? 0 : blocks_.back().token_begin + blocks_.back().n_tokens;
    }

    // Drops tail blocks beyond `count` (the destructive-truncate analogue). Only
    // valid while the dropped tail was never part of a committed working set.
    void truncate(std::uint32_t count) {
        if (count > blocks_.size()) {
            throw std::invalid_argument("kvmem: truncate count exceeds block count");
        }
        blocks_.resize(count);
    }

    void set_tier(std::uint32_t id, KvTier tier) {
        if (id >= blocks_.size()) { throw std::invalid_argument("kvmem: bad block id"); }
        blocks_[id].tier = tier;
    }

    void add_scores(std::uint32_t id, double retrieval, double profile) {
        if (id >= blocks_.size()) { throw std::invalid_argument("kvmem: bad block id"); }
        KvBlockMeta& b   = blocks_[id];
        b.retrieval_score = retrieval;
        b.profile_score   = profile;
        b.attn_score      = retrieval + profile;
    }

    // Preview-only selection (no state change).
    [[nodiscard]] KvSelection preview_select(const KvSelectConfig& cfg,
                                             std::span<const std::uint32_t> mandatory) const {
        return kvmem_select_blocks(blocks_, cfg, mandatory);
    }

    // Commits a selection: computes the stage-in/out diff and the per-block remap
    // plan, then updates tier-independent ledger state. The executor (K1/K2) is
    // responsible for the physical moves; a failed move must leave the repository
    // untouched, which is why the commit is split from the preview.
    [[nodiscard]] KvWindowPlan set_selection(const KvSelection& selection, const KvSelectConfig& cfg,
                                             bool force_raw_refresh = false) {
        const std::uint32_t n = static_cast<std::uint32_t>(blocks_.size());
        std::vector<std::uint8_t> now_selected(n, 0);
        for (const std::uint32_t id : selection.block_ids) {
            if (id >= n) { throw std::invalid_argument("kvmem: selection references unknown id"); }
            now_selected[id] = 1;
        }
        (void)cfg;  // bands are baked into `selection` by the caller; kept for symmetry

        KvWindowPlan plan;
        // Stage-out: resident but no longer selected. Their baked_pos survives so a
        // future re-selection can de-rotate from there instead of raw-K rebuilding.
        for (std::uint32_t i = 0; i < n; ++i) {
            KvBlockMeta& b = blocks_[i];
            if (b.tier == KvTier::Gpu && (now_selected[i] == 0 || force_raw_refresh)) {
                plan.stage_out.push_back(b.id);
                b.in_working_set = false;
            }
        }

        // Pack selected blocks contiguously, ascending (K0 ids are token-ordered).
        std::uint64_t window_pos = 0;
        for (const std::uint32_t id : selection.block_ids) {
            KvBlockMeta& b = blocks_[id];
            const bool was_resident = b.in_working_set && b.tier == KvTier::Gpu;
            if (was_resident) {
                if (b.baked_pos == static_cast<std::int64_t>(window_pos)) {
                    ++plan.retained_position_stable;
                } else {
                    ++plan.retained_position_moved;
                }
            }
            const bool cold = b.tier != KvTier::Gpu || force_raw_refresh;
            if (!b.in_working_set || force_raw_refresh) { plan.stage_in.push_back(b.id); }

            KvRemap rm;
            rm.block_id = b.id;
            rm.n_tokens = b.n_tokens;
            rm.from_base = b.baked_pos;
            rm.to_base   = static_cast<std::int64_t>(window_pos);
            const bool same_position = b.baked_pos == rm.to_base;
            // A cold block has no valid rotated working K on device: its tier record
            // holds V but K authority is the raw mirror, so position equality alone
            // is never enough to skip assembly.
            rm.skip        = same_position && !cold && !force_raw_refresh;
            rm.raw_refresh = false;
            if (!rm.skip) {
                const std::uint64_t delta =
                    b.baked_pos > rm.to_base
                        ? static_cast<std::uint64_t>(b.baked_pos - rm.to_base)
                        : static_cast<std::uint64_t>(rm.to_base - b.baked_pos);
                const bool remap_limit =
                    cfg.raw_refresh_remaps > 0 && b.remap_count >= cfg.raw_refresh_remaps;
                const bool delta_limit = cfg.raw_refresh_abs_delta > 0 &&
                                         (b.remap_abs_delta >= cfg.raw_refresh_abs_delta ||
                                          delta >= cfg.raw_refresh_abs_delta -
                                                       std::min(b.remap_abs_delta,
                                                                cfg.raw_refresh_abs_delta));
                rm.raw_refresh = cold || remap_limit || delta_limit;
                if (rm.raw_refresh) {
                    b.remap_count     = 0;
                    b.remap_abs_delta = 0;
                } else {
                    ++b.remap_count;
                    b.remap_abs_delta += delta;
                }
            }
            plan.remaps.push_back(rm);

            b.in_working_set = true;
            b.baked_pos      = rm.to_base;
            window_pos += b.n_tokens;
        }
        plan.total_window_tokens = window_pos;
        return plan;
    }

private:
    std::vector<KvBlockMeta> blocks_;
};

} // namespace ninfer::kvmem
