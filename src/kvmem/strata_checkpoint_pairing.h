#pragma once

// The validation-before-mutation order follows Strata's MIT-licensed
// conversation_snapshot_validate -> conversation_snapshot_restore contract:
// Niko1221/Strata @ 9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4,
// src/core/conversation_state.cpp:382-410. License: third_party/strata/LICENSE.
//
// NInfer owns separate Main, MTP and StateImage stores. Their original-position
// q0 snapshots form ONE recovery boundary, not three independently publishable
// checkpoints. The existing store still validates actual page/Host/Cold owners,
// first-write revisions and epochs; this CPU-only adapter checks cross-store
// shape/identity, stages every preflight before the first destructive operation
// and requires the caller to poison a partially restored branch.

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::kvmem {

// Host StateImage includes the GDN recurrent/conv state and the continuation
// hidden and partial statistics described by the target's real image layout.
// The Program, not this adapter, owns and checks the StateImage replica.
struct StrataStateImageBinding {
    std::uint32_t prefix_frontier = 0;
    std::size_t image_bytes = 0;
    std::size_t expected_image_bytes = 0;
    bool host_copy_complete = false;
    bool exact_prefix_identity = false;
    bool exclusive_state_source = false;
};

// Checkpoints here are real KVAddressSpaceStore::OriginalPrefixCheckpoint
// instances: address, logical_frontier, lineage_epoch, prefix_revisions.
// Their physical owner validation must still call store.validate_original_prefix.
template <class MainCheckpoint, class MtpCheckpoint>
void validate_strata_original_pair(
    std::span<const TokenId> checkpoint_prefix, std::span<const TokenId> live_prompt,
    const MainCheckpoint& main, const MtpCheckpoint& mtp,
    const StrataStateImageBinding& state, std::uint32_t page_tokens) {
    if (page_tokens == 0 || checkpoint_prefix.empty() ||
        checkpoint_prefix.size() > std::numeric_limits<std::uint32_t>::max() ||
        live_prompt.size() < checkpoint_prefix.size() ||
        !std::equal(checkpoint_prefix.begin(), checkpoint_prefix.end(), live_prompt.begin())) {
        throw std::logic_error("Strata pairing: checkpoint is not an exact live token prefix");
    }
    const auto q0 = static_cast<std::uint32_t>(checkpoint_prefix.size());
    const auto pages = (static_cast<std::uint64_t>(q0) + page_tokens - 1) / page_tokens;
    if (main.logical_frontier != q0 || mtp.logical_frontier != q0 ||
        main.lineage_epoch == 0 || mtp.lineage_epoch == 0 ||
        main.prefix_revisions.size() != pages || mtp.prefix_revisions.size() != pages) {
        throw std::logic_error("Strata pairing: Main/MTP original KV prefix differs");
    }
    if (state.prefix_frontier != q0 || state.image_bytes == 0 ||
        state.image_bytes != state.expected_image_bytes ||
        !state.host_copy_complete || !state.exact_prefix_identity ||
        !state.exclusive_state_source) {
        throw std::logic_error("Strata pairing: incomplete q0 StateImage or prefix identity");
    }
}

// The callbacks named validate_* must not mutate physical or logical owners.
// The Program's existing query-prefix checkpoint and the two tier checkpoints
// stay authoritative; this function is only the validation/publication seam.
// If a CUDA transfer or destructive truncate fails *after* preflight, the
// caller's noexcept poison_branch must block all further original-position
// use and drain any outstanding GPU readers before releasing that branch.
template <class MainCheckpoint, class MtpCheckpoint,
          class ValidateMain, class ValidateMtp, class ValidateState,
          class RewindMain, class RewindMtp, class RestoreState, class PoisonBranch>
[[nodiscard]] std::uint32_t restore_strata_original_pair(
    std::span<const TokenId> checkpoint_prefix, std::span<const TokenId> live_prompt,
    const MainCheckpoint& main, const MtpCheckpoint& mtp,
    const StrataStateImageBinding& state, std::uint32_t page_tokens,
    ValidateMain&& validate_main, ValidateMtp&& validate_mtp,
    ValidateState&& validate_state,
    RewindMain&& rewind_main, RewindMtp&& rewind_mtp,
    RestoreState&& restore_state, PoisonBranch&& poison_branch) {
    static_assert(std::is_nothrow_invocable_v<PoisonBranch&>,
                  "a failed paired rewind requires a noexcept branch poison/drain");
    validate_strata_original_pair(checkpoint_prefix, live_prompt, main, mtp, state, page_tokens);
    // The preflight is complete BEFORE Main's first mutation. In particular,
    // a stale MTP page, changed StateImage identity, or lost Host source must
    // reject even when Main alone is a valid checkpoint.
    std::invoke(validate_main);
    std::invoke(validate_mtp);
    std::invoke(validate_state);
    try {
        const auto m = static_cast<std::uint32_t>(std::invoke(rewind_main));
        const auto d = static_cast<std::uint32_t>(std::invoke(rewind_mtp));
        if (m == 0 || d != m) {
            throw std::logic_error("Strata pairing: Main/MTP restored different physical windows");
        }
        std::invoke(restore_state);
        return m;
    } catch (...) {
        std::invoke(poison_branch);
        throw;
    }
}

// Original query recovery and speculative acceptance must share the same
// executed token/frontier identity. Plan ALL batch rows before any row's
// StateImage replay fold or Main/MTP commit is enqueued. One row may accept
// the full window while another terminates after only its first verified
// token; a cancelled row publishes no token or StateImage.
struct StrataSpeculativePending {
    std::uint32_t lane = 0;
    std::uint32_t base_execution = 0;
    std::uint32_t base_ledger = 0;
    std::uint32_t main_valid = 0;
    std::uint32_t mtp_valid = 0;
    std::uint32_t produced = 0;
    std::uint32_t accepted = 0;
    std::uint32_t record_width = 0;
    bool terminal = false;
    bool cancelled = false;
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct StrataSpeculativeCommit {
    std::uint32_t lane = 0;
    std::uint32_t main_frontier = 0;
    std::uint32_t mtp_frontier = 0;
    std::uint32_t ledger_frontier = 0;
    std::uint32_t state_commit_columns = 0;
    std::uint32_t hidden_select_index = 0;
    bool repair_hidden = false;
    bool retire = false;
    bool cancelled = false;
};

// Caller owns the fixed per-lane array. Validate *all* rows before writing
// even one result, so one bad terminal row cannot partially publish a plan.
// No per-decode heap allocation, including mixed terminal/cancel batches.
[[nodiscard]] inline std::size_t plan_strata_mixed_accept(
    std::span<const StrataSpeculativePending> rows,
    std::span<StrataSpeculativeCommit> output) {
    if (rows.empty() || rows.size() > kMaximumConcurrency || output.size() < rows.size()) {
        throw std::invalid_argument("Strata pairing: invalid speculative batch capacity");
    }
    const auto width = rows.front().record_width;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (r.record_width == 0 || r.record_width != width ||
            r.produced == 0 || r.produced > width ||
            r.main_valid != r.base_execution || r.mtp_valid != r.base_execution ||
            static_cast<std::uint64_t>(r.base_execution) + r.accepted >
                std::numeric_limits<std::uint32_t>::max() ||
            static_cast<std::uint64_t>(r.base_ledger) + r.accepted >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("Strata pairing: speculative Main/MTP/state base diverged");
        }
        if ((r.cancelled && (!r.terminal || r.accepted != 0 ||
                             r.prefix_execution_split_after)) ||
            (!r.cancelled && (r.accepted == 0 || r.accepted > r.produced ||
                              (!r.terminal && r.accepted != r.produced))) ||
            (r.prefix_execution_split_after &&
             (*r.prefix_execution_split_after == 0 ||
              *r.prefix_execution_split_after > r.accepted))) {
            throw std::logic_error("Strata pairing: unlicensed partial speculative prefix");
        }
        if (std::any_of(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(i),
                        [&](const auto& previous) { return previous.lane == r.lane; })) {
            throw std::logic_error("Strata pairing: duplicate speculative lane");
        }
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const bool partial_terminal = !r.cancelled && r.terminal && r.accepted < r.produced;
        const auto frontier = r.base_execution + r.accepted;
        output[i] = {
            .lane = r.lane,
            .main_frontier = frontier,
            .mtp_frontier = frontier,
            .ledger_frontier = r.base_ledger + r.accepted,
            .state_commit_columns = r.cancelled ? 0U : r.accepted,
            .hidden_select_index = partial_terminal ? r.accepted - 1 : r.produced - 1,
            .repair_hidden = partial_terminal,
            .retire = r.terminal,
            .cancelled = r.cancelled,
        };
    }
    return rows.size();
}

} // namespace ninfer::kvmem
