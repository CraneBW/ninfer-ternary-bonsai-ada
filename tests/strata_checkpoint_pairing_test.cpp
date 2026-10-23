// Strata 9de04cd snapshot/restore pairing adapted to NInfer's real
// Main/MTP/StateImage q0 protocol. Pure CPU: exercise a rejected Main/MTP
// mismatch before any writer mutates, partial terminal, cancellation and
// failure-after-partial-rewind retirement. No existing GPU fixture is rerun.
#include "kvmem/strata_checkpoint_pairing.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::kvmem;

void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "Strata checkpoint pairing: %s\n", message);
        std::exit(1);
    }
}

template<class Action>
void rejected(Action&& action, const char* message) {
    bool threw = false;
    try { action(); }
    catch (const std::logic_error&) { threw = true; }
    require(threw, message);
}

struct NativeCheckpoint {
    std::uint32_t logical_frontier;
    std::uint64_t lineage_epoch;
    std::vector<std::uint64_t> prefix_revisions;
};

void check_original_pair() {
    // q0 crosses a page boundary, preserving one 1-column tail. The actual
    // native checkpoints use exactly these logical_frontier/epoch/revisions
    // members, and the physical page count is ceil(q0 / 64).
    std::vector<TokenId> frozen(65);
    for (std::size_t i = 0; i < frozen.size(); ++i) { frozen[i] = 100 + static_cast<TokenId>(i); }
    std::vector<TokenId> live = frozen;
    live.insert(live.end(), {7, 8, 9}); // probe/future tokens cannot enter q0.
    NativeCheckpoint main{65, 9, {101, 103}};
    NativeCheckpoint mtp{65, 11, {31, 33}};
    StrataStateImageBinding state{
        .prefix_frontier = 65,
        .image_bytes = 160,
        .expected_image_bytes = 160,
        .host_copy_complete = true,
        .exact_prefix_identity = true,
        .exclusive_state_source = true,
    };
    std::string order;
    int main_slot = 99, mtp_slot = 99;
    std::array<int, 4> complete_state{1, 2, 3, 4}; // GDN, conv, hidden, tail
    bool poisoned = false;

    auto good = [&](bool bad_mtp_validate, bool bad_mtp_restore, bool bad_state_restore,
                    bool wrong_window) {
        return restore_strata_original_pair(
            frozen, live, main, mtp, state, 64,
            [&] { order += "M"; },
            [&] {
                order += "D";
                if (bad_mtp_validate) { throw std::logic_error("stale MTP first-write revision"); }
            },
            [&] { order += "S"; },
            [&]() -> std::uint32_t { order += "m"; main_slot = 64; return 64; },
            [&]() -> std::uint32_t {
                order += "d";
                if (bad_mtp_restore) { throw std::runtime_error("MTP transfer failed"); }
                mtp_slot = wrong_window ? 63 : 64;
                return static_cast<std::uint32_t>(mtp_slot);
            },
            [&] {
                order += "s";
                if (bad_state_restore) { throw std::runtime_error("Host StateImage copy failed"); }
                complete_state = {10, 20, 30, 40};
            },
            [&]() noexcept { order += "P"; poisoned = true; });
    };

    require(good(false, false, false, false) == 64, "paired restore window");
    require(order == "MDSmds" && main_slot == mtp_slot && !poisoned &&
            complete_state == std::array<int, 4>{10, 20, 30, 40},
            "whole q0 state and Main/MTP published after all preflights");

    // Backend owner rejection MUST occur before Main has truncated even its
    // future page. Validating Main alone cannot authorize any mutation.
    order.clear(); poisoned = false; main_slot = mtp_slot = 99;
    complete_state = {1, 2, 3, 4};
    rejected([&] { (void)good(true, false, false, false); },
             "stale draft preflight accepted");
    require(order == "MD" && main_slot == 99 && mtp_slot == 99 && !poisoned &&
            complete_state == std::array<int, 4>{1, 2, 3, 4},
            "bad draft mutated Main/StateImage before preflight");

    auto check_metadata_reject = [&](const char* cause) {
        order.clear(); poisoned = false; main_slot = mtp_slot = 99;
        rejected([&] { (void)good(false, false, false, false); }, cause);
        require(order.empty() && main_slot == 99 && mtp_slot == 99 && !poisoned,
                "bad pairing changed owners or invoked a validator");
    };
    state.image_bytes -= 1;
    check_metadata_reject("incomplete StateImage payload admitted");
    state.image_bytes += 1;
    state.exact_prefix_identity = false;
    check_metadata_reject("unrelated StateImage accepted");
    state.exact_prefix_identity = true;
    mtp.prefix_revisions.pop_back();
    check_metadata_reject("missing draft partial-tail revision admitted");
    mtp.prefix_revisions.push_back(33);
    mtp.lineage_epoch = 0;
    check_metadata_reject("stale MTP lineage epoch admitted");
    mtp.lineage_epoch = 11;
    live[5]++;
    check_metadata_reject("changed live token prefix accepted");
    live[5]--;
    state.exclusive_state_source = false;
    check_metadata_reject("unsettled StateImage fork admitted");
    state.exclusive_state_source = true;

    order.clear(); poisoned = false; main_slot = mtp_slot = 99;
    rejected([&] { (void)good(false, false, false, true); },
             "mismatched Main/MTP restored window admitted");
    require(order == "MDSmdP" && poisoned &&
            main_slot == 64 && mtp_slot == 63,
            "mismatched restored windows did not poison branch before state mutation");

    order.clear(); poisoned = false; main_slot = mtp_slot = 99;
    rejected([&] {
        try { (void)good(false, true, false, false); }
        catch (const std::runtime_error&) { throw std::logic_error("MTP write failed"); }
    }, "failed draft restore not propagated");
    require(order == "MDSmdP" && poisoned && main_slot == 64 && mtp_slot == 99,
            "partial physical mutation did not revoke whole checkpoint branch");

    order.clear(); poisoned = false; main_slot = mtp_slot = 99;
    rejected([&] {
        try { (void)good(false, false, true, false); }
        catch (const std::runtime_error&) { throw std::logic_error("state write failed"); }
    }, "failed state restore not propagated");
    require(order == "MDSmdsP" && poisoned &&
            main_slot == 64 && mtp_slot == 64,
            "StateImage failure did not revoke Main/MTP restored branch");
}

void check_mixed_speculative_accept() {
    // Three concurrent MTP lanes share one GPU/GDN record width, but one row
    // keeps its full window, another stops after two accepted tokens, and the
    // third is cancelled without publishing any speculative StateImage.
    std::array<StrataSpeculativePending, 3> pending{{
        { .lane = 1, .base_execution = 320, .base_ledger = 1000,
          .main_valid = 320, .mtp_valid = 320, .produced = 5,
          .accepted = 5, .record_width = 5, .terminal = false,
          .prefix_execution_split_after = std::nullopt },
        { .lane = 2, .base_execution = 512, .base_ledger = 700,
          .main_valid = 512, .mtp_valid = 512, .produced = 5,
          .accepted = 2, .record_width = 5, .terminal = true,
          .prefix_execution_split_after = 1 },
        { .lane = 3, .base_execution = 1024, .base_ledger = 2000,
          .main_valid = 1024, .mtp_valid = 1024, .produced = 5,
          .accepted = 0, .record_width = 5, .terminal = true,
          .cancelled = true, .prefix_execution_split_after = std::nullopt },
    }};
    std::array<StrataSpeculativeCommit, 3> plan{};
    require(plan_strata_mixed_accept(pending, plan) == pending.size(),
            "every speculative row has an independent plan");
    require(plan[0].main_frontier == 325 && plan[0].mtp_frontier == 325 &&
            plan[0].ledger_frontier == 1005 && plan[0].state_commit_columns == 5 &&
            !plan[0].repair_hidden && !plan[0].retire,
            "full accepted continuation lost Main/MTP/GDN alignment");
    require(plan[1].main_frontier == 514 && plan[1].mtp_frontier == 514 &&
            plan[1].ledger_frontier == 702 && plan[1].state_commit_columns == 2 &&
            plan[1].hidden_select_index == 1 && plan[1].repair_hidden &&
            plan[1].retire,
            "partially terminal row leaked rejected draft state");
    require(plan[2].main_frontier == 1024 && plan[2].mtp_frontier == 1024 &&
            plan[2].ledger_frontier == 2000 && plan[2].state_commit_columns == 0 &&
            plan[2].cancelled && plan[2].retire,
            "cancelled row published provisional StateImage");

    // Produce a real owned batch plan before mutating any lane: if any row is
    // invalid, even a previously accepted first row must stay untouched.
    std::array<std::uint32_t, 3> main{320, 512, 1024};
    std::array<std::uint32_t, 3> mtp{320, 512, 1024};
    std::array<std::uint32_t, 3> state_fold{};
    auto apply = [&] {
        std::array<StrataSpeculativeCommit, 3> committed{};
        const auto count = plan_strata_mixed_accept(pending, committed);
        for (std::size_t i = 0; i < count; ++i) {
            main[i] = committed[i].main_frontier;
            mtp[i] = committed[i].mtp_frontier;
            state_fold[i] = committed[i].state_commit_columns;
        }
    };
    pending[1].record_width = 4;
    std::array<StrataSpeculativeCommit, 3> untouched{};
    untouched[0].lane = 99;
    untouched[0].state_commit_columns = 77;
    rejected([&] { (void)plan_strata_mixed_accept(pending, untouched); },
             "mixed-width validation wrote an output row");
    require(untouched[0].lane == 99 && untouched[0].state_commit_columns == 77 &&
            untouched[1].lane == 0,
            "a rejected mixed batch overwrote a prior output record");
    rejected([&] { apply(); }, "mixed GDN fold widths admitted");
    require(main == std::array<std::uint32_t, 3>{320, 512, 1024} &&
            mtp == main && state_fold == std::array<std::uint32_t, 3>{0, 0, 0},
            "invalid second row mutated first row's Main/MTP/StateImage");
    pending[1].record_width = 5;
    pending[1].terminal = false;
    rejected([&] { apply(); }, "short nonterminal prefix was accepted");
    pending[1].terminal = true;
    pending[2].accepted = 1;
    rejected([&] { apply(); }, "cancelled row accepted speculative token");
    pending[2].accepted = 0;
    pending[2].lane = 1;
    rejected([&] { apply(); }, "duplicate owner lane admitted");
    pending[2].lane = 3;
    pending[1].main_valid = 513;
    rejected([&] { apply(); }, "mismatched Main/MTP base accepted");
    pending[1].main_valid = 512;
    pending[1].prefix_execution_split_after = 3;
    rejected([&] { apply(); }, "execution split beyond accepted prefix admitted");
    pending[1].prefix_execution_split_after = 1;
    pending[0].base_execution = std::numeric_limits<std::uint32_t>::max() - 1;
    pending[0].main_valid = pending[0].mtp_valid = pending[0].base_execution;
    rejected([&] { apply(); }, "committed frontier overflow admitted");
    pending[0].base_execution = pending[0].main_valid = pending[0].mtp_valid = 320;
    apply();
    require(main == std::array<std::uint32_t, 3>{325, 514, 1024} &&
            mtp == main && state_fold == std::array<std::uint32_t, 3>{5, 2, 0},
            "valid mixed commit failed to publish the matched accepted prefix");
}

} // namespace

int main() {
    check_original_pair();
    check_mixed_speculative_accept();
    std::puts("STRATA_CHECKPOINT_PAIRING CPU PASS");
}
