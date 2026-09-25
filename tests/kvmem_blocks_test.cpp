// Unit test for the KVMem K0 host decision layer (standalone; build with
// `builder.exe -test tests/kvmem_blocks_test.cpp`).
#include "kvmem/kvmem_blocks.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace ninfer::kvmem;

static int failures = 0;
#define CHECK(cond, ...)                                      \
    do {                                                      \
        if (!(cond)) {                                        \
            ++failures;                                       \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);   \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

static KvBlockRepository make_chain(std::uint32_t blocks, std::uint32_t block_tokens = 64) {
    KvBlockRepository repo;
    for (std::uint32_t i = 0; i < blocks; ++i) {
        KvBlockMeta m;
        m.id          = i;
        m.token_begin = static_cast<std::uint64_t>(i) * block_tokens;
        m.n_tokens    = block_tokens;
        repo.append(m);
    }
    return repo;
}

// 1) Everything fits: passthrough selection; first placement is cold/raw.
static void test_passthrough() {
    KvBlockRepository repo = make_chain(8);
    KvSelectConfig cfg;
    cfg.budget_tokens = 8 * 64;
    const KvSelection sel = repo.preview_select(cfg, {});
    CHECK(sel.block_ids.size() == 8, "passthrough: expected all 8 blocks, got %zu",
          sel.block_ids.size());
    CHECK(std::is_sorted(sel.block_ids.begin(), sel.block_ids.end()),
          "passthrough: selection not ascending");

    KvWindowPlan plan = repo.set_selection(sel, cfg);
    CHECK(plan.total_window_tokens == 8 * 64, "passthrough: window tokens %llu",
          (unsigned long long)plan.total_window_tokens);
    CHECK(plan.stage_in.size() == 8, "passthrough: all blocks stage in (%zu)",
          plan.stage_in.size());
    CHECK(plan.stage_out.empty(), "passthrough: no stage out");
    bool ok = true;
    std::uint64_t pos = 0;
    for (const KvRemap& rm : plan.remaps) {
        ok = ok && !rm.skip;             // cold blocks never skip on first placement
        ok = ok && rm.raw_refresh;       // cold -> rebuild from raw-K
        ok = ok && rm.to_base == static_cast<std::int64_t>(pos);
        pos += rm.n_tokens;
    }
    CHECK(ok, "passthrough: first placement must be cold/raw at packed slots");
}

// 2) Over budget: bands survive, quota top-k fills the middle, ascending order;
//    an identical reselection is fully position-stable (all remaps skip).
static void test_over_budget() {
    KvBlockRepository repo = make_chain(64);
    KvSelectConfig cfg;
    cfg.budget_tokens = 16 * 64;
    cfg.sink_tokens   = 64;   // small budgets need explicit bands (auto clamps at 1024)
    cfg.recent_tokens = 64;
    for (std::uint32_t i = 0; i < 64; ++i) {
        repo.add_scores(i, static_cast<double>(i), static_cast<double>(63 - i));
    }
    const KvSelection sel = repo.preview_select(cfg, {});
    CHECK(sel.block_ids.size() == 16, "over-budget: expected 16, got %zu", sel.block_ids.size());
    CHECK(std::is_sorted(sel.block_ids.begin(), sel.block_ids.end()),
          "over-budget: selection not ascending");
    CHECK(std::find(sel.block_ids.begin(), sel.block_ids.end(), 0u) != sel.block_ids.end(),
          "over-budget: sink block 0 missing");
    CHECK(std::find(sel.block_ids.begin(), sel.block_ids.end(), 63u) != sel.block_ids.end(),
          "over-budget: recent block 63 missing");

    repo.set_selection(sel, cfg);
    // Executor contract: only the staged blocks become GPU-resident.
    for (std::uint32_t id : sel.block_ids) { repo.set_tier(id, KvTier::Gpu); }
    KvWindowPlan plan = repo.set_selection(sel, cfg);
    CHECK(plan.stage_in.empty(), "reselect-all: nothing to stage in (%zu)", plan.stage_in.size());
    CHECK(plan.stage_out.empty(), "reselect-all: nothing to stage out");
    CHECK(plan.retained_position_stable == 16, "reselect-all: stable=%u",
          plan.retained_position_stable);
    CHECK(plan.retained_position_moved == 0, "reselect-all: moved=%u",
          plan.retained_position_moved);
    for (const KvRemap& rm : plan.remaps) { CHECK(rm.skip, "reselect-all: remap must skip"); }
}

// 3) Shifting window: front eviction moves every block; the drift counters
//    advance per move and the raw-refresh threshold fires visibly (round 2).
static void test_shift_and_drift() {
    KvBlockRepository repo = make_chain(32);
    KvSelectConfig cfg;
    cfg.budget_tokens      = 16 * 64;
    cfg.sink_tokens        = 64;
    cfg.recent_tokens      = 64;
    cfg.raw_refresh_remaps = 2;  // a block's 3rd in-place move rebuilds from raw-K

    // Explicit initial window 0..15 (the quota preview would fill ties newest-first).
    KvSelection initial;
    for (std::uint32_t i = 0; i < 16; ++i) { initial.block_ids.push_back(i); }
    repo.set_selection(initial, cfg);
    for (std::uint32_t id : initial.block_ids) { repo.set_tier(id, KvTier::Gpu); }

    std::uint32_t prev_raw = 0;
    for (int round = 0; round < 5; ++round) {
        KvBlockMeta m;
        m.id          = 32 + round;
        m.token_begin = (32 + round) * 64ull;
        m.n_tokens    = 64;
        repo.append(m);
        repo.set_tier(m.id, KvTier::Host);

        KvSelection sel;
        for (std::uint32_t i = static_cast<std::uint32_t>(round) + 1;
             i <= static_cast<std::uint32_t>(round) + 16; ++i) {
            sel.block_ids.push_back(i);
        }
        KvWindowPlan plan = repo.set_selection(sel, cfg);
        CHECK(plan.total_window_tokens == 16 * 64, "shift r%d: window=%llu", round,
              (unsigned long long)plan.total_window_tokens);
        CHECK(plan.stage_out.size() == 1, "shift r%d: one block out (%zu)", round,
              plan.stage_out.size());
        CHECK(plan.stage_in.size() == 1, "shift r%d: one block in (%zu)", round,
              plan.stage_in.size());
        CHECK(plan.remaps.front().to_base == 0, "shift r%d: window starts at slot 0", round);
        // Executor contract: a staged-out block's tier drops to Host once its payload
        // is archived. set_selection plans; the executor acts and reports back here.
        for (std::uint32_t id : plan.stage_out) { repo.set_tier(id, KvTier::Host); }

        std::uint32_t raw = 0;
        for (const KvRemap& rm : plan.remaps) { raw += rm.raw_refresh ? 1u : 0u; }
        CHECK(raw >= 1, "shift r%d: the staged-in block always rebuilds", round);
        if (round == 2) {
            // Blocks that moved twice without a rebuild now cross remap_count=2.
            CHECK(raw > prev_raw, "shift r2: drift threshold must fire (%u)", raw);
        }
        prev_raw = raw;
    }
}

// 4) Mandatory blocks survive within budget.
static void test_mandatory() {
    KvBlockRepository repo = make_chain(48);
    KvSelectConfig cfg;
    cfg.budget_tokens = 8 * 64;
    cfg.sink_tokens   = 64;
    cfg.recent_tokens = 64;
    const std::vector<std::uint32_t> mandatory{10, 11, 12};
    const KvSelection sel = repo.preview_select(cfg, mandatory);
    CHECK(sel.block_ids.size() == 8, "mandatory: expected 8, got %zu", sel.block_ids.size());
    for (std::uint32_t id : mandatory) {
        CHECK(std::find(sel.block_ids.begin(), sel.block_ids.end(), id) != sel.block_ids.end(),
              "mandatory: block %u missing", id);
    }
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_passthrough();
    std::printf("passthrough ok\n");
    test_over_budget();
    std::printf("over_budget ok\n");
    test_shift_and_drift();
    std::printf("shift ok\n");
    test_mandatory();
    std::printf("mandatory ok\n");
    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
