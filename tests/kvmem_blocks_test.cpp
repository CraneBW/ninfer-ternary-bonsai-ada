// Unit test for the KVMem K0 host decision layer (standalone; build with
// `builder.exe -test tests/kvmem_blocks_test.cpp`).
#include "kvmem/kvmem_blocks.h"
#include "kvmem/kvmem_lexical.h"

#include <algorithm>
#include <cmath>
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

static void test_lexical_history_rescue() {
    constexpr std::uint32_t page_tokens = 64;
    std::vector<std::int32_t> tokens(49 * page_tokens, 7);
    tokens[15 * page_tokens + 24] = 9001;
    tokens[15 * page_tokens + 25] = 9002;
    tokens[15 * page_tokens + 26] = 9003;
    tokens[48 * page_tokens] = 9002;
    tokens[48 * page_tokens + 1] = 9001;
    tokens[48 * page_tokens + 2] = 9003;
    const auto lexical = kvmem_lexical_recall_pages(tokens, 48 * page_tokens,
                                                     48 * page_tokens + 3);
    CHECK(lexical == std::vector<std::uint32_t>({15, 16}),
          "lexical: exact rare-name page and continuation must rank first");
    CHECK(kvmem_lexical_recall_pages(tokens, 48 * page_tokens, 49 * page_tokens) == lexical,
          "lexical: a 64-token query still has the same rare-name evidence");
    auto long_tokens = tokens;
    long_tokens.resize(48 * page_tokens + 129, 7);
    CHECK(kvmem_lexical_recall_pages(long_tokens, 48 * page_tokens,
                                     48 * page_tokens + 128) == lexical,
          "lexical: 128-token query is within the supported boundary");
    CHECK(kvmem_lexical_recall_pages(long_tokens, 48 * page_tokens,
                                     48 * page_tokens + 129).empty(),
          "lexical: queries beyond 128 tokens must not override semantic scores");
    auto common_tokens = tokens;
    std::fill(common_tokens.begin() + 48 * page_tokens, common_tokens.end(), 7);
    CHECK(kvmem_lexical_recall_pages(common_tokens, 48 * page_tokens,
                                     49 * page_tokens).empty(),
          "lexical: common-only query has no historical rare-name evidence");
    const std::vector<std::int32_t> common_query{7, 7, 7};
    CHECK(kvmem_lexical_recall_pages(tokens, 48 * page_tokens,
                                     std::span<const std::int32_t>(common_query)).empty(),
          "lexical: content-only query must ignore rare serialized framing");
    std::vector<std::int32_t> deep_tokens(81 * page_tokens, 7);
    const std::vector<std::int32_t> deep_query{9001, 9002, 9003, 9004,
                                              9011, 9012, 9013, 9014, 9015, 9016};
    std::copy_n(deep_query.begin(), 4, deep_tokens.begin() + 49 * page_tokens);
    for (const auto page : {77u, 78u}) {
        std::copy(deep_query.begin() + 4, deep_query.end(),
                  deep_tokens.begin() + page * page_tokens);
    }
    CHECK(kvmem_lexical_recall_pages(deep_tokens, 80 * page_tokens,
                                     std::span<const std::int32_t>(deep_query)) ==
              std::vector<std::uint32_t>({49, 50}),
          "lexical: repeated historical question words outranked a unique subject");
    // Real213K: rare previous-question boilerplate and a distinct entity name
    // tie in scalar rank. Retain both signatures; only the old identical name
    // is a demotion candidate, never an unrelated tied fact.
    auto tied = std::vector<std::int32_t>(81 * page_tokens, 7);
    const std::vector<std::int32_t> tied_query{9101, 9102, 9201, 9202};
    tied[20 * page_tokens] = tied[40 * page_tokens] = 9101;
    tied[20 * page_tokens + 1] = tied[40 * page_tokens + 1] = 9102;
    tied[78 * page_tokens] = 9201;
    tied[78 * page_tokens + 1] = 9202;
    std::vector<std::uint32_t> tied_older;
    const auto tied_candidates = kvmem_lexical_recall_pages(
        tied, 80 * page_tokens, std::span<const std::int32_t>(tied_query), page_tokens,
        &tied_older, 4);
    CHECK(tied_candidates == std::vector<std::uint32_t>({78, 79, 40, 41}) &&
              tied_older == std::vector<std::uint32_t>({20, 21}),
          "lexical: previous-question tie erased distinct evidence or newer version");
    std::vector<std::int32_t> boundary_tokens(49 * page_tokens, 7);
    boundary_tokens[15 * page_tokens + 62] = 9001;
    boundary_tokens[15 * page_tokens + 63] = 9002;
    boundary_tokens[16 * page_tokens] = 9010; // Value on the following page.
    boundary_tokens[48 * page_tokens] = 9001;
    boundary_tokens[48 * page_tokens + 1] = 9002;
    CHECK(kvmem_lexical_recall_pages(boundary_tokens, 48 * page_tokens,
                                     48 * page_tokens + 2) == lexical,
          "lexical: subject at page end must retain the following value page");
    tokens[15 * page_tokens + 25] = 7;
    tokens[15 * page_tokens + 26] = 7;
    CHECK(kvmem_lexical_recall_pages(tokens, 48 * page_tokens,
                                     48 * page_tokens + 3).empty(),
          "lexical: a single rare token is not enough to rescue a page");

    KvBlockRepository repo = make_chain(49);
    repo.protect_system_prefix(page_tokens);
    repo.protect_action_span(48 * page_tokens, 49 * page_tokens);
    KvSelectConfig cfg;
    cfg.block_tokens = page_tokens;
    cfg.budget_tokens = 8 * page_tokens;
    cfg.sink_tokens = page_tokens;
    cfg.recent_tokens = 2 * page_tokens;
    cfg.action_lrfu = true;
    const auto selected = repo.preview_select(cfg, {}, lexical);
    for (std::uint32_t id : {0u, 15u, 16u, 47u, 48u}) {
        CHECK(std::find(selected.block_ids.begin(), selected.block_ids.end(), id) !=
                  selected.block_ids.end(),
              "lexical: selected window lost required block %u", id);
    }
    CHECK(selected.block_ids.size() == 8, "lexical: candidate increased the GPU budget");
    const std::vector<std::uint32_t> duplicate_invalid{15, 15, 9999, 16};
    CHECK(repo.preview_select(cfg, {}, duplicate_invalid).block_ids == selected.block_ids,
          "lexical: duplicate or invalid page IDs must not alter selection");
    auto updates = boundary_tokens;
    updates[35 * page_tokens + 62] = 9001;
    updates[35 * page_tokens + 63] = 9002;
    std::vector<std::uint32_t> older;
    const auto latest = kvmem_lexical_recall_pages(
        updates, 48 * page_tokens,
        std::span<const std::int32_t>(updates).subspan(48 * page_tokens, 2), page_tokens, &older);
    CHECK(latest == std::vector<std::uint32_t>({35, 36}) &&
              older == std::vector<std::uint32_t>({15, 16}),
          "lexical: equal historical matches did not prefer the latest occurrence");
    repo.add_scores(15, 1000, 1000);
    const auto updated = repo.preview_select(cfg, {}, latest, older);
    CHECK(std::find(updated.block_ids.begin(), updated.block_ids.end(), 15U) ==
              updated.block_ids.end() && updated.block_ids.size() == 8,
          "lexical: old hot version displaced the current fact or shrank the budget");
    const std::array<std::uint32_t, 1> protected_old{15};
    const auto protected_update = repo.preview_select(cfg, protected_old, latest, older);
    CHECK(std::find(protected_update.block_ids.begin(), protected_update.block_ids.end(), 15U) !=
              protected_update.block_ids.end(),
          "lexical: version preference overrode a mandatory page");
    repo.add_scores(15, -1000, -1000);
    cfg.action_lrfu = false;
    CHECK(repo.preview_select(cfg, {}, lexical).block_ids ==
              repo.preview_select(cfg, {}).block_ids,
          "lexical: disabled policy must preserve the baseline selection");
    cfg.action_lrfu = true;

    // When protected prefix + action + recent already fill the budget,
    // lexical recall is optional and must never evict one of those pages.
    repo.protect_system_prefix(2 * page_tokens);
    cfg.budget_tokens = 4 * page_tokens;
    cfg.recent_tokens = page_tokens;
    const std::vector<std::uint32_t> mandatory{2};
    const auto crowded = repo.preview_select(cfg, mandatory, lexical);
    CHECK(crowded.block_ids.size() == 4 &&
              std::find(crowded.block_ids.begin(), crowded.block_ids.end(), 0u) !=
                  crowded.block_ids.end() &&
              std::find(crowded.block_ids.begin(), crowded.block_ids.end(), 1u) !=
                  crowded.block_ids.end() &&
              std::find(crowded.block_ids.begin(), crowded.block_ids.end(), 2u) !=
                  crowded.block_ids.end() &&
              std::find(crowded.block_ids.begin(), crowded.block_ids.end(), 48u) !=
                  crowded.block_ids.end(),
          "lexical: high-priority pages may not be displaced");
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

// 5) Retrieval ranking (K3): a block the query scored high enters the middle
//    quota; the same block is absent when nothing carries a score, i.e. the
//    unranked K2 order is preserved whenever the scorer contributes nothing.
static void test_retrieval_quota() {
    const auto select_with = [](bool ranked) {
        KvBlockRepository repo = make_chain(64);
        KvSelectConfig cfg;
        cfg.budget_tokens = 16 * 64;
        cfg.sink_tokens   = 64;
        cfg.recent_tokens = 64;
        if (ranked) { repo.add_scores(40, 1.0, 0.0); }
        return repo.preview_select(cfg, {});
    };
    const auto kept = [](const KvSelection& sel, std::uint32_t id) {
        return std::find(sel.block_ids.begin(), sel.block_ids.end(), id) != sel.block_ids.end();
    };

    const KvSelection unranked = select_with(false);
    CHECK(unranked.block_ids.size() == 16, "retrieval: expected 16, got %zu",
          unranked.block_ids.size());
    CHECK(!kept(unranked, 40u), "retrieval: unscored blocks must stay out of the window");

    const KvSelection ranked = select_with(true);
    CHECK(ranked.block_ids.size() == 16, "retrieval: expected 16, got %zu",
          ranked.block_ids.size());
    CHECK(kept(ranked, 40u), "retrieval: the top-scoring block must be retained");
    CHECK(std::is_sorted(ranked.block_ids.begin(), ranked.block_ids.end()),
          "retrieval: selection not ascending");
}

static void test_action_lrfu() {
    KvBlockRepository repo = make_chain(32);
    std::vector<float> evidence(32, 0.0f);
    evidence[6] = 0.8f;
    repo.record_lrfu_scores(evidence, 0.5, 1.0);
    evidence[6] = 0.0f;
    evidence[7] = 0.2f;
    repo.record_lrfu_scores(evidence, 0.5, 1.0);
    CHECK(std::fabs(repo.blocks()[6].profile_score -
                    0.8 * std::exp2(-0.5)) < 1e-5,
          "LRFU did not decay the previous retrieval hit");
    CHECK(repo.blocks()[6].retrieval_score == 0.0 &&
              std::fabs(repo.blocks()[7].profile_score - 0.2) < 1e-5,
          "previous-query heat contaminated current retrieval or lost its history profile");
    CHECK(std::fabs(repo.blocks()[7].retrieval_score - 0.2) < 1e-5,
          "LRFU did not accumulate new retrieval evidence");
    const double completed_profile = repo.blocks()[6].profile_score;
    std::vector<float> partial(32, 0.0f);
    repo.set_retrieval_scores(partial);
    CHECK(repo.blocks()[6].profile_score == completed_profile &&
              repo.blocks()[6].retrieval_score == 0.0,
          "partial-query preview advanced or erased completed LRFU history");
    repo.set_retrieval_scores(evidence);

    repo.protect_action_span(10 * 64, 12 * 64);
    KvSelectConfig cfg;
    cfg.budget_tokens = 8 * 64;
    cfg.sink_tokens = 64;
    cfg.recent_tokens = 64;
    cfg.action_lrfu = true;
    const KvSelection first = repo.preview_select(cfg, {});
    CHECK(std::find(first.block_ids.begin(), first.block_ids.end(), 10U) != first.block_ids.end(),
          "current action block 10 was evicted");
    CHECK(std::find(first.block_ids.begin(), first.block_ids.end(), 11U) != first.block_ids.end(),
          "current action block 11 was evicted");
    repo.protect_action_span(20 * 64, 21 * 64);
    CHECK(repo.blocks()[10].stratum == BlockStratum::StandardContext,
          "old action range was not demoted");
    CHECK(repo.blocks()[20].stratum == BlockStratum::ActionProtected,
          "new action range was not protected");

    // Closed thinking is deprioritized only when a complete 64-token page
    // belongs to the reasoning span; the current action always wins overlap.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> thinking{
        {4U * 64U, 9U * 64U + 20U}, {20U * 64U, 24U * 64U}};
    repo.set_closed_thinking_spans(thinking);
    CHECK(repo.blocks()[4].stratum == BlockStratum::EphemeralThinking &&
          repo.blocks()[8].stratum == BlockStratum::EphemeralThinking,
          "complete thinking blocks did not get the cold tier");
    CHECK(repo.blocks()[9].stratum == BlockStratum::StandardContext,
          "partially covered thinking page must retain normal tier");
    CHECK(repo.blocks()[20].stratum == BlockStratum::ActionProtected,
          "reasoning tier incorrectly overrode the latest action");
    // Even an opposed (negative cosine) thinking score should not improve
    // its rank merely because a demotion rule was applied.
    KvBlockRepository cold = make_chain(32);
    for (std::uint32_t id = 0; id < 32; ++id) {
        cold.add_scores(id, -0.8, 0.0);
    }
    cold.add_scores(5, -0.4, 0.0);
    cold.add_scores(6, -0.2, 0.0);
    cold.set_stratum(6, BlockStratum::EphemeralThinking);
    KvSelectConfig one_middle;
    one_middle.budget_tokens = 3 * 64;
    one_middle.sink_tokens = 64;
    one_middle.recent_tokens = 64;
    one_middle.retrieval_blocks = 1;
    one_middle.action_lrfu = true;
    const auto cold_selection = cold.preview_select(one_middle, {});
    CHECK(std::find(cold_selection.block_ids.begin(), cold_selection.block_ids.end(), 5U) !=
              cold_selection.block_ids.end(),
          "ordinary negative-score context should win retrieval quota");
    CHECK(std::find(cold_selection.block_ids.begin(), cold_selection.block_ids.end(), 6U) ==
              cold_selection.block_ids.end(),
          "negative thinking score incorrectly outranks ordinary content");
    repo.set_closed_thinking_spans({});
    CHECK(repo.blocks()[4].stratum == BlockStratum::StandardContext,
          "revised conversation did not clear previous thinking demotion");
    repo.protect_system_prefix(3 * 64 + 1);
    const auto protected_selection = repo.preview_select(cfg, {});
    for (std::uint32_t id = 0; id <= 3; ++id) {
        CHECK(repo.blocks()[id].stratum == BlockStratum::PinnedSink,
              "system/tool declaration page %u lost its tier", id);
        CHECK(std::find(protected_selection.block_ids.begin(),
                        protected_selection.block_ids.end(), id) !=
                  protected_selection.block_ids.end(),
              "system/tool declaration page %u was evicted", id);
    }
    KvSelectConfig too_small = cfg;
    too_small.budget_tokens = 3 * 64;
    bool refused = false;
    try {
        (void)repo.preview_select(too_small, {});
    } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused, "budget below protected system prefix must be rejected");
}

// 256K-class prompt (4096 pages): repeated cosine scores must not turn a
// top-fraction gate into "retain every tied page". This is a pure host test;
// no large model or GPU memory is required.
static void test_action_lrfu_long_ties() {
    constexpr std::uint32_t n = 262144 / 64;
    KvBlockRepository repo = make_chain(n);
    std::vector<float> scores(n, 0.25f);
    repo.record_lrfu_scores(scores, 0.5, 0.1);
    const auto blocks = repo.blocks();
    constexpr std::size_t kept = 410;  // ceil(0.1 * 4096)
    std::size_t hits = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (blocks[i].lrfu_score != 0.0) { ++hits; }
        CHECK((blocks[i].lrfu_score != 0.0) == (i >= n - kept),
              "LRFU tied score priority mismatch on 256K page %zu", i);
    }
    CHECK(hits == kept, "LRFU top-fraction retained %zu rather than %zu", hits, kept);
    KvSelectConfig cfg;
    cfg.action_lrfu = true;
    cfg.budget_tokens = 16384;
    const auto selection = repo.preview_select(cfg, {});
    CHECK(selection.block_ids.size() == 256,
          "256K-class selection retained %zu rather than 256 pages", selection.block_ids.size());
    std::fill(scores.begin(), scores.end(), 0.0f);
    repo.record_lrfu_scores(scores, 0.5, 0.1);
    CHECK(std::abs(repo.blocks().back().lrfu_score - 0.25 * std::exp2(-0.5)) < 1e-6,
          "256K-class LRFU previous hit did not decay");
    CHECK(repo.blocks().front().lrfu_score == 0,
          "256K-class LRFU suppressed evidence reappeared");

    // Simulate consecutive query turns over a full 256K history. A fact far
    // outside the recent band should remain retrievable after the active
    // action span changes, while closed reasoning loses its special priority.
    constexpr std::uint32_t fact = 701;
    constexpr std::uint32_t old_action = 1700;
    constexpr std::uint32_t new_action = 2900;
    repo.protect_system_prefix(128);
    repo.protect_action_span(old_action * 64ULL, (old_action + 2ULL) * 64ULL);
    repo.set_closed_thinking_spans({});
    std::fill(scores.begin(), scores.end(), 0.0f);
    scores[fact] = 1.0f;
    repo.record_lrfu_scores(scores, 0.5, 1.0);
    auto first = repo.preview_select(cfg, {});
    CHECK(std::find(first.block_ids.begin(), first.block_ids.end(), fact) != first.block_ids.end(),
          "256K history fact missing from first query working set");
    CHECK(std::find(first.block_ids.begin(), first.block_ids.end(), old_action) !=
              first.block_ids.end(), "256K old active action missing");
    repo.protect_action_span(new_action * 64ULL, (new_action + 2ULL) * 64ULL);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 1> closed_thinking{{
        {old_action * 64U, (old_action + 2U) * 64U}
    }};
    repo.set_closed_thinking_spans(closed_thinking);
    repo.record_lrfu_scores(scores, 0.5, 1.0);
    const auto second = repo.preview_select(cfg, {});
    CHECK(std::find(second.block_ids.begin(), second.block_ids.end(), fact) != second.block_ids.end(),
          "256K history fact lost after action turnover");
    CHECK(std::find(second.block_ids.begin(), second.block_ids.end(), new_action) !=
              second.block_ids.end(), "256K new active action missing");
    CHECK(repo.blocks()[old_action].stratum == BlockStratum::EphemeralThinking,
          "256K old action was not demoted after closing reasoning span");
    CHECK(repo.blocks()[new_action].stratum == BlockStratum::ActionProtected,
          "256K current action was not protected");
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_lexical_history_rescue();
    std::printf("lexical_history_rescue ok\n");
    test_passthrough();
    std::printf("passthrough ok\n");
    test_over_budget();
    std::printf("over_budget ok\n");
    test_shift_and_drift();
    std::printf("shift ok\n");
    test_mandatory();
    std::printf("mandatory ok\n");
    test_retrieval_quota();
    std::printf("retrieval ok\n");
    test_action_lrfu();
    std::printf("action_lrfu ok\n");
    test_action_lrfu_long_ties();
    std::printf("action_lrfu_256k_ties ok\n");
    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
