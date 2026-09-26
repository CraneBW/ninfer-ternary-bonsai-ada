// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// Store-level generation-reclaim test: a real page pool + KVAddressSpaceStore with a real
// generation headroom, driving the decode-shaped lifecycle (activate -> ensure -> commit ->
// compact -> grow again) rather than the completion-time one.
//
// It pins down three things the decode path relies on:
//   1. kvmem_compact_generation reclaims down to `pool - generation headroom` and hands the
//      evicted pages back to the pool, so a window that has grown to the whole pool keeps
//      decoding instead of running it dry.
//   2. The rewire keeps content with its block: every window slot's payload follows the ledger.
//      The V plane carries a per-block marker (V is never re-phased), which is what lets the
//      test name the evicted blocks without reading the private ledger.
//   3. A stage-in on a FULL pool succeeds. The evicted pages are returned before the stage-in
//      reservation is taken; taking the reservation first would find nothing free and fail the
//      compaction that generation depends on.
//
// Run: builder.exe -test tests/kvmem_generation_test.cu

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_rerope.cuh"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::targets::qwen3_6::detail;

namespace {

// The pool is deliberately large enough that the generation window lands on the automatic
// sink/recent band floor (1024 + 4096 tokens): a budget of 4K..5K tokens has to fall back to
// proportional bands instead of refusing to plan, which is exactly the range a pool of this
// size produces.
constexpr int kPool   = 96;  // physical pool capacity (page groups)
constexpr int kHeads  = 4;   // the re-phase kernel takes 2 or 4 KV heads
constexpr int kLayers = 1;
constexpr int kTokens = kPagedKVPageSize;

constexpr std::uint32_t kBudgetPages  = 72;  // --kvmem-budget, in pages
constexpr std::uint32_t kReservePages = 24;  // --kvmem-gen-reserve, in pages
constexpr int kWindowPlane = 1;              // V codes: layer 0 -> plane 1

int failures = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++failures;                                     \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

std::int64_t elems_per_page(int leading) {
    return static_cast<std::int64_t>(leading) * kTokens * kHeads;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        KVPageGeometry geometry;
        geometry.page_tokens        = kTokens;
        geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < kLayers; ++layer) {
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});    // K codes
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});    // V codes
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});    // K scales
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});    // V scales
        }

        LayoutBuilder builder;
        const DeviceKVPagePoolLayout pool_layout =
            plan_device_kv_page_pool(builder, DeviceKVPagePoolSpec{.page_group_count = kPool,
                                                                   .geometry = geometry});
        const KVExecutionTableLayout tables_layout =
            plan_kv_execution_tables(builder, KVExecutionTableSpec{.logical_page_capacity = kPool,
                                                                   .table_rows = 1});
        const std::size_t backing_bytes =
            pool_layout.payload_bytes() + tables_layout.metadata_bytes() + (1u << 20);
        void* backing = nullptr;
        CHECK(cudaMalloc(&backing, backing_bytes) == cudaSuccess, "cudaMalloc backing");

        DeviceKVPagePool pool(DeviceSpan{backing, backing_bytes}, pool_layout);
        KVExecutionTablePool tables(DeviceSpan{backing, backing_bytes}, tables_layout, pool);
        LogicalKVPageStore pages(pool, kPool);
        // The store takes its window and headroom explicitly: the front ends resolve them, so a
        // direct construction has to ask for them itself.
        KVAddressSpaceStore store(
            pages, tables, 2, kPool,
            KvMemOptions{.enabled = true,
                         .budget_tokens = kBudgetPages * kTokens,
                         .gen_reserve_tokens = kReservePages * kTokens});
        CHECK(store.kvmem_enabled(), "KVMem did not enable itself");
        CHECK(store.kvmem_gen_reserve_tokens() == kReservePages * kTokens,
              "generation headroom resolved to %u (want %u)", store.kvmem_gen_reserve_tokens(),
              kReservePages * kTokens);
        // The headroom is clamped to the room the pool has beyond the window, never past the
        // minimum compaction window, and an explicit zero turns it off.
        {
            KVAddressSpaceStore narrow(pages, tables, 2, kPool,
                                       KvMemOptions{.enabled = true,
                                                    .budget_tokens = kPool * kTokens});
            CHECK(narrow.kvmem_gen_reserve_tokens() == 0,
                  "a pool that only spans the window reserved %u tokens",
                  narrow.kvmem_gen_reserve_tokens());
            KVAddressSpaceStore greedy(
                pages, tables, 2, kPool,
                KvMemOptions{.enabled = true,
                             .budget_tokens = KVAddressSpaceStore::kMinimumWindowPages * kTokens,
                             .gen_reserve_tokens = kPool * kTokens});
            CHECK(greedy.kvmem_gen_reserve_tokens() ==
                      (kPool - KVAddressSpaceStore::kMinimumWindowPages) * kTokens,
                  "an oversized headroom was not clamped: %u",
                  greedy.kvmem_gen_reserve_tokens());
            KVAddressSpaceStore off(pages, tables, 2, kPool,
                                    KvMemOptions{.enabled = true,
                                                 .budget_tokens = kBudgetPages * kTokens,
                                                 .gen_reserve_tokens = 0});
            CHECK(off.kvmem_gen_reserve_tokens() == 0, "zero headroom was not honored: %u",
                  off.kvmem_gen_reserve_tokens());
        }

        const std::vector<Tensor> planes{pool.plane(0), pool.plane(1), pool.plane(2),
                                         pool.plane(3)};
        const std::int64_t window_bytes = elems_per_page(256);  // I8 V codes: one byte each

        const auto write_mark = [&](int phys, std::uint8_t mark) {
            CHECK(cudaMemcpy(static_cast<std::uint8_t*>(planes[kWindowPlane].data) +
                                 static_cast<std::size_t>(phys) * window_bytes,
                             &mark, sizeof(mark), cudaMemcpyHostToDevice) == cudaSuccess,
                  "H2D mark");
        };
        const auto read_mark = [&](int phys) {
            std::uint8_t mark = 0;
            CHECK(cudaMemcpy(&mark,
                             static_cast<const std::uint8_t*>(planes[kWindowPlane].data) +
                                 static_cast<std::size_t>(phys) * window_bytes,
                             sizeof(mark), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "D2H mark");
            return mark;
        };
        // Fill the whole V page so a stale copy can never be mistaken for a live one.
        const auto fill_page = [&](int phys) {
            std::vector<std::uint8_t> blank(static_cast<std::size_t>(window_bytes), 0xFF);
            CHECK(cudaMemcpy(static_cast<std::uint8_t*>(planes[kWindowPlane].data) +
                                 static_cast<std::size_t>(phys) * window_bytes,
                             blank.data(), blank.size(), cudaMemcpyHostToDevice) == cudaSuccess,
                  "H2D blank");
        };

        // ---- activate and map the WHOLE pool: 8 ledger blocks over 8 device pages ----
        std::optional<KVAddressSpaceHandle> handle = store.create_inactive();
        CHECK(handle.has_value(), "create_inactive");
        store.activate(*handle, kPool, 0);
        store.ensure_mapped_to_tokens(*handle, kPool * kTokens);
        store.commit_frontier(*handle, kPool * kTokens);

        const auto slot_phys = [&](std::size_t slot) {
            const std::vector<LogicalKVPageHandle> logicals = store.window_page_handles(*handle);
            return pool.physical_index_of(pages.physical(logicals[slot]));
        };
        const auto mark_slots = [&]() {
            const std::vector<LogicalKVPageHandle> logicals = store.window_page_handles(*handle);
            std::vector<std::uint8_t> marks;
            for (std::size_t slot = 0; slot < logicals.size(); ++slot) {
                marks.push_back(read_mark(
                    pool.physical_index_of(pages.physical(logicals[slot]))));
            }
            return marks;
        };
        for (int blk = 0; blk < kPool; ++blk) {
            const int phys = slot_phys(static_cast<std::size_t>(blk));
            fill_page(phys);
            write_mark(phys, static_cast<std::uint8_t>(blk + 1));
        }
        CHECK(store.window_page_handles(*handle).size() == static_cast<std::size_t>(kPool),
              "the pool did not map every page");
        CHECK(pool.available_pages() == 0, "the pool was not fully claimed: %u free",
              pool.available_pages());
        std::printf("mapped %d pages, pool free=%u\n", kPool, pool.available_pages());

        // ---- generate past the budget: the reclaim has to hand pages back ----
        // The growth bound is read BEFORE the compaction. The packed window drops the evicted
        // pages from the address, so afterwards the address only names its window: a caller that
        // reclaimed first and read the bound second would re-establish 6 pages, not 8, and the
        // room it had just freed could never be used again.
        const std::uint32_t logical = store.mapping_limit(*handle);
        CHECK(logical == static_cast<std::uint32_t>(kPool), "growth bound = %u", logical);
        const std::uint32_t w1 = store.kvmem_compact_generation(*handle);
        CHECK(w1 == kBudgetPages * kTokens, "generation reclaim window = %u (want %u)", w1,
              kBudgetPages * kTokens);
        CHECK(store.mapping_limit(*handle) == kBudgetPages, "the reclaimed bound = %u",
              store.mapping_limit(*handle));
        CHECK(store.window_page_handles(*handle).size() == kBudgetPages,
              "window is %zu pages (want %u)", store.window_page_handles(*handle).size(),
              kBudgetPages);
        CHECK(pool.available_pages() == kReservePages,
              "the reclaim did not return the headroom: free=%u (want %u)", pool.available_pages(),
              kReservePages);

        const std::vector<std::uint8_t> marks1 = mark_slots();
        std::vector<std::uint8_t> present(kPool + 1, 0);
        for (const std::uint8_t mark : marks1) { present[mark] = 1; }
        CHECK(marks1.front() == 1, "the sink block left slot 0: mark=%u",
              static_cast<unsigned>(marks1.front()));
        CHECK(marks1.back() == kPool, "the newest block left the tail: mark=%u",
              static_cast<unsigned>(marks1.back()));
        std::vector<std::uint8_t> evicted;
        for (int blk = 0; blk < kPool; ++blk) {
            if (present[blk + 1] == 0) { evicted.push_back(static_cast<std::uint8_t>(blk + 1)); }
        }
        CHECK(evicted.size() == static_cast<std::size_t>(kPool - kBudgetPages),
              "evicted %zu blocks (want %u)", evicted.size(), kPool - kBudgetPages);
        std::printf("reclaim -> window %u, marks", w1);
        for (const std::uint8_t mark : marks1) { std::printf(" %u", static_cast<unsigned>(mark)); }
        std::printf("  evicted");
        for (const std::uint8_t mark : evicted) { std::printf(" %u", static_cast<unsigned>(mark)); }
        std::printf("\n");

        // The reclaim runs on every decode round that has run out of room, so a pass over a
        // window that already fits the headroom has to be a fixed point: same window, same
        // marks, same free pages, no churn.
        const std::vector<std::uint8_t> marks_before_repeat = marks1;
        const std::uint32_t repeat = store.kvmem_compact_generation(*handle);
        CHECK(repeat == kBudgetPages * kTokens, "a repeat reclaim returned %u", repeat);
        CHECK(mark_slots() == marks_before_repeat, "a repeat reclaim churned the window");
        CHECK(pool.available_pages() == kReservePages, "a repeat reclaim changed the free pages: %u",
              pool.available_pages());

        // ---- grow again on the reclaimed room, then force a stage-in on a FULL pool ----
        store.resize_entitlement(*handle, logical);
        store.ensure_mapped_to_tokens(*handle, kPool * kTokens);
        store.commit_frontier(*handle, kPool * kTokens);
        CHECK(pool.available_pages() == 0, "the pool was not refilled: free=%u",
              pool.available_pages());
        {
            const std::vector<LogicalKVPageHandle> logicals = store.window_page_handles(*handle);
            CHECK(logicals.size() == static_cast<std::size_t>(kPool),
                  "refilled window is %zu pages", logicals.size());
            // Ledger ids continue from the first mapping, so the two appended blocks are
            // kPool and kPool + 1 in ids, and land on the two reclaimed slots.
            for (std::size_t slot = kBudgetPages; slot < logicals.size(); ++slot) {
                const int phys = pool.physical_index_of(pages.physical(logicals[slot]));
                fill_page(phys);
                write_mark(phys, static_cast<std::uint8_t>(kPool + 1 + (slot - kBudgetPages)));
            }
        }
        std::printf("refilled the pool: free=%u\n", pool.available_pages());

        const std::uint8_t victim = evicted.front();
        const std::uint32_t w2 =
            store.kvmem_compact(*handle, kBudgetPages * kTokens, nullptr,
                                {static_cast<std::uint32_t>(victim - 1)});
        CHECK(w2 == kBudgetPages * kTokens, "stage-in compaction window = %u (want %u)", w2,
              kBudgetPages * kTokens);
        const std::vector<std::uint8_t> marks2 = mark_slots();
        bool revived = false;
        for (const std::uint8_t mark : marks2) { revived = revived || mark == victim; }
        CHECK(revived, "the mandatory block %u did not come back to the device",
              static_cast<unsigned>(victim));
        CHECK(marks2.front() == 1, "the sink block left slot 0 after the stage-in: mark=%u",
              static_cast<unsigned>(marks2.front()));
        std::printf("stage-in on a full pool -> window %u, marks", w2);
        for (const std::uint8_t mark : marks2) { std::printf(" %u", static_cast<unsigned>(mark)); }
        std::printf("\n");

        store.deactivate(*handle);
        std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
        cudaFree(backing);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
