// OpenSpec 3.3 Cold-only forced page recall. This exercises the new
// bounded per-layer file-backed H2D producer, not the previously accepted
// synchronous Cold regression suite.
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6::detail;

namespace {
constexpr std::uint32_t kLayers = 16;
constexpr std::uint32_t kTokens = kPagedKVPageSize;
constexpr std::uint32_t kPhysical = 8;
int failures = 0;
#define CHECK(ok, ...)                                                         \
    do {                                                                       \
        if (!(ok)) {                                                            \
            ++failures;                                                        \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);                    \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                               \
        }                                                                      \
    } while (0)

enum class Case { Success, PreadFailure, PartialH2dFailure };

void forced_cold_layer_recall(Case test_case) {
#if defined(_WIN32)
    CHECK(::_putenv_s("NINFER_KVMEM_HOST_PAGES", "1") == 0, "Host quota");
    CHECK(::_putenv_s("NINFER_KVMEM_LAYER_READY", "1") == 0, "Layer opt-in");
    CHECK(::_putenv_s("NINFER_KVMEM_LAYER_READY_TRACE", "1") == 0, "Layer trace");
#else
    CHECK(::setenv("NINFER_KVMEM_HOST_PAGES", "1", 1) == 0, "Host quota");
    CHECK(::setenv("NINFER_KVMEM_LAYER_READY", "1", 1) == 0, "Layer opt-in");
    CHECK(::setenv("NINFER_KVMEM_LAYER_READY_TRACE", "1", 1) == 0, "Layer trace");
#endif
    DeviceContext device;
    KVPageGeometry geometry;
    geometry.page_tokens = kTokens;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        geometry.planes.push_back({DType::I8, 256, 4, 256});
        geometry.planes.push_back({DType::I8, 256, 4, 256});
        geometry.planes.push_back({DType::FP16, 4, 4, 256});
        geometry.planes.push_back({DType::FP16, 4, 4, 256});
    }
    LayoutBuilder builder;
    const auto pool_layout = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = kPhysical, .geometry = geometry});
    const auto table_layout = plan_kv_execution_tables(
        builder, KVExecutionTableSpec{.logical_page_capacity = kPhysical, .table_rows = 1});
    DeviceBuffer backing(pool_layout.payload_bytes() + table_layout.metadata_bytes() + (1u << 20));
    CUDA_CHECK(cudaMemset(backing.p, 0, backing.bytes));
    DeviceKVPagePool pool({backing.p, backing.bytes}, pool_layout);
    KVExecutionTablePool tables({backing.p, backing.bytes}, table_layout, pool);
    LogicalKVPageStore pages(pool, kPhysical);
    DeviceBuffer metadata(kPhysical * 3 * sizeof(std::int32_t));
    KVAddressSpaceStore store(pages, tables, 1, kPhysical, KvMemOptions{.enabled = true},
                             {metadata.p, metadata.bytes});
    const auto handle = store.create_inactive();
    if (!handle) { throw std::runtime_error("failed to allocate KVMem address"); }
    store.activate(*handle, kPhysical, 0);
    store.ensure_mapped_to_tokens(*handle, 6 * kTokens);
    store.commit_frontier(*handle, 6 * kTokens);

    // Make V codes semantically observable independent of rephased K. Source
    // page 1 and 2 must carry their original patterns through Host or NVMe.
    const auto originally = store.window_page_handles(*handle);
    CHECK(originally.size() == 6, "expected six original K/V blocks");
    for (std::uint32_t block = 0; block < originally.size(); ++block) {
        const auto physical = pool.physical_index_of(pages.physical(originally[block]));
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const auto plane = pool.plane(layer * 4 + 1);
            const std::size_t page_bytes = static_cast<std::size_t>(plane.nb[3]);
            std::vector<std::uint8_t> host(page_bytes,
                static_cast<std::uint8_t>(block * 17 + layer * 3 + 7));
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(plane.data) +
                static_cast<std::size_t>(physical) * page_bytes, host.data(), host.size(),
                cudaMemcpyHostToDevice));
        }
    }
    store.deactivate(*handle);

    // First selection parks blocks 1 and 2 under a one-page Host budget.
    // At least one must spill to Cold. The next explicit selection requests
    // BOTH again, forcing one genuine Cold stage-in in a two-page recall.
    const std::array<std::uint32_t, 4> first{0, 3, 4, 5};
    const auto w1 = store.kvmem_compact(*handle, 4 * kTokens, device.stream, {},
                                       false, 0, first);
    CHECK(w1 == 4 * kTokens && store.kvmem_cold_bytes() != 0,
          "initial one-Host-page compaction did not spill to Cold");
    const std::array<std::uint32_t, 4> second{0, 1, 2, 5};
    if (test_case == Case::PreadFailure) {
        // A real std::async Cold prefetch delivers a read error via get().
        store.enable_kvmem_async_d2h(device);
#if defined(_WIN32)
        CHECK(::_putenv_s("NINFER_KVMEM_TEST_COLD_PREAD_FAILURE", "1") == 0,
              "enable pread error receipt");
#else
        CHECK(::setenv("NINFER_KVMEM_TEST_COLD_PREAD_FAILURE", "1", 1) == 0,
              "enable pread error receipt");
#endif
    } else if (test_case == Case::PartialH2dFailure) {
#if defined(_WIN32)
        CHECK(::_putenv_s("NINFER_KVMEM_TEST_PARTIAL_LAYER_H2D_FAILURE", "1") == 0,
              "enable partial H2D error receipt");
#else
        CHECK(::setenv("NINFER_KVMEM_TEST_PARTIAL_LAYER_H2D_FAILURE", "1", 1) == 0,
              "enable partial H2D error receipt");
#endif
    }
    if (test_case != Case::Success) {
        bool received_injected_error = false;
        try {
            (void)store.kvmem_compact(*handle, 4 * kTokens, device.stream, {},
                                      false, 0, second, std::nullopt, true);
        } catch (const std::runtime_error& error) {
            received_injected_error =
                std::string_view(error.what()).find("injected") != std::string_view::npos;
            std::printf("COLD_LAYER_FAIL captured=%s\n", error.what());
        }
        CHECK(received_injected_error, "expected injected Cold/partial H2D failure");
        CHECK(store.kvmem_layer_failure_drains() == 1,
              "failed compaction did not run one bounded layer drain");
        CHECK(store.kvmem_layer_pending_pages() == 0,
              "failed compaction left pending source/Cold pinned pages");
        CHECK(store.kvmem_layer_ready_for(*handle) == nullptr,
              "failed compaction published a layer-ready ticket");
        CHECK(store.kvmem_inflight_pages() == 0,
              "failed compaction left async physical pages in flight");
        // An aborted request must also be disposable: no orphaned stage-in
        // physical allocations should remain after releasing its address.
        const bool released = store.release(*handle);
        CHECK(released, "failed compact address could not be released");
        CHECK(pool.allocated_pages() == 0,
              "failed compact leaked %u physical pages after address release",
              pool.allocated_pages());
        return;
    }
    const auto w2 = store.kvmem_compact(*handle, 4 * kTokens, device.stream, {},
                                       false, 0, second, std::nullopt, true);
    CHECK(w2 == 4 * kTokens, "forced two-page recall has wrong window=%u", w2);
    auto* consumer = store.kvmem_layer_ready_for(*handle);
    CHECK(consumer != nullptr, "Cold layer producer did not publish a consumer ticket");
    if (!consumer) { throw std::runtime_error("missing Cold layer consumer"); }
    const auto selected = store.window_page_handles(*handle);
    CHECK(selected.size() == 4, "Cold layer window should have four pages");
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        consumer->wait_before_layer(layer, device.stream);
        for (std::size_t slot = 1; slot <= 2; ++slot) {
            const auto physical = pool.physical_index_of(pages.physical(selected.at(slot)));
            const auto plane = pool.plane(layer * 4 + 1);
            const std::size_t bytes = static_cast<std::size_t>(plane.nb[3]);
            std::vector<std::uint8_t> values(bytes);
            CUDA_CHECK(cudaMemcpyAsync(values.data(),
                static_cast<const std::byte*>(plane.data) +
                    static_cast<std::size_t>(physical) * bytes,
                bytes, cudaMemcpyDeviceToHost, device.stream));
            // pageable allocations need explicit completion before inspection
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            const auto expected = static_cast<std::uint8_t>(second[slot] * 17 + layer * 3 + 7);
            CHECK(!values.empty() && values.front() == expected && values.back() == expected,
                  "Cold layer V payload changed block=%u layer=%u", second[slot], layer);
        }
        consumer->release_after_layer(layer, device.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    CHECK(store.kvmem_inflight_pages() == 0, "Cold restore left a page in flight");
    CHECK(store.release(*handle), "Cold layer address release failed");
    CHECK(store.kvmem_layer_ready_for(*handle) == nullptr,
          "released Cold window exposed a stale generation ticket");
}
} // namespace

int main(int argc, char** argv) {
    Case test_case = Case::Success;
    if (argc == 2 && std::strcmp(argv[1], "--pread-failure") == 0) {
        test_case = Case::PreadFailure;
    } else if (argc == 2 && std::strcmp(argv[1], "--partial-h2d-failure") == 0) {
        test_case = Case::PartialH2dFailure;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: kvmem_layer_cold_test [--pread-failure|--partial-h2d-failure]\n");
        return 2;
    }
    try { forced_cold_layer_recall(test_case); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "COLD_LAYER exception: %s\n", e.what());
        return 2;
    }
    std::printf("COLD_LAYER_READY failures=%d VERDICT: %s\n", failures,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
