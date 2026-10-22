// OpenSpec 3.3: actual Main16 source-K H2D/rephase producer (not the prior
// resident-only test). Two bounded pinned sources, one moved 64-column page
// and one identity 17-column tail. No old test matrix is re-executed.
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_layer_ready.h"
#include "kvmem/kvmem_source_k.h"

#include <cuda_fp16.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;

namespace {
constexpr std::uint32_t kLayers = 16;
constexpr std::uint32_t kColumns = kPagedKVPageSize;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kPhysical = 6;
int failures = 0;

#define CHECK(condition, ...)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                     \
            ++failures;                                                        \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);                    \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                               \
        }                                                                      \
    } while (0)

void test_source_ready() {
    DeviceContext device;
    KVPageGeometry geometry;
    geometry.page_tokens = kColumns;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        geometry.planes.push_back({DType::I8, 256, kHeads, 256});
        geometry.planes.push_back({DType::I8, 256, kHeads, 256});
        geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
        geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
    }
    LayoutBuilder builder;
    const auto layout = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = kPhysical, .geometry = geometry});
    DeviceBuffer backing(layout.payload_bytes() + (1u << 20));
    CUDA_CHECK(cudaMemset(backing.p, 0xA5, backing.bytes));
    DeviceKVPagePool pool({backing.p, backing.bytes}, layout);

    // K-only first-write payload, quantized once at its original baked
    // positions; source captures preserve those precise int8 codes/scales.
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t plane_kind : {0u, 2u}) {
            const std::uint32_t plane_index = layer * 4 + plane_kind;
            const auto plane = pool.plane(plane_index);
            const std::uint32_t leading = plane_kind == 0 ? 256 : 4;
            const auto stride = static_cast<std::size_t>(plane.nb[3]);
            for (std::uint32_t page = 0; page < 2; ++page) {
                std::vector<std::uint8_t> data(stride, 0xA5);
                for (std::uint32_t head = 0; head < kHeads; ++head) {
                    for (std::uint32_t t = 0; t < kColumns; ++t) {
                        for (std::uint32_t element = 0; element < leading; ++element) {
                            const auto index =
                                (static_cast<std::size_t>(head) * kColumns + t) * leading + element;
                            if (plane_kind == 0) {
                                data[index] = static_cast<std::uint8_t>(
                                    static_cast<std::int8_t>(
                                        static_cast<int>((layer * 11 + page * 17 +
                                            head * 7 + t * 3 + element * 13) % 69) - 34));
                            } else {
                                const float value = 0.03f + 0.002f *
                                    static_cast<float>((layer + page + head + t + element) % 9);
                                const auto bits = __half_as_ushort(__float2half(value));
                                std::memcpy(data.data() + index * sizeof(bits), &bits, sizeof(bits));
                            }
                        }
                    }
                }
                CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(plane.data) + page * stride,
                                      data.data(), stride, cudaMemcpyHostToDevice));
            }
        }
    }

    ImmutableKStore source(geometry);
    std::array<ImmutableKStore::Record, 2> records;
    source.capture(records[0], pool, 0, 0, 64, device.stream);
    source.capture(records[1], pool, 1, 64, 17, device.stream);

    // References first, before the async producer starts. This is the original
    // per-page production route, so the check isolates the new scheduling and
    // per-layer launch granularity rather than repeating an old suite.
    source.restore(records[0], pool, 4, 128, device.stream);
    source.restore(records[1], pool, 5, 64, device.stream);

    LayerReadyFence fence;
    fence.begin(device.stream);
    const std::array<ImmutableKStore::RestoreRequest, 2> requests{{
        {&records[0], 2, 128}, {&records[1], 3, 64}}};
    source.prepare_layer_sources(requests, pool, fence.producer_stream());
    std::array<cudaEvent_t, kLayers> events{};
    for (std::uint32_t i = 0; i < kLayers; ++i) { events[i] = fence.ready_event(i); }
    struct SourceCallback {
        ImmutableKStore* source;
        DeviceKVPagePool* pool;
    } callback{&source, &pool};
    kvmem_rerope_pages_layer_ready(pool, nullptr, nullptr, nullptr, 0, {}, events,
                                   fence.producer_stream(),
                                   +[](void* context, std::uint32_t layer, cudaStream_t stream) {
        auto& owner = *static_cast<SourceCallback*>(context);
        owner.source->enqueue_layer_source(layer, *owner.pool, stream);
    }, &callback);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) { fence.mark_ready(layer); }

    // A producer event alone does not authorize any arbitrary consumer stream.
    // The release must follow the *same* stream's final actual KV read, or a
    // recycled physical page can overwrite an in-flight attention reader.
    cudaStream_t unrelated = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking));
    bool rejected_wrong_wait = false;
    try { fence.wait_before_layer(0, unrelated); }
    catch (const std::logic_error&) { rejected_wrong_wait = true; }
    CHECK(rejected_wrong_wait, "layer-ready accepted an unbound consumer wait");

    const auto code_bytes = static_cast<std::size_t>(pool.plane(0).nb[3]);
    const auto scale_bytes = static_cast<std::size_t>(pool.plane(2).nb[3]);
    const auto per_pair = code_bytes + scale_bytes;
    PinnedHostBuffer consumed(kLayers * 2 * per_pair);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        fence.wait_before_layer(layer, device.stream);
        if (layer == 0) {
            // Test after the legitimate wait, so rejection cannot be caused by
            // the separate 'no wait yet' precondition.
            bool rejected_wrong_release = false;
            try { fence.release_after_layer(layer, unrelated); }
            catch (const std::logic_error&) { rejected_wrong_release = true; }
            CHECK(rejected_wrong_release, "layer-ready accepted an unbound reader release");
        }
        for (std::uint32_t page = 0; page < 2; ++page) {
            auto* destination = static_cast<std::byte*>(consumed.data()) +
                (layer * 2 + page) * per_pair;
            const auto code_plane = pool.plane(layer * 4);
            const auto scale_plane = pool.plane(layer * 4 + 2);
            CUDA_CHECK(cudaMemcpyAsync(destination,
                 static_cast<std::byte*>(code_plane.data) + (page + 2) * code_plane.nb[3],
                 code_bytes, cudaMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(cudaMemcpyAsync(destination + code_bytes,
                 static_cast<std::byte*>(scale_plane.data) + (page + 2) * scale_plane.nb[3],
                 scale_bytes, cudaMemcpyDeviceToHost, device.stream));
        }
        fence.release_after_layer(layer, device.stream);
    }
    CUDA_CHECK(cudaStreamDestroy(unrelated));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    source.drain_layer_source();
    fence.retire();

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t page = 0; page < 2; ++page) {
            const auto* actual = static_cast<const std::byte*>(consumed.data()) +
                (layer * 2 + page) * per_pair;
            std::vector<std::byte> expected(per_pair);
            const auto codes = pool.plane(layer * 4);
            const auto scales = pool.plane(layer * 4 + 2);
            CUDA_CHECK(cudaMemcpy(expected.data(),
                static_cast<std::byte*>(codes.data) + (page + 4) * codes.nb[3],
                code_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(expected.data() + code_bytes,
                static_cast<std::byte*>(scales.data) + (page + 4) * scales.nb[3],
                scale_bytes, cudaMemcpyDeviceToHost));
            CHECK(std::memcmp(actual, expected.data(), per_pair) == 0,
                  "Main16 source H2D/rebuild mismatch layer=%u source=%u", layer, page);
            if (page == 1) {
                // The 17-column identity tail did not transfer stale K bits
                // into future, unwritten columns of the new physical page.
                const auto untouched_code = (0 * kColumns + 17) * 256;
                const auto untouched_scale = (0 * kColumns + 17) * 4 * sizeof(__half);
                CHECK(actual[untouched_code] == std::byte{0xA5} &&
                      actual[code_bytes + untouched_scale] == std::byte{0xA5},
                      "Main16 partial-tail H2D overwrote uncommitted columns layer=%u", layer);
            }
        }
    }

    // New-path failure only: an invalid duplicate target is rejected before
    // submitting any producer writes; cancellation drains an incomplete run.
    auto duplicate = requests;
    duplicate[1].physical = 2;
    bool rejected = false;
    try { source.prepare_layer_sources(duplicate, pool, fence.producer_stream()); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected, "source pipeline accepted overlapping physical destinations");
    LayerReadyFence cancelled;
    cancelled.begin(device.stream);
    source.prepare_layer_sources(requests, pool, cancelled.producer_stream());
    source.enqueue_layer_source(0, pool, cancelled.producer_stream());
    source.drain_layer_source();
    cancelled.drain_consumers();
    CHECK(!cancelled.active(), "cancelled source producer not drained");
    source.release(records[0]);
    source.release(records[1]);
    CHECK(source.occupied_bytes() == 0, "source backing records leaked");
}
} // namespace

int main() {
    try { test_source_ready(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "source layer test exception: %s\n", error.what());
        return 2;
    }
    std::printf("SOURCE_LAYER_READY failures=%d VERDICT: %s\n",
                failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
