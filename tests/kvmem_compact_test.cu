#include "kvmem/prefix_checkpoint_file.h"
#include <fstream>
#include <iterator>
// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// Store-level K1b compaction test: a real LogicalKVPageStore +
// KVAddressSpaceStore over a real page pool, driving the lifecycle exactly
// like the runtime (activate -> ensure -> commit -> deactivate -> compact),
// then verifying the compacted window byte-for-byte where position-free and
// against a CPU re-phase model for group 0.
//
// Run: builder.exe -test tests/kvmem_compact_test.cu

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <thread>

#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_blocks.h"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::targets::qwen3_6::detail;
using namespace ninfer::kvmem;

namespace {

constexpr int kPages  = 8;   // physical pool capacity (page groups)
constexpr int kHeads  = 4;
constexpr int kLayers = 2;
constexpr int kTokens = kPagedKVPageSize;
constexpr int kDim    = kKVCacheInt8HeadDim;
constexpr int kGroups = kKVCacheInt8Groups;

int failures = 0;
struct CopyGate {
    std::atomic<bool> open{false};
    std::atomic<bool> finished{false};
};

void block_copy_until_released(void* pointer) {
    auto& gate = *static_cast<CopyGate*>(pointer);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!gate.open.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    gate.finished.store(true, std::memory_order_release);
}
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++failures;                                     \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

float host_freq(int pair) { return std::pow(1e7f, -2.0f * static_cast<float>(pair) / 64.0f); }

void hadamard256_warp(float* e) {
    for (int stride = 1; stride <= 16; stride <<= 1) {
        for (int r = 0; r < 8; ++r) {
            float next[32];
            float* col = e + r * 32;
            for (int l = 0; l < 32; ++l) {
                const int peer = l ^ stride;
                next[l]        = (l & stride) == 0 ? (col[l] + col[peer]) : (col[peer] - col[l]);
            }
            for (int l = 0; l < 32; ++l) { col[l] = next[l]; }
        }
    }
    for (int l = 0; l < 32; ++l) {
        for (int span = 1; span < 8; span <<= 1) {
            for (int base = 0; base < 8; base += 2 * span) {
                for (int offset = 0; offset < span; ++offset) {
                    const float low                    = e[(base + offset) * 32 + l];
                    const float high                   = e[(base + offset + span) * 32 + l];
                    e[(base + offset) * 32 + l]        = low + high;
                    e[(base + offset + span) * 32 + l] = low - high;
                }
            }
        }
        for (int r = 0; r < 8; ++r) { e[r * 32 + l] *= 0.0625f; }
    }
}

void rope_group0(float* e, std::int32_t position, bool inverse) {
    // Text MRoPE: all three axes carry the same position.
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(position) * host_freq(p);
        float s         = std::sin(phi);
        const float c   = std::cos(phi);
        if (inverse) { s = -s; }
        const float a = e[p];
        const float b = e[p + 32];
        e[p]          = a * c - b * s;
        e[p + 32]     = b * c + a * s;
    }
}

// Quantize a rotated group the way the append path does.
void quantize_group(const float* g, std::int8_t* codes, unsigned short* scale_bits) {
    float absmax = 0.0f;
    for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(g[i])); }
    const unsigned short bits =
        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
    const float s   = __half2float(__ushort_as_half(bits));
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < 64; ++i) {
        const int q = std::max(-127, std::min(127, static_cast<int>(std::lrintf(g[i] * inv))));
        codes[i]    = static_cast<std::int8_t>(q);
    }
    *scale_bits = bits;
}

// The stored K row model for block `page` baked at window position `baked`:
// Quant_G64(H256(RoPE_baked(raw))).
void expected_k_row(const std::vector<float>& raw, int page, int layer, int head, int token,
                    std::int32_t baked, std::int8_t* codes, unsigned short* scale_bits) {
    float row[kDim];
    const std::size_t base =
        (((static_cast<std::size_t>(page) * kLayers + layer) * kHeads + head) * kTokens + token) *
        kDim;
    for (int d = 0; d < kDim; ++d) { row[d] = raw[base + static_cast<std::size_t>(d)]; }
    rope_group0(row, baked, false);
    hadamard256_warp(row);
    for (int g = 0; g < kGroups; ++g) {
        quantize_group(row + g * 64, codes + g * 64, &scale_bits[g]);
    }
}

// The kernel's float sequence applied to the STORED quantized 256-dim row:
// dequant_G64 -> H256^-1 -> un-RoPE(from) -> RoPE(to) -> H256 -> quantize_G64.
void expected_rephase(const std::int8_t* stored_codes, const unsigned short* stored_bits,
                      std::int32_t from_pos, std::int32_t to_pos, std::int8_t* codes,
                      unsigned short* scale_bits) {
    float row[kDim];
    for (int g = 0; g < kGroups; ++g) {
        const float s = __half2float(__ushort_as_half(stored_bits[g]));
        for (int i = 0; i < 64; ++i) {
            row[g * 64 + i] = static_cast<float>(stored_codes[g * 64 + i]) * s;
        }
    }
    hadamard256_warp(row);
    rope_group0(row, from_pos, true);
    rope_group0(row, to_pos, false);
    hadamard256_warp(row);
    for (int g = 0; g < kGroups; ++g) {
        quantize_group(row + g * 64, codes + g * 64, &scale_bits[g]);
    }
}

// Plane addressing (PageMajor, element units of the plane dtype).
std::int64_t elems_per_page(int leading) {
    return static_cast<std::int64_t>(leading) * kTokens * kHeads;
}
std::int64_t in_page_elems(int leading, int head, int token, int d) {
    return static_cast<std::int64_t>(leading) * (static_cast<std::int64_t>(head) * kTokens + token) +
           d;
}

// A live KVMem address may own a 32-page device claim inside a 36-page
// physical pool. A prefill can exhaust the 32 reserved pages while leaving
// four physically free; the next decode page must trigger reclaim at 32,
// never at physical page 36. Recreate the same lifecycle as a chunked prefill:
// compact, restore the logical growth entitlement, regrow to the claim, then
// compact for generation again. Repeat to exercise Host/NVMe spill as well.
void test_generation_device_claim(bool cold_prefetch = false) {
    constexpr std::uint32_t physical_pages = 36;  // --kv-device-tokens 2304
    constexpr std::uint32_t budget_pages   = 32;  // --kvmem-budget 2048
    constexpr std::uint32_t reserve_pages  = 2;   // --kvmem-gen-reserve 128
    constexpr std::uint32_t logical_pages  = 64;  // --max-context 4096
#if defined(_WIN32)
    CHECK(::_putenv_s("NINFER_KVMEM_HOST_PAGES", "2") == 0, "set Host cap");
    CHECK(::_putenv_s("NINFER_KVMEM_COLD_DIR", ".") == 0, "set cold directory");
#else
    CHECK(::setenv("NINFER_KVMEM_HOST_PAGES", "2", 1) == 0, "set Host cap");
    CHECK(::setenv("NINFER_KVMEM_COLD_DIR", ".", 1) == 0, "set cold directory");
#endif

    KVPageGeometry geometry;
    geometry.page_tokens        = kTokens;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    geometry.planes.push_back({DType::I8, 256, kHeads, 256});
    geometry.planes.push_back({DType::I8, 256, kHeads, 256});
    geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
    geometry.planes.push_back({DType::FP16, 4, kHeads, 256});

    LayoutBuilder builder;
    const auto pool_layout = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages, .geometry = geometry});
    const auto table_layout = plan_kv_execution_tables(
        builder, KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = 1});
    const std::size_t allocation_bytes =
        pool_layout.payload_bytes() + table_layout.metadata_bytes() + (1U << 20);
    void* backing = nullptr;
    CHECK(cudaMalloc(&backing, allocation_bytes) == cudaSuccess, "claim: cudaMalloc");
    if (!backing) { throw std::runtime_error("claim: no device backing"); }
    CHECK(cudaMemset(backing, 0, allocation_bytes) == cudaSuccess, "claim: zero KV pages");
    // Keep the original claim fixture synchronous. The opt-in variant only
    // enables the two-slot Cold queue at the final restored-evidence batch.
    std::unique_ptr<DeviceContext> prefetch_device;
    if (cold_prefetch) { prefetch_device = std::make_unique<DeviceContext>(); }
    {
        DeviceKVPagePool pool(DeviceSpan{backing, allocation_bytes}, pool_layout);
        KVExecutionTablePool tables(DeviceSpan{backing, allocation_bytes}, table_layout, pool);
        LogicalKVPageStore pages(pool, logical_pages);
        DeviceBuffer rerope_meta(3ULL * logical_pages * sizeof(std::int32_t));
        KVAddressSpaceStore store(
            pages, tables, 1, logical_pages,
            KvMemOptions{.enabled = true,
                         .budget_tokens = budget_pages * kTokens,
                         .gen_reserve_tokens = reserve_pages * kTokens,
                         .action_lrfu = true}, {rerope_meta.p, rerope_meta.bytes});
        const auto handle = store.create_inactive();
        CHECK(handle.has_value(), "claim: allocate address");
        if (!handle) { throw std::logic_error("claim: address allocation failed"); }

        store.activate(*handle, logical_pages, 0);
        CHECK(store.mapping_limit(*handle) == logical_pages,
              "claim: logical entitlement must cover full context");
        CHECK(store.kvmem_device_claim_pages(*handle) == budget_pages,
              "claim: device entitlement must be budget pages (not physical pool)");
        CHECK(pool.capacity_pages() == physical_pages,
              "claim: test requires four unclaimed physical pages");
        CHECK(store.kvmem_gen_reserve_tokens() == reserve_pages * kTokens,
              "claim: generation reserve is missing");

        // A prefix filled right to the KVMem budget consumes every reserved
        // page while the GPU still has physical capacity. The next token must
        // reclaim before ensure_mapped_to_tokens can attempt page 33.
        store.ensure_mapped_to_tokens(*handle, budget_pages * kTokens);
        store.commit_frontier(*handle, budget_pages * kTokens);
        CHECK(pool.reserved_pages() == 0 && pool.allocated_pages() == budget_pages,
              "claim: active budget should consume its exact reservation");
        CHECK(store.kvmem_reclaim_watermark_pages(*handle) == budget_pages - 1,
              "claim: typed bounded lead must retain one growth page");
        CHECK(!store.kvmem_generation_reclaim_due(*handle, (budget_pages - 1) * kTokens),
              "claim: reclaim must not run before crossing early watermark page 31");
        CHECK(store.kvmem_generation_reclaim_due(*handle, budget_pages * kTokens + 1),
              "claim: page 33 must reclaim even with physical pool room left");

        for (int round = 0; round < 3; ++round) {
            const std::uint32_t logical_bound = store.mapping_limit(*handle);
            CHECK(logical_bound == logical_pages,
                  "claim round %d: lost logical growth entitlement (%u)", round, logical_bound);
            const std::uint32_t window = store.kvmem_compact_generation(*handle);
            CHECK(window == (budget_pages - reserve_pages) * kTokens,
                  "claim round %d: reclaim window is %u", round, window);
            CHECK(pool.allocated_pages() == budget_pages - reserve_pages,
                  "claim round %d: active window occupies %u pages", round,
                  pool.allocated_pages());
            store.resize_entitlement(*handle, logical_bound);
            CHECK(pool.reserved_pages() == reserve_pages,
                  "claim round %d: missing %u pages of reservation after reclaim", round,
                  pool.reserved_pages());
            store.ensure_mapped_to_tokens(*handle, budget_pages * kTokens);
            store.commit_frontier(*handle, budget_pages * kTokens);
            CHECK(pool.allocated_pages() == budget_pages && pool.reserved_pages() == 0,
                  "claim round %d: reserved room failed to materialize", round);
            CHECK(store.kvmem_generation_reclaim_due(*handle, budget_pages * kTokens + 1),
                  "claim round %d: next decode boundary must reclaim again", round);
        }
        CHECK(store.kvmem_cold_bytes() > 0,
              "claim: repeated compactions must exercise low-capacity cold storage");

        // Completion-time compaction may restore cold blocks into an *active*
        // address that still owns its two reserved growth pages. Reserve every
        // other free physical page as if it belonged to another lane, so no
        // spare unclaimed capacity exists. The next window grows 30 -> 32 and
        // must borrow this address's entitlement for its stage-in pages. The
        // previous independent pool.reserve() failed here after the K0 ledger
        // had already been committed, corrupting a later prefix continuation.
        const std::uint32_t logical_before = store.mapping_limit(*handle);
        CHECK(store.kvmem_compact_generation(*handle) ==
                  (budget_pages - reserve_pages) * kTokens,
              "claim: final generation reclaim to 30 pages");
        store.resize_entitlement(*handle, logical_before);
        const std::uint32_t spare = pool.available_pages();
        CHECK(spare == physical_pages - budget_pages,
              "claim: expected four spare pages before competing reservation");
        auto competing = pool.reserve(spare);
        CHECK(competing.has_value(), "claim: competing lane reservation failed");
        CHECK(pool.available_pages() == 0 && pool.reserved_pages() == spare + reserve_pages,
              "claim: competing lane must leave zero unclaimed capacity");
        const std::uint32_t lexical[] = {7, 8};
        store.set_kvmem_lexical_pages(*handle, lexical);
        CHECK(store.kvmem_lexical_restore_needed(*handle),
              "claim: archived evidence must request restore even while the question fits");
        CHECK(!store.kvmem_generation_reclaim_due(
                  *handle, store.committed_frontier(*handle) + 1),
              "claim: lexical restore must not depend on generation capacity pressure");
        const std::uint64_t stages_before = store.kvmem_usage().stage_in_blocks;
        const std::uint64_t cold_reads_before = store.kvmem_usage().nvme_prefetch_submitted;
        if (cold_prefetch) { store.enable_kvmem_async_d2h(*prefetch_device); }
        CHECK(store.kvmem_compact(*handle, budget_pages * kTokens,
                                  cold_prefetch ? prefetch_device->stream : nullptr) ==
                  budget_pages * kTokens,
              "claim: completion window must grow using its own active reservation");
        if (cold_prefetch) {
            (void)store.retire_kvmem_transfers(true);
            const auto usage = store.kvmem_usage();
            CHECK(usage.nvme_prefetch_submitted >= cold_reads_before + 2 &&
                      usage.nvme_prefetch_submitted == usage.nvme_prefetch_completed,
                  "claim: two pending Cold sources must finish before publishing (%llu/%llu)",
                  static_cast<unsigned long long>(usage.nvme_prefetch_submitted),
                  static_cast<unsigned long long>(usage.nvme_prefetch_completed));
            CHECK(store.kvmem_inflight_pages() == 0,
                  "claim: Cold batch left a GPU source in flight");
            std::printf("claim Cold queue submitted=%llu completed=%llu wait_us=%llu "
                        "transfer_us=%llu staging=%zu\n",
                        static_cast<unsigned long long>(usage.nvme_prefetch_submitted - cold_reads_before),
                        static_cast<unsigned long long>(usage.nvme_prefetch_completed),
                        static_cast<unsigned long long>(usage.nvme_prefetch_wait_us),
                        static_cast<unsigned long long>(usage.transfer_us),
                        usage.source_k_staging_bytes);
        }
        CHECK(!store.kvmem_lexical_restore_needed(*handle),
              "claim: completed restore must make both evidence pages device-visible");
        CHECK(store.kvmem_usage().stage_in_blocks >= stages_before + reserve_pages,
              "claim: selected Host/NVMe blocks did not stage back in");
        CHECK(store.window_page_handles(*handle).size() == budget_pages &&
                  pool.allocated_pages() == budget_pages &&
                  pool.reserved_pages() == spare,
              "claim: stage-in must consume two own pages and preserve competing claim");
        if (competing) { competing->release(); }
        store.deactivate(*handle);
        CHECK(store.release(*handle), "claim: address release");
        CHECK(store.kvmem_cold_bytes() == 0 && pool.allocated_pages() == 0 &&
                  pool.reserved_pages() == 0,
              "claim: cold and physical pages must all be released");
    }
    CHECK(cudaFree(backing) == cudaSuccess, "claim: release GPU backing");
}

} // namespace

// A real 16-layer PageMajor GPU producer and sixteen independent consumer
// waits. Unlike the small two-layer store fixture, this tests the actual SM89
// Main16 geometry and the same resident-K rephase math on both streams.
void test_layer_ready() {
    constexpr int layers = 16;
    constexpr int physical_pages = 8;
#if defined(_WIN32)
    CHECK(::_putenv_s("NINFER_KVMEM_LAYER_READY", "1") == 0, "enable layer-ready opt-in");
#else
    CHECK(::setenv("NINFER_KVMEM_LAYER_READY", "1", 1) == 0, "enable layer-ready opt-in");
#endif
    DeviceContext device;
    KVPageGeometry geometry;
    geometry.page_tokens = kTokens;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    for (int layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back({DType::I8, 256, kHeads, 256});
        geometry.planes.push_back({DType::I8, 256, kHeads, 256});
        geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
        geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
    }
    LayoutBuilder builder;
    const auto layout = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages, .geometry = geometry});
    const auto tables_layout = plan_kv_execution_tables(
        builder, KVExecutionTableSpec{.logical_page_capacity = physical_pages, .table_rows = 1});
    DeviceBuffer backing(layout.payload_bytes() + tables_layout.metadata_bytes() + (1U << 20));
    CUDA_CHECK(cudaMemset(backing.p, 0, backing.bytes));
    DeviceKVPagePool pool({backing.p, backing.bytes}, layout);
    std::array<std::vector<std::int8_t>, layers> original_codes;
    std::array<std::vector<unsigned short>, layers> original_scales;
    for (int layer = 0; layer < layers; ++layer) {
        const Tensor codes = pool.plane(static_cast<std::size_t>(layer) * 4);
        const Tensor scales = pool.plane(static_cast<std::size_t>(layer) * 4 + 2);
        const auto code_stride = static_cast<std::size_t>(codes.nb[3]);
        const auto scale_stride = static_cast<std::size_t>(scales.nb[3]);
        std::vector<std::byte> code_bytes(code_stride * physical_pages);
        std::vector<std::byte> scale_bytes(scale_stride * physical_pages);
        original_codes[layer].resize(kHeads * kTokens * kDim);
        original_scales[layer].resize(kHeads * kTokens * kGroups);
        for (int head = 0; head < kHeads; ++head) {
            for (int t = 0; t < kTokens; ++t) {
                const auto element = static_cast<std::size_t>(head * kTokens + t);
                for (int d = 0; d < kDim; ++d) {
                    const auto value = static_cast<std::int8_t>(
                        ((layer * 11 + head * 7 + t * 3 + d * 13) % 47) - 23);
                    original_codes[layer][element * kDim + d] = value;
                    for (int physical = 0; physical < physical_pages; ++physical) {
                        code_bytes[static_cast<std::size_t>(physical) * code_stride +
                                   in_page_elems(kDim, head, t, d)] = static_cast<std::byte>(value);
                    }
                }
                for (int g = 0; g < kGroups; ++g) {
                    const auto bits = __half_as_ushort(__float2half(0.025f +
                        0.002f * static_cast<float>((layer + head + t + g) % 7)));
                    original_scales[layer][element * kGroups + g] = bits;
                    for (int physical = 0; physical < physical_pages; ++physical) {
                        const auto offset = static_cast<std::size_t>(physical) * scale_stride +
                            static_cast<std::size_t>(in_page_elems(kGroups, head, t, g)) * 2;
                        std::memcpy(scale_bytes.data() + offset, &bits, sizeof(bits));
                    }
                }
            }
        }
        CHECK(cudaMemcpy(codes.data, code_bytes.data(), code_bytes.size(), cudaMemcpyHostToDevice)
                  == cudaSuccess, "layer-ready input K codes H2D layer=%d", layer);
        CHECK(cudaMemcpy(scales.data, scale_bytes.data(), scale_bytes.size(), cudaMemcpyHostToDevice)
                  == cudaSuccess, "layer-ready input K scales H2D layer=%d", layer);
    }
    PinnedHostBuffer metadata(3 * sizeof(std::int32_t));
    DeviceBuffer device_metadata(3 * sizeof(std::int32_t));
    const KvmemReropeWorkspace workspace{
        {static_cast<std::int32_t*>(metadata.data()), 3},
        {static_cast<std::int32_t*>(device_metadata.p), 3}};
    const std::array<std::int32_t, 1> sync_page{1};
    const std::array<std::int32_t, 1> async_page{0};
    const std::array<std::int32_t, 1> from{64};
    const std::array<std::int32_t, 1> to{128};
    kvmem_rerope_pages(pool, sync_page.data(), from.data(), to.data(), 1, workspace, device.stream);

    LayerReadyFence ready;
    ready.begin(device.stream);
    std::array<cudaEvent_t, layers> events{};
    for (int l = 0; l < layers; ++l) { events[l] = ready.ready_event(l); }
    kvmem_rerope_pages_layer_ready(pool, async_page.data(), from.data(), to.data(), 1,
                                   workspace, events, ready.producer_stream());
    for (int l = 0; l < layers; ++l) { ready.mark_ready(l); }

    std::size_t code_page_bytes = static_cast<std::size_t>(pool.plane(0).nb[3]);
    std::size_t scale_page_bytes = static_cast<std::size_t>(pool.plane(2).nb[3]);
    PinnedHostBuffer readback(static_cast<std::size_t>(layers) * (code_page_bytes + scale_page_bytes));
    for (int l = 0; l < layers; ++l) {
        const auto codes = pool.plane(static_cast<std::size_t>(l) * 4);
        const auto scales = pool.plane(static_cast<std::size_t>(l) * 4 + 2);
        auto* target = static_cast<std::byte*>(readback.data()) +
            static_cast<std::size_t>(l) * (code_page_bytes + scale_page_bytes);
        ready.wait_before_layer(l, device.stream);
        CHECK(cudaMemcpyAsync(target, codes.data, code_page_bytes, cudaMemcpyDeviceToHost,
                              device.stream) == cudaSuccess,
              "layer-ready consumer K read layer=%d", l);
        CHECK(cudaMemcpyAsync(target + code_page_bytes, scales.data, scale_page_bytes,
                              cudaMemcpyDeviceToHost, device.stream) == cudaSuccess,
              "layer-ready consumer scale read layer=%d", l);
        ready.release_after_layer(l, device.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    ready.retire();

    for (int l = 0; l < layers; ++l) {
        const auto codes = pool.plane(static_cast<std::size_t>(l) * 4);
        const auto scales = pool.plane(static_cast<std::size_t>(l) * 4 + 2);
        const auto* result = static_cast<const std::byte*>(readback.data()) +
            static_cast<std::size_t>(l) * (code_page_bytes + scale_page_bytes);
        std::vector<std::byte> sync_codes(code_page_bytes), sync_scales(scale_page_bytes);
        CHECK(cudaMemcpy(sync_codes.data(), static_cast<const std::byte*>(codes.data) + codes.nb[3],
                         code_page_bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
              "layer-ready sync reference codes layer=%d", l);
        CHECK(cudaMemcpy(sync_scales.data(), static_cast<const std::byte*>(scales.data) + scales.nb[3],
                         scale_page_bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
              "layer-ready sync reference scales layer=%d", l);
        CHECK(std::memcmp(result, sync_codes.data(), code_page_bytes) == 0 &&
                  std::memcmp(result + code_page_bytes, sync_scales.data(), scale_page_bytes) == 0,
              "layer-ready GPU output differs from synchronous kernel layer=%d", l);
        // Independent CPU dequant/Hadamard/RoPE oracle on a real 64-column tail.
        std::int8_t expected_codes[kDim];
        unsigned short expected_scales[kGroups];
        const int head = l % kHeads;
        const int token = (l * 5 + 13) % kTokens;
        const auto index = static_cast<std::size_t>(head * kTokens + token);
        expected_rephase(original_codes[l].data() + index * kDim,
                         original_scales[l].data() + index * kGroups,
                         from[0] + token, to[0] + token, expected_codes, expected_scales);
        for (int d = 0; d < kDim; ++d) {
            const auto output = static_cast<std::int8_t>(
                result[in_page_elems(kDim, head, token, d)]);
            CHECK(std::abs(int(output) - int(expected_codes[d])) <= 2,
                  "layer-ready CPU oracle layer=%d token=%d dim=%d", l, token, d);
        }
    }
    CHECK(ready.epoch() == 1 && !ready.active(), "layer-ready epoch/retirement");

    // Simulate a cancelled prefill: its producer has already been submitted,
    // but no layer consumer is allowed to use or release the target. Drain
    // both streams before dropping the pinned metadata / target lifetime.
    LayerReadyFence abandoned;
    abandoned.begin(device.stream);
    for (int l = 0; l < layers; ++l) { events[l] = abandoned.ready_event(l); }
    kvmem_rerope_pages_layer_ready(pool, sync_page.data(), to.data(), from.data(), 1,
                                   workspace, events, abandoned.producer_stream());
    for (int l = 0; l < layers; ++l) { abandoned.mark_ready(l); }
    abandoned.drain_consumers();
    CHECK(!abandoned.active(), "layer-ready cancellation must drain queued GPU producer");

    // A validation failure must never record a ready event or publish a page.
    LayerReadyFence rejected;
    rejected.begin(device.stream);
    for (int l = 0; l < layers; ++l) { events[l] = rejected.ready_event(l); }
    const std::array<std::int32_t, 1> invalid_page{-1};
    bool refused = false;
    try {
        kvmem_rerope_pages_layer_ready(pool, invalid_page.data(), from.data(), to.data(), 1,
                                       workspace, events, rejected.producer_stream());
    } catch (const std::invalid_argument&) { refused = true; }
    rejected.drain_consumers();
    CHECK(refused && !rejected.active(),
          "layer-ready failed prepare must remain uncommitted and drained");

    // Drive the actual KVAddressSpaceStore producer route, including real
    // block selection/re-mapping and the generation-bound prefill ticket.
    KVExecutionTablePool tables({backing.p, backing.bytes}, tables_layout, pool);
    LogicalKVPageStore pages(pool, physical_pages);
    DeviceBuffer store_metadata(3 * physical_pages * sizeof(std::int32_t));
    KVAddressSpaceStore store(pages, tables, 1, physical_pages,
                             KvMemOptions{.enabled = true},
                             {store_metadata.p, store_metadata.bytes});
    const auto address = store.create_inactive();
    CHECK(address.has_value(), "layer-ready store address created");
    if (!address) { throw std::runtime_error("layer-ready address unavailable"); }
    store.activate(*address, physical_pages, 0);
    store.ensure_mapped_to_tokens(*address, 6 * kTokens);
    store.commit_frontier(*address, 6 * kTokens);
    store.deactivate(*address);
    const std::uint32_t compacted = store.kvmem_compact(
        *address, 4 * kTokens, device.stream, {}, false, 0, {}, std::nullopt, true);
    CHECK(compacted == 4 * kTokens, "layer-ready store compacted window=%u", compacted);
    auto* consumer = store.kvmem_layer_ready_for(*address);
    CHECK(consumer != nullptr, "layer-ready real store producer was not bound");
    if (consumer != nullptr) {
        const auto window = store.window_page_handles(*address);
        CHECK(window.size() == 4, "layer-ready store retained four pages");
        if (window.empty()) { throw std::logic_error("layer-ready selected no pages"); }
        const int physical = pool.physical_index_of(pages.physical(window.back()));
        PinnedHostBuffer store_reads(code_page_bytes * layers);
        for (int l = 0; l < layers; ++l) {
            consumer->wait_before_layer(l, device.stream);
            const auto plane = pool.plane(static_cast<std::size_t>(l) * 4);
            CHECK(cudaMemcpyAsync(static_cast<std::byte*>(store_reads.data()) +
                                      static_cast<std::size_t>(l) * code_page_bytes,
                                  static_cast<const std::byte*>(plane.data) +
                                      static_cast<std::size_t>(physical) * plane.nb[3],
                                  code_page_bytes, cudaMemcpyDeviceToHost, device.stream) == cudaSuccess,
                  "layer-ready store consumer D2H layer=%d", l);
            consumer->release_after_layer(l, device.stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    CHECK(store.kvmem_layer_ready_for(KVAddressSpaceHandle{}) == nullptr,
          "layer-ready stale/empty address cannot claim current ticket");
    // A second plan may select different blocks even with the same budget;
    // it must use the synchronous fallback and revoke the old event epoch.
    (void)store.kvmem_compact(*address, 4 * kTokens, device.stream);
    CHECK(store.kvmem_layer_ready_for(*address) == nullptr,
          "layer-ready previous ticket must be revoked before next compact");
    CHECK(store.kvmem_inflight_pages() == 0,
          "layer-ready store must retire all inflight physical pages");
    std::printf("layer-ready Main16 GPU fence+CPU oracle+store compact finished\n");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(4242);
    if (argc == 2 && std::strcmp(argv[1], "--layer-ready") == 0) {
        try { test_layer_ready(); }
        catch (const std::exception& e) {
            std::printf("layer-ready exception: %s\n", e.what());
            return 2;
        }
        std::printf("layer-ready failures=%d  VERDICT: %s\n", failures,
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }
    const bool claim_prefetch = argc == 2 && std::strcmp(argv[1], "--claim-prefetch") == 0;
    if (claim_prefetch || (argc == 2 && std::strcmp(argv[1], "--claim") == 0)) {
        try {
            test_generation_device_claim(claim_prefetch);
        } catch (const std::exception& error) {
            std::printf("claim exception: %s\n", error.what());
            return 2;
        }
        std::printf("claim failures=%d  VERDICT: %s\n", failures,
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }
    const bool checkpoint = argc == 2 && std::strcmp(argv[1], "--checkpoint") == 0;
    // Exercise the real Cold read-ahead owner on a small native page geometry.
    // Keep initial demotions synchronous so the same independent K/V oracle
    // establishes a stable archived source before enabling the async route.
    const bool cold_prefetch = argc == 2 && std::strcmp(argv[1], "--cold-prefetch") == 0;
    const bool exercise_cold = checkpoint || cold_prefetch ||
                               (argc == 2 && std::strcmp(argv[1], "--cold") == 0);
    const bool cancel_async = argc == 2 && std::strcmp(argv[1], "--async-cancel") == 0;
    const bool exercise_async = cancel_async ||
                                (argc == 2 && std::strcmp(argv[1], "--async") == 0);
    if (exercise_cold) {
        // One Host page forces the remaining archived pages through the disk
        // ladder; the same CPU oracle below verifies their K/V after restore.
#if defined(_WIN32)
        CHECK(::_putenv_s("NINFER_KVMEM_HOST_PAGES", "1") == 0, "set Host cap");
        CHECK(::_putenv_s("NINFER_KVMEM_COLD_DIR", ".") == 0, "set cold directory");
#else
        CHECK(::setenv("NINFER_KVMEM_HOST_PAGES", "1", 1) == 0, "set Host cap");
        CHECK(::setenv("NINFER_KVMEM_COLD_DIR", ".", 1) == 0, "set cold directory");
#endif
    }
    try {
        DeviceContext device;
        CopyGate copy_gate;
        KVPageGeometry geometry;
        geometry.page_tokens        = kTokens;
        geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < kLayers; ++layer) {
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
        }

        LayoutBuilder builder;
        const DeviceKVPagePoolLayout pool_layout =
            plan_device_kv_page_pool(builder, DeviceKVPagePoolSpec{.page_group_count = kPages,
                                                                   .geometry = geometry});
        const KVExecutionTableLayout tables_layout =
            plan_kv_execution_tables(builder, KVExecutionTableSpec{.logical_page_capacity = kPages,
                                                                   .table_rows = 1});
        const std::size_t backing_bytes =
            pool_layout.payload_bytes() + tables_layout.metadata_bytes() + (1u << 20);
        void* backing = nullptr;
        CHECK(cudaMalloc(&backing, backing_bytes) == cudaSuccess, "cudaMalloc backing");

        DeviceKVPagePool pool(DeviceSpan{backing, backing_bytes}, pool_layout);
        KVExecutionTablePool tables(DeviceSpan{backing, backing_bytes}, tables_layout, pool);
        LogicalKVPageStore pages(pool, kPages);
        // The store takes its window explicitly. The process-wide switch is resolved by the
        // front ends, so a direct construction has to ask for KVMem itself.
        DeviceBuffer rerope_meta(3ULL * kPages * sizeof(std::int32_t));
        KVAddressSpaceStore store(pages, tables, 2, kPages,
                                  ninfer::KvMemOptions{.enabled = true},
                                  {rerope_meta.p, rerope_meta.bytes});
        if (exercise_async) { store.enable_kvmem_async_d2h(device); }
        std::printf("stores ok\n");

        // ---- activate + map 6 pages ----
        std::optional<KVAddressSpaceHandle> handle = store.create_inactive();
        CHECK(handle.has_value(), "create_inactive");
        store.activate(*handle, kPages, 0);
        store.ensure_mapped_to_tokens(*handle, 6 * kTokens);
        store.commit_frontier(*handle, 6 * kTokens);
        std::printf("mapped 6 pages\n");

        // ---- physical plane handles ----
        std::vector<Tensor> planes;
        std::vector<std::int64_t> plane_strides(static_cast<std::size_t>(kLayers * 4));
        for (int p = 0; p < kLayers * 4; ++p) {
            planes.push_back(pool.plane(static_cast<std::size_t>(p)));
            plane_strides[static_cast<std::size_t>(p)] = elems_per_page(
                geometry.planes[static_cast<std::size_t>(p)].leading_extent);
        }
        auto phys_of_slot = [&](int slot) {
            const std::vector<LogicalKVPageHandle> logicals = store.window_page_handles(*handle);
            CHECK(static_cast<int>(logicals.size()) == 6, "expected 6 logicals, got %zu",
                  logicals.size());
            return pool.physical_index_of(pages.physical(logicals[static_cast<std::size_t>(slot)]));
        };

        // ---- fill: raw K row baked at the natural slot; deterministic fillers for V ----
        std::vector<float> raw(static_cast<std::size_t>(kPages) * kLayers * kHeads * kTokens *
                               kDim);
        for (auto& v : raw) { v = static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f; }
        // Simulated ledger: the K codes/scales the device SHOULD hold per block,
        // as baked at baked_now[blk]. Updated through the same chain the kernel
        // runs, so skip blocks compare bitwise and moved blocks within one
        // code step (GPU sincosf vs CPU sin/cos ulp noise).
        std::vector<int> baked_now(static_cast<std::size_t>(kPages), -1);
        std::vector<char> resident(static_cast<std::size_t>(kPages), 0);
        std::vector<std::int8_t> sim_codes(static_cast<std::size_t>(kPages) * kLayers * kHeads *
                                           kTokens * kDim);
        std::vector<unsigned short> sim_bits(static_cast<std::size_t>(kPages) * kLayers * kHeads *
                                             kTokens * kGroups);
        const auto sim_idx = [&](int blk, int l, int h, int t) {
            return ((static_cast<std::size_t>(blk) * kLayers + l) * kHeads + h) * kTokens + t;
        };

        std::vector<std::vector<std::uint8_t>> host_plane(static_cast<std::size_t>(kLayers * 4));
        std::vector<std::int64_t> plane_bytes_total(static_cast<std::size_t>(kLayers * 4));
        for (int p = 0; p < kLayers * 4; ++p) {
            const int leading = geometry.planes[static_cast<std::size_t>(p)].leading_extent;
            const int elem =
                geometry.planes[static_cast<std::size_t>(p)].dtype == DType::FP16 ? 2 : 1;
            const std::int64_t bytes = elems_per_page(leading) * elem * 6;
            plane_bytes_total[static_cast<std::size_t>(p)] = bytes;
            host_plane[static_cast<std::size_t>(p)].resize(static_cast<std::size_t>(bytes));
            auto* base = host_plane[static_cast<std::size_t>(p)].data();
            for (std::int64_t b = 0; b < bytes; ++b) {
                base[b] = static_cast<std::uint8_t>(p * 17 + static_cast<int>(b % 251)) * 31 + 7;
            }
        }
        for (int i = 0; i < 6; ++i) {
            const std::size_t kc_page = static_cast<std::size_t>(elems_per_page(256) * i);
            const std::size_t ks_page = static_cast<std::size_t>(elems_per_page(4) * i) * 2;
            for (int l = 0; l < kLayers; ++l) {
                const std::size_t kc = static_cast<std::size_t>(l * 4);
                const std::size_t ks = static_cast<std::size_t>(l * 4 + 2);
                for (int h = 0; h < kHeads; ++h) {
                    for (int t = 0; t < kTokens; ++t) {
                        std::int8_t codes[kDim];
                        unsigned short bits[kGroups];
                        // The engine convention: content is baked at the block's
                        // window slot (== ledger baked_pos).
                        expected_k_row(raw, i, l, h, t, i * kTokens + t, codes, bits);
                        auto* dst = reinterpret_cast<std::int8_t*>(
                                        host_plane[kc].data() + kc_page) +
                                    in_page_elems(256, h, t, 0);
                        for (int d = 0; d < kDim; ++d) { dst[d] = codes[d]; }
                        auto* ks_ptr = reinterpret_cast<unsigned short*>(
                                           host_plane[ks].data() + ks_page) +
                                       in_page_elems(4, h, t, 0);
                        for (int g = 0; g < kGroups; ++g) { ks_ptr[g] = bits[g]; }
                        const std::size_t si = sim_idx(i, l, h, t);
                        for (int d = 0; d < kDim; ++d) {
                            sim_codes[si * kDim + static_cast<std::size_t>(d)] = codes[d];
                        }
                        for (int g = 0; g < kGroups; ++g) {
                            sim_bits[si * kGroups + static_cast<std::size_t>(g)] = bits[g];
                        }
                    }
                }
            }
        }
        // Upload the 6 filled pages to their physical slots.
        for (int i = 0; i < 6; ++i) {
            const int phys = phys_of_slot(i);
            for (int p = 0; p < kLayers * 4; ++p) {
                const int leading = geometry.planes[static_cast<std::size_t>(p)].leading_extent;
                const int elem =
                    geometry.planes[static_cast<std::size_t>(p)].dtype == DType::FP16 ? 2 : 1;
                CHECK(cudaMemcpy(static_cast<std::uint8_t*>(planes[static_cast<std::size_t>(p)].data) +
                                     phys * plane_strides[static_cast<std::size_t>(p)] * elem,
                                 host_plane[static_cast<std::size_t>(p)].data() +
                                     static_cast<std::size_t>(plane_bytes_total[static_cast<std::size_t>(p)] / 6 * i),
                                 static_cast<std::size_t>(plane_bytes_total[static_cast<std::size_t>(p)] / 6),
                                 cudaMemcpyHostToDevice) == cudaSuccess,
                      "H2D page %d plane %d", i, p);
            }
        }
        std::printf("filled 6 pages\n");
        const auto source_codes = sim_codes;
        const auto source_bits = sim_bits;

        // Immutable K must survive working-page mutation and keep mixed-birth
        // tail columns. It must never overwrite V. The CPU rephase oracle
        // above independently decodes the represented initial INT8 boundary.
        {
            ninfer::kvmem::ImmutableKStore source(geometry);
            ninfer::kvmem::ImmutableKStore::Record record;
            const int physical = phys_of_slot(0);
            source.capture(record, pool, physical, 0, 17, device.stream);
            std::array<std::int32_t, 64> origins;
            for (int t = 0; t < kTokens; ++t) { origins[t] = t; }
            ninfer::kvmem::kvmem_rebuild_source_k(pool, physical, origins, 192, 64, device.stream);
            source.capture(record, pool, physical, 192, 64, device.stream);
            source.restore(record, pool, physical, 0, device.stream);
            for (int l = 0; l < kLayers; ++l) {
                const auto code_bytes = static_cast<std::size_t>(elems_per_page(256));
                const auto scale_bytes = static_cast<std::size_t>(elems_per_page(4) * 2);
                std::vector<std::int8_t> actual(code_bytes);
                std::vector<unsigned short> scales(scale_bytes / 2);
                CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<std::byte*>(planes[l*4].data) +
                    physical * code_bytes, code_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(scales.data(), static_cast<std::byte*>(planes[l*4+2].data) +
                    physical * scale_bytes, scale_bytes, cudaMemcpyDeviceToHost));
                for (int h = 0; h < kHeads; ++h) for (int t = 0; t < kTokens; ++t) {
                    const auto si = sim_idx(0,l,h,t);
                    std::int8_t intermediate[kDim], expected[kDim];
                    unsigned short middle_scales[kGroups], expected_scales[kGroups];
                    if (t < 17) {
                        std::copy_n(source_codes.data()+si*kDim,kDim,expected);
                        std::copy_n(source_bits.data()+si*kGroups,kGroups,expected_scales);
                    } else {
                        expected_rephase(source_codes.data()+si*kDim,source_bits.data()+si*kGroups,
                                          t,192+t,intermediate,middle_scales);
                        expected_rephase(intermediate,middle_scales,192+t,t,expected,expected_scales);
                    }
                    for (int d=0;d<kDim;++d) {
                        CHECK(std::abs(int(actual[in_page_elems(256,h,t,d)])-int(expected[d]))<=1,
                              "immutable mixed-column CPU oracle layer=%d h=%d t=%d d=%d",l,h,t,d);
                    }
                    for (int g=0;g<kGroups;++g) {
                        const auto actual_scale=__half2float(__ushort_as_half(scales[in_page_elems(4,h,t,g)]));
                        const auto expected_scale=__half2float(__ushort_as_half(expected_scales[g]));
                        CHECK(std::abs(actual_scale-expected_scale)<2e-5f,
                              "immutable source scale oracle layer=%d h=%d t=%d g=%d",l,h,t,g);
                    }
                }
                // Reset fixture; later cold rounds must start at its original boundary.
                for (int p=l*4;p<l*4+4;++p) {
                    const auto size=host_plane[p].size()/6;
                    CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(planes[p].data)+physical*size,
                                           host_plane[p].data(),size,cudaMemcpyHostToDevice));
                }
            }
            record.columns = 17; // rollback discards the appended source suffix
            source.capture(record,pool,physical,0,64,device.stream);
            CHECK(record.positions[16]==16 && record.positions[17]==17 &&
                      record.positions[63]==63,"immutable rollback must recapture new columns");
            source.release(record);
            CHECK(source.occupied_bytes()==0 && !record.backing.valid(),
                  "immutable source release reclaims backing record");
        }

        // Ledger snapshot before compaction: blocks 0..5 resident at their
        // natural slots.
        for (int i = 0; i < 6; ++i) {
            resident[static_cast<std::size_t>(i)]  = 1;
            baked_now[static_cast<std::size_t>(i)] = i * kTokens;
        }
        store.deactivate(*handle);
        std::printf("deactivated\n");
        const auto original_handles = store.window_page_handles(*handle);
        if (exercise_async) {
            CHECK(pages.can_destructive_truncate_inactive(original_handles[1], kTokens - 1),
                  "async: unpinned source initially permits a version change");
        }

        // ---- compact 6 pages -> 4 ----
        if (exercise_async) {
            CHECK(cudaLaunchHostFunc(device.transfer_stream, block_copy_until_released,
                                     &copy_gate) == cudaSuccess,
                  "async: queue controlled copy delay");
        }
        const std::uint32_t w1 = store.kvmem_compact(
            *handle, 4 * kTokens, exercise_async ? device.stream : nullptr);
        if (exercise_async) {
            CHECK(store.kvmem_inflight_pages() == 2, "async: two source pages remain in flight");
            CHECK(pool.allocated_pages() == 6, "async: source storage is not free before copy");
            CHECK(!pool.reserve(3),
                  "async: promised source releases are not allocatable free pages");
            std::uint32_t pinned = 0;
            for (const auto logical : original_handles) {
                if (pages.source_pins(logical) == 0) { continue; }
                ++pinned;
                CHECK(!pages.can_release_reference(logical, false),
                      "async: in-flight source reference cannot be released");
                const auto epoch = pages.content_epoch(logical);
                bool refused = false;
                try { pages.destructive_truncate_inactive(logical, kTokens - 1); }
                catch (const std::logic_error&) { refused = true; }
                CHECK(refused && pages.content_epoch(logical) == epoch,
                      "async: source pin must prevent a stale epoch publication");
            }
            CHECK(pinned == 2 && store.kvmem_usage().host_bytes >=
                                      2 * plan_host_kv_page_layout(geometry).page_stride,
                  "async: source pins and destination quota remain owned");
            CHECK(!store.retire_kvmem_transfers(), "async: incomplete batch cannot publish");
            if (cancel_async) {
                CHECK(store.release(*handle), "async cancel: release inactive address");
                CHECK(copy_gate.finished.load(std::memory_order_acquire),
                      "async cancel: address release must drain queued access");
                CHECK(pool.allocated_pages() == 0 && store.kvmem_usage().host_bytes == 0 &&
                          store.kvmem_inflight_pages() == 0,
                      "async cancel: held sources and destinations must be reclaimed");
                const auto next = store.create_inactive();
                CHECK(next.has_value() && !store.valid(*handle),
                      "async cancel: reused address must invalidate old generation");
                if (next) {
                    store.activate(*next, 1, 0);
                    store.ensure_mapped_to_tokens(*next, kTokens, device.stream);
                    store.commit_frontier(*next, kTokens);
                    store.deactivate(*next);
                    CHECK(store.release(*next), "async cancel: next request can release");
                }
                device.synchronize();
                CHECK(cudaFree(backing) == cudaSuccess, "async cancel: free backing");
                std::printf("async cancel failures=%d VERDICT: %s\n", failures,
                            failures == 0 ? "PASS" : "FAIL");
                return failures == 0 ? 0 : 1;
            }
            copy_gate.open.store(true, std::memory_order_release);
            CHECK(store.retire_kvmem_transfers(true), "async: retire after copy completion");
            CHECK(store.kvmem_inflight_pages() == 0 && pool.allocated_pages() == 4,
                  "async: completed batch releases its held source pages");
            device.synchronize();
        }
        if (checkpoint) {
            const auto path = std::filesystem::temp_directory_path() /
                ("ninfer-prefix-roundtrip-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            store.activate(*handle,kPages,0);
            {
                kvmem::PrefixCheckpointWriter file(path.string());
                store.save_benchmark_prefix(*handle,file.stream(),device.stream);file.finish();
            }
            const auto previous=*handle;
            store.deactivate(*handle);
            CHECK(store.release(*handle), "checkpoint: release original complete owner");
            handle=store.create_inactive();
            if(!handle) throw std::runtime_error("checkpoint: fresh address unavailable");
            store.activate(*handle,kPages,0);
            {
                kvmem::PrefixCheckpointReader file(path.string());
                store.load_benchmark_prefix(*handle,file.stream(),device.stream);file.finish();
            }
            CHECK(!store.valid(previous), "checkpoint: old generation remains invalid");
            const auto second=path.string()+".roundtrip";
            {
                kvmem::PrefixCheckpointWriter file(second);
                store.save_benchmark_prefix(*handle,file.stream(),device.stream);file.finish();
            }
            std::ifstream a(path,std::ios::binary),b(second,std::ios::binary);
            const std::vector<char> before((std::istreambuf_iterator<char>(a)),{});
            const std::vector<char> after((std::istreambuf_iterator<char>(b)),{});
            CHECK(before==after, "checkpoint: all native/raw-K/metadata bytes across generation");
            a.close();b.close();
            { // A real corrupted payload must be rejected before a store load.
                std::fstream f(second,std::ios::binary|std::ios::in|std::ios::out);
                f.seekp(32);const char bad=static_cast<char>(after.at(32)^1);f.write(&bad,1);
            }
            bool corrupt=false;
            try { kvmem::PrefixCheckpointReader file(second); }
            catch(const kvmem::SnapshotCorrupt&) { corrupt=true; }
            CHECK(corrupt,"checkpoint: corrupted payload rejected before mutation");
            std::filesystem::remove(path);std::filesystem::remove(second);
            std::filesystem::remove(path.string()+".xxh3idx");
            std::filesystem::remove(second+".xxh3idx");
            store.deactivate(*handle);
            std::printf("checkpoint native/Host/Cold/rawK generation roundtrip complete\n");
        }
        CHECK(w1 == 4 * kTokens, "compact window = %u (want 256)", w1);
        std::printf("compact #1 -> window %u tokens\n", w1);

        // Per-page download helpers (plane dtype aware).
        auto download_page = [&](int plane, int phys) {
            const int leading = geometry.planes[static_cast<std::size_t>(plane)].leading_extent;
            const int elem =
                geometry.planes[static_cast<std::size_t>(plane)].dtype == DType::FP16 ? 2 : 1;
            std::vector<std::uint8_t> buf(
                static_cast<std::size_t>(elems_per_page(leading) * elem));
            CHECK(cudaMemcpy(buf.data(),
                             static_cast<std::uint8_t*>(planes[static_cast<std::size_t>(plane)].data) +
                                 phys * plane_strides[static_cast<std::size_t>(plane)] * elem,
                             buf.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "D2H plane %d", plane);
            return buf;
        };

        // Verify the compacted window: slot s holds block sel[s]; V planes
        // bitwise untouched; skipped K blocks bitwise untouched; moved K blocks
        // re-baked from the stored quantized values through the SAME H256
        // rotate chain the kernel runs.
        auto verify_window = [&](const int* sel, int window_pages) {
            const std::vector<LogicalKVPageHandle> logicals = store.window_page_handles(*handle);
            CHECK(static_cast<int>(logicals.size()) == window_pages, "logicals %zu (want %d)",
                  logicals.size(), window_pages);
            for (int s = 0; s < window_pages; ++s) {
                const int blk = sel[s];
                // Skip is a WHOLE-BLOCK property: resident before this round AND
                // already sitting at the target slot. Compute it once -- the
                // baked_now update below would poison tokens 1..63 otherwise.
                const bool blk_skip = resident[static_cast<std::size_t>(blk)] != 0 &&
                                      baked_now[static_cast<std::size_t>(blk)] == s * kTokens;
                const int baked_pre = baked_now[static_cast<std::size_t>(blk)];
                std::vector<std::int8_t> new_codes;
                std::vector<unsigned short> new_bits;
                if (!blk_skip) {
                    new_codes.resize(sim_codes.size());
                    new_bits.resize(sim_bits.size());
                }
                const int phys =
                    pool.physical_index_of(pages.physical(logicals[static_cast<std::size_t>(s)]));
                for (int l = 0; l < kLayers; ++l) {
                    // V codes and V scales: bitwise vs the block's original host fill.
                    const auto vcd = download_page(l * 4 + 1, phys);
                    const std::size_t v_src =
                        static_cast<std::size_t>(elems_per_page(256) * blk);
                    CHECK(std::equal(vcd.begin(), vcd.end(),
                                     host_plane[static_cast<std::size_t>(l * 4 + 1)].begin() +
                                         v_src),
                          "v-codes slot %d layer %d bitwise", s, l);
                    const auto vsd = download_page(l * 4 + 3, phys);
                    const std::size_t vs_src =
                        static_cast<std::size_t>(elems_per_page(4) * 2 * blk);
                    CHECK(std::equal(vsd.begin(), vsd.end(),
                                     host_plane[static_cast<std::size_t>(l * 4 + 3)].begin() +
                                         vs_src),
                          "v-scales slot %d layer %d bitwise", s, l);

                    const auto kcd = download_page(l * 4, phys);
                    const auto ksd = download_page(l * 4 + 2, phys);
                    for (int h = 0; h < kHeads; ++h) {
                        for (int t = 0; t < kTokens; ++t) {
                            const std::size_t si = sim_idx(blk, l, h, t);
                            std::int8_t want_codes[kDim];
                            unsigned short want_bits[kGroups];
                            if (blk_skip) {
                                for (int d = 0; d < kDim; ++d) {
                                    want_codes[d] =
                                        sim_codes[si * kDim + static_cast<std::size_t>(d)];
                                }
                                for (int g = 0; g < kGroups; ++g) {
                                    want_bits[g] =
                                        sim_bits[si * kGroups + static_cast<std::size_t>(g)];
                                }
                            } else {
                                const bool cold = resident[static_cast<std::size_t>(blk)] == 0;
                                const auto* codes = cold ? source_codes.data() : sim_codes.data();
                                const auto* bits = cold ? source_bits.data() : sim_bits.data();
                                const int from = cold ? blk*kTokens : baked_pre;
                                if (cold && from==s*kTokens) {
                                    std::copy_n(codes+si*kDim,kDim,want_codes);
                                    std::copy_n(bits+si*kGroups,kGroups,want_bits);
                                } else {
                                    expected_rephase(codes+si*kDim,bits+si*kGroups,
                                                     from+t,s*kTokens+t,want_codes,want_bits);
                                }
                                for (int d = 0; d < kDim; ++d) {
                                    new_codes[si * kDim + static_cast<std::size_t>(d)] =
                                        want_codes[d];
                                }
                                for (int g = 0; g < kGroups; ++g) {
                                    new_bits[si * kGroups + static_cast<std::size_t>(g)] =
                                        want_bits[g];
                                }
                            }

                            for (int g = 0; g < kGroups; ++g) {
                                const std::size_t scale_elem =
                                    static_cast<std::size_t>(in_page_elems(4, h, t, g));
                                const unsigned short got_bits =
                                    static_cast<unsigned short>(ksd[scale_elem * 2]) |
                                    static_cast<unsigned short>(ksd[scale_elem * 2 + 1]) << 8;
                                const float got_scale  = __half2float(__ushort_as_half(got_bits));
                                const float want_scale = __half2float(__ushort_as_half(want_bits[g]));
                                if (blk_skip) {
                                    CHECK(got_bits == want_bits[g],
                                          "skip scale bitwise s%d l%d h%d t%d g%d", s, l, h, t, g);
                                } else {
                                    CHECK(std::fabs(got_scale - want_scale) <=
                                              1e-3f * want_scale + 1e-7f,
                                          "scale s%d l%d h%d t%d g%d", s, l, h, t, g);
                                }
                                for (int i = 0; i < 64; ++i) {
                                    const int d = g * 64 + i;
                                    const std::int8_t got_code = static_cast<std::int8_t>(
                                        kcd[static_cast<std::size_t>(in_page_elems(256, h, t, d))]);
                                    if (blk_skip) {
                                        CHECK(got_code == want_codes[d],
                                              "skip code bitwise s%d l%d h%d t%d d%d", s, l, h, t,
                                              d);
                                    } else {
                                        const float got  = static_cast<float>(got_code) * got_scale;
                                        const double diff = std::fabs(
                                            static_cast<double>(got) -
                                            static_cast<double>(want_codes[d]) * want_scale);
                                        CHECK(diff <= 1.01 * static_cast<double>(got_scale),
                                              "code s%d l%d h%d t%d d%d", s, l, h, t, d);
                                    }
                                }
                            }
                        }
                    }
                }
                // Commit this slot's pending updates (the chain read the
                // pre-round sim throughout the loops above).
                if (!blk_skip) {
                    baked_now[static_cast<std::size_t>(blk)] = s * kTokens;
                    for (int l = 0; l < kLayers; ++l) {
                        for (int h = 0; h < kHeads; ++h) {
                            for (int t = 0; t < kTokens; ++t) {
                                const std::size_t si = sim_idx(blk, l, h, t);
                                for (int d = 0; d < kDim; ++d) {
                                    sim_codes[si * kDim + static_cast<std::size_t>(d)] =
                                        new_codes[si * kDim + static_cast<std::size_t>(d)];
                                }
                                for (int g = 0; g < kGroups; ++g) {
                                    sim_bits[si * kGroups + static_cast<std::size_t>(g)] =
                                        new_bits[si * kGroups + static_cast<std::size_t>(g)];
                                }
                            }
                        }
                    }
                }
            }
        };

        // Ledger expectation (budget 4 blocks, sink 1 / recent 1, quota ties
        // newest-first): selection {0,3,4,5} -- 0 is the sink page (stays at
        // slot 0), 3/4/5 move down, {1,2} are evicted to the host arena.
        int sel1[4] = {0, 3, 4, 5};
        verify_window(sel1, 4);
        CHECK(store.kvmem_usage().stage_out_blocks == 2 &&
                  (exercise_cold ? store.kvmem_cold_bytes() >=
                                       plan_host_kv_page_layout(geometry).page_stride
                                 : store.kvmem_usage().transfer_runs == 1),
              "compact #1 stage-out wrong: blocks=%llu runs=%llu cold_bytes=%zu",
              static_cast<unsigned long long>(store.kvmem_usage().stage_out_blocks),
              static_cast<unsigned long long>(store.kvmem_usage().transfer_runs),
              store.kvmem_cold_bytes());
        resident[1] = 0; resident[2] = 0;  // evicted by compact #1
        for (int s = 0; s < 4; ++s) { baked_now[static_cast<std::size_t>(sel1[s])] = s * kTokens; }

        // ---- round trip: force a host block back in via the mandatory hook ----
        // mandatory={1}: block 1 (evicted by compact #1) must be re-selected.
        // Expected selection {0,1,4,5}: 1 restored from the host arena and
        // re-phased 64->64 (identity rotation, but a full requantize pass);
        // 2 and 3 are newly evicted.
        if (exercise_async) {
            copy_gate.open.store(false, std::memory_order_release);
            copy_gate.finished.store(false, std::memory_order_release);
            CHECK(cudaLaunchHostFunc(device.transfer_stream, block_copy_until_released,
                                     &copy_gate) == cudaSuccess,
                  "async: delay required H2D restore");
        }
        const std::uint32_t w2 = store.kvmem_compact(*handle, 4 * kTokens, exercise_async ? device.stream : nullptr, {1});
        if (exercise_async) {
            CHECK(copy_gate.finished.load(std::memory_order_acquire) &&
                      store.kvmem_usage().async_restore_submitted == 1,
                  "async: required history must finish H2D before compact returns");
            CHECK(store.kvmem_usage().async_wait_us > 0,
                  "async: uncovered required-page delay must be reported");
            copy_gate.open.store(true, std::memory_order_release);
        }
        CHECK(w2 == 4 * kTokens, "compact #2 window = %u", w2);
        int sel2[4] = {0, 1, 4, 5};
        verify_window(sel2, 4);
        CHECK(store.kvmem_usage().stage_in_blocks == 1 && store.kvmem_usage().stage_out_blocks == 3,
              "compact #2 stage counts wrong: in=%llu out=%llu",
              static_cast<unsigned long long>(store.kvmem_usage().stage_in_blocks),
              static_cast<unsigned long long>(store.kvmem_usage().stage_out_blocks));
        resident[1] = 1; resident[2] = 0; resident[3] = 0;  // compact #2 diff
        for (int s = 0; s < 4; ++s) { baked_now[static_cast<std::size_t>(sel2[s])] = s * kTokens; }

        // ---- compact #3: restore two adjacent host blocks {2, 3} in one H2D run ----
        // Blocks 2 and 3 sit at adjacent host arena pages (pages 1 and 2), and
        // releasing {1, 4} leaves a contiguous free device run [1..4], so the
        // 2-page stage-in merges into 1 H2D run (plus 2 D2H runs for non-adjacent
        // device pages 1 and 4 -> 3 runs total instead of 4).
        const std::uint64_t runs_before_c3 = store.kvmem_usage().transfer_runs;
        const std::uint32_t w3 = store.kvmem_compact(*handle, 4 * kTokens, exercise_async ? device.stream : nullptr, {2, 3});
        CHECK(w3 == 4 * kTokens, "compact #3 window = %u", w3);
        int sel3[4] = {0, 2, 3, 5};
        verify_window(sel3, 4);
        const std::uint64_t c3_runs = store.kvmem_usage().transfer_runs - runs_before_c3;
        CHECK(store.kvmem_usage().stage_in_blocks == 3 &&
                  store.kvmem_usage().stage_out_blocks == 5 &&
                  (exercise_cold ? store.kvmem_cold_bytes() > 0 : c3_runs == 3),
              "compact #3 stage-in unexpected: in=%llu out=%llu c3_runs=%llu cold_bytes=%zu",
              static_cast<unsigned long long>(store.kvmem_usage().stage_in_blocks),
              static_cast<unsigned long long>(store.kvmem_usage().stage_out_blocks),
              static_cast<unsigned long long>(c3_runs), store.kvmem_cold_bytes());

        if (exercise_cold) {
            resident[1] = 0;
            resident[2] = 1;
            resident[3] = 1;
            resident[4] = 0;
            for (int s = 0; s < 4; ++s) {
                baked_now[static_cast<std::size_t>(sel3[s])] = s * kTokens;
            }

            // Block 1 currently owns the only Host slot. Retain blocks 2
            // and 4 instead: the executor must move block 1 Host->NVMe
            // before it can archive block 3, then bring block 4 back from
            // NVMe. Keeping block 2 at slot 1 avoids a third K re-phase that
            // would exceed this test's existing one-LSB numerical oracle.
            const std::uint64_t spills_before = store.kvmem_host_to_cold_spills();
            if (cold_prefetch) { store.enable_kvmem_async_d2h(device); }
            CHECK(store.kvmem_compact(*handle, 4 * kTokens,
                                      cold_prefetch ? device.stream : nullptr, {2, 4}) == 4 * kTokens,
                  "compact #4 cold restore");
            int sel4[4] = {0, 2, 4, 5};
            std::printf("verifying cold compact #4\n");
            verify_window(sel4, 4);
            CHECK(store.kvmem_host_to_cold_spills() > spills_before,
                  "Host cache pressure must spill the old page to NVMe");
            resident[3] = 0;
            resident[4] = 1;
            for (int s = 0; s < 4; ++s) {
                baked_now[static_cast<std::size_t>(sel4[s])] = s * kTokens;
            }

            // Re-select the page that was Host-backed, then spilled to disk;
            // verify its recovered K/V with the independent CPU oracle.
            CHECK(store.kvmem_compact(*handle, 4 * kTokens,
                                      (exercise_async || cold_prefetch) ? device.stream : nullptr, {1}) == 4 * kTokens,
                  "compact #5 former-Host cold restore");
            int sel5[4] = {0, 1, 4, 5};
            std::printf("verifying cold compact #5\n");
            verify_window(sel5, 4);
            if (cold_prefetch) {
                CHECK(store.kvmem_usage().nvme_prefetch_submitted >= 2 &&
                          store.kvmem_usage().nvme_prefetch_submitted ==
                              store.kvmem_usage().nvme_prefetch_completed,
                      "cold read-ahead did not retire its real Cold page owners");
                CHECK(store.retire_kvmem_transfers(true) || store.kvmem_inflight_pages() == 0,
                      "cold read-ahead left unretired H2D/D2H work");
            }
        }

        if (exercise_async) {
            // A later batch must not inherit the first batch's ready state.
            CHECK(store.release(*handle), "async: finish the first address before reuse");
            handle = store.create_inactive();
            if (!handle) { throw std::runtime_error("async: address reuse failed"); }
            store.activate(*handle, kPages, 0);
            store.ensure_mapped_to_tokens(*handle, 6 * kTokens, device.stream);
            store.commit_frontier(*handle, 6 * kTokens);
            copy_gate.open.store(false, std::memory_order_release);
            copy_gate.finished.store(false, std::memory_order_release);
            CHECK(cudaLaunchHostFunc(device.transfer_stream, block_copy_until_released,
                                     &copy_gate) == cudaSuccess,
                  "async: delay the second batch independently");
            CHECK(store.kvmem_compact(*handle, 4 * kTokens, device.stream) == 4 * kTokens,
                  "async: second batch compacts");
            CHECK(store.kvmem_inflight_pages() == 2 && !store.retire_kvmem_transfers(),
                  "async: old completion must not retire the new batch");
            // Six allocated + two reserved fill this pool. Re-establishing
            // four growth pages must wait for actual release, not spend a promise.
            store.resize_entitlement(*handle, kPages);
            CHECK(copy_gate.finished.load(std::memory_order_acquire) &&
                      pool.allocated_pages() == 4 && pool.reserved_pages() == 4 &&
                      store.kvmem_inflight_pages() == 0 &&
                      store.kvmem_usage().async_completed ==
                          store.kvmem_usage().async_submitted,
                  "async: full pool must harvest before reserving growth");
            copy_gate.open.store(true, std::memory_order_release);
            store.deactivate(*handle);
        }
        CHECK(store.release(*handle), "release KVMem address");
        CHECK(store.kvmem_cold_bytes() == 0, "address release reclaims cold records");
        CHECK(pool.allocated_pages() == 0 && pool.reserved_pages() == 0 &&
                  store.kvmem_usage().host_bytes == 0 && store.kvmem_inflight_pages() == 0,
              "address release must reclaim all Device, Host and in-flight storage");

        // Reused prefill may map an empty growth tail before reselection.
        // It must never become a committed interior page or checkpoint coverage.
        handle = store.create_inactive();
        if (!handle) { throw std::runtime_error("unwritten-tail address creation failed"); }
        store.activate(*handle, kPages, 0);
        store.ensure_mapped_to_tokens(*handle, 7 * kTokens, device.stream);
        store.commit_frontier(*handle, 5 * kTokens);
        const auto written_window = store.kvmem_compact(
            *handle, 4 * kTokens, exercise_async ? device.stream : nullptr);
        CHECK(written_window == 4 * kTokens &&
                  store.committed_frontier(*handle) == written_window,
              "unwritten growth pages entered the committed window");
        (void)store.retire_kvmem_transfers(true);
        store.set_checkpoint_requirement(*handle, written_window);
        store.deactivate(*handle);
        CHECK(store.release(*handle) && pool.allocated_pages() == 0 &&
                  pool.reserved_pages() == 0,
              "unwritten-tail checkpoint leaked pages or reservation");
        std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
        cudaFree(backing);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
