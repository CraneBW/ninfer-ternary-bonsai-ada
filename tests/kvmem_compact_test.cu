// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// Store-level K1b compaction test: a real LogicalKVPageStore +
// KVAddressSpaceStore over a real page pool, driving the lifecycle exactly
// like the runtime (activate -> ensure -> commit -> deactivate -> compact),
// then verifying the compacted window byte-for-byte where position-free and
// against a CPU re-phase model for group 0.
//
// Run: builder.exe -test tests/kvmem_compact_test.cu

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_blocks.h"
#include "kvmem/kvmem_rerope.cuh"
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

void hadamard64_warp(float* e) {
    for (int bit = 1; bit < 32; bit <<= 1) {
        float a[32], b[32];
        for (int l = 0; l < 32; ++l) {
            const int p   = l ^ bit;
            const bool hi = (l & bit) != 0;
            a[l] = hi ? e[p] - e[l] : e[l] + e[p];
            b[l] = hi ? e[p + 32] - e[l + 32] : e[l + 32] + e[p + 32];
        }
        for (int l = 0; l < 32; ++l) {
            e[l]      = a[l];
            e[l + 32] = b[l];
        }
    }
    for (int l = 0; l < 32; ++l) {
        const float A = e[l];
        const float B = e[l + 32];
        e[l]      = (A + B) * 0.125f;
        e[l + 32] = (A - B) * 0.125f;
    }
}

void rope_group0(float* e, std::int32_t position, bool inverse) {
    // Text MRoPE: all three axes carry the same position.
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(position) * host_freq(p);
        float s = std::sin(phi);
        const float c = std::cos(phi);
        if (inverse) { s = -s; }
        const float a = e[p];
        const float b = e[p + 32];
        e[p]      = a * c - b * s;
        e[p + 32] = b * c + a * s;
    }
}

// Quantize a rotated group the way the append path does.
void quantize_group(const float* g, std::int8_t* codes, unsigned short* scale_bits) {
    float absmax = 0.0f;
    for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(g[i])); }
    const unsigned short bits =
        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
    const float s = __half2float(__ushort_as_half(bits));
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < 64; ++i) {
        const int q = std::max(-127, std::min(127, static_cast<int>(std::lrintf(g[i] * inv))));
        codes[i] = static_cast<std::int8_t>(q);
    }
    *scale_bits = bits;
}

// The stored group-0 model for block `page` baked at window position
// `baked`: Quant(H64(RoPE_baked(raw))).
void expected_group0(const std::vector<float>& raw, int page, int layer, int head, int token,
                     std::int32_t baked, std::int8_t* codes, unsigned short* scale_bits) {
    float g[64];
    for (int l = 0; l < 32; ++l) {
        g[l]      = raw[((static_cast<std::size_t>(page) * kLayers + layer) * kHeads + head) *
                            kTokens * 64 +
                        static_cast<std::size_t>(token) * 64 + l];
        g[l + 32] = raw[((static_cast<std::size_t>(page) * kLayers + layer) * kHeads + head) *
                            kTokens * 64 +
                        static_cast<std::size_t>(token) * 64 + l + 32];
    }
    rope_group0(g, baked, false);
    hadamard64_warp(g);
    quantize_group(g, codes, scale_bits);
}

// The kernel's float sequence applied to the STORED quantized group:
// dequant -> H64^-1 -> un-RoPE(from) -> RoPE(to) -> H64 -> quantize.
// `stored` holds the 64 group-0 codes of the block as baked at from_pos.
void expected_rephase(const std::int8_t* stored, unsigned short stored_bits,
                      std::int32_t from_pos, std::int32_t to_pos, std::int8_t* codes,
                      unsigned short* scale_bits) {
    float g[64];
    const float s = __half2float(__ushort_as_half(stored_bits));
    for (int l = 0; l < 32; ++l) {
        g[l]      = static_cast<float>(stored[l]) * s;
        g[l + 32] = static_cast<float>(stored[l + 32]) * s;
    }
    hadamard64_warp(g);
    rope_group0(g, from_pos, true);
    rope_group0(g, to_pos, false);
    hadamard64_warp(g);
    quantize_group(g, codes, scale_bits);
}

// Plane addressing (PageMajor, element units of the plane dtype).
std::int64_t elems_per_page(int leading) {
    return static_cast<std::int64_t>(leading) * kTokens * kHeads;
}
std::int64_t in_page_elems(int leading, int head, int token, int d) {
    return static_cast<std::int64_t>(leading) * (static_cast<std::int64_t>(head) * kTokens + token) +
           d;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(4242);
    try {
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
        KVAddressSpaceStore store(pages, tables, 2, kPages,
                                  ninfer::KvMemOptions{.enabled = true});
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

        // ---- fill: raw K group-0 baked at the natural slot; fillers elsewhere ----
        std::vector<float> raw(static_cast<std::size_t>(kPages) * kLayers * kHeads * kTokens * 64);
        for (auto& v : raw) { v = static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f; }
        // Simulated ledger: the group-0 the device SHOULD hold per block, as
        // baked at baked_now[blk]. Updated through the same chain the kernel
        // runs, so skip blocks compare bitwise and moved blocks within one
        // code step (GPU sincosf vs CPU sin/cos ulp noise).
        std::vector<int> baked_now(static_cast<std::size_t>(kPages), -1);
        std::vector<char> resident(static_cast<std::size_t>(kPages), 0);
        std::vector<std::int8_t> sim_codes(static_cast<std::size_t>(kPages) * kLayers * kHeads *
                                           kTokens * 64);
        std::vector<unsigned short> sim_bits(static_cast<std::size_t>(kPages) * kLayers * kHeads *
                                             kTokens);
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
                        std::int8_t codes[64];
                        unsigned short bits;
                        // The engine convention: content is baked at the block's
                        // window slot (== ledger baked_pos).
                        expected_group0(raw, i, l, h, t, i * kTokens + t, codes, &bits);
                        auto* dst = reinterpret_cast<std::int8_t*>(
                                        host_plane[kc].data() + kc_page) +
                                    in_page_elems(256, h, t, 0);
                        for (int d = 0; d < 64; ++d) { dst[d] = codes[d]; }
                        auto* ks_ptr = reinterpret_cast<unsigned short*>(
                            host_plane[ks].data() + ks_page) +
                                       in_page_elems(4, h, t, 0);
                        ks_ptr[0] = bits;
                        const std::size_t si = sim_idx(i, l, h, t);
                        for (int d = 0; d < 64; ++d) {
                            sim_codes[si * 64 + static_cast<std::size_t>(d)] = codes[d];
                        }
                        sim_bits[si] = bits;
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

        // Ledger snapshot before compaction: blocks 0..5 resident at their
        // natural slots.
        for (int i = 0; i < 6; ++i) {
            resident[static_cast<std::size_t>(i)] = 1;
            baked_now[static_cast<std::size_t>(i)] = i * kTokens;
        }
        store.deactivate(*handle);
        std::printf("deactivated\n");

        // ---- compact 6 pages -> 4 ----
        const std::uint32_t w1 = store.kvmem_compact(*handle, 4 * kTokens);
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

        // Verify the compacted window: slot s holds block sel[s]; V planes and
        // K groups 1..3 bitwise untouched; group 0 re-baked from the stored
        // quantized values through the SAME rotate chain the kernel runs (the
        // fresh-bake ideal differs by the double-quantization noise).
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
                // Per-slot pending updates: the chain must read the PRE-round
                // sim for every token -- updating inside the token loop would
                // poison tokens 1..63 with slot-0's post state.
                std::vector<std::int8_t> new_codes;
                std::vector<unsigned short> new_bits;
                if (!blk_skip) {
                    // Sized like the sim ledger: the index si embeds the block id.
                    new_codes.resize(sim_codes.size());
                    new_bits.resize(sim_bits.size());
                }
                const int phys =
                    pool.physical_index_of(pages.physical(logicals[static_cast<std::size_t>(s)]));
                for (int l = 0; l < kLayers; ++l) {
                    // V codes: bitwise vs the block's original host fill
                    // (the fill pattern carries a global byte offset, so the
                    // comparison must slice the source at the same page).
                    const auto vcd = download_page(l * 4 + 1, phys);
                    const std::size_t v_src =
                        static_cast<std::size_t>(elems_per_page(256) * blk);
                    CHECK(std::equal(vcd.begin(), vcd.end(),
                                     host_plane[static_cast<std::size_t>(l * 4 + 1)].begin() +
                                         v_src),
                          "v-codes slot %d layer %d bitwise", s, l);
                    const auto kcd = download_page(l * 4, phys);
                    const auto ksd = download_page(l * 4 + 2, phys);
                    for (int h = 0; h < kHeads; ++h) {
                        for (int t = 0; t < kTokens; ++t) {
                            // scale: group 0 fp16 at element (h*64+t)*4
                            const std::size_t scale_elem =
                                static_cast<std::size_t>(in_page_elems(4, h, t, 0));
                            const unsigned short got_bits =
                                static_cast<unsigned short>(ksd[scale_elem * 2]) |
                                static_cast<unsigned short>(ksd[scale_elem * 2 + 1]) << 8;
                            const float got_scale = __half2float(__ushort_as_half(got_bits));
                            std::int8_t want_codes[64];
                            unsigned short want_bits;
                            const std::size_t kc_page =
                                static_cast<std::size_t>(elems_per_page(256) * blk);
                            const std::size_t ks_page =
                                static_cast<std::size_t>(elems_per_page(4) * blk) * 2;
                            const std::int8_t* stored =
                                reinterpret_cast<const std::int8_t*>(
                                    host_plane[static_cast<std::size_t>(l * 4)].data() +
                                    kc_page) +
                                in_page_elems(256, h, t, 0);
                            const unsigned short stored_bits =
                                *reinterpret_cast<const unsigned short*>(
                                    host_plane[static_cast<std::size_t>(l * 4 + 2)].data() +
                                    ks_page) +
                                in_page_elems(4, h, t, 0);
                            const std::size_t si = sim_idx(blk, l, h, t);
                            if (blk_skip) {
                                // The kernel left this block untouched.
                                for (int d = 0; d < 64; ++d) {
                                    want_codes[d] = sim_codes[si * 64 + static_cast<std::size_t>(d)];
                                }
                                want_bits = sim_bits[si];
                            } else {
                                // CPU mirror of the kernel's float sequence,
                                // reading the PRE-round sim.
                                expected_rephase(&sim_codes[si * 64], sim_bits[si],
                                                 baked_pre + t, s * kTokens + t, want_codes,
                                                 &want_bits);
                                for (int d = 0; d < 64; ++d) {
                                    new_codes[si * 64 + static_cast<std::size_t>(d)] = want_codes[d];
                                }
                                new_bits[si] = want_bits;
                            }
                            const float want_scale = __half2float(__ushort_as_half(want_bits));
                            if (s == 1 && l == 0 && h == 0 && t == 1) {
                                std::printf("DBG2: got_scale=%.6f want_scale=%.6f "
                                            "from=%d sim0=%d\n",
                                            got_scale, want_scale,
                                            baked_now[static_cast<std::size_t>(blk)],
                                            static_cast<int>(sim_codes[si * 64]));
                                for (int d = 0; d < 6; ++d) {
                                    std::printf("  d%d got=%d want=%d\n", d,
                                                static_cast<int>(static_cast<std::int8_t>(
                                                    kcd[static_cast<std::size_t>(
                                                        in_page_elems(256, h, t, d))])),
                                                static_cast<int>(want_codes[d]));
                                }
                            }
                            CHECK(std::fabs(got_scale - want_scale) <=
                                      1e-3f * want_scale + 1e-7f,
                                  "scale s%d l%d h%d t%d", s, l, h, t);
                            for (int d = 0; d < 64; ++d) {
                                const float got = static_cast<float>(static_cast<std::int8_t>(
                                                      kcd[static_cast<std::size_t>(
                                                          in_page_elems(256, h, t, d))])) *
                                                  got_scale;
                                const double diff =
                                    std::fabs(static_cast<double>(got) - want_codes[d] * want_scale);
                                CHECK(diff <= 1.01 * static_cast<double>(got_scale),
                                      "code s%d l%d h%d t%d d%d", s, l, h, t, d);
                            }
                            // K groups 1..3: untouched vs the block's
                            // original host fill.
                            for (int d = 64; d < 256; ++d) {
                                const std::size_t byte_idx = static_cast<std::size_t>(
                                    in_page_elems(256, h, t, d));
                                const std::size_t src_idx =
                                    static_cast<std::size_t>(elems_per_page(256) * blk) +
                                    byte_idx;
                                if (kcd[byte_idx] != host_plane[static_cast<std::size_t>(l * 4)][src_idx]) {
                                    CHECK(false, "k g1-3 s%d l%d h%d t%d d%d", s, l, h, t, d);
                                    break;
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
                                for (int d = 0; d < 64; ++d) {
                                    sim_codes[si * 64 + static_cast<std::size_t>(d)] =
                                        new_codes[si * 64 + static_cast<std::size_t>(d)];
                                }
                                sim_bits[si] = new_bits[si];
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
        resident[1] = 0; resident[2] = 0;  // evicted by compact #1
        for (int s = 0; s < 4; ++s) { baked_now[static_cast<std::size_t>(sel1[s])] = s * kTokens; }

        // ---- round trip: force a host block back in via the mandatory hook ----
        // mandatory={1}: block 1 (evicted by compact #1) must be re-selected.
        // Expected selection {0,1,4,5}: 1 restored from the host arena and
        // re-phased 64->64 (identity rotation, but a full requantize pass);
        // 2 and 3 are newly evicted.
        const std::uint32_t w2 = store.kvmem_compact(*handle, 4 * kTokens, nullptr, {1});
        CHECK(w2 == 4 * kTokens, "compact #2 window = %u", w2);
        int sel2[4] = {0, 1, 4, 5};
        verify_window(sel2, 4);
        resident[1] = 1; resident[2] = 1; resident[3] = 0;  // compact #2 diff

        std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
        cudaFree(backing);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
