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

        // Ledger snapshot before compaction: blocks 0..5 resident at their
        // natural slots.
        for (int i = 0; i < 6; ++i) {
            resident[static_cast<std::size_t>(i)]  = 1;
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
                                expected_rephase(&sim_codes[si * kDim], &sim_bits[si * kGroups],
                                                 baked_pre + t, s * kTokens + t, want_codes,
                                                 want_bits);
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
        CHECK(store.kvmem_usage().stage_out_blocks == 2 && store.kvmem_usage().transfer_runs == 1,
              "compact #1 stage-out did not merge consecutive pages: blocks=%llu runs=%llu",
              static_cast<unsigned long long>(store.kvmem_usage().stage_out_blocks),
              static_cast<unsigned long long>(store.kvmem_usage().transfer_runs));
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
        const std::uint32_t w3 = store.kvmem_compact(*handle, 4 * kTokens, nullptr, {2, 3});
        CHECK(w3 == 4 * kTokens, "compact #3 window = %u", w3);
        int sel3[4] = {0, 2, 3, 5};
        verify_window(sel3, 4);
        const std::uint64_t c3_runs = store.kvmem_usage().transfer_runs - runs_before_c3;
        CHECK(store.kvmem_usage().stage_in_blocks == 3 &&
                  store.kvmem_usage().stage_out_blocks == 5 && c3_runs == 3,
              "compact #3 stage-in did not merge adjacent host pages: in=%llu out=%llu c3_runs=%llu",
              static_cast<unsigned long long>(store.kvmem_usage().stage_in_blocks),
              static_cast<unsigned long long>(store.kvmem_usage().stage_out_blocks),
              static_cast<unsigned long long>(c3_runs));

        std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
        cudaFree(backing);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
