// Unit test for the K3 retrieval scorer (standalone; build with
// `builder.exe -test tests/kvmem_score_test.cu`).
//
// A real page pool is filled the way the sm_86 int8-group64 append path fills it
// (RoPE at the token's position, then the H256 quantization rotation, then
// per-group fp16 scale), and the captured query is the pre-RoPE Q the prefill
// schedule stores. The kernel's per-block cosine is compared against a CPU model
// of the same pipeline, so a wrong plane layout, scale index, GQA head mapping,
// quantization-rotation inverse, or un-rotation shows up as a mismatch.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_bridge.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::kvmem;

namespace {

constexpr int kPages    = 3;
constexpr int kHeads    = 4;   // KV heads
constexpr int kQHeads   = 24;  // GQA group size 6
constexpr int kLayers   = 2;
constexpr int kTokens   = kPagedKVPageSize;
constexpr int kDim      = kKVCacheInt8HeadDim;
constexpr int kGroups   = kKVCacheInt8Groups;
constexpr int kRotary   = 64;  // Text MRoPE rotary dims

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

// Plane-local element offsets: [head][token][dim] for codes, [head][token][group]
// for scales (page-major planes, see paged_kv_element_offset).
std::int64_t code_off(int head, int token, int d) {
    return static_cast<std::int64_t>(kDim) * (head * kTokens + token) + d;
}
std::int64_t scale_off(int head, int token, int group) {
    return static_cast<std::int64_t>(kGroups) * (head * kTokens + token) + group;
}
constexpr std::int64_t kCodePageStride  = kDim * kTokens * kHeads;
constexpr std::int64_t kScalePageStride = kGroups * kTokens * kHeads;

// Mirrors kTextRopeInvFrequency: theta^(-2*pair/64).
float host_freq(int pair) { return std::pow(1e7f, -2.0f * static_cast<float>(pair) / 64.0f); }

// The Text MRoPE rotation on the rotary dims: pairs (p, p + 32) for p < 32.
void rope_rotary(float* row, std::int32_t position, bool inverse) {
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(position) * host_freq(p);
        float sine      = std::sin(phi);
        const float cosi = std::cos(phi);
        if (inverse) { sine = -sine; }
        const float a = row[p];
        const float b = row[p + 32];
        row[p]        = a * cosi - b * sine;
        row[p + 32]   = b * cosi + a * sine;
    }
}

// The production H256 quantization rotation, mirrored from
// normalized_hadamard_d256_inplace: dims are indexed as d = lane + 32 * r. It is
// its own inverse, so the scorer reuses it to undo the rotation.
void hadamard256_host(float* e) {
    for (int stride = 1; stride <= 16; stride <<= 1) {
        for (int r = 0; r < 8; ++r) {
            float next[32];
            for (int lane = 0; lane < 32; ++lane) {
                const float self = e[32 * r + lane];
                const float peer = e[32 * r + (lane ^ stride)];
                next[lane]       = (lane & stride) == 0 ? self + peer : peer - self;
            }
            for (int lane = 0; lane < 32; ++lane) { e[32 * r + lane] = next[lane]; }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {
        for (int base = 0; base < 8; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                for (int lane = 0; lane < 32; ++lane) {
                    const float low                       = e[32 * (base + offset) + lane];
                    const float high                      = e[32 * (base + offset + span) + lane];
                    e[32 * (base + offset) + lane]        = low + high;
                    e[32 * (base + offset + span) + lane] = low - high;
                }
            }
        }
    }
    for (int i = 0; i < kDim; ++i) { e[i] *= 0.0625f; }
}

void quantize64(const float* group, std::int8_t* codes, unsigned short* scale_bits) {
    float absmax = 0.0f;
    for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(group[i])); }
    const unsigned short bits =
        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
    const float scale = __half2float(__ushort_as_half(bits));
    const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < 64; ++i) {
        const int q = scale > 0.0f
                          ? std::max(-127, std::min(127, static_cast<int>(std::lrintf(group[i] * inv))))
                          : 0;
        codes[i] = static_cast<std::int8_t>(q);
    }
    *scale_bits = bits;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
float bf16_value(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0f;
    std::memcpy(&value, &wide, 4);
    return value;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(90210);

    void* backing    = nullptr;
    void* d_q_tensor = nullptr;
    void* d_capture  = nullptr;
    try {
        KVPageGeometry geometry;
        geometry.page_tokens        = kTokens;
        geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < kLayers; ++layer) {
            geometry.planes.push_back({DType::I8, kDim, kHeads, kDim});
            geometry.planes.push_back({DType::I8, kDim, kHeads, kDim});
            geometry.planes.push_back({DType::FP16, kGroups, kHeads, kDim});
            geometry.planes.push_back({DType::FP16, kGroups, kHeads, kDim});
        }

        LayoutBuilder builder;
        const DeviceKVPagePoolLayout pool_layout =
            plan_device_kv_page_pool(builder, DeviceKVPagePoolSpec{.page_group_count = kPages,
                                                                   .geometry = geometry});
        const std::size_t backing_bytes = pool_layout.payload_bytes() + (1u << 20);
        CHECK(cudaMalloc(&backing, backing_bytes) == cudaSuccess, "cudaMalloc backing");

        // ---- the prefill capture: column `kTokens - 1` of a [dim, q_heads, tokens] Q ----
        const std::uint32_t column = static_cast<std::uint32_t>(kTokens - 1);
        std::vector<std::uint16_t> q_layer(static_cast<std::size_t>(kQHeads) * kDim);
        for (std::uint16_t& bits : q_layer) {
            bits = bf16_bits(static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f);
        }
        std::vector<std::uint16_t> q_tensor(static_cast<std::size_t>(kDim) * kQHeads * kTokens, 0);
        for (int head = 0; head < kQHeads; ++head) {
            for (int d = 0; d < kDim; ++d) {
                const std::int64_t at =
                    static_cast<std::int64_t>(d) +
                    static_cast<std::int64_t>(kDim) *
                        (head + static_cast<std::int64_t>(kQHeads) * column);
                q_tensor[static_cast<std::size_t>(at)] =
                    q_layer[static_cast<std::size_t>(head) * kDim + d];
            }
        }

        // ---- raw K per (page, layer, kv head): random, except the last page which
        //      follows the query's GQA-averaged row so its score must pin at ~1 ----
        std::vector<std::int32_t> baked(kPages);
        for (int page = 0; page < kPages; ++page) { baked[page] = 1000 + page * kTokens; }
        const auto raw_value = [&](int page, int layer, int head, int token, int d) {
            if (page == kPages - 1) {
                const int group = kQHeads / kHeads;
                float sum       = 0.0f;
                for (int gi = 0; gi < group; ++gi) {
                    sum += bf16_value(q_layer[static_cast<std::size_t>(head * group + gi) * kDim + d]);
                }
                return sum / static_cast<float>(group) * 0.9f;
            }
            return static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f;
        };

        std::vector<std::vector<std::int8_t>> codes(kLayers);
        std::vector<std::vector<unsigned short>> scales(kLayers);
        for (int layer = 0; layer < kLayers; ++layer) {
            codes[layer].resize(static_cast<std::size_t>(kCodePageStride) * kPages);
            scales[layer].resize(static_cast<std::size_t>(kScalePageStride) * kPages);
            for (int page = 0; page < kPages; ++page) {
                for (int head = 0; head < kHeads; ++head) {
                    for (int token = 0; token < kTokens; ++token) {
                        float rotated[kDim];
                        for (int d = 0; d < kDim; ++d) {
                            rotated[d] = raw_value(page, layer, head, token, d);
                        }
                        rope_rotary(rotated, baked[page] + token, false);
                        hadamard256_host(rotated);
                        for (int group = 0; group < kGroups; ++group) {
                            std::int8_t group_codes[64];
                            unsigned short group_scale = 0;
                            quantize64(rotated + group * 64, group_codes, &group_scale);
                            const std::int64_t at =
                                static_cast<std::int64_t>(page) * kCodePageStride +
                                code_off(head, token, group * 64);
                            for (int i = 0; i < 64; ++i) {
                                codes[layer][static_cast<std::size_t>(at + i)] = group_codes[i];
                            }
                            scales[layer][static_cast<std::size_t>(
                                static_cast<std::int64_t>(page) * kScalePageStride +
                                scale_off(head, token, group))] = group_scale;
                        }
                    }
                }
            }
        }

        {
            DeviceKVPagePool pool(DeviceSpan{backing, backing_bytes}, pool_layout);
            for (int layer = 0; layer < kLayers; ++layer) {
                CHECK(cudaMemcpy(pool.plane(static_cast<std::size_t>(layer * 4)).data,
                                 codes[layer].data(), codes[layer].size(),
                                 cudaMemcpyHostToDevice) == cudaSuccess,
                      "upload K codes layer %d", layer);
                CHECK(cudaMemcpy(pool.plane(static_cast<std::size_t>(layer * 4 + 2)).data,
                                 scales[layer].data(), scales[layer].size() * 2,
                                 cudaMemcpyHostToDevice) == cudaSuccess,
                      "upload K scales layer %d", layer);
            }

            const std::size_t capture_bytes =
                static_cast<std::size_t>(kLayers) * kQHeads * kDim * 2;
            CHECK(cudaMalloc(&d_q_tensor, q_tensor.size() * 2) == cudaSuccess, "malloc q tensor");
            CHECK(cudaMalloc(&d_capture, capture_bytes) == cudaSuccess, "malloc capture");
            CHECK(cudaMemcpy(d_q_tensor, q_tensor.data(), q_tensor.size() * 2,
                             cudaMemcpyHostToDevice) == cudaSuccess,
                  "upload query tensor");
            CHECK(cudaMemset(d_capture, 0, capture_bytes) == cudaSuccess, "zero capture");
            for (int layer = 0; layer < kLayers; ++layer) {
                kvmem_capture_query(d_q_tensor, kQHeads, kDim, kTokens, column, d_capture,
                                    static_cast<std::uint32_t>(layer), nullptr);
            }
            cudaDeviceSynchronize();

            std::vector<std::uint16_t> captured(static_cast<std::size_t>(kLayers) * kQHeads * kDim);
            cudaMemcpy(captured.data(), d_capture, capture_bytes, cudaMemcpyDeviceToHost);
            int capture_mismatch = 0;
            for (int head = 0; head < kQHeads; ++head) {
                for (int d = 0; d < kDim; ++d) {
                    const std::uint16_t want = q_layer[static_cast<std::size_t>(head) * kDim + d];
                    for (int layer = 0; layer < kLayers; ++layer) {
                        if (captured[(static_cast<std::size_t>(layer) * kQHeads + head) * kDim + d] !=
                            want) {
                            ++capture_mismatch;
                        }
                    }
                }
            }
            CHECK(capture_mismatch == 0,
                  "query capture must lift the last column bitwise (%d differ)", capture_mismatch);

            // ---- score the ledger: two random pages, one query-aligned, one parked ----
            const KvmemQueryCapture query{d_capture, static_cast<std::uint32_t>(kLayers),
                                          static_cast<std::uint32_t>(kQHeads),
                                          static_cast<std::uint32_t>(kDim)};
            std::vector<std::int32_t> pages{kPages - 1, 0, -1};
            std::vector<std::int32_t> positions{baked[kPages - 1], baked[0], 0};
            const float sentinel = -9.0f;
            std::vector<float> scores{0.0f, 0.0f, sentinel};
            // A non-default stream, like the engine's compute stream: the host
            // read-back must be ordered against it.
            cudaStream_t score_stream = nullptr;
            CHECK(cudaStreamCreate(&score_stream) == cudaSuccess, "create score stream");
            kvmem_score_blocks(pool, query, pages.data(), positions.data(),
                               static_cast<std::uint32_t>(pages.size()), scores.data(),
                               score_stream);
            cudaStreamSynchronize(score_stream);
            cudaStreamDestroy(score_stream);
            CHECK(scores[2] == sentinel,
                  "a block with no device page must be left untouched (got %.3f)", scores[2]);
            CHECK(scores[0] > 0.98f,
                  "a block whose raw K follows the query must score near 1 (got %.5f)", scores[0]);

            // ---- CPU model of the kernel: un-rotate each token, then cosine ----
            float worst = 0.0f;
            for (int slot = 0; slot < 2; ++slot) {
                const int page  = pages[slot];
                double total    = 0.0;
                for (int layer = 0; layer < kLayers; ++layer) {
                    for (int head = 0; head < kHeads; ++head) {
                        const int group = kQHeads / kHeads;
                        float q_bar[kDim];
                        for (int d = 0; d < kDim; ++d) {
                            float sum = 0.0f;
                            for (int gi = 0; gi < group; ++gi) {
                                sum += bf16_value(captured[(
                                    static_cast<std::size_t>(layer) * kQHeads + head * group + gi) *
                                                          kDim +
                                                      d]);
                            }
                            q_bar[d] = sum / static_cast<float>(group);
                        }
                        float mean_k[kDim];
                        for (int d = 0; d < kDim; ++d) { mean_k[d] = 0.0f; }
                        for (int token = 0; token < kTokens; ++token) {
                            float row[kDim];
                            for (int d = 0; d < kDim; ++d) {
                                const std::int64_t code_at =
                                    static_cast<std::int64_t>(page) * kCodePageStride +
                                    code_off(head, token, d);
                                const std::int64_t scale_at =
                                    static_cast<std::int64_t>(page) * kScalePageStride +
                                    scale_off(head, token, d / 64);
                                row[d] = static_cast<float>(
                                             codes[layer][static_cast<std::size_t>(code_at)]) *
                                         __half2float(__ushort_as_half(
                                             scales[layer][static_cast<std::size_t>(scale_at)]));
                            }
                            hadamard256_host(row);
                            rope_rotary(row, baked[page] + token, true);
                            for (int d = 0; d < kDim; ++d) { mean_k[d] += row[d]; }
                        }
                        double dot = 0.0, q_norm = 0.0, k_norm = 0.0;
                        for (int d = 0; d < kDim; ++d) {
                            dot += static_cast<double>(q_bar[d]) * mean_k[d];
                            q_norm += static_cast<double>(q_bar[d]) * q_bar[d];
                            k_norm += static_cast<double>(mean_k[d]) * mean_k[d];
                        }
                        const double den = q_norm * k_norm;
                        total += den > 0.0 ? dot / std::sqrt(den) : 0.0;
                    }
                }
                const float expected = static_cast<float>(total / kLayers / kHeads);
                const float diff     = std::fabs(expected - scores[slot]);
                worst                = std::max(worst, diff);
                std::printf("page %d score %.5f expected %.5f\n", page, scores[slot], expected);
                CHECK(diff < 2e-3f, "page %d cosine mismatch (kernel %.5f, cpu %.5f)", page,
                      scores[slot], expected);
            }
            std::printf("worst cosine diff = %.6f\n", worst);
        }

        if (d_capture != nullptr) { cudaFree(d_capture); }
        if (d_q_tensor != nullptr) { cudaFree(d_q_tensor); }
        cudaFree(backing);
    } catch (const std::exception& error) {
        std::printf("FAIL exception: %s\n", error.what());
        ++failures;
    }

    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
