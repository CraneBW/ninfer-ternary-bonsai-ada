// Unit test for the KVMem K2 re-phase kernel (standalone; build with
// `builder.exe -test tests/kvmem_rerope_test.cu`).
//
// Pipeline under test (int8-group64 K page, MRoPE text):
//   stored = Quant(H64(RoPE_src(raw)))
//   kernel re-bakes src -> dst, compared against a CPU reference that mirrors
//   the exact float op sequence. Groups 1..3 are position-free and must be
//   bitwise untouched.
#include "kvmem/kvmem_rerope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::kvmem;

namespace {

constexpr int kHeads  = 4;
constexpr int kPages  = 2;
constexpr int kTokens = kPagedKVPageSize;  // 64
constexpr int kDim    = kKVCacheInt8HeadDim;
constexpr int kGroups = kKVCacheInt8Groups;

int failures = 0;
#define CHECK(cond, ...)                                      \
    do {                                                      \
        if (!(cond)) {                                        \
            ++failures;                                       \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);   \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

// Mirrors kTextRopeInvFrequency: theta^(-2*pair/64) with the Qwen3.6 text theta.
float host_freq(int pair) { return std::pow(1e7f, -2.0f * static_cast<float>(pair) / 64.0f); }

// -- warp-faithful H64 on a 64-element group (lane l owns e[l], e[l+32]) --
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

// MRoPE text: pair p in [0,32) pairs (p, p+32), axis = p % 3.
void rope_group0(float* e, const std::int32_t pos3[3], bool inverse) {
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(pos3[p % 3]) * host_freq(p);
        float s = std::sin(phi);
        const float c = std::cos(phi);
        if (inverse) { s = -s; }
        const float a = e[p];
        const float b = e[p + 32];
        e[p]      = a * c - b * s;
        e[p + 32] = b * c + a * s;
    }
}

std::int64_t code_off(int pg, int h, int t, int d) {
    return static_cast<std::int64_t>(kDim) * kTokens * (static_cast<std::int64_t>(h) + kHeads * pg) +
           static_cast<std::int64_t>(kDim) * t + d;
}
std::int64_t scale_off(int pg, int h, int t, int g) {
    return static_cast<std::int64_t>(kGroups) * kTokens *
               (static_cast<std::int64_t>(h) + kHeads * pg) +
           static_cast<std::int64_t>(kGroups) * t + g;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(1234);

    const std::size_t code_count  = static_cast<std::size_t>(kDim) * kTokens * kHeads * kPages;
    const std::size_t scale_count = static_cast<std::size_t>(kGroups) * kTokens * kHeads * kPages;

    // -- random raw K + source/window positions --
    std::vector<float> raw(code_count);
    for (auto& v : raw) { v = static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f; }
    std::vector<std::int32_t> src_pos3(3 * kTokens * kPages);
    std::vector<std::int32_t> dst_pos3(3 * kTokens * kPages);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int t = 0; t < kTokens; ++t) {
            const std::int32_t src = pg * kTokens + t + 100000;  // absolute position
            const std::int32_t dst = pg * kTokens + t;           // compacted window slot
            for (int a = 0; a < 3; ++a) {
                src_pos3[a * (kTokens * kPages) + pg * kTokens + t] = src;
                dst_pos3[a * (kTokens * kPages) + pg * kTokens + t] = dst;
            }
        }
    }

    // -- stored = Quant(H64(RoPE_src(raw))), the append pipeline on host --
    std::vector<std::int8_t> stored_codes(code_count);
    std::vector<unsigned short> stored_scales(scale_count);
    std::vector<float> group(64);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const std::int32_t pos3[3] = {src_pos3[(pg * kTokens + t) * 3 + 0],
                                              src_pos3[(pg * kTokens + t) * 3 + 1],
                                              src_pos3[(pg * kTokens + t) * 3 + 2]};
                for (int g = 0; g < kGroups; ++g) {
                    for (int l = 0; l < 32; ++l) {
                        group[l]      = raw[code_off(pg, h, t, g * 64 + l)];
                        group[l + 32] = raw[code_off(pg, h, t, g * 64 + l + 32)];
                    }
                    if (g == 0) { rope_group0(group.data(), pos3, false); }
                    hadamard64_warp(group.data());
                    float absmax = 0.0f;
                    for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(group[i])); }
                    const unsigned short bits =
                        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                    const float s = __half2float(__ushort_as_half(bits));
                    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        const int q =
                            std::max(-127, std::min(127, static_cast<int>(std::lrintf(group[i] * inv))));
                        stored_codes[code_off(pg, h, t, g * 64 + i)] = static_cast<std::int8_t>(q);
                        stored_scales[scale_off(pg, h, t, g)]        = bits;
                    }
                }
            }
        }
    }

    // -- device run: re-bake src -> dst --
    std::int8_t* d_codes = nullptr;
    unsigned short* d_scales = nullptr;
    std::int32_t *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_codes, code_count);
    cudaMalloc(&d_scales, scale_count * 2);
    cudaMalloc(&d_src, src_pos3.size() * 4);
    cudaMalloc(&d_dst, dst_pos3.size() * 4);
    cudaMemcpy(d_codes, stored_codes.data(), code_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, stored_scales.data(), scale_count * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src_pos3.data(), src_pos3.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst_pos3.data(), dst_pos3.size() * 4, cudaMemcpyHostToDevice);
    kvmem_rerope_int8_g64_pages<kHeads>(d_codes, reinterpret_cast<__half*>(d_scales), kPages,
                                        d_src, d_dst, 0);
    cudaDeviceSynchronize();

    std::vector<std::int8_t> out_codes(code_count);
    std::vector<unsigned short> out_scales(scale_count);
    cudaMemcpy(out_codes.data(), d_codes, code_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_scales.data(), d_scales, scale_count * 2, cudaMemcpyDeviceToHost);

    // -- check 1: groups 1..3 bitwise untouched --
    std::size_t touched = 0;
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                for (int g = 1; g < kGroups; ++g) {
                    for (int i = 0; i < 64; ++i) {
                        const std::size_t idx = static_cast<std::size_t>(
                            code_off(pg, h, t, g * 64 + i));
                        if (out_codes[idx] != stored_codes[idx]) { ++touched; }
                    }
                    if (out_scales[static_cast<std::size_t>(scale_off(pg, h, t, g))] !=
                        stored_scales[static_cast<std::size_t>(scale_off(pg, h, t, g))]) {
                        ++touched;
                    }
                }
            }
        }
    }
    CHECK(touched == 0, "groups 1..3 must be bitwise untouched (%zu changed)", touched);

    // -- check 2: group 0 vs CPU reference (mirrors the kernel's float sequence) --
    double worst = 0.0;
    int bad_codes = 0, scale_mismatch = 0;
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const std::int32_t pos3[3] = {src_pos3[(pg * kTokens + t) * 3 + 0],
                                              src_pos3[(pg * kTokens + t) * 3 + 1],
                                              src_pos3[(pg * kTokens + t) * 3 + 2]};
                const std::int32_t posd3[3] = {dst_pos3[(pg * kTokens + t) * 3 + 0],
                                               dst_pos3[(pg * kTokens + t) * 3 + 1],
                                               dst_pos3[(pg * kTokens + t) * 3 + 2]};
                for (int l = 0; l < 32; ++l) {
                    const float s = __half2float(__ushort_as_half(
                        stored_scales[static_cast<std::size_t>(scale_off(pg, h, t, 0))]));
                    group[l]      = static_cast<float>(stored_codes[code_off(pg, h, t, l)]) * s;
                    group[l + 32] = static_cast<float>(stored_codes[code_off(pg, h, t, l + 32)]) * s;
                }
                hadamard64_warp(group.data());
                rope_group0(group.data(), pos3, true);
                rope_group0(group.data(), posd3, false);
                hadamard64_warp(group.data());
                float absmax = 0.0f;
                for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(group[i])); }
                const unsigned short ref_bits =
                    __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                const std::size_t got_scale_idx =
                    static_cast<std::size_t>(scale_off(pg, h, t, 0));
                // GPU sincosf vs CPU sin/cos differ in the last ulp, so the absmax
                // can round to the neighboring fp16. Value-level tolerance, not bits.
                const float ref_scale = __half2float(__ushort_as_half(ref_bits));
                const float got_scale =
                    __half2float(__ushort_as_half(out_scales[got_scale_idx]));
                if (std::fabs(got_scale - ref_scale) >
                    1e-3f * ref_scale + 1e-7f) {
                    ++scale_mismatch;
                }
                for (int i = 0; i < 64; ++i) {
                    const float got =
                        static_cast<float>(out_codes[code_off(pg, h, t, i)]) * got_scale;
                    const double d = std::fabs(static_cast<double>(got) - group[i]);
                    worst = std::max(worst, d);
                    if (d > 1.01f * got_scale) { ++bad_codes; }
                }
            }
        }
    }
    CHECK(scale_mismatch == 0, "group-0 scales must match the CPU model (%d differ)",
          scale_mismatch);
    CHECK(bad_codes == 0, "group-0 codes within one step of the CPU model (%d bad)", bad_codes);
    std::printf("worst dequant diff = %.6f\n", worst);

    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
