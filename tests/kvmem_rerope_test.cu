// Unit test for the KVMem K2 re-phase kernel (standalone; build with
// `builder.exe -test tests/kvmem_rerope_test.cu`).
//
// Pipeline under test (int8-group64 K page in the D256 Hadamard basis, MRoPE text):
//   stored   = Quant_G64(H256(RoPE_src(raw)))
//   re-baked = Quant_G64(H256(RoPE_dst(RoPE_src^-1(H256(Dequant_G64(stored))))))
// Compared against:
//   1. A CPU reference mirroring the exact float op sequence across all 4 groups.
//   2. Reconstruction accuracy in the un-rotated domain (dims [0,64) vs RoPE_dst(raw),
//      dims [64,256) vs position-free raw).
//   3. Bitwise equivalence between kvmem_rerope_int8_g64_pages and the batched
//      indexed kvmem_rerope_int8_g64_planes launcher.
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

// -- warp-faithful H256 on a 256-element row (lane l owns e[l + 32*r]) --
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

// MRoPE text: pair p in [0,32) pairs (p, p+32), axis = p % 3.
void rope_group0(float* e, const std::int32_t pos3[3], bool inverse) {
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(pos3[p % 3]) * host_freq(p);
        float s         = std::sin(phi);
        const float c   = std::cos(phi);
        if (inverse) { s = -s; }
        const float a = e[p];
        const float b = e[p + 32];
        e[p]          = a * c - b * s;
        e[p + 32]     = b * c + a * s;
    }
}

void quantize_group(const float* group, std::int8_t* codes, unsigned short* scale_bits) {
    float absmax = 0.0f;
    for (int i = 0; i < 64; ++i) { absmax = std::max(absmax, std::fabs(group[i])); }
    const unsigned short bits =
        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
    const float s   = __half2float(__ushort_as_half(bits));
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < 64; ++i) {
        const int q = std::max(-127, std::min(127, static_cast<int>(std::lrintf(group[i] * inv))));
        codes[i]    = static_cast<std::int8_t>(q);
    }
    *scale_bits = bits;
}

std::int64_t code_off(int pg, int h, int t, int d) {
    return static_cast<std::int64_t>(kDim) * kTokens *
               (static_cast<std::int64_t>(h) + kHeads * pg) +
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
    std::vector<std::int32_t> h_pages(kPages);
    std::vector<std::int32_t> h_from_bases(kPages);
    std::vector<std::int32_t> h_to_bases(kPages);
    const int pos_stride = kTokens * kPages;
    for (int pg = 0; pg < kPages; ++pg) {
        h_pages[pg]      = pg;
        h_from_bases[pg] = 100000 + pg * 320;
        h_to_bases[pg]   = pg * kTokens;
        for (int t = 0; t < kTokens; ++t) {
            const std::int32_t src = h_from_bases[pg] + t;
            const std::int32_t dst = h_to_bases[pg] + t;
            for (int a = 0; a < 3; ++a) {
                src_pos3[a * pos_stride + pg * kTokens + t] = src;
                dst_pos3[a * pos_stride + pg * kTokens + t] = dst;
            }
        }
    }

    // -- stored = Quant_G64(H256(RoPE_src(raw))), the append pipeline on host --
    std::vector<std::int8_t> stored_codes(code_count);
    std::vector<unsigned short> stored_scales(scale_count);
    std::vector<float> row(kDim);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const int col              = pg * kTokens + t;
                const std::int32_t pos3[3] = {src_pos3[0 * pos_stride + col],
                                              src_pos3[1 * pos_stride + col],
                                              src_pos3[2 * pos_stride + col]};
                for (int d = 0; d < kDim; ++d) { row[d] = raw[code_off(pg, h, t, d)]; }
                rope_group0(row.data(), pos3, false);
                hadamard256_warp(row.data());
                for (int g = 0; g < kGroups; ++g) {
                    quantize_group(row.data() + g * 64, &stored_codes[code_off(pg, h, t, g * 64)],
                                   &stored_scales[scale_off(pg, h, t, g)]);
                }
            }
        }
    }

    // -- device run 1: kvmem_rerope_int8_g64_pages --
    std::int8_t* d_codes     = nullptr;
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
    kvmem_rerope_int8_g64_pages<kHeads>(d_codes, reinterpret_cast<__half*>(d_scales), kPages, d_src,
                                        d_dst, 0);
    cudaDeviceSynchronize();

    std::vector<std::int8_t> out_codes(code_count);
    std::vector<unsigned short> out_scales(scale_count);
    cudaMemcpy(out_codes.data(), d_codes, code_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_scales.data(), d_scales, scale_count * 2, cudaMemcpyDeviceToHost);

    // -- device run 2: batched kvmem_rerope_int8_g64_planes --
    std::int32_t *d_pages = nullptr, *d_from = nullptr, *d_to = nullptr;
    cudaMalloc(&d_pages, kPages * sizeof(std::int32_t));
    cudaMalloc(&d_from, kPages * sizeof(std::int32_t));
    cudaMalloc(&d_to, kPages * sizeof(std::int32_t));
    cudaMemcpy(d_codes, stored_codes.data(), code_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, stored_scales.data(), scale_count * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pages, h_pages.data(), kPages * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_from, h_from_bases.data(), kPages * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_to, h_to_bases.data(), kPages * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    kvmem_rerope_int8_g64_planes<kHeads>(d_codes, reinterpret_cast<__half*>(d_scales), d_pages,
                                         d_from, d_to, kPages, 0);
    cudaDeviceSynchronize();

    std::vector<std::int8_t> indexed_codes(code_count);
    std::vector<unsigned short> indexed_scales(scale_count);
    cudaMemcpy(indexed_codes.data(), d_codes, code_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(indexed_scales.data(), d_scales, scale_count * 2, cudaMemcpyDeviceToHost);

    CHECK(out_codes == indexed_codes && out_scales == indexed_scales,
          "batched indexed plane launcher must match per-page launcher bitwise");

    // -- check 1: all 4 groups vs CPU reference (mirrors the kernel's float sequence) --
    // -- check 2: un-H256 reconstruction vs RoPE_dst(raw[0..63]) and raw[64..255] --
    double worst          = 0.0;
    double worst_rotary   = 0.0;
    double worst_nonrot   = 0.0;
    int bad_codes         = 0;
    int scale_mismatch    = 0;
    std::vector<float> recon(kDim);
    std::vector<float> ideal(kDim);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const int col               = pg * kTokens + t;
                const std::int32_t pos3[3]  = {src_pos3[0 * pos_stride + col],
                                               src_pos3[1 * pos_stride + col],
                                               src_pos3[2 * pos_stride + col]};
                const std::int32_t posd3[3] = {dst_pos3[0 * pos_stride + col],
                                               dst_pos3[1 * pos_stride + col],
                                               dst_pos3[2 * pos_stride + col]};
                for (int g = 0; g < kGroups; ++g) {
                    const float s = __half2float(__ushort_as_half(
                        stored_scales[static_cast<std::size_t>(scale_off(pg, h, t, g))]));
                    for (int i = 0; i < 64; ++i) {
                        row[g * 64 + i] =
                            static_cast<float>(stored_codes[code_off(pg, h, t, g * 64 + i)]) * s;
                    }
                }
                hadamard256_warp(row.data());
                rope_group0(row.data(), pos3, true);
                rope_group0(row.data(), posd3, false);
                hadamard256_warp(row.data());

                for (int g = 0; g < kGroups; ++g) {
                    std::int8_t ref_codes[64];
                    unsigned short ref_bits = 0;
                    quantize_group(row.data() + g * 64, ref_codes, &ref_bits);
                    const std::size_t got_scale_idx =
                        static_cast<std::size_t>(scale_off(pg, h, t, g));
                    const float ref_scale = __half2float(__ushort_as_half(ref_bits));
                    const float got_scale =
                        __half2float(__ushort_as_half(out_scales[got_scale_idx]));
                    if (std::fabs(got_scale - ref_scale) > 1e-3f * ref_scale + 1e-7f) {
                        ++scale_mismatch;
                    }
                    for (int i = 0; i < 64; ++i) {
                        const float got =
                            static_cast<float>(out_codes[code_off(pg, h, t, g * 64 + i)]) *
                            got_scale;
                        recon[g * 64 + i] = got;
                        const double d    = std::fabs(static_cast<double>(got) - row[g * 64 + i]);
                        worst             = std::max(worst, d);
                        if (d > 1.01f * got_scale) { ++bad_codes; }
                    }
                }

                // Undo H256 on the GPU's re-baked output and compare against
                // ideal RoPE_dst(raw).
                hadamard256_warp(recon.data());
                for (int d = 0; d < kDim; ++d) { ideal[d] = raw[code_off(pg, h, t, d)]; }
                rope_group0(ideal.data(), posd3, false);
                for (int d = 0; d < 64; ++d) {
                    worst_rotary =
                        std::max(worst_rotary, std::fabs(static_cast<double>(recon[d]) - ideal[d]));
                }
                for (int d = 64; d < kDim; ++d) {
                    worst_nonrot =
                        std::max(worst_nonrot, std::fabs(static_cast<double>(recon[d]) - ideal[d]));
                }
            }
        }
    }
    CHECK(scale_mismatch == 0, "all 4 group scales must match the CPU model (%d differ)",
          scale_mismatch);
    CHECK(bad_codes == 0, "all 256 codes within one step of the CPU model (%d bad)", bad_codes);
    CHECK(worst_rotary < 0.04, "un-H256 rotary reconstruction error too large (%.6f)",
          worst_rotary);
    CHECK(worst_nonrot < 0.04, "un-H256 non-rotary reconstruction error too large (%.6f)",
          worst_nonrot);
    std::printf("worst dequant diff = %.6f  rotary_recon = %.6f  nonrot_recon = %.6f\n", worst,
                worst_rotary, worst_nonrot);

    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_pages);
    cudaFree(d_from);
    cudaFree(d_to);

    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
