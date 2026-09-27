// Standalone harness for the REAL folded-basis rotation kernels.
//
// It compiles the very same ternary_rotation_kernels.cuh the engine ships and dumps the kernel
// input (activation + signs) and output as raw f32, so a numpy oracle can check the documented
// transform   y = H * (s * (P * x))   and   h = s * (H * z)
// against the real device code.  Outputs go to an ASCII path: a native MSVC/nvcc binary cannot
// open a Chinese path (its literals are decoded with the ANSI code page).
//
// Build:
//   nvcc -O2 -std=c++20 -arch=sm_89 -I <src> -o rot_test rot_test.cu
//   ./rot_test <dump-dir>/          # then: python3 ../oracle_rot.py <dump-dir>/
//
// The author's copy hard-coded a Windows build root for the dump directory and needed
// `-Xcompiler /wd4819` (MSVC code page); here the directory is argv[1], so the tree is
// relocatable, and the flag is dropped because it is MSVC-only.

#include "ops/linear/ternary/ternary_rotation_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::string g_out_dir = "./bm2out/";

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

// Deterministic +-1 signs, identical for both width layouts; the oracle reads the dumped file
// rather than recomputing, so no formula has to be reproduced in numpy.
std::vector<float> make_signs(std::int32_t count, std::uint32_t seed) {
    std::vector<float> signs(static_cast<std::size_t>(count));
    std::uint32_t state = seed;
    for (auto& s : signs) {
        state = state * 1664525u + 1013904223u;
        s     = ((state >> 16) & 1u) ? -1.0f : 1.0f;
    }
    return signs;
}

std::vector<float> make_input(std::int32_t k, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(k) * tokens);
    std::uint32_t state = seed;
    for (auto& v : values) {
        state = state * 1664525u + 1013904223u;
        // spread over +-1 so the transform's cancellations are visible
        v = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f) - 1.0f;
    }
    return values;
}

void write_f32(const std::string& name, const std::vector<float>& data) {
    const std::string path = g_out_dir + name;
    std::FILE* file        = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { fail(path.c_str()); }
    if (std::fwrite(data.data(), sizeof(float), data.size(), file) != data.size()) {
        fail("short write");
    }
    std::fclose(file);
}

std::vector<float> bf16_to_f32(const std::vector<__nv_bfloat16>& data) {
    std::vector<float> out(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) { out[i] = __bfloat162float(data[i]); }
    return out;
}

struct Case {
    const char* tag;
    std::int32_t k;
    std::int32_t tokens;
    std::int32_t perm_hd;
    std::int32_t perm_nk;
    std::int32_t perm_rep;
    bool inverse;
};

void run_case(const Case& c, std::uint32_t seed) {
    const std::int32_t n_blk  = c.k / 1024;
    const std::int32_t blocks = n_blk;

    std::vector<float> host_signs = make_signs(c.k, seed ^ 0x51u);
    std::vector<float> host_x     = make_input(c.k, c.tokens, seed);

    float* dev_signs = nullptr;
    check(cudaMalloc(&dev_signs, host_signs.size() * sizeof(float)), "cudaMalloc signs");
    check(cudaMemcpy(dev_signs, host_signs.data(), host_signs.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "copy signs");

    std::vector<__nv_bfloat16> host_a(host_x.size());
    for (std::size_t i = 0; i < host_x.size(); ++i) {
        host_a[i] = __float2bfloat16_rn(host_x[i]);
    }
    __nv_bfloat16* dev_in  = nullptr;
    __nv_bfloat16* dev_out = nullptr;
    check(cudaMalloc(&dev_in, host_a.size() * sizeof(__nv_bfloat16)), "cudaMalloc in");
    check(cudaMalloc(&dev_out, host_a.size() * sizeof(__nv_bfloat16)), "cudaMalloc out");
    check(cudaMemcpy(dev_in, host_a.data(), host_a.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice),
          "copy in");

    const unsigned grid =
        static_cast<unsigned>(((static_cast<long long>(blocks) * c.tokens) + 7) / 8);
    const unsigned block = 8 * 32;

    if (c.inverse) {
        // in place: pass the same buffer twice through the dedicated kernel
        ninfer::ops::detail::ternary_rotate_inverse_inplace_bf16_kernel<<<grid, block>>>(
            dev_in, dev_signs, n_blk, c.k, c.tokens);
        check(cudaGetLastError(), "inverse kernel");
        check(cudaMemcpy(dev_out, dev_in, host_a.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToDevice),
              "copy back");
    } else {
        ninfer::ops::detail::ternary_rotate_bf16_kernel<<<grid, block>>>(
            dev_in, dev_out, dev_signs, n_blk, c.k, c.tokens, c.perm_hd, c.perm_nk, c.perm_rep,
            /*inverse=*/0);
        check(cudaGetLastError(), "forward kernel");
    }
    check(cudaDeviceSynchronize(), "sync");

    std::vector<__nv_bfloat16> host_out(host_a.size());
    check(cudaMemcpy(host_out.data(), dev_out, host_out.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy out");

    const std::string tag = c.tag;
    write_f32(tag + ".in.f32", host_x);
    write_f32(tag + ".signs.f32", host_signs);
    write_f32(tag + ".out.f32", bf16_to_f32(host_out));
    std::printf("case %-10s k=%-6d tokens=%d n_blk=%d perm=(%d,%d,%d) inverse=%d  -> %s%s\n",
                c.tag, c.k, c.tokens, n_blk, c.perm_hd, c.perm_nk, c.perm_rep,
                c.inverse ? 1 : 0, g_out_dir.c_str(), tag.c_str());

    cudaFree(dev_signs);
    cudaFree(dev_in);
    cudaFree(dev_out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        g_out_dir = argv[1];
        if (!g_out_dir.empty() && g_out_dir.back() != '/') { g_out_dir.push_back('/'); }
    }
    std::printf("dump dir: %s\n", g_out_dir.c_str());
    std::printf("device: %s\n", []() -> const char* {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { return "unknown"; }
        return prop.name;
    }());
    // width 5120 (5 sign rows), decode-shaped and token-tiled
    run_case({"plain_t1", 5120, 1, 0, 0, 1, false}, 0x1234u);
    run_case({"plain_t3", 5120, 3, 0, 0, 1, false}, 0x5678u);
    // width 6144 with the GDN output permutation (128, 16, 3)
    run_case({"perm_t1", 6144, 1, 128, 16, 3, false}, 0x9abcu);
    run_case({"perm_t2", 6144, 2, 128, 16, 3, false}, 0xdef0u);
    // width 17408 (17 sign rows, the MLP down input)
    run_case({"wide_t1", 17408, 1, 0, 0, 1, false}, 0x2468u);
    // inverse (token embedding) in place
    run_case({"inv_t2", 5120, 2, 0, 0, 1, true}, 0x1357u);
    std::printf("ALL KERNELS RAN\n");
    return 0;
}
