// SPDX-License-Identifier: MIT
// Reuse vendored GGML load/MMA/typed writer and R17/R19's independent I64 TU.
// RDNA3 supplies the same I64/J128/4-warp row mapping; use NVIDIA math and
// occupancy1, not AMD MMA or its occupancy hint. All original TUs remain I128.
// Down projections now own the complete K instead of Stream-K partial sums:
// association may differ in FP32, so each arm needs the independent CPU oracle.
#define NINFER_GGML_MMQ_IQ3_S_I64 1
#define NINFER_GGML_MMQ_IQ3_XXS_I64 1
#define NINFER_GGML_MMQ_IQ4_XS_I64 1
#define mul_mat_q ninfer_r28_i64_mul_mat_q
#include "ggml_bridge_internal.cuh"
#undef mul_mat_q
#undef NINFER_GGML_MMQ_IQ3_S_I64
#undef NINFER_GGML_MMQ_IQ3_XXS_I64
#undef NINFER_GGML_MMQ_IQ4_XS_I64

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ninfer::ops::gguf::detail {
namespace {

bool bundle_enabled() {
    static const bool value = [] {
        const char* raw = std::getenv("NINFER_R28_BUNDLE");
        if (raw == nullptr || std::strcmp(raw, "0") == 0) { return false; }
        if (std::strcmp(raw, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_R28_BUNDLE must be 0 or 1");
    }();
    return value;
}

bool bundle_trace() {
    static const bool value = [] {
        const char* raw = std::getenv("NINFER_R28_BUNDLE_TRACE");
        if (raw == nullptr || std::strcmp(raw, "0") == 0) { return false; }
        if (std::strcmp(raw, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_R28_BUNDLE_TRACE must be 0 or 1");
    }();
    return value;
}

template <ggml_type type>
void launch_i64(const void* weight, std::int64_t row_bytes, int rows, int k,
                const void* activation, int columns, __nv_bfloat16* out,
                std::int64_t out_column_stride, cudaStream_t stream) {
    constexpr auto config = ggml_cuda_mmq_get_config_ampere(type, 128, false);
    static_assert(config.I == 64 && config.J == 128 && config.nthreads == 128);
    static_assert(config.occupancy == 1 && config.K_vram == 256 && config.stream_k);
    static_assert(config.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 && !config.fallback);
    static_assert(ggml_cuda_type_traits<type>::qk == 256);
    constexpr int shared = 38400;
    if (mmq_get_nbytes_shared(config, device_facts().cc) != shared ||
        device_facts().shared_per_block_optin < shared) {
        throw std::runtime_error("R28 I64: unexpected shared-memory geometry");
    }
    int device = 0;
    check(cudaGetDevice(&device), "R28 I64 device");
    static std::array<std::once_flag, 16> raised;
    if (device < 0 || device >= int(raised.size())) {
        throw std::invalid_argument("R28 I64: device ordinal out of range");
    }
    std::call_once(raised[device], [] {
        check(cudaFuncSetAttribute(
                  ninfer_r28_i64_mul_mat_q<type, 128, false, GGML_PREC_Q8, __nv_bfloat16>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "R28 I64 shared memory");
        check(cudaFuncSetAttribute(
                  ninfer_r28_i64_mul_mat_q<type, 128, false, GGML_PREC_Q8, __nv_bfloat16>,
                  cudaFuncAttributePreferredSharedMemoryCarveout, 100), "R28 I64 shared carveout");
    });
    const int blocks = (rows / config.I) * (columns / config.J);
    // Both admitted geometries divide I/J exactly: one complete K per block,
    // no private FP32 plane, partial sums, or Stream-K fixup destination.
    const uint3 blocks_fd = init_fastdiv_values(k / 256);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(columns / config.J);
    ninfer_r28_i64_mul_mat_q<type, 128, false, GGML_PREC_Q8, __nv_bfloat16>
        <<<dim3(blocks, 1, 1), dim3(32, 4, 1), shared, stream>>>(
            static_cast<const char*>(weight), static_cast<const int*>(activation),
            nullptr, nullptr, out, nullptr, nullptr, blocks_fd, rows, columns,
            static_cast<int>(row_bytes / ggml_cuda_type_traits<type>::bs), columns,
            static_cast<int>(out_column_stride), one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
    check(cudaGetLastError(), "R28 I64 full-K BF16 launch");
    trace_matrix_output(type, rows, k, columns, 64, 128, 128, blocks, false, true);
    if (bundle_trace()) {
        // IQ4_XS has two distinct shapes; do not let one consume the other's tag.
        static std::array<std::once_flag, 2> traced;
        std::call_once(traced[rows == 17408 ? 1 : 0], [&] {
            std::fprintf(stderr,
                "[r28-mmq-i64] policy=1 type=%d N=%d K=%d T=%d I=64 J=128 "
                "threads=128 blocks=%d shared=38400 fixup=0 output=bf16-direct "
                "column_stride=%lld kernel=ninfer_r28_i64_mul_mat_q\n",
                int(type), rows, k, columns, blocks, static_cast<long long>(out_column_stride));
        });
    }
}

} // namespace

bool try_r28_i64_bf16(GgmlType type, const void* weight, std::int64_t row_bytes,
                       int rows, int k, const void* activation, int columns,
                       __nv_bfloat16* out, std::int64_t out_column_stride,
                       cudaStream_t stream) {
    if (!bundle_enabled() || columns != 512 || out_column_stride != rows ||
        device_facts().cc != GGML_CUDA_CC_ADA_LOVELACE) { return false; }
    const bool down = rows == 5120 && k == 17408;
    const bool iq4_up = type == GgmlType::IQ4_XS && rows == 17408 && k == 5120;
    if (!down && !iq4_up) { return false; }
    switch (type) {
    case GgmlType::IQ3_S:
        if (!down || row_bytes != 7480) { return false; }
        launch_i64<GGML_TYPE_IQ3_S>(weight, row_bytes, rows, k, activation, columns,
                                    out, out_column_stride, stream);
        return true;
    case GgmlType::IQ3_XXS:
        if (!down || row_bytes != 6664) { return false; }
        launch_i64<GGML_TYPE_IQ3_XXS>(weight, row_bytes, rows, k, activation, columns,
                                      out, out_column_stride, stream);
        return true;
    case GgmlType::IQ4_XS:
        if (row_bytes != (down ? 9248 : 2720)) { return false; }
        launch_i64<GGML_TYPE_IQ4_XS>(weight, row_bytes, rows, k, activation, columns,
                                     out, out_column_stride, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninfer::ops::gguf::detail
