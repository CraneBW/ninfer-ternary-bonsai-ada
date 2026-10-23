// Reuse vendored GGML's load/MMA/FP32-fold/BF16 writer (MIT; see
// third_party/ggml-quants/README.ninfer.md). The IQ3_S row/warp ratio is also
// used by its RDNA3 configuration; keep Ada's K256/Stream-K/math unchanged.
// The configuration macro is local to this TU. Unique kernel registration and
// a non-template bridge entry keep the original IQ3_S instantiations separate.
#define NINFER_GGML_MMQ_IQ3_S_I64 1
#define mul_mat_q ninfer_r17_iq3s_i64_mul_mat_q
#include "ggml_bridge_internal.cuh"
#undef mul_mat_q
#undef NINFER_GGML_MMQ_IQ3_S_I64

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ninfer::ops::gguf::detail {
namespace {

bool iq3_s_i64_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GGUF_IQ3_S_MMQ_I64");
        if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_GGUF_IQ3_S_MMQ_I64 must be 0 or 1");
    }();
    return enabled;
}

bool large_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GGUF_LARGE_PREFILL");
        if (!value || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_GGUF_LARGE_PREFILL must be 0 or 1");
    }();
    return enabled;
}

constexpr auto kConfig = ggml_cuda_mmq_get_config_ampere(GGML_TYPE_IQ3_S, 128, false);
static_assert(kConfig.I == 64 && kConfig.J == 128 && kConfig.nthreads == 128);
static_assert(kConfig.occupancy == 1 && kConfig.K_vram == 256 && kConfig.stream_k);
static_assert(kConfig.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 && !kConfig.fallback);

constexpr int kRows = 17408;
constexpr int kK = 5120;
constexpr int kColumns = 512;
constexpr int kShared = 38400;
constexpr int kBlocks = (kRows / kConfig.I) * (kColumns / kConfig.J);
static_assert(kBlocks == 1088);
static_assert(ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::qk == 256);
static_assert(ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::bs == 110);

} // namespace

// A separate non-template entry avoids ODR/weak selection of the baseline's
// launch_matrix/matrix_product_bf16_impl, whose J128 has a different I.
bool try_iq3_s_i64_bf16(const void* weight, std::int64_t row_bytes, int rows, int k,
                       const void* activation, int columns, __nv_bfloat16* out,
                       std::int64_t out_column_stride, cudaStream_t stream) {
    if (!iq3_s_i64_enabled() || rows != kRows || k != kK ||
        (columns != kColumns && (columns != 1024 || !large_prefill_enabled())) ||
        row_bytes != 2200 || out_column_stride != kRows ||
        device_facts().cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    // Keep the represented N/K and full-K arithmetic. R28's existing dynamic
    // columns/blocks organization extends only the opt-in plain-prefill width;
    // T512 retains its original route and T1024 needs independent qualification.
    if (mmq_get_nbytes_shared(kConfig, device_facts().cc) != kShared ||
        device_facts().shared_per_block_optin < kShared) {
        throw std::runtime_error("R17 IQ3_S I64: unexpected shared-memory geometry");
    }

    int device = 0;
    check(cudaGetDevice(&device), "R17 IQ3_S I64 device");
    static std::array<std::once_flag, 16> raised;
    if (device < 0 || device >= int(raised.size())) {
        throw std::invalid_argument("R17 IQ3_S I64: device ordinal out of range");
    }
    std::call_once(raised[device], [] {
        check(cudaFuncSetAttribute(
                  ninfer_r17_iq3s_i64_mul_mat_q<GGML_TYPE_IQ3_S, 128, false,
                                             GGML_PREC_Q8, __nv_bfloat16>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, kShared),
              "R17 IQ3_S I64 shared memory");
        check(cudaFuncSetAttribute(
                  ninfer_r17_iq3s_i64_mul_mat_q<GGML_TYPE_IQ3_S, 128, false,
                                             GGML_PREC_Q8, __nv_bfloat16>,
                  cudaFuncAttributePreferredSharedMemoryCarveout, 100),
              "R17 IQ3_S I64 shared carveout");
    });

    const uint3 blocks_fd = init_fastdiv_values(kK / 256);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(columns / 128);
    const int blocks = (kRows / kConfig.I) * (columns / kConfig.J);
    ninfer_r17_iq3s_i64_mul_mat_q<GGML_TYPE_IQ3_S, 128, false,
                                GGML_PREC_Q8, __nv_bfloat16>
        <<<dim3(blocks, 1, 1), dim3(32, 4, 1), kShared, stream>>>(
            static_cast<const char*>(weight), static_cast<const int*>(activation),
            nullptr, nullptr, out, nullptr, nullptr, blocks_fd, rows, columns,
            static_cast<int>(row_bytes / 110), columns,
            static_cast<int>(out_column_stride), one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
    check(cudaGetLastError(), "R17 IQ3_S I64 product launch");
    trace_matrix_output(GGML_TYPE_IQ3_S, rows, k, columns, 64, 128, 128,
                        blocks, false, true);

    static const bool trace = [] {
        const char* value = std::getenv("NINFER_GGUF_MMQ_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (trace) {
        static std::array<std::once_flag, 2> traced;
        std::call_once(traced[columns == 1024], [columns, blocks] {
            std::fprintf(stderr,
                "[r17-iq3s-i64] kernel=ninfer_r17_iq3s_i64_mul_mat_q "
                "type=21 N=17408 K=5120 T=%d I=64 J=128 threads=128 "
                "blocks=%d shared=38400 fixup=0 output=bf16-direct "
                "carveout_hint=100\n", columns, blocks);
        });
    }
    return true;
}

} // namespace ninfer::ops::gguf::detail
