#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;

    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 14336 && weight.k == 5120) {
        launch_geometry<Bf16GemvGeometry<14336, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<5120, 6144>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear decode: unsupported exact problem");
}

void launch_bf16_v3_individual(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    if (weight.k != 5120 || (weight.n != 1024 && weight.n != 6144) || x.ne[0] != 5120 ||
        out.ne[0] != weight.n || x.ne[1] <= 0 || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("bf16 v3 individual projection: invalid geometry");
    }
    // Reuse the FP32-accumulating production BF16 GEMV, one token at a time.
    // This fallback preserves source rows and needs no extra workspace; the
    // original larger fused BF16 problems retain their tuned GEMM kernels.
    for (std::int32_t t = 0; t < x.ne[1]; ++t) {
        Tensor token_input(static_cast<__nv_bfloat16*>(x.data) +
                               static_cast<std::int64_t>(t) * weight.k,
                           DType::BF16, {weight.k, 1});
        Tensor token_output(static_cast<__nv_bfloat16*>(out.data) +
                                static_cast<std::int64_t>(t) * weight.n,
                            DType::BF16, {weight.n, 1});
        if (weight.n == 1024) {
            launch_geometry<Bf16GemvGeometry<1024, 5120>>(
                token_input, weight, token_output, stream);
        } else {
            launch_geometry<Bf16GemvGeometry<6144, 5120>>(
                token_input, weight, token_output, stream);
        }
    }
}

} // namespace ninfer::ops::detail
