#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 linear decode: invalid exact problem");
    }

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_exact<Nvfp4AttnInputGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_exact<Nvfp4GdnInputGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_exact<Nvfp4MlpGateUpGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_exact<Nvfp4Residual6144Geometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_exact<Nvfp4Residual17408Geometry>(x, weight, out, stream);
        return;
    }
}

void launch_nvfp4_v3_row_view(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    if (!is_nvfp4_v3_row_view(weight.n, weight.k) || x.ne[0] != weight.k ||
        out.ne[0] != weight.n || x.ne[1] <= 0 || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("nvfp4 v3 row view: unsupported shape");
    }
    // The existing exact GEMV kernel accepts both raw plane pointers and the
    // two divisors. A row view merely adjusts the base of each plane; no
    // conversion, repacking, or shape-guessing is performed.
    for (std::int32_t t = 0; t < x.ne[1]; ++t) {
        const Tensor token_x(static_cast<__nv_bfloat16*>(x.data) +
                                 static_cast<std::int64_t>(t) * weight.k,
                             DType::BF16, {weight.k, 1});
        Tensor token_out(static_cast<__nv_bfloat16*>(out.data) +
                             static_cast<std::int64_t>(t) * weight.n,
                         DType::BF16, {weight.n, 1});
        switch (weight.n) {
        case 1024:
            launch_exact<Nvfp4GemvGeometry<1024, 5120>>(token_x, weight, token_out, stream);
            break;
        case 2048:
            launch_exact<Nvfp4GemvGeometry<2048, 5120>>(token_x, weight, token_out, stream);
            break;
        case 6144:
            launch_exact<Nvfp4GemvGeometry<6144, 5120>>(token_x, weight, token_out, stream);
            break;
        case 17408:
            launch_exact<Nvfp4GemvGeometry<17408, 5120>>(token_x, weight, token_out, stream);
            break;
        default:
            throw std::invalid_argument("nvfp4 v3 row view: unsupported output width");
        }
    }
}

} // namespace ninfer::ops::detail
