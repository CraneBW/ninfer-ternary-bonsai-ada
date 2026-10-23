// ninfer::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope.cuh"

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {

// Baked native table: theta = 1e7 over the 64-wide text rotary dimension, one divisor per rotated
// pair. A YaRN-extended Program replaces it through set_text_rope_frequencies().
__device__ __constant__ float kTextRopeInvFrequency[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

__device__ __constant__ int kTextRopeLongPrecision = 1;
__device__ __constant__ double kTextRopeInvFrequencyWide[32] = {
    1, 0.60429639023813286, 0.36517412725483772, 0.22067340690845899,
    0.1333521432163324, 0.080584218776148187, 0.048696752516586311, 0.029427271762092817,
    0.017782794100389229, 0.010746078283213174, 0.006493816315762113, 0.0039241897584845363,
    0.0023713737056616554, 0.0014330125702369627, 0.00086596432336006539, 0.00052329911468149473,
    0.00031622776601683794, 0.00019109529749704405, 0.00011547819846894582, 6.9783058485986635e-05,
    4.2169650342858222e-05, 2.5482967479793464e-05, 1.5399265260594919e-05, 9.3057204092969904e-06,
    5.6234132519034912e-06, 3.3982083289425593e-06, 2.0535250264571461e-06, 1.2409377607517195e-06,
    7.4989420933245585e-07, 4.5315836376008179e-07, 2.7384196342643614e-07, 1.6548170999431814e-07
};

} // namespace ninfer::ops

namespace ninfer::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// One CTA per token: the 256-thread block is the fastest choice only while the whole launch is
// resident. Six of these CTAs fit per SM, so the wave ends at six times the device SM count;
// past that the narrower full-chunk block wins.
constexpr int kLargeBlockCtasPerSm = 6;

int large_block_wave_capacity() { return kLargeBlockCtasPerSm * device_sm_count(); }

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block,
                        cudaStream_t stream) {
    const int tokens = positions.ne[0];
    rope_fixed_kernel<Mode, QHeads, KHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k));
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= large_block_wave_capacity()) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, stream);
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock>
        <<<tokens * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
            k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
            token_stride(k));
}

bool launch_fixed_pair(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                       cudaStream_t stream) {
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160, stream);
        }
        return true;
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, stream);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, stream);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, stream);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads>
void launch_fixed_single(const Tensor& positions, Tensor& x, cudaStream_t stream) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, stream);
}

template <int Heads>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x, cudaStream_t stream) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, stream);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, stream);
        return true;
    }
    return false;
}

bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                                  cudaStream_t stream) {
    if (!bf16x2_aligned(x)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, stream);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, stream);
            return true;
        }
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (launch_text_single<24>(positions, axes, x, stream) ||
            launch_text_single<4>(positions, axes, x, stream) ||
            launch_text_single<16>(positions, axes, x, stream) ||
            launch_text_single<2>(positions, axes, x, stream)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, stream);
        return true;
    }
    return false;
}

void launch_generic(const Tensor& positions, int rotary_dim, float theta, Tensor* q, Tensor* k,
                    cudaStream_t stream) {
    constexpr int block = 128;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    rope_generic_kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), positions.ne[1],
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens, token_stride(q),
        token_stride(k));
}

// Native table used to restore the baked divisors when a Program publishes no YaRN scaling.
constexpr float kDefaultTextRopeInvFrequency[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

} // namespace

void set_text_rope_frequencies_symbol(const float* inv_freq_32) {
    const float* source = inv_freq_32 != nullptr ? inv_freq_32 : kDefaultTextRopeInvFrequency;
    CUDA_CHECK(cudaMemcpyToSymbol(kTextRopeInvFrequency, source, sizeof(float) * 32));
    double wide[32];
    for (int pair=0;pair<32;++pair) {
        wide[pair] = inv_freq_32 ? static_cast<double>(inv_freq_32[pair])
            : std::pow(1.0e7,-static_cast<double>(pair)/32.0);
    }
    const char* requested=std::getenv("NINFER_TEXT_ROPE_LONG_PRECISION");
    if (requested && std::strcmp(requested,"0") && std::strcmp(requested,"1")) {
        throw std::invalid_argument("NINFER_TEXT_ROPE_LONG_PRECISION must be 0 or 1");
    }
    const int precise=(!requested || std::strcmp(requested,"0")) ? 1 : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(kTextRopeInvFrequencyWide,wide,sizeof(wide)));
    CUDA_CHECK(cudaMemcpyToSymbol(kTextRopeLongPrecision,&precise,sizeof(precise)));
}

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream) {
    if (!launch_fixed_pair(positions, rotary_dim, theta, q, k, stream)) {
        launch_generic(positions, rotary_dim, theta, &q, &k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream) {
    if (!launch_fixed_single_dispatch(positions, rotary_dim, theta, x, stream)) {
        launch_generic(positions, rotary_dim, theta, &x, nullptr, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
