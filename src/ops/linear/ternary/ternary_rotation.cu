// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_rotation.h"

#include "core/device.h"
#include "ops/linear/ternary/ternary_rotation_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

static_assert(sizeof(__nv_bfloat16) == 2, "the rotation workspace math assumes a 2-byte BF16");

// Shared validation for both directions: the transform width, the matching sign block, and the
// [k, tokens] BF16 activation.
void require_rotation_operands(const Tensor& x, const Weight& weight, const char* what) {
    const std::int32_t k      = weight.k;
    const std::int32_t tokens = x.ne[1];

    if (k <= 0 || tokens <= 0) {
        throw std::invalid_argument(std::string(what) + ": K and tokens must be positive");
    }
    if ((k % kBlockSize) != 0) {
        throw std::invalid_argument(std::string(what) +
                                    ": K must be a multiple of the block size");
    }
    if (weight.hadamard_signs == nullptr || weight.hadamard_n_blk <= 0) {
        throw std::invalid_argument(std::string(what) + ": weight carries no sign block");
    }
    if (weight.hadamard_n_blk != k / kBlockSize) {
        throw std::invalid_argument(std::string(what) +
                                    ": sign row count does not match the input width");
    }
    if (x.dtype != DType::BF16 || x.data == nullptr) {
        throw std::invalid_argument(std::string(what) + ": activation must be BF16");
    }
    if (x.ne[0] != k) {
        throw std::invalid_argument(std::string(what) +
                                    ": activation width does not match K");
    }
}

// The transform is one warp per (1024-block, token) pair, so the grid is fixed by the data and
// there is no tiling to do -- the only free parameter is how those warps are PACKED, and it is
// worth measuring rather than assuming. A decode-shaped rotation (k=5120, T=3) is 15 warps: at 8
// per block that is two blocks on two of the card's 66 SMs and costs 4.74 us, against 3.40 at one
// warp per block. What is being paid is the latency of one warp's 32-load chain, not bandwidth --
// the wpb=1 time is FLAT in k (17408 costs the same as 5120) and flat in block count (5, 15 and 51
// blocks all measure 3.40 us), so spreading the warps over more SMs is free and packing them
// together is not.
//
// The prefill regime is the opposite case: 5120 warps at one per block would sit on the
// 32-blocks-per-SM resident limit instead of the 48-warp one, so a large grid keeps the original
// packing. The threshold is where one block per SM stops covering the card.
int rotation_warps_per_block(std::int64_t warps) {
    static const int configured = [] {
        const char* value = std::getenv("NINFER_TERNARY_ROTATE_WPB");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8) ? parsed : 0;
    }();
    if (configured != 0) { return configured; }
    return warps <= 128 ? 1 : kWarpsPerBlock;
}

std::int64_t rotation_warps(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::int64_t>(k / kBlockSize) * static_cast<std::int64_t>(tokens);
}

} // namespace

void launch_ternary_rotation(const Tensor& x, Tensor& out, const Weight& weight,
                             cudaStream_t stream) {
    require_rotation_operands(x, weight, "ternary rotation");
    if (out.dtype != DType::BF16 || out.data == nullptr) {
        throw std::invalid_argument("ternary rotation: out must be BF16");
    }
    if (out.ne[0] != weight.k || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("ternary rotation: expected [K,T] x and [K,T] out");
    }

    // The permutation is only admitted when its geometry actually spans the input width; a
    // partially declared permutation would silently transform a malformed basis. It is applied only
    // when the artifact is known to be tile-ordered (see ternary_gdn_perm_enabled()).
    const bool permuted = ternary_gdn_perm_enabled() && weight.hadamard_perm_rep > 1;
    if (permuted) {
        const std::int64_t span = static_cast<std::int64_t>(weight.hadamard_perm_hd) *
                                  weight.hadamard_perm_nk * weight.hadamard_perm_rep;
        if (weight.hadamard_perm_hd <= 0 || weight.hadamard_perm_nk <= 0 ||
            span != weight.k) {
            throw std::invalid_argument(
                "ternary rotation: folded permutation geometry does not span K");
        }
    }
    const std::int32_t perm_hd  = permuted ? weight.hadamard_perm_hd : 0;
    const std::int32_t perm_nk  = permuted ? weight.hadamard_perm_nk : 0;
    const std::int32_t perm_rep = permuted ? weight.hadamard_perm_rep : 1;

    const std::int64_t warps = rotation_warps(weight.k, x.ne[1]);
    const int warp_block     = rotation_warps_per_block(warps);
    ternary_rotate_bf16_kernel<<<static_cast<unsigned>((warps + warp_block - 1) / warp_block),
                                 warp_block * kThreadsPerWarp, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
        weight.hadamard_signs, weight.hadamard_n_blk, weight.k, x.ne[1], perm_hd, perm_nk,
        perm_rep, /*inverse=*/0);
    CUDA_CHECK(cudaGetLastError());
}

void launch_ternary_rotation_inverse_inplace(Tensor& data, const Weight& weight,
                                             cudaStream_t stream) {
    require_rotation_operands(data, weight, "ternary rotation (inverse)");
    // The token-embedding table is the only inverse-mapped weight and carries no permutation
    // (llama-model.cpp gates P on `.ssm_out.` alone), so one declared here is a contract error.
    if (weight.hadamard_perm_rep > 1) {
        throw std::invalid_argument(
            "ternary rotation: the embedding path does not carry a folded permutation");
    }

    const std::int64_t warps = rotation_warps(weight.k, data.ne[1]);
    const int warp_block     = rotation_warps_per_block(warps);
    ternary_rotate_inverse_inplace_bf16_kernel<<<
        static_cast<unsigned>((warps + warp_block - 1) / warp_block),
        warp_block * kThreadsPerWarp, 0, stream>>>(
        static_cast<__nv_bfloat16*>(data.data), weight.hadamard_signs, weight.hadamard_n_blk,
        weight.k, data.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
