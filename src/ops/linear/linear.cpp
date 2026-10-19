// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/gguf/gguf_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

// The Qwen3.6 v3 container binds independent semantic M128 rows of two
// grouped Q4 parents. These are not standalone upstream fused tensor shapes,
// but the existing row-split Q4 SIMT/MMA kernels operate on arbitrary N and
// can consume the exact row views without re-encoding. Keep this route local:
// q4_dispatch.cpp is owned by another work session.
bool is_q4_v3_text_slice(std::int32_t n, std::int32_t k) {
    return k == 5120 && (n == 2048 || n == 17408);
}

void q4_v3_slice_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                          LinearPolicy policy, cudaStream_t stream) {
    if (!is_q4_v3_text_slice(w.n, w.k) || x.ne[1] <= 0 ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
        throw std::invalid_argument("Qwen3.6 v3 Q4 slice: unsupported geometry/policy");
    }
    if (x.ne[1] == 1) {
        detail::launch_q4_gemv_r1_w8_direct(x, w, out, stream);
    } else if (x.ne[1] <= 7) {
        detail::launch_q4_simt_r8_c4(x, w, out, stream);
    } else if (x.ne[1] <= 16) {
        detail::launch_q4_simt_r8_c8(x, w, out, stream);
    } else {
        detail::launch_q4_mma_r64_c128(x, w, out, stream);
    }
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (is_gguf(w.qtype)) {
        if (!workspace) {
            throw std::invalid_argument("GGUF linear needs a workspace");
        }
        detail::gguf_linear_dispatch(x, w, out, *workspace, stream);
        return;
    }
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        if (is_q4_v3_text_slice(w.n, w.k)) {
            q4_v3_slice_dispatch(x, w, out, policy, stream);
            return;
        }
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::PTQ1_0_G128:
    case QType::PQ2_0_G128:
        detail::ternary_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }
    if (is_gguf(qtype)) {
        return detail::gguf_linear_workspace_capacity_bytes(
            qtype, output_rows, input_rows, min_tokens, max_tokens);
    }

    switch (qtype) {
    case QType::Q4G64_F16S:
        if (is_q4_v3_text_slice(output_rows, input_rows)) {
            if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
                throw std::invalid_argument("Qwen3.6 v3 Q4 slice: unsupported policy");
            }
            return 0;
        }
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if ((!detail::is_nvfp4_linear_problem(output_rows, input_rows) &&
             !detail::is_nvfp4_v3_row_view(output_rows, input_rows)) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::PTQ1_0_G128:
    case QType::PQ2_0_G128:
        // The GEMM itself decodes inside the kernel and needs no workspace, but the folded basis
        // does: the activation must be mapped into the rotated basis (P, then signs, then the
        // normalized Hadamard) before the matmul, and that needs a [K, T] BF16 scratch. Omitting
        // these cases would make every ternary weight die here with "linear workspace:
        // unsupported weight qtype" before reaching dispatch.
        (void)detail::select_ternary_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_ternary_launch(output_rows, input_rows, max_tokens, policy);
        return detail::ternary_rotation_workspace_bytes(input_rows, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninfer::ops
