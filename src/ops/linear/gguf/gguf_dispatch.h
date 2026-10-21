#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] gguf::GgmlType gguf_type(QType format);

// Bound scratch for all token widths in [min_tokens, max_tokens].
[[nodiscard]] std::size_t gguf_linear_workspace_capacity_bytes(
    QType qtype, std::int32_t rows, std::int32_t k, std::int32_t min_tokens,
    std::int32_t max_tokens);

// Direct GGUF block products, byte-for-byte weights and BF16 activation.
void gguf_linear_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                          WorkspaceArena& workspace, cudaStream_t stream);

// IndividualGdn-only packed BF16 rows: K=5120, Q/K N=2048 at rows
// 0/2048 or V N=6144 at row 4096, in contiguous [10240,T]. The caller
// owns the destination; no intermediate output allocation or copy is added.
void gguf_linear_gdn_rows_bf16_dispatch(
    const Tensor& x, const Weight& weight, Tensor& packed, std::int32_t row_begin,
    WorkspaceArena& workspace, cudaStream_t stream);

// Main Individual MLP only: gate/up are already BF16 [17408,512].
// The activation is explicitly BF16-RN before the unchanged GGML quantizer.
// This writes the original projected BF16 [5120,512]; residual remains separate.
void gguf_linear_silu_mul_down_dispatch(
    const Tensor& gate, const Tensor& up, const Weight& weight, Tensor& projected,
    WorkspaceArena& workspace, cudaStream_t stream);

// Exact FP32 accumulation output for vocabulary ranking. Unlike BF16 logits,
// it preserves close winners and deterministic score/id ties for TopK.
void gguf_linear_dispatch_f32(const Tensor& x, const Weight& weight, Tensor& out,
                              WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
