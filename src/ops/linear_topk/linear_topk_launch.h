#pragma once

#include "core/tensor.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void linear_topk_w8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_w8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                                const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, const LinearTopKWorkspace& workspace,
                           cudaStream_t stream);
void linear_topk_q4_m64_launch(const Tensor& hidden, const Weight& head,
                               const Tensor& row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream);
// GGUF fallback: reduce exact FP32 logits into the same 16-key merge layout.
void gguf_linear_topk_keys_launch(const Tensor& logits, const Tensor* id_map,
                                  int valid_rows, const LinearTopKWorkspace& workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
