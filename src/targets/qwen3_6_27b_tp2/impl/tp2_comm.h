#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {

// NCCL communication executes on the same stream as rank-local CUDA kernels.
// Both ranks must issue identical collective sequences and shapes.
void init();
[[nodiscard]] int rank();
void allreduce(Tensor& residual, cudaStream_t stream);
void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
