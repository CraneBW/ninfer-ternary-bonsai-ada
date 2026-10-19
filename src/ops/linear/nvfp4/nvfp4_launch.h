#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
// A16 semantic row views of physical v3 NVFP4 parents, including multi-token
// input. No persistent or shared workspace is needed.
void launch_nvfp4_v3_row_view(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
