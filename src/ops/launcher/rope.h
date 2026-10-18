#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream);

// Replaces the 32 text RoPE divisors, or restores the baked native table when inv_freq_32 is null.
void set_text_rope_frequencies_symbol(const float* inv_freq_32);

} // namespace ninfer::ops::detail
