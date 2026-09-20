// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy);

// Rotate the activation into the folded basis when the weight needs it, then run the GEMM.
void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream);

// Run the GEMM when the activation is ALREADY in the folded basis. Callers that feed several
// folded weights from one activation (the split GDN input projection feeds three) rotate once and
// then call this, instead of paying for -- and diverging on -- a redundant rotation per weight.
void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream);

// Same, for a weight that writes a ROW RANGE of a larger fused output. `out` must already point at
// the range start (Tensor::slice over ne[0], the contiguous axis, does that) and out_row_stride is
// the parent's row count -- which is also the token stride under ninfer's ne[0]-contiguous layout.
void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
