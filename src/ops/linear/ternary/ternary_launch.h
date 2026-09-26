// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

#include "core/tensor.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The token tile is the only schedule knob on the reference path: one token per CTA at
// T = 1 (decode), eight otherwise (a small prefill). Both share one kernel template.
//
// out_row_stride is the row count of the OUT tensor's parent allocation (the token stride), so a
// caller can have several weights write disjoint row ranges of one fused output -- the split GDN
// and attention parents do exactly that. Pass w.n when the output is the whole tensor.
//
// The int8 rung needs an activation-quantization scratch buffer, which the caller owns because this
// op runs inside captured CUDA graphs (a lazy cudaMalloc at launch time would be illegal there).
// Callers that have no workspace -- the *_basis entry points used by the fused GDN/attention
// parents -- pass a null scratch and therefore stay on the bf16 rungs.
using TernaryLaunch = void (*)(const Tensor&, const Weight&, Tensor&, std::int32_t, cudaStream_t,
                               TernaryS8Scratch);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);
void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);

// Runs the int8 activation-quantization pass ONCE for a folded activation that several projections
// are about to share, and stamps the shape into `scratch` so their launches skip it.
//
// The fused parents are why this exists: attn_input_proj hands one activation to four
// ternary_dispatch_basis calls and gdn_input_proj hands one to three, and every one of them used to
// redo the absmax reduction and the code pass over the same k x T bytes -- measured at 144 of the
// 400 quantize launches in one prefill, i.e. 36% of that kernel's 8.17 ms, plus the L2 traffic of
// reading x twice per redundant pass.
//
// Costs nothing when the caller does not use it: launch_pq2_mma_s8 quantizes whenever the scratch's
// recorded shape does not match, which is always true for a scratch nobody pre-quantized.
void quantize_ternary_s8_activation(const Tensor& x, TernaryS8Scratch& scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
