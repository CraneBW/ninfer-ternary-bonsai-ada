#pragma once

// The fused epilogues of the ternary rungs (G4): a value transform applied at the store of a
// ternary GEMM, so that the elementwise op a consumer would otherwise run as a second kernel -- the
// residual add after linear_add, the silu*mul after linear_swiglu -- is folded into the kernel that
// already has the value in a register.
//
// Why this is not in ternary_s8_scratch.h. That header is the host-safe half of the int8 rung and is
// included by two host translation units (ternary_dispatch.cpp, ternary_rotation.cpp); its opening
// paragraph records what happens when device-only syntax reaches one of them. Everything here is
// __device__-qualified and belongs on the other side of that line, next to the kernels that call it.
// Nothing in the ternary path needs an epilogue to be host-visible: the only TU that constructs one
// is ternary_rowsplit_gemm.cu, and that is a .cu. (<cuda_bf16.h> is included for the residual
// epilogue this header exists to carry -- it reads the __nv_bfloat16 tensor it is about to
// overwrite -- not for the identity below, which needs nothing from it.)
//
// Why a template TYPE with a by-value POD argument, rather than an enum or a function pointer. This
// is copied deliberately from the engine's existing fused-epilogue path, Fp8AddResidualEpilogue in
// ops/linear_add/fp8/fp8_linear_add_epilogue.cuh (constructed at
// ops/linear_add/fp8/fp8_linear_add_a8.cu:27), because that path already solves this exact problem
// for this exact consumer class, and a second mechanism would be a second thing to get wrong:
//
//   * The epilogue's state IS its arguments: a pointer to the tensor the store is about to
//     overwrite, plus the token stride that pointer is indexed by. A function pointer carries
//     neither, and the call would land in a warp that has finished all of its mma and is about to
//     issue four stores -- an indirect call there is paid per stored element, not per CTA.
//   * An enum makes the choice a runtime one, which forces every epilogue's code into every
//     instantiation and puts a branch in front of each store, for a decision that is a compile-time
//     property of the call site. As a type parameter the identity costs nothing at all: it inlines
//     to its own argument, so ptxas sees the program that was shipped before the parameter existed.
//
// The index contract of apply() is the STORE's own index expression, on purpose. The kernel writes
// out[token * out_row_stride + row], so an epilogue handed (row, token) together with that same
// stride recomputes the identical address from the identical two integers -- a fused residual is a
// read-modify-write of the destination and there is no second way to compute where the element
// lives. That is also why apply() is called INSIDE the store's `row < rows && col < tokens` guard
// and never before it: a real epilogue dereferences memory, so applying it to an out-of-range
// (row, token) pair would read or write an element the store itself must not touch.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct TernaryIdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

} // namespace ninfer::ops::detail
