#pragma once

#include "core/arena.h"
#include "ops/linear_attention/gated_delta_net/common.h"

namespace ninfer::ops::detail::gated_delta_net {

// Startup-fixed so planning and CUDA Graph execution use the same storage contract.
bool recurrent_qk_prepared_enabled();

struct RecurrentQkScratch {
    Tensor q;
    Tensor k;
};

template <class Allocator>
RecurrentQkScratch allocate_recurrent_qk(Allocator& allocator, std::int32_t heads,
                                        std::int32_t tokens) {
    return {allocator.alloc(DType::FP32, {kStateDim, heads, tokens}),
            allocator.alloc(DType::FP32, {kStateDim, heads, tokens})};
}

} // namespace ninfer::ops::detail::gated_delta_net
