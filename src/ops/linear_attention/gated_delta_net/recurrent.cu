#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

// Adapted from tuxKOH/ninfer-V100X2 5c515130 prepare_recurrent_qk_kernel (Apache-2.0,
// third_party/ninfer_v100x2/LICENSE). Extend its FP32 staging to native short/batched
// recurrence and raw-record verification, not the donor's Volta wide-prefill route.
__global__ void prepare_recurrent_qk_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                            float* normalized_q, float* normalized_k,
                                            std::int64_t rows) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * kNumWarps + threadIdx.y;
    if (row >= rows) { return; }
    const int lane = threadIdx.x;
    const auto d = static_cast<std::uint32_t>(lane * kQkPerLane);
    RawQkLane qr = load_raw_qk_lane(q + row * kStateDim, d);
    RawQkLane kr = load_raw_qk_lane(k + row * kStateDim, d);
    normalize_qk_lane<true>(qr.value, lane);
    normalize_qk_lane<true>(kr.value, lane);
    store_vec(normalized_q + row * kStateDim + d,
              make_float4(qr.value[0], qr.value[1], qr.value[2], qr.value[3]));
    store_vec(normalized_k + row * kStateDim + d,
              make_float4(kr.value[0], kr.value[1], kr.value[2], kr.value[3]));
}

template <bool NormalizeQK>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_bf16_direct_kernel<NormalizeQK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const float*>(state_read.data),
        static_cast<float*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out,
                                         cudaStream_t stream, const float* prepared_q,
                                         const float* prepared_k) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const BatchUpdateAccess access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        static_cast<float*>(ssm_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        state_slot_stride,
        scale,
    };
    if (prepared_q != nullptr) {
        const PreparedAccess<BatchUpdateAccess> prepared{access, prepared_q, prepared_k};
        recurrent_prepared_kernel<false><<<grid, block, 0, stream>>>(prepared, 1);
    } else {
        recurrent_batch_update_kernel<NormalizeInputs><<<grid, block, 0, stream>>>(access);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Residency target for the record kernel. See the note on the kernel itself: it moves ~148 MB of
// state a round at 265 GB/s, and the question is whether that is a register/occupancy limit or the
// serial dependency chain of the recurrence. Env-selectable so one build can answer it.
int record_min_blocks() {
    static const int mb = [] {
        const char* value = std::getenv("NINFER_GDN_RECORD_MINBLOCKS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 2 || parsed == 6 || parsed == 8) ? parsed : 2;
    }();
    return mb;
}

template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream, const float* prepared_q,
                                   const float* prepared_k) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    if (prepared_q != nullptr) {
        const PreparedAccess<RecordAccess<Masked>> prepared{access, prepared_q, prepared_k};
        switch (record_min_blocks()) {
        case 6:
            recurrent_prepared_kernel<true, RecordAccess<Masked>, 6>
                <<<grid, block, 0, stream>>>(prepared, q.ne[2]);
            break;
        case 8:
            recurrent_prepared_kernel<true, RecordAccess<Masked>, 8>
                <<<grid, block, 0, stream>>>(prepared, q.ne[2]);
            break;
        default:
            recurrent_prepared_kernel<true><<<grid, block, 0, stream>>>(prepared, q.ne[2]);
            break;
        }
    } else {
        switch (record_min_blocks()) {
        case 6: recurrent_record_kernel<Masked, 6><<<grid, block, 0, stream>>>(access); break;
        case 8: recurrent_record_kernel<Masked, 8><<<grid, block, 0, stream>>>(access); break;
        default: recurrent_record_kernel<Masked><<<grid, block, 0, stream>>>(access); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states,
                              const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                              std::int32_t active_width, cudaStream_t stream) {
    const FoldAccess<Geometry> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity * records.spec.width,
        active_width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_prepare_recurrent_qk(const Tensor& q, const Tensor& k, Tensor& normalized_q,
                                  Tensor& normalized_k, cudaStream_t stream) {
    const auto rows = q.ne[1] * q.ne[2] * q.ne[3];
    prepare_recurrent_qk_kernel<<<static_cast<unsigned>((rows + kNumWarps - 1) / kNumWarps),
                                  dim3(kWarpSize, kNumWarps), 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<float*>(normalized_q.data), static_cast<float*>(normalized_k.data), rows);
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_direct_fixed<true>(q, k, v, g, beta, scale, ssm_state, ssm_state, out,
                                            stream);
    } else {
        launch_recurrent_direct_fixed<false>(q, k, v, g, beta, scale, ssm_state, ssm_state, out,
                                             stream);
    }
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream, const float* prepared_q, const float* prepared_k) {
    if (prepared_q != nullptr) {
        const auto heads = head_map::of(q.ne[1], v.ne[1]);
        const dim3 grid(static_cast<unsigned>(v.ne[1]), 1,
                        static_cast<unsigned>(kStateDim / kBlockDv));
        const DirectAccess access{
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
            static_cast<const float*>(beta.data), static_cast<const float*>(ssm_state_in.data),
            static_cast<float*>(ssm_state_out.data), static_cast<__nv_bfloat16*>(out.data), heads,
            static_cast<std::int32_t>(q.ne[2]), scale};
        const PreparedAccess<DirectAccess> prepared{access, prepared_q, prepared_k};
        recurrent_prepared_kernel<false><<<grid, dim3(kWarpSize, kNumWarps), 0, stream>>>(
            prepared, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    } else if (normalize_qk) {
        launch_recurrent_direct_fixed<true>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                            out, stream);
    } else {
        launch_recurrent_direct_fixed<false>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                             out, stream);
    }
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream, const float* prepared_q,
                                   const float* prepared_k) {
    if (normalize_qk) {
        launch_recurrent_batch_update_fixed<true>(q, k, v, g, beta, scale, ssm_states,
                                                  source_state_slots, destination_state_slots, out,
                                                  stream, prepared_q, prepared_k);
    } else {
        launch_recurrent_batch_update_fixed<false>(q, k, v, g, beta, scale, ssm_states,
                                                   source_state_slots, destination_state_slots, out,
                                                   stream, prepared_q, prepared_k);
    }
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream, const float* prepared_q,
                             const float* prepared_k) {
    if (valid_columns.data == nullptr) {
        launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                             initial_state_slots, key_record, value_record,
                                             gate_record, out, stream, prepared_q, prepared_k);
    } else {
        launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                            initial_state_slots, key_record, value_record,
                                            gate_record, out, stream, prepared_q, prepared_k);
    }
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        std::int32_t active_width, cudaStream_t stream) {
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry48x48>(records, states, rows, active_rows, active_width,
                                                   stream);
        return;
    }
    if (records.spec.layers == FoldGeometry30x32::kLayers &&
        records.spec.qk_heads == FoldGeometry30x32::kQkHeads &&
        records.spec.value_heads == FoldGeometry30x32::kValueHeads &&
        records.spec.conv_channels == FoldGeometry30x32::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry30x32>(records, states, rows, active_rows, active_width,
                                                   stream);
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace ninfer::ops::detail::gated_delta_net
