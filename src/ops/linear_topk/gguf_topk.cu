#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_radix_sort.cuh>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerCta = 256;
using BlockSort = cub::BlockRadixSort<std::uint64_t, kRowsPerCta, 1>;

__global__ __launch_bounds__(kRowsPerCta, 2)
void gguf_topk_from_logits_kernel(const float* __restrict__ logits,
                                  const std::int32_t* __restrict__ id_map,
                                  std::uint64_t* __restrict__ partial_keys,
                                  int total_rows, int valid_rows, int producer_groups) {
    const int tid = static_cast<int>(threadIdx.x);
    const int group = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int row = group * kRowsPerCta + tid;
    __shared__ typename BlockSort::TempStorage storage;

    std::uint64_t key[1] = {0};
    if (row < total_rows && row < valid_rows) {
        const float score = logits[static_cast<std::int64_t>(column) * total_rows + row];
        const std::int32_t token_id = id_map ? id_map[row] : row;
        if (isfinite(score) && token_id >= 0 && token_id < kLinearTopKFullValidRows) {
            key[0] = score_id_order_key(score, token_id);
        }
    }
    BlockSort(storage).SortDescending(key);
    if (tid < kLinearTopK) {
        partial_keys[(static_cast<std::int64_t>(column) * producer_groups + group) *
                         kLinearTopK + tid] = key[0];
    }
}

} // namespace

void gguf_linear_topk_keys_launch(const Tensor& logits, const Tensor* id_map,
                                  int valid_rows, const LinearTopKWorkspace& workspace,
                                  cudaStream_t stream) {
    const int rows = logits.ne[0];
    const int columns = logits.ne[1];
    const int groups = (rows + kRowsPerCta - 1) / kRowsPerCta;
    if (logits.dtype != DType::FP32 || !logits.is_contiguous() ||
        rows < valid_rows || valid_rows <= 0 || columns != workspace.columns ||
        groups != workspace.producer_groups || workspace.rows_per_producer != kRowsPerCta) {
        throw std::invalid_argument("GGUF topk: invalid FP32 logits or workspace");
    }
    const auto* map = id_map ? static_cast<const std::int32_t*>(id_map->data) : nullptr;
    const dim3 grid(static_cast<unsigned>(groups), static_cast<unsigned>(columns));
    gguf_topk_from_logits_kernel<<<grid, kRowsPerCta, 0, stream>>>(
        static_cast<const float*>(logits.data), map,
        static_cast<std::uint64_t*>(workspace.partial_keys.data),
        rows, valid_rows, groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
