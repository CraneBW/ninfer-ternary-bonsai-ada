// zatfung：Turing (sm_75) / Ampere (sm_86) 上的 FP8 A8 路径替身。
//
// 背景
// ----
// FP8 A8 路由最终落到 `mma.sync.aligned.m16n8k32...e4m3.e4m3.f32` —— FP8 tensor core
// 自 Ada (sm_89) 起才有。sm_75/sm_86 完全没有 FP8 tensor-core 通路，因此承载这些
// 指令的翻译单元根本无法为这两档架构编译：
//
//     ptxas ... error : Feature 'mma with FP8 floating point type'
//                       requires .target sm_89 or higher
//     ptxas fatal   : Ptx assembly aborted due to errors
//
// 这些 .cu 由 src/CMakeLists.txt 在 `^(75|86)$` 时从 ninfer_ops 里过滤掉。
// 本文件补上被过滤掉的入口定义，使 Op 边界保持链接完整，并把"路线被误选"
// 变成一条精确的运行时错误，而不是一个链接失败（后者会把真正的原因埋在
// 几百行 LNK2019 里）。
//
// 什么仍然可用
// ------------
// **FP8 权重在 sm_75/sm_86 上照常可用**：`LinearPolicy::A16Only` 会把它路由到
// A16 反量化 GEMM（把 e4m3 解成 bf16 再做 bf16 MMA），相关翻译单元正常参与编译。
// 例：`ops/linear_topk/fp8.cu`、`ops/linear_topk/fp8_m64.cu` 走
// `fp8_e4m3x2_to_bf16x2_bits`，属 A16 路径，故未被过滤。
// **只有 A8（FP8 激活）计算与 FP8 E4M3 KV-cache 注意力不可用。**
//
// 未覆盖的反例：`--kv-dtype fp8` 需要 FP8 注意力内核 → 在 sm_75/sm_86 上请改用
// `--kv-dtype int8` 或 `--kv-dtype bf16`。

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_fp8_a8() {
    throw std::runtime_error(
        "FP8 A8 execution requires an sm_89 (Ada) or newer GPU; "
        "this zatfung build targets sm_75/sm_86, which have no FP8 tensor-core path. "
        "Use an FP8 weight through its A16 dequantizing route instead.");
}

[[noreturn]] void reject_fp8_kv() {
    throw std::runtime_error(
        "FP8 E4M3 KV-cache attention requires an sm_89 (Ada) or newer GPU; "
        "on sm_75/sm_86 use --kv-dtype int8 or --kv-dtype bf16.");
}

} // namespace

// --- Linear / projection A8 路由 ---------------------------------------------------------------

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    reject_fp8_a8();
}

// --- FP8 KV-cache 因果注意力 --------------------------------------------------------------------

void causal_attention_small_t_fp8_launch(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float,
    PagedKVBatchLayerView, CausalAttentionExecutionEnvelope, std::int32_t, std::int32_t, Tensor&,
    Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_cached_small_t_fp8_launch(const Tensor&, const Tensor&, float,
                                                const PagedKVLayerView&,
                                                CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                Tensor&, Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_prompt_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                        Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_prompt_fp8_attention_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&, Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

} // namespace ninfer::ops::detail
