#pragma once

// zatfung 架构特性表（自 ninfer-2080ti-22g 的 Turing port 引入，并补上 Ada 特化）。
//
// 用途：把"这一档架构有什么、没有什么"收敛成一处编译期常量，让内核层用
//
//     if constexpr (CurrentArchTraits::has_cp_async) { ... } else { ... }
//
// 表达降级，而不是在每个文件里散落 #if defined(NINFER_SMxx)。
// 迁移是渐进的：本文件先落地并随 SM75/SM86 门控合并逐处接入，现有代码继续用
// NINFER_SM* 宏工作，两者并存不冲突。
//
// 数值来源：
//   - SM75  : Turing (TU102/TU104)，64 KiB shared/SM，无 cp.async（cp.async 自 sm_80 起），
//             无 bf16/tf32 tensor core（Turing 为 fp16/int8 MMA）
//   - SM86  : Ampere (GA102/GA10x)，100 KiB shared/SM 可分配，第 3 代 tensor core
//   - SM89  : Ada (AD102/AD10x)，与 SM86 同为 100 KiB/SM 档位；本 fork 的原始目标线
//   - SM120A: Blackwell (GB202)，高共享内存调度，含 NVFP4 与 PDL
//
// 注意 CudaArch 只列四档：SM89 与 SM86 在 shared-memory / MMA 形状上同档，
// 因此 traits 数值相同，但保留独立枚举以便 Ada 专属路径（如 FP8 tensor core、
// sergiuszm 线的 i8 KV 内核 rk4v4/rk4v4-e8）能按档位区分。

#include <cstddef>

namespace ninfer::core {

enum class CudaArch {
    SM75,   // Turing  (e.g. RTX 2080 Ti, TU102)
    SM86,   // Ampere  (e.g. RTX 3090, GA102 / RTX 3060 Ti, GA104)
    SM89,   // Ada     (e.g. RTX 4090, AD102) — 本 fork 原始目标
    SM120A, // Blackwell (e.g. RTX 5090, GB202)
};

template <CudaArch Arch>
struct ArchTraits;

template <>
struct ArchTraits<CudaArch::SM75> {
    static constexpr CudaArch arch = CudaArch::SM75;
    static constexpr const char* name = "sm_75";

    static constexpr bool has_cp_async  = false; // cp.async 自 sm_80 引入
    static constexpr bool has_bf16_mma  = false; // Turing 只有 fp16/int8 tensor core
    static constexpr bool has_tf32_mma  = false;
    static constexpr bool has_pdl       = false; // PDL 自 sm_90 引入
    static constexpr bool has_nvfp4     = false;
    static constexpr bool has_ldmatrix  = true;

    // MMA tile dimensions
    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 8;

    static constexpr int int8_mma_m = 8;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 16;

    // Hardware limits
    static constexpr std::size_t max_smem_per_sm = 64 * 1024;
    static constexpr int max_warps_per_sm        = 32;
    static constexpr int max_threads_per_sm      = 1024;
    static constexpr int max_blocks_per_sm       = 16;
};

template <>
struct ArchTraits<CudaArch::SM86> {
    static constexpr CudaArch arch = CudaArch::SM86;
    static constexpr const char* name = "sm_86";

    static constexpr bool has_cp_async  = true;
    static constexpr bool has_bf16_mma  = true;
    static constexpr bool has_tf32_mma  = true;
    static constexpr bool has_pdl       = false;
    static constexpr bool has_nvfp4     = false;
    static constexpr bool has_ldmatrix  = true;

    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 16;

    static constexpr int int8_mma_m = 16;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 32;

    static constexpr std::size_t max_smem_per_sm = 100 * 1024;
    static constexpr int max_warps_per_sm        = 48;
    static constexpr int max_threads_per_sm      = 1536;
    static constexpr int max_blocks_per_sm       = 16;
};

// Ada 与 Ampere 在共享内存档位与 MMA 形状上同档，故数值一致；
// 差异只存在于具体指令支持（FP8 tensor core）与派生架构的调度选择上，
// 那些走 NINFER_SM89 门控，不在这里表达。
template <>
struct ArchTraits<CudaArch::SM89> {
    static constexpr CudaArch arch = CudaArch::SM89;
    static constexpr const char* name = "sm_89";

    static constexpr bool has_cp_async  = true;
    static constexpr bool has_bf16_mma  = true;
    static constexpr bool has_tf32_mma  = true;
    static constexpr bool has_pdl       = false;
    static constexpr bool has_nvfp4     = false;
    static constexpr bool has_ldmatrix  = true;

    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 16;

    static constexpr int int8_mma_m = 16;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 32;

    static constexpr std::size_t max_smem_per_sm = 100 * 1024;
    static constexpr int max_warps_per_sm        = 48;
    static constexpr int max_threads_per_sm      = 1536;
    static constexpr int max_blocks_per_sm       = 16;
};

template <>
struct ArchTraits<CudaArch::SM120A> {
    static constexpr CudaArch arch = CudaArch::SM120A;
    static constexpr const char* name = "sm_120a";

    static constexpr bool has_cp_async  = true;
    static constexpr bool has_bf16_mma  = true;
    static constexpr bool has_tf32_mma  = true;
    static constexpr bool has_pdl       = true;
    static constexpr bool has_nvfp4     = true;
    static constexpr bool has_ldmatrix  = true;

    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 16;

    static constexpr int int8_mma_m = 16;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 32;

    static constexpr std::size_t max_smem_per_sm = 100 * 1024;
    static constexpr int max_warps_per_sm        = 48;
    static constexpr int max_threads_per_sm      = 1536;
    static constexpr int max_blocks_per_sm       = 32;
};

// 当前编译目标。宏由顶层 CMakeLists.txt 按 CMAKE_CUDA_ARCHITECTURES 定义；
// 未定义即 120a 参考实现（与上游行为一致）。
#if defined(NINFER_SM75)
using CurrentArchTraits = ArchTraits<CudaArch::SM75>;
#elif defined(NINFER_SM86)
using CurrentArchTraits = ArchTraits<CudaArch::SM86>;
#elif defined(NINFER_SM89)
using CurrentArchTraits = ArchTraits<CudaArch::SM89>;
#else
using CurrentArchTraits = ArchTraits<CudaArch::SM120A>;
#endif

} // namespace ninfer::core
