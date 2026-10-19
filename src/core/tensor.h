// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S           = 0,
    Q5G64_F16S           = 1,
    Q6G64_F16S           = 2,
    W8G32_F16S           = 3,
    BF16_CTRL            = 4,
    FP32_CTRL            = 5,
    I32_CTRL             = 6,
    NVFP4                = 7,
    FP8_E4M3FN_ROW_BF16S = 8,
    // Prism-private ternary (Bonsai 2 27B): group 128, base-3 / 2-bit codes.
    PTQ1_0_G128          = 9,
    PQ2_0_G128           = 10,
    // ggml/GGUF blocks are retained byte-for-byte in v3 artifacts.
    GGUF_Q8_0            = 11,
    GGUF_Q2_K            = 12,
    GGUF_Q3_K            = 13,
    GGUF_Q4_K            = 14,
    GGUF_Q5_K            = 15,
    GGUF_Q6_K            = 16,
    GGUF_IQ2_XXS         = 17,
    GGUF_IQ2_XS          = 18,
    GGUF_IQ2_S           = 19,
    GGUF_IQ3_XXS         = 20,
    GGUF_IQ3_S           = 21,
    GGUF_IQ1_S           = 22,
    GGUF_IQ1_M           = 23,
    GGUF_IQ4_NL          = 24,
    GGUF_IQ4_XS          = 25,
    GGUF_Q4_0            = 26,
    GGUF_Q5_0            = 27,
    GGUF_Q2_0            = 28,
    T2_G128_FP16         = 29,
    Q2_G64_FP16         = 30,
    FP8_E4M3FN_ROW_FP16 = 31,
};

[[nodiscard]] constexpr bool is_gguf(QType format) {
    return format >= QType::GGUF_Q8_0 && format <= QType::GGUF_Q2_0;
}

struct GgufBlockShape {
    int elements = 0;
    int bytes = 0;
};

[[nodiscard]] constexpr GgufBlockShape gguf_block_shape(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return {32, 34};
    case QType::GGUF_Q2_K: return {256, 84};
    case QType::GGUF_Q3_K: return {256, 110};
    case QType::GGUF_Q4_K: return {256, 144};
    case QType::GGUF_Q5_K: return {256, 176};
    case QType::GGUF_Q6_K: return {256, 210};
    case QType::GGUF_IQ2_XXS: return {256, 66};
    case QType::GGUF_IQ2_XS: return {256, 74};
    case QType::GGUF_IQ2_S: return {256, 82};
    case QType::GGUF_IQ3_XXS: return {256, 98};
    case QType::GGUF_IQ3_S: return {256, 110};
    case QType::GGUF_IQ1_S: return {256, 50};
    case QType::GGUF_IQ1_M: return {256, 56};
    case QType::GGUF_IQ4_NL: return {32, 18};
    case QType::GGUF_IQ4_XS: return {256, 136};
    case QType::GGUF_Q4_0: return {32, 18};
    case QType::GGUF_Q5_0: return {32, 22};
    case QType::GGUF_Q2_0: return {64, 18};
    default: return {};
    }
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    GgufBlocks          = 4,
    RowInterleaved      = 5,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // Prism ternary weights are stored folded into a rotated basis: the model computes
    // y = W' * (H * (s * P * x)), so every activation feeding one of the folded weights must
    // be mapped by (signs, then normalized Sylvester-Hadamard) before the matmul, and the
    // token-embedding lookup must be mapped by (rotate, then signs) afterwards.
    //
    // hadamard_signs points at the sign block for THIS weight's input width (each width owns
    // k/1024 rows of 1024 F32 +-1 values, and row b within the width is selected by the
    // activation's 1024-block index). nullptr means no transform. hadamard_n_blk = k/1024.
    const float* hadamard_signs = nullptr;
    std::int32_t hadamard_n_blk = 0;

    // Folded-basis feature permutation P, applied BEFORE the signs and the rotation:
    //
    //     x.view(perm_hd, perm_nk, perm_rep) -> swap the last two axes -> flatten
    //
    // Only the GDN output projection stores grouped columns while the runtime produces tiled
    // v-heads, so perm_rep > 1 enables it and everything else leaves it at 1. Geometry comes
    // from the model's head counts: perm_hd = K/n_v, perm_nk = n_k, perm_rep = n_v/n_k.
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;

    // Optional ggml input gather, used for Qwen GDN column permutations.
    const std::int32_t* input_columns = nullptr;
    // Per-source divisor metadata for v3 NVFP4 stacked planes.
    const void* weight_divisors = nullptr;
    std::int32_t weight_divisor_rows = 0;
};

} // namespace ninfer
