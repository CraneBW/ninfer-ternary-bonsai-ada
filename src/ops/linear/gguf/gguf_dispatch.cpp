#include "ops/linear/gguf/gguf_dispatch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

gguf::GgmlType gguf_type(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return gguf::GgmlType::Q8_0;
    case QType::GGUF_Q2_K: return gguf::GgmlType::Q2_K;
    case QType::GGUF_Q3_K: return gguf::GgmlType::Q3_K;
    case QType::GGUF_Q4_K: return gguf::GgmlType::Q4_K;
    case QType::GGUF_Q5_K: return gguf::GgmlType::Q5_K;
    case QType::GGUF_Q6_K: return gguf::GgmlType::Q6_K;
    case QType::GGUF_IQ2_XXS: return gguf::GgmlType::IQ2_XXS;
    case QType::GGUF_IQ2_XS: return gguf::GgmlType::IQ2_XS;
    case QType::GGUF_IQ2_S: return gguf::GgmlType::IQ2_S;
    case QType::GGUF_IQ3_XXS: return gguf::GgmlType::IQ3_XXS;
    case QType::GGUF_IQ3_S: return gguf::GgmlType::IQ3_S;
    case QType::GGUF_IQ1_S: return gguf::GgmlType::IQ1_S;
    case QType::GGUF_IQ1_M: return gguf::GgmlType::IQ1_M;
    case QType::GGUF_IQ4_NL: return gguf::GgmlType::IQ4_NL;
    case QType::GGUF_IQ4_XS: return gguf::GgmlType::IQ4_XS;
    case QType::GGUF_Q4_0: return gguf::GgmlType::Q4_0;
    case QType::GGUF_Q5_0: return gguf::GgmlType::Q5_0;
    case QType::GGUF_Q2_0: return gguf::GgmlType::Q2_0;
    default: throw std::invalid_argument("GGUF product: unsupported block encoding");
    }
}

namespace {

constexpr std::int32_t kMatrixBatchColumns = 128;
constexpr std::size_t kWorkspaceAlignmentBudget = 4 * 256;

void require_weight(const Weight& w) {
    if (!is_gguf(w.qtype) || w.layout != QuantLayout::GgufBlocks || !w.qdata ||
        w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("GGUF product: expected resident GGUF blocks");
    }
    const GgufBlockShape block = gguf_block_shape(w.qtype);
    if (!block.elements || !block.bytes || w.k % block.elements != 0) {
        throw std::invalid_argument("GGUF product: K must comprise whole blocks");
    }
}

std::size_t workspace_bytes(QType format, std::int32_t rows, std::int32_t k,
                            std::int32_t max_tokens) {
    const GgufBlockShape block = gguf_block_shape(format);
    if (!is_gguf(format) || rows <= 0 || k <= 0 || max_tokens <= 0 ||
        block.elements == 0 || k % block.elements != 0 || k % 32 != 0) {
        throw std::invalid_argument("GGUF workspace: unsupported shape");
    }
    const auto type = gguf_type(format);
    const std::size_t vector = gguf::vector_activation_bytes(k, 8);
    std::size_t widest = vector;
    if (max_tokens > gguf::kMaxVectorColumns && rows % 128 == 0 && k % 256 == 0 &&
        gguf::has_matrix_kernel(type)) {
        const int width = std::min(kMatrixBatchColumns, max_tokens);
        const std::size_t matrix =
            gguf::matrix_activation_bytes(k, width) +
            static_cast<std::size_t>(rows) * width * sizeof(float) +
            gguf::matrix_fixup_bytes(type, rows, width);
        widest = std::max(widest, matrix);
    }
    return widest + kWorkspaceAlignmentBudget;
}

} // namespace

std::size_t gguf_linear_workspace_capacity_bytes(
    QType qtype, std::int32_t rows, std::int32_t k,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("GGUF workspace: invalid token interval");
    }
    return workspace_bytes(qtype, rows, k, max_tokens);
}

void gguf_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    require_weight(w);
    if (!x.data || !out.data || x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        !x.is_contiguous() || !out.is_contiguous() || x.ne[0] != w.k ||
        out.ne[0] != w.n || x.ne[1] != out.ne[1] || x.ne[1] <= 0) {
        throw std::invalid_argument("GGUF product: expected BF16 [K,T] -> [N,T]");
    }

    const auto type = gguf_type(w.qtype);
    const auto block = gguf_block_shape(w.qtype);
    const std::int64_t row_bytes =
        static_cast<std::int64_t>(w.k / block.elements) * block.bytes;
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    const bool wide_kernel =
        gguf::has_matrix_kernel(type) && w.k % 256 == 0 && w.n % 128 == 0;

    for (int first = 0; first < x.ne[1];) {
        // IQ1_M has no integer MMQ kernel, and widths not divisible by 256
        // cannot safely use ggml's padded MMQ. Both remain on vector products.
        const bool use_matrix = wide_kernel && x.ne[1] - first > gguf::kMaxVectorColumns;
        const int count = std::min(
            x.ne[1] - first, use_matrix ? kMatrixBatchColumns : gguf::kMaxVectorColumns);
        auto scope = workspace.scope();
        if (!use_matrix) {
            void* codes = workspace.alloc_bytes(
                gguf::vector_activation_bytes(w.k, count)).data;
            gguf::quantize_vector_activation(input + static_cast<std::int64_t>(first) * w.k,
                                             w.k, count, w.input_columns, codes, stream);
            gguf::VectorOutput target;
            target.bf16 = output + static_cast<std::int64_t>(first) * w.n;
            target.column_stride = w.n;
            gguf::vector_product(type, w.qdata, row_bytes, w.n, w.k, codes,
                                 count, target, stream);
        } else {
            void* codes = workspace.alloc_bytes(
                gguf::matrix_activation_bytes(w.k, count)).data;
            auto* plane = static_cast<float*>(
                workspace.alloc_bytes(static_cast<std::size_t>(w.n) * count * sizeof(float)).data);
            const std::size_t fixup_bytes =
                gguf::matrix_fixup_bytes(type, w.n, count);
            void* fixup = fixup_bytes ? workspace.alloc_bytes(fixup_bytes).data : nullptr;
            gguf::quantize_matrix_activation(type,
                input + static_cast<std::int64_t>(first) * w.k, w.k, count,
                w.input_columns, codes, stream);
            gguf::matrix_product(type, w.qdata, row_bytes, w.n, w.k, codes, count,
                                 plane, w.n, fixup, stream);
            gguf::VectorOutput target;
            target.bf16 = output + static_cast<std::int64_t>(first) * w.n;
            target.column_stride = w.n;
            gguf::store_plane(plane, w.n, w.n, count, target, stream);
        }
        first += count;
    }
}

} // namespace ninfer::ops::detail
