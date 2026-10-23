#include "ops/linear/gguf/gguf_dispatch.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::gguf {
void quantize_matrix_silu_mul_activation(
    GgmlType type, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int k, int columns, void* out, cudaStream_t stream);
}

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

constexpr std::size_t kWorkspaceAlignmentBudget = 4 * 256;

bool large_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GGUF_LARGE_PREFILL");
        if (!value || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_GGUF_LARGE_PREFILL must be 0 or 1");
    }();
    return enabled;
}

std::int32_t matrix_batch_columns() {
    // Reuse the wide MMQ route already used by gguf_project. Ada's 48MiB L2
    // benefits from issuing adjacent column tiles together instead of reading
    // the whole weight again for each 128-token slice. Keep other devices on
    // their existing policy; neither activation quantization nor MMQ changes.
    thread_local int previous_device = -1;
    thread_local std::int32_t columns = 128;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        throw std::runtime_error("GGUF batch policy: cannot query current device");
    }
    if (device != previous_device) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
            throw std::runtime_error("GGUF batch policy: cannot query compute capability");
        }
        // The same cap sizes workspace and slices execution. Raising only
        // the public prefill chunk would otherwise still issue two T512 Ops.
        columns = major == 8 && minor == 9 ? (large_prefill_enabled() ? 1024 : 512) : 128;
        previous_device = device;
    }
    return columns;
}

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
        const int width = std::min(matrix_batch_columns(), max_tokens);
        // A shorter final slice can need stream-k fixup even when the widest
        // grid divides evenly. Use gguf_project's existing all-tile bound.
        const auto fixup = static_cast<std::size_t>(gguf::multiprocessor_count()) *
                           128 * 128 * sizeof(float);
        const std::size_t matrix =
            gguf::matrix_activation_bytes(k, width) +
            static_cast<std::size_t>(rows) * width * sizeof(float) +
            fixup;
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

void gguf_linear_dispatch_impl(const Tensor& x, const Weight& w, Tensor& out,
                               WorkspaceArena& workspace, cudaStream_t stream,
                               bool fp32_out, bool gdn_rows = false, std::int32_t row_begin = 0,
                               const Tensor* silu_up = nullptr) {
    require_weight(w);
    if (!x.data || !out.data || x.dtype != DType::BF16 ||
        out.dtype != (fp32_out ? DType::FP32 : DType::BF16) ||
        !x.is_contiguous() || !out.is_contiguous() || x.ne[0] != w.k ||
        (!gdn_rows && out.ne[0] != w.n) ||
        (gdn_rows && (fp32_out || w.k != 5120 || out.ne[0] != 10240 ||
                      !((w.n == 2048 && (row_begin == 0 || row_begin == 2048)) ||
                        (w.n == 6144 && row_begin == 4096)))) ||
        x.ne[1] != out.ne[1] || x.ne[1] <= 0) {
        throw std::invalid_argument("GGUF product: invalid activation or output geometry");
    }

    // The GDN caller owns one packed [Q,K,V] buffer. Only the BF16 store
    // address/stride changes; activation codes, product and rounding stay here.
    const std::int32_t output_stride = gdn_rows ? 10240 : w.n;
    const auto type = gguf_type(w.qtype);
    if (silu_up != nullptr &&
        (fp32_out || gdn_rows || w.k != 17408 || w.n != 5120 || x.ne[1] != 512 ||
         w.input_columns != nullptr || !silu_up->data || silu_up->dtype != DType::BF16 ||
         !silu_up->is_contiguous() || silu_up->ne[0] != w.k || silu_up->ne[1] != 512 ||
         !gguf::has_matrix_kernel(type) ||
         std::min(matrix_batch_columns(), static_cast<std::int32_t>(x.ne[1])) != 512)) {
        throw std::invalid_argument("R29 SiLU/down: expected contiguous main GGUF K17408 N5120 T512 without gather");
    }
    const auto block = gguf_block_shape(w.qtype);
    const std::int64_t row_bytes =
        static_cast<std::int64_t>(w.k / block.elements) * block.bytes;
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const bool wide_kernel =
        gguf::has_matrix_kernel(type) && w.k % 256 == 0 && w.n % 128 == 0;

    // R29 T16 proof only; adapted from R20 scratch checked dispatch counters.
    // Count the route after each actual checked enqueue, including the direct
    // BF16 early-continue case. Trace=0 leaves the product policy unchanged.
    const bool trace_t16 = x.ne[1] == 16 && !fp32_out && !gdn_rows && [] {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_R29_BUNDLE_TRACE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        return enabled;
    }();
    int matrix_calls = 0, vector_calls = 0;
    int vector_columns[2] = {0, 0};
    for (int first = 0; first < x.ne[1];) {
        // IQ1_M has no integer MMQ kernel, and widths not divisible by 256
        // cannot safely use ggml's padded MMQ. Both remain on vector products.
        const bool use_matrix = wide_kernel && x.ne[1] - first > gguf::kMaxVectorColumns;
        const int count = std::min(
            x.ne[1] - first, use_matrix ? matrix_batch_columns() : gguf::kMaxVectorColumns);
        auto scope = workspace.scope();
        if (!use_matrix) {
            void* codes = workspace.alloc_bytes(
                gguf::vector_activation_bytes(w.k, count)).data;
            gguf::quantize_vector_activation(input + static_cast<std::int64_t>(first) * w.k,
                                             w.k, count, w.input_columns, codes, stream);
            gguf::VectorOutput target;
            if (fp32_out) {
                target.f32 = static_cast<float*>(out.data) +
                             static_cast<std::int64_t>(first) * output_stride + row_begin;
            } else {
                target.bf16 = static_cast<__nv_bfloat16*>(out.data) +
                              static_cast<std::int64_t>(first) * output_stride + row_begin;
            }
            target.column_stride = output_stride;
            gguf::vector_product(type, w.qdata, row_bytes, w.n, w.k, codes,
                                 count, target, stream);
            if (trace_t16) {
                if (vector_calls < 2) { vector_columns[vector_calls] = count; }
                ++vector_calls;
            }
        } else {
            void* codes = workspace.alloc_bytes(
                gguf::matrix_activation_bytes(w.k, count)).data;
            if (silu_up != nullptr) {
                gguf::quantize_matrix_silu_mul_activation(
                    type, input + static_cast<std::int64_t>(first) * w.k,
                    static_cast<const __nv_bfloat16*>(silu_up->data) +
                        static_cast<std::int64_t>(first) * w.k,
                    w.k, count, codes, stream);
            } else {
                gguf::quantize_matrix_activation(type,
                    input + static_cast<std::int64_t>(first) * w.k, w.k, count,
                    w.input_columns, codes, stream);
            }
            if (!fp32_out && gguf::try_matrix_product_bf16(
                    type, w.qdata, row_bytes, w.n, w.k, codes, count,
                    static_cast<__nv_bfloat16*>(out.data) +
                        static_cast<std::int64_t>(first) * output_stride + row_begin,
                    output_stride, stream)) {
                if (trace_t16) { ++matrix_calls; }
                first += count;
                continue;
            }
            auto* plane = fp32_out
                              ? static_cast<float*>(out.data) +
                                    static_cast<std::int64_t>(first) * w.n
                              : static_cast<float*>(workspace.alloc_bytes(
                                    static_cast<std::size_t>(w.n) * count * sizeof(float)).data);
            const std::size_t fixup_bytes =
                gguf::matrix_fixup_bytes(type, w.n, count);
            void* fixup = fixup_bytes ? workspace.alloc_bytes(fixup_bytes).data : nullptr;
            gguf::matrix_product(type, w.qdata, row_bytes, w.n, w.k, codes, count,
                                 plane, w.n, fixup, stream);
            if (!fp32_out) {
                gguf::VectorOutput target;
                target.bf16 = static_cast<__nv_bfloat16*>(out.data) +
                              static_cast<std::int64_t>(first) * output_stride + row_begin;
                target.column_stride = output_stride;
                gguf::store_plane(plane, w.n, w.n, count, target, stream);
            }
            if (trace_t16) { ++matrix_calls; }
        }
        first += count;
    }
    if (trace_t16) {
        const char* policy = std::getenv("NINFER_R29_BUNDLE");
        const int selected = policy && policy[0] == '1' && policy[1] == '\0';
        std::fprintf(stderr,
            "[r29-gguf-dispatch] policy=%d type=%d N=%d K=%d T=16 fp32=0 route=%s matrix_calls=%d vector_calls=%d vector_columns=%d,%d\n",
            selected, int(type), w.n, w.k, matrix_calls ? "mmq" : "vector",
            matrix_calls, vector_calls, vector_columns[0], vector_columns[1]);
    }
}

bool try_gguf_shared_vector_dispatch(
    const Tensor& x, std::span<const GgufProduct> products,
    WorkspaceArena& workspace, cudaStream_t stream) {
    // Adapted from gguf_linear.cpp::gguf_project, itself using the vendored
    // llama.cpp q8_1/DP4A path. The same-gather restriction keeps the scratch
    // bound equal to ONE ordinary product; validate all destinations first.
    if (products.size() < 2 || products.size() > 4 ||
        x.dtype != DType::BF16 || !x.data || !x.is_contiguous() ||
        x.ne[1] < 1 || x.ne[1] > gguf::kMaxVectorColumns || x.ne[2] != 1) {
        return false;
    }
    const auto* first_weight = products.front().weight;
    if (!first_weight) { return false; }
    for (const auto& product : products) {
        const auto* w = product.weight;
        const auto* y = product.out;
        if (!w || !is_gguf(w->qtype) || w->layout != QuantLayout::GgufBlocks ||
            !w->qdata || w->n <= 0 || w->k != x.ne[0] || w->k % 32 != 0 ||
            w->input_columns != first_weight->input_columns ||
            product.epilogue != GgufEpilogue::Store || product.f32 ||
            !y || !y->data || y->dtype != DType::BF16 || !y->is_contiguous() ||
            y->ne[1] != x.ne[1] || y->ne[2] != 1 || product.row < 0 ||
            static_cast<std::int64_t>(product.row) + w->n > y->ne[0]) {
            return false;
        }
        const auto block = gguf_block_shape(w->qtype);
        if (!block.elements || w->k % block.elements != 0) { return false; }
    }
    auto scope = workspace.scope();
    void* codes = workspace.alloc_bytes(
        gguf::vector_activation_bytes(x.ne[0], x.ne[1])).data;
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data),
        x.ne[0], x.ne[1], first_weight->input_columns, codes, stream);
    for (const auto& product : products) {
        const Weight& w = *product.weight;
        const auto block = gguf_block_shape(w.qtype);
        gguf::VectorOutput output;
        output.bf16 = static_cast<__nv_bfloat16*>(product.out->data) + product.row;
        output.column_stride = product.out->ne[0];
        gguf::vector_product(gguf_type(w.qtype), w.qdata,
            static_cast<std::int64_t>(w.k / block.elements) * block.bytes,
            w.n, w.k, codes, x.ne[1], output, stream);
    }
    static const bool trace = [] {
        const char* value = std::getenv("NINFER_SM89_SHARED_VECTOR_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (trace) {
        static std::atomic<unsigned> emitted{0};
        const unsigned shape = first_weight->n == 10240 ? 0 :
                               first_weight->n == 17408 ? 1 : 2;
        const unsigned bit = 1u << ((x.ne[1] - 1) * 3 + shape);
        if (!(emitted.fetch_or(bit, std::memory_order_relaxed) & bit)) {
            std::fprintf(stderr,
                "SM89_SHARED_VECTOR T=%d K=%d products=%zu first_rows=%d quantizers=1 bf16_boundary=1\n",
                x.ne[1], x.ne[0], products.size(), first_weight->n);
        }
    }
    return true;
}

void gguf_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    gguf_linear_dispatch_impl(x, w, out, workspace, stream, false);
}

void gguf_linear_gdn_rows_bf16_dispatch(
    const Tensor& x, const Weight& w, Tensor& packed, std::int32_t row_begin,
    WorkspaceArena& workspace, cudaStream_t stream) {
    gguf_linear_dispatch_impl(x, w, packed, workspace, stream, false, true, row_begin);
    // One marker per actual shape is enough for the qualification and graph
    // route receipts, without a line for every layer of a long prefill.
    static const bool trace = [] {
        const char* value = std::getenv("NINFER_R28_BUNDLE_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (trace) {
        static std::atomic<unsigned> seen{0};
        const unsigned bucket = x.ne[1] == 512 ? 0 : x.ne[1] == 8 ? 1 : 2;
        const unsigned bit = 1u << (bucket + (w.n == 6144 ? 3 : 0));
        if ((seen.fetch_or(bit, std::memory_order_relaxed) & bit) == 0) {
            std::fprintf(stderr,
                "[r28-gdn-packed] N=%d K=%d T=%d row_begin=%d stride=10240 output=bf16 same_product=1\n",
                w.n, w.k, x.ne[1], row_begin);
        }
    }
}

void gguf_linear_silu_mul_down_dispatch(
    const Tensor& gate, const Tensor& up, const Weight& weight, Tensor& projected,
    WorkspaceArena& workspace, cudaStream_t stream) {
    gguf_linear_dispatch_impl(gate, weight, projected, workspace, stream,
                              false, false, 0, &up);
}

void gguf_linear_dispatch_f32(const Tensor& x, const Weight& w, Tensor& out,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    gguf_linear_dispatch_impl(x, w, out, workspace, stream, true);
}

} // namespace ninfer::ops::detail
