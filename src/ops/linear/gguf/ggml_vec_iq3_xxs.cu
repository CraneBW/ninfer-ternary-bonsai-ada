// R26 isolated bundle TU; baseline kernel remains unchanged.
#include "ggml_bridge_vec.cuh"
#if defined(NINFER_SM89)
#include "ggml_vec_r26_signlut.cuh"
#include "ggml_vec_r27_reduce_scatter.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::gguf::detail::vec {
namespace {
bool bundle_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_R26_BUNDLE");
        if (v == nullptr || std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "1") == 0) return true;
        throw std::invalid_argument("NINFER_R26_BUNDLE must be 0 or 1");
    }();
    return enabled;
}
void trace_bundle_decode(const VecArgs& args, bool candidate) {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_R26_BUNDLE_TRACE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    static bool emitted = false;
    if (!enabled || emitted) return;
    emitted = true;
    std::fprintf(stderr,
        "[r26-decode-signlut] policy=%d type=18 N=%d K=%d T=8 row_bytes=%lld "
        "nonFused=1 output=bf16-store kernel=%s\n",
        int(candidate), args.rows, args.k, static_cast<long long>(args.row_bytes),
        candidate ? "r26::sign_lut_kernel<18,8,false,5120>"
                  : "vec::kernel<18,8,false,5120>");
}
} // namespace

template <>
void launch_k<GGML_TYPE_IQ3_XXS, 8, false>(const VecArgs& args, cudaStream_t stream) {
    if (r27::projection_eligible<GGML_TYPE_IQ3_XXS>(args)) {
        r27::launch<GGML_TYPE_IQ3_XXS>(args, stream);
        r27::trace_projection<GGML_TYPE_IQ3_XXS>(args);
        return;
    }
    const bool eligible = args.rows == 17408 && args.k == 5120 && args.row_bytes == 1960 &&
        args.out.bf16 != nullptr && args.out.f32 == nullptr &&
        args.out.column_stride == args.rows && args.out.epilogue == Epilogue::Store &&
        device_facts().cc == GGML_CUDA_CC_ADA_LOVELACE;
    if (eligible) {
        if (r27::bundle_enabled()) {
            r27::launch<GGML_TYPE_IQ3_XXS>(args, stream);
            r27::trace<GGML_TYPE_IQ3_XXS>(args, true);
            return;
        }
        const bool candidate = bundle_enabled();
        if (candidate) r26::sign_lut_launch<GGML_TYPE_IQ3_XXS, 8, false, 5120>(args, stream);
        else launch<GGML_TYPE_IQ3_XXS, 8, false, 5120>(args, stream);
        r27::trace<GGML_TYPE_IQ3_XXS>(args, false, candidate);
        trace_bundle_decode(args, candidate); // after the actual checked launch
        return;
    }
    switch (args.k) {
    case 5120: return launch<GGML_TYPE_IQ3_XXS, 8, false, 5120>(args, stream);
    case 6144: return launch<GGML_TYPE_IQ3_XXS, 8, false, 6144>(args, stream);
    case 10240: return launch<GGML_TYPE_IQ3_XXS, 8, false, 10240>(args, stream);
    case 17408: return launch<GGML_TYPE_IQ3_XXS, 8, false, 17408>(args, stream);
    default: return launch<GGML_TYPE_IQ3_XXS, 8, false, 0>(args, stream);
    }
}
} // namespace ninfer::ops::gguf::detail::vec
#endif
NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_IQ3_XXS);
