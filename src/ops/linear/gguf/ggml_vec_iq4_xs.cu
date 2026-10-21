// R27 narrow T8 epilogue bundle; other products retain their existing routes.
#include "ggml_bridge_vec.cuh"
#if defined(NINFER_SM89)
#include "ggml_vec_r27_reduce_scatter.cuh"

namespace ninfer::ops::gguf::detail::vec {
template <>
void launch_k<GGML_TYPE_IQ4_XS, 8, false>(const VecArgs& args, cudaStream_t stream) {
    const bool eligible = args.rows == 17408 && args.k == 5120 && args.row_bytes == 2720 &&
        args.out.bf16 != nullptr && args.out.f32 == nullptr &&
        args.out.column_stride == args.rows && args.out.epilogue == Epilogue::Store &&
        device_facts().cc == GGML_CUDA_CC_ADA_LOVELACE;
    if (eligible) {
        const bool candidate = r27::bundle_enabled();
        if (candidate) r27::launch<GGML_TYPE_IQ4_XS>(args, stream);
        else launch<GGML_TYPE_IQ4_XS, 8, false, 5120>(args, stream);
        r27::trace<GGML_TYPE_IQ4_XS>(args, candidate);
        return;
    }
    switch (args.k) {
    case 5120: return launch<GGML_TYPE_IQ4_XS, 8, false, 5120>(args, stream);
    case 6144: return launch<GGML_TYPE_IQ4_XS, 8, false, 6144>(args, stream);
    case 10240: return launch<GGML_TYPE_IQ4_XS, 8, false, 10240>(args, stream);
    case 17408: return launch<GGML_TYPE_IQ4_XS, 8, false, 17408>(args, stream);
    default: return launch<GGML_TYPE_IQ4_XS, 8, false, 0>(args, stream);
    }
}
} // namespace ninfer::ops::gguf::detail::vec
#endif
NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_IQ4_XS);
