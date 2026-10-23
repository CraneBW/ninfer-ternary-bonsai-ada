#pragma once

// R27: preserve the original lane-0 FP32 butterfly tree and operand direction,
// retain only fields the destination lanes consume, then store cooperatively.
// Decoder/product/slice ordering and persistent-row mapping are the production
// ggml_bridge_vec.cuh's. Distributed output ownership is adapted from
// Don-Chad/ninfer-3090 d8c29253 ggml_mmvq.cuh:137-146 (MIT/Strata attribution in
// that source); its activation scale/FMA/shared-input arithmetic is not copied.
// See perf-ada-round26-bundle/next-decode-research.md for the full lane proof.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

namespace ninfer::ops::gguf::detail::r27 {

inline bool bundle_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_R27_BUNDLE");
        if (value == nullptr || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("NINFER_R27_BUNDLE must be 0 or 1");
    }();
    return enabled;
}

inline bool projections_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM89_R27_PROJECTIONS");
        if (!value || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument("NINFER_SM89_R27_PROJECTIONS must be 0 or 1");
    }();
    return enabled;
}

template <ggml_type type>
inline bool projection_eligible(const VecArgs& args) {
    using D = vec::Decoder<type>;
    const bool rows = args.rows == 1024 || args.rows == 2048 || args.rows == 6144 || args.rows == 10240 ||
                      args.rows == 16384 || args.rows == 34816;
    return projections_enabled() && rows && args.k == 5120 &&
           args.row_bytes == (5120 / D::kBlockElems) * D::kBlockBytes &&
           args.out.bf16 && !args.out.f32 && args.out.column_stride == args.rows &&
           args.out.epilogue == Epilogue::Store &&
           device_facts().cc == GGML_CUDA_CC_ADA_LOVELACE;
}

template <ggml_type type>
inline void trace_projection(const VecArgs& args) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM89_R27_PROJECTIONS_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) { return; }
    const unsigned shape = args.rows == 1024 ? 0 : args.rows == 2048 ? 1 : args.rows == 6144 ? 2 :
                           args.rows == 10240 ? 3 : args.rows == 16384 ? 4 : 5;
    const unsigned bit = 1u << (shape * 2 + (args.out.column_stride != args.rows));
    static std::atomic<unsigned> emitted{0};
    if (!(emitted.fetch_or(bit, std::memory_order_relaxed) & bit)) {
        std::fprintf(stderr, "SM89_R27_PROJECTION type=%d N=%d K=5120 T=8 stride=%lld original_tree=1\n",
                     int(type), args.rows, static_cast<long long>(args.out.column_stride));
    }
}

// Each stage first computes the original canonical low-lane + high-lane
// subtrees of every live field, then discards the half this lane will not own.
// Indices are compile-time constants after unrolling; no dynamic register-array
// selection, no new memory, and every full-mask shuffle has all 32 participants.
template <int Active, int Offset>
__device__ __forceinline__ void canonical_halve(float (&values)[16], int lane) {
    static_assert(Active == Offset && Active >= 2 && Active <= 16);
    const int low_lane = lane & ~Offset;
    const int high_lane = lane | Offset;
#pragma unroll
    for (int i = 0; i < Active / 2; ++i) {
        const float lo0 = __shfl_sync(0xFFFFFFFFu, values[i], low_lane);
        const float hi0 = __shfl_sync(0xFFFFFFFFu, values[i], high_lane);
        const float lo1 = __shfl_sync(0xFFFFFFFFu, values[i + Active / 2], low_lane);
        const float hi1 = __shfl_sync(0xFFFFFFFFu, values[i + Active / 2], high_lane);
        const float first = __fadd_rn(lo0, hi0);
        const float second = __fadd_rn(lo1, hi1);
        values[i] = (lane & Offset) != 0 ? second : first;
    }
}

template <ggml_type type>
__launch_bounds__(kVecWarps * 32) __global__ void reduce_scatter_store_kernel(VecArgs p) {
    using D = vec::Decoder<type>;
    static_assert(type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS ||
                  type == GGML_TYPE_IQ4_XS);
    constexpr int kRows = 2;
    constexpr int T = 8;
    constexpr int KS = 5120;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kTable = D::kTableWords > 0 ? D::kTableWords : 4;
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) {
            table[i] = src[i];
        }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = KS;
    constexpr int slices = k / 32;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const int sub = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * kRows; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * kRows) {
        const std::uint8_t* rowp[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            rowp[r] = p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes +
                lane_offset;
        }
        // Flattening only changes the spelling of the original acc[r][j].
        float acc[kRows * T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) acc[r * T + j] = 0.0f;
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const auto* q =
                    reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                vec::Slice sl;
                D::decode(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes,
                          sub, table, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) acc[r * T + j] += D::dot(sl, a[j], d[j], sum[j]);
            }
        };
#pragma unroll
        for (int it = 0; it < KS / 1024; ++it) slice(it);

        canonical_halve<16, 16>(acc, lane);
        canonical_halve<8, 8>(acc, lane);
        canonical_halve<4, 4>(acc, lane);
        canonical_halve<2, 2>(acc, lane);
        // The even lanes are the low subtrees at the final offset1 stage,
        // exactly the canonical operand direction of the old lane0 result.
        const float other = __shfl_xor_sync(0xFFFFFFFFu, acc[0], 1);
        const float value = __fadd_rn(acc[0], other);
        if ((lane & 1) == 0) {
            const int r = lane >> 4;
            const int j = (lane >> 1) & 7;
            p.out.bf16[std::int64_t(j) * p.out.column_stride + row0 + r] =
                __float2bfloat16(value);
        }
    }
}

template <ggml_type type>
inline void launch(const VecArgs& args, cudaStream_t stream) {
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, reduce_scatter_store_kernel<type>, kVecWarps * 32, 0),
              "r27 vector occupancy");
        return std::max(blocks, 1);
    }();
    const int groups = (args.rows + 2 * kVecWarps - 1) / (2 * kVecWarps);
    const int blocks = std::min(groups, resident * device_facts().sm_count);
    reduce_scatter_store_kernel<type><<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "r27 reduce-scatter store launch");
}

template <ggml_type type>
inline void trace(const VecArgs& args, bool candidate, bool r26_candidate = false) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_R27_BUNDLE_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    static bool emitted = false;
    if (!enabled || emitted) return;
    emitted = true;
    const char* name = candidate ? "r27::reduce_scatter_store_kernel" :
        (r26_candidate ? "r26::sign_lut_kernel" : "vec::kernel");
    std::fprintf(stderr,
        "[r27-decode-reduce-scatter] policy=%d type=%d N=%d K=%d T=8 row_bytes=%lld "
        "nonFused=1 output=bf16-store r26=%d kernel=%s<%d%s\n",
        int(candidate), int(type), args.rows, args.k, static_cast<long long>(args.row_bytes),
        int(r26_candidate), name, int(type), candidate ? ">" : ",8,false,5120>");
}

} // namespace ninfer::ops::gguf::detail::r27
