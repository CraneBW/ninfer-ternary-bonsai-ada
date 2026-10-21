#pragma once
// SPDX-License-Identifier: Apache-2.0
// R26 scratch bundle only. Original dense body/launcher, format loads and FP
// expressions are copied from this project's ggml_bridge_vec.cuh unchanged.
// Immutable sign masks adapt Strata's signed8t/sign_entry (MIT, Niko1221 and
// Strata contributors), not its HIP/MMA kernels, activation or FP arithmetic.
// https://github.com/Niko1221/Strata/blob/main/src/prefill/moe_fused_iq.cu#L228
// IQ3_S:16 sign nibbles; IQ3_XXS:128 7-bit fields with parity folded in once.
#include "ggml_bridge_vec.cuh"

namespace ninfer::ops::gguf::detail::vec::r26 {

template <ggml_type type> struct SignPolicy;

template <> struct SignPolicy<GGML_TYPE_IQ3_S> {
    using Mask = uint2;
    static constexpr int kEntries = 16;
    __device__ __forceinline__ static Mask mask(unsigned b) {
        const std::uint32_t ones = (b * 0x00204081u) & 0x01010101u;
        return make_uint2(ones, ones * 0xFFu);
    }
    __device__ __forceinline__ static void decode(
        const std::uint8_t* b, int u, const std::uint32_t* grid,
        const Mask* masks, Slice& s) {
        const std::uint32_t q0 = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1 = u32_a2(b + 6 + 8 * u);
        const std::uint32_t qh = b[66 + u];
        const std::uint32_t sg = u32_a2(b + 74 + 4 * u);
        const int sc = (b[106 + u / 2] >> (4 * (u & 1))) & 0xF;
        s.f[0] = half_at(b) * float(1 + 2 * sc);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t idx = (((e < 4 ? q0 : q1) >> (8 * (e & 3))) & 0xFF) |
                                      ((qh << (8 - e)) & 0x100);
            const Mask m = masks[(sg >> (4 * e)) & 0xF];
            s.w[e] = int((grid[idx] ^ m.y) + m.x);
        }
    }
};

template <> struct SignPolicy<GGML_TYPE_IQ3_XXS> {
    using Mask = uint4;
    static constexpr int kEntries = 128;
    __device__ __forceinline__ static Mask mask(unsigned b) {
        const std::uint32_t sg = ksigns(b);
        const std::uint32_t lo = ((sg & 0xFu) * 0x00204081u) & 0x01010101u;
        const std::uint32_t hi = ((sg >> 4) * 0x00204081u) & 0x01010101u;
        return make_uint4(lo, lo * 0xFFu, hi, hi * 0xFFu);
    }
    __device__ __forceinline__ static void decode(
        const std::uint8_t* b, int u, const std::uint32_t* grid,
        const Mask* masks, Slice& s) {
        const std::uint32_t q0 = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1 = u32_a2(b + 6 + 8 * u);
        const std::uint32_t aux = u32_a2(b + 66 + 4 * u);
        s.f[0] = half_at(b) * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const std::uint32_t q = p < 2 ? q0 : q1;
            const Mask m = masks[(aux >> (7 * p)) & 0x7F];
            s.w[2 * p] = int((grid[(q >> (16 * (p & 1))) & 0xFF] ^ m.y) + m.x);
            s.w[2 * p + 1] = int((grid[(q >> (16 * (p & 1) + 8)) & 0xFF] ^ m.w) + m.z);
        }
    }
};

template <ggml_type type, int T, bool Fused, int KS>
__launch_bounds__(kVecWarps * 32) __global__ void sign_lut_kernel(VecArgs p) {
    static_assert((type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) &&
                  T == 8 && !Fused && KS == 5120);
    using P = SignPolicy<type>;
    using Mask = typename P::Mask;
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    __shared__ __align__(16) std::uint32_t table[kTable];
    __shared__ __align__(16) Mask sign_masks[P::kEntries];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) { table[i] = src[i]; }
        for (int i = threadIdx.x; i < P::kEntries; i += kVecWarps * 32) {
            sign_masks[i] = P::mask(i);
        }
        __syncthreads();
    }
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int k      = KS != 0 ? KS : p.k;
    const int slices = k / 32;
    const int step   = Fused ? 1 : kRows;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const int sub                  = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * step) {
        const std::uint8_t* rowp[kRows];
        if constexpr (Fused) {
            rowp[0] = p.weight + std::int64_t(row0) * p.row_bytes + lane_offset;
            rowp[1] = p.weight + std::int64_t(p.rows + row0) * p.row_bytes + lane_offset;
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                rowp[r] = p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes + lane_offset;
            }
        }
        float acc[kRows][T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j]            = ds.x;
                sum[j]          = ds.y;
                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                P::decode(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sign_masks, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] += D::dot(sl, a[j], d[j], sum[j]); }
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) { slice(it); }
        } else {
            for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (lane == 0) {
            if constexpr (Fused) {
#pragma unroll
                for (int j = 0; j < T; ++j) { vec_store(p.out, row0, j, vec_gate(acc[0][j], p.out.activation) * acc[1][j]); }
            } else {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (row0 + r < p.rows) {
#pragma unroll
                        for (int j = 0; j < T; ++j) { vec_store(p.out, row0 + r, j, acc[r][j]); }
                    }
                }
            }
        }
    }
}

template <ggml_type type, int T, bool Fused, int KS>
void sign_lut_launch(const VecArgs& args, cudaStream_t stream) {
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, sign_lut_kernel<type, T, Fused, KS>,
                                                            kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1);
    }();
    const int per_block = kVecWarps * (Fused ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int blocks    = std::min(groups, resident * device_facts().sm_count);
    sign_lut_kernel<type, T, Fused, KS><<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "vector product launch");
}

} // namespace ninfer::ops::gguf::detail::vec::r26
