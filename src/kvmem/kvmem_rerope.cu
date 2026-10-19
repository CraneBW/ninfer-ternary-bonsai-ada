// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// nvcc-compiled bridge for the KVMem K2 re-phase kernel (see kvmem_bridge.h).

#include "kvmem/kvmem_bridge.h"

#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_rerope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::kvmem {

void kvmem_rerope_page(DeviceKVPagePool& pool, std::int32_t physical_page,
                       const std::int32_t* src_pos, const std::int32_t* dst_pos,
                       cudaStream_t stream) {
    const KVPageGeometry& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("kvmem rerope: expected layer-grouped KV planes");
    }
    // Text cache plane groups are [K-codes, V-codes, K-scale, V-scale]; the
    // K-code plane's head_extent is the KV head count.
    const int kv_heads = geometry.planes.front().head_extent;
    const auto rephase = [&](auto heads) {
        constexpr int H = decltype(heads)::value;
        for (std::size_t layer = 0; (layer + 1) * 4 <= geometry.planes.size(); ++layer) {
            const std::size_t kc = layer * 4;
            const std::size_t ks = layer * 4 + 2;
            kvmem_rerope_int8_g64_plane<H>(
                static_cast<std::int8_t*>(pool.plane(kc).data),
                static_cast<__half*>(pool.plane(ks).data), physical_page, src_pos, dst_pos,
                stream);
        }
    };
    if (kv_heads == 4) {
        rephase(std::integral_constant<int, 4>{});
    } else if (kv_heads == 2) {
        rephase(std::integral_constant<int, 2>{});
    } else {
        throw std::invalid_argument("kvmem rerope: unsupported KV head count");
    }
}

void kvmem_rerope_pages(DeviceKVPagePool& pool, const std::int32_t* physical_pages,
                        const std::int32_t* from_bases, const std::int32_t* to_bases,
                        std::uint32_t n_blocks, cudaStream_t stream) {
    if (physical_pages == nullptr || from_bases == nullptr || to_bases == nullptr ||
        n_blocks == 0) {
        return;
    }
    const KVPageGeometry& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("kvmem rerope: expected layer-grouped KV planes");
    }
    const int kv_heads = geometry.planes.front().head_extent;

    std::vector<std::int32_t> host_meta(static_cast<std::size_t>(n_blocks) * 3);
    std::memcpy(host_meta.data(), physical_pages, sizeof(std::int32_t) * n_blocks);
    std::memcpy(host_meta.data() + n_blocks, from_bases, sizeof(std::int32_t) * n_blocks);
    std::memcpy(host_meta.data() + static_cast<std::size_t>(n_blocks) * 2, to_bases,
                sizeof(std::int32_t) * n_blocks);

    std::int32_t* d_meta = nullptr;
    const std::size_t meta_bytes = host_meta.size() * sizeof(std::int32_t);
    KVMEM_CUDA_CHECK(cudaMalloc(&d_meta, meta_bytes));
    KVMEM_CUDA_CHECK(
        cudaMemcpyAsync(d_meta, host_meta.data(), meta_bytes, cudaMemcpyHostToDevice, stream));
    const std::int32_t* d_pages = d_meta;
    const std::int32_t* d_from  = d_meta + n_blocks;
    const std::int32_t* d_to    = d_meta + static_cast<std::size_t>(n_blocks) * 2;

    const auto rephase = [&](auto heads) {
        constexpr int H = decltype(heads)::value;
        for (std::size_t layer = 0; (layer + 1) * 4 <= geometry.planes.size(); ++layer) {
            const std::size_t kc = layer * 4;
            const std::size_t ks = layer * 4 + 2;
            kvmem_rerope_int8_g64_planes<H>(static_cast<std::int8_t*>(pool.plane(kc).data),
                                            static_cast<__half*>(pool.plane(ks).data), d_pages,
                                            d_from, d_to, n_blocks, stream);
        }
    };
    if (kv_heads == 4) {
        rephase(std::integral_constant<int, 4>{});
    } else if (kv_heads == 2) {
        rephase(std::integral_constant<int, 2>{});
    } else {
        KVMEM_CUDA_CHECK(cudaFree(d_meta));
        throw std::invalid_argument("kvmem rerope: unsupported KV head count");
    }
    KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
    KVMEM_CUDA_CHECK(cudaFree(d_meta));
}

} // namespace ninfer::kvmem
