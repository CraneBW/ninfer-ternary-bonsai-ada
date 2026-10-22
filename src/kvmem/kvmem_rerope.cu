// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// nvcc-compiled bridge for the KVMem K2 re-phase kernel (see kvmem_bridge.h).

#include "kvmem/kvmem_bridge.h"

#include "core/paged_kv_cache.h"
#include "core/device.h"
#include "kvmem/kvmem_rerope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::kvmem {

namespace {
struct SourcePositions { std::int32_t values[64]; };
#if defined(NINFER_SM89)
bool source_layer_batch_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_KVMEM_SOURCE_LAYER_BATCH");
        if (value == nullptr || std::strcmp(value, "1") == 0) { return true; }
        if (std::strcmp(value, "0") == 0) { return false; }
        throw std::invalid_argument("NINFER_KVMEM_SOURCE_LAYER_BATCH must be 0 or 1");
    }();
    return enabled;
}

// Reuse context_kv_materialize's by-value DeviceLayers + grid.z addressing.
// Source: tancau/ninfer-kvmem-ring@04721a8637f1302ff73419144e34996bb80454dd,
// src/ops/context_kv_materialize/materialize.cu. Only the pointer-table pattern
// is reused; the immutable K codec and per-token operation below are unchanged.
struct SourceLayerPlanes {
    std::int8_t* codes[16];
    __half* scales[16];
};
static_assert(sizeof(SourceLayerPlanes) == 16 * 2 * sizeof(void*));

// Reuse R14's by-value layer table and grid.z addressing. Add a page ordinal
// above the layer ordinal; every warp still calls the identical token helper.
// Raw arrays avoid relying on std::array device annotations in kernel code.
struct SourcePageMetadata {
    std::int32_t physical;
    SourcePositions positions;
    std::int32_t destination;
    std::int32_t columns;
};
struct SourceRestoreBatch {
    SourceLayerPlanes planes;
    SourcePageMetadata pages[2];
    std::int32_t layers;
};
static_assert(sizeof(SourcePageMetadata) == 268);
static_assert(sizeof(SourceRestoreBatch) == 800);
// One 800-byte grid-constant argument; comfortably below the older 4KiB
// CUDA parameter bound as well as the larger modern bound. No metadata H2D.
static_assert(sizeof(SourceRestoreBatch) < 4096);

__global__ void rebuild_source_pages_layers_kernel(
    const __grid_constant__ SourceRestoreBatch batch) {
    const int token = static_cast<int>(blockIdx.x);
    const int page_index = static_cast<int>(blockIdx.z) / batch.layers;
    const int layer = static_cast<int>(blockIdx.z) % batch.layers;
    const auto& page = batch.pages[page_index];
    if (token >= page.columns) { return; }
    const auto source = page.positions.values[token];
    const auto to = page.destination + token;
    if (source == to) { return; }
    const std::int32_t src[3] = {source, source, source};
    const std::int32_t dst[3] = {to, to, to};
    kvmem_rerope_int8_g64_token<4>(batch.planes.codes[layer], batch.planes.scales[layer],
                                   page.physical, blockIdx.y, token, src, dst, threadIdx.x);
}

__global__ void rebuild_source_layers_kernel(
    const __grid_constant__ SourceLayerPlanes planes, int physical,
    const __grid_constant__ SourcePositions positions, int destination, int columns) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= columns) { return; }
    const std::int32_t src[3] = {positions.values[token], positions.values[token],
                                  positions.values[token]};
    const auto to = destination + token;
    if (positions.values[token] == to) { return; }
    const std::int32_t dst[3] = {to, to, to};
    const int layer = static_cast<int>(blockIdx.z);
    kvmem_rerope_int8_g64_token<4>(planes.codes[layer], planes.scales[layer], physical,
                                   blockIdx.y, token, src, dst, threadIdx.x);
}
#endif
template <int Heads>
__global__ void rebuild_source_kernel(std::int8_t* codes, __half* scales, int physical,
                                      SourcePositions positions, int destination, int columns) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= columns) { return; }
    const std::int32_t src[3] = {positions.values[token], positions.values[token],
                                  positions.values[token]};
    const auto to = destination + token;
    if (positions.values[token] == to) { return; }
    const std::int32_t dst[3] = {to, to, to};
    kvmem_rerope_int8_g64_token<Heads>(codes, scales, physical, blockIdx.y, token,
                                       src, dst, threadIdx.x);
}
} // namespace

void kvmem_rebuild_source_k(DeviceKVPagePool& pool, std::int32_t physical,
                            const std::array<std::int32_t, 64>& source_positions,
                            std::int32_t destination, std::uint32_t columns,
                            cudaStream_t stream) {
    const auto& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 || columns == 0 || columns > 64) {
        throw std::invalid_argument("immutable K rebuild geometry is invalid");
    }
    SourcePositions positions;
    std::memcpy(positions.values, source_positions.data(), sizeof(positions.values));
#if defined(NINFER_SM89)
    if (geometry.planes.size() == 16 * 4 && geometry.planes.front().head_extent == 4 &&
        geometry.device_plane_order == PagedKVPlaneOrder::PageMajor &&
        source_layer_batch_enabled()) {
        SourceLayerPlanes planes{};
        for (std::size_t layer = 0; layer < 16; ++layer) {
            planes.codes[layer] = static_cast<std::int8_t*>(pool.plane(layer * 4).data);
            planes.scales[layer] = static_cast<__half*>(pool.plane(layer * 4 + 2).data);
        }
        rebuild_source_layers_kernel<<<dim3(columns, 4, 16), 32, 0, stream>>>(
            planes, physical, positions, destination, columns);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
        return;
    }
#endif
    const auto rebuild = [&](auto heads) {
        constexpr int H = decltype(heads)::value;
        for (std::size_t i = 0; i < geometry.planes.size(); i += 4) {
            rebuild_source_kernel<H><<<dim3(columns, H), 32, 0, stream>>>(
                static_cast<std::int8_t*>(pool.plane(i).data),
                static_cast<__half*>(pool.plane(i + 2).data), physical, positions,
                destination, columns);
        }
    };
    if (geometry.planes.front().head_extent == 4) { rebuild(std::integral_constant<int, 4>{}); }
    else if (geometry.planes.front().head_extent == 2) { rebuild(std::integral_constant<int, 2>{}); }
    else { throw std::invalid_argument("immutable K rebuild head count is unsupported"); }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
}

void kvmem_rebuild_source_k_batch(DeviceKVPagePool& pool,
                                  std::span<const SourcePageRestore> pages,
                                  cudaStream_t stream) {
    const auto& geometry = pool.geometry();
    if (pages.empty() || pages.size() > 2 || geometry.page_tokens != 64 ||
        geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("immutable K rebuild batch geometry/count is invalid");
    }
    // Validate the whole group before submitting either page. The store caller
    // checks the stronger lease/generation and immutable-source ownership.
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const auto& page = pages[i];
        if (page.physical < 0 || static_cast<std::uint32_t>(page.physical) >= pool.capacity_pages() ||
            page.columns == 0 || page.columns > 64 ||
            page.destination > std::numeric_limits<std::int32_t>::max() -
                static_cast<std::int32_t>(page.columns - 1) ||
            (i != 0 && page.physical == pages[0].physical)) {
            throw std::invalid_argument("immutable K rebuild batch extent/physical page is invalid");
        }
    }
    const int heads = geometry.planes.front().head_extent;
    if (heads != 4 && heads != 2) {
        throw std::invalid_argument("immutable K rebuild batch head count is unsupported");
    }
    for (std::size_t i = 0; i < geometry.planes.size(); i += 4) {
        const auto& codes = geometry.planes[i];
        const auto& scales = geometry.planes[i + 2];
        if (codes.dtype != DType::I8 || codes.leading_extent != 256 || codes.head_extent != heads ||
            scales.dtype != DType::FP16 || scales.leading_extent != 4 || scales.head_extent != heads) {
            throw std::invalid_argument("immutable K rebuild batch requires INT8-G64 K planes");
        }
    }
    if (pages.size() == 1) {
        const auto& page = pages.front();
        kvmem_rebuild_source_k(pool, page.physical, page.positions, page.destination,
                                page.columns, stream);
        return;
    }
#if defined(NINFER_SM89)
    const auto layers = geometry.planes.size() / 4;
    bool batch_geometry = heads == 4 && (layers == 16 || layers == 1) &&
        geometry.device_plane_order == PagedKVPlaneOrder::PageMajor;
    for (std::size_t i = 0; batch_geometry && i < geometry.planes.size(); ++i) {
        const auto& plane = geometry.planes[i];
        batch_geometry = plane.head_extent == 4 &&
            (i % 4 < 2 ? plane.dtype == DType::I8 && plane.leading_extent == 256
                       : plane.dtype == DType::FP16 && plane.leading_extent == 4);
    }
    if (batch_geometry && source_layer_batch_enabled()) {
        SourceRestoreBatch batch{};
        batch.layers = static_cast<std::int32_t>(layers);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            batch.planes.codes[layer] = static_cast<std::int8_t*>(pool.plane(layer * 4).data);
            batch.planes.scales[layer] = static_cast<__half*>(pool.plane(layer * 4 + 2).data);
        }
        std::uint32_t columns = 0;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const auto& page = pages[i];
            batch.pages[i].physical = page.physical;
            std::memcpy(batch.pages[i].positions.values, page.positions.data(),
                        sizeof(batch.pages[i].positions.values));
            batch.pages[i].destination = page.destination;
            batch.pages[i].columns = static_cast<std::int32_t>(page.columns);
            if (page.columns > columns) { columns = page.columns; }
        }
        rebuild_source_pages_layers_kernel<<<
            dim3(columns, 4, static_cast<unsigned>(layers * pages.size())), 32, 0, stream>>>(batch);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
        return;
    }
#endif
    for (const auto& page : pages) {
        kvmem_rebuild_source_k(pool, page.physical, page.positions, page.destination,
                                page.columns, stream);
    }
}

void kvmem_rebuild_source_k_batch_layer(DeviceKVPagePool& pool,
                                        std::span<const SourcePageRestore> pages,
                                        std::uint32_t layer, cudaStream_t stream) {
#if defined(NINFER_SM89)
    const auto& geometry = pool.geometry();
    if (pages.empty() || pages.size() > 2 || layer >= 16 ||
        geometry.page_tokens != kPagedKVPageSize ||
        geometry.device_plane_order != PagedKVPlaneOrder::PageMajor ||
        geometry.planes.size() != 16 * 4 || stream == nullptr ||
        stream == cudaStreamLegacy) {
        throw std::invalid_argument("source layer rebuild requires Main16 and one/two pages");
    }
    for (std::size_t plane = 0; plane < geometry.planes.size(); ++plane) {
        const auto& entry = geometry.planes[plane];
        const bool scale = plane % 4 >= 2;
        if (entry.head_extent != 4 || entry.leading_extent != (scale ? 4 : 256) ||
            entry.dtype != (scale ? DType::FP16 : DType::I8)) {
            throw std::invalid_argument("source layer rebuild requires INT8-G64 geometry");
        }
    }
    SourceRestoreBatch batch{};
    batch.layers = 1; // grid.z names a page; each page uses this one layer plane
    batch.planes.codes[0] = static_cast<std::int8_t*>(pool.plane(layer * 4).data);
    batch.planes.scales[0] = static_cast<__half*>(pool.plane(layer * 4 + 2).data);
    std::uint32_t max_columns = 0;
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const auto& page = pages[i];
        if (page.physical < 0 ||
            static_cast<std::uint32_t>(page.physical) >= pool.capacity_pages() ||
            page.columns == 0 || page.columns > kPagedKVPageSize || page.destination < 0 ||
            page.destination > std::numeric_limits<std::int32_t>::max() -
                static_cast<std::int32_t>(page.columns - 1) ||
            (i != 0 && page.physical == pages[0].physical)) {
            throw std::invalid_argument("source layer rebuild physical/columns invalid");
        }
        batch.pages[i].physical = page.physical;
        batch.pages[i].destination = page.destination;
        batch.pages[i].columns = static_cast<std::int32_t>(page.columns);
        std::memcpy(batch.pages[i].positions.values, page.positions.data(),
                    sizeof(batch.pages[i].positions.values));
        max_columns = std::max(max_columns, page.columns);
    }
    rebuild_source_pages_layers_kernel<<<
        dim3(max_columns, 4, static_cast<unsigned>(pages.size())), 32, 0, stream>>>(batch);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
#else
    (void)pool; (void)pages; (void)layer; (void)stream;
    throw std::invalid_argument("source layer rebuild requires SM89");
#endif
}

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
                        std::uint32_t n_blocks, KvmemReropeWorkspace workspace,
                        cudaStream_t stream) {
    if (physical_pages == nullptr || from_bases == nullptr || to_bases == nullptr ||
        n_blocks == 0) {
        return;
    }
    const KVPageGeometry& geometry = pool.geometry();
    if (geometry.planes.empty() || geometry.planes.size() % 4 != 0) {
        throw std::invalid_argument("kvmem rerope: expected layer-grouped KV planes");
    }
    const int kv_heads = geometry.planes.front().head_extent;

    const std::size_t meta_count = static_cast<std::size_t>(n_blocks) * 3;
    if (workspace.host_meta.size() < meta_count || workspace.device_meta.size() < meta_count) {
        throw std::invalid_argument("kvmem rerope: metadata workspace is too small");
    }
    std::int32_t* host_meta = workspace.host_meta.data();
    std::memcpy(host_meta, physical_pages, sizeof(std::int32_t) * n_blocks);
    std::memcpy(host_meta + n_blocks, from_bases, sizeof(std::int32_t) * n_blocks);
    std::memcpy(host_meta + static_cast<std::size_t>(n_blocks) * 2, to_bases,
                sizeof(std::int32_t) * n_blocks);

    std::int32_t* d_meta = workspace.device_meta.data();
    const std::size_t meta_bytes = meta_count * sizeof(std::int32_t);
    KVMEM_CUDA_CHECK(
        cudaMemcpyAsync(d_meta, host_meta, meta_bytes, cudaMemcpyHostToDevice, stream));
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
        throw std::invalid_argument("kvmem rerope: unsupported KV head count");
    }
    KVMEM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void kvmem_rerope_pages_layer_ready(DeviceKVPagePool& pool,
                                   const std::int32_t* physical_pages,
                                   const std::int32_t* from_bases,
                                   const std::int32_t* to_bases,
                                   std::uint32_t n_blocks, KvmemReropeWorkspace workspace,
                                   std::span<const cudaEvent_t> layer_ready,
                                   cudaStream_t producer,
                                   KvmemLayerEnqueue enqueue_source_layer,
                                   void* source_owner) {
    if ((n_blocks != 0 && (!physical_pages || !from_bases || !to_bases)) ||
        (n_blocks == 0 && !enqueue_source_layer) ||
        (enqueue_source_layer && !source_owner) ||
        producer == nullptr || producer == cudaStreamLegacy || layer_ready.size() != 16) {
        throw std::invalid_argument("layer-ready requires resident pages or a source producer");
    }
    const auto& geometry = pool.geometry();
    if (geometry.page_tokens != kPagedKVPageSize || geometry.planes.size() != 16 * 4 ||
        geometry.device_plane_order != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("layer-ready rephase requires Main16 PageMajor");
    }
    for (std::size_t i = 0; i < geometry.planes.size(); ++i) {
        const auto& p = geometry.planes[i];
        const bool scale = (i % 4) >= 2;
        if (p.head_extent != 4 || p.leading_extent != (scale ? 4 : 256) ||
            p.dtype != (scale ? DType::FP16 : DType::I8)) {
            throw std::invalid_argument("layer-ready rephase requires INT8-G64 KV geometry");
        }
    }
    const std::size_t count = static_cast<std::size_t>(n_blocks) * 3;
    if (workspace.host_meta.size() < count || workspace.device_meta.size() < count) {
        throw std::invalid_argument("layer-ready rephase metadata workspace too small");
    }
    for (const cudaEvent_t ready : layer_ready) {
        if (!ready) { throw std::invalid_argument("layer-ready rephase missing event"); }
    }
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        if (physical_pages[i] < 0 ||
            static_cast<std::uint32_t>(physical_pages[i]) >= pool.capacity_pages()) {
            throw std::invalid_argument("layer-ready rephase physical target out of range");
        }
    }
    std::int32_t* d = workspace.device_meta.data();
    if (n_blocks != 0) {
        auto* h = workspace.host_meta.data();
        std::memcpy(h, physical_pages, sizeof(std::int32_t) * n_blocks);
        std::memcpy(h + n_blocks, from_bases, sizeof(std::int32_t) * n_blocks);
        std::memcpy(h + static_cast<std::size_t>(n_blocks) * 2, to_bases,
                    sizeof(std::int32_t) * n_blocks);
        CUDA_CHECK(cudaMemcpyAsync(d, h, count * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   producer));
    }
    for (std::size_t layer = 0; layer < layer_ready.size(); ++layer) {
        if (enqueue_source_layer) {
            enqueue_source_layer(source_owner, static_cast<std::uint32_t>(layer), producer);
        }
        if (n_blocks != 0) {
            const std::size_t plane = layer * 4;
            kvmem_rerope_int8_g64_planes<4>(
                static_cast<std::int8_t*>(pool.plane(plane).data),
                static_cast<__half*>(pool.plane(plane + 2).data),
                d, d + n_blocks, d + 2 * static_cast<std::size_t>(n_blocks),
                n_blocks, producer);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(layer_ready[layer], producer));
    }
}

} // namespace ninfer::kvmem
