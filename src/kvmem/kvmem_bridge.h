// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// Pure-C++ bridge declarations for the KVMem K2 re-phase kernel.
//
// logical_kv_store.h is included from host translation units compiled by
// cl.exe, where `<<<>>>` launch syntax cannot parse. The store therefore
// calls this bridge; the only TU that instantiates the kernels is
// kvmem_rerope.cu (compiled by nvcc).

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <array>
#include <cstdio>

namespace ninfer {

class DeviceKVPagePool;

} // namespace ninfer

namespace ninfer::kvmem {

// A prefill-only dependency contract between a KVMem producer and the exact
// attention layer that consumes its restored physical KV planes. The producer
// owns the events and their lifetime; no caller may treat a submitted H2D or
// rephase operation as completed until wait_before_layer has been enqueued.
// Graph decode must use its separate stream/graph retirement contract.
class LayerReadyConsumer {
public:
    virtual ~LayerReadyConsumer() = default;
    virtual void wait_before_layer(std::uint32_t layer, cudaStream_t stream) = 0;
    virtual void release_after_layer(std::uint32_t layer, cudaStream_t stream) = 0;
};

// Lightweight CUDA error reporter for host-side executor code (the store).
// kvmem stays self-contained: core/device.h carries the SM gate and must not
// be pulled into every TU that includes this header.
inline void kvmem_cuda_check_impl(cudaError_t err, const char* expr, const char* file,
                                  int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expr, file, line,
                     cudaGetErrorString(err));
    }
}

#ifndef KVMEM_CUDA_CHECK
#define KVMEM_CUDA_CHECK(expr) \
    ::ninfer::kvmem::kvmem_cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#endif

// Re-bakes one physical page's K phases on every layer plane of the pool.
//
// `physical_page` is the page index within each plane (planes are page-major).
// `src_pos`/`dst_pos` are DEVICE arrays in axis-major layout
// [axis * 64 + token], holding the absolute MRoPE positions the page's K is
// currently baked at and the window positions it must carry after the move.
// Re-phasing operates in the shared D256 Hadamard basis across all four K
// quantization groups; the V planes are position-free and pass through
// untouched.
void kvmem_rerope_page(ninfer::DeviceKVPagePool& pool, std::int32_t physical_page,
                       const std::int32_t* src_pos, const std::int32_t* dst_pos,
                       cudaStream_t stream);

// Reuse the existing decode/un-RoPE/RoPE/encode math with immutable source
// positions (a partial page can have columns born at different window slots).
void kvmem_rebuild_source_k(ninfer::DeviceKVPagePool& pool, std::int32_t physical_page,
                            const std::array<std::int32_t, 64>& source_positions,
                            std::int32_t destination_base, std::uint32_t columns,
                            cudaStream_t stream);

// Bounded two-page restore metadata. The caller retains its immutable source
// owners and pinned input buffers until the stream completes, as for the
// single-page entry. Destination physical pages in a batch must be distinct.
struct SourcePageRestore {
    std::int32_t physical;
    std::array<std::int32_t, 64> positions;
    std::int32_t destination;
    std::uint32_t columns;
};

// Exactly one or two pages, potentially with different valid-column counts.
// SM89 Main16/H4 and MTP1/H4 use one launch for two pages; the original token
// math, identity-column skip, and single-page entry remain unchanged. Other
// supported geometries retain their original single-page launch sequence.
void kvmem_rebuild_source_k_batch(ninfer::DeviceKVPagePool& pool,
                                  std::span<const SourcePageRestore> pages,
                                  cudaStream_t stream);

// Exactly one Main16 attention-plane group, with one or two immutable K
// sources (including partial 64-token tails). Uses the same un-RoPE/re-RoPE
// math and by-value metadata as kvmem_rebuild_source_k_batch, but permits a
// per-layer H2D -> rebuild -> consumer-ready event sequence.
void kvmem_rebuild_source_k_batch_layer(ninfer::DeviceKVPagePool& pool,
                                        std::span<const SourcePageRestore> pages,
                                        std::uint32_t layer, cudaStream_t stream);

// Batched variant: re-bakes `n_blocks` moved physical pages across all layer
// planes of the pool using one H2D metadata upload and one kernel launch per
// layer. `physical_pages`, `from_bases`, and `to_bases` are HOST arrays of
// length `n_blocks`, where each moved block `i` transitions from scalar base
// position `from_bases[i]` to `to_bases[i]`.
struct KvmemReropeWorkspace {
    std::span<std::int32_t> host_meta;
    std::span<std::int32_t> device_meta;
};

using KvmemLayerEnqueue = void (*)(void* owner, std::uint32_t layer, cudaStream_t stream);

void kvmem_rerope_pages(ninfer::DeviceKVPagePool& pool, const std::int32_t* physical_pages,
                        const std::int32_t* from_bases, const std::int32_t* to_bases,
                        std::uint32_t n_blocks, KvmemReropeWorkspace workspace,
                        cudaStream_t stream);

// SM89 Main16 PageMajor opt-in: identical K codec/rephase kernels, but the
// independent producer stream publishes one completion event per full-attention
// layer. Its pinned metadata and device workspace must remain owned until the
// final event completes; the consumer stream waits before reading each layer.
// Unlike kvmem_rerope_pages, this does not host-synchronize the producer.
void kvmem_rerope_pages_layer_ready(ninfer::DeviceKVPagePool& pool,
                                   const std::int32_t* physical_pages,
                                   const std::int32_t* from_bases,
                                   const std::int32_t* to_bases,
                                   std::uint32_t n_blocks, KvmemReropeWorkspace workspace,
                                   std::span<const cudaEvent_t> layer_ready,
                                   cudaStream_t producer,
                                   KvmemLayerEnqueue enqueue_source_layer = nullptr,
                                   void* source_owner = nullptr);

// ---- query-driven retention (K3): persistent retrieval evidence ----
//
// The window's middle quota ranks blocks by how strongly the request's query
// attends to them. The reference harvests pre-RoPE Q over the current user turn
// and compares it against each block's mean raw K. Both sides must be compared
// before the position rotation: a token's key is position independent, so
// ranking blocks written at different distances only works once RoPE is out of
// the way.
//
// The cached K is int8-group64 quantized under the H256 quantization rotation.
// A block's mean K is therefore reconstructed by undoing H256 (an involution),
// un-rotating the rotary dims at the position the page is baked at, and summing
// the raw rows. The sum is kept on the host for every block, so a block parked
// in host memory still carries evidence after it leaves the device.
struct KvmemQueryCapture {
    void* sums = nullptr;  // DEVICE f32, [layers][q_heads][head_dim] pre-RoPE query sums
    std::uint32_t layers      = 0;
    std::uint32_t q_heads     = 0;
    std::uint32_t head_dim    = 0;
    std::uint32_t kv_heads    = 0;
    std::uint32_t query_begin = 0;
    std::uint32_t query_end   = 0;

    [[nodiscard]] bool valid() const noexcept { return sums != nullptr; }
};

// Accumulates the pre-RoPE Q columns of [query_begin, query_end) that this
// chunk carries. `q_rope` is [head_dim, q_heads, tokens] (bf16) and column j
// stands for absolute prompt position `chunk_begin + j`; `sums` is the
// capture's layer `layer` slot. Columns outside the query span contribute
// nothing, so prefilling a prompt in chunks fills the same sum a single pass
// would.
void kvmem_accumulate_query(const void* q_rope, std::uint32_t q_heads, std::uint32_t head_dim,
                            std::uint32_t tokens, std::uint32_t chunk_begin,
                            std::uint32_t query_begin, std::uint32_t query_end, void* sums,
                            std::uint32_t layer, cudaStream_t stream);

// Reconstructs four independent raw-K sums per 64-token page and writes them
// to `mean_k_out` (host): [blocks][layers][kv_heads][subblocks=4][dim] f32.
// `page_indices[i]` is the block's physical page within the pool's planes;
// `token_counts[i]` is the number of written tokens in the block (a window's
// last page may be partial), so unwritten columns are not folded into the mean.
void kvmem_export_mean_k(ninfer::DeviceKVPagePool& pool, const std::int32_t* page_indices,
                         const std::int32_t* baked_positions, const std::uint32_t* token_counts,
                         std::uint32_t n_blocks, float* mean_k_out, cudaStream_t stream);

} // namespace ninfer::kvmem
