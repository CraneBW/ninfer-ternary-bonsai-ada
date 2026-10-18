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
#include <cstdio>

namespace ninfer {

class DeviceKVPagePool;

} // namespace ninfer

namespace ninfer::kvmem {

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

#define KVMEM_CUDA_CHECK(expr) \
    ::ninfer::kvmem::kvmem_cuda_check_impl((expr), #expr, __FILE__, __LINE__)

// Re-bakes one physical page's K phases on every layer plane of the pool.
//
// `physical_page` is the page index within each plane (planes are page-major).
// `src_pos`/`dst_pos` are DEVICE arrays in axis-major layout
// [axis * 64 + token], holding the absolute MRoPE positions the page's K is
// currently baked at and the window positions it must carry after the move.
// Only group 0 (dims [0,64)) is touched; groups 1..3 and the V planes are
// position-free and pass through untouched.
void kvmem_rerope_page(ninfer::DeviceKVPagePool& pool, std::int32_t physical_page,
                       const std::int32_t* src_pos, const std::int32_t* dst_pos,
                       cudaStream_t stream);

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

// Reconstructs the raw mean-K sum of every device-resident block and writes it
// to `mean_k_out` (host), laid out [n_blocks][layers][kv_heads][head_dim] f32.
// `page_indices[i]` is the block's physical page within the pool's planes;
// `token_counts[i]` is the number of written tokens in the block (a window's
// last page may be partial), so unwritten columns are not folded into the mean.
void kvmem_export_mean_k(ninfer::DeviceKVPagePool& pool, const std::int32_t* page_indices,
                         const std::int32_t* baked_positions, const std::uint32_t* token_counts,
                         std::uint32_t n_blocks, float* mean_k_out, cudaStream_t stream);

} // namespace ninfer::kvmem
