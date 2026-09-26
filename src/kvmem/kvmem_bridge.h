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

// ---- query-driven retention (K3): retrieval scoring ----
//
// The window's middle quota ranks blocks by how strongly the request's query
// attends to them. The query is the last prompt token's Q and a block's score is
// the cosine between that Q and the block's mean-K over its 64 tokens -- the
// laamaafung/v26 retrieval signal.
//
// Both sides are compared BEFORE the position rotation, and that is the whole
// point of the signal: a token's key is position independent, so ranking blocks
// written at different distances from the query only works once RoPE is out of
// the way (the reference harvests pre-RoPE K/Q for the same reason). The cached
// K is int8-group64 quantized under the H256 quantization rotation, so the
// scorer undoes H256 (an involution), un-rotates the rotary dims at the
// position the page is actually baked at, and compares against the raw Q.
struct KvmemQueryCapture {
    const void* data = nullptr;  // DEVICE bf16, [layers][q_heads][head_dim]
    std::uint32_t layers   = 0;
    std::uint32_t q_heads  = 0;
    std::uint32_t head_dim = 0;

    [[nodiscard]] bool valid() const noexcept { return data != nullptr; }
};

// Copies column `column` of a pre-RoPE Q tensor laid out
// [head_dim, q_heads, tokens] (bf16) into the capture's layer `layer` slot.
void kvmem_capture_query(const void* q_rope, std::uint32_t q_heads, std::uint32_t head_dim,
                         std::uint32_t tokens, std::uint32_t column, void* dst,
                         std::uint32_t layer, cudaStream_t stream);

// Scores each of `n_blocks` ledger blocks against the captured query and writes
// one cosine per block to `scores_out` (host). `page_indices[i]` is the block's
// physical page within the pool's planes, or negative when its K is not
// device-resident: such a block keeps its 0 ("no evidence"), so a run without a
// usable query reproduces the unranked selection exactly. `baked_positions[i]`
// is the absolute position the block's first token is baked at, which is what
// the un-rotation inverts.
void kvmem_score_blocks(ninfer::DeviceKVPagePool& pool, const KvmemQueryCapture& query,
                        const std::int32_t* page_indices, const std::int32_t* baked_positions,
                        std::uint32_t n_blocks, float* scores_out, cudaStream_t stream);

} // namespace ninfer::kvmem
