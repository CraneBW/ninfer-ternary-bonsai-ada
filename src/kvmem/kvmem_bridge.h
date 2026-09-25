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

} // namespace ninfer::kvmem
