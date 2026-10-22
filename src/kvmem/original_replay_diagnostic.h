#pragma once

#include "kvmem/prefix_checkpoint_file.h"
#include <charconv>
#include <optional>
#include <span>

namespace ninfer::kvmem {

// Causal control for R21 only. These records contain inputs/selection/digests,
// never answers. Separate from the qualified canonical prefix archive ABI.
// No work or allocations occur unless an explicit diagnostic path is supplied.
struct OriginalReplayDiagnostic {
    struct Digest {
        std::uint64_t low, high;
        friend bool operator==(const Digest&, const Digest&) = default;
    };
    std::uint32_t q0 = 0, frontier = 0;
    std::vector<std::int32_t> prefix;
    std::vector<std::uint32_t> selected;
    Digest state{}, main{}, mtp{};

    static Digest digest(const void* data, std::size_t bytes) {
        const auto h = XXH3_128bits(data, bytes);
        return {h.low64, h.high64};
    }
    template<class Store, class Handle, class Stream>
    static Digest native(Store& store, Handle handle, Stream stream) {
        XXH3_state_t hash;
        XXH3_INITSTATE(&hash);
        XXH3_128bits_reset(&hash);
        store.visit_original_window(handle, [&](const void* data, std::size_t bytes) {
            XXH3_128bits_update(&hash, data, bytes);
        }, stream);
        const auto h = XXH3_128bits_digest(&hash);
        return {h.low64, h.high64};
    }
    static bool selected_query(std::uint32_t q0) {
        const char* value = std::getenv("NINFER_KVMEM_ORIGINAL_DIAG_Q0");
        if (!value) { return false; }
        std::uint32_t wanted = 0;
        const auto end = value + std::strlen(value);
        const auto r = std::from_chars(value, end, wanted);
        if (r.ec != std::errc{} || r.ptr != end || !wanted) {
            throw std::invalid_argument("original diagnostic needs a positive exact q0");
        }
        return q0 == wanted;
    }
    void save(const char* path) const {
        PrefixCheckpointWriter file(path);
        auto& out = file.stream();
        out.scalar<std::uint64_t>(0x3147414944323152ULL); // R21DIAG1
        out.scalar(q0); out.scalar(frontier);
        out.vector(prefix); out.vector(selected);
        out.pod(state); out.pod(main); out.pod(mtp);
        file.finish();
    }
    static OriginalReplayDiagnostic load(const char* path, std::uint32_t q0,
            std::span<const std::int32_t> prefix, Digest state, std::uint32_t budget) {
        PrefixCheckpointReader file(path);
        auto& in = file.stream();
        in.expect<std::uint64_t>(0x3147414944323152ULL);
        OriginalReplayDiagnostic control;
        control.q0 = in.scalar<std::uint32_t>();
        control.frontier = in.scalar<std::uint32_t>();
        control.prefix = in.vector<std::int32_t>(q0);
        control.selected = in.vector<std::uint32_t>(budget / 64);
        control.state = in.pod<Digest>(); control.main = in.pod<Digest>(); control.mtp = in.pod<Digest>();
        file.finish();
        if (control.q0 != q0 || control.prefix.size() != prefix.size() ||
            !std::equal(control.prefix.begin(), control.prefix.end(), prefix.begin()) ||
            control.state != state || control.frontier > budget || control.selected.empty() ||
            !std::is_sorted(control.selected.begin(), control.selected.end()) ||
            std::adjacent_find(control.selected.begin(), control.selected.end()) != control.selected.end() ||
            control.selected.back() != (q0 - 1U) / 64) {
            throw std::invalid_argument("original no-probe control identity/coverage mismatch");
        }
        return control;
    }
};
} // namespace ninfer::kvmem
