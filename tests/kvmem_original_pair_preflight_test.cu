// OpenSpec S1: KVMem/ActKV original-prefix pairing preflight before any tier mutation.
// Reuses the already qualified R21 native storage rig (originally perf-ada-round21);
// no model, Engine, changed mathematics, or full-history inference.
#include <cuda_runtime.h>
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "kvmem/kvmem_layout.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6::detail;
namespace {
constexpr int kPage = 64, kHeads = 4, kLayers = 2, kDim = 256, kGroups = 4;
constexpr int kPhysical = 12, kLogical = 16;
constexpr std::uint8_t kGuard = 0xA5;
std::uint64_t compared_bytes = 0;
std::uint64_t compared_means = 0;
std::uint64_t expected_hash = 14695981039346656037ULL;
std::uint64_t actual_hash = 14695981039346656037ULL;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}
void cuda_ok(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc)); }
}
void hash_bytes(std::uint64_t& value, std::span<const std::uint8_t> bytes) {
    for (const auto byte : bytes) { value = (value ^ byte) * 1099511628211ULL; }
}
template <class F>
void expect_rejected(F&& action, const char* what) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, what);
}
int coefficient(int original, int layer, int head, int token, int version) {
    return 1 + (original * 3 + layer + head + token + version * 2) % 7;
}
std::vector<std::uint8_t> cpu_row(int plane, int original, int head, int token, int version) {
    const int kind = plane % 4, layer = plane / 4;
    const int leading = kind < 2 ? kDim : kGroups;
    std::vector<std::uint8_t> row(leading * (kind < 2 ? 1 : 2));
    if (kind == 0) {
        // Walsh column64: H256(codes) has exactly raw[64] = 16*C.
        // raw[0..63] are zero, so inverse RoPE cannot change this oracle.
        const int c = coefficient(original, layer, head, token, version);
        for (int d = 0; d < leading; ++d) {
            row[d] = static_cast<std::uint8_t>(static_cast<std::int8_t>((d & 64) ? -c : c));
        }
    } else if (kind == 1) {
        for (int d = 0; d < leading; ++d) {
            const int q = (original * 29 + layer * 11 + head * 7 + token * 3 + d + version * 41) % 235 - 117;
            row[d] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
    } else {
        for (int g = 0; g < leading; ++g) {
            // K scales exactly1; V scales are distinct finite represented halves.
            const std::uint16_t bits = kind == 2 ? 0x3C00 :
                static_cast<std::uint16_t>(0x3C00 + (original * 31 + layer * 13 + head * 7 + token + g + version * 17) % 64);
            std::memcpy(row.data() + 2 * g, &bits, sizeof(bits));
        }
    }
    return row;
}

struct Backing {
    void* pointer = nullptr;
    std::size_t bytes = 0;
    ~Backing() { if (pointer) { (void)cudaFree(pointer); } }
};
struct Rig {
    // Declared first so the allocation outlives store/pool/table owners.
    Backing allocation;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::unique_ptr<LogicalKVPageStore> pages;
    std::unique_ptr<DeviceBuffer> rerope;
    std::unique_ptr<KVAddressSpaceStore> store;
    DeviceContext& device;
    KVAddressSpaceHandle handle;
    bool opened = false;

    explicit Rig(DeviceContext& device_) : device(device_) {
        KVPageGeometry geometry;
        geometry.page_tokens = kPage;
        geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < kLayers; ++layer) {
            geometry.planes.push_back({DType::I8, kDim, kHeads, 256});
            geometry.planes.push_back({DType::I8, kDim, kHeads, 256});
            geometry.planes.push_back({DType::FP16, kGroups, kHeads, 256});
            geometry.planes.push_back({DType::FP16, kGroups, kHeads, 256});
        }
        LayoutBuilder builder;
        const auto pl = plan_device_kv_page_pool(builder,
            DeviceKVPagePoolSpec{.page_group_count = kPhysical, .geometry = geometry});
        const auto tl = plan_kv_execution_tables(builder,
            KVExecutionTableSpec{.logical_page_capacity = kLogical, .table_rows = 1});
        allocation.bytes = pl.payload_bytes() + tl.metadata_bytes() + (1U << 20);
        cuda_ok(cudaMalloc(&allocation.pointer, allocation.bytes), "allocate tiny pool");
        cuda_ok(cudaMemset(allocation.pointer, kGuard, allocation.bytes), "initialize guards");
        pool = std::make_unique<DeviceKVPagePool>(DeviceSpan{allocation.pointer, allocation.bytes}, pl);
        tables = std::make_unique<KVExecutionTablePool>(DeviceSpan{allocation.pointer, allocation.bytes}, tl, *pool);
        pages = std::make_unique<LogicalKVPageStore>(*pool, kLogical);
        rerope = std::make_unique<DeviceBuffer>(3ULL * kLogical * sizeof(std::int32_t));
        store = std::make_unique<KVAddressSpaceStore>(*pages, *tables, 1, kLogical,
            KvMemOptions{.enabled = true, .budget_tokens = 10 * kPage, .gen_reserve_tokens = kPage},
            DeviceSpan{rerope->p, rerope->bytes});
        require(store->kvmem_original_positions(), "fixture requires frozen original-position flag1");
        store->install_kvmem_query(kLayers, kHeads, kDim, kHeads);
        open();
    }
    void open() {
        const auto created = store->create_inactive();
        require(created.has_value(), "allocate address");
        handle = *created;
        store->activate(handle, kLogical, 0);
        opened = true;
    }
    int physical(int slot) const {
        const auto logical = store->logical_page(handle, slot);
        return pool->physical_index_of(pages->physical(logical));
    }
    std::byte* row_pointer(int plane, int slot, int head, int token) const {
        const auto tensor = pool->plane(plane);
        return static_cast<std::byte*>(tensor.data) + physical(slot) * tensor.nb[3] +
            head * tensor.nb[2] + token * tensor.nb[1];
    }
    void write(int slot, int original, int begin, int end, int version) {
        for (int plane = 0; plane < kLayers * 4; ++plane) {
            for (int head = 0; head < kHeads; ++head) {
                for (int token = begin; token < end; ++token) {
                    const auto bytes = cpu_row(plane, original, head, token, version);
                    cuda_ok(cudaMemcpy(row_pointer(plane, slot, head, token), bytes.data(), bytes.size(),
                        cudaMemcpyHostToDevice), "write independent CPU fixture bytes");
                }
            }
        }
    }
    void check_page(int slot, int original, int columns, int split = kPage, int suffix_version = 0) {
        for (int plane = 0; plane < kLayers * 4; ++plane) {
            for (int head = 0; head < kHeads; ++head) {
                for (int token = 0; token < columns; ++token) {
                    const auto expected = cpu_row(plane, original, head, token,
                        token < split ? 0 : suffix_version);
                    std::vector<std::uint8_t> actual(expected.size());
                    cuda_ok(cudaMemcpy(actual.data(), row_pointer(plane, slot, head, token), actual.size(),
                        cudaMemcpyDeviceToHost), "read complete represented rows");
                    require(actual == expected, "native K/V/scale bytes differ at original=" +
                        std::to_string(original) + " plane=" + std::to_string(plane) +
                        " token=" + std::to_string(token));
                    hash_bytes(expected_hash, expected);
                    hash_bytes(actual_hash, actual);
                    compared_bytes += actual.size();
                }
            }
        }
    }
    void check_positions(std::span<const std::uint32_t> originals, int last_columns = kPage) const {
        std::vector<std::uint32_t> expected;
        for (std::size_t slot = 0; slot < originals.size(); ++slot) {
            const int columns = slot + 1 == originals.size() ? last_columns : kPage;
            for (int token = 0; token < columns; ++token) { expected.push_back(originals[slot] * kPage + token); }
        }
        require(store->kvmem_window_token_positions(handle) == expected, "packed -> original token map differs");
        require(store->committed_frontier(handle) == expected.size(), "packed committed frontier differs");
        require(store->mapped_pages(handle) == originals.size(), "mapped page coverage differs");
        const auto last = store->logical_page(handle, static_cast<std::uint32_t>(originals.size() - 1));
        require(pages->committed_columns(last) == static_cast<std::uint32_t>(last_columns), "partial tail coverage differs");
    }
    void compact(std::span<const std::uint32_t> selected) {
        const auto result = store->kvmem_compact(handle, selected.size() * kPage,
            device.stream, {}, true, 0, selected);
        require(result != 0, "exact selection did not compact");
        cuda_ok(cudaStreamSynchronize(device.stream), "complete native restore");
        (void)store->retire_kvmem_transfers(true);
    }
    void check_guard() const {
        std::array<std::uint8_t, 256> bytes;
        cuda_ok(cudaMemcpy(bytes.data(), static_cast<std::byte*>(allocation.pointer) + allocation.bytes - bytes.size(),
            bytes.size(), cudaMemcpyDeviceToHost), "read external allocation guard");
        require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == kGuard; }), "external guard overwritten");
    }
    void close() {
        if (!opened) { return; }
        (void)store->retire_kvmem_transfers(true);
        store->deactivate(handle);
        require(store->release(handle), "release original address");
        opened = false;
        require(pool->allocated_pages() == 0 && pool->reserved_pages() == 0 && store->kvmem_cold_bytes() == 0,
            "release retained Device/Host/Cold ownership");
        check_guard();
    }
    ~Rig() {
        if (opened) {
            try { close(); } catch (...) { }
        }
    }
};

void check_partial_mean(const KVAddressSpaceStore::OriginalPrefixCheckpoint& checkpoint,
                        int columns, int split = kPage, int suffix_version = 0) {
    using namespace ninfer::kvmem;
    constexpr std::size_t stride = kLayers * kHeads * kMeanKeySubblocks * kDim;
    require(checkpoint.partial_mean_tokens == static_cast<std::uint32_t>(columns) &&
        checkpoint.partial_mean.size() == stride, "partial mean coverage/shape differs");
    for (int layer = 0; layer < kLayers; ++layer) {
        for (int head = 0; head < kHeads; ++head) {
            for (std::uint32_t sub = 0; sub < kMeanKeySubblocks; ++sub) {
                float sum = 0.0f;
                for (int token = sub * kMeanKeyTokensPerSubblock;
                     token < std::min(columns, static_cast<int>((sub + 1) * kMeanKeyTokensPerSubblock)); ++token) {
                    sum += static_cast<float>(16 * coefficient(5, layer, head, token,
                        token < split ? 0 : suffix_version));
                }
                for (int d = 0; d < kDim; ++d) {
                    const auto offset = ((layer * kHeads + head) * kMeanKeySubblocks + sub) * kDim + d;
                    const float expected = d == 64 ? sum : 0.0f;
                    require(checkpoint.partial_mean[offset] == expected,
                        "partial mean differs from independent Walsh-basis oracle");
                    ++compared_means;
                }
            }
        }
    }
}

[[maybe_unused]] void case_native_move(DeviceContext& device) {
    Rig rig(device);
    rig.store->ensure_mapped_to_tokens(rig.handle, 6 * kPage);
    for (int original = 0; original < 6; ++original) { rig.write(original, original, 0, kPage, 0); }
    rig.store->commit_frontier(rig.handle, 6 * kPage);
    const std::array<std::uint32_t, 4> selected{0, 2, 4, 5};
    rig.compact(selected);
    rig.check_positions(selected);
    for (int slot = 0; slot < 4; ++slot) { rig.check_page(slot, selected[slot], kPage); }
    require(rig.store->kvmem_cold_bytes() != 0, "one-page Host cap did not exercise Cold ownership");
    rig.close();
    std::puts("{\"case\":\"native_nonzero_page_move\",\"verdict\":\"PASS\",\"full_kv_scales_exact\":true,\"guard_exact\":true,\"released\":true}");
}

[[maybe_unused]] void case_partial_rewind(DeviceContext& device) {
    Rig rig(device);
    constexpr std::uint32_t q0 = 5 * kPage + 17;
    rig.store->ensure_mapped_to_tokens(rig.handle, q0);
    for (int original = 0; original < 6; ++original) {
        rig.write(original, original, 0, original == 5 ? 17 : kPage, 0);
    }
    rig.store->commit_frontier(rig.handle, q0);
    const auto q0_checkpoint = rig.store->capture_original_prefix(rig.handle, q0, device.stream);
    check_partial_mean(q0_checkpoint, 17);
    require(q0_checkpoint.prefix_revisions.size() == 6, "q0 prefix page revision count differs");

    rig.store->ensure_mapped_to_tokens(rig.handle, 9 * kPage);
    rig.write(5, 5, 17, kPage, 1);
    for (int original = 6; original < 9; ++original) { rig.write(original, original, 0, kPage, 1); }
    rig.store->commit_frontier(rig.handle, 9 * kPage);
    // Make the live partial-page mean include probe columns, so restoring17 is necessary.
    rig.store->kvmem_refresh_scores(rig.handle, device.stream);
    const std::array<std::uint32_t, 4> probe_view{0, 1, 7, 8};
    rig.compact(probe_view);
    rig.check_positions(probe_view);
    require(rig.store->kvmem_cold_bytes() != 0, "partial/future pages did not migrate to Cold");

    const auto packed = rig.store->rewind_original_prefix(rig.handle, q0_checkpoint, device.stream);
    require(packed == 2 * kPage + 17, "q0 rewind returned wrong current-map frontier");
    const std::array<std::uint32_t, 3> restored{0, 1, 5};
    rig.check_positions(restored, 17);
    rig.check_page(0, 0, kPage);
    rig.check_page(1, 1, kPage);
    rig.check_page(2, 5, 17);
    const auto restored_checkpoint = rig.store->capture_original_prefix(rig.handle, q0, device.stream);
    check_partial_mean(restored_checkpoint, 17);
    expect_rejected([&] { (void)rig.store->rewind_original_prefix(rig.handle, q0_checkpoint, device.stream); },
        "consumed same-address checkpoint was accepted again");
    auto replaced_revision = restored_checkpoint;
    replaced_revision.prefix_revisions[0] ^= 1;
    expect_rejected([&] { (void)rig.store->rewind_original_prefix(rig.handle, replaced_revision, device.stream); },
        "replaced first-write revision was accepted");
    const std::array<std::uint32_t, 1> forbidden_future{7};
    expect_rejected([&] { rig.compact(forbidden_future); }, "parked future directory survived rewind");
    rig.check_positions(restored, 17);

    // Reuse the partial page with a NEW suffix; old future bytes must not become licensed.
    rig.store->resize_entitlement(rig.handle, kLogical);
    rig.store->ensure_mapped_to_tokens(rig.handle, packed + 6);
    rig.write(2, 5, 17, 23, 2);
    rig.store->commit_frontier(rig.handle, packed + 6);
    rig.check_positions(restored, 23);
    rig.check_page(2, 5, 23, 17, 2);
    const auto new_checkpoint = rig.store->capture_original_prefix(rig.handle, q0 + 6, device.stream);
    check_partial_mean(new_checkpoint, 23, 17, 2);
    const auto stale_generation = new_checkpoint;
    rig.close();
    rig.open();
    rig.store->ensure_mapped_to_tokens(rig.handle, kPage);
    rig.write(0, 0, 0, kPage, 0);
    rig.store->commit_frontier(rig.handle, kPage);
    expect_rejected([&] { (void)rig.store->rewind_original_prefix(rig.handle, stale_generation, device.stream); },
        "released/recreated address generation accepted old checkpoint");
    const std::array<std::uint32_t, 1> fresh{0};
    rig.check_positions(fresh);
    rig.check_page(0, 0, kPage);
    rig.close();
    std::puts("{\"case\":\"partial_q0_probe_cold_rewind_append\",\"verdict\":\"PASS\",\"future_invisible\":true,\"partial_mean_cpu_exact\":true,\"lineage_reuse_rejected\":true,\"generation_reuse_rejected\":true,\"guard_exact\":true,\"released\":true}");
}

void case_paired_validation_before_mutation(DeviceContext& device) {
    Rig main(device);
    Rig draft(device);
    constexpr std::uint32_t q0 = 5 * kPage + 17;
    constexpr std::uint32_t probe_frontier = 7 * kPage;
    const auto stage = [&](Rig& rig) {
        rig.store->ensure_mapped_to_tokens(rig.handle, q0);
        for (int page = 0; page < 6; ++page) {
            rig.write(page, page, 0, page == 5 ? 17 : kPage, 0);
        }
        rig.store->commit_frontier(rig.handle, q0);
        const auto checkpoint = rig.store->capture_original_prefix(rig.handle, q0, device.stream);
        rig.store->ensure_mapped_to_tokens(rig.handle, probe_frontier);
        rig.write(5, 5, 17, kPage, 1);
        rig.write(6, 6, 0, kPage, 1);
        rig.store->commit_frontier(rig.handle, probe_frontier);
        return checkpoint;
    };
    const auto main_q0 = stage(main);
    const auto draft_q0 = stage(draft);
    require(main.store->committed_frontier(main.handle) == probe_frontier,
            "Main probe frontier was not established");
    require(draft.store->committed_frontier(draft.handle) == probe_frontier,
            "MTP probe frontier was not established");

    auto stale_draft = draft_q0;
    stale_draft.prefix_revisions[0] ^= 1;
    expect_rejected([&] {
        // This is the production Program's order: validate both BEFORE the
        // first destructive rewind. A stale MTP revision must not alter Main.
        main.store->validate_original_prefix(main.handle, main_q0);
        draft.store->validate_original_prefix(draft.handle, stale_draft);
        (void)main.store->rewind_original_prefix(main.handle, main_q0, device.stream);
    }, "paired checkpoint accepted replaced MTP revision");
    require(main.store->committed_frontier(main.handle) == probe_frontier,
            "failed MTP preflight partially rewound Main");
    require(draft.store->committed_frontier(draft.handle) == probe_frontier,
            "failed MTP preflight changed its own frontier");
    main.store->validate_original_prefix(main.handle, main_q0);
    draft.store->validate_original_prefix(draft.handle, draft_q0);

    const auto main_restored =
        main.store->rewind_original_prefix(main.handle, main_q0, device.stream);
    const auto draft_restored =
        draft.store->rewind_original_prefix(draft.handle, draft_q0, device.stream);
    require(main_restored == draft_restored && main_restored == q0,
            "paired rewind does not restore one q0 execution frontier");
    // A subsequent DMA/StateImage failure after either destructive rewind must
    // poison both owners, even if one had completed its own native restoration.
    main.store->invalidate_original_prefix_branch(main.handle);
    draft.store->invalidate_original_prefix_branch(draft.handle);
    expect_rejected([&] { main.store->validate_original_prefix(main.handle, main_q0); },
                    "poisoned Main original owner remained reusable");
    expect_rejected([&] { draft.store->validate_original_prefix(draft.handle, draft_q0); },
                    "poisoned MTP original owner remained reusable");
    main.close();
    draft.close();
    std::puts("{\"case\":\"original_pair_preflight\",\"verdict\":\"PASS\","
              "\"bad_draft_rejected_before_main_mutation\":true,"
              "\"matched_main_mtp_restored\":true,\"whole_branch_poison_rejected\":true,"
              "\"released\":true}");
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        require(argc <= 2, "usage: ninfer_kvmem_original_pair_preflight_test [COLD_DIRECTORY]");
        const auto directory = argc == 2
            ? std::filesystem::absolute(argv[1])
            : std::filesystem::temp_directory_path() /
              ("ninfer-kvmem-pair-preflight-" + std::to_string(::getpid()));
        if (argc == 1) {
            require(std::filesystem::create_directory(directory),
                    "isolated pair preflight cold directory already exists");
        } else {
            std::filesystem::create_directories(directory);
        }
        require(::setenv("NINFER_KVMEM_ORIGINAL_POSITIONS", "1", 1) == 0, "set original-position policy");
        require(::setenv("NINFER_KVMEM_HOST_PAGES", "1", 1) == 0, "set one-page Host cap");
        require(::setenv("NINFER_KVMEM_COLD_DIR", directory.c_str(), 1) == 0, "set unique Cold backing directory");
        ::unsetenv("NINFER_KVMEM_TRACE");
        DeviceContext device;
        case_paired_validation_before_mutation(device);
        require(expected_hash == actual_hash, "full native byte hash differs");
        std::printf("{\"summary\":true,\"cases\":1,\"verdict\":\"PASS\",\"compared_bytes\":%llu,\"compared_mean_values\":%llu,\"expected_fnv64\":\"%016llx\",\"actual_fnv64\":\"%016llx\",\"models_loaded\":0}\n",
            static_cast<unsigned long long>(compared_bytes), static_cast<unsigned long long>(compared_means),
            static_cast<unsigned long long>(expected_hash), static_cast<unsigned long long>(actual_hash));
        if (argc == 1) { std::filesystem::remove_all(directory); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native-prefix-probe FAIL: %s\n", error.what());
        return 1;
    }
}
