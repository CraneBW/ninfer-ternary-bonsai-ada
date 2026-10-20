#include "kvmem/kvmem_blocks.h"
#include "kvmem/kvmem_cold_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace ninfer::kvmem;

namespace {
int failures = 0;

void check(bool valid, const char* message) {
    if (!valid) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

template <typename F>
bool throws_invalid_argument(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_disk_roundtrip(const std::string& directory) {
    constexpr std::size_t page_bytes = 513; // include trailing bytes outside sector alignment
    std::vector<std::byte> first(page_bytes);
    std::vector<std::byte> second(page_bytes);
    std::vector<std::byte> output(page_bytes);
    for (std::size_t i = 0; i < page_bytes; ++i) {
        first[i] = static_cast<std::byte>((i * 37 + 5) % 256);
        second[i] = static_cast<std::byte>((i * 13 + 191) % 256);
    }
    ColdKVStore store(page_bytes, directory);
    const auto one = store.store(first);
    const auto two = store.store(second);
    check(store.occupied_pages() == 2 && store.occupied_bytes() == page_bytes * 2,
          "two cold pages must be accounted");
    store.load(one, output);
    check(output == first, "first page must round-trip byte-exactly");
    store.load(two, output);
    check(output == second, "second page must round-trip byte-exactly");
    check(throws_invalid_argument([&] { store.load(one, std::span<std::byte>(output).first(512)); }),
          "loading a truncated destination must be rejected");
    check(throws_invalid_argument([&] { (void)store.store(std::span<const std::byte>(first).first(1)); }),
          "storing a short page must be rejected");

    check(store.release(one), "first page releases");
    check(!store.release(one) && !store.valid(one), "double release is refused");
    const auto replacement = store.store(second);
    check(replacement.slot == one.slot && replacement.generation != one.generation,
          "released disk extent is recycled with a new generation");
    check(throws_invalid_argument([&] { store.load(one, output); }),
          "stale handle cannot read recycled data");
    store.load(replacement, output);
    check(output == second, "recycled page is byte-exact");
    store.load(two, output);
    check(output == second, "unrelated live page is unchanged by slot reuse");
    check(store.release(two) && store.release(replacement) && store.occupied_bytes() == 0,
          "cold slots are reclaimed on release");
}

void test_many_cold_pages(const std::string& directory) {
    // A 256K-token context has 4096 64-token blocks. Exercise first-time
    // allocation, bulk reclamation and slot reuse without a model/GPU.
    constexpr std::size_t pages = 4096;
    constexpr std::size_t page_bytes = 64;
    ColdKVStore store(page_bytes, directory);
    std::vector<ColdKVStore::Handle> handles;
    handles.reserve(pages);
    std::vector<std::byte> data(page_bytes);
    std::vector<std::byte> output(page_bytes);
    for (std::size_t page = 0; page < pages; ++page) {
        std::fill(data.begin(), data.end(), static_cast<std::byte>(page % 251));
        handles.push_back(store.store(data));
    }
    check(store.occupied_pages() == pages &&
              store.occupied_bytes() == pages * page_bytes,
          "all 4096 cold extents must be accounted");
    for (std::size_t page : {std::size_t{0}, pages / 2, pages - 1}) {
        store.load(handles[page], output);
        check(std::all_of(output.begin(), output.end(), [&](std::byte value) {
                  return value == static_cast<std::byte>(page % 251);
              }), "cold extents survive 4096 allocations");
    }
    for (std::size_t page = 0; page < pages; page += 2) {
        check(store.release(handles[page]), "bulk release succeeds");
    }
    check(store.occupied_pages() == pages / 2, "bulk release updates page count");
    std::fill(data.begin(), data.end(), std::byte{0x7d});
    for (std::size_t page = 0; page < pages / 2; ++page) {
        const auto replacement = store.store(data);
        check(!store.valid(handles[(pages / 2 - 1 - page) * 2]),
              "old cold generation stays invalid after bulk reuse");
        store.load(replacement, output);
        check(output == data, "reused cold extent is byte-exact");
    }
    check(store.occupied_pages() == pages, "recycled cold extents remain bounded");
}

void test_cold_priority() {
    KvBlockMeta normal;
    normal.id = 1;
    normal.token_begin = 64;
    normal.lrfu_score = 0.1;
    KvBlockMeta closed_think = normal;
    closed_think.id = 2;
    closed_think.token_begin = 128;
    closed_think.stratum = BlockStratum::EphemeralThinking;
    closed_think.lrfu_score = 0.9;
    check(kvmem_colder_than(closed_think, normal),
          "closed thinking is spilled before high-priority ordinary context");
    KvBlockMeta cool_normal = normal;
    cool_normal.lrfu_score = 0.02;
    check(kvmem_colder_than(cool_normal, normal), "lower-LRFU context spills first");
    KvBlockMeta old_normal = normal;
    old_normal.token_begin = 0;
    check(kvmem_colder_than(old_normal, normal), "older equal-score context spills first");
    KvBlockMeta action = normal;
    action.stratum = BlockStratum::ActionProtected;
    KvBlockMeta sink = normal;
    sink.stratum = BlockStratum::PinnedSink;
    check(!kvmem_cold_spillable(action) && !kvmem_cold_spillable(sink),
          "live action and system/tool prefix resist Host pressure");
    check(kvmem_cold_spillable(normal) && kvmem_cold_spillable(closed_think),
          "ordinary and closed-thinking pages are spillable");
}
} // namespace

int main(int argc, char** argv) {
    const std::string directory = argc > 1 ? argv[1] : "./tmp-nvcc/kvmem-cold-test";
    std::filesystem::create_directories(directory);
    test_disk_roundtrip(directory);
    test_many_cold_pages(directory);
    test_cold_priority();
    check(std::filesystem::is_empty(directory),
          "anonymous cold file is unlinked even while its store is alive");
    std::printf("kvmem cold store: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL",
                failures);
    return failures != 0;
}
