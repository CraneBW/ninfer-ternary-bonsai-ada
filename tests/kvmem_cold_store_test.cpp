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
    test_cold_priority();
    check(std::filesystem::is_empty(directory),
          "anonymous cold file is unlinked even while its store is alive");
    std::printf("kvmem cold store: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL",
                failures);
    return failures != 0;
}
