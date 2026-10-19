#include "ninfer/ops/gdn_input_proj.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace ninfer;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

// The folded ternary parents project through the batched compose route of the snapshot Op. That
// route keeps the [channels, batch * width] projected plane and the projection's own [hidden,
// batch * width] rotation buffer alive together, so the capacity statement has to cover both. It
// once covered only the plane, and a two-lane decode of a ternary artifact ran out of the arena
// the Program sized from this query.
int test_batched_capacity_covers_the_projection_workspace() {
    constexpr std::int32_t kQueryRows   = 2048;
    constexpr std::int32_t kKeyRows     = 2048;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kChannels    = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kBatch       = 2;
    constexpr std::int32_t kWidth       = 1;
    constexpr std::size_t kElementBytes = sizeof(std::uint16_t);

    const std::size_t plane = static_cast<std::size_t>(kChannels) * kBatch * kWidth * kElementBytes;
    const std::size_t rotation =
        static_cast<std::size_t>(kHidden) * kBatch * kWidth * kElementBytes;
    int failures        = 0;
    const std::size_t single = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, kWidth, kWidth);
    const std::size_t batched = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, kBatch, kWidth, kWidth);
    failures +=
        check(batched >= plane + rotation,
              "batched snapshot capacity holds the projected plane and the folded ternary rotation "
              "buffer together");
    failures += check(batched > single, "batched snapshot capacity grows with the batch");
    return failures;
}

} // namespace

int main() {
    const int failures = test_batched_capacity_covers_the_projection_workspace();
    if (failures == 0) { std::cout << "gdn snapshot capacity tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
