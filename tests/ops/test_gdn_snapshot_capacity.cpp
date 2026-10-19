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
        kQueryRows, kKeyRows, kValueRows, 1, kWidth, kWidth, /*folded_ternary_rotation=*/true);
    const std::size_t batched = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, kBatch, kWidth, kWidth, /*folded_ternary_rotation=*/true);
    failures +=
        check(batched >= plane + rotation,
              "batched snapshot capacity holds the projected plane and the folded ternary rotation "
              "buffer together");
    failures += check(batched > single, "batched snapshot capacity grows with the batch");
    return failures;
}

// The row geometry above is shared: the folded ternary artifact and the groupwise-int Q4/Q5 and W8
// ones resolve to the same query, so the capacity function takes the encoding as a flag. Only the
// ternary parents keep the rotation buffer, and only they materialize a plane on the fused widths,
// so an int artifact must still see the tight statement it saw before the ternary port.
int test_the_encoding_selects_the_statement() {
    constexpr std::int32_t kQueryRows   = 2048;
    constexpr std::int32_t kKeyRows     = 2048;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kChannels    = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kBatch       = 2;
    constexpr std::int32_t kFusedWidth  = 6;
    constexpr std::int32_t kRecordWidth = 2; // ReplaySSM's record domain starts at T=2
    constexpr std::size_t kElementBytes = sizeof(std::uint16_t);

    const std::size_t fused_plane = static_cast<std::size_t>(kChannels) * kFusedWidth * kElementBytes;
    const std::size_t fused_rotation =
        static_cast<std::size_t>(kHidden) * kFusedWidth * kElementBytes;
    const std::size_t int_single = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, 1, 1);
    const std::size_t ternary_single = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, 1, 1, /*folded_ternary_rotation=*/true);
    const std::size_t int_fused = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, kFusedWidth, kFusedWidth);
    const std::size_t ternary_fused = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, kFusedWidth, kFusedWidth,
        /*folded_ternary_rotation=*/true);
    const std::size_t int_record = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, kBatch, kRecordWidth, kRecordWidth);
    const std::size_t ternary_record = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, kBatch, kRecordWidth, kRecordWidth,
        /*folded_ternary_rotation=*/true);
    const std::size_t record_rotation =
        static_cast<std::size_t>(kHidden) * kBatch * kRecordWidth * kElementBytes;

    int failures = 0;
    failures += check(int_single < ternary_single,
                      "the folded ternary statement adds the rotation buffer on top of the plane");
    failures += check(int_fused == 0,
                      "the groupwise-int statement stays empty on a width the fused schedule covers");
    failures += check(ternary_fused >= fused_plane + fused_rotation,
                      "the folded ternary statement materializes the plane on every width");
    failures += check(int_record == 0,
                      "the groupwise-int record statement needs no transient storage");
    failures += check(ternary_record >= record_rotation,
                      "the folded ternary record statement holds the rotation buffer");
    return failures;
}

} // namespace

int main() {
    int failures = test_batched_capacity_covers_the_projection_workspace();
    failures += test_the_encoding_selects_the_statement();
    if (failures == 0) { std::cout << "gdn snapshot capacity tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
