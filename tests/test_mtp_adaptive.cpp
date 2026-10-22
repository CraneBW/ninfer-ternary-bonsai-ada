#include "targets/qwen3_6/impl/runtime/mtp_adaptive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace {

using ninfer::targets::qwen3_6::detail::kAdaptiveMtpMaximumBatch;
using ninfer::targets::qwen3_6::detail::MtpAdaptiveBatchController;
using ninfer::targets::qwen3_6::detail::MtpAdaptiveSignal;
using ninfer::targets::qwen3_6::detail::minimum_adaptive_window;
using ninfer::targets::qwen3_6::detail::mtp_adaptive_cohort_key;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "mtp adaptive test failed: %s\n", message);
        std::exit(1);
    }
}

bool close_to(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-4F;
}

// One round's selection over a cohort whose rows all carry the same signal.
std::uint32_t select_once(MtpAdaptiveBatchController& controller, MtpAdaptiveSignal& signal,
                          std::uint32_t available, std::uint32_t room,
                          std::uint64_t cohort_key) {
    std::array<const MtpAdaptiveSignal*, kAdaptiveMtpMaximumBatch> signals{&signal};
    const std::array<std::uint32_t, kAdaptiveMtpMaximumBatch> available_rows{available};
    const std::array<std::uint32_t, kAdaptiveMtpMaximumBatch> room_rows{room};
    return controller.select(std::span<const MtpAdaptiveSignal* const>(signals.data(), 1),
                             std::span<const std::uint32_t>(available_rows.data(), 1),
                             std::span<const std::uint32_t>(room_rows.data(), 1), cohort_key);
}

} // namespace

// The adaptive controller prices a wider verification round from the drafts' measured survival and
// the round cost model. This fixture pins the prior shape, the evidence ramp, and the two behaviors
// the MTP decode loop depends on: a batch keeps its width while the evidence is flat, and it narrows
// once the wide positions stop being accepted.
int main() {
    // A fresh signal reproduces the prior survival curve: 1 committed token plus the prior of each
    // verified position, with the continuation floor (0.60 of the previous position's survival)
    // lifting the deep positions above their prior until the first observations arrive.
    {
        const MtpAdaptiveSignal signal;
        require(close_to(signal.expected_tokens(1, 7), 1.54F), "position-1 prior");
        require(close_to(signal.expected_tokens(2, 7), 1.864F), "position-2 prior");
        require(close_to(signal.expected_tokens(3, 7), 2.0584F), "position-3 prior");
    }

    // Evidence dominates the prior: an always-accepted window approaches its full width, an
    // always-rejected one collapses to one token.
    {
        MtpAdaptiveSignal accepted;
        MtpAdaptiveSignal rejected;
        for (int round = 0; round < 16; ++round) {
            accepted.observe(7, 7);
            rejected.observe(7, 0);
        }
        require(accepted.expected_tokens(7, 7) > 6.0F, "accepted window holds its width");
        require(rejected.expected_tokens(7, 7) < 2.0F, "rejected window collapses");
        require(accepted.expected_tokens(7, 7) > accepted.expected_tokens(4, 7),
                "expected tokens grow with the window");
    }

    // A confident tail is what licenses a probe above the current width.
    {
        MtpAdaptiveSignal signal;
        for (int round = 0; round < 6; ++round) { signal.observe(7, 7); }
        require(signal.confident_tail(3), "three accepted positions license a wider probe");
        require(!signal.confident_tail(7) || signal.confident_tail(6), "tail streak is monotone");
    }

    // Wide-survival evidence widens the round; the result never leaves [1, draft window].
    {
        MtpAdaptiveSignal signal;
        for (int round = 0; round < 12; ++round) { signal.observe(7, 7); }
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        std::uint32_t widest = 0;
        for (int round = 0; round < 64; ++round) {
            const std::uint32_t window = select_once(controller, signal, 7, 4096, 11);
            require(window >= 1 && window <= 7, "selection stays inside the graph widths");
            widest = window;
            controller.observe_execution(1, window, 0.010);
        }
        require(widest >= 3, "a fully accepted window widens past the floor");
    }

    // Once the wide positions stop being accepted the controller narrows again.
    {
        MtpAdaptiveSignal signal;
        for (int round = 0; round < 12; ++round) { signal.observe(7, 7); }
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        for (int round = 0; round < 40; ++round) {
            const std::uint32_t window = select_once(controller, signal, 7, 4096, 11);
            controller.observe_execution(1, window, 0.010);
        }
        require(controller.selected_window() >= 3, "the floor holds while drafts stay available");
        std::uint32_t narrowest = controller.selected_window();
        for (int round = 0; round < 64; ++round) {
            signal.observe(7, 1);
            const std::uint32_t window = select_once(controller, signal, 7, 4096, 11);
            controller.observe_execution(1, window, 0.010);
            narrowest = std::min(narrowest, window);
        }
        // The round narrows once the wide positions stop surviving. It does not necessarily stay
        // narrow: only the executed widths carry cost samples, so a width the batch has not run yet
        // is priced from the measured shape and can look marginally cheaper than the incumbent. The
        // decode loop only relies on the round actually contracting, not on a settled narrow width.
        require(narrowest <= 4, "dead wide drafts narrow the round");
        require(controller.selected_window() >= 3 && controller.selected_window() <= 6,
                "the round does not return to the full window after the contraction");
    }

    // A perfect run of rounds lands on the widest admissible graph at once: the landing is a
    // decision about where the round should run, not a width to climb one draft at a time. The
    // controller still spends the first rounds at the floor, because that is where the evidence
    // comes from and where a wrong landing would be caught.
    {
        MtpAdaptiveSignal signal;
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        for (int round = 0; round < 3; ++round) {
            const std::uint32_t window = select_once(controller, signal, 7, 4096, 5);
            require(window < 7, "the landing waits for its evidence");
            controller.observe_execution(1, window, 0.010);
            signal.observe(window, window);
        }
        require(select_once(controller, signal, 7, 4096, 5) == 7,
                "three perfect rounds land on the widest verification width");
    }

    // Rows without enough room cap the selectable width: a one-token budget cannot verify drafts.
    {
        MtpAdaptiveSignal signal;
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        const std::uint32_t window = select_once(controller, signal, 0, 1, 7);
        require(window <= 3, "no room keeps the round narrow");
    }

    // A different live request set must not inherit a widened width or its transition counter.
    // XOR of lane IDs used to collide for {0,1} and {2,3}; the lane epoch additionally distinguishes
    // a new request occupying the same lane after a prior request has retired.
    {
        const std::array<std::uint32_t, 2> first_lanes{0, 1};
        const std::array<std::uint32_t, 2> other_lanes{2, 3};
        const std::array<std::uint32_t, 2> reversed_lanes{1, 0};
        std::array<std::uint64_t, kAdaptiveMtpMaximumBatch> epochs{};
        epochs.fill(1);
        const auto first_key = mtp_adaptive_cohort_key(first_lanes, epochs);
        const auto other_key = mtp_adaptive_cohort_key(other_lanes, epochs);
        require(first_key != other_key, "different lane sets distinguish cohorts");
        require(first_key == mtp_adaptive_cohort_key(reversed_lanes, epochs),
                "cohort identity is independent of row ordering");
        ++epochs[0];
        require(first_key != mtp_adaptive_cohort_key(first_lanes, epochs),
                "lane reuse with a new epoch distinguishes cohorts");

        MtpAdaptiveSignal strong;
        MtpAdaptiveSignal fresh;
        for (int round = 0; round < 12; ++round) { strong.observe(7, 7); }
        const std::array<const MtpAdaptiveSignal*, 2> strong_rows{&strong, &strong};
        const std::array<const MtpAdaptiveSignal*, 2> fresh_rows{&fresh, &fresh};
        const std::array<std::uint32_t, 2> available{7, 7};
        const std::array<std::uint32_t, 2> room{4096, 4096};
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        for (int round = 0; round < 16; ++round) {
            const auto width = controller.select(strong_rows, available, room, first_key);
            controller.observe_execution(2, width, 0.010);
        }
        require(controller.selected_window() >= 6, "strong first cohort widened");
        const auto second_width = controller.select(fresh_rows, available, room, other_key);
        require(second_width == 3, "new cohort begins from narrow startup width");
        require(!controller.transitioned(), "new cohort startup is not a width transition");
    }

    require(minimum_adaptive_window(2) == 2 && minimum_adaptive_window(5) == 3,
            "the adaptive floor is the draft window below three, otherwise three");
    std::printf("mtp adaptive tests passed\n");
    return 0;
}
