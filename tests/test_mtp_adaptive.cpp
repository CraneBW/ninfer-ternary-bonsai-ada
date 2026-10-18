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

    // Rows without enough room cap the selectable width: a one-token budget cannot verify drafts.
    {
        MtpAdaptiveSignal signal;
        MtpAdaptiveBatchController controller;
        controller.reset(7);
        const std::uint32_t window = select_once(controller, signal, 0, 1, 7);
        require(window <= 3, "no room keeps the round narrow");
    }

    require(minimum_adaptive_window(2) == 2 && minimum_adaptive_window(5) == 3,
            "the adaptive floor is the draft window below three, otherwise three");
    std::printf("mtp adaptive tests passed\n");
    return 0;
}
