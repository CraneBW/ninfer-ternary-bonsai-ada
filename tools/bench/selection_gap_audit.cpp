#include "kvmem/original_replay_diagnostic.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

// Read the already frozen R21 no-probe causal-control receipt. This does not instantiate the model
// or the device, mutate state, replay queries or infer model accuracy from lexical page coverage.
int main(int argc, char** argv) {
    if (argc != 2) { return 2; }
    using ninfer::kvmem::OriginalReplayDiagnostic;
    using ninfer::kvmem::PrefixCheckpointReader;
    PrefixCheckpointReader file(argv[1]);
    auto& in = file.stream();
    in.expect<std::uint64_t>(0x3147414944323152ULL);
    const std::uint32_t q0       = in.scalar<std::uint32_t>();
    const std::uint32_t frontier = in.scalar<std::uint32_t>();
    const auto prefix           = in.vector<std::int32_t>(q0);
    constexpr std::uint32_t page_tokens = 64;
    const std::uint32_t full_pages = (q0 + page_tokens - 1) / page_tokens;
    const auto pages            = in.vector<std::uint32_t>(full_pages);
    const auto state            = in.pod<OriginalReplayDiagnostic::Digest>();
    const auto main             = in.pod<OriginalReplayDiagnostic::Digest>();
    const auto mtp              = in.pod<OriginalReplayDiagnostic::Digest>();
    file.finish();
    if (q0 == 0 || prefix.size() != q0 || pages.empty() || !std::is_sorted(pages.begin(), pages.end()) ||
        std::adjacent_find(pages.begin(), pages.end()) != pages.end() ||
        pages.back() != (q0 - 1) / page_tokens || frontier > pages.size() * page_tokens) {
        throw std::runtime_error("R21 control selected blocks are not a valid original prefix");
    }
    // Reuse exactly the same q0, prefix and full StateImage binding as the no-probe arm.
    const auto bound = OriginalReplayDiagnostic::load(argv[1], q0, prefix, state,
                                                      full_pages * page_tokens);
    if (bound.main != main || bound.mtp != mtp || bound.selected != pages ||
        bound.frontier != frontier) {
        throw std::runtime_error("R21 control changed between reads");
    }

    std::uint32_t first_absent = full_pages;
    std::uint32_t gaps = 0;
    std::uint32_t previous = 0;
    for (const std::uint32_t page : pages) {
        if (page != previous) {
            if (first_absent == full_pages) { first_absent = previous; }
            ++gaps;
        }
        previous = page + 1;
    }
    const std::uint32_t missing = full_pages - static_cast<std::uint32_t>(pages.size());
    if (missing != 0 && first_absent == full_pages) {
        throw std::runtime_error("nonempty gaps must begin before the retained final q0 page");
    }
    std::cout << "{\"q0\":" << q0
              << ",\"native_frontier\":" << frontier
              << ",\"pages_total\":" << full_pages
              << ",\"pages_retained\":" << pages.size()
              << ",\"pages_omitted\":" << missing
              << ",\"first_omitted_page\":" << (missing ? first_absent : full_pages)
              << ",\"first_omitted_token\":" << (missing ? first_absent * page_tokens : q0)
              << ",\"gap_regions\":" << gaps
              << ",\"state_digest_matches_frozen_control\":true"
              << ",\"native_digest_matches_frozen_control\":true}\n";
}
