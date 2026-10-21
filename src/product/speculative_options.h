#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (options.adaptive_mtp && options.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("--adaptive-mtp requires --spec mtp");
    }
    if (options.lookup_ngram && options.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("--lookup-ngram requires --spec mtp");
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // R29: fixed-window MTP may use the existing T16 GGUF/MMQ path.
        // Keep the CLI, decode frame, model ceiling and mtp_round width in step.
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
        }
        if (options.draft_tokens > 7 && (options.adaptive_mtp || options.lookup_ngram)) {
            throw std::invalid_argument("MTP windows above 7 require fixed drafting without lookup");
        }
        // The adaptive floor is three: a two-wide round buys almost nothing over a one-wide one,
        // so a ceiling below three would leave the controller with nothing to choose between.
        if (options.adaptive_mtp && options.draft_tokens < 3) {
            throw std::invalid_argument("--adaptive-mtp requires --draft-tokens in [3,7]");
        }
        // The lookup offers a prediction only from three tokens up: a shorter one is not worth
        // displacing a draft the head already holds. A window too narrow to verify three would
        // leave the option with nothing it is allowed to draft.
        if (options.lookup_ngram && options.draft_tokens < 3) {
            throw std::invalid_argument("--lookup-ngram requires --draft-tokens in [3,7]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
