#include "targets/qwen3_6/impl/runtime/ngram_lookup.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::NgramLookup;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "ngram lookup test failed: %s\n", message);
        std::exit(1);
    }
}

// Leading text a "document" starts with: long enough that the passage's runs occur exactly once,
// and disjoint from the passage's token values.
std::vector<TokenId> leading_tokens() {
    std::vector<TokenId> tokens;
    for (TokenId index = 0; index < 4096; ++index) {
        tokens.push_back(1000 + index * 7 % 4093);
    }
    return tokens;
}

std::vector<TokenId> passage_tokens() {
    return {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56};
}

} // namespace

// --lookup-ngram exists to draft copied text. This fixture pins the contract the MTP round relies
// on: a predicted token is the continuation of an earlier occurrence of the same eight-token run, a
// request cannot predict the text it is standing on, an unseen run predicts nothing, and slicing
// the appends (prefill pieces, one committed prefix at a time) indexes the same stream.
int main() {
    const std::vector<TokenId> leading = leading_tokens();
    const std::vector<TokenId> passage = passage_tokens();
    const std::array<TokenId, 3> trailing{70001, 70002, 70003};

    std::vector<TokenId> prompt = leading;
    prompt.insert(prompt.end(), passage.begin(), passage.end());
    prompt.insert(prompt.end(), trailing.begin(), trailing.end());

    NgramLookup lookup;
    lookup.append(std::span<const TokenId>(prompt));
    require(lookup.size() == static_cast<std::uint32_t>(prompt.size()),
            "the index counts the appended tokens");

    // The newest context is the text the request is standing on; it has no continuation of its own
    // in the index, so nothing is predicted until the request repeats something.
    std::array<TokenId, 7> prediction{};
    require(lookup.predict(std::span<TokenId>(prediction)) == 0,
            "an unseen context predicts nothing");

    // The model reproduces the passage. Once its first eight tokens are committed, the lookup finds
    // the earlier occurrence and offers that occurrence's continuation.
    lookup.append(std::span<const TokenId>(passage.data(), 8));
    const std::uint32_t predicted = lookup.predict(std::span<TokenId>(prediction));
    require(predicted == 7, "a copied run predicts up to the prediction limit");
    for (std::uint32_t index = 0; index < predicted; ++index) {
        require(prediction[index] == passage[8 + index],
                "the prediction continues the earlier occurrence");
    }

    // Committing the prediction moves the run forward: the next prediction starts where the
    // passage's earlier occurrence continued, which is the text that followed it, not a repeat of
    // the tokens just committed.
    lookup.append(std::span<const TokenId>(prediction.data(), predicted));
    const std::uint32_t followed = lookup.predict(std::span<TokenId>(prediction));
    require(followed >= 2, "the run keeps predicting what followed the occurrence");
    require(prediction[0] == passage[15], "the last passage token comes first");
    require(prediction[1] == trailing[0], "then the text that followed the passage");

    // A narrower ask is honored: the prediction never exceeds the room the round has for drafts.
    std::array<TokenId, 2> narrow{};
    require(lookup.predict(std::span<TokenId>(narrow)) == 2, "the ask bounds the prediction");
    require(narrow[0] == prediction[0], "a narrower ask returns the same prefix");

    // Unrelated text never produces a prediction: eight tokens of context have to repeat exactly.
    NgramLookup prose;
    prose.append(std::span<const TokenId>(leading));
    const std::array<TokenId, 8> novel{9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008};
    prose.append(std::span<const TokenId>(novel));
    require(prose.predict(std::span<TokenId>(prediction)) == 0,
            "a context that never occurred predicts nothing");

    // The same run followed by a different token the second time is a template (a JSON object with
    // one varied field), not a copy: it is retired as soon as the divergence lands inside the
    // agreement window, so a prediction never drafts the varied part.
    NgramLookup templated;
    std::vector<TokenId> template_text = leading;
    template_text.insert(template_text.end(), passage.begin(), passage.end());
    template_text.push_back(6001);
    template_text.insert(template_text.end(), 64, 7000);
    template_text.insert(template_text.end(), passage.begin(), passage.end());
    template_text.push_back(6002);
    templated.append(std::span<const TokenId>(template_text));
    templated.append(std::span<const TokenId>(passage.data(), 8));
    require(templated.predict(std::span<TokenId>(prediction)) == 0,
            "a run whose continuation diverges inside the window is not a copy");

    // A run that agrees for a whole prediction still drafts: the divergence has to land inside the
    // agreement window before the lookup gives up on it. Eight identical tokens are the shortest
    // shared continuation it can prove, and a prediction is never wider than that.
    NgramLookup agreeing;
    std::vector<TokenId> agreeing_text = leading;
    agreeing_text.insert(agreeing_text.end(), passage.begin(), passage.end());
    agreeing_text.insert(agreeing_text.end(), 64, 7000);
    agreeing_text.insert(agreeing_text.end(), passage.begin(), passage.end());
    agreeing_text.push_back(6002);
    agreeing.append(std::span<const TokenId>(agreeing_text));
    agreeing.append(std::span<const TokenId>(passage.data(), 8));

    // The agreement window reaches past the first continuation token: a run whose continuation
    // agreed once and then diverged is a repeated phrase as well, and is retired before it can
    // draft the wrong continuation.
    NgramLookup diverging;
    std::vector<TokenId> diverging_text = leading;
    const auto run_begin                  = passage.begin();
    const auto run_end                    = passage.begin() + 8;
    diverging_text.insert(diverging_text.end(), run_begin, run_end);
    diverging_text.insert(diverging_text.end(), {61, 62, 63});
    diverging_text.insert(diverging_text.end(), 64, 7000);
    diverging_text.insert(diverging_text.end(), run_begin, run_end);
    diverging_text.insert(diverging_text.end(), {61, 62, 64});
    diverging_text.insert(diverging_text.end(), 65, 7001);
    diverging.append(std::span<const TokenId>(diverging_text));
    diverging.append(std::span<const TokenId>(run_begin, 8));
    require(diverging.predict(std::span<TokenId>(prediction)) == 0,
            "a run that diverges past the first token is not a copy");

    // A prediction stops where the earlier occurrence's continuation stops being a copy of it. The
    // index records a run only when its continuation is already known, so a hit always has at least
    // one token; kMinimumPrediction then decides whether that is worth displacing the draft head.
    require(NgramLookup::kMaximumPrediction >= NgramLookup::kMinimumPrediction,
            "the prediction bounds are ordered");
    require(NgramLookup::kMinimumPrediction > 1, "one token is not worth a lookup round");
    require(NgramLookup::kMaximumPrediction <= NgramLookup::kAgreementTokens,
            "a prediction never reaches past the continuation both occurrences agreed on");

    // Appending in slices indexes the same stream as one append: prefill pieces and committed
    // prefixes arrive in pieces of one length or another, never as the whole document at once.
    NgramLookup sliced;
    for (std::size_t begin = 0; begin < prompt.size(); begin += 97) {
        const std::size_t end = std::min(begin + 97, prompt.size());
        sliced.append(std::span<const TokenId>(prompt.data() + begin, end - begin));
    }
    sliced.append(std::span<const TokenId>(passage.data(), 8));
    require(sliced.predict(std::span<TokenId>(prediction)) == 7,
            "sliced appends index the same stream as one append");

    std::puts("ngram lookup tests passed");
    return 0;
}
