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
    std::vector<TokenId> tokens;
    for (TokenId token = 41; token < 81; ++token) { tokens.push_back(token); }
    return tokens;
}

} // namespace

// --lookup-ngram exists to draft copied text. This fixture pins the contract the MTP round relies
// on: a predicted token is the continuation of an earlier occurrence of the same eight-token run, a
// run counts as copied only once the tokens on both sides of the match agree, an unseen run predicts
// nothing, an example in the prompt is not a copy however it is reached, and slicing the appends
// (prefill pieces, one committed prefix at a time) indexes the same stream.
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

    // The model reproduces the passage. The first copied tokens already complete a run whose earlier
    // occurrence is in the prompt, but the tokens before that occurrence are the document's, not the
    // ones the copy has written: the match alone is a local pattern, and its continuation is what
    // the request is about to decide for itself.
    lookup.append(std::span<const TokenId>(passage.data(), 8));
    require(lookup.predict(std::span<TokenId>(prediction)) == 0,
            "a copy is not predicted before its left side corroborates the occurrence");

    // Once enough of the copy stands before the run, both sides of the earlier occurrence agree and
    // its continuation is the text the request is about to reproduce.
    lookup.append(std::span<const TokenId>(passage.data() + 8, 16));
    const std::uint32_t predicted = lookup.predict(std::span<TokenId>(prediction));
    require(predicted == 7, "a corroborated copy predicts up to the prediction limit");
    for (std::uint32_t index = 0; index < predicted; ++index) {
        require(prediction[index] == passage[24 + index],
                "the prediction continues the earlier occurrence");
    }

    // Committing the prediction moves the run forward: the next prediction starts where the
    // passage's earlier occurrence continued, not at a repeat of the tokens just committed.
    lookup.append(std::span<const TokenId>(prediction.data(), predicted));
    const std::uint32_t followed = lookup.predict(std::span<TokenId>(prediction));
    require(followed == 7, "the run keeps predicting as the copy advances");
    require(prediction[0] == passage[31], "the prediction advances with the copy");

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

    // A single occurrence in the prompt is an example, not a copy. Generated JSON reaches the runs
    // of the example again -- the request is writing its own object, which happens to share the
    // eight tokens -- and one or two matching tokens on the left are coincidence, not corroboration.
    // The earlier occurrence's continuation is another example, not the answer.
    NgramLookup example;
    std::vector<TokenId> example_text = leading;
    example_text.push_back(900);
    example_text.push_back(901);
    example_text.insert(example_text.end(), passage.begin(), passage.begin() + 8);
    example_text.push_back(6001);
    example_text.insert(example_text.end(), 64, 7000);
    example.append(std::span<const TokenId>(example_text));
    example.append(std::span<const TokenId>(std::array<TokenId, 2>{900, 901}));
    example.append(std::span<const TokenId>(passage.data(), 8));
    require(example.predict(std::span<TokenId>(prediction)) == 0,
            "an example in the prompt is not a copy in progress");

    // The same run followed by a different token the second time is a template (a JSON object with
    // one varied field), not a copy: it is retired as soon as the divergence lands on the run, even
    // when the request reaches it with a copy's worth of corroboration behind it.
    NgramLookup templated;
    std::vector<TokenId> template_text = leading;
    template_text.insert(template_text.end(), passage.begin(), passage.begin() + 8);
    template_text.push_back(6001);
    template_text.insert(template_text.end(), 64, 7000);
    template_text.insert(template_text.end(), passage.begin(), passage.begin() + 8);
    template_text.push_back(6002);
    templated.append(std::span<const TokenId>(template_text));
    templated.append(std::span<const TokenId>(std::array<TokenId, 16>{7000, 7000, 7000, 7000, 7000, 7000,
                                                                      7000, 7000, 7000, 7000, 7000, 7000,
                                                                      7000, 7000, 7000, 7000}));
    templated.append(std::span<const TokenId>(passage.data(), 8));
    require(templated.predict(std::span<TokenId>(prediction)) == 0,
            "a run whose continuation diverges on the run is not a copy");

    // The agreement window reaches past the first continuation token: a run whose continuation
    // agreed once and then diverged is a repeated phrase as well, and is retired before it can
    // draft the wrong continuation.
    NgramLookup diverging;
    std::vector<TokenId> diverging_text = leading;
    const auto run_begin                = passage.begin();
    const auto run_end                  = passage.begin() + 8;
    diverging_text.insert(diverging_text.end(), run_begin, run_end);
    diverging_text.insert(diverging_text.end(), {61, 62, 63});
    diverging_text.insert(diverging_text.end(), 64, 7000);
    diverging_text.insert(diverging_text.end(), run_begin, run_end);
    diverging_text.insert(diverging_text.end(), {61, 62, 64});
    diverging_text.insert(diverging_text.end(), 65, 7001);
    diverging.append(std::span<const TokenId>(diverging_text));
    diverging.append(std::span<const TokenId>(std::array<TokenId, 16>{7000, 7000, 7000, 7000, 7000, 7000,
                                                                      7000, 7000, 7000, 7000, 7000, 7000,
                                                                      7000, 7000, 7000, 7000}));
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
    sliced.append(std::span<const TokenId>(passage.data(), 24));
    require(sliced.predict(std::span<TokenId>(prediction)) == 7,
            "sliced appends index the same stream as one append");
    require(prediction[0] == passage[24], "sliced appends corroborate the same occurrence");

    std::puts("ngram lookup tests passed");
    return 0;
}
