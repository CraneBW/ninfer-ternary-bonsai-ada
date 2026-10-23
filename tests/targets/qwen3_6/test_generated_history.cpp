// Official-tokenizer regression and optional real-Engine frozen-history continuation probe.
// The GPU fixture prefills saved Pi output; it does not re-run a creative quality campaign.
#include "ninfer/engine.h"
#include "runtime/engine/test_access.h"
#include "targets/qwen3_6/impl/frontend/generated_history_cache.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace q = ninfer::targets::qwen3_6;
namespace fi = q::frontend_internal;
using Json = nlohmann::ordered_json;

static void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
static std::string read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "cannot read fixture");
    return {std::istreambuf_iterator<char>(f), {}};
}
static q::FrontendResources resources(const std::string& dir) {
    q::FrontendResources r;
    r.tokenizer_json = read(dir + "/tokenizer.json");
    r.tokenizer_config_json = read(dir + "/tokenizer_config.json");
    r.generation_config_json = read(dir + "/generation_config.json");
    r.chat_template_jinja = read(dir + "/chat_template.jinja");
    r.preprocessor_config_json = read(dir + "/preprocessor_config.json");
    r.video_preprocessor_config_json = read(dir + "/video_preprocessor_config.json");
    return r;
}
static ninfer::PromptInput input(const Json& fixture) {
    ninfer::PromptInput p;
    p.options.enable_thinking = true;
    p.options.preserve_thinking = true;
    p.options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    for (const auto& tool : fixture.at("tools")) { p.options.tool_jsons.push_back(tool.dump()); }
    for (const auto& m : fixture.at("messages")) {
        ninfer::ChatMessage message;
        const auto role = m.at("role").get<std::string>();
        message.role = role == "system" ? ninfer::ChatRole::System :
                       role == "assistant" ? ninfer::ChatRole::Assistant :
                       role == "tool" ? ninfer::ChatRole::Tool : ninfer::ChatRole::User;
        if (m.contains("content") && !m.at("content").is_null()) {
            std::string text;
            if (m.at("content").is_string()) { text = m.at("content"); }
            else {
                for (const auto& part : m.at("content")) {
                    require(part.at("type") == "text", "media is outside this fixture");
                    text += part.at("text").get<std::string>();
                }
            }
            message.parts.push_back({.kind = ninfer::MessagePartKind::Text,
                                     .text = std::move(text)});
        }
        message.reasoning_content = m.value("reasoning_content", "");
        for (const auto& call : m.value("tool_calls", Json::array())) {
            message.tool_calls.push_back({.id = call.at("id"),
                .name = call.at("function").at("name"),
                .arguments_json = call.at("function").at("arguments").get<std::string>()});
        }
        message.tool_call_id = m.value("tool_call_id", "");
        p.messages.push_back(std::move(message));
    }
    return p;
}
static std::vector<int> noncanonical(const fi::Tokenizer& t, std::span<const int> ids,
                                     const std::string& needle, const std::string& left,
                                     std::optional<std::size_t> observed_frontier = std::nullopt) {
    const std::string bytes = t.decode(ids);
    const auto content_begin = bytes.find("<parameter=content>");
    auto found = bytes.find(needle, content_begin == std::string::npos ? 0 : content_begin);
    if (observed_frontier) {
        require(*observed_frontier < ids.size(), "observed mismatch is outside the frozen history");
        found = 0;
        for (std::size_t i = 0; i < *observed_frontier; ++i) { found += t.decoded_token(ids[i]).bytes.size(); }
        require(std::string_view(bytes).substr(found).starts_with(needle),
                "frozen history no longer matches the observed mismatch position");
    }
    require(found != std::string::npos, "frozen output lacks the observed Chinese text");
    std::size_t begin = 0, offset = 0;
    while (begin < ids.size() && offset < found) { offset += t.decoded_token(ids[begin++]).bytes.size(); }
    require(offset == found, "observed divergence is not at a token boundary");
    std::size_t end = begin;
    while (end < ids.size() && offset < found + needle.size()) {
        offset += t.decoded_token(ids[end++]).bytes.size();
    }
    const std::string covered = bytes.substr(found, offset - found);
    auto alternative = t.encode(left);
    const auto tail = t.encode(covered.substr(left.size()));
    alternative.insert(alternative.end(), tail.begin(), tail.end());
    require(!std::equal(alternative.begin(), alternative.end(), ids.begin() + begin, ids.begin() + end),
            "fixture did not change BPE segmentation");
    std::vector<int> result(ids.begin(), ids.begin() + begin);
    result.insert(result.end(), alternative.begin(), alternative.end());
    result.insert(result.end(), ids.begin() + end, ids.end());
    require(t.decode(result) == bytes, "fixture changed text bytes");
    return result;
}
static void unit(const fi::Tokenizer& t) {
    fi::GeneratedHistoryCache cache;
    const std::string text = "<|im_start|>assistant\n<think>\nOK\n</think>\n\n下雪<|im_end|>";
    auto canonical = t.encode(text);
    auto original = noncanonical(t, canonical, "下雪", "下");
    cache.remember(t, original);
    fi::EncodedChat base;
    base.input_ids = t.encode(text + "\n<|im_start|>user\nnext<|im_end|>\n");
    base.rewrite_checkpoint = q::RewriteCheckpointSpec{q::RewriteCheckpointKind::ResponseReplay,
                                                      static_cast<std::uint32_t>(canonical.size())};
    base.rewrite_execution_frontiers = {static_cast<std::uint32_t>(canonical.size())};
    base.message_boundaries = {0, static_cast<std::uint32_t>(canonical.size())};
    base.cache_boundaries = {static_cast<std::uint32_t>(base.input_ids.size())};
    auto chat = base;
    const auto hit = cache.apply(t, chat);
    require(hit.has_value() && std::equal(original.begin(), original.end(), chat.input_ids.begin()),
            "same bytes did not recover original ids");
    require(chat.rewrite_checkpoint->frontier == original.size() &&
            chat.rewrite_execution_frontiers[0] == original.size() &&
            chat.message_boundaries[1] == original.size() &&
            chat.cache_boundaries[0] == chat.input_ids.size(), "frontier remap failed");
    auto changed = base;
    changed.input_ids[1] = t.encode("user").front();
    require(!cache.apply(t, changed), "changed history was accepted");
    auto literal = base;
    const auto marker = std::find(literal.input_ids.begin(), literal.input_ids.end(), 248045);
    require(marker != literal.input_ids.end(), "official structural token missing");
    const auto index = marker - literal.input_ids.begin();
    const auto ordinary = t.encode("<|im_start|>", {.parse_added_tokens = false});
    literal.input_ids.erase(literal.input_ids.begin() + index);
    literal.input_ids.insert(literal.input_ids.begin() + index, ordinary.begin(), ordinary.end());
    require(t.decode(literal.input_ids) == t.decode(base.input_ids), "literal guard fixture changed bytes");
    require(!cache.apply(t, literal), "literal text was confused with a structural marker");
    fi::GeneratedHistoryCache partial;
    original.pop_back();
    partial.remember(t, original);
    chat = base;
    require(!partial.apply(t, chat), "unfinished generation became a completed entry");
    std::cout << "CPU byte equality, structural identity, remap, edited history, terminal guards PASS\n";
}

int main(int argc, char** argv) try {
    const char* dir = std::getenv("NINFER_OFFICIAL_TOKENIZER_DIR");
    if (!dir) { std::cout << "Set NINFER_OFFICIAL_TOKENIZER_DIR\n"; return 77; }
    const auto r = resources(dir);
    const fi::Tokenizer tokenizer({r.tokenizer_json, r.tokenizer_config_json, r.generation_config_json});
    if (argc == 1) { unit(tokenizer); return 0; }
    require(argc == 6, "usage: test model fixture backend mode receipt");
    const std::string backend = argv[3], mode = argv[4];
    require(mode == "canonical" || mode == "candidate" || mode == "original", "unknown mode");
    const Json fixture = Json::parse(read(argv[2]));
    auto p = input(fixture);
    auto frontend = q::FrontendTestAccess::create_component(r, false);
    auto prepared = frontend.prepare(p);
    const auto& canonical = q::FrontendTestAccess::inspect(prepared);
    auto original = noncanonical(tokenizer, canonical.token_ids, fixture.at("needle"), fixture.at("left"),
                                  fixture.at("mismatch_index").get<std::size_t>());
    // Last completed assistant, before its tool result and the next generation opener.
    std::size_t end = 0;
    // tool markup is ordinary vocabulary, so locate the following im_end by decoded bytes.
    const auto full_bytes = tokenizer.decode(original);
    const auto call_end = full_bytes.rfind("</tool_call>");
    require(call_end != std::string::npos, "fixture has no tool call");
    std::size_t offset = 0;
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (offset > call_end && tokenizer.decoded_token(original[i]).bytes == "<|im_end|>") {
            end = i; break;
        }
        offset += tokenizer.decoded_token(original[i]).bytes.size();
    }
    require(end > 0 && end < original.size(), "no completed assistant boundary");
    // Restore the exact chat/GDN provenance along with the alternative BPE segmentation.
    // Importing bare ids would correctly fail Program identity even after a perfect token hit.
    fi::EncodedChat encoded;
    encoded.input_ids = canonical.token_ids;
    encoded.rewrite_checkpoint = canonical.identity.rewrite_checkpoint;
    encoded.rewrite_execution_frontiers = canonical.identity.rewrite_execution_frontiers;
    fi::GeneratedHistoryCache fixture_cache;
    fixture_cache.remember(tokenizer, {original.begin(), original.begin() + end + 1});
    require(fixture_cache.apply(tokenizer, encoded).has_value(), "cannot map fixture provenance");
    const auto prepare_original = [&](std::size_t count, bool starts_in_reasoning) {
        auto data = q::PreparedPromptAccess::take(frontend.prepare(p));
        data.token_ids.assign(original.begin(), original.begin() + count);
        data.token_types.assign(count, 0);
        data.positions.resize(3 * count);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < count; ++i) { data.positions[axis * count + i] = i; }
        }
        data.identity.rewrite_execution_frontiers = encoded.rewrite_execution_frontiers;
        std::erase_if(data.identity.rewrite_execution_frontiers,
                      [&](auto frontier) { return frontier > count; });
        data.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
        if (data.identity.rewrite_checkpoint && data.identity.rewrite_checkpoint->frontier > count) {
            data.identity.rewrite_checkpoint.reset();
        }
        data.context_cache.opportunities.clear();
        data.query_begin = 0;
        data.query_end = count;
        data.action_begin = 0;
        data.action_end = count;
        data.closed_thinking_spans.clear();
        data.lexical_query_token_ids.clear();
        data.starts_in_reasoning = starts_in_reasoning;
        return ninfer::PreparedPromptTestAccess::adopt(q::PreparedPromptAccess::restore(std::move(data)),
            starts_in_reasoning ? ninfer::SamplingMode::Thinking : ninfer::SamplingMode::NonThinking);
    };
    ninfer::EngineOptions options;
    options.artifact_path = argv[1];
    options.max_context = 262144;
    options.kv_capacity = ninfer::KvCapacityPolicy::device_budget(36864);
    options.kv_storage = ninfer::KvStoragePolicy::explicit_storage(ninfer::KvCacheStorage::Int8Group64);
    options.prefill_chunk = 1024;
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    options.kvmem = {.enabled = true, .budget_tokens = 32768, .gen_reserve_tokens = 512};
    options.context_cache.host_kv_capacity_bytes = 128ULL << 20;
    if (backend == "mtp4") {
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else { require(backend == "ar", "unknown backend"); }
    ninfer::Engine engine(options);
    ninfer::RequestOptions request;
    request.execution.sampling.temperature = 0;
    request.execution.sampling.presence_penalty = 0;
    request.execution.sampling.seed = 42;
    request.execution.requested_output_tokens = 1;
    const auto prime = engine.generate(prepare_original(end, false), request);
    require(prime.finish_reason == ninfer::FinishReason::StopToken &&
            prime.generated_token_ids == std::vector<int>{248046}, "frozen tool prefix did not terminate naturally");
    request.execution.requested_output_tokens = 64;
    const ninfer::GenerationObservationOptions observation{.phase_timings = true};
    const auto result = mode == "original"
        ? engine.submit(prepare_original(original.size(), true), request,
                        ninfer::OutputConsumerMode::Aggregate, observation).wait()
        : engine.submit(engine.prepare(p), request,
                        ninfer::OutputConsumerMode::Aggregate, observation).wait();
    Json receipt{{"mode", mode}, {"backend", backend}, {"source_tokens", end},
        {"prompt_tokens", result.prompt.prompt_tokens}, {"reused_tokens", result.reused_prompt_tokens},
        {"ttft_s", result.timings.first_token_seconds}, {"prefill_s", result.timings.prompt_wall_seconds},
        {"generated_ids", result.generated_token_ids}, {"content", result.content},
        {"reasoning", result.reasoning}, {"scope", "frozen Pi history with observed BPE alternative; real GPU prefill/continuation"}};
    std::ofstream(argv[5]) << receipt.dump(2) << '\n';
    if (mode == "candidate" || mode == "original") {
        require(result.reused_prompt_tokens >= end, "completed tool history was rebuilt");
    } else { require(result.reused_prompt_tokens < end, "baseline unexpectedly reused the noncanonical endpoint"); }
    if (mode == "candidate") {
        p.messages.front().parts.front().text += " Changed history must invalidate the old prefix.";
        request.execution.requested_output_tokens = 1;
        const auto changed = engine.generate(engine.prepare(p), request);
        require(changed.reused_prompt_tokens == 0, "edited history reused old GPU state");
        receipt["changed_history_reused"] = changed.reused_prompt_tokens;
    }
    receipt["state"] = "PASS";
    std::ofstream(argv[5]) << receipt.dump(2) << '\n';
    std::cout << receipt.dump() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
}
