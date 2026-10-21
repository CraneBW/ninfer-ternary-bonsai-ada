#include "ninfer/engine.h"
#include "../qwen3_6/speculative_page_boundary.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
ninfer::RequestOptions request(std::uint32_t count = 24, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = count;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = reuse;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::PromptInput chat_input(const nlohmann::json& input, bool automatic_shared) {
    ninfer::PromptInput prompt;
    prompt.options.enable_thinking = false;
    // Match serve/openai_common.cpp: these replay requests carry no explicit
    // shared-prefix marker and disable Engine-inferred shared captures.
    prompt.context_cache.allow_engine_automatic_shared_prefixes = automatic_shared;
    for (const auto& source : input.at("messages")) {
        ninfer::ChatMessage message;
        const auto role = source.at("role").get<std::string>();
        if (role == "system") { message.role = ninfer::ChatRole::System; }
        else if (role == "user") { message.role = ninfer::ChatRole::User; }
        else if (role == "assistant") { message.role = ninfer::ChatRole::Assistant; }
        else if (role == "tool") { message.role = ninfer::ChatRole::Tool; }
        else { throw std::invalid_argument("unsupported recall fixture role"); }
        if (source.contains("content") && source.at("content").is_string()) {
            message.parts.push_back({.kind = ninfer::MessagePartKind::Text,
                                     .text = source.at("content").get<std::string>()});
        }
        message.tool_call_id = source.value("tool_call_id", std::string{});
        for (const auto& call : source.value("tool_calls", nlohmann::json::array())) {
            message.tool_calls.push_back({.id = call.at("id").get<std::string>(),
                .name = call.at("function").at("name").get<std::string>(),
                .arguments_json = call.at("function").at("arguments").get<std::string>()});
        }
        prompt.messages.push_back(std::move(message));
    }
    return prompt;
}

void recall_requests(ninfer::Engine& engine, const std::filesystem::path& fixtures,
                     nlohmann::json& evidence,
                     std::vector<std::vector<ninfer::TokenId>>& observed,
                     bool shared_boundary, bool semantic_only) {
    for (const auto& [name, expected] :
         {std::pair{"page_boundary", "violet-cinder-58"},
          std::pair{"tool_result", "teal-jasmine-61"},
          std::pair{"paraphrase", "cobalt-orchid-73"}}) {
        if ((std::string_view(name) == "paraphrase") != semantic_only) { continue; }
        if (shared_boundary && std::string_view(name) != "page_boundary") { continue; }
        for (int turn = 1; turn <= 2; ++turn) {
            std::ifstream source(fixtures / (std::string(name) + "-turn" +
                                             std::to_string(turn) + ".json"));
            require(source.good(), "recall fixture is missing");
            const auto input = nlohmann::json::parse(source);
            auto options = request(input.at("max_tokens").get<std::uint32_t>());
            options.stop.include_model_defaults = true;
            std::cout << "recall START " << name << " turn=" << turn << std::endl;
            const auto result =
                engine.generate(engine.prepare(chat_input(input, shared_boundary)), options);
            const auto usage = engine.runtime_stats().kvmem;
            observed.push_back(result.generated_token_ids);
            evidence["requests"].push_back({{"case", name}, {"turn", turn},
                {"ids", result.generated_token_ids}, {"content", result.content},
                {"reused", result.reused_prompt_tokens},
                {"finish", static_cast<int>(result.finish_reason)},
                {"drafted", result.speculative.drafted_tokens},
                {"accepted", result.speculative.accepted_tokens},
                {"rounds", result.speculative.rounds},
                {"answer_correct", result.content == (turn == 1 ? "ACK" : expected)},
                {"committed_decode", engine.runtime_stats().committed_decode_tokens},
                {"inflight", usage.inflight_pages}});
            std::cout << "recall " << name << " turn=" << turn << " content=" << result.content
                      << " accepted=" << result.speculative.accepted_tokens << "/"
                      << result.speculative.drafted_tokens << std::endl;
            if (!shared_boundary || turn == 1) {
                require(result.content == (turn == 1 ? "ACK" : expected),
                        "recall answer is incorrect");
            } else {
                // This configuration regression checks the previously failing
                // capture capacity and transfer parity. Shared/fixed geometry
                // has no forced lexical-recall guarantee; report quality too.
                require(!result.content.empty(), "shared capture produced no answer");
            }
            if (turn == 2 && (std::string_view(name) == "page_boundary" || semantic_only)) {
                require(result.reused_prompt_tokens != 0,
                        "cross-page fixture bypassed the reusable history bridge");
            }
            require(usage.inflight_pages == 0, "recall left an in-flight page at completion");
        }
    }
}

void branch_requests(ninfer::Engine& engine, const std::filesystem::path& fixtures,
                     nlohmann::json& evidence,
                     std::vector<std::vector<ninfer::TokenId>>& observed) {
    for (int step = 0; step < 3; ++step) {
        std::ifstream file(fixtures / (std::string("page_boundary-turn") +
                                      (step == 0 ? "1.json" : "2.json")));
        require(file.good(), "branch fixture is missing");
        const auto input = nlohmann::json::parse(file);
        auto prompt = chat_input(input, false);
        prompt.context_cache.session_key = step == 1 ? "branch-B" : "branch-A";
        auto options = request(input.at("max_tokens").get<std::uint32_t>());
        options.stop.include_model_defaults = true;
        const auto roots = engine.runtime_stats().root_selections;
        const auto result = engine.generate(engine.prepare(std::move(prompt)), options);
        const auto stats = engine.runtime_stats();
        observed.push_back(result.generated_token_ids);
        evidence["requests"].push_back({{"case", "retained_branch"}, {"turn", step + 1},
            {"content", result.content}, {"ids", result.generated_token_ids},
            {"reused", result.reused_prompt_tokens}, {"finish", static_cast<int>(result.finish_reason)},
            {"committed_decode", stats.committed_decode_tokens},
            {"drafted", result.speculative.drafted_tokens},
            {"root_selections", stats.root_selections}, {"inflight", stats.kvmem.inflight_pages}});
        require(result.content == (step == 0 ? "ACK" : "violet-cinder-58"),
                "retained compressed branch lost its historical fact");
        if (step == 1) {
            require(result.reused_prompt_tokens == 0 && stats.root_selections == roots + 1,
                    "incomplete compressed Fork was not rebuilt from root");
        }
        if (step == 2) {
            require(result.reused_prompt_tokens != 0,
                    "root branch fallback destroyed the retained original session");
        }
        std::cout << "branch step=" << step << " content=" << result.content
                  << " reused=" << result.reused_prompt_tokens << std::endl;
    }
}
}

// Opt-in, bounded Qwen3.8 comparison. Reuse the existing speculative boundary
// fixture for stop/partial-verify/next-request settlement rather than duplicating it.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_KVMEM_WEIGHTS");
    if (!artifact || !*artifact) { std::cout << "skip: no Qwen3.8 artifact\n"; return 77; }
    const std::string route = argc > 1 ? argv[1] : "base";
    const bool recall = argc > 2;
    const bool shared_boundary = argc > 3 && std::string_view(argv[3]) == "shared-boundary";
    const bool semantic_only = argc > 3 && std::string_view(argv[3]) == "semantic";
    const bool branch_only = argc > 3 && std::string_view(argv[3]) == "branch";
    nlohmann::json evidence;
    try {
        require(route == "base" || route == "graph" || route == "mtp" || route == "dflash2",
                "unknown async integration route");
        std::vector<std::vector<ninfer::TokenId>> reference;
        std::vector<ninfer::TokenId> frozen_prompt;
        for (const bool async : {false, true}) {
            ninfer::EngineOptions options;
            options.artifact_path = artifact;
            options.managed_weights = true;
            options.max_context = 3072;
            options.kv_capacity = ninfer::KvCapacityPolicy::device_budget(1024);
            options.prefill_chunk = 256;
            options.use_cuda_graph = route != "base";
            options.kv_storage = ninfer::KvStoragePolicy::explicit_storage(
                ninfer::KvCacheStorage::Int8Group64);
            options.kvmem.enabled = true;
            options.kvmem.budget_tokens = 512;
            options.kvmem.gen_reserve_tokens = 128;
            options.kvmem.action_lrfu = true;
            if (recall) { options.kvmem.lrfu_top_fraction = 0.3; }
            options.kvmem.async_transfers = async;
            options.context_cache.host_kv_capacity_bytes = 128ULL << 20;
            if (route == "mtp" || route == "dflash2") {
                options.speculative.backend = route == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                                               : ninfer::SpeculativeBackend::DFlash2;
                options.speculative.draft_tokens = 3;
                options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            }
            if (branch_only) {
                require(route == "base", "branch fixture uses the base route");
                options.managed_runtime_capacity_bytes = 2ULL << 30;
                options.context_cache.max_private_continuations = 4;
                options.context_cache.device_state_slots = 2;
                options.context_cache.host_state_slots = 2;
            }
            ninfer::Engine engine(options);
            if (!recall && frozen_prompt.empty()) {
                std::string text;
                for (int i = 0; i < 110; ++i) {
                    text += "Archive note " + std::to_string(i) + ": the record is unchanged.\n";
                }
                text += "Count from one to twenty: one, two, three,";
                frozen_prompt = engine.tokenize_text(text);
                require(frozen_prompt.size() > 1024 && frozen_prompt.size() < 2800,
                        "frozen fixture must cross the physical pool");
            }
            auto& arm = evidence[async ? "async" : "sync"];
            std::vector<std::vector<ninfer::TokenId>> observed;
            if (recall) {
                if (branch_only) { branch_requests(engine, argv[2], arm, observed); }
                else { recall_requests(engine, argv[2], arm, observed, shared_boundary, semantic_only); }
                if (route == "mtp" || route == "dflash2") {
                    for (const auto& row : arm.at("requests")) {
                        require(row.at("drafted").get<std::uint64_t>() != 0,
                                "recall did not exercise the speculative backend");
                    }
                }
            }
            for (int turn = 0; !recall && turn < 3; ++turn) {
                auto prompt = frozen_prompt;
                if (turn == 1) {
                    prompt.insert(prompt.end(), observed[0].begin(), observed[0].begin() + 12);
                    auto suffix = engine.tokenize_text("\nContinue the count:");
                    prompt.insert(prompt.end(), suffix.begin(), suffix.end());
                }
                const auto result = engine.generate(engine.prepare_tokens(prompt), request());
                require(result.generated_token_ids.size() == 24 &&
                            result.finish_reason == ninfer::FinishReason::OutputLimit,
                        "overpool request failed to settle its exact output budget");
                if (route == "mtp" || route == "dflash2") {
                    require(result.speculative.rounds != 0 && result.speculative.drafted_tokens != 0,
                            "speculative route was bypassed");
                }
                observed.push_back(result.generated_token_ids);
                arm["requests"].push_back({{"ids", result.generated_token_ids},
                    {"reused", result.reused_prompt_tokens},
                    {"drafted", result.speculative.drafted_tokens},
                    {"accepted", result.speculative.accepted_tokens},
                    {"rounds", result.speculative.rounds},
                    {"committed_decode", engine.runtime_stats().committed_decode_tokens}});
                std::cout << route << " " << (async ? "async" : "sync") << " turn=" << turn
                          << " tokens=" << result.generated_token_ids.size()
                          << " accepted=" << result.speculative.accepted_tokens << "/"
                          << result.speculative.drafted_tokens << std::endl;
            }
            if (!recall && (route == "mtp" || route == "dflash2")) {
                ninfer::test::speculative_page_boundary(engine);
                arm["page_boundary"] = "PASS";
            }
            // Destruction of an unconsumed queued handle is the public cancel API.
            if (!recall) {
                auto active = engine.submit(engine.prepare_tokens(frozen_prompt), request(8));
                { auto queued = engine.submit(engine.prepare_tokens(frozen_prompt), request(8)); }
                require(active.wait().generated_token_ids.size() == 8, "queued cancellation damaged active request");
                const auto following = engine.generate(engine.prepare_tokens(frozen_prompt), request(8));
                require(following.generated_token_ids.size() == 8, "request after cancellation failed");
            }
            const auto usage = engine.runtime_stats().kvmem;
            arm["submitted"] = usage.async_submitted;
            arm["completed"] = usage.async_completed;
            arm["discarded"] = usage.async_discarded;
            arm["inflight"] = usage.inflight_pages;
            arm["compactions"] = usage.compactions;
            arm["restores"] = usage.async_restore_submitted;
            arm["history_replay_tokens"] = usage.history_replay_tokens;
            arm["query_replay_tokens"] = usage.query_replay_tokens;
            arm["history_replay_rounds"] = usage.history_replay_rounds;
            arm["source_k_bytes"] = usage.source_k_bytes;
            if (recall && !shared_boundary) {
                require(usage.history_replay_tokens != 0,
                        "recall fixture bypassed canonical history reconstruction");
            }
            require(usage.compactions > 1 && (!async || usage.async_submitted != 0),
                    "integration fixture did not exercise page compaction/transfer");
            require(usage.inflight_pages == 0 &&
                        usage.async_completed + usage.async_discarded == usage.async_submitted,
                    "terminal request left an unaccounted transfer batch");
            if (!async) { reference = observed; }
            else { require(observed == reference, "sync/async generated token IDs differ"); }
            if (recall && async) {
                for (std::size_t i = 0; i < arm.at("requests").size(); ++i) {
                    for (const char* key : {"reused", "committed_decode", "finish", "content"}) {
                        require(arm.at("requests")[i].at(key) ==
                                    evidence.at("sync").at("requests")[i].at(key),
                                "sync/async recall commit or reuse differs");
                    }
                }
            }
        }
        std::cout << evidence.dump(2) << "\nPASS " << route << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cout << evidence.dump(2) << '\n';
        std::cerr << "FAIL " << route << ": " << error.what() << std::endl;
        return 1;
    }
}
