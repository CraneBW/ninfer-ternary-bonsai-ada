#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
    // prompt_cache_key names the conversation lineage a request belongs to. The frontend turns it
    // into a session key, so a re-entering turn continues the lineage's own checkpoint instead of
    // forking a copy beside it.
    std::optional<std::string> session_key;
};

// Returns the marker kind a prompt_cache_breakpoint declares, or nullopt when the field is absent.
// A declared boundary is a write candidate of its own: scope 'shared' (the default) publishes a
// prefix other sessions can reuse, scope 'private' records a checkpoint on this request's lineage.
[[nodiscard]] std::optional<ninfer::PromptCacheMarkerKind> parse_openai_prompt_cache_breakpoint(
    const RequestJson& value, std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
