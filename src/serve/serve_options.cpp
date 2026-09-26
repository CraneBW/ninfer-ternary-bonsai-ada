#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    // Packed int4 (port sergiuszm/ninfer-4090); solo el build sm_89 los soporta — el planner
    // rechaza el arranque en otra arquitectura (ver validate_target_options).
    if (value == "rk4v4") { return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (value == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] "
           "[--kv-device-tokens N] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--log-stats-interval-ms N] [--device N] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] "
           "[--max-long-anchors-per-continuation N] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|rk4v4|rk4v4-e8 | "
           "--cache-type-k <bf16|int8|fp8|nvfp4|int4|int4-e8> "
           "--cache-type-v <bf16|int8|fp8|nvfp4|int4>] "
           "[--kvmem [--kvmem-budget N] [--kvmem-gen-reserve N]] "
           "[--spec mtp|dflash|dflash2 --draft-tokens N] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--vision] [--vision-max-tokens N] [--no-cuda-graph] [--no-prefix-reuse] "
           "[--lm-head-draft] [--no-thinking] [--preserve-thinking] "
           "[--tolerant-tool-calls] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy] [--wddm-evictable-budget]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --model-id overrides the artifact identity.model_id reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --vision-max-tokens sets the Vision scratchpad token capacity (default 8192)\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom (" +
           std::to_string(kKvMemAutomaticHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB under --kvmem, whose pool is sized by the working set it needs)\n"
           "       --kv-device-tokens caps the KV page pool below --max-context; the rest of the "
           "context is backed by host memory, so it needs the host KV tier\n"
           "       --cache-type-k/--cache-type-v choose each side of the KV cache; an omitted "
           "side stays at bf16, and the pair must have a kernel (see the server error for the "
           "supported list)\n"
           "       --kvmem keeps the device KV pool a resident working set and backs the rest of "
           "the context with host memory; --kvmem-budget narrows the working set in tokens "
           "(0 or omitted keeps the whole context). Without --kv-capacity or --kv-device-tokens "
           "the pool is sized automatically from the working set it needs\n"
           "       --kvmem-gen-reserve keeps that many tokens of the pool free for decoding "
           "(default " +
           std::to_string(kDefaultKvMemGenReserveTokens) +
           "; 0 disables generation-time reclaim). Too small a reserve evicts freshly retrieved "
           "content mid-generation; a pool that cannot afford it clamps it down\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       context cache defaults: device-state=max-concurrency, private=2x concurrency, "
           "shared=max(max-concurrency,4), anchors=2; Host state=8 slots, Host KV=8192 MiB\n"
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       --tolerant-tool-calls recovers complete Qwen calls with malformed "
           "wrapper/suffix output\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n"
           "       --wddm-evictable-budget budgets runtime memory against total VRAM on dedicated "
           "GPUs, ignoring the WDDM process budget (Windows only; the GPU must not drive a "
           "concurrent desktop GPU workload)\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool kv_device_tokens_explicit   = false;
    bool context_capacity_explicit   = false;
    bool kv_dtype_explicit           = false;
    bool kvmem_switch                = false;
    bool kvmem_budget_given          = false;
    std::uint32_t kvmem_budget       = 0;
    bool kvmem_gen_reserve_given     = false;
    std::uint32_t kvmem_gen_reserve  = 0;
    std::optional<KvKeyStorage> kv_key_storage;
    std::optional<KvValueStorage> kv_value_storage;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--kv-device-tokens") {
            const int value =
                parse_nonnegative_int(require_value("--kv-device-tokens"), "kv-device-tokens");
            if (value == 0) {
                throw std::invalid_argument("--kv-device-tokens must be positive");
            }
            options.kv_capacity        = KvCapacityPolicy::device_budget(
                static_cast<std::uint32_t>(value));
            kv_device_tokens_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--kv-dtype") {
            options.kv_storage =
                KvStoragePolicy::explicit_storage(parse_kv_dtype(require_value("--kv-dtype")));
            kv_dtype_explicit = true;
        } else if (arg == "--cache-type-k") {
            const std::string_view text = require_value("--cache-type-k");
            const auto parsed           = parse_kv_key_storage(text);
            if (!parsed) {
                throw std::invalid_argument(
                    "--cache-type-k must be bf16, int8, fp8, nvfp4, int4, or int4-e8");
            }
            kv_key_storage = *parsed;
        } else if (arg == "--cache-type-v") {
            const std::string_view text = require_value("--cache-type-v");
            const auto parsed           = parse_kv_value_storage(text);
            if (!parsed) {
                throw std::invalid_argument(
                    "--cache-type-v must be bf16, int8, fp8, nvfp4, or int4");
            }
            kv_value_storage = *parsed;
        } else if (arg == "--kvmem") {
            kvmem_switch = true;
        } else if (arg == "--kvmem-budget") {
            const int value =
                parse_nonnegative_int(require_value("--kvmem-budget"), "kvmem-budget");
            if (value == 0) { throw std::invalid_argument("--kvmem-budget must be positive"); }
            kvmem_budget       = static_cast<std::uint32_t>(value);
            kvmem_budget_given = true;
        } else if (arg == "--kvmem-gen-reserve") {
            const int value = parse_nonnegative_int(require_value("--kvmem-gen-reserve"),
                                                    "kvmem-gen-reserve");
            kvmem_gen_reserve       = static_cast<std::uint32_t>(value);
            kvmem_gen_reserve_given = true;
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-max-tokens") {
            const int val = parse_nonnegative_int(require_value("--vision-max-tokens"),
                                                  "vision-max-tokens");
            if (val <= 0) {
                throw std::invalid_argument("--vision-max-tokens must be positive");
            }
            options.vision_max_tokens = static_cast<std::uint32_t>(val);
            options.enable_vision     = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--tolerant-tool-calls") {
            options.tolerant_tool_calls = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--wddm-evictable-budget") {
            options.wddm_evictable_budget = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (kv_capacity_explicit && kv_device_tokens_explicit) {
        throw std::invalid_argument("--kv-capacity and --kv-device-tokens cannot be combined");
    }
    options.kvmem = resolve_kvmem_options(kvmem_switch, kvmem_budget_given, kvmem_budget,
                                          kvmem_gen_reserve_given, kvmem_gen_reserve);
    if ((kvmem_budget_given || kvmem_gen_reserve_given) && !options.kvmem.enabled) {
        throw std::invalid_argument(
            "--kvmem-budget and --kvmem-gen-reserve size the KVMem window, so they need --kvmem "
            "(or NINFER_KVMEM=1) to be enabled");
    }
    if (!kv_capacity_explicit && !kv_device_tokens_explicit) {
        // KVMem keeps the device pool a resident working set, so a front end that enabled it
        // without choosing a pool gets an automatic one, bounded by the working set the runtime
        // needs instead of by whatever device memory happens to be left. Without the host KV
        // tier the context has no backing, so the pool keeps covering it.
        options.kv_capacity = options.kvmem.enabled && options.allow_prefix_reuse
                                  ? KvCapacityPolicy::automatic(kKvMemAutomaticHeadroomBytes)
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (kv_dtype_explicit && (kv_key_storage || kv_value_storage)) {
        throw std::invalid_argument(
            "--kv-dtype sets both sides of the KV cache, so it cannot be combined with "
            "--cache-type-k or --cache-type-v");
    }
    if (!kv_dtype_explicit && (kv_key_storage || kv_value_storage)) {
        // An omitted side stays at bf16, so a single-sided request that has no kernel
        // fails here with the supported-pair list instead of running the wrong decode.
        const auto resolved = resolve_kv_cache_storage(
            kv_key_storage.value_or(KvKeyStorage::BFloat16),
            kv_value_storage.value_or(KvValueStorage::BFloat16));
        if (!resolved) {
            throw std::invalid_argument(
                "--cache-type-k/--cache-type-v select a pair without a kernel; supported "
                "pairs: " +
                std::string(kSupportedKvCachePairList));
        }
        options.kv_storage = KvStoragePolicy::explicit_storage(*resolved);
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    if (kv_device_tokens_explicit && !options.allow_prefix_reuse) {
        throw std::invalid_argument(
            "--kv-device-tokens needs the host KV tier to back the context, so it cannot be "
            "combined with --no-prefix-reuse");
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace ninfer::serve
