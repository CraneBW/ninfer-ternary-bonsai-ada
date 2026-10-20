#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include <cstdint>
#include <array>
#include <ninfer/types.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail {

// The geometry of a registered text model, independent of its storage format. Keep it as a
// value so the binder can validate an artifact's declared tensors before consuming any of them.
// Runtime/Program still uses TextConfig's compile-time constants; supporting another shape
// requires a separately instantiated Program (or a future runtime-shape dispatch).
struct ModelShape {
    int hidden;
    int layers;
    int intermediate;
    int output_rows;
    int token_domain;
    int gdn_conv_kernel;
    int gdn_key_heads;
    int gdn_key_head_dim;
    int gdn_value_heads;
    int gdn_value_head_dim;
    int query_heads;
    int kv_heads;
    int head_dim;
    int rotary_dim;
    int full_attention_interval;
    int mtp_layers;
    int draft_head_rows;

    [[nodiscard]] constexpr int key_dim() const { return gdn_key_heads * gdn_key_head_dim; }
    [[nodiscard]] constexpr int value_dim() const { return gdn_value_heads * gdn_value_head_dim; }
    [[nodiscard]] constexpr int convolution_dim() const { return 2 * key_dim() + value_dim(); }
    [[nodiscard]] constexpr int query_size() const { return query_heads * head_dim; }
    [[nodiscard]] constexpr int kv_size() const { return kv_heads * head_dim; }
    [[nodiscard]] constexpr int attention_input_rows() const {
        return 2 * (query_size() + kv_size());
    }
    [[nodiscard]] constexpr int gdn_input_rows() const { return 2 * (key_dim() + value_dim()); }
    [[nodiscard]] constexpr bool is_full_attention(int layer) const {
        return (layer + 1) % full_attention_interval == 0;
    }
    [[nodiscard]] constexpr int full_attention_layers() const {
        return layers / full_attention_interval;
    }
    [[nodiscard]] constexpr int gdn_layers() const { return layers - full_attention_layers(); }
};

// The output matrix is padded for the selected kernels. Only token IDs in
// [0, token_domain) are tokenizer-addressable and valid sampling results.
inline constexpr ModelShape kUnshardedModelShape{
    .hidden                  = 5120,
    .layers                  = 64,
    .intermediate            = 17408,
    .output_rows             = 248320,
    .token_domain            = static_cast<int>(qwen3_6::kTokenDomain),
    .gdn_conv_kernel         = 4,
    .gdn_key_heads           = 16,
    .gdn_key_head_dim        = 128,
    .gdn_value_heads         = 48,
    .gdn_value_head_dim      = 128,
    .query_heads             = 24,
    .kv_heads                = 4,
    .head_dim                = 256,
    .rotary_dim              = 64,
    .full_attention_interval = qwen3_6::kHybridAttentionInterval,
    .mtp_layers              = 1,
    .draft_head_rows         = 131072,
};

inline constexpr ModelShape kModelShape = [] {
    auto shape = kUnshardedModelShape;
    shape.intermediate /= 2;
    shape.gdn_key_heads /= 2;
    shape.gdn_value_heads /= 2;
    shape.query_heads /= 2;
    shape.kv_heads /= 2;
    return shape;
}();

struct TextConfig {
    static constexpr int hidden       = kModelShape.hidden;
    static constexpr int layers       = kModelShape.layers;
    static constexpr int intermediate = kModelShape.intermediate;
    static constexpr int output_rows  = kModelShape.output_rows;
    static constexpr int token_domain = kModelShape.token_domain;

    static constexpr int gdn_conv_kernel      = kModelShape.gdn_conv_kernel;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = kModelShape.gdn_key_heads;
    static constexpr int gdn_key_head_dim     = kModelShape.gdn_key_head_dim;
    static constexpr int gdn_value_heads      = kModelShape.gdn_value_heads;
    static constexpr int gdn_value_head_dim   = kModelShape.gdn_value_head_dim;
    static constexpr int query_heads          = kModelShape.query_heads;
    static constexpr int kv_heads             = kModelShape.kv_heads;
    static constexpr int head_dim             = kModelShape.head_dim;
    static constexpr int rotary_dim           = kModelShape.rotary_dim;
    static constexpr int full_attention_interval = kModelShape.full_attention_interval;
    static constexpr float rms_epsilon        = 1.0e-6F;
    static constexpr float rope_theta         = 1.0e7F;

    static constexpr int key_dim               = kModelShape.key_dim();
    static constexpr int value_dim             = kModelShape.value_dim();
    static constexpr int convolution_dim       = kModelShape.convolution_dim();
    static constexpr int query_size            = kModelShape.query_size();
    static constexpr int kv_size               = kModelShape.kv_size();
    static constexpr int query_projection_rows = 2 * query_size;
    static constexpr int attention_input_rows  = kModelShape.attention_input_rows();
    static constexpr int gdn_input_rows        = kModelShape.gdn_input_rows();
    static constexpr int gdn_control_rows      = 2 * gdn_value_heads;
    static constexpr int draft_head_rows       = kModelShape.draft_head_rows;

    static constexpr int mtp_layers               = kModelShape.mtp_layers;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = attention_input_rows;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return kModelShape.is_full_attention(layer);
    }
    [[nodiscard]] static constexpr int full_attention_layers() {
        return kModelShape.full_attention_layers();
    }
    [[nodiscard]] static constexpr int gdn_layers() { return kModelShape.gdn_layers(); }
    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return qwen3_6::full_attention_index(layer);
    }
    [[nodiscard]] static constexpr int gdn_index(int layer) { return qwen3_6::gdn_index(layer); }
};

static_assert(TextConfig::full_attention_interval == qwen3_6::kHybridAttentionInterval);
static_assert(TextConfig::full_attention_layers() == 16);
static_assert(TextConfig::gdn_layers() == 48);

struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

struct DFlashConfig {
    static constexpr bool supported             = true;
    static constexpr SpeculativeBackend backend = SpeculativeBackend::DFlash2;
    static constexpr bool coherent_selector     = true;
    static constexpr int layers                 = 5;
    static constexpr int local_layers           = 5;
    static constexpr int full_layers            = 0;
    static constexpr int local_capacity         = 2048;
    static constexpr int feature_layers         = 5;
    static constexpr int feature_rows           = feature_layers * TextConfig::hidden;
    static constexpr int hidden                 = TextConfig::hidden;
    static constexpr int intermediate           = TextConfig::intermediate;
    static constexpr int query_heads            = 32;
    static constexpr int kv_heads               = 8;
    static constexpr int head_dim               = 128;
    static constexpr int query_size             = 4096;
    static constexpr int kv_size                = 1024;
    static constexpr int query_key_value_rows   = query_size + 2 * kv_size;
    static constexpr int dynamic_conv_rows      = 1280;
    static constexpr int selector_dim           = 256;
    static constexpr int mlp_gate_up_rows       = 2 * intermediate;
    static constexpr int mask_token             = 248070;
    static constexpr float rms_epsilon          = 1.0e-6F;
    static constexpr float rope_theta           = 1.0e7F;
    static constexpr float attention_scale      = 0.08838834764831845F;
    static constexpr std::array<int, feature_layers> target_feature_layers{5, 19, 33, 47, 61};
};

inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7; // see kMtpDecodeMaximumDrafts
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail
