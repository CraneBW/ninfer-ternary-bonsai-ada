#pragma once

// YaRN context extension (Peng et al., "YaRN: Efficient Context Window Extension of Large
// Language Models", Eq. 19). Past the native rotary window the low-frequency RoPE divisors are
// ramped down by the scale factor F = capacity / native, and the final dot-product scale is
// multiplied by m = 0.1*ln(F) + 1, matching the reference port this file follows.

#include <array>
#include <cmath>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct YarnConfig {
    // 1 for a capacity inside the native window; the inv_freq table is then left untouched so
    // the native path stays bit-exact with the baked device table.
    float scale_factor                = 1.0F;
    float attn_temperature_multiplier = 1.0F;
    std::array<float, 32> inv_freq{};
};

// Largest supported context, as a multiple of the native window. Qwen3.6's 262,144-token rotary
// window is extended to 384K (1.5x) and 512K (2x) here; beyond that the frequency ramp removes
// too much positional resolution to be useful.
inline constexpr std::uint32_t kMaximumYarnScale = 2;

[[nodiscard]] inline constexpr std::uint32_t maximum_yarn_context(
    std::uint32_t native_context) noexcept {
    return native_context * kMaximumYarnScale;
}

// 32 split-half RoPE divisors, i.e. one per rotated pair of the 64-wide text rotary dimension.
[[nodiscard]] inline YarnConfig compute_yarn_config(std::uint32_t native_context,
                                                    std::uint32_t context_capacity) {
    constexpr float kBaseTheta = 1.0e7F;
    constexpr float kBeta      = 32.0F;
    constexpr float kAlpha     = 1.0F;
    constexpr double kPi       = 3.14159265358979323846;

    YarnConfig cfg;
    if (context_capacity <= native_context) { return cfg; }

    const double native = static_cast<double>(native_context);
    const double kRlow  = native / static_cast<double>(kBeta);
    const double kRhigh = native / static_cast<double>(kAlpha);
    cfg.scale_factor = static_cast<float>(static_cast<double>(context_capacity) / native);
    cfg.attn_temperature_multiplier = 0.1F * std::log(cfg.scale_factor) + 1.0F;
    for (int i = 0; i < 32; ++i) {
        const double base_freq = std::pow(static_cast<double>(kBaseTheta),
                                          -2.0 * static_cast<double>(i) / 64.0);
        const double wavelength = (2.0 * kPi) / base_freq;
        double gamma            = 0.0;
        if (wavelength > kRhigh) {
            gamma = 1.0;
        } else if (wavelength > kRlow) {
            gamma = (wavelength - kRlow) / (kRhigh - kRlow);
        }
        cfg.inv_freq[i] =
            static_cast<float>(base_freq * ((1.0 - gamma) + gamma / static_cast<double>(cfg.scale_factor)));
    }
    return cfg;
}

} // namespace ninfer::targets::qwen3_6::detail
