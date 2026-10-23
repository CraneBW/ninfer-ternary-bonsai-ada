#include "targets/qwen3_6/impl/runtime/yarn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

using ninfer::targets::qwen3_6::detail::compute_yarn_config;
using ninfer::targets::qwen3_6::detail::kMaximumYarnScale;
using ninfer::targets::qwen3_6::detail::maximum_yarn_context;

constexpr std::uint32_t kNative = 262144;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "yarn test failed: %s\n", message);
        std::exit(1);
    }
}

bool close_to(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance * std::fmax(1.0F, std::fabs(expected));
}

// The baked native table is theta^(-2*pair/64) for theta = 1e7, which is what the rope kernel
// rotates with when no YaRN scaling is published.
float native_divisor(int pair) {
    return static_cast<float>(std::pow(1.0e7, -2.0 * static_cast<double>(pair) / 64.0));
}

} // namespace

// A capacity inside the native window must leave the baked table alone: the caller publishes
// nothing, so scale stays 1 and the native rotation is bit-exact.
int main() {
    require(kMaximumYarnScale == 2, "the supported extension is twice the native window");
    require(maximum_yarn_context(kNative) == 524288, "512K is the largest supported context");
    for (const std::uint32_t capacity : {std::uint32_t{1}, std::uint32_t{8192}, kNative}) {
        const auto native = compute_yarn_config(kNative, capacity);
        require(native.scale_factor == 1.0F, "a native capacity reports scale one");
        require(native.attn_temperature_multiplier == 1.0F,
                "a native capacity keeps the native attention temperature");
        for (const float divisor : native.inv_freq) {
            require(divisor == 0.0F, "a native capacity leaves the published table untouched");
        }
    }

    // 384K: one and a half times the native window.
    const auto wide = compute_yarn_config(kNative, 393216);
    require(close_to(wide.scale_factor, 1.5F, 1e-6F), "384K scales the window by 1.5");
    require(close_to(wide.attn_temperature_multiplier, 0.1F * std::log(1.5F) + 1.0F, 1e-6F),
            "384K applies the YaRN temperature multiplier");
    // The highest frequencies keep their native divisors, the lowest ones are divided by F.
    require(close_to(wide.inv_freq[0], 1.0F, 1e-6F), "pair zero is unaffected by the ramp");
    constexpr int kLast = 31;
    require(close_to(wide.inv_freq[kLast], native_divisor(kLast) / 1.5F, 1e-5F),
            "the lowest pair is divided by the scale factor");
    for (int pair = 1; pair < 32; ++pair) {
        require(wide.inv_freq[pair] < wide.inv_freq[pair - 1], "the divisors stay ordered");
        require(wide.inv_freq[pair] <= native_divisor(pair) * 1.0001F,
                "the ramp never exceeds the native divisor");
    }

    // 512K: the largest supported extension.
    const auto widest = compute_yarn_config(kNative, 524288);
    require(close_to(widest.scale_factor, 2.0F, 1e-6F), "512K scales the window by two");
    require(close_to(widest.attn_temperature_multiplier, 0.1F * std::log(2.0F) + 1.0F, 1e-6F),
            "512K applies the YaRN temperature multiplier");
    require(close_to(widest.inv_freq[kLast], native_divisor(kLast) / 2.0F, 1e-5F),
            "the lowest pair halves at scale two");

    // Only wavelengths longer than native/32 are touched: the ramp is monotone in the pair index
    // and continuous at both beta and alpha bounds.
    const auto just_past = compute_yarn_config(kNative, kNative + 64);
    require(close_to(just_past.scale_factor, 1.0002442F, 1e-5F), "the scale follows the capacity");
    for (int pair = 0; pair < 32; ++pair) {
        const float native_pair = native_divisor(pair);
        require(just_past.inv_freq[pair] <= native_pair * 1.000001F,
                "a barely extended window never raises a divisor");
        require(just_past.inv_freq[pair] >= native_pair / 1.001F,
                "a barely extended window keeps the native divisors");
    }

    // Fixed-factor policy from the KVMem native backend applies even inside
    // the nominal context. Native/default calculations above remain unchanged.
    for (const float factor : {1.0F, 2.0F, 4.0F}) {
        const auto fixed = compute_yarn_config(kNative, 212992, factor);
        require(fixed.scale_factor == factor, "fixed factor does not follow capacity");
        require(close_to(fixed.attn_temperature_multiplier,
                         static_cast<float>(1.0 + std::log(double(factor)) / 10.0), 1e-6F),
                "fixed factor temperature matches independent double expression");
        if (factor == 1) {
            for (const float divisor : fixed.inv_freq) require(divisor == 0, "fixed one uses baked table");
        } else {
            require(fixed.inv_freq[0] == 1.0F, "fixed factor preserves shortest wavelength");
            require(close_to(fixed.inv_freq[31], native_divisor(31) / factor, 1e-7F),
                    "fixed factor scales longest wavelength");
        }
    }
    for (const float bad : {0.5F, 5.0F, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected=false;
        try { (void)compute_yarn_config(kNative, 212992, bad); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid fixed factor rejected");
    }

    std::puts("yarn tests passed");
    return 0;
}
