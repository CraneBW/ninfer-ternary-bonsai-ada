#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{20, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    constexpr std::array new_problem_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, new_problem_invocations});
    return failures;
}

int verify_v3_parent_slice_geometry() {
    constexpr std::uintptr_t base = 0x10000000;
    constexpr std::uint64_t parent_rows = 16384;
    constexpr std::uint64_t columns = 5120;
    constexpr std::uint64_t payload_bytes = parent_rows * columns * 9 / 16 + 4;
    constexpr std::uint64_t first_row = 2048;
    Weight weight{};
    weight.payload = reinterpret_cast<const void*>(base);
    weight.payload_bytes = payload_bytes;
    weight.qtype = QType::NVFP4;
    weight.layout = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype = DType::FP8_E4M3FN;
    weight.group_size = weight.group = 16;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = 6144;
    weight.k = weight.shape[1] = weight.padded_shape[1] = columns;
    weight.qdata = reinterpret_cast<const void*>(base + first_row * columns / 2);
    weight.scales = reinterpret_cast<const void*>(
        base + parent_rows * columns / 2 + first_row * columns / 16);
    weight.weight_scale_divisor = 512.0F;
    weight.input_scale_divisor = 128.0F;
    auto expect_rejected = [&](const Weight& candidate) {
        try {
            (void)ops::detail::validate_nvfp4_weight(candidate, "NVFP4 v3 slice test");
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const auto geometry = ops::detail::validate_nvfp4_weight(weight, "NVFP4 v3 slice test");
    if (geometry.code_plane_bytes != 6144ULL * columns / 2) {
        std::cerr << "invalid accepted NVFP4 row slice size\n";
        return 1;
    }
    Weight corrupt = weight;
    corrupt.scales = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(corrupt.scales) + 16);
    if (!expect_rejected(corrupt)) {
        std::cerr << "invalid NVFP4 scale-row slice accepted\n";
        return 1;
    }
    corrupt = weight;
    corrupt.qdata = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(corrupt.qdata) + columns / 2);
    if (!expect_rejected(corrupt)) {
        std::cerr << "unaligned NVFP4 parent row accepted\n";
        return 1;
    }
    corrupt = weight;
    corrupt.input_scale_divisor = 0;
    if (!expect_rejected(corrupt)) {
        std::cerr << "invalid NVFP4 activation divisor accepted\n";
        return 1;
    }
    return 0;
}

int run_v3_row_views() {
    constexpr std::array invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    int failures = verify_v3_parent_slice_geometry();
    for (const std::int32_t rows : {1024, 2048, 6144, 17408}) {
        failures += run_shape("NVFP4_V3_A16", ActivationCompute::A16, make_nvfp4_weight,
                              {rows, 5120, 908U + static_cast<std::uint32_t>(rows),
                               Comparison::Sampled, true, invocations});
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--v3-rowview")) {
            std::cerr << "usage: test_nvfp4_a16 [--v3-rowview]\n";
            return 2;
        }
        const int failures = argc == 2 ? run_v3_row_views() : run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
