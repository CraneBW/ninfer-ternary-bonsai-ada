#include "kvmem/kvmem_retrieval.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "kvmem retrieval test failed: %s\n", message);
        std::exit(1);
    }
}

bool close_to(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-5F;
}

} // namespace

// The scorer measures a resident and an evicted block on the same evidence: the
// per-layer raw mean-K sum kept beside the block. This fixture pins the GQA
// aggregation the reference uses (max over the group, not the mean) and the
// layer average over layers that actually saw a query token.
int main() {
    constexpr std::uint32_t kLayers = 2;
    constexpr std::uint32_t kQueryHeads = 2;
    constexpr std::uint32_t kKvHeads = 1;
    constexpr std::uint32_t kHeadDim = 2;
    constexpr std::uint32_t kBlocks = 3;

    // Layer 1 has no query evidence, so it must not dilute the layer average.
    std::vector<float> query(kLayers * kQueryHeads * kHeadDim, 0.0F);
    query[0] = 2.0F;  // head 0: (1, 0)
    query[1] = 0.0F;
    query[2] = 0.0F;  // head 1: (0, 1)
    query[3] = 2.0F;

    std::vector<float> mean_k(kBlocks * kLayers * kKvHeads * kHeadDim, 0.0F);
    mean_k[0] = 4.0F;  // block 0 layer 0: sum (4, 0) over 4 tokens
    mean_k[1] = 0.0F;
    mean_k[2] = 4.0F;  // block 1 layer 0: sum (-4, 0)
    mean_k[3] = 0.0F;
    // block 2 has no mean-K yet (never resident long enough to export).
    std::vector<std::uint32_t> tokens{4, 4, 0};

    const ninfer::kvmem::MeanKeyQueryView view{query.data(), kLayers, kHeadDim, kQueryHeads,
                                               kKvHeads};
    const std::vector<float> scores =
        ninfer::kvmem::mean_key_scores(kBlocks, mean_k.data(), tokens.data(), view);

    require(scores.size() == kBlocks, "one score per block");
    // Block 0 aligns with head 0 exactly; GQA max keeps that 1.0 where a group
    // mean would have reported 0.707. Inactive layer 1 must not halve it.
    require(close_to(scores[0], 1.0F), "aligned block scores one");
    // Block 1 is opposed by head 0 and orthogonal to head 1, so its best cosine is 0.
    require(close_to(scores[1], 0.0F), "opposed block scores zero");
    require(close_to(scores[2], 0.0F), "block without mean-K scores zero");

    std::puts("kvmem retrieval tests passed");
    return 0;
}
