// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// Host-side mean-key retrieval scoring for KVMem K3.
//
// The reference implementation persists one F32 sum of raw K per block, layer and KV head while
// the block is resident, so a block parked in host RAM still carries retrieval evidence. This
// scorer consumes those sums plus the request's raw query sums and reproduces the reference's
// GQA-max cosine: for each layer and KV head take the best cosine over the query heads of that
// KV head's group, average the KV heads, then average the layers that saw a query token.

#pragma once

#include "kvmem/kvmem_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace ninfer::kvmem {

struct MeanKeyQueryView {
    const float* sums = nullptr;  // [layers][query_heads][head_dim] pre-RoPE query sums
    std::uint32_t layers      = 0;
    std::uint32_t head_dim    = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads    = 0;
};

// One score per block: best query-to-subblock cosine (16 tokens per centroid),
// aggregated by GQA-max and averaged over active layers and KV heads.
inline std::vector<float> mean_key_scores(std::uint32_t blocks, const float* mean_k,
                                          const std::uint32_t* block_tokens,
                                          const MeanKeyQueryView& query) {
    std::vector<float> scores(blocks, 0.0f);
    if (blocks == 0 || mean_k == nullptr || block_tokens == nullptr || query.sums == nullptr ||
        query.layers == 0 || query.head_dim == 0 || query.query_heads == 0 || query.kv_heads == 0 ||
        query.query_heads % query.kv_heads != 0) {
        return scores;
    }
    const std::uint32_t group = query.query_heads / query.kv_heads;
    const std::size_t q_layer_stride =
        static_cast<std::size_t>(query.query_heads) * query.head_dim;
    const std::size_t k_layer_stride =
        static_cast<std::size_t>(query.kv_heads) * kMeanKeySubblocks * query.head_dim;
    const std::size_t k_block_stride = static_cast<std::size_t>(query.layers) * k_layer_stride;

    // Norm of each query head's sum. The token count cancels between the numerator and the
    // denominator, so the cosine of the means is the cosine of the sums.
    std::vector<float> query_norms(static_cast<std::size_t>(query.layers) * query.query_heads,
                                   0.0f);
    std::vector<std::uint8_t> active_layer(query.layers, 0);
    std::uint32_t active_layers = 0;
    for (std::uint32_t layer = 0; layer < query.layers; ++layer) {
        const float* layer_sums = query.sums + static_cast<std::size_t>(layer) * q_layer_stride;
        bool active = false;
        for (std::uint32_t head = 0; head < query.query_heads; ++head) {
            const float* sums = layer_sums + static_cast<std::size_t>(head) * query.head_dim;
            float norm2 = 0.0f;
            for (std::uint32_t d = 0; d < query.head_dim; ++d) {
                norm2 = std::fma(sums[d], sums[d], norm2);
            }
            const float norm = std::sqrt(norm2);
            query_norms[static_cast<std::size_t>(layer) * query.query_heads + head] = norm;
            active = active || norm > 0.0f;
        }
        if (active) {
            active_layer[layer] = 1;
            ++active_layers;
        }
    }
    if (active_layers == 0) { return scores; }

    const auto score_block = [&](std::uint32_t block) {
        const std::uint32_t tokens = block_tokens[block];
        if (tokens == 0) { return; }
        const float* block_k = mean_k + static_cast<std::size_t>(block) * k_block_stride;
        float total = 0.0f;
        for (std::uint32_t layer = 0; layer < query.layers; ++layer) {
            if (active_layer[layer] == 0) { continue; }
            const float* layer_k =
                block_k + static_cast<std::size_t>(layer) * k_layer_stride;
            float layer_score = 0.0f;
            for (std::uint32_t kv_head = 0; kv_head < query.kv_heads; ++kv_head) {
                float best = -1.0e30f;
                for (std::uint32_t sub = 0;
                     sub < kMeanKeySubblocks && sub * kMeanKeyTokensPerSubblock < tokens;
                     ++sub) {
                    const float* key =
                        layer_k +
                        (static_cast<std::size_t>(kv_head) * kMeanKeySubblocks + sub) *
                            query.head_dim;
                    float key_norm2 = 0.0f;
                    for (std::uint32_t d = 0; d < query.head_dim; ++d) {
                        key_norm2 = std::fma(key[d], key[d], key_norm2);
                    }
                    const float key_norm = std::sqrt(key_norm2);
                    if (!(key_norm > 0.0f)) { continue; }
                    for (std::uint32_t g = 0; g < group; ++g) {
                        const std::uint32_t q_head = kv_head * group + g;
                        const float q_norm = query_norms[
                            static_cast<std::size_t>(layer) * query.query_heads + q_head];
                        if (!(q_norm > 0.0f)) { continue; }
                        const float* q_sums =
                            query.sums +
                            (static_cast<std::size_t>(layer) * query.query_heads + q_head) *
                                query.head_dim;
                        float dot = 0.0f;
                        for (std::uint32_t d = 0; d < query.head_dim; ++d) {
                            dot = std::fma(q_sums[d], key[d], dot);
                        }
                        best = std::max(best, dot / (q_norm * key_norm + 1.0e-6f));
                    }
                }
                if (best > -1.0e29f) { layer_score += best; }
            }
            total += layer_score / static_cast<float>(query.kv_heads);
        }
        scores[block] = total / static_cast<float>(active_layers);
    };

    unsigned workers = std::thread::hardware_concurrency();
    workers           = std::clamp(workers, 1u, 8u);
    if (blocks < 64 || workers <= 1) {
        for (std::uint32_t block = 0; block < blocks; ++block) { score_block(block); }
        return scores;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        const std::uint32_t begin =
            static_cast<std::uint32_t>((static_cast<std::uint64_t>(blocks) * worker) / workers);
        const std::uint32_t end = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(blocks) * (worker + 1U)) / workers);
        threads.emplace_back([&, begin, end] {
            for (std::uint32_t block = begin; block < end; ++block) { score_block(block); }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    return scores;
}

} // namespace ninfer::kvmem
