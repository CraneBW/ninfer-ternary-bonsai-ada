#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

// 7, not the 5 the reference port shipped, and the ceiling is structural rather than tuned: one
// round verifies K+1 tokens and the verify path runs on the 8-wide small-t tile, so 7 is the last K
// whose width (8) still fits it. K=8 would make T=9, which falls off that tile -- measured 13.3 ms
// per verify round at T=8 against 40 ms at T=9, i.e. the extra draft costs more than it can return.
//
// Why raising it at all: the per-position survival rate is a property of the TASK, not of the
// model, and it varies far more than the 5 assumed. Measured over three fixtures at K=5 (survival
// per position, fitted T0~15.83 ms and c~1.94 ms per draft):
//   en-code (code editing)   ~0.76  -> optimum is K=4
//   mtp-count   (counting)   ~0.95  -> optimum is K=7
//   mtp-template (repetitive) ~0.995 -> optimum is beyond 7; its histogram reads 43,43,42,42,42,
//                                       essentially no decay
// The cost of a draft is fixed (~1.94 ms) and the return is the survival probability, so the
// crossover sits at s ~= 0.75: below it more K loses, above it more K wins. A single deployment-wide
// K cannot serve all three, which is why this is a CEILING and --draft-tokens picks within it.
// Adaptive K would need a captured graph per K (this constant is part of the graph shape) and is a
// separate change.
//
// This number is written in four places and they are kept in step by hand, because each is a
// different kind of thing rather than four copies of one: this one is the captured graph's width,
// kMaximumMtpDraftTokens in the 27b config is the model's own ceiling, the range check in
// product/speculative_options.h is CLI validation, and the T bound in ops/wrapper/mtp_round.cpp is
// a runtime invariant. Raise all four together; raising fewer fails loudly at one of those checks
// rather than producing a wrong answer.
inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 7;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_valid_columns{};
    // DFlash uses logical positions for its own attention. Target verification carries a separate
    // continuation RoPE position so multimodal rows retain their per-sequence rope_delta.
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth>
        target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    TensorRegion produced_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion verify_positions;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    Tensor produced_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor proposal_valid_columns;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor verify_positions;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
