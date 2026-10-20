# KVMem Strata × Action-LRFU real native GSQ v3 smoke

2026-10-20, RTX 4070 Ti SUPER sm_89. This documents **existing implemented
features** and an additional actual long-prompt request; it is not a claim of
a new TP2 or 256K-context implementation.

The native KVMem host code already implements `BlockStratum` with
`PinnedSink`, `ActionProtected`, `EphemeralThinking`, and standard context;
retrieval/profile quotas, time-decayed LRFU, thinking-span demotion, and
Host/NVMe cold-spill priority. The Qwen3.6 runtime passes
`--kvmem-action-lrfu` and scoring parameters into `record_lrfu_scores`,
`protect_system_prefix`, `protect_action_span` and
`set_closed_thinking_spans` during live query-driven scoring. The separate
unit tests passed:

```text
./build/tests/kvmem_blocks_standalone
  action_lrfu ok; failures=0 VERDICT: PASS
./build/tests/ninfer_kvmem_retrieval_test
  kvmem retrieval tests passed
./build/tests/ninfer_kvmem_cold_store_test
  kvmem cold store: PASS (0 failures)
```

## Actual GSQ v3 KVMem request

Used the existing SHA-256 verified complete GSQ model at
`/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer`
(SHA-256
`29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`).

```sh
NINFER_KVMEM_TRACE=1 ./build/apps/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port 18107 --managed-weights \
  --max-context 2048 --kv-device-tokens 1024 \
  --kvmem --kvmem-budget 512 --kvmem-gen-reserve 128 \
  --kvmem-action-lrfu --kvmem-lrfu-lambda 0.5 \
  --kvmem-lrfu-top-frac 0.9 --prefill-chunk 256 \
  --host-kv-mib 128 --max-concurrency 1 \
  --default-max-tokens 8 --kv-dtype int8 --greedy \
  --no-thinking --no-cuda-graph --log-level info
```

An actual Chat Completions user prompt contained 110 repeated ordinary
reference notes, followed by `What is 2+2? Reply only the digit.` The
tokenizer returned **1,898 prompt tokens** (larger than the 512-token
working set), and the HTTP request returned **200**, stop, output `4`
(two output tokens) in 1.51 seconds.

Log: `tmp-nvcc/gsq-v3-kvmem-action-lrfu-smoke.log`.

```text
KVMEM_TRACE score id=1 tier=1 tokens=64 retrieval=0.25432
KVMEM_TRACE score id=29 tier=0 tokens=43 retrieval=0.66448
KVMEM_TRACE select window_tokens=512 ids=0,23,24,25,26,27,28,29,
KVMEM_TRACE cold bytes=0 blocks=0 closed_thinking=0 host_capacity=70254592 spills=0
req#1 done | openai-chat | stop token | prompt 1,898 | output 2
```

Live `/metrics` returned `ninfer:kvmem_host_capacity_bytes 70254592`;
`ninfer:kvmem_cold_bytes` and `host_to_cold_spills_total` both zero for
this small test. The trace independently demonstrates that middle pages
reside in Host tier (`tier=1`) and the selected GPU window retains the
sink and newest high-scoring pages. The server was shut down normally.

**Limitations:** One long-prompt request proves live scoring, window
selection, Host/GPU tiering and correct short answer; it does not establish
multi-turn LRFU recall quality, correctness of all tool-span strata, 256K
context, or actual NVMe spill in this run. Cold tier movement was separately
verified in the earlier KVMem regression. TP2 multi-GPU native execution is
unrelated and remains unverified on this one-GPU machine.

## Later 256K-class CPU LRFU scheduling fix

The prior `top_fraction` positive-score threshold retained every item tied at
the cutoff. Equal cosine scores across a large set of pages could therefore
admit significantly more than the intended fraction. `record_lrfu_scores`
now selects an **exact** `ceil(top_fraction * positive_count)` top-k using
deterministic new-page-first tie breaking, while preserving the existing
time-decayed LRFU accumulation.

The standalone `tests/kvmem_blocks_test.cpp` was extended to 4096 pages of
64 tokens (262,144 tokens total), with 4096 identical positive scores and a
10% selection rate. Exactly 410 scored pages survive, followed by a working
set of 256 selected pages under a 16,384-token device budget. A subsequent
query verifies decay, and two action/query passes exercise historical fact
retrieval, protection of the latest action, and demotion of closed thinking.
The standalone C++ test builds and passes on CPU (`failures=0`). This verifies
**host decision mathematics at 256K scale**, not actual 256K model inference
or an end-to-end recall-quality benchmark.

## 4096-page ColdKVStore metadata growth and reuse

The on-disk `ColdKVStore` previously called `free_.reserve(slots_.size()+1)`
for every first-time slot allocation, causing repeated free-list reallocations
and potentially quadratic metadata-copy work as cold pages accumulate. New
slots now reserve free-list capacity geometrically, still *before* publication
so `release()` can remain `noexcept`.

The targeted `ninfer_kvmem_cold_store_test` passed (0 failures) after adding
4096 64-byte cold records, representing the page count of 256K tokens at
64 tokens/page. It checked byte-exact samples across the allocation span,
release of 2048 slots, reuse with generation invalidation, and bounded active
page accounting. This is a CPU/NVMe-backing data-structure and I/O check,
not a 256K GPU inference or throughput claim.

## Two-turn historical detail recall: 512-token KVMem missed a fact

A separate single-GPU native Qwen3.8 GSQ v3 test used an artificial archive
of 171 notes. The fact in the **middle** of the first user turn said:
`The Indigo Archive display label is cobalt-orchid-73.` The first Chat
Completions request contained 2,940 prompt tokens and returned `ACK` (HTTP
200). A second request supplied that history plus assistant `ACK` and a
question asking for the exact display label (2,968 prompt tokens).

With `--max-context 3072 --kv-device-tokens 1024 --kvmem
--kvmem-budget 512 --host-kv-mib 128 --kvmem-action-lrfu
--kvmem-lrfu-lambda 0.5 --kvmem-lrfu-top-frac 0.3`, the second request
returned **`Indigo Archive` (incorrect)**, despite HTTP 200 and 2,933
cached prompt tokens (98.8%). The final selection trace showed eight
64-token pages with ids `0,40,41,42,43,44,45,46`; it excluded the page
containing the earlier exact label. No Cold spill was observed (0 cold bytes,
0 Host→Cold spills; 13 compactions total). This specifically demonstrates
a retrieval/retention miss, not NVMe data loss.

An independent no-KVMem control served the **same second request** with
`--kv-device-tokens 3072`; HTTP 200 returned the exact
`cobalt-orchid-73`. The models used the same verified full GSQ weights,
single RTX 4070 Ti SUPER, greedy decoding and no thinking. Both servers
were stopped afterwards. Evidence:

```text
tmp-nvcc/kvmem-multiturn-first-request.json
tmp-nvcc/kvmem-multiturn-first-response.json
tmp-nvcc/kvmem-multiturn-second-request.json
tmp-nvcc/kvmem-multiturn-second-response.json
tmp-nvcc/kvmem-multiturn-fullkv-control-response.json
tmp-nvcc/gsq-v3-kvmem-multiturn-20261020.log
tmp-nvcc/gsq-v3-fullkv-control-20261020.log
```

**Outstanding quality gap:** query-conditioned page ranking and device-window
retention do not guarantee exact recall of an arbitrary fact far back in the
history. This single artificial example does not estimate recall rates, and
changing global ranking weights without a broader oracle could regress other
tasks. Do not mark 256K/multi-turn semantic accuracy as accepted. This test
also does not exercise a Host→Cold spill; Cold byte accounting and restoration
were established separately.
