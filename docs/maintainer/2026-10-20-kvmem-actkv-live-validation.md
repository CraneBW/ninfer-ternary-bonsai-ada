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
