# Qwen3.6 27B v3 Groupwise MTP: first live Ada inference

2026-10-20, RTX 4070 Ti SUPER (sm_89, one GPU). Used the existing complete,
previously SHA-256-verified `qwen3_6_27b.ninfer` without downloading or
rehashing any model weights.

## Actual failure and fix

The original full MTP startup loaded 15.7 GiB of v3 Groupwise + MTP weights
and initialized the engine, but failed during warmup with `w8 linear:
unsupported shape or T`. The v3 MTP uses two independent W8 gate/up parents
with **[17408,5120]** logical row-major matrices, whereas the dispatcher had
only the older packed MTP gate/up shape **[34816,5120]** and its specialized
small-T kernel. This was a real execution error, not a metadata rejection.

Added an explicit `[17408,5120]` case to the existing W8 A16 dispatch, using
the already-implemented **shape-generic SIMT** schedule for T<=16 and the
generic MMA schedule thereafter. The change does not modify the external
dirty `src/ops/linear/q4/q4_dispatch.cpp`.

Focused single-GPU oracle:

```sh
TMPDIR=$PWD/tmp-nvcc cmake --build build --target ninfer_linear_w8_a16_test ninfer-serve -j2
./build/tests/ninfer_linear_w8_a16_test --v3-mtp-individual
# OK W8_A16 Linear (T=1,3,9,17)
```

## First real MTP request — PASS

One native `ninfer-serve` process, `--managed-weights`, `--spec mtp
--draft-tokens 2`, `--max-context 256 --kv-capacity 256 --kv-dtype int8
--greedy --no-thinking --no-cuda-graph`, one request, no dual-GPU execution.
Small state and Host KV settings were used because the static weights consumed
15.7 GiB, leaving only 69.6 MiB free after startup.

| Observation | Result |
|---|---|
| Native model startup and warmup | PASS, 9.8s warmup |
| User prompt | `What is 1+1? Answer only the number.` |
| API | HTTP 200; `finish_reason=stop` |
| Text | `2` |
| Draft accepted | 2/2 |
| Usage | 24 prompt / 2 completion tokens |

Local evidence (untracked):

```text
tmp-nvcc/qwen36-v3-groupwise-mtp-20261020.log
tmp-nvcc/qwen36-v3-groupwise-mtp-fixed-20261020.log
tmp-nvcc/qwen36-v3-groupwise-mtp-fixed-response.json
```

The server terminated after the request. This confirms short real Groupwise
MTP execution, **not** long-context acceptance, throughput optimization,
NVFP4 MTP, or multimodal combinations.
