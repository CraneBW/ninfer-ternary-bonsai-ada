# Qwen3.6 27B v3 Groupwise: real Ada inference

Verified 2026-10-20 on RTX 4070 Ti SUPER sm_89 (16 GiB) with native
`build/apps/ninfer-serve`. The complete source file is
`/home/ran/ninfer-work/real-models/qwen3_6_27b.ninfer`, size 17,495,538,688 bytes,
previous downloader-verified SHA-256
`9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b`.
No new download or whole-file hashing was needed.

## Implementation and targeted test

The old full native startup loaded 15.3 GiB, but warmup rejected
`q4 linear: unsupported shape or T (n=2048, k=5120, t=9)`.
Actual v3 groupwise logical projections select consecutive rows of shared
`row_split_k128_v1` Q4 parents, with individual shapes `(N,K)`
`(2048,5120)` (GDN Q/K) and `(17408,5120)` (independent MLP gate/up).
The original CUDA Q4 SIMT, GEMV and MMA kernels already consume arbitrary
row-split N via runtime bounds, so the public linear dispatcher now selects
these existing kernels for only those two exact v3 shapes; it does not
delete the 27B shape checks or alter the stored weight bytes.

The separate, pre-existing dirty modification to
`src/ops/linear/q4/q4_dispatch.cpp` belongs to another agent, and was
**not** edited or included in this patch.

```sh
TMPDIR=$PWD/tmp-nvcc cmake --build build \
  --target ninfer_linear_q4_a16_test ninfer-serve -j2
./build/tests/ninfer_linear_q4_a16_test --v3-individual
# OK Q4_A16 Linear
```

The targeted Q4 test independently checks FP32/FP64 CPU quantization/oracle
against GPU output at T=1,3,9,17 with both exact newly supported N values,
including the existing input/weight preservation tests. It does not rerun
the full 124-test suite.

## Full model, real HTTP requests

```sh
./build/apps/ninfer-serve \
  /home/ran/ninfer-work/real-models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18106 --managed-weights \
  --max-context 512 --kv-capacity 512 --host-kv-mib 64 \
  --max-concurrency 1 --default-max-tokens 16 --kv-dtype int8 \
  --greedy --no-thinking --no-cuda-graph --log-level info
```

Evidence: `tmp-nvcc/qwen36-v3-groupwise-q4fixed-smoke.log` and the actual
non-stream OpenAI Chat Completions responses. The model loaded **15.3 GiB**
of real weights, completed warmup in **14.5 seconds**, and generated:

| Prompt | Response | Outcome |
|---|---|---|
| `Reply with: hello` | `hello` | HTTP 200, stop, 5.1 seconds |
| `What is 1+1? Answer only the number.` | `2` | HTTP 200, stop, 6.4 seconds |

Service was gracefully shut down. With about 67 MiB of GPU memory remaining,
this is a short real-generation **correctness smoke**, not a performance
benchmark or proof of long-context stability. Vision, MTP and DFlash2 for
this exact groupwise artifact, as well as TP2 two-rank runtime and Strata ×
ActKV, remain separate acceptance scopes.
