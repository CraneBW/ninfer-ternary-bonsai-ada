# Qwen3.6 27B v3 NVFP4 on Ada: real runtime and semantic row-parent validation

2026-10-20. RTX 4070 Ti SUPER, sm_89, single GPU (16 GiB), managed weights.

The original full v3 NVFP4 source in `real-models/qwen3_6_27b_nvfp4.ninfer`
is **18,324,354,820 bytes**, previously downloaded and SHA-256 verified:
`0448262d15df2ae4fda761540c110bc19e7c3b4f43c0938e4d50474429cda083`.
No downloading or hashing was repeated during this work.

## Actual issues and implemented fixes

On the original native target the 16 GiB weight load succeeded, then failed
`materialized_weight: NVFP4 requires target-validated weight and input divisors`.

- v3 uses `Use.auxiliaries.activation_input_divisor` to associate a separate
  FP32 scalar with each NVFP4 logical projection. Source-file metadata has
  485 NVFP4 semantic bindings, 247 physical parents, and every relevant Use
  has this divisor. Binding now reads and records the exact physical encoded
  parent weight divisor plus each Use's input divisor, validating positivity
  and finiteness when *real* weights reach ModelView. Shape-only sparse
  headers remain inspectable without inventing non-zero payload bytes.
- Shared `block_scale_k16_m128x4_v1` parents have distinct code and scale
  planes. Logical views require exact M128 row boundaries; their two base
  pointers derive from independent row offsets in the parent. The Op validates
  the complete parent payload size, code/scale intervals, row alignment,
  divisor values, and parent capacity. It never treats a parent row slice as
  a newly packed standalone object.
- Production CUDA A16 GEMV dispatch adds the actual v3 individual NVFP4
  shapes `(N,K)` of `(1024,5120)`, `(2048,5120)`, `(6144,5120)` and
  `(17408,5120)`. Exact existing fused formats and native 1024/27B
  schedules remain unchanged. V3 row-view A16 falls back to per-token GEMV,
  rather than manufacturing a non-existent W4A4 fused kernel.
- On sm_89/sm_86/sm_75, NVFP4 A4 permission selects the available BF16
  activation path. Blackwell W4A4 remains enabled where compiled. A16 longer
  sequences compose existing linear plus SwiGLU/GDN materialized paths rather
  than incorrectly invoking Blackwell-only TMA kernels.
- Some actual v3 BF16 attention parents are independent `(6144,5120)` and
  `(1024,5120)` weights, not the original fused BF16 profiles. Those are
  now served by existing BF16 FP32-accumulating CUDA GEMV kernels.

## Focused numerical verification

Both targeted builds completed using `TMPDIR=$PWD/tmp-nvcc`, Release Ninja
and `-j2` (without running the full 124-test suite).

```sh
TMPDIR=$PWD/tmp-nvcc cmake --build build --target ninfer_linear_nvfp4_a16_test -j2
./build/tests/ninfer_linear_nvfp4_a16_test --v3-rowview
# OK NVFP4_A16 Linear

TMPDIR=$PWD/tmp-nvcc cmake --build build --target ninfer_linear_bf16_a16_test -j2
./build/tests/ninfer_linear_bf16_a16_test --v3-individual
# OK BF16_A16 Linear

./build/tests/ninfer_qwen3_6_27b_load_plan_test --v3-header-only
# real v3 GSQ text metadata validated
```

The NVFP4 targeted mode covers all four new geometries, two token counts,
independent CPU sampled numeric oracle, and a parent-slice shape/pointer
validator with negative scale-offset, row-alignment and input-divisor cases.
The BF16 targeted mode covers both independent output widths at T=1/9 with
FP64 CPU dot-product comparisons. These are numerical correctness tests,
not throughput measurements.

## Real whole-model CUDA request

```sh
./build/apps/ninfer-serve /home/ran/ninfer-work/real-models/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18105 --managed-weights \
  --max-context 512 --kv-capacity 512 --host-kv-mib 64 \
  --max-concurrency 1 --default-max-tokens 16 --kv-dtype int8 \
  --greedy --no-thinking --no-cuda-graph --log-level info
```

Log: `tmp-nvcc/qwen36-v3-nvfp4-bf16-slice-smoke.log`.
It loaded **16.0 GiB** of real weights, completed warmup in **18.3 s**,
and served two actual OpenAI Chat Completions requests:

| Prompt | Response | API |
|---|---|---|
| `Reply with: hello` | `hello` | HTTP 200, `stop` |
| `What is 1+1? Answer only the number.` | `2` | HTTP 200, `stop` |

The recorded requests took 5.5 and 6.8 seconds, respectively; this model
nearly fills the GPU and unified-memory paging constrains throughput.
Do not generalize these times to a performance benchmark or long context.
The test service was stopped cleanly after the requests.

**Scope:** Native v3 NVFP4 short-text real generation has been demonstrated.
This is not a claim that every 27B feature (Vision, DFlash2/MTP in NVFP4,
long context, TP2) is complete or that all contexts are stable. GSQ v3
Vision/MTP/DFlash2 and Bonsai2B are tracked in separate reports.
