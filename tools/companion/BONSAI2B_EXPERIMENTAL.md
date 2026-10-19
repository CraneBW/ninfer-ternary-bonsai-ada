# Companion Bonsai 2B v3 exact-shape and 512-point Hadamard patch

This patch applies to the **separate** `ninfer-all` Companion upstream fork at
baseline `b2913b6`. It is stored in this main Ada repository; the Companion
working tree has deliberately **not** been committed or pushed. The complete
patch is `tools/companion/bonsai2b-smallshape-experimental.patch` (2,449 lines,
44 changed files at this checkpoint). It includes `include/ninfer/ops`,
`src/models/qwen3_5`, `src/ops` and `tests/ops` — **not** just the original
small-geometry GPU changes.

Apply to a clean matching checkout using `git apply --check <patch>` followed
by `git apply <patch>`. The existing dirty Companion checkout already has these
changes; `git apply --reverse --check <patch>` PASSED on 2026-10-20. Do not
apply a second time there.

## Verified model and discovered root cause

- Input: `/home/ran/ninfer-work/real-models/bonsai2_2b_v3.ninfer`;
  650,907,136 bytes; downloader verified whole-file SHA-256
  `0601756954b5fc24e9056b56e80bad6f40f8db9985e47ddf421eb833e76d7b63`.
- v3 metadata explicitly contains `hadamard_block_size = 512`.
- Previously: real load (608.4 MiB), warmup and HTTP 200, but incoherent
  multilingual output under both BF16/INT8 KV. The Companion runtime was
  applying 1024-point rotations to the 512-point converted checkpoint.
- Fixed: separate real CUDA 512-point butterfly/embedding code paths,
  normalization `1/sqrt(512)`, explicit per-model block-size from v3 metadata
  through Qwen3.5 Load/Bindings/Runtime and op APIs. 1024-point kernels remain
  separately available and are still the default when metadata does not
  specify otherwise. No process-global rotation environment variable.
- The explicit 512 route covers token embeddings, input projections,
  attention, GDN control/conv/output, RMSNorm/gated RMSNorm/Sigmoid, FFN
  gate/up/down and output head. Where 1024 fused producers cannot represent
  512, operations use exact 512-composition fallbacks.
- Added Bonsai2B geometries for ternary T2 linear, GDN 16/2048 gates,
  6144-channel conv, attention 8 Q/2 KV heads with head_dim 256 and
  query/gate/key/value projections. Existing 27B geometries remain supported.

## Focused numerical tests — PASS

Independent CUDA `sm_89` `nvcc` compile: 512 Hadamard launcher and T2
embedding PASS; C++ syntax checks for all seven changed Qwen3.5 model
translation units PASS. For real model linking, the 76 Qwen3.5 translation
units from loading/runtime were separately recompiled into **private** archives
without writing to Ninja's build directory, and the ad-hoc `ninfer-serve`
executable successfully linked. This is a diagnostic executable, **not** the
formal `build-ada-companion/apps/ninfer-serve` build.

`tmp-nvcc/bonsai2b-adhoc/test-hadamard-h512 --block512` PASS, using CPU
reference/oracles for forward/inverse Walsh, round trips, in-place operation,
T2 rotated embedding (512 and 2048 widths), SwiGLU, RMSNorm, gated RMSNorm,
sigmoid and invalid 256-point block-size rejection.

`tmp-nvcc/bonsai2b-adhoc/test-hadamard-h512 --block1024-smoke` PASS:
original 1024 path including 5120/6144 widths and fused producer comparison.

Other pre-existing focused GPU tests passed before the 512 change:
`test-t2-a16 --bonsai2b`, `test-gdn-input --bonsai2b`, `test-gdn-gating
--bonsai2b`, `test-attn-input --bonsai2b`, `test-causal-conv --bonsai2b`,
`test-softmax`. Their outputs and GPU oracle executables remain in
`tmp-nvcc/bonsai2b-adhoc/` and were not pointlessly rerun here.

## Real Bonsai2B generation — PASSED with private executable

Executable: `tmp-nvcc/bonsai2b-adhoc/ninfer-serve-512-metadata` using full,
SHA-256-verified Bonsai 2B v3 weights, GPU RTX 4070 Ti SUPER sm_89,
`--max-context 1024 --kv-capacity 1024 --host-kv-mib 256 --greedy
--no-thinking --no-cuda-graph`.

BF16 KV HTTP Chat Completions (3/3 HTTP200, finish `stop`):

| Prompt | Exact model response |
|---|---|
| Say hello. | Hello! How can I help you today? |
| What is 1+1? Answer only the number. | 2 |
| Respond with only green. | Green. |

INT8 KV HTTP Chat Completions (5/5 HTTP200, finish `stop`):

| Prompt | Exact model response |
|---|---|
| Say hello. | Hello! How can I help you today? |
| What is 1+1? Answer only the number. | 2 |
| What is the capital of France? Reply with the city name. | Paris |
| I said: apple. Repeat only the fruit. | apple |
| Three messages: remember 37, answer OK, then recall | 37 |

Logs: `tmp-nvcc/bonsai2b-adhoc/real-512-metadata-{bf16,int8}.log`.
Both test services were gracefully shut down. The original broken output
is preserved in previous `real-startup-fixed.d.log` and
`real-bf16-smoke.log` for comparison.

## Remaining acceptance and isolation

The already-running `build-ada-companion` Ninja PID 662841 must complete its
own build before executing a formal rebuilt `apps/ninfer-serve`. **Do not run
another Ninja targeting that directory concurrently, stop its process, or
overwrite its `.o`/libraries.** Retest the formal binary once up to date;
substantial quality/performance/context tests and other Companion models are
not implied by these focused semantic requests.

This checkpoint patch must not be confused with full v3 Vision inference,
27B Groupwise/NVFP4 serving, TP2 native multi-GPU execution, or Strata ×
ActKV, which are separate remaining features in the main task.
