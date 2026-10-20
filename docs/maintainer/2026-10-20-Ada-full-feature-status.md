# NInfer Ada full-feature work: verified state and remaining work

2026-10-20, native host `/home/ran/ninfer-work`, one RTX 4070 Ti SUPER 16 GiB.
The user's latest priority is **complete code and necessary tests before any
further model downloading**, not cancellation of downloads or full-model
acceptance. The three 27B model files and Bonsai2B below are already full
files verified by the downloader. Never restart those downloads or treat the
size of a preallocated `.ranges.part` as completion.

## Evidence-backed functionality

| Scope | Verified, narrow outcome | Report / code checkpoint |
|---|---|---|
| KVMem cold/claim/continuous requests | Four real model HTTP200, nonzero Host→cold spills and reservation/claim fixes | `2a60553`, `docs/kvmem-capacity-decoupling.md` |
| GSQ Qwen3.8 27B v3 base text | Full SHA-256 verified real model, managed-weight Chat Completions HTTP200 | `docs/maintainer/2026-10-20-v3-gsq-live-validation.md` |
| GSQ v3 MTP | Production ModelView physical-parent GPU oracle; real speculative HTTP200, accepted 2/2 | `66ca5c7`, GSQ report |
| GSQ v3 DFlash2 | 66 physical-parent device/source plane oracle, production ModelView; real speculative HTTP200, accepted 2/2 | `8c4f409`, GSQ report |
| GSQ v3 Vision | 333 real physical parents device oracle; one real 64×64 PNG with correct gradient text response, truncated at 24 tokens | `989457c`, `629dbdd` |
| Companion Bonsai2B v3 | 512-point Hadamard explicit per-model routing, 512/1024 GPU CPU oracle; BF16/INT8 sensible real requests including multi-turn, formally linked serve binary HTTP200 | `tools/companion/BONSAI2B_EXPERIMENTAL.md`, `d62128d`, `21cb76e` |
| Qwen3.6 27B v3 NVFP4 | 16.0 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; independent NVFP4/BF16 CUDA numerical tests | `docs/maintainer/2026-10-20-v3-qwen36-nvfp4-live-validation.md`, `4862f3a`, `dad6059` |
| Qwen3.6 27B v3 Groupwise | 15.3 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; Q4 v3 row-slice CUDA CPU oracle PASS | `docs/maintainer/2026-10-20-v3-qwen36-groupwise-live-validation.md`, `84576dd`, `071e539` |
| Strata × ActKV | Existing `BlockStratum` and Action-LRFU wiring confirmed. Standalone protection/decay/retrieval/cold tests PASS, plus real 1,898-token GSQ request, 512-token selected window and output `4`, HTTP200 | `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md`, `e495e19` |
| TP2 shards & proxy guards | V3 byte-exact sharder has tests; proxy now refuses a single physical GPU, mismatched ranks, bad ports, header-only v3 files; six preflight tests PASS | `tools/tp2/README.md`, `c2123ba` |
| Native TP2 communication/lockstep groundwork | Compiled optional NCCL BF16 sum/all-gather transport and two-rank engine admission/prefill/decode lockstep; three focused tests pass (including a mocked NCCL ABI and cross-process divergence detection). Neither native v3 rank execution nor real NCCL/two-GPU correctness has been verified | `src/runtime/tp2/`, `src/runtime/engine/tp_lockstep.h`, `tests/runtime/` |

## Still missing: do **not** declare full implementation complete

1. **Actual native v3 TP2 tensor-parallel GPU executor.**
   The new optional NCCL data-plane implementation issues real NCCL BF16
   all-reduce/all-gather calls when built with NCCL, and the host runtime now
   has two-process lockstep synchronization. On this host the NCCL development
   library is missing, so only the failure path and mocked NCCL call ABI were
   compiled/tested; no live NCCL operation was issued. The remaining central
   work is rank-local v3 mixed-GGUF physical ModelView binding, the TP2
   projection/reduction/head execution order, native target registration and
   graph capture interoperability. The proxy/sharder/preflight alone do not
   perform those steps. The
   reference `/home/ran/ninfer-work/ecosystem/ninfer-v100-tpx` has a v2
   `src/targets/qwen3_6_27b_tp2` implementation with `tp2_comm.cpp`,
   `tp2_mailbox.cu`, and `runtime/engine/tp_lockstep.h`. Porting must map the
   true v3 mixed-GGUF rank parent ModelViews, output/reduction sequences and
   graph capture. The user has only **one** CUDA GPU today, so any dual-rank
   native E2E performance/correctness claim requires a second real GPU.
2. **Broader Companion model coverage.** The separate Companion fork has
   the tested Bonsai2B patch but Gemma4/Flash-Next/Bonsai2 4B/9B/27B whole
   model weights are not currently full SHA-256 verified local `.ninfer`
   files. Prefix/header `.part` files are not complete models. Implement
   codecs/geometry and targeted oracle first, then trusted downloads and real
   inference when ready. **Do not commit/push** the Companion work tree;
   refresh tested/experimental patches in the main `tools/companion/`.
3. **Broader acceptance.** Vision video/multi-image, Groupwise/NVFP4 native
   MTP/DFlash2/vision combinations, long-context semantic accuracy and
   multi-turn cold-tier LRFU across compactions have not been completely
   validated by the narrow successful smoke tests above.

## Current protected state and next safe steps

- Main repository: `/home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada`,
  local `master`, push only to `origin master` with lower-case Conventional
  Commit messages. The pre-existing dirty
  `src/ops/linear/q4/q4_dispatch.cpp` **belongs to another session** and
  must not be overwritten, staged or reset. Never run `git reset`/`git clean`.
- Companion: `/home/ran/ninfer-work/ecosystem/ninfer-all`, baseline
  `b2913b6`, dirty by design, is an independent upstream fork. Do not
  commit or push it. The tested/experimental patch lives in main
  `tools/companion/bonsai2b-smallshape-experimental.patch`.
- Avoid repeating the entire 124-test suite or starting two Ninja processes
  in the same build directory. For native Ada, set `TMPDIR=$PWD/tmp-nvcc`
  before `cmake --build build --target <specific-target> -j2`.
- Preserve all other local modifications and `/home/ran/zatfung` unrelated
  files. Do not change GitHub issues #7/#4 or look for private HF credentials.
- Use a real two-GPU runner for the eventual TP2 collective acceptance, not
  an emulated proxy. Real absence of the second GPU is a verification blocker,
  not permission to claim success.

The full user objective remains open. This record is an implementation
checkpoint rather than a `session_finish` signal.
