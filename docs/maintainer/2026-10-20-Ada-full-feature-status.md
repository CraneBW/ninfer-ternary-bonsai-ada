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
| GSQ v3 Vision+MTP and Vision+DFlash2 | Two separate **real single-GPU** combined Vision/speculative PNG Chat Completions HTTP200; MTP accepted 12/20 draft tokens and DFlash2 accepted 11/23. The DFlash2 startup left only 53.8 MiB VRAM free; not a long-context or multi-image acceptance | `docs/maintainer/2026-10-20-v3-combined-vision-spec-single-gpu.md` |
| GSQ v3 multi-image and short MP4 input | One 2-image PNG Chat Completions HTTP200, accurate gradient/blue comparison; one 2-second H.264 `video_url` HTTP200 confirmed video-to-text pipeline, but response included an unsupported timer detail, so video understanding accuracy is NOT proven | `docs/maintainer/2026-10-20-v3-vision-multi-image-video-single-gpu.md` |
| Companion Bonsai2B v3 | 512-point Hadamard explicit per-model routing, 512/1024 GPU CPU oracle; BF16/INT8 sensible real requests including multi-turn, formally linked serve binary HTTP200 | `tools/companion/BONSAI2B_EXPERIMENTAL.md`, `d62128d`, `21cb76e` |
| Companion other model v3 configs | Compiled actual C++ config parsers against header-only Bonsai27B (64 layers/1024 Hadamard), Gemma4 (30 layers, MTP+Vision), and Flash-Next (48 layers, 512 experts, MTP+Vision); all PASS, but no corresponding full verified weights and no live inference | `tools/companion/HEADER_CONFIG_ORACLE.md`, `tools/companion/validate_cpp_v3_configs.cpp` |
| Qwen3.6 27B v3 NVFP4 | 16.0 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; independent NVFP4/BF16 CUDA numerical tests | `docs/maintainer/2026-10-20-v3-qwen36-nvfp4-live-validation.md`, `4862f3a`, `dad6059` |
| Qwen3.6 27B NVFP4 + MTP | Existing full file with 16.4 GiB managed weight allocation; real `--spec mtp` startup, HTTP200 `2`, accepted 2/2; only 74.2 MiB free device VRAM | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B NVFP4 + Vision | Native gradient PNG HTTP200, 16.3 GiB managed weights; 24-token response correctly broadly identifies gradient but took 32.3 s, with only 34.2 MiB free VRAM | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B v3 Groupwise | 15.3 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; Q4 v3 row-slice CUDA CPU oracle PASS | `docs/maintainer/2026-10-20-v3-qwen36-groupwise-live-validation.md`, `84576dd`, `071e539` |
| Qwen3.6 27B v3 Groupwise + MTP | Fixed previously unsupported independent W8 [17408,5120] gate/up v3 MTP shape with existing generic SIMT/MMA launchers; focused CUDA T=1,3,9,17 oracle PASS; full 15.7 GiB startup, MTP 2/2 accepted and answer `2`, HTTP200 | `docs/maintainer/2026-10-20-qwen36-groupwise-mtp-live-validation.md` |
| Qwen3.6 27B Groupwise + Vision | Native gradient PNG HTTP200 after correcting request model identity; accurate broad gradient description truncated at 24 tokens; 30.0 s request due to low free VRAM. Qwen3.6 Groupwise and NVFP4 artifacts do not contain DFlash2 physical component; DFlash2 startup correctly rejects them | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B Vision + MTP combinations | Native image+speculative combined generation HTTP200 on both Groupwise and NVFP4; respective MTP acceptance 2/2 and 4/6, correct short gradient yes/no answers; memory pressure remains severe | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Strata × ActKV | Existing `BlockStratum` and Action-LRFU wiring confirmed. Standalone protection/decay/retrieval/cold tests PASS, plus real 1,898-token GSQ request, 512-token selected window and output `4`, HTTP200 | `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md`, `e495e19` |
| 256K-class host Action-LRFU | Corrected exact top-fraction selection under tied retrieval scores; 4096-page CPU unit checks cover deterministic top-410, 256-page device working set, two action/query passes and temporal decay. No full 256K model inference claim | `src/kvmem/kvmem_blocks.h`, `tests/kvmem_blocks_test.cpp` |
| TP2 shards & proxy guards | V3 byte-exact sharder has tests; proxy now refuses a single physical GPU, mismatched ranks, bad ports, header-only v3 files; six preflight tests PASS | `tools/tp2/README.md`, `c2123ba` |
| Native TP2 communication/lockstep groundwork | Compiled optional NCCL BF16 sum/all-gather transport and two-rank engine admission/prefill/decode lockstep; three focused tests pass (including a mocked NCCL ABI and cross-process divergence detection). Neither native v3 rank execution nor real NCCL/two-GPU correctness has been verified | `src/runtime/tp2/`, `src/runtime/engine/tp_lockstep.h`, `tests/runtime/` |
| V3 TP2 rank-local metadata | The exact Qwen3.8 27B GSQ sharder produced rank0/rank1 directories directly from the verified source header. Native C++ manifest verified 656 core projection bindings and rank-specific Qwen3.8 ModelView inspector verified 963 text semantic tensors on each rank. No full shard copy or dual-GPU run | `src/runtime/tp2/v3_rank_manifest.cpp`, `src/targets/qwen3_6_27b_tp2/impl/load/bindings.cpp` |
| V3 rank physical materialization interface | Compiles a rank-local Binder/MaterializationPlan selecting 656 core logical weights' physical parents only once, plus GDN input-column mapping. Exposes CUDA logical matrix row views with real GGUF block-byte offsets and BF16 shape bounds. This stage is compile-verified only; no full shard upload or rank-local live inference has been demonstrated | `src/runtime/tp2/v3_rank_materialization.{h,cpp}` |
| Native TP2 registered model source | Distinct Qwen3.8 v3 rank target with half-width text geometries, v3 ModelView/materialization, NCCL warmup and rank partial-residual AllReduce; `ninfer-serve` compiles. MTP projections/sharded MLP, replicated Vision and DFlash2 feature paths now pass native v3 semantic inspection for both ranks (MTP 16, Vision 441, DFlash2 91, combined Vision+MTP and Vision+DFlash2); DFlash2 keeps its full unsharded FFN width. No live NCCL, GPU materialization of full shards, TP2 speculative/vision inference or combined correctness was demonstrated | `src/targets/qwen3_6_27b_tp2/`, `tests/runtime/test_tp2_v3_manifest.cpp` |

## Still missing: do **not** declare full implementation complete

1. **Native TP2 runtime acceptance, remaining feature combinations.**
   The rank-local target and direct NCCL collective execution source now
   compile and link into the service, using genuine v3 semantic bindings and
   half-width text operators. On this host the NCCL development library is
   missing and the available GPU count is one. No live NCCL collective,
   full shard CUDA upload, graph-capture interoperability, numeric comparison,
   or full TP2 request has been demonstrated. The code guards unavailable
   NCCL instead of falling back to two non-communicating single-GPU servers.
   Vision/MTP/DFlash2 rank feature semantics have been bound/shape-validated
   and their startup selection is now wired, but still lack TP2 end-to-end
   and combined numerical verification. The
   reference `/home/ran/ninfer-work/ecosystem/ninfer-v100-tpx` has a v2
   `src/targets/qwen3_6_27b_tp2` implementation with `tp2_comm.cpp`,
   `tp2_mailbox.cu`, and `runtime/engine/tp_lockstep.h`. Porting must map the
   true v3 mixed-GGUF rank parent ModelViews, output/reduction sequences and
   graph capture. The user explicitly requires **single-GPU-only testing**;
   do not launch a dual-rank inference test, and do not claim dual-rank E2E
   correctness or performance without such evidence.
2. **Broader Companion model coverage.** The separate Companion fork has
   the tested Bonsai2B patch but Gemma4/Flash-Next/Bonsai2 4B/9B/27B whole
   model weights are not currently full SHA-256 verified local `.ninfer`
   files. Prefix/header `.part` files are not complete models. Implement
   codecs/geometry and targeted oracle first, then trusted downloads and real
   inference when ready. **Do not commit/push** the Companion work tree;
   refresh tested/experimental patches in the main `tools/companion/`.
3. **Broader acceptance.** Beyond the initial multi-image and short MP4
   smoke tests, high-resolution/long video semantics, Groupwise/NVFP4 native
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
- User-directed verification scope is **single GPU only**. Do not schedule
  actual two-GPU inference tests or emulate two GPUs on the one available card;
  native two-rank numerical behavior therefore remains unverified.

The full user objective remains open. This record is an implementation
checkpoint rather than a `session_finish` signal.
