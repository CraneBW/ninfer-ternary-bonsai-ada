# NInfer Ada full-feature work: verified state and remaining work

Latest detailed performance round: existing Qwen3.8 original213K12/12 and current
P4 54/54 calls pass. Short213K TTFT medians are20.701s sync /18.963s async;
10240history tokens per arm are avoided,63/63NVMe read-ahead tasks complete,
and large Host transactions split into bounded sub-batches. SM89 MMQ uses512
columns with the same CUDA kernel, checked against independent CPU decoding.
P4 deep async gain0.542% misses10%, so sync stays default. One full-model NCU
capture caused global OOM; it is preserved and replaced with small-Op profiling
and hard-limited transient jobs. See the [live tuning report](2026-10-20-kvmem-actkv-live-validation.md#详细性能调优ttftnvme读前sm89-mmq与当前p4).

Earlier functional closure: existing Qwen3.8 passes44/44 calls: managed64K2/2,
original W4K/physical8K/32K12/12, retained A→B→A branch6/6, current
Graph/MTP3/DFlash2 short semantics12/12 and frozen W32K/physical36K/256K12/12.
The256K prompts contain261411–261576tokens; each mode has468compactions,
3099Cold spills and zero terminal in-flight resources. Managed runtime now has
an explicit production budget, incomplete compressed Forks safely rebuild from
root, and source/memory/internal-replay metrics are exposed.
Original213K12/12 evidence is retained without rerunning that matrix.
That phase deferred TTFT/flow/P4; the later detailed performance round above
has now run. Sync remains default. See the [current live record](2026-10-20-kvmem-actkv-live-validation.md#本轮最终功能结果).

2026-10-20, native host `/home/ran/ninfer-work`, one RTX 4070 Ti SUPER 16 GiB.
The latest instruction is **validate existing complete models only**, without
new model downloads. Qwen3.8 remains primary for long-context, media and P4. The three 27B model files and Bonsai2B below are already full
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
| GSQ v3 multi-image, high-resolution and bounded video | One 2-image PNG Chat Completions HTTP200, accurate gradient/blue comparison; one 2-second H.264 `video_url` HTTP200 confirmed video-to-text pipeline. Original-pixel audit confirms the reported timer really exists, correcting the earlier annotation; broader video understanding accuracy remains unqualified. Later decoder fixes passed host tests. New six-call media qualification passes 1536×1024 panels, two images/cross-turn reuse, 60-second/120-frame temporal colors, and high-resolution MTP/DFlash2; actual 1536/960 vision-token processing is recorded | `docs/maintainer/2026-10-20-v3-vision-multi-image-video-single-gpu.md` |
| Companion Bonsai2B v3 | 512-point Hadamard explicit per-model routing, 512/1024 GPU CPU oracle; BF16/INT8 sensible real requests including multi-turn, formally linked serve binary HTTP200 | `tools/companion/BONSAI2B_EXPERIMENTAL.md`, `d62128d`, `21cb76e` |
| Companion other model v3 configs | Compiled actual C++ config parsers against header-only Bonsai27B (64 layers/1024 Hadamard), Gemma4 (30 layers, MTP+Vision), and Flash-Next (48 layers, 512 experts, MTP+Vision); all PASS, but no corresponding full verified weights and no live inference | `tools/companion/HEADER_CONFIG_ORACLE.md`, `tools/companion/validate_cpp_v3_configs.cpp` |
| Qwen3.6 27B v3 NVFP4 | 16.0 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; independent NVFP4/BF16 CUDA numerical tests | `docs/maintainer/2026-10-20-v3-qwen36-nvfp4-live-validation.md`, `4862f3a`, `dad6059` |
| Qwen3.6 27B NVFP4 + MTP | Existing full file with 16.4 GiB managed weight allocation; real `--spec mtp` startup, HTTP200 `2`, accepted 2/2; only 74.2 MiB free device VRAM | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B NVFP4 + Vision | Native gradient PNG HTTP200, 16.3 GiB managed weights; 24-token response correctly broadly identifies gradient but took 32.3 s, with only 34.2 MiB free VRAM | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B v3 Groupwise | 15.3 GiB, warmup PASS, native HTTP200 outputs `hello` and `2`; Q4 v3 row-slice CUDA CPU oracle PASS | `docs/maintainer/2026-10-20-v3-qwen36-groupwise-live-validation.md`, `84576dd`, `071e539` |
| Qwen3.6 27B v3 Groupwise + MTP | Fixed previously unsupported independent W8 [17408,5120] gate/up v3 MTP shape with existing generic SIMT/MMA launchers; focused CUDA T=1,3,9,17 oracle PASS; full 15.7 GiB startup, MTP 2/2 accepted and answer `2`, HTTP200 | `docs/maintainer/2026-10-20-qwen36-groupwise-mtp-live-validation.md` |
| Qwen3.6 27B Groupwise + Vision | Native gradient PNG HTTP200 after correcting request model identity; accurate broad gradient description truncated at 24 tokens; 30.0 s request due to low free VRAM. Qwen3.6 Groupwise and NVFP4 artifacts do not contain DFlash2 physical component; DFlash2 startup correctly rejects them | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Qwen3.6 27B Vision + MTP combinations | Native image+speculative combined generation HTTP200 on both Groupwise and NVFP4; respective MTP acceptance 2/2 and 4/6, correct short gradient yes/no answers; memory pressure remains severe | `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md` |
| Strata × ActKV | Existing `BlockStratum` and Action-LRFU wiring confirmed. Standalone protection/decay/retrieval/cold tests PASS, plus real 1,898-token GSQ request, 512-token selected window and output `4`, HTTP200. The original exact-label failure is closed. Final unchanged eight-case panel is 8/8, including paraphrase/update/tool/cross-page; original roughly 5K-token deep history is 6/6 facts in full/sync/async. Complete-query/profile timing, old-version soft priority and empty growth-page selection were repaired; no cold-tier semantic acceptance is implied | `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md`, `e495e19` |
| Functional gap closure and256K frozen facts | 44/44 current calls correct; managed budget, complete-history Fork admission, source/replay telemetry; W4K/32K12/12 and W32K/256K12/12. Closed before the later tuning round | `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md` |
| Original213K frozen facts, W32K | Current base/no-Graph sync/async12/12 correct complete calls, identical usage/selection,2471Cold spills per arm, terminal in-flight0. Historical replay TTFT23–26s; later tuning reaches20.701/18.963s and refreshes base P4, without broader long combinations | `docs/maintainer/2026-10-20-213k-state-replay-repair.md` |
| 256K-class host Action-LRFU and cold extent growth | Corrected exact top-fraction selection under tied retrieval scores; 4096-page CPU unit checks cover deterministic top-410, 256-page device working set, two action/query passes and temporal decay. ColdKVStore free-list now grows geometrically; 4096-page cold allocation/reuse checks pass. No full 256K model inference claim | `src/kvmem/kvmem_blocks.h`, `src/kvmem/kvmem_cold_store.h`, `tests/kvmem_blocks_test.cpp`, `tests/kvmem_cold_store_test.cpp` |
| TP2 shards & proxy guards | V3 byte-exact sharder has tests; proxy now refuses a single physical GPU, mismatched ranks, bad ports, header-only v3 files; six preflight tests PASS | `tools/tp2/README.md`, `c2123ba` |
| Native TP2 communication/lockstep groundwork | Compiled optional NCCL BF16 sum/all-gather transport and two-rank engine admission/prefill/decode lockstep; three focused tests pass (including a mocked NCCL ABI and cross-process divergence detection). Neither native v3 rank execution nor real NCCL/two-GPU correctness has been verified | `src/runtime/tp2/`, `src/runtime/engine/tp_lockstep.h`, `tests/runtime/` |
| V3 TP2 rank-local metadata | The exact Qwen3.8 27B GSQ sharder produced rank0/rank1 directories directly from the verified source header. Native C++ manifest verified 656 core projection bindings and rank-specific Qwen3.8 ModelView inspector verified 963 text semantic tensors on each rank. No full shard copy or dual-GPU run | `src/runtime/tp2/v3_rank_manifest.cpp`, `src/targets/qwen3_6_27b_tp2/impl/load/bindings.cpp` |
| V3 rank physical materialization interface | Compiles a rank-local Binder/MaterializationPlan selecting 656 core logical weights' physical parents only once, plus GDN input-column mapping. Exposes CUDA logical matrix row views with real GGUF block-byte offsets and BF16 shape bounds. This stage is compile-verified only; no full shard upload or rank-local live inference has been demonstrated | `src/runtime/tp2/v3_rank_materialization.{h,cpp}` |
| Native TP2 registered model source | Distinct Qwen3.8 v3 rank target with half-width text geometries, v3 ModelView/materialization, NCCL warmup and rank partial-residual AllReduce; `ninfer-serve` compiles. MTP projections/sharded MLP, replicated Vision and DFlash2 feature paths now pass native v3 semantic inspection for both ranks (MTP 16, Vision 441, DFlash2 91, combined Vision+MTP and Vision+DFlash2); DFlash2 keeps its full unsharded FFN width. No live NCCL, GPU materialization of full shards, TP2 speculative/vision inference or combined correctness was demonstrated | `src/targets/qwen3_6_27b_tp2/`, `tests/runtime/test_tp2_v3_manifest.cpp` |

## Latest existing-model qualification

No new weights were downloaded. Existing Qwen3.6 Groupwise, NVFP4 and Companion
Bonsai2B each pass two-turn ACK/37 recall. Qwen3.8 passes six bounded media
calls: 1536×1024 panels, two images and reuse, 60-second/120-frame color order,
and high-resolution MTP3/DFlash2. Broader OCR/arbitrary-media semantics remain
unqualified.

Earlier32K/64K qualifications passed five frozen facts in both modes, including
64K Cold; their exact receipts remain historical. The later original213K
sequence now passes all six calls in both modes with the same32K window.
The newest functional round also passes the original W4K/physical8K/32K
counterexample12/12 and frozen W32K/physical36K/256K12/12. Their old failures
remain preserved. The extra full-KV-failing ambiguous semantic control was not
rerun or relabeled. These are finite fact-set qualifications.

The older P4 deep gain0.92% predates immutable-source/canonical replay. The
current detailed-tuning binary passes18/18sessions,54/54correct requests with
deep paired gain0.542%, below10%; **sync remains default**.
See the live validation's “Existing-model qualification” section for exact
lengths, resources, scope, all failures and the predeclared stop decisions.

## Still missing: do **not** declare full implementation complete

Implementation update: Qwen3.8 GSQ is the primary validation model. The previously recorded roughly 3K/5K single-GPU recall counterexamples
are closed in their original bounded inputs: final eight-case panel **8/8**, and fixed deep history **6/6
facts per full/sync/async mode**. Final compressed deep replay has 14/14 complete
SSE streams with matching content/usage; a newly exposed terminal checkpoint
abort was fixed by rolling back ahead-mapped unwritten pages before selection.
Current content retrieval is separate from history profile; a no-lexical
complete-query restore precedes the assistant opener without replaying GDN.
Old equal-version hot pages no longer refill soft slots ahead of the new fact,
and rarer subject anchors rank ahead of repeated question-template words.

The unchanged semantic pair also passes base, Graph, MTP3/Graph and
DFlash2-K3/Graph: 16/16 calls, equal sync/async token IDs, reuse and commit,
2933-token second-turn reuse and zero terminal in-flight pages. Prior cross-page/
tool P3 and default-capture evidence remains valid. This whole issue-closure
follow-up preserves 147 attempts, 145 completed calls and two fixed terminal
aborts; earlier four wrong completed answers remain in the ledger. These are
small synthetic roughly 3K/5K cases, not full long-context/media acceptance.
Details and closure ledger: `2026-10-20-kvmem-actkv-live-validation.md`.

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
3. **Broader acceptance.** Beyond the bounded panel-color/multi-image/60-second video tests,
   high-detail OCR and arbitrary high-resolution/long video semantics, Groupwise/NVFP4 native
   MTP/DFlash2/vision combinations, long-context semantic accuracy and
   multi-turn cold-tier LRFU across compactions have not been completely
   validated by the bounded tests above. The previously open exact-label,
   paraphrase, update, tool and six-fact fixed-history counterexamples are now
   closed. This does not qualify arbitrary32K–256K documents or unrestricted
   Host→Cold semantics; the current finite213K facts are qualified separately above.

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

The broader model-family and unrestricted media objective remains unqualified.
The current bounded task uses existing artifacts and records the actual tested
length/configuration frontier, rather than treating future coverage as passed.

## 2026-10-20 implementation checkpoint: Qwen3.8 first

Known-fault closure is complete through the original P3 integrations. The final
unchanged recall panel is 8/8, the deep-history full/sync/async controls each
answer all six facts, and the final two compressed streams finish all 14 calls.
The native semantic matrix adds 16/16 calls across base/Graph/MTP3/DFlash2 with
identical sync/async token IDs, commit/reuse and terminal resource accounting.
The existing cross-page/tool matrix and default shared-capture repair are
retained. Failed candidates, including two correct-text-but-incomplete streams,
remain in the resolve ledger; no failure is removed from the denominator.

Bounded D2H/H2D, actual-free arbitration, planned rephase workspace and terminal
transfer draining remain implemented. The latest repair reuses the existing
rollback to exclude unwritten growth pages from historical selection. Vision
and concurrency>1 still reject async; speculative output remains bounded by its
funded tail. TP2 stays paused and missing model artifacts are prerequisites.
No new download, commit or push was performed.

Historical P4 decision (before immutable K and canonical replay): the refreshed 18-session/54-request base/no-Graph comparison
completed with all expected answers correct and matching sync/async outputs and
usage. Median paired deep-session wall reduction is 0.92%, first overpool 2.90%,
shallow -0.32%. **Sync stays default; async remains experimental.** No depth=2
or extra deployment-performance pairs were triggered. The earlier 1.50% run,
whose deep recall was only 1/6, remains historical in `docs/performance.md`.
Neither performance result establishes unrestricted long-context recall. The
latest32K-window original213K facts pass12/12; the subsequent performance round
reduces measured short-query TTFT to20.701/18.963s and runs current P4 (0.542%).
The frozen256K facts passed12/12 in the earlier functional round and were not rerun during tuning.
The live functional record is the current boundary.
