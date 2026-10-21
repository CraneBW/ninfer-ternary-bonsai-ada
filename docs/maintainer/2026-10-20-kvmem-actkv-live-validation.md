# KVMem Strata × Action-LRFU real native GSQ v3 smoke

> 持续调优补充：已采用项目`ninfer-perf-tuning`经验，补齐完整尺寸CPU oracle，
> 并将约1%的顺序测量差异降为筛选线索。更大MMQ批次与共享激活候选均未转正，生产实现恢复。
> 现有Graph在同一二进制、三对交替的限定夹具上有decode收益；详细证据和原生执行含义见文末新节。

> 最新性能轮：原213K六题sync/async12/12及当前P4的54/54请求正确完整。
> 五个短问题TTFT中位数sync20.701s、async18.963s；每臂少重算10240历史tokens。
> NVMe读前63/63，大批拆分622子批，终态在飞0；SM89复用MMQ512批宽已通过CPU oracle。
> P4异步深轮仅0.542%，未过10%，sync仍默认。曾有1次整模型NCU采集global OOM；
> 已停用该采集方式，后续任务使用独立内存限额，失败和旧原题证据保留。详情见文末。

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

## Qwen3.8 content-only lexical recall implementation

Implementation resumed on 2026-10-20 using the existing Qwen3.8 GSQ artifact,
one RTX 4070 Ti SUPER and the original 512-token working-set configuration.
The new lexical standalone test is now called by `main()`, covers the actual
128/129-token limit, common-only queries, cross-page values, duplicate/invalid
candidate IDs and protected-budget priority, and passes (`failures=0`).

Native chat rendering and tokenization independently reproduce the original
2,968-token prompt, query span `[2939,2961)` and historical pages 16/17. Page 16
contains the complete `cobalt-orchid-73` fact. The original requests now answer
`ACK` then `cobalt-orchid-73`, with final selected pages
`0,16,17,42,43,44,45,46` and 2,933 cached prompt tokens.

The first draft also matched serialized role/control tokens as rare evidence:
questions without a relevant fact name could spuriously nominate pages 0/1.
The frontend now supplies separately encoded **message-content-only** query
tokens to lexical matching. The existing semantic query span is unchanged.
Native audits now return no lexical candidates for the paraphrase and arithmetic
controls. Media inputs and serialized queries longer than 128 tokens keep this
supplement disabled; selection uses it only with Action-LRFU, after protected
prefix/mandatory/action/recent pages and within the existing budget.

A frozen eight-case synthetic panel used the same model, greedy decoding,
INT8 KV, no thinking and no CUDA Graph. Full KV ran each complete second-turn
history once; both compressed arms ran the two turns. This totals **40 HTTP
requests**, plus the earlier two-request original-case pilot (42 new requests
in this implementation stage). No model request failed or was silently retried.
A driver startup with an invalid full-KV-only option failed before sending any
request; its command and log remain in the evidence directory.

| Case | Full KV | Previous compressed policy | Content-only lexical policy |
|---|---|---|---|
| Original exact label | Correct | Wrong | Correct |
| Early historical label | Correct | Wrong | Correct |
| Late historical label | Correct | Correct | Correct |
| Label crossing a page boundary | Correct | Wrong | Wrong |
| Semantic paraphrase | Correct | Wrong | Wrong |
| Updated fact with the same name | Correct | Wrong | Wrong |
| Historical long tool result | Correct | Wrong | Wrong |
| Arithmetic/no lexical candidate | Correct | Correct | Correct |

Thus this panel is **8/8 full KV, 2/8 old compressed, 4/8 candidate**, with no
new failure among the two previously correct compressed cases. These are small
synthetic engineering cases, not an estimate of general recall accuracy.

The cross-page case selects pages 16/17 containing the value but still answers
incorrectly; candidate inclusion therefore does not prove answer correctness.
These were **round-end** selections, not proof of visibility while computing
the question. The pre-query timing fix and updated results are recorded below.
The updated fact also reaches that round-end window but is not correctly answered.
The long tool-result case nominates pages 16/17 but retains only
`0,40,41,42,43,44,45,46`: the protected last call/result span consumes the
working-set quota before lexical candidates. The candidate does not override
those protections. At this checkpoint these remained quality gaps; no weights or special strings were
added to force those cases through, and the frozen cases were not changed.

The supplemental policy is retained for the bounded short-text Action-LRFU
domain. This is an original-regression fix and a limited improvement, **not**
256K/multi-turn semantic acceptance or equivalence to full KV. Async transfer
work must compare the same selected windows and keep these failures separate.

Evidence under `tmp-nvcc/`:

- `p0-20261020-cpu.log`, `p0-20261020-build-content-only.log`
- `p0-20261020-original-token-audit.json`
- `p0-20261020-recall-panel/manifest.json` (frozen requests and hashes)
- `p0-20261020-recall-panel/content-only-audits.json`
- `p0-20261020-recall-panel/results.json`, per-arm commands/logs/responses
- `p0-20261020-panel-run-v2.log`, `p0-freeze-panel.py`, `p0-run-panel.py`

The earlier manifest's token audit reflects the initial framed-query draft;
`content-only-audits.json` records the corrected policy on the same frozen
requests. No complete model was downloaded or hashed again. The external
`q4_dispatch.cpp` was not edited; its implementation-start SHA-256 was
`d0a14eaa1f097b3e964589af8215d331af1741394f3932b3493293af786efc5a`.

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

## 2026-10-20 P1: bounded background D2H on Qwen3.8

The implementation uses one independently owned completion event per batch,
source-ready ordering from compute to the transfer stream, pinned immutable
sources, and reserved Host targets. Retirement verifies the complete batch
before publishing Host slots and releasing Device pages. The default remains
`--kvmem-transfer-mode sync`; the experimental `async` mode is explicitly opt-in.

The focused CPU/options and native CUDA checks passed. The delayed-copy check
keeps two Device sources and Host targets allocated while completion is blocked;
it rejects destructive epoch changes, checks a later batch against its own event,
and verifies cancellation drains the copy before address-generation reuse.
Position-free V code/scale data remain byte-exact; moved K uses the existing
independent CPU H256/re-RoPE oracle. Lazy kernel module loading initially consumed
the deliberate gate timeout; only these controlled-delay tests set
`CUDA_MODULE_LOADING=EAGER`. Failed development runs remain in `tmp-nvcc/`.

Same-binary Qwen3.8 GSQ replay of the unchanged original two requests:

| Mode | First response | Exact historical label | Submitted / completed | In-flight at end |
|---|---|---|---:|---:|
| sync | `ACK` | `cobalt-orchid-73` | 0 / 0 | 0 |
| async | `ACK` | `cobalt-orchid-73` | 10 / 10 | 0 |

Async copied 82,182,144 payload bytes, recorded 978 microseconds of D2H submission
wall-time and 0 exposed retirement-wait microseconds, with two synchronous
fallbacks. These short correctness replays do not establish a speedup. Initial
support is one active text request, base/no-Graph, with H2D and Cold work still
synchronous at this checkpoint. Later P2/P3 records supersede only that scope.

Evidence: `p1-20261020-{targeted-tests,second-batch-tests}.log`,
`p1-20261020-original-{sync,async}.{command.json,log,metrics.txt}`, response JSONs,
and `p1-20261020-original-results.json` under `tmp-nvcc/`.
Submission-error cleanup and all-or-nothing publication are implemented; this
record does not claim a forced fatal CUDA-context-error experiment or a successful
publication of a mutated pinned source. Pins deliberately refuse such mutation.

## 2026-10-20 P2/P3: early H2D, Graph and speculative pools

H2D reuses the existing destination-pin / publish / abort-reservation primitives.
Its targets are private until copy completion; compute rephases disjoint resident
pages while transfer restores history, then rephases restored pages before table
publication. A compute→transfer event also protects Device slots reused after
D2H. Required pages wait if no independent work remains. Cold retains the existing
synchronous staging semantics and format. The typed 16 MiB bound applies in both
directions; actual free capacity excludes all in-flight sources and reservations.

Base generation now starts reclamation with half its configured growth reserve
still unused, capped in typed payload bytes; sync/async share that watermark and
selection. Rephase Device metadata belongs to the persistent layout (12 bytes per
logical page, shared sequentially by text/backend); each store owns the same-sized
startup pinned Host metadata. No per-compaction cudaMalloc/free remains.

The existing native byte oracle passed on transfer-stream gather/scatter with
PageMajor, HeadMajor and asymmetric code/scale layouts. The INT8 H256 compaction
oracle passed sync, delayed D2H/H2D, cancellation and a second independent batch.
A full-pool case proved a promise cannot be allocated and growth waits for actual
retirement. Existing `--claim` and `--cold` cases passed, including Host→NVMe→GPU
and release/reuse; final native checks verify all Device/Host/in-flight storage
returns to zero. Evidence: `p2-20261020-{native-tests,claim,cold}.log` and
`p3-20261020-native-final.log` under `tmp-nvcc/`.

Same frozen native-token inputs, INT8 KV, 3×24-token overpool requests per arm,
then existing speculative page-boundary and queued-cancellation/follow-up fixtures:

| Qwen3.8 GSQ route | Output token IDs / committed decode / reuse | Async submitted = completed | H2D batches | Final in-flight |
|---|---|---:|---:|---:|
| Base, no Graph | identical | 30 | 5 | 0 |
| Base, Graph | identical | 30 | 5 | 0 |
| MTP3, Graph | identical, including accepted/drafted counts | 40 | 0 | 0 |
| DFlash2 K3, Graph | identical, including accepted/drafted counts | 20 | 0 | 0 |

MTP per-request acceptance was 16/18, 15/21 and 16/18; DFlash2 was 17/17,
16/18 and 17/17 in both arms. The reused page-boundary fixture exercises a stop
at the committed boundary, partial speculative settlement, rejected/unused tails
and subsequent request reuse. MTP and DFlash2 share no backend history payload:
MTP has its own full-context/provisional pool, while DFlash2 cyclic state remains
inside the existing StateImage path. No Vision support is claimed; async startup
with Vision or concurrency=2 was explicitly rejected.

Two sync-MTP failures were preserved before the compatibility fix: the backend
mistakenly inherited Main's selected-window cap, and Main had not charged room
for growth/provisional verification. The fix preserves the exact entitlement
check, gives MTP its independent physical claim and funds Main's extra pages
from real pool capacity. The selected history budget is unchanged. If generation
exhausts this funded speculative tail, the existing live-geometry restriction
still yields a controlled error; arbitrary-length speculative decode is not
claimed. Source comparison reused the ring device-footprint contract and
ninfer-all's resource-dimension diagnostic, recorded in the reference assessment.

The final pending batches now retire before request-checkpoint publication, so
no transfer from a completed request is left unaccounted. The original MTP probe
showing one final in-flight page remains preserved and is superseded by the final
40/40, zero-in-flight run. These are correctness/lifetime results, not performance
results. Evidence: `p3-20261020-{base,graph}-results.json`,
`p3-20261020-{mtp,dflash2}-final-results.json`, matching logs, and
`p3-20261020-reject-{concurrency,vision}.log`.

## 2026-10-20 P4 decision and handoff

The frozen 18-session comparison completed 54/54 requests, no operational
failures/retries. Median paired wall reductions: shallow -0.10%, first overpool
2.90%, seven-turn deep 1.50%. Outputs and transfer shapes match; end in-flight
and Cold bytes are zero. The deep gain misses the 10% gate, so sync remains the
default, depth stays one and no extra Graph/MTP deployment performance run was
started. Full scope and timings are recorded in `docs/performance.md`.

Deep semantic failures remain open: each arm/session answered only 1/6 label
questions correctly (plus ACK, giving 2/7 requests). These are separate from
byte/position/transfer transparency. The P0 frozen panel's four remaining
failures and this additional fixed corpus stay available; no tuning or model
switch was used to hide them. Later long-history qualification must address
recall before expanding to 32K/64K/256K or claiming new semantic coverage.

Current handoff: Qwen3.8 remains the preferred model; bounded async D2H/H2D and
base/Graph/MTP3/DFlash2-K3 correctness are implemented, with explicit Vision /
multi-active-request rejection and funded-tail limitations. Preserve single-batch
and sync-default settings. TP2 remains paused. Work is uncommitted; protected
external `src/ops/linear/q4/q4_dispatch.cpp` remains untouched. The user's
source-reuse-first principle is recorded in workspace `AGENTS.md`; check related
KVMem/ActKV/Strata/Sync/gguftoninfer source before designing another subsystem.

Final-binary stability confirmation reused only the frozen original, late and
no-lexical cases (six requests). Answers remained `cobalt-orchid-73`,
`silver-lotus-42` and `4`, with ACK first turns; the owned server exited 0.
This checks the final watermark/allocator changes against the original regression
and the two old-correct cases, without repeating the whole quality panel or
adding performance samples. The initial P0 40-request panel plus two pilot
requests and these six confirmations remain within 48 requests. Evidence:
`final-20261020-recall-stability-{run.log,results.json}` and per-turn receipts.
The affected build/native checks, `git diff --check` and strict OpenSpec validation
passed. No full-suite campaign, extra model download, commit or push was run.


## 2026-10-20 P5: restore lexical evidence before computing the question

The next source comparison found a timing gap: the old lexical trace nominated
correct pages, but the completed window was reselected only at round end when
the short second-turn suffix fit the resident window. That final selection did
not make the evidence visible to the current question. Both cross-page and
updated-fact failures ended with a stop token, not an output-length limit.

Reference `ecosystem/kvmem/backends/patches/ninfer-kvmem.patch` at
`faf3c22582adf9d99ff072c543ea30ca03a8e2ca` restores selected history before query
replay; ActKV's selector retains mandatory/protected ordering. This implementation
uses that sequencing with the existing lexical candidates: once all candidate
pages have been written, an exclusive movable window with missing Host/Cold
evidence can reselect through existing prefill compact, H2D and re-RoPE **before
question prefill**, even without capacity pressure. The incoming piece still
has space within the same 512-token claim. Protection priorities are unchanged.
There is no new generic query replay/checkpoint system or model-specific string.
Fixed rewrite/endpoint and shared geometry are not forcibly moved; absent
candidates take the existing path. Trace `missing_after` distinguishes an
attempt from successful admission when protections consume the quota.

The frozen P0 request bytes and hashes were reused on Qwen3.8 GSQ, greedy,
INT8 KV, no thinking/Graph/speculation, physical pool 1024 tokens, selected budget
512, generation reserve 128, prefill chunk 256 and Host capacity 128 MiB.
Prior full-KV and compressed controls were reused, not rerun. The follow-up
manifest limited execution to 16 sync requests (eight two-turn cases) plus two
async requests for the same cross-page case, without retries or added cases.

| Case | Reused full KV | Initial content-only policy | Pre-query restore |
|---|---|---|---|
| Original exact label | Correct | Correct | Correct |
| Early historical label | Correct | Correct | Correct |
| Late historical label | Correct | Correct | Correct |
| Label crossing a page boundary | Correct | Wrong | Correct: `violet-cinder-58` |
| Semantic paraphrase | Correct | Wrong | Wrong: no lexical candidate |
| Updated fact with the same name | Correct | Wrong | Correct: `copper-lilac-28` |
| Historical long tool result | Correct | Wrong | Wrong: protected action consumes quota |
| Arithmetic/no lexical candidate | Correct | Correct | Correct |

Thus the same small panel advances **4/8 to 6/8**, with no previously correct
case lost. Across the historical policy stages it is 2/8→4/8→6/8; full KV remains
8/8 on its preserved control. This is not a general accuracy estimate.
All **18/18 requests** completed. The async cross-page replay returned the same
content, finish reasons and prompt/cache/completion counts as sync; both owned
servers exited 0. No P4 performance rerun was performed, so its 1.50% result
belongs to the earlier binary. Sync remains default; no new speedup is claimed.

For the cross-page case, trace selects `0,16,17,42,43,45,46` with a 448-token
selection cap before the 35-token suffix, at history cursor 2962 < query begin
2968. Its valid packed frontier is 402. The update selects
`0,37,38,39,43,45,46` at cursor 2957 < query begin 2963, before its 36-token
suffix. Both full values are now in the input view before question computation.
The tool-result case still excludes 16/17 before its question, preserving the
protection failure rather than bypassing it.

Only `ninfer-serve` and the affected compact-test target were rebuilt. The
existing `--claim` regression now supplies lexical evidence on Host/Cold while
there is no generation-pressure trigger, restores it with a competing pool
reservation, and checks that both pages become Device-visible. It passes with
all physical/Cold resources released. After the model panel a conservative
shared-view guard and the extra trace field were added; those received the
affected build and capacity regression, without another model replay.

Evidence: `tmp-nvcc/p5-20261020-prequery-recall/` contains the frozen manifest,
commands, raw responses, `run.log`, per-mode trace logs, `results.json`, derived
`analysis.json`, build logs and `claim.log` / `final-claim.log`. The driver is
`tmp-nvcc/p5-run-prequery-recall.py`; original P0 requests/controls remain intact.
No commit/push or model download was performed; the external Q4 file was not
edited. Next quality work is the paraphrase retrieval signal and the completed
historical tool action's protection scope; long-context escalation remains gated.


## 2026-10-20 P3 closure for the latest chat recall path

The user requested continued implementation through the original P3 boundary
(Graph/MTP/DFlash2). This follow-up fixes current-user tool protection and tests
the latest pre-query restore through real chat preparation. It does not start a
new P4 performance campaign or expand to other models, Vision or TP2.

Changes and source reuse:

- ActKV's last-user query/retrieval suffix (`ecosystem/melis-actkv`,
  `tools/llama-kvmem-server.cpp`, and `docs/kvmem-chat-tools-plan.md` T5) guided
  the current-action boundary. Active tool call/adjacent results remain
  protected; after a new user message the completed tool action returns to
  ordinary retrievable history. Existing frontend tests cover both transitions.
- MTP now consumes Main's **exact selected logical page IDs**, not its own
  independently ranked window with merely the same length. Typed payloads,
  reservations and transfers remain independent. Across the two accepted MTP
  arms, trace contains **62 paired Main/MTP selections**, all identical.
- The real chat reuse exposed a dimensional error hidden by raw-token tests:
  MTP checkpoint readiness compared compacted backend coverage to the original
  roughly 3K-token ledger frontier. It now compares to `kv_frontier`; private
  DFlash readiness uses the same coordinate contract.
- The ring fork's `LOCAL FIX (mtp-bridge mapping)` in
  `src/models/qwen3_5/program/prefill.cpp` was adapted directly: map before
  bridge writes. Both bridge branches now use compacted cache/RoPE positions
  and publish the corresponding backend frontier. Input slicing/history matching
  remain in original ledger coordinates. The accepted MTP chat case exercises
  the BeforeSuffix bridge; this is not an exhaustive bridge geometry sweep.
- The first native fixture attempt used Engine-default automatic shared
  captures, while the original OpenAI request disables inferred captures in
  `serve/openai_common.cpp`. The fixture was aligned to that original protocol.
  The concrete default-capture failure was also fixed: an early shared capture
  must defer full long-prompt mapping to the existing bounded prefill loop,
  following `start_sequence`, rather than map all 2969 tokens into an eight-page
  claim. A separate four-call sync/async default-capture regression completed;
  both exact-label answers happened to be correct too. Shared/fixed geometry
  still has no blanket forced lexical-recall guarantee.

The immutable P0 cross-page and long-tool requests are preserved under
`tests/fixtures/kvmem_recall/`, with the same bytes/hashes and full-KV controls.
No new question or larger answer limit was introduced. Configuration is Qwen3.8
GSQ on the same single Ada GPU, managed weights, max context 3072, Main physical
pool 1024, working budget 512, generation reserve 128, chunk 256, INT8 KV,
LRFU lambda 0.5/top fraction 0.3, greedy and no thinking. MTP/DFlash2 use width 3.

| Path | Accepted matrix calls | Historical answers in each mode | Async batches completed/submitted | H2D batches | Final in-flight |
|---|---:|---|---:|---:|---:|
| base/no-Graph | 8 | `violet-cinder-58`, `teal-jasmine-61` | 38/38 | 5 | 0 |
| base/Graph | 8 | same exact labels | 38/38 | 5 | 0 |
| MTP3/Graph | 8 | same exact labels | 62/62 across typed pools | 2 | 0 |
| DFlash2-K3/Graph | 8 | same exact labels | 31/31 | 1 | 0 |

The accepted matrix is **32/32 completed calls**, comprising 16 ACK turns and
16 correct historical-label answers. Within every path sync/async token IDs,
content, finish reasons, committed-decode and reuse counts agree. The cross-page
second turn reuses **2962 prompt tokens** in all modes; the long-tool replay is
a root/cache-miss path. MTP drafts were actually verified (ACK 3/3 and each
label 6/6). DFlash2 exercised rejection/partial acceptance (ACK 1/3 and each
label 5/8). These short examples do not establish long-generation or general
semantic accuracy. We did not rerun the complete eight-case panel and do not
present a newly measured 7/8 score; the earlier paraphrase failure remains open.

Failure accounting is separate from the accepted matrix: the initial automatic
shared-capture capacity error failed one call; the first MTP run completed one ACK
then failed its checkpoint reuse call. Both attempts and logs remain intact.
After their causes were fixed, only the affected/unrun paths were executed.
The separate default-shared-capture regression adds four completed calls. Thus
this whole follow-up contains **39 attempted Engine calls, 37 completed and two
preserved failures**, including the duplicated first MTP ACK; no silent retries.
The formal matrix has 32 calls, plus four default-capture boundary calls.

Only the affected frontend/integration targets and `ninfer-serve` were built.
The frontend regression and the native matrix passed. Production mathematical
kernels and allocator formats are unchanged, so no full suite or repeated kernel
oracle campaign was run. The final shared-capture guard affects the additional
configuration above; normal OpenAI-mode matrix calls already passed without
that branch and were not rerun. P3 is closed at this declared scope. Sync remains
default, async experimental, and no new speedup claim is made.

Evidence: `tmp-nvcc/p3-latest-20261020/manifest.json`, `summary.json`,
`base.log` (first failure), `repair-1/` (accepted base/Graph and failed MTP),
`repair-2/` (accepted MTP/DFlash2), and `shared-capture-repair/` (four default
capture calls). Each part retains commands/manifests/logs/results; affected build
and frontend logs are alongside them. Reproduce a single matrix path with
the command below from the main repository:

```sh
NINFER_QWEN3_8_KVMEM_WEIGHTS=/absolute/model.ninfer \
  build/tests/ninfer_qwen3_8_kvmem_async_real_test graph tests/fixtures/kvmem_recall
```

No commit/push, model download or changes to the external Q4 file were made.


## 2026-10-20 Known-issue closure: complete query, versions and committed history

User request: resolve the problems previously encountered, keeping Qwen3.8 GSQ
as the primary model and reusing existing projects before adding machinery.
This closes reproducible faults from the recorded single-GPU mainline. It does
not turn absent hardware/weights, unsupported modes or future acceptance into
completed implementation.

### Root causes and changes

1. **Semantic evidence arrived after the answer.** KVMem/ActKV separate current
   retrieval from historical profile. We adapted `set_retrieval_scores` and
   `commit_retrieval_round` from the local KVMem patch (5698–5703): pure current
   content cosine, separate decayed LRFU profile, one profile commit per completed
   prompt. Partial prefill previews no longer add/decay the same query repeatedly.
   For a long text prompt with no qualified lexical candidates, the original
   prefill splits at `query_end`, commits/settles the question once, restores
   semantic history via existing compact/H2D/rephase, then computes the assistant
   opener. No hidden checkpoint/replay, duplicate GDN update or larger budget.
2. **Old same-name evidence remained hot.** Newest-on-tie lexical rescue alone
   left the earlier label eligible to refill a soft profile slot. Equal older
   matches and their next pages now rank behind other optional candidates.
   Mandatory/system/action/recent protection still wins; the window does not
   shrink. Restricting semantic fallback to queries without lexical candidates
   alone did not fix this; that failed candidate is retained.
3. **Repeated question instructions outranked a name.** In the Saffron query,
   six template tokens packed into two old question pages outweighed four unique
   subject tokens. Lexical ranking now compares rarest anchor DF before combined
   weight/hits and newest-on-tie, with the original two-distinct-token admission
   rule. The actual subject/value page 49 and continuation 50 replace the wrong
   page 78. This uses token evidence, without subject/label-specific strings.
4. **A reserved empty page became committed history.** The six-fact answers
   initially all appeared, but the final stream aborted in
   `rebuild_checkpoint_protection`. A tail-column-only candidate did not fix it.
   Trace showed `mapped=16, committed=960`: an unwritten growth page entered the
   recent band and became an interior hole after append. Active compaction now
   reuses `destructive_truncate(handle, committed_frontier)` to remove ahead-
   mapped empty pages and their uncommitted ledger suffix before selection.
   Resident retained-tail coverage also comes from the logical page's committed
   columns. Protection checks remain strict; the repair does not mark empty
   columns valid. The final replay trims that empty page once in each mode and
   finishes normally. The store regression maps seven pages, commits five,
   compacts four, publishes a checkpoint and releases all allocation/reservation.
5. **The reported video timer hallucination was an annotation error.** The
   original request bytes match the saved MP4 (SHA-256
   `b2ee73ff600180506c44763e4bc8662bdcb3cca60d88c203642036fdb53152cc`).
   Visual inspection of decoded original first/last frames confirms an actual
   timer changing from 00:00:00 to 00:00:01. The allegation is withdrawn in the
   active Vision report. This pixel audit needed no new inference; it does not
   qualify broader video semantics.

### Final bounded evidence

All new model calls use the existing full Qwen3.8 GSQ artifact, greedy/no-thinking,
unchanged frozen requests and output bounds. Full-KV controls are reused when
already comparable. No new performance campaign, model download or TP2 run.

- **Original eight cases:** final code 8/8, 16/16 completed two-turn HTTP calls.
  This includes paraphrase, same-name update, tool-result and cross-page labels;
  full KV was 8/8. Final directory: `tmp-nvcc/resolve-20261020/panel-closure/`.
  Main physical pool 1024, working budget 512, reserve 128, chunk 256, INT8.
- **Original deep fixed history:** full KV 6/6 facts plus ACK; final sync and
  async each 6/6 facts plus ACK. All 14 compressed SSE requests have terminal
  events and usage, both owned servers exit 0, and content/usage/cache counts
  match across modes. Prompt lengths extend to 5160 tokens; physical pool 2048,
  working budget 1024, reserve 256, chunk 256, INT8, no Graph. Source full control:
  `deep/`; final compressed replay: `deep-unwritten-repair/`, both under the same
  resolve directory. This is a quality replay, not a new latency measurement.
- **Semantic P3:** the unchanged paraphrase request pair enters native Engine
  chat preparation through all four paths, two turns × sync/async = 16 calls.
  Every answer is correct. Complete token IDs/content/finish/reuse/commit match
  between modes; each second turn reuses 2933 prompt tokens. MTP/DFlash2 execute
  actual speculative drafts. All terminal in-flight pages are zero.

| Semantic path | Calls | Async completed/submitted | H2D batches | Turn-2 accepted/drafted | Main/MTP selection pairs |
|---|---:|---:|---:|---:|---:|
| base/no-Graph | 4 | 15/15 | 4 | n/a | n/a |
| base/Graph | 4 | 15/15 | 4 | n/a | n/a |
| MTP3/Graph | 4 | 24/24 | 2 | 6/6 | 24, all exact |
| DFlash2-K3/Graph | 4 | 12/12 | 1 | 5/10 | n/a |

The pre-existing cross-page/tool P3 matrix remains 32/32 completed, with its
separate four-call default-capture boundary; see the preceding section. It was
not blindly repeated. No all-context or all-input semantic guarantee follows.

### Preserved denominators and necessary checks

The resolve directory contains manifests, immutable input hashes, commands, raw
JSON/SSE/receipts, trace logs and stage results. `closure-ledger.json` records
**147 model-call attempts, 145 completed and two terminal aborts** for this whole
follow-up, including exploratory candidates and controls. Four completed answers
were wrong in earlier candidates: same-name update twice, Saffron once in each
transfer mode. Correct text without a terminal was counted as failure in both
`deep-final` and the unsuccessful `deep-tail-repair`. Nothing was overwritten to
turn those into passes.

| Stage | Attempts / completed | Result |
|---|---:|---|
| original paraphrase pilot | 2/2 | correct |
| initial panel / semantic-gate-only repair | 16/16 each | 7/8 each, old updated label wrong |
| versioned panel | 16/16 | 8/8 |
| deep full/sync/async control | 21/21 | 6/6, 5/6, 5/6 facts |
| specificity panel | 16/16 | 8/8 |
| first deep repair / tail-only candidate | 7/6 each | final stream aborted in both |
| unwritten-page root repair | 14/14 | 6/6 facts per mode, terminal complete |
| semantic P3 | 16/16 | all four paths pass |
| final-code original panel | 16/16 | 8/8 |

Only affected serve/native/store targets were built; standalone block decision
checks and the async compact/store regression passed. The two manual async-store
invocations initially omitted the **already existing** CTest environment
`CUDA_MODULE_LOADING=EAGER`; extending the gate timeout did not repair that setup
and was reverted. The original CMake contract explicitly preloads modules before
its deliberately blocked transfer callback. Replaying with that existing
contract passed (`unwritten-compact-async-eager.log`); production loading policy
was not changed. CUDA documents the corresponding lazy-load concurrency hazard
in its [programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html).
These two test-runner failures are separate from the model-call denominator.
No full suite, repeated kernel oracle or new performance sweep was run.

### Issue closure ledger and boundaries

| Previously encountered item | Current disposition and evidence |
|---|---|
| Original exact/early/late/cross-page facts | Closed within original panel: final 8/8, original full-KV control retained |
| Semantic paraphrase | Closed on unchanged case and all four native routes; current retrieval/complete-query timing fixed |
| Same-name update returns old label | Closed: `copper-lilac-28`, old hot equal matches demoted only in soft fill |
| Historical tool result consumes current protection | Closed by prior P3 current-user boundary; final tool label `teal-jasmine-61` remains correct |
| Deep six-fact misses | Closed on original fixed history: full/sync/async each 6/6 facts, no higher output bound |
| Answer emitted, terminal checkpoint aborts | Closed: empty growth mappings rolled back; 14/14 final deep streams complete, servers exit 0 |
| MTP independent selection / reuse coordinate / bridge mapping | Closed in prior P3; latest semantic MTP adds 24 exact selection pairs and 2933-token reuse |
| Automatic shared capture maps long prompt beyond claim | Closed by prior four-call bounded capture fixture; no universal shared recall claim |
| DMA pin/reserve/event lifetime, actual-free vs promised-free, final drain | Existing bounded implementation retained; store release and latest native terminal counters pass |
| Timer detail called hallucinated | Annotation corrected from original pixels; no model repair needed |
| Async did not meet the default speedup gate | Closed decision: sync stays default, async opt-in. Earlier 1.50% deep median is from its frozen candidate, not a timing claim for this later retrieval code |
| Vision/concurrency>1 async and unfunded speculative tail | Explicit supported-domain rejection/capacity contract retained; not bypassed to fake support |
| TP2/NCCL/two physical GPUs | Still paused by user scope and host prerequisites; not a solved single-GPU defect |
| Other v3 model families without verified complete artifacts | Await qualified weights and per-model execution; no redownload of existing Qwen3.8 |
| 32K–256K/cold multi-turn, high-res/long media and wider combinations | Unqualified future acceptance, not covered by these roughly 3K/5K text cases |

No commit/push/archive was requested or performed. External-owned Q4 changes,
Companion's dirty state and unrelated work are preserved. OpenSpec task 9 tracks
this closure; raw failed candidates remain available for audit.

Final delivery checks: affected builds, block decisions, async compact/store under
its existing EAGER contract, `git diff --check` and OpenSpec strict validation all
pass. External Q4 SHA-256 remains
`d0a14eaa1f097b3e964589af8215d331af1741394f3932b3493293af786efc5a`.


## 2026-10-20 Existing-model qualification and current-code P4

Latest user scope: **validate existing complete models only; no downloads**.
Qwen3.8 remains primary. New verification used the existing native binaries;
no inference kernel, allocator or model math was changed. Only bounded
benchmark/qualification tooling and active documentation were extended.

### Existing artifacts and media

Qwen3.6 Groupwise, Qwen3.6 NVFP4 and Companion Bonsai2B each completed the same
two-turn memory task (`ACK`, then `37`), six calls total. Qwen3.8 additionally
completed the P4 and longer-context paths below. Artifact paths remain the four
previously verified full files under `/home/ran/ninfer-work/real-models/`.
No model was downloaded, converted or redundantly whole-file-hashed.

The initial Bonsai wrapper passed the primary CLI's `--managed-weights`, which
Companion rejects; that zero-request startup is retained. Its corrected native
CLI uses `--kv-capacity 1024` and ordinary 608.4 MiB resident weights. The dirty
Companion tree was not rebuilt, committed or pushed.

Six Qwen3.8 media calls all passed: 1536×1024 panel colors, two 256×256 images,
cross-turn second-image recall, one 60-second/120-frame color-order video, and
high-resolution image + MTP3 / DFlash2 K3. Production preprocessing records
1536 vision tokens for the large image and 960 for the clip; native decode is
120/120 source frames. Actual MTP accepted/drafted is 6/6; DFlash2 is 4/6.
This is bounded color/order/reuse acceptance, not arbitrary OCR or video
reasoning. See the active Vision report for the exact processing boundaries.

### Long-context outcomes: a qualified 64K frontier, not 256K acceptance

The fact documents use native-tokenizer-sized notes with three independent
names near 3.3%, 36.7% and 70%, a replacement value near 96.7%, and a historical
tool result. Questions, fixed assistant replay and 24-token output bounds are
frozen before compressed inference. Each eligible stage has one ACK plus five
fact questions. The expected values remain
`silver-maple-41`, `amber-willow-92`, `copper-lotus-35`,
`copper-lilac-28` and `teal-jasmine-61`.

The first 32K full-KV control completed seven original calls: ACK and all five
explicit facts were correct, but the additional ambiguous “that archive”
question returned the preceding tool label. Its raw control score is **6/7**;
that semantic item is unqualified even without compression. Before compressed
inference, a separate manifest froze the five full-KV-answerable facts with
identical bytes and bounds. The failed control item is retained, not counted
as a pass or used to attribute compression damage.

| Configuration / logical ceiling | Actual first→last observed prompt tokens | Completed / attempted | Eligible fact outcome per mode |
|---|---:|---:|---|
| full-KV 32K control | 31,969→32,165 | 7/7 | 5/5 explicit facts; additional semantic control wrong |
| 4K working window, 32K, sync | 31,969→31,996 | 2/2 | first fact wrong; repeated two-call diagnostic also wrong |
| 32K working window, 32K | 31,969→32,134 | 12/12 | sync 5/5, async 5/5 |
| 32K working window, 64K | 64,712→64,877 | 12/12 | sync 5/5, async 5/5 |
| 32K working window, 213K-class | 212,276→212,303 | 2/2 | ACK correct, first Topaz fact wrong; gate stopped |
| 256K | not executed | 0 | blocked by the 213K quality gate |
| 32K working window, forced-Cold 64K | 64,712→64,877 | 12/12 | sync 5/5, async 5/5 |

The 4K diagnostic proves the correct logical pages 14/15 were present before the
question; it still answered the Saffron value. This is a compression-quality
boundary beyond the earlier roughly 3K/5K fixtures, not a demonstrated missing-
page timing bug. One explicit production alternative used a 32K window, with a
32K physical pool for the 32K tier and 36,864 physical tokens thereafter. No
working-budget sweep was performed. At 213K the first fact again answered
`copper-lotus-35` instead of `silver-maple-41`. No full-KV control at that length
was run, so attribution to model vs compression is unresolved. No 213K async
arm or 256K inference was sent after this failure. The frozen 256K tokenizer
input exists but is not inference evidence.

Long tests use ordinary resident 10.9 GiB Qwen3.8 text weights, INT8, 512-token
prefill, greedy/no-thinking/no-Graph, reserve 512, Action-LRFU 0.5/0.3. The
initial managed full-KV attempt was rejected before sending requests by the
existing 1 GiB managed-runtime limit. The text weights fit beside these runtime
allocations in ordinary device memory, so no planner guard was removed. A
separate early 36K physical/32K logical startup was also rejected and corrected
to keep physical capacity at or below the logical ceiling.

The 32K tier fits the 32K window and records zero compactions; the 64K tier
records 78 compactions, with async submitting/completing 2/2 batches and 84
fallbacks. At accepted 32K/64K lengths, full response/usage/cache records match between
sync/async and terminal in-flight pages/bytes are zero. These are statement-
level known-fact tests, not unrestricted document or universal semantic recall.
512-token chunks can exceed the 16 MiB async batch cap; many large transfers
use the existing synchronous fallback. Mode parity does not imply that every
long-context copy was asynchronous, and no long-context latency gain is claimed.

### Actual NVMe-backed Cold scenario

The global `--host-kv-mib` also funds logical handles/admission; it is distinct
from the KVMem archive. Setting it to 32 MiB failed startup at 64K, and 1024 MiB
with the 32K working claim failed request admission. Those cases are retained
with respectively zero sends and one rejected SSE request. The final usable
configuration is **2048 MiB global Host KV**,
`NINFER_KVMEM_HOST_PAGES=16` (34,603,008-byte archive), and an explicit
`NINFER_KVMEM_COLD_DIR` under the workspace's `/dev/nvme0n1p4` Btrfs mount.
The default `/tmp` is tmpfs here and was not used for the accepted Cold run.

Both final arms completed all six requests and returned the five exact facts;
response/usage/cache counts match. Each recorded **547 Host→Cold spills**,
723 stage-in and 1225 stage-out blocks, 1,061,879,808 Cold bytes at the cached
endpoint, and zero in-flight pages/bytes. Async submitted/completed 2/2 batches;
Cold restores and oversized batches retain their bounded synchronous paths.
The 16-page warm cache cannot stand in for the whole 64K history. Existing
byte/position-free Cold oracles remain the numerical basis; this adds the real
model fact/lifecycle path. The backing file is on NVMe but remains OS-cacheable;
this is not SSD latency, flush durability or restart-recovery evidence.

### Current P4 and retained denominators

Current-code P4 completed **18/18 sessions, 54/54 requests, all answers correct**,
including all six old deep facts in every arm. Outputs/usage match. Median
paired reductions: shallow -0.32%, first overpool 2.90%, deep **0.92%**. The
10% deep gate is not met: sync stays default, async remains opt-in; no depth=2
or extra performance samples. Detailed phase/resource results are in
`docs/performance.md` and `tmp-nvcc/p4-qualified-20261020/`.

This qualification follow-up preserves **116 request attempts, 115 complete
terminals, four wrong completed answers and one admission rejection**, plus
four zero-request startup failures. The wrong answers comprise the full-KV
semantic control, two 4K-window Topaz attempts and the 213K Topaz attempt.
Repeated controls/modes are not independent samples; no pooled accuracy score
is claimed. All are itemized in
`tmp-nvcc/qualification-20261020/qualification-ledger.json`.

The accepted output is a usable configuration boundary: existing model pairs
and bounded media pass; long factual recall passes through 64K with the 32K
window, including an actual file-backed Cold scenario; **213K failed and 256K
is NOT_RUN**. The procedure is complete under its predeclared stop gates;
this is not a claim that the failed lengths passed. No weights were downloaded,
no full test suite or unnecessary native rebuild was run, and no commit/push
was performed. Source reuse: native Serve wire clients/P4 schedule, Companion's
long-context marker depths, native tokenizer audit, and existing documented
Host/Cold configuration/rollback paths.

Final checks for this verification-only follow-up: affected Python tools import/
compile, strict media fact matching, response/usage parity and terminal gauges,
`git diff --check` and OpenSpec strict validation pass. Existing native binaries
were reused; no new C++/CUDA build was required. External Q4 SHA-256 remains
`d0a14eaa1f097b3e964589af8215d331af1741394f3932b3493293af786efc5a`.


## 213K diagnosis and immutable K source (2026-10-20 follow-up)

This section follows the user's request to diagnose 213K and inspect previously
left gaps. The original 32K-window, resident Qwen3.8 GSQ, INT8, chunk512,
sync ACK/Topaz requests were replayed unchanged with trace. Both requests
completed; ACK was correct, Topaz again returned `copper-lotus-35` instead of
`silver-maple-41`. The 357 selections show source page99 (value at tokens
6358–6363) and companion100 already present before the Topaz question. The previous round ends at
slots4352/4416; a further incoming-suffix pressure reselection retains511pages
and puts the fact at2496/2560 before question computation. The lexical candidate is correct. This is an answer
quality failure with selected evidence, not this case's missing-page trigger.
The full-KV213K and exact selected-view controls added below now qualify
this specific failure; the prior32K control alone could not establish it. Evidence: `tmp-nvcc/diagnose-213k-20261020/original-trace/` and
`original-trace-analysis.json`; original qualification failures stay intact.

### Controlled comparison and diagnosis

| Path | Input / evidence | Exact Topaz answer |
|---|---|---|
| Original compressed | Same213K history,32K budget; fact99/100 selected | `copper-lotus-35` (FAIL) |
| Immutable-source compressed | Same requests/budget; source file3,587,899,392bytes,3322captures/17727rebuilds | `copper-lotus-35` (FAIL) |
| Full KV at213K | Same original ACK/Topaz requests;212992physical tokens, INT8, noKVMem | `silver-maple-41` (PASS) |
| Fresh replay of actual question view | Exactly511selected pages clipped at reuse frontier212269,32685historical tokens plus unchanged34-token suffix =32719tokens | `silver-maple-41` (PASS) |

The full-KV control uses a separately linked diagnostic executable: only its
managed runtime budget is8GiB (capped by freeVRAM minus512MiB); production
`registry.cpp` and production managed1GiB cap are unchanged. It admitted
7.09GiB runtime, completed ACK and Topaz with valid terminals and exit0.
Weight migration makes this a quality control, not a performance comparison.
Both original requests and model paths are unchanged. Native context is262144,
so no YaRN extension is activated at this length.

The first selected-view probe used the preceding512-page round-final window
(32783tokens), and correctly answered too. That probe was initially mislabeled
as the question view. It is preserved, but the final localization uses the
subsequent511-page view that the program selects BEFORE computing the question.
Original and immutable-source question views overlap507/511pages; the final
probe uses the immutable-source failing arm's exact511pages, including the
fact. The raw-token control enters a fresh publicEngine and adds no wrapper
or replacement question. `result.tokens.json` and `result.audit.json` freeze
the exact native IDs and selection.

**Conclusion:** selected content is sufficient and full213K inference can answer.
The failure lies in history-conditioned compressed inference (cached K/V, GDN
state and their position/mask transitions), rather than a lack of the fact in
the selected content. These controls do NOT yet isolate a specific K/V versus
GDN branch, nor rule out every residual rephase error. Immutable-source repair
alone is insufficient.213K remains FAILED;256K remains NOT_RUN. The next useful
change is to compare/repair state reconstruction across compaction, borrowing
the reference checkpoint/restore/replay mechanism with consistent GDN state;
full-window replay here is a diagnostic, not a production performance solution.

Evidence under `tmp-nvcc/diagnose-213k-20261020/`: `diagnosis.json`,
`actual-question-selection.json`, `full-kv-control/` (including exact two-line
registry diff and build commands), `selected-view-control/` (prior-window
probe) and `actual-question-view-control/` (final exact question-window probe).

### Immutable-source implementation

The previous executor recognized `raw_refresh` but only reset counters and
re-quantized the current working K. This follow-up adapts the immutable-source
policy from KVMem/ActKV `kvmem_store.cpp:837` and its default refresh bounds
(8 remaps / 262144 cumulative position delta). The native implementation keeps
first-write K codes/scales plus per-column birth positions, decodes that
immutable source to the raw-K domain and rebuilds working K on cold restore
or threshold refresh. This preserves the initial INT8 error; it is not a
pre-quantization BF16/raw-K capture. V stays with the existing Host/Cold owner.
No new rotation arithmetic or quantizer is introduced.

`src/kvmem/kvmem_source_k.h` reuses ColdKVStore files and native Host tensor
strides with one pinned K-page staging buffer per typed store. A Qwen3.8 source
page is1081344 bytes; the full213K history can add about3.35GiB of backing,
independent of global Host admission and ordinary archived KV. Partial pages
retain immutable earlier columns and capture new columns with their own birth
positions; truncation discards the overwritten source suffix. Address release
reclaims records. The source path is selected by `NINFER_KVMEM_COLD_DIR`;
its file may be OS-cached. The diagnostic points it at the existing /home NVMe
mount; default /tmp on this host is tmpfs.

The affected serve/compact targets built. Sync, async, cancellation, forced
Cold and device-claim checks all passed (5 checks); the added mixed-birth
column oracle checks source reconstruction, scales and rollback recapture.
This does not by itself establish that immutable K fixes the real model answer.

### 上一轮功能收尾时的源码对照缺口（历史快照）

以下保留该轮原状态；性能轮的最新处置与结果见文末。

| 项目 | 当前状态与证据 | 后续 |
|---|---|---|
| 213K压缩状态质量 | 原六请求sync/async12/12正确，完整GDN/KV选页视图重建及不同锚点候选已修；本轮不重复213K矩阵 | 减少重建属于后续TTFT调优，不归因某个未分离的数学算子 |
| 首写K数值边界 | 首次INT8编码是明确支持的语义边界，已有独立数值/生命周期检查；不是尚未接入不可变来源 | 未出现需要pre-codec BF16才能修复的反例，不无依据新增数学路径 |
| NVMe预取 | 当前同步读取是正确性回退，引用项目有有界预取可复用 | 按用户要求放到下一轮性能调优 |
| 大批异步回退 | 超过16MiB正确回退，现有原题搬运透明性通过 | 拆批和流水深度放到下一轮性能调优 |
| managed runtime容量 | 已接生产Engine/Serve显式预算 `--managed-runtime-mib`，默认上限1GiB并留512MiB初始设备余量；2GiB预算真实W32K/64K两原题均通过 | 无需诊断专用registry；未重复原完整KV213K控制，不借此声称所有大预算均实测 |
| 独立内存预算 | 全局Host cache容量、普通KVMem Host占用/容量、来源K有效文件字节/可复用文件范围/staging已分别观测；`--host-kv-mib`与`NINFER_KVMEM_HOST_PAGES`各控原有范围 | 不宣称统一RAM上限；文件/OS页缓存、堆元数据与固定pinned staging含义已分开 |
| 来源和回放观测 | 来源捕获/重建及history/query回放tokens/rounds已接KvMemUsage和Prometheus；physical_prefill另计额外工作 | cached_tokens仍为逻辑复用前沿；新指标供后续性能归因 |
| 历史证据映射 | 已在19-Oct原交接文件顶部更正：所引日志6.70GiB/65536物理KV/41.3tok/s，不能支持原131072/98304/51.5的配置引用 | 保留历史答题与原文，不与当前10.9GiB GSQ混算 |
| 更小工作集质量 | 原W4096/物理8192/32K六请求sync/async12/12正确完整；原题和参数不变 | 原256K档W32768/物理36864六题两模式也12/12正确，实际首轮261411 tokens；不外推任意文档 |
| 当前性能证据 | 本轮不测P4、不减重放量；旧0.92%成绩不适用于新源码 | TTFT、NVMe预取、大批拆分及P4统一留下一轮；sync默认不变 |
| 保留私有Fork | 已拒绝无法继承完整归档的压缩Fork候选，允许root重建并保留原来源；A→B→A两模式6/6正确，B复用0、回A复用2962，IDs一致 | 尚未实现零重算的共享归档Fork；这是性能/能力扩展，不再静默丢历史 |
| 当前组合资格 | Graph、MTP3、DFlash2短语义案例12/12正确完整，sync/async IDs、commit/reuse一致、复用2933，在飞0；实际执行了canonical重建 | 未声称这些组合已完成256K或任意媒体验证 |

The previously fixed query timing, content-only rare-anchor ordering, MTP bridge
mapping, unwritten growth trimming, FFmpeg packet handling and frame accounting
remain implemented; no TODO-only stub was found for those scoped paths. This
source comparison is not a certification of every ecosystem feature. TP2 and
new-weight downloads remain outside this work.

### Post-repair lower-frontier and typed-backend regression

The unchanged64K Cold requests were frozen separately as a lower-tier regression,
not a length escalation. Current-binary sync/async completed12/12 requests with
all five facts correct in each arm, same outputs/usage,549Host→Cold spills per
arm and terminal in-flight pages/bytes0. Immutable K source backing reached
1,096,482,816bytes with1027capture operations and4155rebuilds per arm. The
archive cap remains16pages and backing is on /dev/nvme0n1p4 (/home, Btrfs).
Evidence: `cold-64k-regression/` under the diagnostic root.

The affected native integration target was relinked; the original semantic
fixture ran on Graph+MTP3 with both transfer modes:4/4 calls correct, tokenIDs,
finish, commit and reuse match,2933tokens reused, terminal in-flight0. This
checks the changed Main/MTP source path, not the full previous P3 matrix.
Evidence: `mtp-regression/result.json` and its trace log.

This follow-up has24completed model calls:2intentionally retained wrong
answers (original/source-repaired213K) and22correct responses across repeated
controls/regressions. They are not independent accuracy samples. There was one
zero-inference helper compile failure (incorrect API member names), corrected
before running the helper. All evidence stays in `ledger.json` and its phase
directories; the preceding116-attempt qualification ledger is unchanged.
No256K, new P4, new weights, commit or push was performed.


## 213K original-sequence repair: current acceptance

The unchanged original six requests now pass sync/async12/12 with Qwen3.8 GSQ,
32768working tokens and36864physical tokens. Both services exit0; outputs,
usage and373selection steps match,2471Host→Cold spills per arm, terminal
in-flight pages/bytes0. Async completes2D2H batches but has386fallbacks and
0async restores; Cold remains synchronous. No async performance-adoption claim.

The missing state transaction was first ported from reference KVMem/ActKV:
complete StateImage snapshot, full-query probe, restore and query replay. It
still failed Topaz, so the final repair reconstructs coherent GDN and all-layer
K/V for the exact selected original-token view, then replays the unchanged
query. Old source/mean/replica data is revoked for the reconstructed epoch.
The next original Indigo question exposed a separate single-candidate tie:
a previous question template erased a different name with identical scalar
statistics. Up to four distinct actual anchor signatures are now retained
within existing protection/budget rules; same-anchor old versions remain lower.

Every internal chunk consumes pre-reserved service work; public prompt progress
is counted once. Per arm, six histories replay196076tokens, plus162query replay
tokens. Cached-token usage still denotes the logical reuse frontier rather than
physical computation saved. This cost is visible in short-query TTFT23–26s.

All intermediate wrong answers and the zero-request startup/build mistakes are
preserved. This follow-up adds17complete calls with2intermediate wrong answers;
the diagnostic root ledger now has41complete calls and4historical wrong answers,
not independent accuracy samples. New256K/P4 and Graph/MTP/DFlash2 matrices were
not run; no weights, commits or pushes were added. Details and exact evidence
are in [the repair report](2026-10-20-213k-state-replay-repair.md) and
`tmp-nvcc/diagnose-213k-20261020/window-replay-v3/analysis.json`.


## 功能缺口收尾：TTFT后续专项处理

本轮复用原诊断预算逻辑、KvMemUsage/metrics聚合与Admission候选/root重建，参考KVMem/ActKV会话共享归档的所有权约束。生产参数增加 `--managed-runtime-mib N`，仅在managed模式可用；默认上限1024MiB，显式预算仍受初始freeVRAM减512MiB以及原容量曲线约束。不是新权重分配或无上限设备超售。

保留压缩会话的Fork原本只复制驻留页，不能继承原地址的archive/source目录。现在在候选阶段判断其是否完整；不完整时不修改来源，其他合法候选/root仍可执行。两模式A→B→A均返回ACK、violet-cinder-58、violet-cinder-58；B从root执行、返回A继续复用2962 tokens，验证未破坏父会话。

新的来源指标区分live文件内容、复用槽文件范围、单页pinned staging和捕获/恢复；全局Host cache、普通archive和来源文件保持各自合同。history/query额外回放和主模型physical prefill token位置另计，不重复计typed pools，也不把该指标解释为FLOPs或decode投机工作量。真实managed64K两请求产生65343 history/44 query重放，physical prefill130146，来源有效1094320128 bytes、文件槽范围1095401472 bytes、staging1081344 bytes。

已完成的最小功能验证：managed64K2/2、原4K工作集32K12/12、保留分支6/6、Graph/MTP3/DFlash2短语义12/12，均有完整终态或原生Engine完成、sync/async IDs一致，终态在飞0。256K原六题两模式12/12正确完整（实际261411至261576 prompt tokens），输出/usage一致，每臂468次压实、3099次Cold spill、终态在飞0。全轮44/44请求正确完成；不重复已有213K矩阵，不下载新权重或运行TP2。

证据集中在 `tmp-nvcc/functional-gap-20261020/`：各phase manifest/results/SSE/metrics，`native-manifest.json`、`native-summary.json`、各route JSON与日志；参数和指标CPU回归通过。一处零推理构建错误（指标重载的整型类型歧义）已修正，原日志保留。旧的失败、控制和资格化分母未改写。

TTFT减少重建、NVMe预取、超过16MiB批次拆分和P4测量明确延期到用户要求的详细性能轮；不将延期性能项标为本轮修复或新采用结论。


### 本轮最终功能结果

| 范围 | 调用结果 | 关键证据 |
|---|---|---|
| managed64K，W32K/物理36864，显式2GiB runtime | 2/2正确完整 | 原ACK/Topaz；生产参数及来源/回放/容量指标实际生效 |
| 原4K工作集，物理8192，32K输入 | sync/async12/12正确完整 | 原六文件/参数不变，31969至32134 prompt tokens，每臂76压实/154Cold spill，在飞0 |
| A→B→A保留分支 | sync/async6/6正确 | B跳过不完整Fork走root，回A复用2962，原会话完整保留，IDs/commit/reuse一致 |
| Graph、MTP3、DFlash2短语义 | 12/12正确完整 | 三路各4，实际canonical重建443tokens/arm，二轮复用2933，投机drafts非零，IDs/commit/reuse一致 |
| 冻结256K档，W32768/物理36864 | sync/async12/12正确完整 | 原六题未改，实际261411至261576 tokens，468压实/3099Cold spill每臂，在飞0，输出/usage一致 |

本轮44模型调用全部正确且完成，没有模型请求失败或自动重试；另保留1次零推理编译错误及修复日志。CPU参数/指标两项通过；只构建受影响目标。`tmp-nvcc/functional-gap-20261020/summary.json`由各原始results、metrics与原生JSON逐项汇总，验证额外回放计数、工作集上限、完整终态及两模式一致性；旧诊断ledger41调用/4个历史错误和此前资格化ledger均保留，不把重复模式当独立准确率样本。

256K是本固定事实集的真实验收，不是任意256K文档或完整KV等价证明；该长档未使用MTP/Graph/DFlash2。新投机组合证据仅上表短语义范围。TP2、缺权重模型和既定不支持域保持原边界。性能项明确延期，无新P4/加速采用结论，无提交/推送。


## 详细性能调优：TTFT、NVMe读前、SM89 MMQ与当前P4

2026-10-20，本轮使用同一已有Qwen3.8 GSQ-RCO-IQ3_S v3完整权重；不重新量化、不下载模型、不改原题。仅一张RTX4070 Ti SUPER，SM89/66SM/16GiB，CUDA13.4.92、driver615.78.08。最终二进制原213K六请求sync/async **12/12正确完整**，当前P4 **18/18会话、54/54请求正确完整**。P4异步深轮中位收益 **0.54%**，未达到10%，保留sync默认与async opt-in。

### 实施内容和源码复用

1. **规范历史前缀缓存**：复用原生`HostStatePool`、`StateImage.copy_to_host/copy_from_host`和KV rollback。W≥4096时启动固定分配每lane四个完整Host StateImage，在W/16、W/8、W/4及最后完整prefill_chunk边界保存规范重放状态。身份包含同一address generation、有序原始token位置和token ID；仅逐项相同前缀可恢复。普通prefill写入某检查点之前会使它失效。保留前缀K从该conditioning epoch的首写来源恢复，V继续使用相同因果前缀的内容，只有后缀撤销旧来源/归档/均值并重算。未命中仍从空状态重建；没有恢复已证实错误的压缩时间线快照。设备工作集不增加，内部service-unit上界保持保守。
2. **NVMe文件层读前**：直接适配KVMem `kvmem_runtime.cpp::start_prefetch`的future模式，结合ActKV `spill_io.cpp`持有extent直到任务结束的生命周期。选页确定后最多两页后台`pread`，和换出及重定位准备重叠。future销毁先于buffer；源record在future收割和GPU读取完成前不会释放。另留一页换出staging，因此普通Cold staging固定三页；Windows文件游标操作互斥。这里证明文件层I/O通路，不声称每次都穿透OS页缓存到SSD。
3. **大批Host搬运拆分**：单次复制提交至多16MiB，早期子批在当前独占compact调用内完成后发布/回收，只有最后D2H子批可跨Engine边界在飞，并继续核验epoch/coverage。H2D目标仍由正常Device预留计费，前子批完成后提交后子批，末批与独立resident rephase重叠。后子批提交前清除旧completion标记，提交异常会排空stream。不会仅因整个Host事务超过16MiB退回同步。
4. **4070算子执行几何**：Nsight Systems确认GGUF MMQ是主要GPU成本。复用同仓`gguf_project`及已vendored ggml矩阵核的完整批宽实现，将普通linear在SM89的每次批宽128提高到512；其他架构保留128。规划和实际执行同步更新，fixup采用每SM一个128×128 FP32 tile的上界，覆盖短尾批。没有新增量化数学内核，生产CUDA SASS逐字一致，256条IMMA指令仍是原实现。

主要源码：`src/targets/qwen3_6/impl/runtime/{program.h,program_impl.h,logical_kv_store.h}`、`src/kvmem/kvmem_cold_store.h`、`src/ops/linear/gguf/gguf_dispatch.cpp`及原有Usage/Serve metrics。CPU参考来自项目本地`.tools/gguf-oracle`中的GGML gguf0.19.0 NumPy解码器（MIT）；生产推理不依赖Python/PyTorch。

### 原213K逐题结果与TTFT

原始文件字节、greedy、max_tokens24、logical212992、实际212276–212441tokens、W32768、物理36864、prefill512、Host8192MiB、resident weights、INT8、base/no-Graph均保持；冷文件在工作区NVMe/Btrfs目录。最终两个模式的正文/usage及 **373次选页完全一致**，每臂2468次Cold spill，终态在飞页/字节为0。

以下TTFT单位秒，以原始wire receipt首个model_output减body_sent计时。历史窗口重放结果与本轮分阶段测量并非交错的二进制A/B，最终任务还有20GiB主存/2GiB swap约束；因此是可复核观察，不能把差值全部归因给某一优化。

| 原题 | 原完整重放sync | 最终sync | 原完整重放async | 最终async |
|---|---:|---:|---:|---:|
| 首次ACK | 201.675 | 175.519 | 195.169 | 191.258 |
| Topaz | 26.094 | 24.025 | 25.145 | 24.470 |
| Indigo | 24.299 | 19.492 | 23.250 | 18.931 |
| Saffron | 24.085 | 21.415 | 23.275 | 18.963 |
| Quartz更新 | 24.256 | 20.253 | 23.539 | 17.616 |
| Jade工具结果 | 24.193 | 20.701 | 23.222 | 19.308 |
| 五个短问题中位数 | 24.256 | 20.701 | 23.275 | 18.963 |

历史完整重放每臂196076tokens；第一阶段缓存/流水二进制为180204、明确复用15872tokens，两模式12/12。最终MMQ候选改变了浮点归约的调度及部分可选选页，实际重放 **185836tokens**、复用 **10240tokens**（相对完整重放少5.22%），query replay仍162tokens。没有声称旧/新二进制全部logits逐位相同。当前同一二进制的sync/async重放量和选页相同，physical_prefill等于logical computed加history/query重放，未将saved tokens重复计入。

最终async实际完成 **986** 个搬运子批（submitted相同），其中 **622** 个为大批中提前完成的子批；回退 **24** 次。旧完整重放记录为2批/386回退。NVMe读前 **63提交/63消费完成**，future.get累计约28.229ms；普通Cold消费H2D与immutable-K的单页恢复仍可能同步等待，不宣称零等待。

四个checkpoint的固定Host payload为 **615817216bytes（587.29MiB/lane）**，另有有界token身份scratch。它独立于全局Host context cache、普通KVMem archive、source-K文件/单页staging；没有实现统一RAM预算。新指标为`kvmem_history_reused_tokens_total`、`kvmem_replay_checkpoint_capacity_bytes`、`kvmem_async_split_batches_total`、`kvmem_nvme_prefetch_submitted_total/completed_total/wait_us_total`。

### 4070上的算子证据

独立参考先用GGML NumPy在CPU解码已有六种格式的128行，再用FP64求和；输入每列使用不同的可精确表示比例。直接验证生产GGUF dispatch，覆盖T=1/16/128/129/256/512/513及Q8非恒等input_columns，均通过既有数值容差。没有用CUDA解码器当权重oracle。

下面只属于 **N128、K5120、T512的实际权重行样本Op**：两次warmup、七次CUDA-event计时的中位数，单位微秒；不能当作27B整模型3倍加速。

| 编码 | 原128批宽 | 512批宽 | Op加速 |
|---|---:|---:|---:|
| gguf_iq4_xs | 106.272 | 32.576 | 3.26× |
| gguf_iq3_s | 109.312 | 33.664 | 3.25× |
| gguf_iq3_xxs | 109.440 | 32.800 | 3.34× |
| gguf_iq2_xs | 125.888 | 38.880 | 3.24× |
| gguf_q2_k | 141.312 | 48.096 | 2.94× |
| gguf_q4_k | 113.568 | 34.816 | 3.26× |

独立小Op的NCU basic采集成功：原批宽一个128列kernel16.06us，新批宽一个512列kernel24.77us；kernel工作量不同，整Op比较使用上表。小样本Memory Throughput24.30%→46.01%，SM Throughput4.81%→12.71%，不作near-peak结论。

原生Engine语义夹具两模式各两轮（4调用）仍通过，token IDs、commit/reuse一致。Nsight只捕获首次generate：MMQ调用11753→6132，cudaLaunchKernel62660→40176，累计GPU kernel1726.78→1600.72ms，首次请求2008.12→1825.09ms（约9.1%观察改善）。这一原生请求证据与小Op结果分开报告。

### 当前二进制P4

保持原三场景、每场景三对、sync/async交错、18会话/54请求及原字节请求；只切换搬运模式，base/no-Graph、W1024、physical2048、prefill256、managed weights、INT8，冷启动readiness后计时，模型加载不计入session wall。候选与基线都54/54正确，候选各对正文/usage一致，终态正常。

| 场景 | 基线sync中位wall ms | 候选sync中位wall ms | 候选async中位wall ms | 候选同对async中位收益 |
|---|---:|---:|---:|---:|
| 浅载 | 232.3 | 222.0 | 222.0 | -0.70% |
| 首次超池 | 1874.6 | 1729.3 | 1678.7 | 2.92% |
| 深轮七请求 | 9313.8 | 8777.1 | 8732.7 | 0.54% |

候选深轮三对收益0.443%、0.542%、1.253%，均为正但中位数低于10%；浅载中位回归0.704%，在3%限制内。**质量通过，默认采用的性能门槛未通过；保持sync默认，不追加样本或调深度来追门槛。** 两二进制sync深轮wall观察下降5.76%，这是先后批次对照，不是交错二进制A/B。

P4深轮短请求TTFT中位数：sync833.137→799.483ms，async834.057→799.268ms；decode中位约39.9–40.1→40.5–40.6tok/s，未据此宣称专用decode内核提升。小P4的W1024不会触发W≥4096的Host前缀缓存，它不能替代213K的重建减量证据。

### 主存不足事件与执行约束

最初非管理员NCU因`RmProfilingAdminOnly=1`失败，但4个模型调用完整。随后已有免密管理员权限允许采集；整模型`--set full`的kernel replay提示“Backing up device memory in system memory”，21:57:28发生 **global OOM**，内核杀掉本轮目标PID978870，profile退出9并影响桌面。该失败没有删除或改成成功，也未重试整模型NCU kernel replay。此前硬件探测的`can_read_counters`不能替代实际权限验证。

改用独立小Op的NCU basic，单次内存峰值约994MiB，4GiB主存/0 swap上限；普通Op计时峰值588/271MiB，不能用这两个冷暖进程差异宣称内存优化。编译上限6–8GiB/0 swap，最终完整推理矩阵在独立systemd transient service中使用 **MemoryMax20GiB、MemorySwapMax2GiB、OOMPolicy=kill、RuntimeMaxSec1800**。最终服务正常退出，报告峰值20GiB、swap2GiB，仍存在内存压力；这只是隔离与硬限额，未完成统一Host内存治理。GPU测量、编译均串行；未主动停止或修改其他项目进程。

### 证据、分母与剩余工作

本轮共145次模型调用尝试：144次完整且期望答案正确，1次为整模型NCU OOM未完成。最终候选66/66来自原213K12调用和P4 54调用；两个原生Nsight各4调用、非管理员NCU4完整调用、第一阶段213K12调用和基线P4 54调用单列，重复问题不作为独立统计样本。两次零推理的执行器/报告路径配置错误保留在日志，不混入模型质量分母。CPU metrics、受影响构建、作用域diff和规格检查见最终收据；外部`q4_dispatch.cpp` SHA256仍为原值。未提交、推送或归档。

证据根目录：`tmp-nvcc/perf-20261020/`。`summary.json`/`summarize.py`可从原收据重新计算；`long-213k{,-final}`、`p4-{baseline,candidate}`保存冻结manifest/SSE/usage/metrics，`native-*.nsys-rep`、`op-mmq*.ncu-repz`、两份相同SASS和CPU oracle可复查。原始213K/256K失败及前轮44调用记录原位保留。

仍待后续：选页在早期变化时，每轮仍须约30K联合GDN/KV重建；immutable-K捕获/恢复仍是单页同步路径；普通Cold消费仍可能等待，剩余24回退不是已消失。下一轮应先在独立、匹配完整N/K的算子夹具中看MMQ或source-K实际瓶颈，再考虑有预算的多视图缓存/共享归档及统一Host额度。禁止为采集再次执行整模型NCU kernel replay。未重跑256K、媒体或长档Graph/MTP/DFlash2，本轮不能替代它们的较早功能证据。

关于GGUF：当前制品已经是`.ninfer`，Engine是原生C++/CUDA；内部保留IQ3_S/IQ4_XS等GGUF权重编码并直接使用原生CUDA/MMQ，未调用外部llama.cpp推理。转换容器/绑定可以保留权重数值，改成不同的原生Q4/Q8码本通常需要重新量化。另一条路线是保留IQ码本移植专用kernel。本轮采用保留权重、优化现有调度的路径。


## 持续调优：完整尺寸与测量纪律

本轮持续目标仍在进行中，未声称已经达到4070 Ti SUPER性能最优。生产模型是已有`qwen3_8_gsq_rco_iq3_s_v3.ninfer`：v3原生容器、NInfer public Engine/C++/CUDA调度执行，内部保留IQ3_S、IQ4_XS、Q4_K等量化块编码并复用GGML CUDA算子。这是“保留GGUF量化块编码的NInfer原生Qwen3.8”；混合编码描述权重表示，推理由NInfer执行，没有调用外部llama.cpp、重新量化或更换模型。

### 本轮实际保留的改动

独立GGML NumPy CPU oracle增加`--full-shapes`，每次只解码128行，再以FP64求和。完整读取已有的`[17408,5120]`独立gate/up、`[34816,5120]`融合parent及`[5120,17408]`down中实际存在的六种格式，共15个entry；不把词表头或整个模型放进RAM。当前生产Op在T=1/8/9/512/1024/2048/2049的 **105/105** case通过原数值门，并保留原Q8非均匀gather参考。新增`--format`、`--entry`、`--tokens`筛选，允许下一轮只采集相关真实尺寸。具体命令见`tools/bench/README.md`。

### 筛选与采用边界

每轮调整前已经检索GitHub MMQ/MMVQ、CUDA Graph、混合gate/up融合、Strata及Reddit线索，并查看本地未接入的gguf_project/ninfer-3090实现。链接与兼容性说明在`tmp-nvcc/perf-ada-round2/research.md`。Reddit异机速度未当作本机证据。

初筛冻结同一原生制品、resident weights、INT8、logical/physical4096、raw token corpus、greedy、无投机，输入2048tokens及随后128decode输出。MMQ提交宽512/1024/2048均有完整尺寸CPU校验记录，但扩大批宽与prefill_chunk未建立端到端采用证据：512批宽/chunk512的先后批次观察约2002prefill tok/s；1024批宽/chunk1024约2003，2048/2048约1942，workspace分别约102/200/395MiB。它们是顺序筛选，不能把百分比差异解释为因果收益或回归；因此保留原512策略。

共享激活候选直接适配gguf_project的gather/layout身份复用：兼容时量化一次，仍各自产生BF16 gate/up再执行原SiLU；不改CUDA数学、码本或工作区上界。35个真实pair case通过独立CPU oracle及两次原linear的BF16逐位比较，另4个非均匀gather case通过。但顺序整模型观察并未显示收益（eager decode42.430→42.008、Graph43.878→43.575tok/s），不足以采用。**没有将这约1%的差异定性为已证实回归**。五个生产文件及pair测试增量已撤回，当前保留完整尺寸工具；候选源码、可回放patch、二进制及原始测量保存在`rejected-shared-activation/`，便于后续按形状重新评价。

### 用户指定经验后的真实交替对照

已完整阅读项目`.claude/skills/ninfer-perf-tuning/SKILL.md`，采用其同一binary、clean、交替、按尺寸分桶与实际分支探针纪律。三元模型历史阈值、耗时及尚未钉死单位的带宽常量不套用到当前制品。当前桌面仍在独显上渲染，记录的背景显存约1.1GiB，未满足该经验的核显显示条件，因此不声称硬件极限。

使用恢复后的同一`ninfer_bench`执行A/B/B/A/A/B，每个进程一次warmup、一次测量，Graph按原工具先prime；参数与输入固定，只有Graph开关不同。每臂计时128个decode输出，模型加载与prime不计入吞吐；温度/SM及显存时钟/功耗采样另存。结果如下，单位tok/s：

| 配对 | eager decode | Graph decode | 同对吞吐收益 |
|---|---:|---:|---:|
| 1 | 40.238 | 41.433 | 2.971% |
| 2 | 40.361 | 41.637 | 3.160% |
| 3 | 40.257 | 42.382 | 5.279% |

同对decode收益中位数 **3.160%**，prefill同对中位仅 **0.018%**。这是既有Graph机制在此限定夹具上的配置证据，不是本轮新增CUDA内核收益，也不外推至213K/KVMem长档。后半轮绝对速度较早批次下降，说明跨时段绝对值不能代替当场成对比值。

### 执行记录与下一步

本轮 **56次模型调用完整**，包括warmup、Graph prime和一次Nsight采集；其中检查的是输出预算与完整终态，不能当作56个独立事实质量样本。没有重跑213K/P4/256K或媒体矩阵：候选未采用，生产源码已恢复到本轮开始状态，旧原题证据原位保留。构建串行两worker、6–8GiB/0swap；独立Op 4GiB/0swap，最终105case峰值约637MiB；Engine 20GiB/2GiB限额，三对交替任务峰值2.1GiB/0swap。没有整模型NCU replay或新的OOM。

最初一次Op候选副本在link完成前复制到了旧binary，实际仍是512；`stale-copy512.log`及副本保留，已排除出候选比较，后续复制在构建终态后进行。Graph级Nsight的逐kernel表只含eager seed部分，未覆盖全部graph节点；不能把它的kernel次数当作128步decode次数，也不能用node trace的host launch时间推算clean收益。

当前主要待解是GGUF decode的实际ALU/缓存/并发瓶颈及其尺寸分桶。下一轮先在完整N/K的小Op采集NCU，再决定每warp行数等schedule是否值得改。每行权重字节和归约总数是约束不变量；activation逻辑读取减少也不等于DRAM读取同幅减少。新候选须同一binary可切、分支探针确认及clean真交替；只有与现有数值边界和原题相容的收益才采用。OpenSpec第15阶段仍有持续优化任务，不归档、不标全局目标完成。

证据根目录：`tmp-nvcc/perf-ada-round2/`；`summarize.py`从原始JSON/日志重新生成`summary.json`，包含全部56调用分母、三对比值、未采用候选和105case计数。没有提交或推送。

## 持续调优：实际NCU与J64候选拒绝

已再次沿源码确认当前制品头为`NINFER\0\x03`、metadata为`qwen3.8-27b`。NInfer Reader/materializer直接上传权重，Program调用本项目Ops，再由已静态链接的CUDA bridge发射kernel。`GgufBlocks`是部分权重的物理编码；不需要再转一次才能获得原生执行，也没有将推理委托给外部llama.cpp。精确执行链见`tmp-nvcc/perf-ada-round3/native-runtime-note.md`。

调整前检索了上游GGML MMQ配置、issue28760、Strata、bc250-rocm及Reddit项目线索。只在独立小Op采集NCU full，单任务4GiB主存/0swap；未再使用整模型kernel replay。

IQ3_S decode，N17408/K5120/T1的DRAM为91.13%、596.93GB/s，N5120/K17408/T1为83.06%、544.25GB/s；大N桶暂不实施四行/warp。prefill N17408/K5120/T512的既有J128使用254寄存器/线程、57.86Kbyte dynamic shared、理论occupancy16.67%、实际16.54%。据此仅试用已经编译的J64实例，保留I128、256线程、K循环、量化数学和512列提交上限；进程冻结env与实际launch探针允许同一binary对照。范围仅SM89、三种IQ格式的N17408/34816、T512，默认未切换。

六个完整N/K桶的两臂CPU oracle共12/12通过；固定seed随机BF16输入的完整输出直接逐字节比较，六对全部相同。关闭trace后每桶按A/B/B/A/A/B运行，单进程两次warmup、七次CUDA-event样本，共18对。下表为同对吞吐比`J128耗时/J64耗时-1`的中位数，负数表示J64更慢，仅是原生Op结果：

| 编码 | N17408/K5120/T512 | N34816/K5120/T512 |
|---|---:|---:|
| IQ4_XS | -12.118% | -13.640% |
| IQ3_S | -18.251% | -20.207% |
| IQ3_XXS | -17.036% | -18.401% |

J64的NCU复测显示寄存器降到168、shared降到48.38Kbyte，但寄存器及shared仍各限制为每SM一CTA，理论occupancy仍16.67%，实际17.32%。与此同时列tile及CTA翻倍：N17408时544→1088；权重tile的逻辑读取/解码次数随之增加，不能把它等同于DRAM字节翻倍。寄存器数量下降没有解除驻留限制，实际clean计时在所有桶都更慢，因此拒绝该候选；不单凭occupancy预测收益。

已保存候选源码、patch和实际计时binary到`tmp-nvcc/perf-ada-round3/rejected-j64/`，恢复生产MMQ头并成功重建Op/bench。仅保留测试工具的单entry原始BF16输出及固定seed随机输入，后者明确标为`BITWISE_CAPTURE`，不冒充CPU数值oracle。当前生产运行策略与本轮开始相同。

本轮48个普通小Op进程完整，另4个小Op NCU采集完整；未运行整模型，不产生新213K/P4/256K质量或性能成绩。既有原题证据继续适用，未因已拒绝候选重复长档。普通Op峰值约704MiB、NCU约1GiB，构建两worker/8GiB限额峰值约1GiB；全部0swap，无新OOM。`summary.json`由本目录`summarize.py`从日志/CSV/输出字节重新生成。

下一候选可参考本地ninfer-3090的BF16打包写回，在无Stream-K fixup的既有MMQ中消除FP32输出plane及独立转换；需要保持FP32累加、最终BF16舍入与地址步长，fixup和FP32输出继续使用原路径。目前尚未实现，不声称这部分已有收益。持续优化目标保持进行中，没有提交、推送或归档。

## 第四轮：GGUF MMQ 最终直接写回 BF16

状态：**ACCEPTED**。数值与受测性能门槛通过，默认已启用并重建；最终默认版本原 213K 六题 **PASS6/6、server exit0**，质量门槛已通过。采用范围限于下述设备、格式、输出类型和已资格化 workload。

本轮复用现有 GGML MMQ 和本地 ninfer-3090 的最终 BF16 物化思路，把符合条件的 `FP32 plane → store_plane → BF16` 改为 MMQ 最终直接写回 BF16。保留 activation 量化、FP32 累加、K 迭代与 J128 调度；通过默认 `Dst=float` 模板保留旧路，以类型贯通所有输出偏移和 stride。来源、代码边界及机制细节见 [research.md](../../tmp-nvcc/perf-ada-round4/research.md)。

默认 `NINFER_GGUF_MMQ_BF16=1`，显式设 0 可回退。直接路线限 SM89、J128、`blocks==tiles && !fixup`、普通 BF16 输出，以及 IQ3_S/IQ3_XXS/IQ4_XS/IQ2_XS/Q2_K/Q4_K 六种格式；FP32 head、MoE、split/fixup、其他设备或不匹配条件继续原路。默认值切换后的实际 probe 已命中 `bf16-direct`。

数值证据为 **34/34 constant CPU-oracle cases 通过、27/27 随机 BF16 完整字节对照相等**。矩阵覆盖六格式和真实 N/K、down 的 T257 单列尾 tile、T513 的 MMQ512+vector1、T1024 连续子批及 fixup 回退。随机固定 seed 的字节一致补充原边界保留证据，不替代独立数学 oracle 或原题质量门槛。原始记录见 [qualified/receipts.json](../../tmp-nvcc/perf-ada-round4/qualified/receipts.json)、[qualified/summary.json](../../tmp-nvcc/perf-ada-round4/qualified/summary.json)。

小 Op 使用同一 binary、两臂进程 env 冻结、trace 关闭，9 个 T512 热点形状按 ABBAAB 进行 27 对计时。各桶中位收益 **+5.104%～+12.354%**，指完整 linear Op；其中 IQ3_S/N17408/K5120 为 +8.481%。Nsight Systems 确认该形状 kernel 从 3 个减至 2 个，移除 `store_plane`；baseline profile 中该转换 kernel 单次为 61.18 us。单主核 NCU 两臂仍为 254 registers/thread、57.86 Kbyte dynamic shared，采样时间 803.04/797.60 us，因此不把完整 Op 收益表述为主核加速约 8%。profile 用于机制确认，收益采用 clean timing。

SASS 显示最终 BF16 转换和 16 位全局写回；第一个完整输出计算体两臂均为 128 IMMA、512 FFMA、512 FMUL。float kernel 还有 partial/fixup 体，整核静态计数不可用于声称数学工作量减半。SASS 没有显式 `.RN` token，RN 依据来自本机 CUDA header 和完整字节对照。详见 [sass/summary.json](../../tmp-nvcc/perf-ada-round4/sass/summary.json)。

Engine 使用已有 resident 模型、INT8 KV、ctx8192、prefill_chunk512、CUDA Graph、无投机。三 fixture 各 r1/warmup1，按 ABBAAB 三对串行运行；**18/18 benchmark 进程完成**，计入 prime、warmup、正式 repetition 共 54 次模型调用。配置与 requested129/generated129/decode128 均已核验。

| Fixture | Paired prefill 吞吐收益中位数 | Paired decode 吞吐差值中位数，仅描述 |
|---|---:|---:|
| pp512 + tg128 | +1.308057% | +0.132035% |
| pp2048 + tg128 | +2.417294% | +0.405347% |
| pp4096 + tg128 | +3.789646% | +1.254741% |

**9/9 prefill 对均为正**；三个 fixture 独立计算，未混合工作量。decode 路未修改，上表 decode 数字不作因果提升结论，也不将这些短 fixture 的收益直接外推为 213K TTFT。完整配置、stdout/stderr、JSON、GPU 前后温度/时钟/功率/显存回执见 [engine-bf16/runner-receipts.json](../../tmp-nvcc/perf-ada-round4/engine-bf16/runner-receipts.json) 和 [engine-bf16/summary.json](../../tmp-nvcc/perf-ada-round4/engine-bf16/summary.json)。

符合条件的输出每元素省去 FP32 plane 的一写一读，逻辑全局字节减少 `8*N*T`，并省去该批 plane 临时需求；这不等于 DRAM 字节按比例减少。本轮两臂 **Engine workspace 峰值和 capacity 均为 106,912,768 字节，未观察到 Engine 峰值下降**，不新增总显存或 Engine workspace 节省结论。

原 213K 六题使用冻结请求，sync KVMem、无 CUDA Graph，逻辑最大上下文 212,992，实际 prompt_tokens 为 212,276～212,441。ACK 和五个此前 full-KV 资格化的 exact/update/tool 事实全部终态正确；独立重组原始 SSE 后逐题与 expected 精确相等，各有 `[DONE]` 和 stop，HTTP200，无取消，server exit0。保留原歧义 paraphrase 题未资格化的边界。质量记录见 [long-213k-default/results.json](../../tmp-nvcc/perf-ada-round4/long-213k-default/results.json) 和 [manifest.json](../../tmp-nvcc/perf-ada-round4/long-213k-default/manifest.json)；该次质量验收不提供新的 213K TTFT 提升结论。

CPU-only [summarize.py](../../tmp-nvcc/perf-ada-round4/summarize.py) 已生成 [summary.json](../../tmp-nvcc/perf-ada-round4/summary.json)：从原始 logs 重新数 34 个 CPU PASS，完整读取并逐字节比较 27 对 BF16 文件及长度，从 18 份原始 Engine JSON 的时间重算 9 对吞吐，解析 Nsight 的实际 3→2 kernel 和 NCU 主核记录。**54 次 benchmark 模型调用包含 prime/warmup，另有 6 道语义质量题，不能合称 60 个独立正确样本**。额外默认/基线 probe 不混入主 34 case。

资源按主线程峰值记录：build 约 3.5G RAM/0 swap，qualified Op 1.8G/0，单 Op NCU 1.1G/0，Nsight Systems 1.6G/0，Engine 11.1G/6.5M swap；最终 213K service 为 4m53、MemoryPeak20G、swap1.8G，在 20G/2G 限额内完成，无新 OOM。资源标记为主线程记录，汇总脚本没有伪称重建该 service 读数。没有全模型 NCU replay。本轮不增加 NVMe 预取或历史恢复实现的完成声明。

第四轮采用完成，整体限时优化 goal 仍 active；用户暂停截止为 2026-10-21 01:04:03+08:00，本次记录时尚未到截止，没有执行暂停或自动恢复。截止依据仓库 `tmp-nvcc/optimization-deadline.json`，active 状态来源为主线程。


## 第五轮：复用现有 MTP 与 Graph 的 decode 配置

已完成同一binary、固定raw corpus、ctx8192/resident/INT8/chunk512/Graph的AB/BA/AB成对测量。base→MTP3在pp512/2048/4096+tg128的成对decode收益中位分别为+83.7945%/+194.7537%/+211.9326%，实际约43–44→80/128/135 output tok/s；prefill分别-2.1203%/-0.4952%/-1.1621%。每矩阵18进程、54次Engine调用（含prime/warmup）、9对，完整129实际生成IDs均相等。不能把54次调用记为54条事实质量样本，不能外推为普通对话或213K吞吐。

K3→K5另一个成对矩阵在相同三fixture的decode收益中位为-6.8425%/+20.3403%/+29.7508%，prefill -1.0608%/-1.0800%/-0.5404%，9对129 IDs也全等。K5低接受率fixture退化；保留K3常用显式配置、K5高接受率可选配置，全局spec默认off。不将两矩阵中位连乘成正式base→K5收益。

原213K冻结六题新增K3+Graph、K5+Graph两arm，均PASS6/6、终态完整、server exit0、最终在飞0。六答案ACK/silver-maple-41/amber-willow-92/copper-lotus-35/copper-lilac-28/teal-jasmine-61。六条request_done合计K3为11轮、33draft/33accepted、perpos[11,11,11]、fallback0；K5为9轮、45draft/44accepted、perpos[9,9,9,9,8]、fallback0。原题短输出仅证明质量，不作长TTFT收益结论。首次K3因backing目录未建而1.732秒0请求启动失败已保留，另建cold-ready目录后通过。

bench新增可选NINFER_BENCH_TOKEN_IDS_PATH，在Engine计时之外直接保存真实生成IDs TSV，默认不启用无IO。补丁与测量二进制保存。报告：`tmp-nvcc/perf-ada-round5/report.md`；两矩阵summary及两个long results同目录。后续轮次另记，本轮不增加async/256K/媒体验收主张。


## 第六至八轮的边界

孤立ldmatrix x4候选数值正确但完整Op退步13.5%–14.2%，已恢复原生产内核；[第六轮报告](../../tmp-nvcc/perf-ada-round6/report.md)。Engine chunk1024单点没有可信正收益且workspace增81.4MiB，保留512；[第七轮报告](../../tmp-nvcc/perf-ada-round7/report.md)。完整producer/worker流水另作第八轮：算子两几何中位+47.11%/+50.21%，16case/arm与完整BF16字节一致；NCU受4GiB隔离限额未完成，Engine配对未运行，默认关闭。新binary原213K验收终态见[第八轮状态](../../tmp-nvcc/perf-ada-round8/status.json)，不沿用第五轮通过结果，也不将算子速度外推为端到端TTFT收益。


第八轮原题终态补充：最终新Serve/pipeline=1/MTP3+Graph/sync的原213K冻结六题6/6正确完整，serverexit0，service4m27.085s、Host20G/1.9Gswap。Engine配对NOT_RUN、NCU4GiB采样未完成；因此默认0，尚未采用。以perf-ada-round8/report.md及status.json为准。


第九轮续记（2026-10-21，用户恢复后）：已补齐完整INT8 producer/worker流水的采用证据，SM89 plain INT8默认NINFER_PROMPT_I8_PIPELINE=1，显式0恢复旧schedule；packed/rotated/SM120原路径保留。相同旧缺省binary、MTP3 optimized/Graph/INT8/ctx36864/chunk512，pp32768+tg16三对ABBAAB prefill分别+5.926524%/+4.750683%/+4.764359%，中位+4.764359%，每对减少847–1057ms；六进程各17个生成ID完整相同，workspace均106912768B。pp512+tg16补充三对中位+0.099936%，含一对-1.221137%，只记无稳定短档收益。decode的16-token伴随观测不作专项成果。沿用数学未变的R8两arm独立oracle/完整字节与新Serve原213K6/6，不重复原六题，也不外推整条213K TTFT。2,014,048B小harness直接实例化实际PagedKVBatchMetadata<false>生产内核，在T512/history32768下6,291,456B完整输出相同；NCU basic与targeted全部4GiB/0swap终态通过，峰值低于700MiB，解决先前分析器隔离OOM。两arm均128regs/93696B shared，occupancy33.33→33.12%，barrier15.452388→11.104926%，但local load/store sectors增加，不能声称无spill或occupancy提高。最终采用构建43.446s/3.6G/0swap，env未设的公共Op实际launch命中新流水。Engine长矩阵4m3.598s/14.2G/0swap、短矩阵22.484s/2.1G/0swap，cgroup文件页计费差异不等于模型内存优化。证据在perf-ada-round9/report.md、status.json和service-receipts.json，计时binary及改默认前source已保存。原一小时暂停已履行，当前用户恢复、不续设闹钟、不关机；OpenSpec15.4继续未完成，不宣称设备最优。下一decode小候选先查实际MTP小列热点；144个GDN临时copy旧trace合计仅122.828us，约0.5%局部理想空间，不为节点数盲目扩API。


第十、十一轮续记（2026-10-21）：优先复用已有源码继续有界筛选，两个候选均拒绝并恢复，R9默认完整INT8流水保持。R10仅IQ4_XS/SM89/nonFused/plainBF16Store/T6和两大MLP桶，复用已有KS0 runtime-K以减少展开寄存器；四个现有CPU oracle通过、随机完整208896/61440B相同，实际小NSYS命中kernel<23,6,false,0>。同binary warm CUDA-event完整Op三对/桶全负，中位-1.293661%/-2.283654%，不扩大Engine/NCU/原题。候选source/patch/binary保留，恢复后确实重编Op，27.062s/432M/0swap；第一次保留旧时间戳导致ninja no-work未作恢复证据。R11复用KVMem Ring/Fusion只缓存两组Q scales；standalone编译两arm均128regs/40Bstack，SASS发生变化。提前kernel-only warm T512/history32768/H24KV4三对均负，中位-2.244749%；6291456B接线输出相同不冒充独立数学oracle。因为早筛未过，没有编译生产候选Op/Engine、没有重复独立oracle/NCU/原213K六题；两生产文件已恢复到已构建R9 baseline相同字节，故无须另作生产重建。两轮都没有新OOM，全部自有句柄终态。完整结果与未运行边界分别在perf-ada-round10/report.md及round11/report.md。OpenSpec15.4继续未完成，不将寄存器数降低或一轮筛选当成达到最优。


第十二、十三轮续记（2026-10-21）：先采集当前R9/MTP5真实NSYS：decode 22次GraphLaunch、39,314 kernels，GPU总739.742ms；vec占87.438%，IQ3_S实际是T6，不能拿旧base/T1热点冒充当前投机路径。prefill GPU 2054.058ms，MMQ（含量化/转换）占86.956%。该profile只用于归因，不作clean性能采用计时。IQ3整数链拆分及IQ2_S direct-BF16扩展已准备，尚未生产接入/资格化。

复用已有MTP7（未增大框架支持上限），固定pp4096+tg128、ctx8192/resident/INT8/chunk512/optimized/Graph同binary K5/K7三对ABBAAB，decode收益+11.526970%/+9.034183%/+11.133378%，中位+11.133378%。六进程各129个实际生成IDs完整一致，也与保留的非投机129 IDs相同。K7每测量17轮、117draft/111accept、fallback0；仅此高接受率fixture，未测短档K7、不连乘R5中位数、不设全局spec默认。

K7原213K首次首请求虽输出ACK，但SSE无终态并报tokens32706需512页/仅511页；后5题未发送，不能算质量通过，非OOM。根因是历史压实只恢复text增长entitlement，backend的mapped+reserved限额被缩小；K5同位置只需32702 token因而未触发。修复保存backend自身mapping_limit，复用现成resize_sequence_kv_entitlement恢复整个bundle，保留页覆盖检查、原题及32K预算。bench help/README同步真实MTP1..7。修复构建44.376s/2.5G/0swap。

修复后新目录原213K六题6/6正确、HTTP200/terminal/usage完整、serverexit0，服务4m28.596s/20G/2Gswap、无新OOM；718次压实、3010次Cold spill、36998次source-K重建，history replay185836/reuse10240/query replay162，在飞0。原失败目录完整保留。K7现在是显式高接受率配置可用，性能数字来自此前resident矩阵（该路径不执行修复分支），不伪称修复后重新计时，也不增加TTFT或P4性能主张。证据在perf-ada-round12/current-profile-analysis.md及round13/report.md/status.json/service-receipts.json。15.4继续未完成。


第十四轮续记（2026-10-21）：已采用SM89/PageMajor/Main16layer/H4不可变source-K的层合并发射，复用现有context_kv_materialize按值pointer table/grid.z模式，逐token仍调用原H256/RoPE/INT8-G64 helper；MTP1及其他几何原路不变。两arm各4独立CPU oracle通过，18,384,064B完整pool dump相同，来源first-write/重复恢复/guard/V/tail/release保持。首次standalone漏RDC与RoPE owner导致baseline oracle失败的零模型调用尝试保留，补齐生产RDC链接后通过，没有放宽数学容差。

Main16/full64/相同单staging、buffered暖文件cache的完整restore Host wall三对ABBAAB：208.332→160.2575、210.6385→161.934、207.6095→163.194us；latency减少23.075908%/23.122316%/21.393771%，中位23.075908%（throughput中位+29.998284%）。计时包括pread/H2D/rebuild/sync，不是kernel-only；不能称物理NVMe冷读或整条TTFT同幅收益。小NSYS实际layer kernel1次，仍有32次H2D/1,081,344B，不声称copy已合并。

新Serve/K7/Graph/sync的原213K六题6/6正确完整，serverexit0；service267.651s/20G/2Gswap，非OOM；718压实/3010Cold spill/36998source-K restores，history replay185836/reuse10240/query162，在飞0，与R13工作量一致。两次长题为单arm质量gate，不能用约4m28→4m27报TTFT加速。最终NINFER_KVMEM_SOURCE_LAYER_BATCH默认1、显式0回退；生产构建33.440s/3.8G/0swap终态通过，未设env实际生产TU小NSYS再次命中一次层合并kernel，所有自有句柄终态。原题只在候选新代码下验收一次，不为host缺省切换再跑六题。报告与精确回执：perf-ada-round14/report.md、status.json、service-receipts.json。

后续R15：IQ2_S补入既有direct-BF16资格范围，已生成真实weight000211独立CPU fixture（28.55MB encoded只读、0.711s/104.9MiB/0swap），可只编译三个scratch对象做Op早筛；生产未改、GPU未运行。R16：已找到ninfer-3090现成CUDA13 copy_batch，可在full64/PageMajor连续plane边界批提交32个H2D，partial/pitched仍须原路；只完成源码/API调查，未实现。P4/整体TTFT新的配对收益仍未验证，OpenSpec15.4继续未完成，目标active，不宣称设备最优。


第十五轮续记（2026-10-21，用户再次明确优先Qwen3.8 GSQ）：仅使用既有qwen3_8_gsq_rco_iq3_s_v3.ninfer。IQ2_S补入既有direct-BF16两级名单，保持完整K/量化/FP32累加/最终BF16边界。三对象scratch构建避免重编其它format，真实N17408/K5120/T512完整linear Op三对ABBAAB收益+6.163656%/+6.868316%/+6.630609%，中位+6.630609%；独立constant/scaled数学候选8个唯一case及baseline首case通过，覆盖另一实际N6144、T257/513/1024子批偏移、fixup/J16回退，5对random完整80,261,120B/arm相同（不冒充random独立oracle）。

Engine新增仅IQ2_S的进程冻结实验selector，两个arm均保持已有六格式BF16=1，避免全局0/1混入R4收益。固定GSQ pp4096+tg128/MTP7/Graph/INT8/chunk512，三对prefill为+0.610787%/+0.273332%/-0.764661%，中位+0.273332%，六run完整129 IDs一致；没有稳定整模型prefill收益，decode伴随数不作本候选成果。因此状态OP_QUALIFIED_OPT_IN_NOT_DEFAULT，NINFER_GGUF_IQ2_S_MMQ_BF16默认0，已资格化实验入口保留，暂不默认采用，也不为它重复原213K六题。生产其它默认保持，后续继续GSQ更大prefill热点，不把算子6.63%外推整模型。

资源：scratch构建16.016s/554MiB、首Op2.837s/449.5MiB、边界Op4.546s/412.9MiB、正常生产构建51.830s/1.4G，均0swap；Engine53.925s/13.1G/0swap。实际生产selector0/1探针通过，但该两小进程未采RAM/cgroup，不伪造受限服务资源回执。详细数值和未运行边界见perf-ada-round15/report.md、status.json、service-receipts.json。15.4继续未完成。


第十六轮续记（2026-10-21，Qwen3.8 GSQ）：已采用SM89/CUDA13/Main16/H4/PageMajor/full64 source-K的批次H2D提交，直接适配ninfer-3090/Infernix的CUDA13 copy_batch，保留Apache-2.0来源。partial、capture D2H、MTP1和其它geometry维持旧copy2D；全部API错误传播，不吞错假装成功。描述数组约768B普通Host元数据，无新pinned payload/Device buffer，原load→copy→rebuild→sync→单staging复用生命周期保持。

固定R14 layer-batch=1，同binary copy-batch0/1三对完整restore Host wall分别163.874→141.5235、160.2235→141.2945、160.8355→139.076us；latency减13.638832%/11.814122%/13.529041%，中位13.529041%，throughput中位+15.645762%。两arm各4独立CPU oracle、18,384,064B完整dump及first-write/guard/repeat/release均通过。小NSYS实际batch API1次、layer kernel1次，GPU仍32段H2D/1,081,344B；不是32段物理DMA合一，不推断CE专用/零SM/与同页rebuild并行，不乘R14/R16中位数，不外推TTFT。

新Serve原213K6/6精确且完整，MTP7/Graph/sync/W32768/physical36864/chunk512，source-copy1/layer1、IQ2_S候选显式0；server/service exit0。用时261.465s/20G RAM/2Gswap，非OOM；718压实/3010Cold spill/36998source-K restores、history replay185836/reuse10240/query162，在飞0。没有靠减少恢复或历史重放量换性能。三次long单arm总时长不能当TTFT成对提速证明。

最终NINFER_KVMEM_SOURCE_COPY_BATCH默认1、显式0回退。采用构建44.533s/2.5G/0swap，实际生产header的默认probe构建6.138s/560.5MiB/0；两个source env未设的小NSYS仍真实命中batch API与层合并，2.333s/1.1G/0，所有自有GPU/build/Serve句柄终态。详见perf-ada-round16/report.md/status.json/service-receipts.json。R15仍仅Op-qualified默认0；R17只准备真实GSQ IQ3_S I64/J128/4warp候选，尚无编译/性能结论。P4新测、整体TTFT稳定收益及设备最优仍未证明，15.4继续未完成。


## R17：IQ2_S与IQ3_S优化共用（2026-10-21）

现有Qwen3.8 GSQ本来就混合使用IQ/QK格式；不同权重按原格式选择内核，权重和精度不变。IQ2_S direct-BF16与IQ3_S I64可同时开启，也继续使用已采用的其它格式direct-BF16、INT8 prompt流水及source-K恢复批处理。收益须测完整组合，不能把单算子6.63%和7.18%相加。

固定本机4070 Ti SUPER、已有GSQ、pp4096+tg128、MTP7 optimized/Graph、INT8、ctx8192、chunk512，另做同binary BOTH0/BOTH1 ABBAAB三对，prefill为 **+1.835805% / +1.275941% / +0.210722%**，中位 **+1.275941%**，中位节省 **25.991701ms**，六次完整129 IDs相同。单独I64的Engine结果仍有负pair，不能用组合覆盖原结论；组合不是四格交互实验，也不证明线性叠加或cold TTFT收益。

生产已接入新的受限IQ3_S入口，两个全局默认仍0。要为这个GSQ制品使用已资格化组合，在**进程启动前**同时设置：

```sh
env NINFER_GGUF_IQ2_S_MMQ_BF16=1 \
    NINFER_GGUF_IQ3_S_MMQ_I64=1 \
    ./build/bench/ninfer_bench \
    --weights /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
    --max-ctx 8192 --prefill-chunk 512 --kv-dtype int8 \
    --spec mtp --draft-tokens 7 --lm-head-draft \
    -pg 4096,128 --warmup 1 -r 1
```

也可与已有Serve命令一起通过env传入；显式0分别回退。IQ3_S I64仍限SM89、N17408/K5120/T512/row_bytes2200/连续BF16，不扩大到其它N/K/架构。global NINFER_GGUF_MMQ_BF16继续保持默认1。其它已采用优化照常保留。

正常构建和生产Op资格通过，生产随机17,825,792B与已测scratch逐字节一致。该新生产binary两项均1的原213K六题 **6/6正确且完整终态**、server/service exit0；20G RAM/2Gswap限额内正常结束，263.980s，峰值达到20G/2G，无新OOM。在飞0，718压实/3010Cold spill/36998source-K恢复、history replay185836/reuse10240/query162保持。只跑一次新组合质量arm，未增加长档配对性能、P4或256K/媒体矩阵。

组合状态为 **QUALIFIED_EXPLICIT_COMBINED_GSQ_PROFILE**。完整数字、独立I64未采用结果、小NCU权限失败与trace预算不足的保留记录见[第十七轮报告](../../tmp-nvcc/perf-ada-round17/report.md)。后续[第十八轮准备](../../tmp-nvcc/perf-ada-round18/README.md)针对实际MTP7/T8 decode热点，尚未构建或测候选；15.4继续未完成。


R18执行终态（2026-10-21）：复用4080逐列消费循环组织，IQ3_S/N17408/K5120/T8维持原decode/dot/规约，在scratch独立kernel减少activation同时存活。独立oracle两例、constant和random完整278528B/arm均一致，但同binary完整Op三对75.584→75.712、74.976→75.808、74.752→75.776us，全部负收益、中位-1.097509%。虽然ptxas寄存器182→80，不能因此推断性能；无runtime local traffic/瓶颈因果结论。候选REJECTED_NO_PUBLIC_OP_GAIN，生产R17不动，不追加NCU/Engine/原213K。构建37.906s/701.2M/0swap（8G/0），screen2.978s/489.4M/0（4G/0）。真实MTP7 NSYS仍用于定位，15.4未完成。下一prefill重点为实占约12.014175%的IQ3_XXS大MLP完整K桶，IQ2XXS新增direct-BF16仅约0.88%桶，降优先；尚无新采用结论。


## R19：增加IQ3_XXS的I64投影（2026-10-21）

现有Qwen3.8 GSQ的IQ3_XXS大MLP目标实际为312次/246.199915ms，约prefill GPU kernel sum的12.014175%。保持现有量化、K累加和最终BF16边界，复用GGML loader/MMA/writer及R17独立TU几何。完整Op三对中位 **+6.258693%**；2独立oracle和完整17,825,792B/arm random字节一致，小NCU证实实际100KiB carveout/2CTA资源条件，不把CTA数当warp数翻倍。

在R17已资格化的两格式组合上固定原两flag=1，仅切新增 `NINFER_GGUF_IQ3_XXS_MMQ_I64=0/1`，同binary **pp4096+tg128、MTP7/Graph、INT8、ctx8192、chunk512**三对prefill增量为 **+0.390812% / +0.849449% / +1.110583%**，中位 **+0.849449%**、中位节省 **17.13358ms**；六次129生成IDs完整一致。该数字来自集成前scratch Engine配对，生产集成仅将相同私有配置改为TU局部macro；生产probe与scratch完整字节相同，未重新跑生产paired性能。不能与R17中位数相加/相乘，也不认领decode或213K TTFT加速。

当前显式三格式GSQ配置在已有命令启动前传入以下三个env，**三个全局默认均0**，各自设0回退：

```sh
env NINFER_GGUF_IQ2_S_MMQ_BF16=1 \
    NINFER_GGUF_IQ3_S_MMQ_I64=1 \
    NINFER_GGUF_IQ3_XXS_MMQ_I64=1 \
    ./build/bench/ninfer_bench \
    --weights /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
    --max-ctx 8192 --prefill-chunk 512 --kv-dtype int8 \
    --spec mtp --draft-tokens 7 --lm-head-draft \
    -pg 4096,128 --warmup 1 -r 1
```

IQ3_XXS I64只覆盖SM89/N17408/K5120/T512/row_bytes1960/连续BF16；其它shape/架构/FP32/fixup保持原路。既有INT8 prompt流水及source-K layer/copy batch仍启用。正常生产构建与unset/candidate独立oracle通过，新生产三flag1的原213K六题 **6/6正确且终态完整**、server/service exit0；262.524s，Host峰值20G/swap2G，限额20G2内正常结束，非OOM。在飞0，718压实/3010Coldspills/36998sourceK恢复及history185836/reuse10240/query162保持。

状态 **QUALIFIED_EXPLICIT_THREE_FORMAT_GSQ_PROFILE**，不是global默认采用。详见[第十九轮完整报告](../../tmp-nvcc/perf-ada-round19/report.md)。R18 decode逐列消费已拒绝，寄存器下降没有带来完整Op收益；R20 T16组件和R21 query-only replay路线仍仅CPU准备/调查，未改MTP上限或历史恢复语义。15.4继续未完成。


## R21：原位置 query-only 恢复试验失败（2026-10-21）

完整原位置KV/原时间线q0 checkpoint路径已做成独立启动实验，默认NINFER_KVMEM_ORIGINAL_POSITIONS=0，已有canonical路径保留。生产构建通过；两个小storage用例通过（2,061,312B K/V/scales与24,576个CPU mean值exact、future撤销、partial tail、stale身份拒绝、guard与释放均通过）。

新模式的原213K验收失败：ACK通过，首事实期望silver-maple-41，实际copper-lilac-28；六计划/两发送/一正确，已按门槛停后四题与所有性能配对。两次完整q0 StateImage字节核对通过，不能因此断言语义质量正确。错误首输出是Main prefill产生的c，不能仅归因后续MTP AR。服务142.123s，RAM上限20G、swap626.1M（上限2G），server正常0、非OOM。

保持实验模式关闭，不采用该模式，不继承R19的6/6，也不把更少重放作为性能成果。失败证据与补丁在tmp-nvcc/perf-ada-round21/；质量未通过，TTFT三对脚本处于BLOCKED_BY_QUALITY。原题/权重/工作集未改。当前继续研究源码中的具体遗漏，并优先推进保留canonical数学的global Host KV缓冲解耦。


R22实际修复及验收（2026-10-21）：已解开KVMem逻辑descriptor下限与global Host payload的耦合，并修复root admission将独立KVMem归档虚报为global Host debt；真实global Host副本/demotion/restore及KVMem自己Host/Cold配额不删。HostKV0的当前canonical三格式GSQ原213K六题6/6、完整终态、server0；248.378s/20G峰值/swap776.3M（20G/2G限制），非OOM，global Host容量与占用0。history185836/reuse10240/query162、compaction718/Coldspills3010/sourceKrebuild36998、在飞0，与R19工作量相同。初次metadata-only版本的准入拒绝保留，未改题也未增大工作集。

R22同一最终binary、仅global HostKV8192/0的ABBAAB三对已全部终态，6fresh processes/36原请求全部正确、所有server/service exit0；unit ninfer-r22-ttft-paired/session59798结束，26min27.490s/整矩阵20G峰值/swap2G。首题长TTFT三对内减少中位-0.110288%，没有稳定改善；后五题TTFT合计三对均正、中位减少2.262371%/2126.038ms。当前C1/canonical/三格式GSQ显式长上下文profile采用--host-kv-mib0，全局默认不变，不宣称首次长prefill加速或RSS少8GiB；原KVMem归档/StateImage保持。R21新原位置模式仍失败且默认0，16.3/16.4和15.4继续未完成。详见目标仓tmp-nvcc/perf-ada-round22/report.md及ttft-paired/ttft-summary.json。

## R24、R26–R27：先理论、批量修改、统一验收（2026-10-21）

R24两槽source-K read-ahead已通过原213K六题6/6、HTTP/terminal/usage完整、server0，251.417s/18.2G峰值/0swap；`NINFER_KVMEM_SOURCE_READ_AHEAD=1`是已资格化opt-in，默认0。warm source恢复三对用时减少中位1.935888%，不能冒充物理NVMe冷读或长TTFT收益。source staging2297856B，canonical history185836/reuse10240/query162、compaction718、Coldspills3010、source restores36998保持。详见[R24报告](../../tmp-nvcc/perf-ada-round24/report.md)。

用户要求先计算理论空间和代价，多个相容部分一次修改后统一验证，整包失败再拆。R26据此组合线程池/四槽/两页重建及组同步/两格式sign LUT，完整数值和生命周期门通过；ABBAAB六fresh Serve/36原请求全部正确、退出0、工作量一致，25min30.839s/整矩阵20G峰值2Gswap。resident decode吞吐中位-0.520329%；长首题用时减少-0.399626%，后五题合计+0.689703%（两正一负），session wall减少-0.284176%。不采用整包，默认0。36998是restore调用数而非kernel数；两页都moved才少一次重建launch，额外staging2297856B。精确实测binary保存在round26/qualified-bin，旧失败与原题保留。见[R26报告](../../tmp-nvcc/perf-ada-round26-bundle/report.md)。

R27一次合并三格式canonical reduce-scatter/合作BF16 Store和Main16四head warp source kernel。三个格式12个无计时Op、6个独立constant oracle、六对完整278528B输出及真实route通过；source两臂各4个独立CPU case、每臂36766912B完整池相同。source新kernel实际34regs/旧35、无spill，但whole restore三对吞吐+4.099892%/+18.487074%/-6.019211%，未稳定，已只撤回source CU。保留三个decode格式，resident三对吞吐+1.601048%/+1.775624%/+0.891646%，中位+1.601048%，6×129 IDs相同；伴随prefill-0.677477%，不认领prefill/TTFT收益。重新链接后GGUF库字节不变，不重复该数值/性能矩阵；只跑一次新Serve decode-only原213K六题，6/6精确正确、HTTP200/terminal/usage完整、server0。245.820s/18.7G峰值/170.6M swap（20G/2G限制），非OOM；canonical history185836/reuse10240/query162、718compactions/3010Coldspills/36998restores及staging2297856B保持。状态QUALIFIED_EXPLICIT_THREE_FORMAT_DECODE_PROFILE，最终三个binary保存在round27/qualified-bin；性能继承相同GGUF模块，不伪称最终binary重跑配对。

R26/R27全局默认均0，R27候选固定R26=0。source原型及理论证据保留，不把理论驻留翻倍当成实测occupancy或吞吐翻倍。15.4、R21原位置验收及新P4性能均继续未完成。见[R27报告](../../tmp-nvcc/perf-ada-round27-bundle/report.md)。


R28批量筛选（2026-10-21）：先计算真实热shape与流量/寄存器/shared成本，再一次合入四个I64/J128完整K MMQ shape和GDN Q/K/V直接打包输出，共七文件，NINFER_R28_BUNDLE缺省0。八个公共MMQ独立constant oracle、GDN七种实际格式/T512/17/8的独立constant oracle与完整153,990,144B输出/guard/Graph比较通过；down规约改变产生少量BF16差异，未承诺bitwise数学不变。

同一binary统一做两组整体ABBAAB，每组六次：短pp4096+tg128的prefill/decode吞吐中位分别−1.114718%/−1.574450%；长pp32768+tg16分别+0.873695%/−0.404652%，长prefill三对有正有负。短129和长17个完整输出IDs各自六次一致，投机工作量各自相同。状态MATH_PASS_REJECTED_NO_STABLE_OVERALL_GAIN，R28=0，R27=1/R26=0基线保留；不为负组合追加原213K、TTFT或P4矩阵，也不逐格式做长性能筛选。构建256.935s/4G/0swap；统一数学5.731s/912.7M/0；短56.021s/13.5G/0、长236.425s/2.1G/0（服务内存峰值不可当作进程RSS比较），全部terminal0。精确实测binaries、配对原值、失败采用门与收据保存在tmp-nvcc/perf-ada-round28-bundle；原题质量NOT_RUN，不能继承旧6/6作本批验收。没有提交推送。

后续R29仍按用户“理论先行、多项相容改动后统一构建/测试”推进：优先复用现成T16 MMQ尝试固定MTP15，并只在原fullK/no-fixup几何合法时直写BF16；同时准备保持BF16舍入边界的SiLU/down激活量化融合。当前仅研究和CPU overlay，尚未生产构建、数学/Graph/质量或性能验收。R21原位置模式仍失败且默认0，15.4及16.1–16.4不勾选；不存在新的暂停期限或关机授权。


R29 / v1.0.0 发布验收（2026-10-21）：按用户要求先算理论与资源成本，将 fixed MTP15/T16、fullK/J16 BF16 writer 与保持BF16舍入的SiLU/down量化融合一次合入19文件后统一构建与数学验收。50次Op（T16十格式含完整Q4_K head22次，融合七格式constant/random28次）和K15混合接受前缀/guards/frame通过。初次eager129IDs正确，Graph在拓扑切换失败；只对明确topologyChanged先实例化替代exec后换，旧失败保留，修复前后Ops/GGML/Engine库字节未变，不重复数学矩阵。

最终同binary A显式六flags0/K7、B六flags1/K7、C六flags1+R29/K15，ABC/CBA/ABC，4K与32K各9进程、warmup1/r1/tg128；每档完整129IDs九次一致，短C Graph与已验eager投机工作一致。B/A短prefill吞吐中位+5.232890%（三对正）、decode+.761975%（两正一负）；C/A长prefill+9.582095%、decode+11.779491%（均三对正）。C/B短decode−15.183451%，长+10.219626%，故采用两个显式profile：短K7、长32K档K15；无通用自动阈值或普遍K15收益。

最终K15/Graph/sync/W32768/物理36864/HostKV0原213K六题6/6精确、HTTP/terminal/usage完整、server/service0；248.360s、cgroup峰值16.9G/swap329.5M（20G/2G，非OOM）。整组history replay185836/reused10240/query162、718压实/3010Coldspill/36998sourceK恢复、staging2297856B保持。单臂API TTFT首题152.592s、后五题中位17.715s，仅作观测，不认领配对TTFT或新P4改善；原位置实验仍失败且默认0，15.4和16.1–16.4仍未完成。

全新README、CMake软件版本1.0.0、完整整体原值与原题/数值证据纳入源代码版本；公开报告见 docs/releases/v1.0.0.md 与 docs/releases/v1.0.0/。本机完整失败和精确二进制保存在 perf-ada-round29-bundle。权重、二进制、Cold后备、profiler数据和外部所有权q4诊断改动不纳入源码提交。发布推送/tag状态以实际收据为准。
