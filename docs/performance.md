# Single-GPU serving performance

本机4070 Ti SUPER持续调优记录见[当前原生Qwen3.8测量](maintainer/2026-10-20-kvmem-actkv-live-validation.md#持续调优完整尺寸与测量纪律)。
当前模型是保留GGUF量化块编码的`.ninfer`原生制品，由NInfer C++/CUDA Engine执行。
完整尺寸CPU oracle已覆盖15组N/K、105个case；更大MMQ提交宽度和共享激活候选尚无采用证据，生产实现保持。
同一二进制三对交替测量确认现有Graph在限定短上下文夹具的decode收益，不能外推到213K或硬件极限。

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded historical measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to the relevant result section. “Not published” describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | Not published |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. The Qwen3.8 NVFP4 saturation reports retain configuration and
values but no tested Git revision; the model page records that provenance limitation.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 DFlash2 outcomes](performance/qwen3.8-27b.md#dflash2-completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.

## 2026-10-20 Ada KVMem async screening: keep sync default

A bounded local comparison used Qwen3.8 GSQ v3 on one RTX 4070 Ti SUPER
(16,376 MiB), driver 615.78.08, CUDA 13.4.92, `sm_89`/66 SM build. This is a
separate Ada measurement, not an update to the RTX 5090 model tables above.
The same binary and complete artifact served both arms, changing only
`--kvmem-transfer-mode sync|async`. Configuration: managed weights, INT8 KV,
8,192-token logical ceiling, 2,048-token physical pool, 1,024-token selected
window, 256-token growth reserve/prefill chunk, one text request, greedy,
no thinking/Graph/speculation. The external-owned Q4 source was not changed.

Inputs and order were frozen before inference: three interleaved pairs in each
of three scenes, **18/18 sessions and 54/54 HTTP requests completed**, no retries
or extra samples. Each session starts a fresh process without inference warmup.
Session wall time starts after readiness and covers every replay request;
model startup and receipt writing are excluded. Async initializes its pinned
Host arena at startup, whereas sync initializes it on first compaction, so these
request-latency numbers must not be attributed solely to DMA overlap. Fixed
assistant histories keep replay inputs identical across arms.

| Scene | Actual prompt tokens | Median session sync / async (ms) | Median paired wall reduction | Initial streaming TTFT sync / async (ms) |
|---|---:|---:|---:|---:|
| Shallow | 32 | 237.52 / 237.31 | **-0.10%** (slower) | 136.06 / 136.79 |
| First overpool | 2,719 | 1,910.69 / 1,856.98 | **2.90%** | 1,855.75 / 1,802.58 |
| Deep, seven turns | 4,969→5,160 | 5,390.27 / 5,323.44 | **1.50%** | 3,228.43 / 3,165.86 |

The paired percentage is the median of the three paired ratios, not the ratio
of the two separately computed medians. Deep paired reductions were 1.496%,
0.560% and 1.500%. All outputs, token counts, compaction counts, stage-in/out
counts and merged transfer-run counts matched. Deep native decode-phase time
summed across seven turns had median 1,145.03 ms sync / 1,133.21 ms async, with
51 completion tokens in both arms (including the per-request initial tokens).

Deep async submitted/completed 33/33 batches, moved 274,661,376 bytes through
47 runs (96 stage-out and 31 stage-in pages), and ended with zero in-flight or
Cold bytes. Median exposed async wait was **1.569 ms per complete session**;
submission wall time was 3.806 ms and the completed H2D phase 5.236 ms. The latter
includes overlapped resident rephase and must not be added as a disjoint cost.
Post-initial-turn median streaming TTFTs were 157–162 ms async and 158–161 ms
sync; TTFT includes preparation/prefill/transport and is not pure paging wait.
Per-turn values and every raw timing are retained in the receipts/analysis.

**Quality remains a blocker for broader historical-recall claims.** Shallow and
ACK control requests matched their expected answers. In each deep session, only
**1/6 historical-label questions** was correct in either arm (the ACK control
makes 2/7 requests correct). The other five outputs were identical failures,
including an abstention and invented labels. No full-KV control was run for this
performance corpus; it is not a new recall-rate acceptance. The separate frozen
P0 panel remains full KV 8/8, old compressed 2/8, candidate 4/8.

Decision: the deep improvement is below the frozen **10%** gate, although the
shallow paired median regression stays below 3%. Keep `sync` as the default and
retain `async` as opt-in experimental. The exposed wait gives no evidence that
in-flight depth is the limiting factor; **no depth=2 candidate, extra samples,
or Graph/MTP performance-confirmation pairs were run**. No async default is
being proposed for those deployment routes. Larger/longer workloads, other
artifacts and semantic recall require their own qualification.

Reproduce the frozen protocol with `python -m tools.bench.kvmem_async freeze DIR
--model /explicit/qwen38.ninfer --binary build/apps/ninfer-serve`, then `run DIR`.
The runner directly reuses `tools.ninfer_serve` / `tools.streaming_http`, refuses
changed inputs/binary and refuses an existing result ledger. This run's evidence
is `tmp-nvcc/p4-20261020-screen/{manifest,results,summary,analysis}.json`, commands,
server logs, metrics and raw SSE receipts. The original failing recall requests
and intermediate failed implementation probes remain preserved separately.

## 2026-10-20 Ada P4 refresh after recall and terminal fixes

The recall/terminal-repaired binary was frozen in `tmp-nvcc/p4-qualified-20261020/`.
**This measurement predates the subsequent immutable-source K repair. The newer
binary adds source capture/rebuild I/O and canonical selected-history replay;
its P4 is NOT_RUN. The original213K facts now pass sync/async12/12, but short-query
TTFT is23–26s; see [the repair report](maintainer/2026-10-20-213k-state-replay-repair.md).
The table below is
historical evidence and does not establish the newer binary's performance.**
The earlier measurement above remains historical. No download, build or other
GPU inference ran alongside this refresh. Same single 4070 Ti SUPER, Qwen3.8
GSQ artifact and original 18-session/54-request protocol/configuration; fresh
server per arm, no inference warmup, only transfer mode changes. The runner now
requires every expected answer and complete terminal before continuing, and
requires sync/async usage equality before any adoption decision.

**18/18 sessions and 54/54 requests completed, 54/54 answers correct.** All six
deep facts are correct in every session (36/36 fact answers plus six ACKs),
outputs and usage match across modes, and terminal in-flight pages/bytes are 0.

| Scene | Median sync / async session ms | Median paired reduction | First TTFT sync / async ms |
|---|---:|---:|---:|
| shallow | 224.63 / 225.35 | -0.32% | 129.68 / 129.40 |
| first overpool | 1851.97 / 1803.34 | 2.90% | 1797.59 / 1749.72 |
| seven-turn deep | 5138.78 / 5091.67 | 0.92% | 3087.83 / 3049.64 |

Deep paired improvements are 0.54%, 0.92% and 3.30%. Deep median summed
native decode time is 1059.53 ms sync / 1054.40 ms async. Async deep submits
and completes 35/35 batches, with 16 restore batches, 287,637,504 transfer bytes,
101 merged runs and 9 bounded fallbacks. Median exposed wait is 6.026 ms;
submit is 5.757 ms and restore phase 16.812 ms. Those overlapping phases must
not be summed as disjoint costs.

**Decision: keep sync default; async stays opt-in experimental.** Quality now
passes, but 0.92% remains below the predeclared 10% deep gate. Shallow regression
is within the 3% limit. No depth=2 or Graph/MTP performance-confirmation campaign
is triggered by these data. The current host/managed-weight request cost is not
a new kernel optimization claim; startup allocation asymmetry noted above still
limits attribution to DMA overlap. No extra samples were added after the result.

Evidence: manifests, commands, raw SSE/receipts, metrics, `results.json`,
`summary.json` and `analysis.json` in the frozen refresh directory. Longer-context
and media qualification use their own bounded correctness runs below this scope.


## Functional closure preceding the detailed performance round

The subsequent44-call Qwen3.8 functional round passes the original W4K/32K
counterexample, frozen W32K/256K facts, retained private branches and current
short Graph/MTP3/DFlash2 cases. It adds explicit managed runtime budgeting and
separate source/memory/internal-replay metrics; see the [live record](maintainer/2026-10-20-kvmem-actkv-live-validation.md#本轮最终功能结果).
At the user's request, TTFT reconstruction reduction, NVMe read-ahead,
large-copy splitting and P4 measurement belong to the next detailed tuning
round. This records the pre-tuning state. Current measurements follow; no default async adoption was made.


## RTX4070 Ti SUPER TTFT and page-pipeline tuning (2026-10-20)

Existing Qwen3.8 GSQ v3 weights and original questions are unchanged. Final
W32768/physical36864/prefill512 original213K calls pass12/12; current P4
passes18/18sessions and54/54correct complete requests. Full results, per-question
TTFTs, provenance and the retained profiling failure are in the
[live tuning report](maintainer/2026-10-20-kvmem-actkv-live-validation.md#详细性能调优ttftnvme读前sm89-mmq与当前p4).

- Canonical-prefix Host checkpoints avoid10240of196076history tokens per arm.
  Final short-query median TTFT is20.701s sync /18.963s async; historical full
  replay was24.256s /23.275s. These sequential observations have different
  memory constraints and are not an interleaved binary A/B.
- Two-page file read-ahead completes63/63tasks, with one additional outgoing
  staging page. Large Host copies complete986sub-batches,622of them earlier
  split batches; fallback24, terminal in-flight0. Source-K and Cold consumers
  can still synchronize.
- SM89 ordinary GGUF linear reuses the existing MMQ implementation with a
  maximum512column submission (previously128); workspace planning includes
  the existing all-tile fixup bound. CPU-decoded actual mixed formats pass
  T=1/16/128/129/256/512/513. N128/K5120/T512 samples improve2.94–3.34x;
  the native first-generate Nsight observation improves about9.1%, not3x.
  Production kernel SASS is identical. Other architectures retain128.
- Final P4 paired async reduction: shallow-0.704%, first-overpool2.922%,
  deep0.542%. Quality passes; deep misses10%, shallow stays within3%.
  **Sync remains default; async remains opt-in.** No extra samples/depth were
  added. Across sequential binary cohorts, sync deep-session wall median
  is9313.8→8777.1ms; this is a separate observational comparison.

A full-model NCU kernel replay caused a real global OOM during device-memory
backup and is preserved as a failed run. Do not repeat that capture strategy.
NCU now uses only an independent small Op fixture, with4GiB/zero-swap hard
limits (about994MiB observed peak). Final model jobs run in a transient systemd
service with20GiB memory /2GiB swap and OOMPolicy=kill. The final service
completed at those peaks, so Host-memory pressure is still a material limit,
not a claim of unified RAM accounting or resolved memory efficiency.

Evidence: `tmp-nvcc/perf-20261020/summary.json`, frozen long/P4 manifests,
raw SSE/receipts/metrics, Nsight Systems/Compute reports, CPU oracle and logs.
There were145model-call attempts,144complete correct calls and one incomplete
full-model profiling OOM. New256K/media/long speculative matrices were not rerun.

## SM89 direct final BF16 MMQ output (2026-10-20/21)

The existing NInfer-native Qwen3.8 artifact retains its IQ/QK block encodings.
MMQ now writes final BF16 directly for six qualified formats on SM89 when J128
uses complete output tiles without fixup. Activation quantization, FP32 sums
and the final rounding boundary are preserved; FP32 heads, MoE and split/fixup
products retain the previous route. Set `NINFER_GGUF_MMQ_BF16=0` before process
startup to restore the FP32 plane and separate conversion.

Same-binary ABBAAB, three pairs per prompt, resident weights, INT8 KV,
context8192, chunk512, CUDA Graph and no speculation:

| Workload | Median paired prefill throughput gain |
|---|---:|
| pp512 + tg128 | 1.308% |
| pp2048 + tg128 | 2.417% |
| pp4096 + tg128 | 3.790% |

All nine prefill pairs improve. Decode is unchanged by this route; its small
observed differences do not establish a decode speedup. The complete native
Op gains5.104–12.354% across nine full-size hotspots; standalone Nsight confirms
three kernels become two. This is not a same-sized speedup of the main MMQ
kernel or a claim about213K TTFT. Engine workspace peak/capacity remain
106,912,768bytes.

Independent CPU oracle34/34 and random complete-BF16-byte comparisons27/27
pass. Final-default original213K six questions pass6/6 with complete terminals,
serverexit0 and no in-flight pages. That is one sync quality run, not a new
async/P4/256K/media qualification. The18benchmark processes comprise54complete
model calls including prime/warmup; keep those separate from the six factual
questions. Long-run peak remains20GiB RAM/1.8GiB swap under20GiB/2GiB limits;
no new OOM occurred.

See the [live record](maintainer/2026-10-20-kvmem-actkv-live-validation.md#第四轮gguf-mmq-最终直接写回-bf16)
and `tmp-nvcc/perf-ada-round4/` for source reuse, raw measurements and boundaries.


## Ada 本机 MTP 配置资格（2026-10-21）

现有保留GGUF量化块编码的原生Qwen3.8、resident/INT8/ctx8192/chunk512/Graph。同binary ABBAAB每fixture三对，完整129 IDs均一致。

| Fixture | base→MTP3 decode中位收益 | MTP3→MTP5 decode中位收益 |
|---|---:|---:|
| pp512+tg128 | +83.7945% | -6.8425% |
| pp2048+tg128 | +194.7537% | +20.3403% |
| pp4096+tg128 | +211.9326% | +29.7508% |

两个矩阵分别配对，不能将中位连乘成三臂收益。短raw corpus结果不外推到普通对话；MTP本轮prefill有0.5%–2.1%小退。K3常用显式配置、K5高接受率可选；全局spec默认off。两配置各通过原213K冻结六题及终态/在飞归零。完整证据见[第五轮报告](../tmp-nvcc/perf-ada-round5/report.md)。


## Qwen3.8 GSQ：IQ2_S / IQ3_S组合opt-in（2026-10-21）

本机4070 Ti SUPER与既有GSQ制品，可在进程启动前同时设 `NINFER_GGUF_IQ2_S_MMQ_BF16=1` 和 `NINFER_GGUF_IQ3_S_MMQ_I64=1`；两个全局默认仍0，显式0分别回退。IQ3_S只改SM89/N17408/K5120/T512/连续BF16这一条已资格化路径，权重与精度保持。其它已采用格式及INT8 prompt流水保持。

同binary BOTH0/BOTH1、MTP7/Graph/INT8/ctx8192/chunk512、pp4096+tg128的三对prefill提升1.835805%/1.275941%/0.210722%，中位1.275941%（节省25.991701ms）；完整129 IDs不变。新生产组合原213K六题6/6完整、serverexit0。该有限fixture的组合收益不能由单Op百分比相加，也不是cold TTFT或设备最优结论。详见[当前实测与使用边界](maintainer/2026-10-20-kvmem-actkv-live-validation.md#r17iq2_s与iq3_s优化共用2026-10-21)。


## Qwen3.8 GSQ：三格式显式组合（2026-10-21，R19）

在上述IQ2_S BF16与IQ3_S I64均1的配置上，增加 `NINFER_GGUF_IQ3_XXS_MMQ_I64=1`。三个global默认均0、各自显式0回退。固定pp4096+tg128/MTP7/Graph/INT8/ctx8192/chunk512，同scratchbinary仅切新增flag，三对prefill增量+0.390812%/+0.849449%/+1.110583%，中位+0.849449%/17.13358ms，完整129 IDs一致。新生产三flag1原213K6/6完整，serverexit0，生产Op与已测scratch全字节相同。性能未重新在生产binary配对，不合成跨轮总百分比或长TTFT收益。当前仅显式GSQ profile资格，详见[完整使用边界及证据](maintainer/2026-10-20-kvmem-actkv-live-validation.md)。

## Qwen3.8 GSQ：canonical 长上下文 Host KV（2026-10-21，R22）

同最终生产binary、三格式flag1、W32768/physical36864/logical212992、INT8/chunk512、C1/MTP7/Graph，唯一变量为global `--host-kv-mib 8192/0`。ABBAAB六个新进程、36个冻结原请求全部正确且正常退出；独立KVMem Host/Cold归档与StateImage池保持。

后五题合计API首个非空输出TTFT三对分别减少4.078605%/2.262371%/0.639831%，中位2.262371%（2126.038ms）。首次完整长prefill三对减少-0.110288%/+6.720079%/-0.720526%，没有稳定收益，不作冷TTFT加速结论。当前这一显式长上下文profile使用`--host-kv-mib 0`，全局默认保持；不代表所有Host内存为0或RSS减少8GiB，不推广到需要global Host副本的其它场景。详见[第22轮报告](../tmp-nvcc/perf-ada-round22/report.md)。

## 批量优化与整体验收（2026-10-21，R26–R27）

按用户最新要求，先计算可消除的工作及额外资源，再合并相容改动；统一数值验证后测整体工作负载。整包无收益时才拆分，不为每项改动重复长题矩阵。

R26同时加入常驻读取线程池、四槽流水、两页重建/组同步及IQ3_S/IQ3_XXS符号表。数学与原213K的36/36请求通过，但resident decode吞吐中位-0.520329%，首次长TTFT用时中位增加0.399626%，后五题合计用时减少0.689703%且一对退步。整包不采用，`NINFER_R26_BUNDLE=0`保留。详见[R26报告](../tmp-nvcc/perf-ada-round26-bundle/report.md)。

R27合并IQ3_S、IQ3_XXS、IQ4_XS的canonical reduce-scatter和合作BF16写回，以及Main16四head warp页重建。统一数学通过；页恢复工作负载两正一负，已撤回该source改动。保留三格式decode，同binary pp4096+tg128/MTP7/Graph/INT8/ctx8192/chunk512的三对吞吐分别+1.601048%/+1.775624%/+0.891646%，中位+1.601048%，六组完整129 IDs相同。伴随prefill吞吐中位-0.677477%，该路径未修改，不认领prefill收益；这些数字也不是长TTFT收益。

最终decode-only重新链接后的GGUF量化库与实测库字节完全相同，新Serve原213K六题6/6精确正确、HTTP/terminal/usage完整、server0；245.820s/18.7G峰值/170.6M swap（20G/2G限制）。状态为显式三格式decode profile已资格化：`NINFER_R27_BUNDLE=1`、固定`NINFER_R26_BUNDLE=0`，全局默认仍0；范围仅SM89/N17408/K5120/T8/nonFused/plain BF16。+1.601048%为上述同binary对照及相同GGUF模块的证据继承，没有在最终链接binary重做配对，也没有长TTFT收益结论。详见[R27报告](../tmp-nvcc/perf-ada-round27-bundle/report.md)。15.4及P4性能仍未完成。


R28批量筛选（2026-10-21）：先计算真实热shape与流量/寄存器/shared成本，再一次合入四个I64/J128完整K MMQ shape和GDN Q/K/V直接打包输出，共七文件，NINFER_R28_BUNDLE缺省0。八个公共MMQ独立constant oracle、GDN七种实际格式/T512/17/8的独立constant oracle与完整153,990,144B输出/guard/Graph比较通过；down规约改变产生少量BF16差异，未承诺bitwise数学不变。

同一binary统一做两组整体ABBAAB，每组六次：短pp4096+tg128的prefill/decode吞吐中位分别−1.114718%/−1.574450%；长pp32768+tg16分别+0.873695%/−0.404652%，长prefill三对有正有负。短129和长17个完整输出IDs各自六次一致，投机工作量各自相同。状态MATH_PASS_REJECTED_NO_STABLE_OVERALL_GAIN，R28=0，R27=1/R26=0基线保留；不为负组合追加原213K、TTFT或P4矩阵，也不逐格式做长性能筛选。构建256.935s/4G/0swap；统一数学5.731s/912.7M/0；短56.021s/13.5G/0、长236.425s/2.1G/0（服务内存峰值不可当作进程RSS比较），全部terminal0。精确实测binaries、配对原值、失败采用门与收据保存在tmp-nvcc/perf-ada-round28-bundle；原题质量NOT_RUN，不能继承旧6/6作本批验收。没有提交推送。

后续R29仍按用户“理论先行、多项相容改动后统一构建/测试”推进：优先复用现成T16 MMQ尝试固定MTP15，并只在原fullK/no-fixup几何合法时直写BF16；同时准备保持BF16舍入边界的SiLU/down激活量化融合。当前仅研究和CPU overlay，尚未生产构建、数学/Graph/质量或性能验收。R21原位置模式仍失败且默认0，15.4及16.1–16.4不勾选；不存在新的暂停期限或关机授权。


R29 / v1.0.0 发布验收（2026-10-21）：按用户要求先算理论与资源成本，将 fixed MTP15/T16、fullK/J16 BF16 writer 与保持BF16舍入的SiLU/down量化融合一次合入19文件后统一构建与数学验收。50次Op（T16十格式含完整Q4_K head22次，融合七格式constant/random28次）和K15混合接受前缀/guards/frame通过。初次eager129IDs正确，Graph在拓扑切换失败；只对明确topologyChanged先实例化替代exec后换，旧失败保留，修复前后Ops/GGML/Engine库字节未变，不重复数学矩阵。

最终同binary A显式六flags0/K7、B六flags1/K7、C六flags1+R29/K15，ABC/CBA/ABC，4K与32K各9进程、warmup1/r1/tg128；每档完整129IDs九次一致，短C Graph与已验eager投机工作一致。B/A短prefill吞吐中位+5.232890%（三对正）、decode+.761975%（两正一负）；C/A长prefill+9.582095%、decode+11.779491%（均三对正）。C/B短decode−15.183451%，长+10.219626%，故采用两个显式profile：短K7、长32K档K15；无通用自动阈值或普遍K15收益。

最终K15/Graph/sync/W32768/物理36864/HostKV0原213K六题6/6精确、HTTP/terminal/usage完整、server/service0；248.360s、cgroup峰值16.9G/swap329.5M（20G/2G，非OOM）。整组history replay185836/reused10240/query162、718压实/3010Coldspill/36998sourceK恢复、staging2297856B保持。单臂API TTFT首题152.592s、后五题中位17.715s，仅作观测，不认领配对TTFT或新P4改善；原位置实验仍失败且默认0，15.4和16.1–16.4仍未完成。

全新README、CMake软件版本1.0.0、完整整体原值与原题/数值证据纳入源代码版本；公开报告见 docs/releases/v1.0.0.md 与 docs/releases/v1.0.0/。本机完整失败和精确二进制保存在 perf-ada-round29-bundle。权重、二进制、Cold后备、profiler数据和外部所有权q4诊断改动不纳入源码提交。发布推送/tag状态以实际收据为准。
