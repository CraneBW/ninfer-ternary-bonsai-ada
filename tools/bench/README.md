# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.
These are the runner's existing presets, not the current local GSQ recommendation. The Engine and
`ninfer_bench` support fixed MTP `K=1..15`; adaptive/lookup MTP remains limited to `K<=7`
(adaptive ceiling at least three). The local Qwen3.8 GSQ / RTX 4070 Ti SUPER recommended profile
remains MTP7 with the optimized proposal head until the release-profile comparison selects otherwise.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v14. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact TARGET=PATH` entries for the registered targets.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset. Use `dflash7` for
Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B DFlash2 K=7, with companion weights
in the selected artifact. `--sampling greedy` selects exact argmax; the default is stochastic.
The local environment uses Python 3.14 (`/usr/bin/python3`). Historical model-page reproduction
entries may name Python 3.11; select that interpreter explicitly when reproducing those records.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v6 result and flattened summaries retain the canonical `weights_id`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v18 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions.

Outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. Corpus runs also write complete responses in `corpus/<point>/results.jsonl` and
per-request phase summaries in that directory; older campaigns may have only point reports and
server logs. Historical model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.


### Bounded Ada qualification with existing models

`python -m tools.bench.kvmem_qualification run DIR` consumes a frozen manifest,
uses the native Serve wire client, saves raw SSE/receipts/metrics and stops at
the first failed answer or terminal. It refuses an existing ledger. Inputs use
explicit artifact/binary paths; no model download or Python inference backend
is involved. Stage configuration can select the existing Companion executable
for Bonsai2B, whose CLI differs from the primary Ada server.

`freeze-long DIR --model /explicit/full.ninfer --binary build/apps/ninfer-serve
--tokenizer-audit /explicit/native-tokenizer-audit --working-tokens 32768`
freezes the 32K/64K/213K/256K ladder and a separate forced-Cold 64K path. The
native tokenizer audit must use this artifact's actual resources. The baseline
32K full-KV control is distinct from compressed sync/async; no full-KV
comparison is claimed at larger tiers. Five known facts cover early/middle/late,
a replacement value and a tool result. One ACK plus five facts per stage,
66 calls maximum with fresh controls; existing comparable controls can be
reused in a separately frozen manifest. Each earlier qualified stage gates the
next length. A 4K working window has a recorded 32K failure and is not a
qualified substitute for the 32K window. This tool does not imply all windows
or arbitrary long-context semantics pass.

Current evidence is under `tmp-nvcc/qualification-20261020/`; see the active
maintainer validation for accepted boundaries and retained failures. Media
manifests additionally capture source hashes, color-order facts and production
request-log patch/token counts. The `kvmem_async` P4 runner now includes actual
answer correctness and equal usage as adoption prerequisites, while keeping
its original three-scene/three-pair bound and 10%/3% thresholds.


## Independent GGUF prefill oracle and memory-bounded tuning

`gguf_prefill_oracle.py` reads 128 rows by default for each represented block format
from an existing v3 artifact. GGML's independent CPU NumPy decoder (`gguf==0.19.0`,
MIT) and FP64 row sums provide the reference for the native Op test. The optional
reference package stays project-local; production inference does not depend on it.

```bash
/usr/bin/python3 tools/bench/gguf_prefill_oracle.py \
  /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
  tmp-nvcc/gguf-cpu-oracle.json \
  --gguf-source /home/ran/ninfer-work/.tools/gguf-oracle
build/tests/ninfer_gguf_linear_test \
  /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
  tmp-nvcc/gguf-cpu-oracle.json --time
```

The optional `--time` records2warmups/7CUDA-event samples for N128/K5120/T512
format slices. Those numbers describe the Op sample only. The no-argument test
still uses self-contained CPU Q8 oracles. Real-model checks now require both
the artifact and its CPU-oracle JSON; the former GPU-dequant reference is removed.

For complete existing projection shapes, add `--full-shapes` to oracle generation.
It streams CPU decoding in at most 128-row chunks and covers existing
`[17408,5120]`, `[34816,5120]`, and `[5120,17408]` tensors: 15 entries in the
current Qwen3.8 artifact, without loading its vocabulary head or whole model.
The resulting test covers T=1/8/9/512/1024/2048/2049; `--time` measures
T=1/512/1024/2048 with two warmups and seven CUDA-event samples each.
`--format gguf_iq3_s`, `--entry gguf_iq3_s_n17408_k5120`, and `--tokens 1`
restrict a run and skip synthetic cases, allowing bounded standalone profiling.
The reference activation uses exactly represented column scales and FP64 CPU
row sums; the original nonuniform synthetic gather oracle remains separate.

For an exact comparison across process-frozen dispatch policies, add
`--dump-output /absolute/output.bf16` with an explicit single `--entry` and
`--tokens`. This writes the complete raw BF16 output once, truncating the
destination; order is `token * rows + row`, two bytes per element, without a
header. `--pattern random` uses a fixed xorshift32 seed and exactly represented
integer/16 inputs. It requires `--dump-output` or `--time`, reports
`BITWISE_CAPTURE`, and checks finite output; it **does not** report an independent
CPU-oracle pass. Compare the complete bytes between policies and retain the
default constant-pattern CPU-oracle check as separate numerical evidence.

These are **NInfer-native Ops over retained GGUF block encodings**. Production
loads `.ninfer` and uses the public C++/CUDA Engine; this tool does not invoke
an external llama.cpp inference backend or requantize the weights.

On SM89, six represented IQ/QK formats can use a direct final BF16 MMQ store
when the existing selector chooses J128 with complete output tiles and no
Stream-K fixup. FP32 accumulation and final BF16 rounding are preserved.
`NINFER_GGUF_MMQ_BF16=0` restores the FP32-plane/conversion route; the qualified
direct route is enabled by default. The environment policy is frozen at first
use, so restart the process to change it. FP32 output, fixup, MoE and other
devices keep their original route. `NINFER_GGUF_MMQ_TRACE=1` emits bounded actual
launch probes including `output=bf16-direct` or `output=fp32`; disable it for
timing. See the [live performance record](../../docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md#第四轮gguf-mmq-最终直接写回-bf16).

Before another tuning round, read
[the project tuning experience](../../.claude/skills/ninfer-perf-tuning/SKILL.md).
Use a single binary, clean interleaved A/B, explicit KV/context/generation
settings, and actual route probes for new dispatches. Sequential percent-scale
screens do not establish causal gains or regressions. A graph-level Nsight
summary omits internal replay-node kernels; node tracing also inflates host
launch timing, so neither should replace clean end-to-end measurements.

A full-model NCU kernel replay exhausted host RAM during device-memory backup.
**Do not profile the complete model with that strategy.** Use a standalone Op
fixture for NCU and Nsight Systems for native execution attribution. The tested
2026-10-20 setup uses transient systemd services with OOMPolicy=kill:

| Job | MemoryMax | MemorySwapMax |
|---|---:|---:|
| Independent Op/NCU | 4GiB | 0 |
| Build, two workers | 6–8GiB | 0 |
| Whole-model qualification/P4 runner | 20GiB | 2GiB |

Wrap the entire runner so its Serve children inherit the limit; set
`--working-directory` explicitly. For example, with a **fresh frozen** P4 folder:

```bash
systemd-run --user --unit=ninfer-p4-bounded --wait --pipe --collect \
  --working-directory=/home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada \
  --property=MemoryMax=20G --property=MemorySwapMax=2G \
  --property=OOMPolicy=kill --property=RuntimeMaxSec=1800 \
  /usr/bin/python3 -m tools.bench.kvmem_async run tmp-nvcc/FRESH-FROZEN-P4
```

Keep GPU jobs and compilation serial. The limits isolate jobs; they do not unify
the Engine Host cache, KVMem archive, source files, OS page cache and checkpoint
images. Final model runs reached20GiB plus2GiB swap, so memory pressure remains
part of the measurement conditions. Evidence and the failed profiler run are
in `tmp-nvcc/perf-20261020/`, summarized by its `summary.json` and `summarize.py`.
