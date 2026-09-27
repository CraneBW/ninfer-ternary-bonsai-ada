[![English](https://img.shields.io/badge/Language-English-2ea44f?style=flat-square)](README.en.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-d73a49?style=flat-square)](README.md)

# Ternary Bonsai 2 27B on **NINFER**

> A 16 GB RTX 4070 Ti SUPER running ternary Bonsai 2 27B (2.125 bits per weight), context
> filled to the model's native 262,144 tokens (int8 KV). Measured on 210K tokens of real
> text (a large file fed as context): 164 s to first token, 88.8 t/s decode, 5.21 GiB of
> runtime memory. With KVMem on or off, decode shows no difference across six task
> configurations (0.5% max) — prefill too, and short contexts as well. KVMem runs alongside
> MTP speculative decoding. A short tool call reaches 232 t/s, writing fiction 89; prefill
> is 2.30k t/s on a 28k prompt. The full matrix — three tasks (pelican-on-a-bicycle SVG,
> Chinese short story, tool call) × long/short context × with/without thinking — plus its
> fixtures and raw data are in the repository; `git checkout kvmem` is this build.

---

> A single-GPU C++20/CUDA inference engine (**NINFER**) ported to Ada (`sm_89`), running the
> ternary quantization of Bonsai 2 27B. The target machine is one **RTX 4070 Ti SUPER (16 GB)**,
> and every number below was measured on it.

Ternary weights are packed `{−1, 0, +1}` at 2 bits per code plus one fp16 scale per 128-weight
group. The storage cost is **2.125 bits per weight**. The familiar "1.58-bit" is log₂3, the
information content of a ternary symbol — both figures are correct and they measure different
things; **the memory system pays the former**. That is roughly half the weight traffic of a
4-bit format, and it is the entire reason the numbers below are where they are: the model reads
7.12 GiB of weights per verify round, and this card reads at a measured 637 GB/s.

**This repository is a derivative work, not upstream NINFER.** Lineage and credits are in §8.

**How to read this document**:

| You want | Go to |
|---|---|
| What this branch does better than the upstream release, and everything it changed | **§2** |
| To run a 262k context on a 16 GB card | **§3** (KVMem) |
| To get it built and running | **§4** |
| To tune flags, pick a KV format, decide how much context fits | **§5** |
| The data (three-way comparison, speed matrix, PPL) | **§6** |
| How the engine picks a kernel by T | **§7** |

---

## 1. What this is

**In one line**: Bonsai 2 27B, ternary-quantized, running on the NINFER engine, tuned for a
single Ada card, with long context made reachable on 16 GB.

Three things separate this repository from upstream. Each has its own section below.

1. **Kernel and dispatch work** (§2) — with the precision difference removed, this machine's
   bf16 prefill kernel is 35% (30k) / 32% (64k) faster than upstream's; with each side's
   default precision restored, long prompts lead by 29%.
2. **KVMem** (§3) — only a resident window of KV stays in device memory and the rest is parked
   in host memory, which is how a 262,144-token logical context fits on one 16 GB card
   **at zero cost in throughput**.
3. **Deployment and configuration in one place** (§4, §5) — build, run, what each flag costs,
   how to choose a KV format.

**What "upstream" means here**:

| | |
|---|---|
| **Source of the ternary port** | **[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary)** on ModelScope — engine-side `patches/`, packing and verification tools, the technical record. **It ships as a Windows `.exe` (MSVC) package** |
| **Direct base of the source tree** | **[Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)** v1.0.8 line |
| **Canonical upstream** | **[Neroued/ninfer](https://github.com/Neroued/ninfer)** — C++20/CUDA architecture, DFlash2, ReplaySSM, Paged KV Cache |

Every "upstream ternary" figure in this document was measured on **a Linux binary rebuilt from
it here with GCC 16.2.1** (`~/ninfer-off-build/build/apps/ninfer`, sha256 `b89c77e9…`), not on
the Windows package upstream publishes. That isolates kernels from toolchain; it also means the
numbers do not describe upstream's shipped binary. Protocol is in §6.1.

---

## 2. What this branch changes against upstream

### 2.1 The changes

**Every measurement behind these figures lives in a code comment.** The gain column is a
same-binary A/B or a median over alternating rounds; protocol in §6.1.

**Prefill**

| Change | Gain | Notes |
|---|---:|---|
| **int8 (`s8`) activation path** | **1.61×** | Upstream's int8-activation × tensor-core path. PPL shift +0.0015% |
| **Token tile into `blockIdx.y` (G2)** | **+26.2%** (T=508) / **+24.7%** (28k) | Bit-identical. The s8 grid is `div_up(n,64)`, **independent of T**, so small-`n` layers ran at 1.2 CTA/SM at any T. Laying the tile into the grid leaves weight traffic unchanged while occupancy goes up |
| **split-K for the T ≤ 64 gap (G3)** | **+7.1%** (T=28) | The K accumulation is re-associated into fp32 partials — deterministic, but **not bit-identical**: PPL +0.041%. Fallback in §5.4 |
| **Prefill token tile by T** | **+42%** (T=42/50) | The 128-wide tile had no token loop, so a T under 128 paid the whole K sweep. T≤64 moves to a 64-wide tile (shared 43264 → 26880 B, 2 → 3 CTA/SM) |
| **Filling the T=9..31 gap** | **+131% ~ +266%** | The two thresholds left a seam exactly where ordinary short prompts (13~35) land. Now split by cost model: T≤16 on an 8-wide tensor-core tile, T≥17 on int8 |
| **s8 threshold refit 33 → 17** | Removes a 74% step at T=32 | Upstream's 33 was measured against its own bf16 path; on this machine the two lines cross at 16/17 |

**Decode**

| Change | Gain | Notes |
|---|---:|---|
| **Small-T tensor-core path for the verify pass** | the bulk of decode | Replaces blocked GEMV |
| **Row block chosen per shape** | **+1.8%** | 16 for `n ≤ 5120`, else 32. The response is **non-monotonic**, so no single global value can be right. Bit-identical |
| **small_t split-K** | **+1.3 ~ 1.4%** | Only shapes whose row grid underfills the card **and** leave ≥4 K steps per slice. **Not bit-identical**; fallback in §5.4 |
| **Rotation launch packing** | — | 274 rotations per round, one per activation — already the floor |
| **Two SASS-driven decode fixes** | — | See `.claude/skills/ninfer-perf-tuning/SKILL.md` |
| **K ceiling 5 → 7** | **+21%** on structured output | `T=K+1≤8`; beyond that it falls out of `small_t` (verify round 13.3 ms → 40 ms). The ceiling lives in four places of different kinds and must be changed together |

**KV and long context**

| Change | Gain | Notes |
|---|---:|---|
| **KV format auto-selected from `--max-context`** | **+16.2%** at 64k | bf16 at ≤16383, fp8 at ≥16384. A pure function, never written back to a field. See §5.3 |
| **KVMem offload** | **262k logical context fits** | Zero cost in throughput. See §3 |

### 2.2 Three-way comparison

Three builds, **same machine, same model file, same fixture set, same flags**
(`--greedy --no-thinking`, `--max-new 8` for prefill, 300 tokens for decode), every cell the
median of **three alternating rounds** — alternating so that GPU thermal drift cannot land
entirely on one build.

| Build | What it is | Binary used here |
|---|---|---|
| **Previous** | This repository at **`bd71d74`** (2026-09-21) — where the last round ended and where `origin/master` sat before this one. **Not "some upstream version" — this repository's own previous commit** | `~/ninfer-work/bin-pre-s8/ninfer`<br><sub>built 2026-09-26 07:06, sha256 `cea0e193…`</sub> |
| **Upstream ternary** | The build defined in §1 (a Linux rebuild of upstream's source) | `~/ninfer-off-build/build/apps/ninfer`<br><sub>sha256 `b89c77e9…`</sub> |
| **This branch** | Current HEAD | `~/ninfer-build/build/apps/ninfer`<br><sub>sha256 `70ecd5ca…`</sub> |

All three are Linux binaries built here with the same GCC (16.2.1); `readelf -p .comment`
confirms it. Build times were 07:06 / 05:59 / 16:33 (2026-09-26).

**Prefill, each at its own default precision** (tok/s, higher is better):

| Prompt | Previous<br>(bf16) | Upstream ternary<br>(int8) | **This branch**<br>(int8) | vs previous | vs upstream |
|---|---:|---:|---:|---:|---:|
| Long (T=1759) | 1.22k | 1.96k | **2.53k** | **+107%** | **+29%** |
| Medium (T=123) | 769.9 | 1.33k | **1.58k** | **+105%** | **+19%** |
| Short (T=38) | 280.6 | 664.1 | **734.7** | **+162%** | **+11%** |
| Tiny (T=15) | 144.4 | 333.8 | **388.4** | **+169%** | +16% |

**Decode** (tok/s, `en-code.json`, 300 tokens):

| Scenario | Previous | Upstream ternary | **This branch** | vs previous | vs upstream |
|---|---:|---:|---:|---:|---:|
| MTP draft 3 | 145.4 | 122.5 | **147.3** | **+1.3%** | **+20%** |
| └ round net (ms) | 20.77 | 24.65 | **20.30** | **−2.3%** | **−18%** |
| No speculation | 64.8 | 57.4 | **64.4** | −0.6% | **+12%** |

> **Mind the MTP acceptance rate**: previous and upstream are both at 67.6%, this branch at
> **66.3%**. The difference comes from split-K (§2.1): over T=17..64 it re-associates the K
> accumulation in fp32, so the numerics are not bit-identical and this prompt's trajectory
> forks onto a slightly lower-acceptance path. **It is not "slower per round", it is "a
> different trajectory"** — with `KSPLIT=0` the acceptance returns to 67.6% immediately.
> Within one binary the choice is deterministic and reproducible; it only matters across
> builds. `NINFER_TERNARY_S8_KSPLIT=0` restores the bit-identical path.

> **Where the two decode gains came from** (both on the verify path at T ≤ 8, neither touches
> prefill): **① row block per shape** — non-monotonic response, no global value works;
> alternating two-way A/B **+1.8%**, with acceptance, histogram and generated text all
> unchanged. **② split-K for small shapes** — only `n ≤ 6144` shapes that still leave ≥4 K
> steps per slice (in this model only 5120×17408), two-way A/B **+1.3~1.4%**, six fixtures
> +0.2~1.8% with no accepted-length regression. ② **is not bit-identical**.

**In one line**: long prompts went from "level" to **29% ahead**, every other prefill case
leads, and decode holds a 20% lead (the acceptance caveat is above).

### 2.3 With the precision difference removed

§2.2 is **each build at its own default precision** — upstream ternary and this branch take
int8, the previous build takes bf16. Turning int8 off (`NINFER_TERNARY_S8=0`) is what shows
which **kernel** is actually faster.

`./three-way-long.sh`, KV pinned explicitly to `--kv-dtype bf16` (**without pinning it, this
repository auto-selects fp8 at these lengths and the other two do not, so you would be
measuring a precision difference rather than a kernel difference** — see §5.3), median of
three alternating rounds:

| Prompt | Precision | Previous | Upstream ternary | **This branch** |
|---|---|---:|---:|---:|
| **28,199**<br><sub>`--max-context 32768`</sub> | each default | 1.17k <sub>(bf16)</sub> | 1.75k <sub>(int8)</sub> | **2.18k** <sub>(int8)</sub> |
| | **both bf16** | 1.17k | **864** | **1.17k** |
| **61,636**<br><sub>`--max-context 65536`</sub> | each default | 1.06k <sub>(bf16)</sub> | 1.51k <sub>(int8)</sub> | **1.81k** <sub>(int8)</sub> |
| | **both bf16** | 1.06k | **804** | **1.06k** |

Three conclusions, **both lengths**:

1. **With the precision difference removed, this machine's bf16 prefill kernel is 35% (30k) /
   32% (64k) faster than upstream ternary's.** Upstream's prefill advantage comes **entirely**
   from its int8 path — true at T=1.7k, 28k and 62k alike.
2. **With precision restored, this repository leads by 25% (30k) / 20% (64k)** (2.18k vs 1.75k;
   1.81k vs 1.51k). The previous build was still level here — the difference is **the token tile
   into `blockIdx.y` (G2, +26%)**, which **only applies to the int8 path**, which is why the
   "both bf16" row below it does not move.
3. **At bf16 the previous build and this one are still level** (1.17k / 1.17k; 1.06k / 1.06k),
   while upstream ternary falls 35%/32% behind. **All of this round's long-prompt gain runs
   through the int8 path**: nothing changed on the bf16 path, and nothing needed to — it was
   already ahead.

> **Three-way comparison above 64k is possible from the CLI, just not with `rk4v4`.** Upstream
> ternary's CLI does not accept `rk4v4` (only its serve does), but **`fp8` is accepted by all
> three** and reaches **233k** — a 124k context needs 3.9 GiB. So go through `fp8` above 64k
> and no serve, and therefore no protocol alignment, is needed. **Not yet measured here.**

### 2.4 Tried and did not work

Negative results are recorded in the code too.

| Attempt | Result | Mechanism |
|---|---|---|
| Widening the small_t tile | **Regression** (45.2 → 19.0 t/s) | The accumulators cost a resident CTA |
| `__launch_bounds__` for occupancy | **Monotonically slower** in three independent measurements | `MinBlocks` caps registers only, **it cannot hold back shared memory** — `TernaryS8Storage<64,4>` × 4 = 103424 B > 102400 B, so a 4th CTA never fits |
| Narrowing the s8 token tile to follow T | **Regression** (six fixtures, +1%) | The kernel is not mma-bound (tensor cores only 17~35% active); halving the tile just spreads each CTA's fixed cost over half the work. **Forcing 32 at T=38 is +19%** — the `tok_base` loop runs twice and the weights are read twice |
| Splitting the LUT | **−12.6%** | Same-address broadcast |
| `--lm-head-draft` | Within noise (en-code −3.6%) | Its head is 356.5 MB against the main output head's 337.7 MB, so traffic goes **up** 6% |
| The `rk4v4-e8` KV format | 3.5× worse than `rk4v4` | Its decode side is a deliberate half-coset approximation that drops part of the E8 shaping gain. **No reason to exist in the current implementation** |

---

## 3. KVMem: long context without full residency

**Merged from [naamfung/zatfung](https://github.com/naamfung/zatfung)**; this repository is
where it was made to coexist with MTP.

### 3.1 What it solves

On a 16 GB card, KV and weights compete for the same memory. A 262,144-token context costs
4.71 GiB at `rk4v4` and 8.06 GiB at `int8` — on top of 7.12 GiB of weights, it does not fit.

KVMem keeps only a **resident window** in the device pool and parks the rest in host memory.
The point is that the pages outside the window are not left in place as holes: the evicted
interval is **compacted away and the remaining pages are re-phased (re-RoPE)**, with the
coordinate system shifted to match. An implementation that selects blocks in place saves no
tile iteration at all and is equivalent to not turning it on — that is why it failed elsewhere.

### 3.2 Turning it on

```bash
./build/apps/ninfer-serve <model.ninfer> \
  --kvmem --kv-dtype int8 \
  --kv-device-tokens 131072 \
  --kvmem-budget 98304 \
  --host-kv-mib 8192
```

| Flag | Effect |
|---|---|
| `--kvmem` | On. **Without it everything stays resident**, exactly as before the merge |
| `--kv-device-tokens N` | Shrinks the device pool below the context; what it frees is what KVMem gives back |
| `--kvmem-budget N` | Token count of the resident window |
| `--kvmem-gen-reserve N` | Pool headroom kept for decode (default 8192) |
| `--host-kv-mib N` | Size of the host-side KV pool |

CLI and server both expose all of these. The environment variables `NINFER_KVMEM` /
`NINFER_KVMEM_BUDGET` / `NINFER_KVMEM_GEN_RESERVE` are the equivalent spelling for frontends
that do not surface them as flags.

**Two hard constraints**:

- **`--kv-dtype int8` is required.** The offload and the scoring address the int8-group64 codec
  planes directly; fp8 cannot go through them.
- **The window size decides quality, and it is not free.** See §3.5.

### 3.3 The speed cost is zero

At the same KV format (int8), KVMem and full residency are identical cell by cell while
**actually offloading**:

| Task | Thinking | Resident-int8 | KVMem-int8 |
|---|---|---:|---:|
| Pelican on a bicycle | off | 108.1 | 108.2 |
| Pelican on a bicycle | on | 74.6 | 74.7 |
| Short story | on | 77.3 | 77.3 |
| Short story | off | 58.8 | 58.9 |
| Tool call | on | 134.3 | 134.5 |
| Tool call | off | 180.4 | 181.3 |

Protocol: 124k prompt, device pool 2,048 pages against 4,096 logical, `host KV pinned
8.00 GiB` (**genuinely offloading**). Prefill matches too (1520~1540 vs 1520~1530), and so do
short contexts that fit the pool without offloading (155.2/155.1, 232.5/232.0). **What it buys
is a 262k logical context on a 16 GB card, and it does not buy it with speed.**

**The merge did not touch the resident path either**: same fixture, same CLI protocol, three
alternating rounds, 64k decode matches cell by cell before and after the merge — bf16 98.3 /
98.3, int8 112.0 / 112.0, fp8 114.5 / 114.4, rk4v4 111.5 / 111.5, with acceptance identical
to the digit (60.1%).

### 3.4 The memory ledger

Every row is the `capacity` line of a `ninfer-serve` startup log.

| Configuration | Weights | Device pool | runtime | Total on card | GPU free |
|---|---:|---:|---:|---:|---:|
| Resident 8k, no MTP | 6.70 | 128/128 | 0.75 | **8.08** | 7.92 |
| Resident 8k, MTP3 | 7.12 | 128/128 | 0.84 | **8.53** | 7.47 |
| Resident 64k, MTP3 | 7.12 | 1024/1024 | 2.76 | **10.45** | 5.55 |
| Resident 128k, MTP3 | 7.12 | 2048/2048 | 4.95 | **12.64** | 3.36 |
| **KVMem 256k logical / 128k pool, no MTP** | 6.70 | 2048/4096 | 4.61 | **11.95** | 4.05 |
| **KVMem 256k logical / 128k pool, MTP3** | 7.12 | 2048/4096 | 5.21 | **12.90** | 3.10 |

Total on card = 16 GiB − GPU free (including roughly 0.6 GiB of CUDA context and the like).
**KVMem runs a 262,144-token logical context in 12.90 GiB; full residency spends 12.64 GiB to
reach 131,072.** Taking the resident path to 262,144 would need 8.06 GiB of KV pool by itself,
putting the total past 16 GiB — it does not start, and the engine names the shortfall in bytes.
The two device-pool columns are "physical pages / logical pages"; the ratio is how much KV sits
in host memory.

**Long-context (124k prompt) prefill**: resident-fp8 **1460**, resident-int8 **1520~1540**,
KVMem-int8 **1520~1530** tok/s. All three tasks land in one band, because prefill depends on
token count alone, not on task content or residency mode.

### 3.5 It is lossy: the window decides recall

The decisive fixture is three **real documents** on unrelated subjects concatenated into
213,723 tokens (Maya 124k + pistols 62k + Tintin 29k), with the question answerable **only from
the middle section** — the answer block is neither in the sink head nor in the recent tail, so
it has to survive on mid-context top-k.

| Window | Result |
|---|---|
| 17,408 (6% of the context) | **Loses the middle** — the model answers "there is no Section B in the text" |
| 74,752 | Answers correctly, `Type 94 Nambu` (decode 47.0, still 27% faster than full residency) |

**So it is a lossy tool that trades memory for long-context speed, not a lossless memory
optimization.** How large the window must be depends on where your task hides its answers; for
long context with mid-document recall, the window has to be big enough.

### 3.6 Running it with MTP

**They can be on at the same time** — this is what this repository added. A draft layer attends
the whole history and cannot keep a window, so its KV is compacted together with the text
(same budget, same frontier) and its page pool is sized by logical capacity.

The original implementation excluded speculation from offload through three
`speculative_backend == None` guards, because both sides share one `kv_offset`: compacting the
text alone shifts the draft layer's RoPE positions by the whole evicted span. The three fixes
(3 files, 77 lines) are: compact the MTP KV too and assert the two windows agree; take the
prefill compaction budget as `min(configured window, pool headroom)` (the original used only
the latter and ate the pool to fullness every time); and make `chunked_kv_prefill`'s test use
the raw prompt length (in compacted coordinates the original test flips false, so after an
offload no piece maps any more).

**Measured**: 213,723 tokens of real text, device pool 2,048 pages / 4,096 logical (window
1,536 pages), MTP draft 3 — TTFT **164.0 s** (prefill 1,302.8 tok/s), decode **88.8 t/s**,
runtime **5.21 GiB**, answer correct. The same text does not start at all with all 2,048 pages
resident.

---

## 4. Deployment

### 4.1 Hardware and prerequisites

| | Requirement |
|---|---|
| GPU | NVIDIA Ada `sm_89` (RTX 40 series). Tuned and verified here on an **RTX 4070 Ti SUPER 16 GB** |
| Memory | ≥ 16 GB (weights 6.70 GiB + MTP 0.42 GiB + KV) |
| OS | Linux (verified on Arch). **Paths must be pure ASCII** |
| Toolchain | CUDA 13.x, GCC 15+, CMake ≥ 3.24 |

**Put the display on the integrated GPU if the board has one.** Driving a desktop from the
discrete card steals its memory bandwidth — about **10%** on this machine. That is not a tuning
knob, it is a machine state.

### 4.2 Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j"$(nproc)"
```

Outputs: `build/apps/{ninfer, ninfer-serve, ninfer-perplexity}`.

> **Two known build traps**, both from concurrent nvcc under `-rdc=true` and unrelated to the
> code:
> 1. **`TMPDIR` on tmpfs makes GCC segfault.** Point it at disk:
>    `export TMPDIR=$PWD/tmp-nvcc`
> 2. The build occasionally reports an ICE (`cc1plus` segfault). **Retry it** — not a code
>    problem. Dropping to `-j6` makes it much rarer.

### 4.3 Model

The model is a `.ninfer` artifact of **Ternary Bonsai 2 27B**. **This repository does not
contain or redistribute the weights** — obtain them from the official channels and observe
their terms, which may not be Apache-2.0. See §8.

### 4.4 Running it

**Command line** (single question):

```bash
./build/apps/ninfer <model.ninfer> \
  --messages prompt.json \
  --max-context 8192 --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --no-thinking
```

`prompt.json` is an OpenAI-style message array:

```json
[{"role": "user", "content": "Hello"}]
```

**Server** (recommended for multi-turn):

```bash
./build/apps/ninfer-serve <model.ninfer> \
  --host 127.0.0.1 --port 8084 \
  --kv-dtype rk4v4 --max-context 262144 --kv-capacity auto \
  --max-concurrency 1 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --no-thinking
```

```bash
curl -s http://127.0.0.1:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}' | jq -r '.choices[0].message.content'
```

> **Prefer `ninfer-serve` over `ninfer` for multi-turn work.** The CLI's prefix cache is
> **hard-disabled**; the server's is on by default, so the same history is prefilled from
> scratch on every turn under the CLI.

**Wiring it to an agent harness: a worked example**

The two snippets above only prove the server is up. **Tool calls plus multi-turn** is a
separate path and needs its own check — run a concrete task through an agent harness:

```bash
# Terminal 1: start the server (§5.1's recommended configuration, but **drop --no-thinking** — see below)
# Terminal 2:
BONSAI_API_KEY=local dsh --profile headless \
  "Draw a 2D animation of a pelican riding a bicycle as ONE self-contained HTML \
   file, inline SVG only, under 120 lines. Output only the code."
```

The harness's model configuration (`~/.dsh/settings.yaml`) points at
`http://127.0.0.1:8080/v1` with model id `Ternary-Bonsai-2-27B`, and **`contextWindow` must
match `--max-context`** — set it larger and the client sends an over-long request, which the
server rejects outright.

**Measured**: finished in **3 min 22 s**, having issued **8 requests** — the agent's tool loop:
it wrote the file itself, ran `wc -l` on it itself, parsed the SVG to verify its structure. The
output was a **58-line, zero-external-reference, renderable single-file HTML**.

> **Two preconditions, without which this example does not terminate**:
> ① **Do not add `--no-thinking` to the server** — an agent loop ends because the model ends it,
> and with the thinking chain off it just rewrites the same file over and over;
> ② do not sample with `--greedy` (that is for reconciliation and makes this worse).
>
> **Why an agent loop rather than one-shot generation**: every step ends in a tool call, so
> **the thinking has a terminus**. A control run with the client output limit raised from
> 32,768 to 199,000 took **3 min 27 s / 8 steps** — it did **not** think more (19% less
> thinking, in fact) and terminated with a finished artifact anyway.
> **The terminus is "every step has to land an action", not the limit.**

### 4.5 Deploying into long sessions

**The server's prefix cache advances turn by turn for an agent loop**, provided the history
shape is recognised. One counterexample: if the caller **drops reasoning** from the history
every turn (what harnesses like Hermes do), and the server anchors turns in the wrong place,
the cache hold **freezes at the start of the session**, the miss grows linearly with the
session, and TTFT degrades from seconds to tens of seconds. Fixed in `d69ef3d` (the anchor is
now "after the last user **or tool** message"), but with any frontend it is worth glancing at
the `cache N (X%, <path>)` column in the server log:

| Path | Meaning |
|---|---|
| `turn closure` / `private endpoint` | Normal; the value should advance every turn |
| `shared prefix` | A different path, also normal |
| **Any path whose value never grows** | Something is wrong; misses will accumulate |

---

## 5. Configuration

### 5.1 Recommended configuration and what each flag costs

```bash
--kv-dtype rk4v4          # context 120k → 262,144; runtime 8.54 → 4.71 GiB
--max-context 262144 --kv-capacity auto
--max-concurrency 1       # interactive latency first; **batch throughput** below
--prefill-chunk 1024      # the default is already optimal (swept 128~2048)
--spec mtp --draft-tokens 3   # decode ×2.3. **The best K moves with the task**, see §5.1
--no-thinking             # decode +31%. **A usage trade-off**: no thinking chain
```

For a 262k context without giving up speed, add KVMem (§3) on top of `--kv-dtype rk4v4`, or
take `int8` + KVMem for better numerics.

**`--max-concurrency` has two readings; do not look at only one.** Decode is **genuinely
batched** (the weight read is amortised), but prefill is **exclusive** (one lane at a time). So:

| | Per-lane latency | Aggregate throughput |
|---|---|---|
| `--max-concurrency 1` | ✅ 113 t/s | 1.00× |
| `2 ~ 3` | falls to 70% / 51% | **about 1.40× / 1.53×** |

Note that **the KV pool scales linearly with concurrency**
(`max_concurrency × page_count(max_context)`), so **on 16 GB, `N>1` and long context (>64k)
are mutually exclusive**.

**The best `--draft-tokens` moves with the task** (measured 2026-09-26, median of three
alternating rounds):

| Task | Per-position survival | Best K | K=3 | K=5 | K=7 |
|---|---:|---:|---:|---:|---:|
| `en-code` (code, low predictability) | ≈0.76 | **5** | 143.4 | **151.0** | 136.6 |
| Counting / structured output | ≈0.95 | **7** | 191.3 | 232.4 | **231.8 (+21%)** |
| Template repetition (upper bound) | ≈0.995 | **≥7** | 197.0 | 248.4 | **289.3 (+47%)** |

**Acceptance is not a constant — it decays with position, and the decay rate is a property of
the task.** The `accepted by pos` histogram prints it directly (at K=7):

```
en-code        56,45,28,23,17,11, 7    ~0.76 per position, a loss from the 4th on
mtp-count      40,39,39,32,30,22,13    ~0.97 for the first four, then it falls
mtp-template   32,32,32,32,32,32,31    barely decays at all, accepted length 7.97 (ceiling 8)
```

So "a bigger draft is always slower" holds only for low-predictability tasks: **on one binary,
`en-code` peaks at K=5 and K=7 is worse than K=3, while template repetition is 47% faster at
K=7.** The cost side is fixed (≈1.94 ms per draft step, independent of task), the benefit side
is survival probability, and they cross at about 0.75 per position.

**K is capped at 7** (`T=K+1≤8`; beyond that it falls out of `small_t` and the verify round
jumps from 13.3 ms to 40 ms). `--draft-tokens` is chosen per task within that. The ceiling is
written in four places of different kinds (capture-graph width, model configuration, CLI
validation, a runtime invariant) that are aligned by hand and must be changed together — see
the comment in `round_state.h`.

`--greedy` buys another ~12% of decode (acceptance 36% → 44%), but it **changes the output
distribution** — **use it for performance reconciliation only**, never as a production default.

**The 16 GB trade-off**: `--max-context 262144` will not start with bf16 KV; it needs a 4-bit
KV format or KVMem.

`--vision` costs about **0.5 GiB** (vision tower 0.27 + workspace 0.24). The media subsystem
has a separate **3 GiB budget cap** (`--media-cache-mib` 1024 + `--media-live-mib` 2048) —
that is a **host-side** allowance, allocated on demand, not part of the VRAM reservation. To
save it you do not have to turn vision off; lower those two numbers.

**`--kv-capacity auto` is equivalent to spelling out `max_context` when `--max-concurrency 1`**
— it is not "adaptive", it only affects the "does not fit, fail loudly" branch.

### 5.2 KVMem's configuration

See §3.2. Three points repeated: **`--kv-dtype int8` is required**; `--kv-device-tokens`
decides how much memory comes back; **`--kvmem-budget` decides quality** (§3.5).

### 5.3 Choosing a KV format

**With no `--kv-dtype`, the format is chosen from `--max-context`: `bf16` at 16383 and below,
`fp8` at 16384 and above.** The threshold is not arbitrary — the table below is where it comes
from. Name a format explicitly to override; all five are accepted.

> **Reachable context moves with how much VRAM is free at startup; it is not a constant of the
> model.** The values below are this machine's ceilings (16 GB card, desktop not on the discrete
> GPU); if `--query-gpu=memory.used` already shows a few hundred MiB taken, these numbers fall
> proportionally. To reproduce: try `--max-context` from large downward; a failure states
> exactly how many bytes are missing (`but only N bytes are available after weights`), and
> dividing that by bytes-per-token gives the boundary.
> Re-measured 2026-09-26: `fp8` starts at a request of 233,000 (planned 233,024 / 3,641 pages /
> runtime 7.66 GiB / 1.01 GiB left) and fails at 234,000; `int8` starts at 213,000. **That
> back-solves to 32,994 bytes per token (32.2 KiB)**, matching the column below.

| KV format | Measured reachable context | Per token | Notes |
|---|---:|---:|---|
| `bf16` | ~120k | 64 KiB | default at `--max-context` ≤ 16383 |
| `fp8` | **233,024** | 32.3 KiB | default at ≥ 16384; **fastest at long context** |
| `int8` | ~213k | 33.0 KiB | the most accurate of the three, but slower than `fp8`; **the only format KVMem accepts** |
| **`rk4v4`** | **262,144 (the model's native ceiling)** | 17.0 KiB | 4-bit, runtime 8.54 → **4.71 GiB** |

**Measured decode** (`--spec mtp --draft-tokens 3`, median of three alternating rounds, every
configuration naming `--kv-dtype` explicitly so this measures kernels and not precision):

| KV format | 30k prompt | 64k prompt | Long-protocol PPL cost |
|---|---:|---:|---:|
| `bf16` | 142.9 tok/s | 96.7 tok/s | — (anchor) |
| `int8` | +6.2% | +13.8% | **+0.077%** |
| **`fp8`** | **+7.1%** | **+16.4%** | +0.143% |
| `rk4v4` | +6.9% | +13.4% | +0.213% |

(The `rk4v4` acceptance in the 64k round was 61.2% against 60.1% for the other three — a KV
format changes attention numerics, the trajectory can fork, so its +13.4% is flattered slightly.
Also: these two anchors were measured before the two verify-path changes in §2.1, so their
absolute values are about 1.3% low — **the percentages are unaffected**, since row block is
chosen per shape and split-K applies only to shapes with `n ≤ 6144` and enough K depth, neither
of which involves the KV format: **all four formats are treated alike**.)

Three readings:

1. **The gain is close to linear in context** (about one point per 3.7k tokens) and below 16k
   it does not clear the run-to-run noise of a single decode measurement — which is exactly why
   the threshold is 16384. The PPL column uses the long protocol (`--context 65536
   --stride 32768`, 124k-token corpus, 3 windows, bf16 anchor 2.099109); **the criterion is
   ≤0.30% and all four pass.**
2. **`rk4v4` uses half the KV bytes of `fp8` and decodes more slowly.** Its decode side adds a
   `kv_cache_inverse_rotate_output_kernel` per layer per attention (16 layers ⇒ 16 more nodes in
   the graph), and that fixed cost eats the bandwidth saving. ⇒ **KV byte count is not the
   constraint on long-context decode.** An earlier arithmetic model that treated KV reads as
   equivalent to weight reads concluded `rk4v4` was mandatory above 110k; it did not account for
   this term, so that conclusion was wrong.
3. So **`rk4v4` is a capacity format, not a speed format**: it is what makes 262,144 (the native
   ceiling) reachable — the `bf16`/`fp8`/`int8` rows run out of memory first, and only it starts.
   **247,646 tokens with a deeply buried answer has been measured correct.** For speed use
   `fp8`; for accuracy use `int8`.

**Server-side cross-check** (`ninfer-serve` + `/v1/chat/completions`, 30k prompt,
`--max-context 40960` — a depth all four fit, since `bf16` is 64 KiB/token and 200K would need
12.8 GB on a 16 GB card. Two alternating rounds, **restarting the server for every
configuration, so the prefix is cold**):

| KV format | prefill | TTFT | decode | MTP acceptance |
|---|---:|---:|---:|---:|
| `bf16` | 2.22k / 2.21k | 12.7 / 12.8 s | **146.6 / 146.6** | 94.4% |
| `int8` | 2.28k / 2.27k | 12.4 / 12.4 s | **157.1 / 157.0** | 94.4% |
| **`fp8`** | 2.25k / 2.24k | 12.6 / 12.6 s | **157.1 / 157.1** | 94.4% |
| `rk4v4` | 2.20k / 2.20k | 12.8 / 12.8 s | **156.1 / 155.9** | 94.4% |

**Two observations**: ① **prefill is essentially independent of the KV format** (3.6% spread)
— prefill is **weight-bandwidth bound** (it reads 7.12 GiB of weights), KV writes are a
rounding error, and changing format will not get a long prompt into context any faster.
② **Only `bf16` is clearly slower at decode**; the other three are about +7%, with `int8` and
`fp8` level and `rk4v4` 0.6% behind. **The ordering matches the CLI table above**, so that
table's protocol is sound.

> ⚠️ Two protocol traps: **every configuration must start from a cold prefix** (from the second
> run on, the hit rate is 99.9% and the `prefill X tok/s` line is either a remainder or absent);
> and **the depth must fit all four formats**.

### 5.4 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NINFER_TERNARY_S8` | on | `=0` falls back to the bf16 path (**use this for same-binary A/B**) |
| `NINFER_TERNARY_S8_MIN_TOKENS` | 17 | T at which s8 takes over |
| `NINFER_TERNARY_S8_KSPLIT` | on | `=0` restores the bit-identical path (§2.2) |
| `NINFER_TERNARY_GAP` | `auto` | Which path T=9..31 takes: `auto`/`small_t`/`s8`/`gemv` |
| `NINFER_TERNARY_GAP_SMALL_T_MAX` | 16 | the `auto` boundary |
| `NINFER_TERNARY_MMA_MIN_TOKENS` | 32 | T at which bf16 MMA takes over |
| `NINFER_TERNARY_SHORT_TILE_TOKENS` | 64 | 64-wide / 128-wide tile boundary |
| `NINFER_TERNARY_VERIFY_CAP` | 8 | T ceiling accepted on the verify round |
| `NINFER_TERNARY_SMALL_T_ROWS` | 32 | small_t rows/CTA (16/32/48) |
| `NINFER_TERNARY_SMALL_T_KSPLIT` | on | `=0` disables the decode-side split-K (§2.2) |
| `NINFER_TERNARY_DECODE` | `small_t` | `=gemv` fallback |
| `NINFER_TERNARY_PREFILL` | `mma` | `block`/`ref` for A/B |
| `NINFER_TERNARY_HADAMARD` | on | `=0` disables rotation (**output is meaningless**; diagnostics only) |
| `NINFER_TERNARY_S8_DEBUG` | off | `=1` prints **the path every call actually took** |
| `NINFER_KVMEM` | off | `=1` enables KVMem (equivalent to `--kvmem`) |
| `NINFER_KVMEM_BUDGET` | 0 | KVMem resident window in tokens (0 = the whole context) |
| `NINFER_KVMEM_GEN_RESERVE` | 8192 | KVMem pool headroom kept for decode |

**After changing dispatch, the first thing to do is turn the probe on and confirm the branch is
actually taken** — speed alone cannot tell you whether you hit the intended path. This is the
most expensive lesson in this repository; see `.claude/skills/ninfer-perf-tuning/SKILL.md`.

---

## 6. Performance data

### 6.1 Before reading these numbers: protocol and precision

**The fixed conditions** (without aligning these, the numbers below are not comparable):

| | Value |
|---|---|
| KV precision | **`bf16`** (the format all three builds accept; `rk4v4` is **not accepted by upstream ternary's CLI**, only its serve) |
| `--max-context` | 8192 (long-prompt scenarios set their own) |
| Sampling | **`--greedy`** — for reconciliation only; it changes the output distribution and is **not a production default** |
| Thinking | **`--no-thinking`** |
| Generation length | `--max-new 8` for prefill scenarios; 300 tokens for decode |
| Rounds | **median of three alternating rounds** (all three builds run in each round, so thermal drift cannot land entirely on one of them) |

> **⚠️ Absolute values drift ±1~2% between time windows; ratios do not.** Same day, same
> machine, same fixtures, same binary, two full sweeps 45 minutes apart: the previous build's
> long-prompt prefill read 1.25k and then 1.22k (−2.4%), decode 146.9 and then 145.4 (−1.0%) —
> and **all three builds moved by the same amount**, so the percentages in these tables are
> stable and the absolute values are not. The conclusion: **use these tables to compare builds,
> not time windows**; to reconcile against another window, **run both configurations on the
> spot**.

**★ Precision has three independent axes, so say which one is moving:**

| Axis | Same across the three builds? | Notes |
|---|---|---|
| **Weights** | **Yes** | Fixed ternary `PQ2_0_G128` at 2.125 bits per weight. **Never a variable** |
| **Activations** | **No** ← the only one that moves | Which GEMM path is taken (below) |
| **KV cache** | **Yes** | Every table in this section is pinned to `bf16` (see the fixed conditions above) |

So "what precision is that build" is too coarse a question. **Only the activation axis moves**,
and specifically:

| Build | Activation path (at `--max-context 8192`) |
|---|---|
| **Previous** | Only two bf16 paths (a small-T tile path and a large-T MMA path). **It had no s8 path yet** |
| **Upstream ternary** | **One more path at prefill: int8** (activations quantized absmax per token); decode still bf16 |
| **This branch** | Same as upstream |

**That int8 path only applies at prefill** — its threshold is T ≥ 17 (`kTernaryS8MinTokens`),
while the verify width ceiling at decode is 16 (`kMaximumVerifyTokens`). **So the decode table
in §2.2 is in fact the same precision across all three builds** (all bf16); what differs is the
trajectory — prefill numerics propagate through the KV, which is why the acceptance cell moves.

**To compare prefill at equal precision, turn int8 off in all three:**

```bash
NINFER_TERNARY_S8=0 <binary> <model> ...      # the previous build has no such switch; setting it is harmless
```

§2.3 therefore gives two rows: **"each default"** (how it would actually be deployed) and
**"both bf16"** (the pure kernel comparison with the activation difference stripped).

### 6.2 The speed matrix: task × context × thinking × residency

Three tasks, each run at short and long context, with and without thinking, at fp8 and int8 KV,
across three configurations: **all-resident fp8**, **all-resident int8**, **KVMem int8**.

Protocol: CLI, `--greedy`, `--max-new 256`, `--spec mtp`; short context `--max-context 8192`
(prompts 69~340 tokens), long context `--max-context 131072` (prompt ~124k); KVMem given
`--kv-device-tokens 131072 --kvmem-budget 98304 --host-kv-mib 8192` (2,048 device pages against
4,096 logical, **genuinely offloading**). **K is each task's own optimum** (see the footnote
table below).

**Short context**

| Task | Thinking | K | Resident-fp8 | Resident-int8 | KVMem-int8 |
|---|---|---:|---:|---:|---:|
| Pelican on a bicycle | off | 5 | 155.2 | 155.2 | 155.1 |
| Pelican on a bicycle | on | 3 | 146.8 | 146.5 | 146.6 |
| Tool call (pure tool_call) | off | 7 | **232.6** | 232.5 | 232.0 |
| Tool call (pure tool_call) | on | 7 | 194.8 | 194.2 | 194.8 |
| Short story | on | 3 | 101.7 | 107.4 | 107.3 |
| Short story | off | 3 | 89.0 | 85.9 | 85.9 |

**Long context (prompt ~124k)**

| Task | Thinking | K | Resident-fp8 | Resident-int8 | KVMem-int8 |
|---|---|---:|---:|---:|---:|
| Pelican on a bicycle | off | 5 | 113.1 | 108.1 | 108.2 |
| Pelican on a bicycle | on | 3 | 71.1 | 74.6 | 74.7 |
| Short story | on | 3 | 78.8 | 77.3 | 77.3 |
| Short story | off | 3 | 57.2 | 58.8 | 58.9 |
| Tool call | on | 7 | 108.2 | 134.3 | 134.5 |
| Tool call | off | 7 | 182.6 | 180.4 | 181.3 |

**Prefill**: short context 852~1970 tok/s (**it rises with prompt length — the fixed overhead
cannot be amortised away, so it does not measure the engine**); long context, all three tasks
in one band: **resident-fp8 1460**, **resident-int8 1520~1540**, **KVMem-int8 1520~1530**.
A 28k prompt adds **2.30k**, joining §2.3's 2.18k (28k) and 1.81k (62k) and this section's
1.53k (124k) into a line that falls monotonically with length.

**Four conclusions**

1. **KVMem's speed cost is zero.** The two int8 columns are identical cell by cell while
   **genuinely offloading**: 108.1/108.2, 74.6/74.7, 77.3/77.3, 58.8/58.9, 134.3/134.5,
   180.4/181.3; prefill too. Short contexts (fits the pool, no offload) match as well
   (155.2/155.1, 232.5/232.0). **What it buys is a 262k logical context on a 16 GB card, and it
   is not paid for in speed.**
2. **Decode is dominated by MTP acceptance**, not by the KV format or residency mode. At long
   context the fp8/int8 difference runs entirely through acceptance: tool-thinking 108.2 vs
   134.3 (−24%), because fp8 quantization changes the output and acceptance drops from 65.5% to
   49.1%; tool-nothink the two are level (182.6 vs 180.4) because acceptance is 96% either way.
3. **Long context costs about a third**: tool 232 → 181 (−22%), fiction 86 → 59 (−31%).
4. **K moves with the task** (table below); fixing one K puts some tasks on their suboptimum.

| Task | Best K from per-position survival | K=3 | K=5 | K=7 |
|---|---:|---:|---:|---:|
| Pelican on a bicycle | 3 thinking / 5 not | 146.4 | **154.8** | 134.9 |
| Short story | 3 | **101.5** | 88.9 | 81.7 |
| Tool call | **7** | 184.1 | 217.0 | **232.4** |

**One trap**: "tool call" is two different tasks whose speeds differ by 2×. **Pure tool_call**
(structured output, 82~96% acceptance) reaches 232 at K=7; **"write advice after the result
comes back"** (natural language, 42% acceptance) reaches 112. Changing the former into the
latter so the output is long enough to measure decode swaps out the task along with it.

### 6.3 Numerics

| Configuration | PPL |
|---|---:|
| This machine's baseline (`NINFER_TERNARY_S8=0`) | **9.69192** |
| This machine's default (int8 path on) | **9.6934** |

Protocol: `ninfer-perplexity --text wiki-slice.txt --context 512 --stride 256` (110 windows /
28,160 tokens).

**split-K and the T=17..64 window** (`NINFER_TERNARY_S8_KSPLIT`, on by default): in this window
the token axis fits only one 64-slice and the row axis underfills the card
(`gridX = div_up(n,64) < 198`), so s8 slices along K to add CTAs, worth **+7.1%** prefill at
T=28. The cost is that the K accumulation is re-associated into fp32 partials and summed —
**deterministic, but not bit-identical**: PPL at `--context 32` is therefore +0.041%, while
**the two numbers above are unaffected** (`--context 512` has T=508, already outside the window;
the same value is read before and after G3).

> **It does not change decode speed, but it does change decode results.** The verify width
> ceiling at decode is 16 (`kMaximumVerifyTokens`), below s8's own threshold of 17
> (`kTernaryS8MinTokens`, see `x.ne[1] >= ternary_s8_min_tokens()` in
> `ternary_rowsplit_gemm.cu`), **so decode does not go through s8 at all**. What changes is
> prefill numerics, so this prompt's output trajectory can fork: it did on `en-code.json`
> (acceptance 67.6% → 66.0%), while `bash.json` (T=31) and `gap-sm.json` (T=27), also inside the
> window, are **byte-identical**. **Forking is a task-dependent event, not a certainty.** To
> restore the bit-identical path: `NINFER_TERNARY_S8_KSPLIT=0`.

**The second split-K, on the verify path** (`NINFER_TERNARY_SMALL_T_KSPLIT`, on by default,
2026-09-26): it points the **other** way — it slices decode at T ≤ 8, exactly the range s8
cannot reach. It applies only to shapes where the row grid underfills the card **and** at least
4 K steps remain per slice (in this model only `5120×17408`, 4 slices, 1280 CTAs), because
shapes with only 1~2 steps per slice measure **slower** (the prologue cannot be amortised):
without that gate the overall result is **−1%**, with it +1.3~1.4%. Six fixtures +0.2~1.8%,
no accepted-length regression, and **PPL is bit-identical between the two configurations**
(prefill never slices, so the two numbers above are unaffected). It is likewise not
bit-identical; the fallback is `NINFER_TERNARY_SMALL_T_KSPLIT=0`.

> **★ Those two numbers are `fp8` KV numbers.** `ninfer-perplexity` pins its KV to `fp8` in
> source (see `apps/perplexity/main.cpp`), **while the `ninfer` CLI selects by `--max-context`**
> (bf16 at ≤16383, fp8 at ≥16384 — §5.3). Different protocols; **do not mix the numbers** —
> §2.2's performance table runs at `--max-context 8192`, which is bf16 KV, while this section is
> fp8 KV.

The int8 path's shift is **+0.0015%**, which is the magnitude of the int8 activation
quantization error itself.

**The KV axis** (same protocol, five formats, first measured 2026-09-26):

| KV format | PPL | vs `bf16` | Per token | Notes |
|---|---:|---:|---:|---|
| **`bf16`** | **9.688451** | — | 64.00 KiB | CLI default at `--max-context` ≤ 16383. The lossless reference |
| `int8` | 9.691663 | +0.033% | 33.00 KiB | **more accurate than `fp8`**, but slower at decode |
| `fp8` | 9.693396 | +0.051% | 32.25 KiB | CLI default at ≥ 16384, and `ninfer-perplexity`'s default |
| `rk4v4` | 9.702774 | **+0.148%** | **17.00 KiB** | the **only** format that reaches 262,144; slower than `fp8` at decode |
| `rk4v4-e8` | 9.726728 | **+0.395%** | 17.00 KiB | **worse than `rk4v4` at identical capacity and speed** |

Three conclusions:

1. **`int8` is more accurate than `fp8`** (+0.033% vs +0.051%) for only 2.3% more bytes. The
   reason is the encoding: `int8-g64` is an 8-bit **fixed point** with one scale per 64
   dimensions, while `fp8-e4m3-r256` has a **3-bit mantissa** and one scale per 256.
   **Per-element precision differs by about 6×.** But "more accurate" does not mean "a better
   default": `fp8` measures faster at decode (+16.4% vs +13.4% at 64k, §5.3), so `fp8` is the
   long-context default and `int8` is the accuracy-first choice — and the only format KVMem
   accepts (§3.2).
2. **`rk4v4` costs +0.148%**, at the low end of the KV formats' "soft door" (0.05%~1%). As the
   **only** format that reaches 262,144 on a 16 GB card, that is a defensible price. This is the
   first time this repository has quantified it.
3. **`rk4v4-e8` is 3.5× worse than `rk4v4` at identical capacity, speed, and bytes per token.**
   Its decode side is a **deliberate half-coset approximation** (the code comment, verbatim:
   *"the D8+0.5 E8 coset is collapsed … **not an exact E8**"*) — part of the E8 shaping gain is
   thrown away. **In the current implementation `rk4v4-e8` has no reason to exist.**

> ⚠️ **This is the short protocol, `--context 512`.** Per-element quantization error does not
> depend on context length (the KV is written once and only read back), so the table is valid
> numerically; but the longer the context the more selective attention becomes, so the cost can
> be **higher** (not lower). **The long protocol has been run** (`--context 65536 --stride
> 32768`, 124k-token corpus, 3 windows, bf16 anchor 2.099109): `int8` +0.077% / `fp8` +0.143% /
> `rk4v4` +0.213%, criterion ≤0.30%, all four pass. **`rk4v4`'s long-protocol cost is 1.44× its
> short-protocol cost**, in the expected direction.

> ⚠️ **Do not compare this 9.69 against numbers from elsewhere.** PPL is a product of its
> protocol: a different corpus, context, or stride is a different number. The upstream
> document's golden criterion is ≈6.445, which is its own protocol. Reconcile against this
> machine's own history.

### 6.4 Two server behaviours that bite

**① `reasoning_effort` defaults to the model's own `xhigh`.**
`result.reasoning_effort.default_effort = ReasoningEffort::XHigh;` at `chat_template.cpp:432`.
**Turning thinking on without naming a level means the highest level** — it is not something the
caller set. To lower it, pass `reasoning_effort: low|medium|high` explicitly (`request.h:142`
accepts none/minimal/low/medium/high/xhigh).

**② `--default-thinking-budget` is a licence, not a brake.**
The thinking/body split relies on the model emitting control tokens (`frontend.cpp:557`); the
budget only feeds a semantic tracker that is **disabled by default entirely**
(`semantic.in_reasoning = starts_in_reasoning && thinking.budget.has_value()`, and the comment
says it is off by default so every token is not decoded twice). Exceeding the budget **raises
`model output exceeded the licensed thinking budget` — it errors out, it does not wrap up for
the model.** ⇒ **It does not solve "it will not stop thinking".**

**Three things for users** (all direct consequences of the two above, none about model
behaviour):

| You want | Do this |
|---|---|
| Control over the thinking level | Pass `reasoning_effort` explicitly; do not rely on the default |
| Thinking that terminates | **Use an agent loop** — every step ends in a tool call, so the thinking has a terminus (a runnable example is in §4.4). `--default-thinking-budget` is another route, but **it errors on overrun — do not treat it as "wraps up automatically"** |
| Speed above all | `--no-thinking` (trade-off in §5.1) |

**"Thinking terminates" measured** (same binary, same machine, server at `--max-context 200000
--kv-dtype fp8 --spec mtp --draft-tokens 3`, **no `--no-thinking`, no
`--default-thinking-budget`** ⇒ thinking unlimited, level the default `xhigh`):

| | Client limit 32,768 | **Client limit raised to 199,000** |
|---|---|---|
| First usable output | 3 min 22 s / 7 steps | **3 min 27 s / 8 steps** |
| Step 1 thinking | 56,783 characters | **34,844 characters** |
| Total thinking | 58,689 characters | **47,487 characters** |
| Output | 58-line single-file HTML | **49-line single-file HTML** |

**Raising the limit did not make it think more (19% less, in fact), and it terminated and
produced output anyway.** ⇒ **The terminus is "every step has to land an action", not the
output limit.** Raising the limit is neither a way to make long thinking converge nor the reason
it fails to.

---

## 7. Engine internals: T dispatch and constants

### 7.1 Dispatch

The ternary linear layers pick a kernel by token count T. **T is not the prompt length** — the
rule, consistent across seven fixtures:

```
the T the linear layers see = prompt_tokens − 4
```

Current defaults:

| T | Path |
|---|---|
| 1 | `small_t` (tensor core) |
| 2..8 | `small_t` (verify round) |
| 9..16 | `small_t_tiled` (8-wide tile, re-entered every 8 tokens) |
| 17 and up | `s8` (int8 activations × int8 weights, tensor core) |
| (with s8 off) ≥32 | `short_mma` / `wide_mma` (bf16) |

After changing dispatch, **the first thing to do is set `NINFER_TERNARY_S8_DEBUG=1` and confirm
the branch is actually taken** — speed alone cannot tell you whether you hit the intended path.

### 7.2 Measured constants on this machine

Re-measure all of these on a different card.

| Quantity | Value |
|---|---|
| Read-only bandwidth ceiling | 637 GB/s (measured; **whether the unit is GB/s or GiB/s is not yet pinned down**, see below) |
| Weights per pass | **6.80 GB = 6.33 GiB** (64 layers at 6461.8 MB + `output_head` 337.7 MB, summed per tensor; excludes MTP) |
| `s8` cost | T=15 → 44.3 ms, T=28 → 51.3 ms. **Not "near-flat"** — the mma count is independent of T (the tile is fixed at 64 wide), so at T=15 **76% of the mmas run on zeros** |
| `small_t` cost | T=9..16 is about **20.5 ms × ceil(T/8)**; **T=1..8 is completely flat** (T=1 and T=4 measure 13.75 / 13.3 ms — same kernel, same weights) |
| `small_t` / `s8` crossover | **T=17** |
| 64-wide / 128-wide tile crossover | **T=64** |
| decode bandwidth | **6.80 GB / 13.75 ms = 494 GB/s** (about 78% of 637) |
| Kernels per decode token | **1224** (all inside one CUDA graph, **64 ns** between frames, only 3.7% genuinely idle) |
| Decode round budget | `ternary_small_t_mma_kernel` **420.6 calls/round × 37.2 µs = 15 646 µs = 79%**; `ternary_rotate_bf16` 274 × 3.3 = 905 µs (4.6%); `recurrent_record` 577 µs; `pq2_mma_s8` 386 µs |

> ⚠️ **The units in the rows above are inconsistent, and that is a known problem.** The 637
> ceiling is used as GB/s in some derivations (6.80 GB ÷ 637 = 10.7 ms) and as GiB/s in others
> (6.65 ÷ 637 = 10.4 ms). The two differ by 7%. **Do not quote percentages until it is pinned
> down.** One clean single-kernel reading would settle it.

**The decode lever is entirely inside that 79%**, and it is already running at about 73% of
effective bandwidth (7.12 GiB of weights ÷ 672 GB/s = an 11.4 ms floor against 15.6 ms
measured). **Cutting bytes cuts the floor directly, and pays better than squeezing kernel
efficiency.**

---

## 8. Lineage and credits

This work stands entirely on **NINFER** and the forks around it.

| Project | Contribution |
|---|---|
| **[Neroued/ninfer](https://github.com/Neroued/ninfer)** | **Canonical upstream NINFER** — C++20/CUDA architecture, DFlash2, ReplaySSM, Paged KV Cache. Apache-2.0 |
| [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | The original RTX 4090 fork; the E8-lattice `rk4v4-e8` KV storage |
| [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) | Ada `sm_89` kernel optimizations, GDN cooperative-launch fix |
| [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) | Win32/MSVC portability layer |
| [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) | Native Windows MSVC compilation base |
| [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) | Early Ampere work |
| **[Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)** | **The direct base of this branch's source tree** |
| **[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary)** (ModelScope) | **The source of the ternary port itself**: engine-side `patches/`, packing and verification `tools/`, `docs/` technical record |
| **[naamfung/zatfung](https://github.com/naamfung/zatfung)** | **Where KVMem comes from** — host-side KV offload (compaction + re-RoPE), windowed continuation, decoupling device budget from logical entitlement. Forked from this repository's `bd71d74`; `src/kvmem/*`, the KVMem logic in `logical_kv_store.h`, and the related documents all come from that line |

**Method references**: ternary encode/decode semantics follow `ggml-quants.c` in the llama.cpp
ecosystem; the folded Hadamard basis follows PrismML's published runtime and its
`prism.hadamard.*` metadata contract; the tensor-core FWT design was informed by the public
HadaCore and TurboQuant work. These are method references only; the code here is an independent
implementation.

### Model weights

The model is **Ternary Bonsai 2 27B**, built on `Qwen/Qwen3.8-27B` with the architecture
unchanged and the weights quantized to ternary over a Hadamard-rotated basis. **Weight copyright
belongs to its authors and publishers — PrismML and the upstream Qwen lineage — and this
repository does not contain or redistribute any model weights.** A `.ninfer` artifact produced
from them is a weight-derived work, so its redistribution obligations follow the *weight*
licence, not this repository's.

The upstream `NOTICE` and `LICENSE` are retained verbatim. Every file this branch modifies
carries a prominent notice at the top, as Apache-2.0 §4(b) requires.

---

## Other documents

| Document | Contents |
|---|---|
| `.claude/skills/ninfer-perf-tuning/SKILL.md` | **Tuning methodology**: measurement discipline, dispatch probes, A/B design, numerical gates, the traps that lie silently |
| [README.md](README.md) | Chinese edition, same structure |
| `docs/` | Upstream product guides (CLI, serving, performance, evaluation, maintainer) |
| `bench/fixtures/speed-matrix/` | The fixtures behind §6.2, the raw results (`results.jsonl`), and the re-run scripts |

---

**This repository is a derivative work. It is not upstream NINFER.** Everything below the
horizontal rule is the upstream README, unchanged.

---

# NInfer 4090 Windows

> Windows port of NInfer for the NVIDIA GeForce RTX 4090 (`sm_89`, Ada Lovelace). Selected checkpoints. Maximum single-GPU inference performance. **100% Native Windows MSVC (no WSL2 required).**

**[⬇️ Descargar versión precompilada portable v1.0.8 (Windows 11) en GitHub Releases](https://github.com/Ambolio/ninfer-4090-windows/releases/download/v1.0.8-windows/ninfer-4090-windows-v1.0.8.zip)**

> 🖥️ **Companion repository (RTX 5090):** [Ambolio/ninfer-5090-windows](https://github.com/Ambolio/ninfer-5090-windows) — the Blackwell (`sm_120a`) sibling branch. Both repos publish the full two-card benchmark tables: see [Benchmarks — v1.0.7 cross-GPU campaign (2026-09-09)](#benchmarks--v107-cross-gpu-campaign-2026-09-09).

NInfer 4090 Windows is a native Windows 11 port of the upstream
[Neroued/ninfer](https://github.com/Neroued/ninfer) C++20/CUDA inference engine,
adapted to the Ada Lovelace architecture (`sm_89`): kernels fitted to the
48 KiB static shared-memory limit, the E8-lattice `rk4v4-e8` KV storage, MTP3
and DFlash2 speculative decoding, and the WDDM evictable-budget bypass. It
runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs.

The performance numbers in this README were **measured with this build** on a
physical RTX 4090 (section [Measured performance — this build](#measured-performance--this-build)).
They are not upstream numbers.

---

## Project Lineage & Credits

This branch stands on the work of the whole NInfer Windows ecosystem. With
gratitude to all of them — in lineage order:

| Contributor | Repository | Contribution |
|---|---|---|
| **Neroued** | [Neroued/ninfer](https://github.com/Neroued/ninfer) | Canonical upstream: C++20/CUDA architecture, DFlash2, ReplaySSM, Paged KV Cache |
| **UDPSendToFailed** | [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | **Creator of the original RTX 4090 fork**; pioneer of the WDDM evictable-budget bypass on Windows WDDM and of E8 lattice (Conway-Sloane) geometric quantization, `rk4v4-e8` |
| **sergiuszm** | [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) | Ada Lovelace `sm_89` kernel optimizations, `rk4v4-e8` adaptation, GDN cooperative-launch fix |
| **natpate** | [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) | Base Win32/MSVC portability layer, unbuffered asynchronous I/O (`OVERLAPPED`), initial Windows scripts |
| **headpiece747** | [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) | Native Windows MSVC compilation base from which this branch descends |
| **Don-Chad** | [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) | Pioneering Ampere work and early compatibility bridges |
| **dylanbrodiefafard** | [dylanbrodiefafard/ninfer](https://github.com/dylanbrodiefafard/ninfer) | v1.0.8 port: incremental host encode (`48d1857`) |
| **nmorgowicz** | [nmorgowicz/ninfer-windows](https://github.com/nmorgowicz/ninfer-windows) | v1.0.8 port: `--tolerant-tool-calls` (`69b0950`) |

Model foundations: **Qwen Team (Alibaba Cloud)** for the foundational model
architectures, **unsloth** for the NVFP4 quantizations, and **z-lab** for the
DFlash companion weights.

This branch would not exist without that work. See [NOTICE](NOTICE) for the
full legal attribution (Apache-2.0 §4) and third-party details.

---

## Relationship to Upstream (v1.0.8)

This branch tracks upstream `b88c0f6f` (v1.0.7: 7 commits post-v1.0.6 —
MoE pipeline/prefetch/L2 ×3, NVFP4 W4A4 TMA, open-addressed BPE table,
unicode NFC-skip, host-arena fix) plus the sm_89 layer, the post-merge
correctness work of 2026-09-08 (int4-KV accumulator fix `6c4f5a10`,
small-T T=7/8 port `f7cef9e9`+`486f647d`), and — new in v1.0.8 — five
verified ports from the NInfer fork ecosystem (2026-09-09 forkscan, A/B'd
against the v1.0.7 binaries on both cards before the deploy; see
[Benchmarks — v1.0.7 cross-GPU campaign](#benchmarks--v107-cross-gpu-campaign-2026-09-09),
subsection "v1.0.8 A/B on this baseline"):

- **GDN gating pairwise-K** (sergiuszm `5d57fed`, sm_89): pairwise
  K-reduction in the GDN gating-projection `MmaUnsplit` kernel. Resolves
  the v1.0.7 borderline `gdn_gating_proj` test (ratio 1.212 → PASS).
- **T=1 double-buffered Ada MMA** (UDPSendToFailed `39a6f20`, sm_89): the
  T=1 draft head runs the double-buffered Ada MMA path.
- **SM-count CTA sizing** (UDPSendToFailed `45a5ae5`, sm_89): CTA wave
  sizes are derived from the target's SM count instead of an RTX 5090
  constant (fixes oversized CTA waves on GPUs with fewer SMs).
- **`--tolerant-tool-calls`** (nmorgowicz `69b0950`, frontend, both
  cards): opt-in serve flag that keeps a complete Qwen tool call even when
  trailing wrapper garbage follows (off by default; the strict parser
  keeps its all-or-nothing behavior).
- **Incremental host encode** (dylanbrodiefafard `48d1857`, frontend,
  both cards): LRU cache of committed history prefixes with
  loop-position splicing — unchanged history is re-encoded incrementally
  instead of from scratch.

### Shared with upstream

- High-performance C++20/CUDA core and 1:1 compatibility with `.ninfer` artifacts.
- MTP3, DFlash2 (`--spec dflash2 --draft-tokens 7`) and DFlash legacy
  (`K=1..15`) speculative decoding, with transactional ReplaySSM for linear
  GDN states.
- HTTP APIs compatible with OpenAI Chat Completions / Responses and Anthropic
  Messages, including streaming, tools, and token counting.
- Low-latency prefix caching with paged Device/Host KV and State retention.

### Added by this fork

- **Native Windows 11 compilation**: CMake + MSVC 2022 + Ninja + CUDA 13.x —
  no WSL2, no virtualization overhead (`build_windows.bat`, `build_v1.0.8.bat`).
- **WDDM bypass (`--wddm-evictable-budget`)**: D3D12/DXGI residency lock that
  budgets runtime memory against total VRAM instead of the WDDM process
  budget, recovering 1.0–1.5 GB of physically retained VRAM (see
  [Windows WDDM note](#windows-wddm-and-dedicated-gpus)). Concept pioneered
  in [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090).
- **Ada Lovelace adaptations (sm_89 only)**:
  - Kernels adapted to the **48 KiB static shared-memory** limit (static
    schedules ≤48 KiB; dynamic `extern __shared__` with 101,376 B opt-in for
    larger tiles).
  - Integration of **`rk4v4-e8`** (real 4-bit quantization over Conway-Sloane
    E8 lattices) as a KV cache storage.
  - MTP3 profile on Qwen3.6-35B-A3B.

### Verified in v1.0.6 (this branch)

- int4-KV correctness: 27B +6.8% vs v1.0.5 (119.4 tok/s, A/B on this
  hardware, commit `6c4f5a10`), outputs coherent across all int4 KV storages.
- Test suite `ninfer_softmax_attention_test --dflash2-only`: 100% pass on
  bf16/fp8/nvfp4/k8v4 (widths 2..16); the pre-port baseline crashed on W=7.
- **DFlash2 end-to-end on sm_89**: 27B + dflash2 (7 draft tokens) measured at
  104.1 tok/s decode, 28.5% draft acceptance — the e2e item pending since the
  v1.0.6 port is now closed (see
  [Measured performance — this build](#measured-performance--this-build)).
- 260,032-token `rk4v4-e8` KV pool at C=4 with the WDDM budget, 96% of the
  24 GB card resident (measured, not estimated).
- **v1.0.7 test suite on Windows (2026-09-09)**: 103/104 green — 7 skipped
  by-design (real-data + sm_89-only cases), 1 documented pre-existing
  borderline (gdn_gating_proj T=4097, deterministic), 2 excluded on Windows
  (BEX64 0xC0000409 in the MSVC test binaries — not the engine: the v1.0.7
  server with real production data (150k-merge tokenizer, 260k profile)
  boots and serves clean, verified with a :8091 smoke).
- **v1.0.8 test suite on Windows (2026-09-09)**: 104/104 executed green
  (407 s) — the v1.0.7 borderline `gdn_gating_proj` (T=4097, ratio 1.212)
  now PASSES with the pairwise-K port; 7 skipped by-design (same as
  v1.0.7); 3 excluded on Windows (`frontend_test`, `softmax_attention_test`,
  `incremental_encode_test` — BEX64 0xC0000409 in the MSVC test binaries,
  zero output at startup: a test-binary artifact, not the engine; the
  v1.0.8 server with real production data boots and serves clean,
  verified with production-artifact smokes).

Details and A/B measurements: [PORT_v1.0.6.md](PORT_v1.0.6.md).

---

## Windows WDDM and dedicated GPUs

**If your GPU is dedicated, enable `--wddm-evictable-budget` to use its VRAM
to the maximum.** On Windows, the WDDM driver model gives every process a
memory *budget* that is a fraction of total VRAM (the OS holds back the rest
for the display compositor and TDR recovery), and a process that exceeds its
budget gets evicted or fails to commit. NInfer on Windows can instead take a
D3D12/DXGI residency lock and budget against **total VRAM**:

- With the flag, this build ran the 35B-A3B profile with **23,651 MiB
  resident of 24,564 MiB (96%)** — 20.6 GiB weights + a 260,032-token
  `rk4v4-e8` KV pool + 8 GiB pinned host KV, at C=4. Without the flag the
  same configuration does not start on a 24 GB card.
- The flag is safe: if a real GPU memory pressure event happens (e.g. a
  fullscreen game, another CUDA process), WDDM still evicts safely — the
  budget just moves from "a fraction of VRAM" to "VRAM minus the hard
  reserves".
- If you still cannot start the server, lower `--max-context` /
  `--kv-capacity` (or use `--kv-capacity auto`) until startup fits.

The flag is a no-op safety net on multi-GPU systems: pair it with
`CUDA_VISIBLE_DEVICES=<index>` to pin the engine to your card.

---

## Measured performance — this build

Measured **2026-09-08 on a physical RTX 4090 (24 GB, `sm_89`)** with the
pre-compiled binary of this branch, `--wddm-evictable-budget` enabled, single
stream, temperature 0.7. Prefill prompts are deterministic (~12.7k and
~56.3k tokens); decode is one 2,048-token free generation. Timings are the
ones the server itself reports (`prompt_per_second`,
`predicted_per_second`); VRAM is `nvidia-smi` on the 4090.

| Profile | Weights | KV pool (resolved) | Prefill 12.7k tok | Prefill 56.3k tok | Decode 2048 tok | Draft acceptance | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B — `rk4v4-e8`, MTP3 d3, C=4, 260k ctx | 20.6 GiB | 260,032 tok (explicit) | **10,916 tok/s** (TTFT 1.2 s) | **9,745 tok/s** (TTFT 5.8 s) | **391.4 tok/s** | 52.8 % (1,254/2,377) | 23,651 MiB (96 %) |
| Qwen3.8-27B — `rk4v4-e8`, MTP3 d3, C=2, 131k ctx | 16.7 GiB | 262,144 tok (auto) | **2,143 tok/s** (TTFT 6.0 s) | **1,910 tok/s** (TTFT 29.5 s) | **97.8 tok/s** | 37.7 % (1,086/2,880) | 23,001 MiB (94 %) |
| Qwen3.8-27B DFlash2 — `rk4v4-e8`, dflash2 d7, C=2, 131k ctx | 18.3 GiB | 138,752 tok (auto) | **2,079 tok/s** (TTFT 6.2 s) | **1,857 tok/s** (TTFT 30.4 s) | **104.1 tok/s** | 28.5 % (1,362/4,780, 7 tok) | 22,497 MiB (92 %) |

Engine startup (weights load + CUDA graphs): 10.5 s / 9.2 s / 10.1 s
respectively.

Reading the table:

- **35B-A3B vs 27B**: Qwen3.6-35B-A3B is a MoE with ~3B active parameters, so
  on the same 4090 it prefills ~5× and decodes ~4× faster than the dense
  Qwen3.8-27B. Pick it when your workload fits a single 24 GB card.
- **DFlash2 vs MTP3 on 27B**: 104.1 vs 97.8 tok/s (+6.6%). DFlash2 (7-token
  drafts, 28.5 % per-draft acceptance) beats MTP3 (3-token drafts, 37.7 %)
  on this hardware — and this is the first end-to-end DFlash2 measurement on
  `sm_89` (the item left open by the v1.0.6 port).
- **v1.0.5 → v1.0.6 (A/B on this 4090, 27B, e8 KV)**: 111.8 → 119.4 tok/s
  (**+6.8 %**), entirely from the int4-KV accumulator fix `6c4f5a10`.
- **`--kv-capacity auto`** resolved 262,144 tokens for 27B and 138,752 for
  27B-DFlash2 on 24 GB — DFlash2 keeps more draft state, hence the smaller
  pool. All of this is only possible with the WDDM budget; without the flag
  none of the three profiles would start at these capacities.

All model artifacts are published by Neroued. The base
[`qwen3_8_27b.ninfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
artifact used above is public and works with `--kv-dtype rk4v4-e8`
(runtime KV quantization). The DFlash2 artifact
(`qwen3_8_27b_dflash2.ninfer`) is the same public base weights with the
DFlash companion merged in via the upstream converter pipeline; as of
2026-09-08 the merged artifact is not yet published in Neroued's public
HuggingFace repos (verified across all of his public NInfer repos).

---

## Comparison with the upstream repository

The upstream project publishes RTX **5090** reference numbers
(docs/performance, v1.0.6, revision `487f8977`, INT8 group-64 KV, auto
capacity). Our build is the same engine core + the sm_89/WDDM layer above, so
the comparison below is *our measured 4090 numbers vs upstream's published
5090 numbers*:

| Metric (single stream) | Upstream 5090 (published) | This build — 4090 (measured) | Ratio |
|---|---:|---:|---:|
| 35B-A3B prefill ~8k tok (INT8 g64 KV) | 17,705 tok/s | 10,916 tok/s (12.7k tok, `rk4v4-e8`) | **62 %** |
| 35B-A3B MTP3 decode C1 | 642.5 tok/s (68.6 % accept) | 391.4 tok/s (52.8 % accept) | **61 %** |
| 27B prefill ~8k tok (INT8 g64 KV) | 3,275 tok/s (Qwen3.8-27B g64) | 2,143 tok/s (12.8k tok, `rk4v4-e8`) | **65 %** |
| 27B MTP3 decode (upstream row is structured output) | 224.4 tok/s (structured) | 97.8 tok/s (free generation, 37.7 % accept) | 44 % |

Honest caveats — the ratio is *not* a pure port-quality number:

1. **Hardware**: Ada Lovelace `sm_89` (24 GB) vs Blackwell `sm_120a` (32 GB).
   60–65 % of the 5090 prefill rate on the 4090 is exactly what the
   generation gap implies; the port itself adds no measurable overhead.
2. **KV dtype**: upstream published tables use INT8 group-64; our runs use
   `rk4v4-e8` (E8-lattice 4-bit), which trades a little accuracy for
   ~2× KV capacity.
3. **Artifacts & acceptance**: draft acceptance depends on the MTP/dflash2
   head baked into the *artifact* and on the prompt content, not on the
   runtime. Our runs used locally quantized artifacts (35B-A3B v2; 27B +
   DFlash2) on free-form generation, while the upstream rows use the public
   artifacts (and, for the 27B decode row, a structured-output scenario).
   That gap is why the decode ratio reads lower than the prefill ratio. To
   isolate the port itself, the like-for-like structured MTP3 point was
   measured on the 5090 (see the
   [ninfer-5090-windows](https://github.com/Ambolio/ninfer-5090-windows)
   README): Windows 5090 g64 structured = 237.6 tok/s = **106 %** of the same
   upstream 224.4 tok/s point — so the 44 % above is hardware + quantization
   + scenario, not the Windows port.
4. **Concurrency**: upstream's C4 35B number (1,213.5 tok/s) is *aggregate*
   over four concurrent streams; our 391.4 tok/s is a single stream on a
   C=4 server (no batching benefit with one request).

Within the *same* hardware, the port's effect is measured separately: the
v1.0.5 → v1.0.6 A/B on this 4090 is **+6.8 %** (int4-KV fix).

---

## Benchmarks — v1.0.7 cross-GPU campaign (2026-09-09)

Measured **2026-09-09** in a single back-to-back campaign on one dual-GPU
machine (Windows 11): the **RTX 4090 (24 GB, `sm_89`)** and the **RTX 5090
(32 GB, `sm_120a`)** — the v1.0.7 binaries (sha256-verified byte-identical to
the production binaries), `--wddm-evictable-budget` on every server, and the
exact per-run argv recorded in each point JSON. Same artifacts, same harness,
same day on both cards.

> 🖥️ **Sibling repositories:** the full per-card data — methodology,
> deviations registry (D1–D11), raw point JSONs and campaign logs — live in
> both [Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)
> and [Ambolio/ninfer-5090-windows](https://github.com/Ambolio/ninfer-5090-windows).
> This section is identical in both repos on purpose, so each page shows the
> numbers for *both* cards.

**Artifacts used in this campaign** (public HuggingFace repos, by Neroued):

| Artifact | Weights | HuggingFace | Used for |
|---|---|---|---|
| `qwen3_6_35b_a3bv2.ninfer` — Qwen3.6-35B-A3B v2 | groupwise-int, 20.6 GiB | [Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | S3 + P0 — both cards |
| `qwen3_8_27b_nvfp4.ninfer` — Qwen3.8-27B | nvfp4, 21.5 GB | [Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | N0 + NS — 5090 |
| `qwen3_8_27b.ninfer` — Qwen3.8-27B | groupwise-int, 16.7 GiB | [Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | NS — 4090 |

(The 35B-A3B "v2" is the current production conversion of the public
Qwen3.6-35B-A3B family; the 27B rows use the two public 27B conversions —
NVFP4 on the 32 GB card, groupwise-int on the 24 GB card.)

### S3 — 35B-A3B v2, MTP3 d3, saturated decode (both cards)

Stochastic 8,192-token generation per request (293-token prompt), at
concurrency C = 1/2/4/8; int8 KV, `auto` capacity. Steady-state committed
decode rate:

| C | 4090 — steady (tok/s) | 5090 — steady (tok/s) | 5090 / 4090 |
|---:|---:|---:|---:|
| 1 | 459.5 | 672.9 | **1.46×** |
| 2 | 660.7 | 974.3 | **1.47×** |
| 4 | 914.3 | 1,336.4 | **1.46×** |
| 8 | 1,095.5 ¹ | 1,544.5 | **1.41×** |

Draft acceptance: 4090 67.0–71.1 % · 5090 66.3–68.6 % — 8/8 real concurrent
requests on both cards.

¹ **24 GB wall with int8:** the `auto` pool on the 4090 resolves 59,648
tokens < the 8×8,485 needed for C=8, so that point re-ran with the documented
ladder `--kv-dtype rk4v4-e8 --kv-capacity 131072` (E8-lattice KV, 748 MiB
pool) — still 8/8 real, mean batch 8.0. The 5090 resolves the full
131,072-token pool with int8 `auto` in 32 GB.

### NS — 27B, MTP3 d3, saturated decode (both cards)

Same protocol (335-token prompt + 8,192 decode); int8 `auto` KV pools of
16,384 / 32,768 / 65,536 / 113,216 (4090) and 16,384 / 32,768 / 65,536 /
131,072 (5090):

| C | 4090 — steady (tok/s) | 5090 — steady (tok/s) | 5090 / 4090 |
|---:|---:|---:|---:|
| 1 | 108.5 | 148.1 | 1.37× |
| 2 | 164.6 | 281.2 | 1.71× |
| 4 | 185.9 | 491.8 | 2.65× |
| 8 | 290.5 | 827.5 | 2.85× |

Draft acceptance: 4090 46.1–47.9 % · 5090 44.9–46.2 %.

⚠ **Not like-for-like:** the 4090 ran the **groupwise-int** 27B artifact
(16.7 GiB) and the 5090 the **NVFP4** one (21.5 GB), so the widening ratio
(1.37× → 2.85×) is hardware *plus* weight quantization — NVFP4 reads fewer
bytes per token, and the gap grows with concurrency. The S3 table above is
the same-artifact comparison: ~1.45× with the identical 35B-A3B v2 on both
cards.

### P0 — 35B-A3B v2, MTP0 (no speculation), NIAH context corpus (both cards)

20 serial requests = 5 seeds × {8k, 64k, 128k, 256k} contexts
(2,311,680 prompt tokens total). Prefill and decode rates per context point:

| Context (tok) | 4090 prefill | 5090 prefill | 4090 TTFT | 5090 TTFT | 4090 decode | 5090 decode |
|---:|---:|---:|---:|---:|---:|---:|
| 7,680 | 12,375.1 | 18,699.8 | 624 ms | 414 ms | 240.7 | 363.9 |
| 64,512 | 9,418.1 | 12,092.7 | 6,875 ms | 5,363 ms | 202.2 | 317.9 |
| 130,048 | 7,193.4 | 8,417.1 | 18,127 ms | 15,503 ms | 173.2 | 278.2 |
| 260,096 | 4,927.2 | 5,261.9 | 52,884 ms | 49,530 ms | 136.3 | 225.5 |

(prefill/decode in tok/s; full-corpus makespan: 4090 **393.9 s** · 5090
**355.3 s**.)

### N0 — 27B NVFP4, MTP0, NIAH context (5090 only)

| Context (tok) | Prefill (tok/s) | TTFT (ms) | Decode (tok/s) |
|---:|---:|---:|---:|
| 7,680 | 9,780.6 | 789 | 75.9 |
| 64,512 | 5,778.1 | 11,190 | 69.6 |
| 130,048 | 3,866.0 | 33,692 | 63.7 |
| 260,096 | 2,331.6 | 111,650 | 54.6 |

(The 27B context point ran on the 5090 only — the 4090 27B context run was
outside the campaign's fast profile; its 27B decode-saturation point is the
4090 column of the NS table.)

### Windows vs upstream Linux parity (same card)

The point of the campaign: same models, same commands, same GPU — the
measured delta is the overhead of the Windows port (WDDM), nothing else.

- **RTX 5090 — parity.** Windows matches or slightly exceeds the numbers
  upstream published for the same card, across every point of this campaign:
  S3 steady 104.7–111.9 % of upstream, NS steady 103.0–107.9 %, P0 prefill
  100.3–105.6 %, P0 decode 105.9–107.6 %, N0 prefill 105.9–117.3 %.
- **RTX 5090 — structured MTP3 decode (follow-up, same day).** The upstream
  "Structured" single-stream point, re-measured like-for-like on this build
  (same 15-request corpus = 3 structured scenarios × 5 fixed seeds, same
  server flags): g64 **237.6 ± 16.8 tok/s @ 87.5 %** vs upstream
  224.4 ± 13.6 @ 89.5 % → **106 %**; NVFP4 **233.8 ± 10.8 @ 89.5 %** vs
  219.8 ± 8.6 @ 90.8 % → **106 %** (see "Comparison with the upstream
  repository" above).
- **RTX 4090 — 38–79 % of the upstream *5090* reference** (S3 71.5–79.3 %,
  NS 37.9–75.4 % — with the quantization caveat above —, P0 63.5–93.9 %):
  that is the Ada-vs-Blackwell hardware gap, not port overhead. Within the
  same hardware the port sits at parity (100–117 % on the 5090; the 4090's
  own v1.0.5 → v1.0.6 A/B — int4-KV fix — measured +6.8 %).

### Validation (same campaign)

- **4090:** full ctest suite on the v1.0.7 build — 106/109 passed (3
  documented failures: 2 = MSVC test-binary artifact `0xC0000409`, 1 = known
  deterministic borderline; the v1.0.7 server with real production data
  boots and serves clean, :8091 smoke) + 7 skipped by design. pytest
  75 passed / 3 skipped / 1 failed — the single failure is a Windows
  path-separator artifact in a converter test (`endswith("/model")`), not
  port logic.
- **5090:** ctest pass covered by the 4090 run (byte-identical trees); the
  v1.0.7 5090 deployment additionally validated its suite 103/103 executed
  green (1 `DISABLED` on Windows — the BEX64 test-binary artifact, engine
  verified clean with a production-artifact smoke). pytest 75/3/1 (same
  path-separator artifact).

### v1.0.8 A/B on this baseline (2026-09-09)

v1.0.8 = v1.0.7 + the five fork ports listed in
[Relationship to Upstream](#relationship-to-upstream-v108) (the three
sm_89 kernel ports apply to the 4090; the two frontend ports apply to both
cards). Same machine, same day, same like-for-like protocol as the campaign
above, A/B'd against the v1.0.7 binaries before the v1.0.8 deploy:

| Point (steady decode tok/s; P0 = NIAH 262,144 makespan in s, lower = better) | 4090 v1.0.7 | 4090 v1.0.8 | 5090 v1.0.7 | 5090 v1.0.8 |
|---|---:|---:|---:|---:|
| S3 35B C1 (int8 auto) | 459.5 | 459.0 | 672.9 | 671.9 |
| S3 35B C2 (int8 auto) | 660.7 | **680.2** | 974.3 | 972.5 |
| S3 35B C4 (int8 auto) | 914.3 | 918.4 | 1,336.4 | 1,334.4 |
| S3 35B C8 (4090: prod shape `rk4v4-e8` 131,072 · 5090: int8 auto) | 1,095.5 | 1,099.0 | 1,544.5 | 1,530.7 |
| P0 35B NIAH makespan (s) | 393.91 | 394.82 | 355.28 | 355.83 |
| NS 27B C1 (4090 `groupwise-int` / 5090 `nvfp4`; see the NS caveats) | 108.5 | 108.5 | 148.1 | 147.7 |
| NS 27B C2 | 164.6 | 159.2 ¹ | 281.2 | 280.0 |
| NS 27B C4 | 185.9 | 183.6 | 491.8 | 495.4 |
| NS 27B C8 | 290.5 | 289.8 | 827.5 | 830.9 |

**Verdict: no regression on any point (±1 %).** The 35B gains +3 % at C=2
on the 4090, the production shape (C8 `rk4v4-e8`) is stable, and the 5090
is pure parity — its v1.0.8 delta is frontend-only, which is exactly the
expected result.

¹ borderline noise band on the 4090 27B reference point (morning v1.0.7
baseline vs evening v1.0.8; the 27B runs on the 4090 only as a standby
reference, not production; the 27B matrix decode stayed flat at
−0.1…−0.9 % on the same day).

---

## Running the server

### Generic startup (shipped as `start_4090.bat`)

Minimal configuration — adjust `--max-context`, `--kv-capacity` and
`--max-concurrency` to your VRAM and workload:

```bat
@echo off
set CUDA_VISIBLE_DEVICES=0
ninfer-serve.exe qwen3_6_35b_a3b.ninfer ^
 --host 127.0.0.1 --port 8080 ^
 --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 ^
 --wddm-evictable-budget --max-concurrency 2 --device-state-slots 2 ^
 --spec mtp --draft-tokens 3 --lm-head-draft ^
 --prefill-chunk 2048
pause
```

### Flag notes

- `--kv-dtype rk4v4-e8` is the E8-lattice 4-bit KV storage (sm_89 branch).
  It is a **runtime** KV quantization: it works with the public
  `qwen3_6_35b_a3b.ninfer` artifact. `bf16`/`int8`/`fp8` are also accepted.
- `--wddm-evictable-budget` — see [Windows WDDM note](#windows-wddm-and-dedicated-gpus).
- `--kv-capacity auto` resolves the largest pool that fits after weights;
  explicit values are fixed for the process lifetime.
- `--spec mtp --draft-tokens 3` (MTP3) or `--spec dflash2 --draft-tokens 7`
  (DFlash2, needs the DFlash2 companion artifact).
- `--max-concurrency` / `--device-state-slots`: one state slot per active
  request; keep them equal for the simplest scheduling.
- Multi-GPU: set `CUDA_VISIBLE_DEVICES` to the index of *your* GPU as seen by
  CUDA. Note that CUDA's enumeration order can differ from `nvidia-smi`'s
  order on multi-GPU systems — verify with a short probe run or by watching
  which card's VRAM moves.

### Verified 260k profile (35B-A3B, this hardware)

The table row that uses 96 % of the 24 GB card:

```bat
ninfer-serve.exe qwen3_6_35b_a3b.ninfer ^
 --host 127.0.0.1 --port 8080 ^
 --max-context 260000 --kv-capacity 260000 --kv-dtype rk4v4-e8 ^
 --wddm-evictable-budget --max-concurrency 4 --device-state-slots 4 ^
 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 2048
```

---

## Supported Models

Primary target:

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Also verified on this branch (measured above). Model artifacts:
**Neroued** (NInfer checkpoints on HuggingFace).

| Model | Artifact | Download |
|---|---|---|
| Qwen3.8-27B | `qwen3_8_27b.ninfer` (`--kv-dtype rk4v4-e8`, MTP3) | [Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B DFlash2 | `qwen3_8_27b_dflash2.ninfer` (dflash2, 7 drafts) | the public artifact above + DFlash companion, merged with the upstream converter pipeline; the merged artifact is not yet on Neroued's public HF (2026-09-08) |

---

## Requirements

- 64-bit Windows 11 (Native, **no WSL2 required**)
- NVIDIA GeForce RTX 4090 (`sm_89`)
- NVIDIA driver with CUDA 13.x support (pre-compiled ZIP)
- Microsoft Visual C++ Redistributable 2015–2022 (x64)
- Source builds only: Visual Studio 2022 BuildTools, CUDA 13.3, CMake 3.28+, Ninja

---

## Installation (Pre-compiled)

**Download the [ninfer-4090-windows-v1.0.8.zip](https://github.com/Ambolio/ninfer-4090-windows/releases/download/v1.0.8-windows/ninfer-4090-windows-v1.0.8.zip) from the [v1.0.8-windows release](https://github.com/Ambolio/ninfer-4090-windows/releases/tag/v1.0.8-windows).**

The ZIP contains `ninfer-serve.exe` with its runtime DLLs (FFmpeg), a generic
`start_4090.bat`, a `download_model.bat`, and a `LEEME.txt` with instructions
and model links.

1. Extract the ZIP to a folder.
2. Run `download_model.bat` to download the `qwen3_6_35b_a3b.ninfer` model
   file (or download manually from
   [HuggingFace](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer)).
3. Double-click `start_4090.bat` to launch the server.
4. Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

---

## Building from Source (For Developers)

### 1. Build Automatically

```cmd
build_v1.0.8.bat
```

Self-contained: sm_89, vision, Release. Needs this tree + MSVC BuildTools +
CUDA 13.3 + Ninja. Pass an alternative build directory as the first argument.

### 2. Manual CMake Build

Open the **x64 Native Tools Command Prompt** and run:

```cmd
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_ENABLE_AVX2=ON -DNINFER_BUILD_MEDIA_ACQUIRE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 32
```

---

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP3 speculative decoding with draft windows from one to five, DFlash2
  (`--spec dflash2 --draft-tokens 7`, verified e2e on this branch), and DFlash
  legacy (`K=1..15`) on the 35B-A3B target;
- BF16, INT8, FP8, and the sm_89-only `rk4v4`/`rk4v4-e8` lattice KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages,
  including streaming, tools, local response state, token counting, and usage
  accounting.

The product boundary remains intentionally small:

- one RTX 4090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload,
  multi-GPU, or distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the
shared Main Text KV pool used by active requests and retained prefixes; `auto`
resolves the largest legal capacity at startup from the memory remaining after
weights. Explicit capacities remain fixed for the process lifetime.

---

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Port notes v1.0.6 (sm_89 decisions, verification)](PORT_v1.0.6.md)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that Neroued develops out of interest. If you find
it useful and would like to support its continued development, you can
[support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not
come with financial returns, promised services or features, or a role in
project decisions.

---

## License & Attribution

This project is licensed under the [Apache License 2.0](LICENSE).

This repository is a Windows MSVC adaptation of the upstream
[Neroued/ninfer](https://github.com/Neroued/ninfer) project, originally
authored by **Neroued** and licensed under the Apache License 2.0. In
accordance with Apache License 2.0 Section 4, all original attribution and
copyright notices are retained; the lineage credits above and the
[NOTICE](NOTICE) file are part of the distribution.

The published artifacts are derived from
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and the
Qwen 3.8 family. NVFP4 quantizations: [unsloth](https://huggingface.co/unsloth).
DFlash companion weights: [z-lab](https://huggingface.co/z-lab). These source
repositories are distributed under their own licenses. Vendored dependencies
retain their own license files under `third_party/`.

**Third-party binary distribution.** The pre-compiled ZIP packages include
FFmpeg shared libraries (avcodec, avformat, avutil, swresample, swscale) from
the BtbN `ffmpeg-master-latest-win64-gpl-shared` build, distributed under
GPL v2 or later; the full license text ships as `LICENSE-FFMPEG.txt` in the
ZIP and in the repository root. The corresponding source is the FFmpeg
source tree of that build (https://ffmpeg.org,
https://github.com/BtbN/FFmpeg-Builds). NVIDIA, CUDA, and RTX are trademarks
of NVIDIA Corporation. Model weights are **not** redistributed with this
project: users download them directly from HuggingFace under the model
owners' own licenses.

**Disclaimer.** This software is provided "as is" (AS IS), without warranty
of any kind, express or implied. The authors are not liable for hardware
damage, system instability, data loss, or overheating resulting from the use
of these binaries or configurations, including configurations that run the
GPU at or near its full memory and power envelope. You use this software at
your own responsibility.
