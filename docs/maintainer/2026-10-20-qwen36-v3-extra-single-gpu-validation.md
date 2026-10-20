# Qwen3.6 27B v3: additional single-GPU feature acceptance

2026-10-20, one RTX 4070 Ti SUPER sm_89, original already-verified
full weight files. Only targeted real requests were run. All completed
servers were stopped after one request; **no TP2 tests**.

## NVFP4 + native MTP

- Source: `real-models/qwen3_6_27b_nvfp4.ninfer`, 16.4 GiB uploaded via
  CUDA managed weights with MTP, KV int8, max context/KV 256 tokens.
- Actual `--spec mtp --draft-tokens 2` warmup PASS (10.5 s).
- Real Chat Completions `What is 1+1? Answer only the number.` returned
  HTTP 200 and `2`, `finish_reason=stop`, accepted **2/2** draft tokens.
- Observed prompt 24 tokens, output 2, request 5.2 s; free device memory
  after startup 74.2 MiB. No throughput or long-context claim.

Evidence:

```text
tmp-nvcc/qwen36-v3-nvfp4-mtp-20261020.log
tmp-nvcc/qwen36-v3-nvfp4-mtp-response.json
```

## Groupwise + native Vision

- Source: `real-models/qwen3_6_27b.ninfer`, real PNG Vision path.
- The first attempt reused an input JSON naming `qwen3.8-27b` and returned
  HTTP 404 `model_not_found` despite successful model load/warmup; this was
  corrected in the request, not by bypassing the model identity guard.
- Correct request: actual 64x64 multicolor gradient PNG as an OpenAI
  `image_url`, with `model=qwen3.6-27b`, `--vision`, managed weights,
  KV int8, context and KV capacity 256, no CUDA graph.
- Real response HTTP 200, output `The image displays a smooth gradient
  transitioning from green on the left to magenta/pink on the right, with no
  distinct` (24-token limit; `finish_reason=length`).
- Prompt 90 / output 24 tokens. Start/warmup PASS; request about **30.0 s**
  (roughly 0.8 tok/s decoding) with only 58.8 MiB free device memory.
  Result supports a narrow multimodal path, not production performance or
  comprehensive semantic accuracy.

Evidence:

```text
tmp-nvcc/qwen36-v3-groupwise-vision-20261020.log
tmp-nvcc/qwen36-v3-groupwise-vision-fixed-20261020.log
tmp-nvcc/qwen36-v3-groupwise-vision-fixed-response.json
```

## DFlash2 is absent from these actual Qwen3.6 v3 artifacts

The validated directory headers for both Groupwise and NVFP4 Qwen3.6 27B
include `text`, `vision`, and `mtp` but **not** a `dflash2` component.
Attempting `--spec dflash2` with the Groupwise file failed **before loading**
with `v3 dflash2: missing text target`; this is the correct rejection for
absent model weights, not a CUDA bug. Do not remove the guard or claim that
these files provide real DFlash2 weights. The Qwen3.8 GSQ artifact does
contain DFlash2 and has independent/combined real request evidence.

## NVFP4 + native Vision — PASS, narrow and memory-constrained

The verified `qwen3_6_27b_nvfp4.ninfer` was separately served with `--vision`
and CUDA managed weights on the same single GPU. It loaded ~16.3 GiB,
completed warmup (13.7 s) and accepted the same RGB gradient PNG in the
native Chat Completions `image_url` format. Response HTTP 200, 90 input and
24 output tokens, truncated by the 24-token output cap, described a smooth
green-to-magenta gradient. The request required **32.3 seconds** due to
device memory pressure, with only **34.2 MiB** free after startup. This
is functional evidence, not acceptable throughput or a stability benchmark.

Evidence (untracked):

```text
tmp-nvcc/qwen36-v3-nvfp4-vision-20261020.log
tmp-nvcc/qwen36-v3-nvfp4-vision-response.json
```

## Groupwise and NVFP4 Vision + MTP — both PASS

Two additional actual Vision/MTP combination tests used the same 64x64 PNG
gradient and a user prompt `Is this a gradient? Answer briefly.` A single
native service process on one RTX 4070 Ti SUPER used managed weights, int8
KV, 256-token context, `--vision --spec mtp --draft-tokens 2`, with small
state and media caches and `--no-cuda-graph`.

| Quantization | Weights | Result | Draft acceptance | Request duration |
|---|---:|---|---:|---:|
| Qwen3.6 Groupwise | 15.9 GiB | HTTP 200; `Yes.`; normal stop | 2/2 | 4.4 s |
| Qwen3.6 NVFP4 | 16.7 GiB | HTTP 200; `Yes, it is a gradient.`; normal stop | 4/6 | 10.6 s |

Both were real requests with an actual image (`media 1`) and correct short
yes/no statements. Groupwise used 86 prompt and 3 output tokens; NVFP4 used
86 prompt and 8 output tokens. These narrow results validate simultaneous
Vision and MTP execution, **not** comprehensive image recognition, quality,
long-context scheduling or production latency. Both servers were terminated.

```text
tmp-nvcc/qwen36-v3-groupwise-vision-mtp-20261020.log
tmp-nvcc/qwen36-v3-groupwise-vision-mtp-response.json
tmp-nvcc/qwen36-v3-nvfp4-vision-mtp-20261020.log
tmp-nvcc/qwen36-v3-nvfp4-vision-mtp-response.json
```

Remaining: extended contexts, realistic image evaluation and high-concurrency
deployment. Near-zero
remaining VRAM explains why those tests were not repeated on this host.
