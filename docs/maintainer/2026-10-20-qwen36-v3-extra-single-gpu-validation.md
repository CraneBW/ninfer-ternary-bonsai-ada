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

Remaining: extended contexts, realistic image evaluation, NVFP4+Vision,
Vision+MTP for Qwen3.6, and high-concurrency deployment. Near-zero
remaining VRAM explains why those tests were not repeated on this host.
