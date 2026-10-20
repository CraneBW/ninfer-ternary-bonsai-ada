# Native Ada Qwen3.8 GSQ: combined Vision + speculative decoding

2026-10-20. Single RTX 4070 Ti SUPER, `sm_89`, 16 GiB. The user explicitly
paused dual-GPU work. These are **two actual single-GPU Chat Completions**
against the already verified full `qwen3_8_gsq_rco_iq3_s_v3.ninfer`; neither
test is a TP2/NCCL acceptance test. No weights were downloaded or rehashed.

The same 64x64 synthetic RGB gradient image was sent in an OpenAI `image_url`
data-URL request (`tmp-nvcc/v3-vision-gradient64-request.json`). Previous
separate text/spec and Vision smoke tests had already passed, but did not
establish their combined scheduling/materialization path.

## Actual HTTP outcomes

| Configuration | Weight load | Real response | Speculative draft accepted | Observed duration |
|---|---:|---|---:|---:|
| `--vision --spec mtp --draft-tokens 2` | 11.5 GiB | HTTP 200, coherent gradient description; 90 prompt/24 output tokens | 12/20 (60.0%) | 549 ms request |
| `--vision --spec dflash2 --draft-tokens 2` | 13.3 GiB | HTTP 200, gradient and corners correctly described; 90 prompt/24 output tokens | 11/23 (47.8%) | 6.1 s request |

Both outputs were limited to 24 tokens (`finish_reason=length`), so these
results demonstrate a narrow executable multimodal/speculative combination,
not general image understanding or long-horizon stability.

The MTP combination used a 512-token context, 512-token KV capacity, INT8 KV,
managed weights, 256 Vision tokens, 128 MiB media cache and 256 MiB live-media
budget. The DFlash2 combination used a 256-token context and capacity, INT8
KV, managed weights, one host state, 128 Vision tokens, 64 MiB media cache and
128 MiB live-media budget. All runs used `--no-thinking --no-cuda-graph` and
greedy decoding. They each used one GPU and one actual service process.

The DFlash2 configuration exposed only **53.8 MiB free device memory after
startup**, with 9.7 seconds warmup and 5.7 seconds first-token latency.
Consequently, do not extrapolate it to higher concurrency, long contexts,
video, or multi-image without additional implementation/validation.

Raw local (untracked) evidence:

```text
tmp-nvcc/gsq-v3-vision-mtp-combined-20261020.log
tmp-nvcc/gsq-v3-vision-mtp-combined-response.json
tmp-nvcc/gsq-v3-vision-dflash2-combined-20261020.log
tmp-nvcc/gsq-v3-vision-dflash2-combined-response.json
```

Both servers were terminated normally after their one request.

## Remaining limits

- Full-length image responses, multi-image and video validation.
- Combined speculative decoding across extended contexts and KV compaction.
- Groupwise/NVFP4 alternate weights with all feature combinations.
- Actual native TP2/NCCL numeric execution, intentionally not attempted.
