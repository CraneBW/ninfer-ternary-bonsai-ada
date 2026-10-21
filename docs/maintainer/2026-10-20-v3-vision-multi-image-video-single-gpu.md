# Qwen3.8 GSQ v3 Vision: multi-image and MP4 single-GPU requests

2026-10-20. Actual native `ninfer-serve` Chat Completions using the already
SHA-256-verified complete Qwen3.8 GSQ 27B v3 artifact on one RTX 4070 Ti
SUPER. These are narrow multimodal transport/preprocessing/inference smoke
tests, **not** broad semantic accuracy evaluations.

## Two image parts in one user turn

- Two `image_url` data-URL PNG parts: first a 64x64 RGB gradient, second a
  64x64 solid-blue image generated in the same local test.
- Prompt asked which image was gradient and which was solid blue.
- The native service reported `media 2`, 167 input and 18 generated tokens,
  `finish_reason=stop`, HTTP 200, total request 717 ms.
- Exact response: `The first image shows a smooth multicolor gradient. The
  second image is solid blue.` This matches the constructed test images.

Local evidence:

```text
tmp-nvcc/v3-vision-two-images-request.json
tmp-nvcc/gsq-v3-multi-image-20261020.log
tmp-nvcc/gsq-v3-multi-image-response.json
```

## Actual short MP4 `video_url`

- Created locally with `ffmpeg` testsrc2 at 64x64, two frames per second,
  two-second duration, H.264 MP4 (~8 KiB).
- Used one `video_url` data-URL in the Chat Completions user message.
- Actual native service reported `media 1`, 52 prompt and 50 generated tokens,
  `finish_reason=stop`, HTTP 200, total request 1.4 seconds.
- The response describes colorful test patterns and a digital timer at the top
  counting from 00:00:00 to 00:00:01. The earlier version of this report called
  the timer a hallucination. **Direct inspection of the original request pixels
  disproves that annotation:** an embedded timer is visible in both the first
  and last decoded frames. See the correction below. This remains a narrow
  clip-level result, not broad video semantic acceptance.

Local evidence:

```text
tmp-nvcc/v3-vision-testsrc2-2s.mp4
tmp-nvcc/v3-vision-short-video-request.json
tmp-nvcc/gsq-v3-video-20261020.log
tmp-nvcc/gsq-v3-video-response.json
```

Both requests ran with `--managed-weights --vision --kv-dtype int8 --greedy
--no-thinking --no-cuda-graph`, an explicit 384-token KV capacity, single
concurrency and small media caches. Servers were shut down after the
request. Neither case establishes long video, multi-video, high-resolution
images, cross-turn replay or strict factual video summarization.

## Decoder packet retention follow-up (host-only)

The FFmpeg `Decoder::frames` loop was corrected to respect
`avcodec_send_packet` returning `EAGAIN`: the packet is not consumed in that
case and must be submitted again after receiving pending frames. Previously
the code could unref that packet before retrying, risking dropped video frames
with buffered decoders. The targeted `ninfer_media_decode_test` passed.

A separate host-only decode probe used both the real 2-second testsrc2 MP4
above and `examples/cli/media/temporal_events.mp4`; decoded and inspected
frame counts agreed exactly at **4/4** and **40/40** frames respectively.
These probes do not intentionally force FFmpeg's `EAGAIN` return, nor do they
demonstrate any model-level semantic-quality change. The timer hallucination
annotation in the earlier inference report is superseded by the original-pixel
audit below; decoder correctness alone still does not imply general model quality.

## Variable-frame-rate container without `nb_frames` — regression fixed

An independently generated 64x64 Matroska testsrc2 clip retains every fifth
frame from a nominal 15-fps input. `ffprobe -count_frames` reports **six**
decoded frames, but the container has `nb_frames=N/A`, a nominal stream rate
of 15 fps and a duration of about 1.734 s. Previously the media planner
estimated about 26 frames via `duration * fps`, requested nonexistent sample
indices and disagreed with the actual decoded frame sequence.

If container `nb_frames` is unavailable, `make_video_plan` now counts real
decoded frames under the existing source-frame and cancellation limits instead
of estimating. The deterministic 9-KiB fixture
`tests/fixtures/media/vfr_sparse_6frames.mkv` is checked by
`ninfer_media_decode_test`: `inspect_video` and `decode_video` both produce
6/6 samples at indices `[0,1,2,3,4,5]`, with 64x64 RGB frames. The existing
4-frame MP4 and 40-frame `temporal_events.mp4` also still decoded at 4/4 and
40/40 in targeted host checks. No whole-model run was repeated; VFR video
understanding quality is therefore not established.

## Original video pixel audit: timer concern retracted

The follow-up decoded the data URL from the **original**
`v3-vision-short-video-request.json`, rather than regenerating a different clip.
Those bytes exactly equal `v3-vision-testsrc2-2s.mp4`, SHA-256
`b2ee73ff600180506c44763e4bc8662bdcb3cca60d88c203642036fdb53152cc`.
Frame 0 visibly contains the top-left `00:00:00` timer; frame 3 shows
`00:00:01` with a fractional suffix. The model's timer observation therefore
has visible evidence. The earlier assertion that this fixture had no timer was
an annotation error, and is withdrawn. No additional model call was needed.

Audit files are under `tmp-nvcc/resolve-20261020/`:
`video-request-original.mp4`, `video-frame0.png`, `video-frame3.png` and
`video-audit.json`. This correction does not establish high-resolution, long
video, multi-video or unrestricted factual-caption reliability.

## Bounded higher-resolution and 60-second temporal qualification

The follow-up uses only the existing complete Qwen3.8 GSQ model on the same
single Ada GPU. Six frozen calls completed with correct independently known
color facts, not merely HTTP200. `tmp-nvcc/qualification-20261020/media/`
contains source PNG/MP4 hashes, manifests, commands, request logs, SSE receipts,
metrics and `analysis.json`.

Configuration: managed weights, INT8, 4096 context/physical KV tokens,
128-token prefill chunks, Vision scratch capacity 2048, no thinking/Graph,
greedy, one active request. Media cache 64 MiB/live 128 MiB; async KVMem is not
enabled for Vision. All three owned servers exited normally.

| Case | Observed processing | Expected / actual fact |
|---|---|---|
| 1536×1024 three vertical panels | 6144 raw patches, 1536 vision tokens | left→right red, green, blue; correct |
| two 256×256 solid images | 512 raw patches, 128 vision tokens total | first red, second blue; correct |
| same images, follow-up question | two media cache hits, 1,572,864 patch bytes reused; 158 prompt tokens cached | second image blue; correct |
| 60-second 128×128 H.264 clip, 2 fps | 120 source frames, native inspect/decode 120/120, 3840 raw patches, 960 vision tokens | red→green→blue across three 20-second sections; correct |
| 1536×1024 + MTP3 | same 1536 vision tokens, accepted/drafted 6/6 | red, green, blue; correct |
| 1536×1024 + DFlash2 K3 | same 1536 vision tokens, accepted/drafted 4/6 | red, green, blue; correct |

The registered resize/sampling policy preserves these already aligned source
dimensions; the clip's source rate equals the target 2 fps and is below the
768-frame ceiling. This qualifies source-resolution panel colors, multi-image
ordering/reuse and one minute of simple temporal color order under these
budgets. It does not qualify high-detail OCR, arbitrary long-video reasoning,
all resolutions, or async+Vision. No new weight download or full kernel suite
was needed. Original smaller smoke results and annotation correction remain
historical evidence rather than being overwritten.
