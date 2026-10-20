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
- The response describes colorful test patterns but also describes a digital
  timer at the top of the video. The test video was not created with a timer,
  so this is **not** reliable video semantic accuracy. The request confirms
  accepted MP4 transport, frame preprocessing, Vision inference and text
  generation only.

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
observed in the earlier inference result remains a known limitation.
