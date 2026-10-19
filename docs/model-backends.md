# NInfer model family dispatch

This fork includes an Ada-targeted NInfer runtime for Qwen3.6/Qwen3.8 27B
with KVMem, Strata/ActKV retention and native GGUF weight kernels. The
[NInfer-all](https://github.com/iamwavecut/ninfer-all) runtime is available as
an **optional companion backend** for additional v3 architectures.
The companion is built separately and retains its own model-specific runtime,
GPU kernels and licensing. It does not overwrite the fork's executable or its
working tree.

Supported routing:

| Artifact | Default backend | Notes |
|---|---|---|
| v2 Qwen3.6/Qwen3.8 | fork | Existing artifact interface |
| v3 Qwen3.6 27B, groupwise-int/NVFP4 | fork | Uses per-tensor format descriptors |
| v3 Qwen3.8 GSQ-RCO IQ3_S 27B | fork | Native GGUF CUDA GEMV/MMQ |
| v3 Qwen3.6/Qwen3.8 27B with `--vision`, `--spec mtp` or `--spec dflash2` | fork | Native semantic binding and execution paths; real GSQ full-model feature validation is still pending |
| v3 Qwen3.8 with `--spec dflash` | NInfer-all | DFlash v1 is companion-only |
| v3 Ternary Bonsai 2 (2B/4B/9B/27B) | NInfer-all | Dynamic configuration and T2/GGUF weights |
| v3 Gemma 4 (26B-A4B) | NInfer-all | Gemma 4 native architecture |
| v3 Qwen3.8-Flash-Next 125B | NInfer-all | Hybrid MoE/expert streaming and n-gram |

The default native executable is `build/apps/ninfer-serve`. Build NInfer-all
from its pinned source separately (the observed integration source revision
is `b2913b6`):

```bash
git clone https://github.com/iamwavecut/ninfer-all.git /path/to/ninfer-all
cmake -S /path/to/ninfer-all -B /path/to/ninfer-all/build-ada -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
cmake --build /path/to/ninfer-all/build-ada --target ninfer-serve -j3
export NINFER_ALL_SERVE=/path/to/ninfer-all/build-ada/apps/ninfer-serve
```

The router detects container version and model architecture from the
artifact's directory. No model weights are copied or converted:

```bash
python3 tools/model_router.py /models/model.ninfer --dry-run --max-context 8192
python3 tools/model_router.py /models/model.ninfer --max-context 8192 --kv-dtype int8
```

Use `--backend native` or `--backend companion` for an explicit engine
choice. The selected process takes over with `exec` and receives all
unknown CLI flags unchanged. Both model paths must be supported by the
selected engine and fit in the selected GPU or memory-offloading profile.

TP2 uses a different, two-rank native engine. See
[`tools/tp2/README.md`](../tools/tp2/README.md) for the v2 sharding and
proxy utilities and their native runtime requirements. A single-GPU build
cannot execute a two-GPU all-reduce.
