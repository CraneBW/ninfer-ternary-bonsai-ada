# Companion native v3 configuration checks without model downloads

Companion experimental checkout (leave dirty/uncommitted):

```text
/home/ran/ninfer-work/ecosystem/ninfer-all
```

The native Companion model code already has separate Qwen3.5 (Bonsai),
Gemma4 and Qwen4Exp (Flash-Next) config parsers. To check the actual v3
model variants without downloading terabytes of weights or triggering the
current stale 339-target Companion Ninja rebuild, compile this **source-level
host-only** test against those precise native translation units:

```sh
cd /home/ran/ninfer-work/ecosystem/ninfer-all
ADA=/home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada
/usr/bin/g++ -std=c++20 -O1 -ffunction-sections -fdata-sections \
  -Wl,--gc-sections -Iinclude -Isrc -Ithird_party \
  "$ADA/tools/companion/validate_cpp_v3_configs.cpp" \
  src/models/gemma4/config.cpp src/models/qwen4_exp/config.cpp \
  src/models/qwen3_5/config.cpp src/models/registry.cpp \
  src/artifact/schema.cpp -o "$ADA/tmp-nvcc/companion-v3-config-oracle"
for name in bonsai2b-v3-header.part bonsai27b-v3-header.part \
            gemma4-v3-header.part flashnext-v3-header.part; do
  "$ADA/tmp-nvcc/companion-v3-config-oracle" \
    "/home/ran/ninfer-work/real-models/$name"
done
```

Observed 2026-10-20 results (all **PASS**):

```text
Qwen3.5/Bonsai native configs OK: 24 layers, Hadamard 512
Qwen3.5/Bonsai native configs OK: 64 layers, Hadamard 1024
Gemma4 text+MTP+Vision native configs OK: 30 layers
Qwen4Exp text+MTP+Vision native configs OK: 48 layers, 512 experts
```

The harness reads only the v3 32-byte entry header and declared JSON
directory. It calls the exact C++ configuration parsers used by the Companion
runtime, including requested MTP/Vision or DFlash2 companion geometry when
present. `bonsai27b-v3-header.part`, `gemma4-v3-header.part`, and
`flashnext-v3-header.part` are **prefix-only files**: neither this test nor
their apparent sizes establish weight completeness, successful physical
materialization, CUDA kernel correctness, or real model responses.

No 4B or 9B Bonsai headers were locally available. The previous Bonsai2B
512-point patch remains separate and the Companion work tree must not be
committed or pushed. The earlier formally linked Bonsai2B server passed real
inference, but a fully clean Ninja rebuild is still unverified and this
small host-only oracle does not close that gap.
