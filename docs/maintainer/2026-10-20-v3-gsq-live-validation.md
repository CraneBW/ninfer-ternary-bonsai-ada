# Qwen3.8 GSQ v3: first real Native text/MTP/DFlash2 service validation

Validated on 2026-10-20, single RTX 4070 Ti SUPER (16 GiB), Native Ada engine
after `b4dc030`. Source artifact is **fully downloaded and SHA-256-verified**:

```text
/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer
15017456896 bytes
SHA-256 29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750
```

The original unmodified v3 frontend rejected the valid model *after* loading
10.9 GiB of weights with `tokenizer_config.json.chat_template does not match
frontend/chat_template.jinja`. These are intentionally different: v3 embeds
the unmodified upstream template in the tokenizer config, but stores a vetted
NInfer derivative as its Jinja resource to support additional developer turns,
continued assistant messages and truncated tool histories.

`b4dc030` registers **only the two exact upstream/derived SHA-256 pairs** for
Qwen3.6 thinking-toggle and Qwen3.8 reasoning-effort. Unknown, tampered and
cross-profile template pairs remain rejected. The compiled C++ renderer now
accepts the vetted derivatives with the matching existing semantics. Its
standalone test includes the exact v3 Jinja bytes and negative cases:

```sh
TMPDIR=$PWD/tmp-nvcc cmake --build build --target ninfer_qwen3_6_frontend_test ninfer-serve -j2
./build/tests/ninfer_qwen3_6_frontend_test  # PASS, exit 0
```

## Reproduction

Start *one* of the modes below at a time, using a local free port. The static
weight set changes by mode; do not run these three profiles simultaneously on
one 16 GiB GPU. `--managed-weights` uses unified memory and permits paging.

```sh
MODEL=/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer
./build/apps/ninfer-serve "$MODEL" --host 127.0.0.1 --port 18100 \
  --managed-weights --max-context 512 --kv-capacity 512 --host-kv-mib 64 \
  --max-concurrency 1 --default-max-tokens 16 --kv-dtype int8 --greedy \
  --no-thinking --no-cuda-graph --log-level info

# Use port 18101 and append: --spec mtp --draft-tokens 2
# Or port 18102 and append: --spec dflash2 --draft-tokens 2

curl -sS -H 'Content-Type: application/json' \
  --data '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply exactly with: hello world."}],"max_tokens":16,"temperature":0}' \
  http://127.0.0.1:18100/v1/chat/completions
```

| Mode | Weight upload | Startup | API observation |
|---|---:|---:|---|
| Text only | 10.9 GiB | 7.0 s | HTTP 200; returned `hello` in first short probe |
| `--spec mtp --draft-tokens 2` | 11.3 GiB | 5.6 s | HTTP 200; `hello world.`; accepted `2/2` |
| `--spec dflash2 --draft-tokens 2` | 13.0 GiB | 7.1 s | HTTP 200; `hello world.`; accepted `2/2` |

Requests were non-stream OpenAI Chat Completions with greedy decoding. The MTP
response contained `draft_n:2,draft_n_accepted:2`; the DFlash2 response reported
the same figures. Both were generated against the *full, real* v3 artifact,
not a shape fixture. DFlash2 profile reported only about **180 MiB device VRAM
remaining** at startup, so leave more headroom for real workloads.

Local diagnostic files (untracked, not needed for reproduction):

```text
tmp-nvcc/gsq-v3-full-smoke-after-template.log
tmp-nvcc/gsq-v3-full-smoke-response.json
tmp-nvcc/gsq-v3-full-mtp-smoke.log
tmp-nvcc/gsq-v3-full-mtp-smoke-response.json
tmp-nvcc/gsq-v3-full-dflash2-smoke.log
tmp-nvcc/gsq-v3-full-dflash2-response.json
```

**Scope:** These are startup and *short real-generation* functional tests, not
throughput benchmarks or long-context stability tests. Vision, groupwise/NVFP4
whole-model inference, 2B Companion and real multi-rank TP2 are separately
tracked and must not be represented as passed by this report.
