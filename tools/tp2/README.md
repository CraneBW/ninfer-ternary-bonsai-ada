# Qwen3.8-27B TP2 shard tools

These programs are adapted from
[`huangserva/ninfer-v100-tpx`](https://github.com/huangserva/ninfer-v100-tpx)
(reference snapshot `9913bc2`, Apache-2.0 license; see `LICENSE.apache-2.0`).

`shard_qwen38_27b.py` creates two rank-specific **v2** artifacts from an
unsharded Qwen3.8-27B NInfer model. Each rank stores half of every supported
Attention/GDN/MLP projected dimension, while shared tensors and frontend
resources are replicated. The converter preserves quantized weight codes and
scales, and `--verify` compares the resulting rank artifacts against the
source.

```bash
python3 -m tools.tp2.shard_qwen38_27b model.ninfer out/qwen38_tp2
python3 -m tools.tp2.shard_qwen38_27b model.ninfer out/qwen38_tp2 --verify 10
python3 -m tools.tp2.test_shard_rules
```

## v3 mixed-GGUF artifact (Qwen3.8-27B)

`shard_v3_qwen38.py` is a separate **stdlib-only** sharder for `NINFER\0\x03`
artifacts. It reads the 32-byte header and embedded JSON directory without
mmap or PyTorch, interprets `bindings.parts.range` in **logical elements**,
and copies quantized GGUF blocks as their original stored bytes. No decoding,
re-quantization, tensor interpolation, or missing-weight substitution occurs.

The source can be just the complete JSON header prefix for planning:

```bash
python3 -m tools.tp2.shard_v3_qwen38 /path/qwen3_8_gsq_rco_iq3_s_v3.ninfer --plan-only
python3 -m tools.tp2.shard_v3_qwen38 /path/qwen3_8_gsq_rco_iq3_s_v3.ninfer --plan-only --plan-json /tmp/tp2-v3-plan.json
```

`--plan-only` outputs actual input bytes available, required complete file
bytes, per-rank payload sizes, and per-mode object counts. The optional JSON
contains both rank directories for inspection; for incomplete input the physical
`input_columns` block assignment remains deliberately unresolved. A plan is
**not** a runnable or complete rank artifact.

Once **all declared input payload bytes** are present, write both ranks:

```bash
python3 -m tools.tp2.shard_v3_qwen38 /path/qwen3_8_gsq_rco_iq3_s_v3.ninfer out/qwen38-tp2
python3 -m tools.tp2.shard_v3_qwen38 /path/qwen3_8_gsq_rco_iq3_s_v3.ninfer out/qwen38-tp2-verified --sha256 29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750
python3 -m unittest -v tests.test_tp2_v3_shard
```

Outputs are `out/qwen38-tp2.rank0.ninfer` and `.rank1.ninfer` in native v3
container framing. Physical byte spans, tensor shapes, binding ranges, object
offsets, and single-file payload sizes are rewritten for each rank. Existing
output files are never silently replaced. Writing validates declared source
length, all physical byte spans, and (on supported filesystems) rejects sparse
holes in weight records. It refuses an incomplete input before creating rank
files. Supply the published full-file digest with `--sha256` to verify all source
bytes before creating either rank; a metadata-only or length-only check does not
establish authentic weights.

Rows are split within each attention Q/G/K/V, GDN Q/K/V/Z, A/B, `a_log`,
`dt_bias`, and MLP gate/up region, including fused GGUF tensors. Columns are
split for attention/GDN output and MLP down; the 10,240-column GDN convolution
is sliced independently across `[2048, 2048, 6144]`. All GGUF column cuts are
at complete 256-logical-element boundaries, including IQ3/IQ4 packed blocks.
BF16/FP32 direct tensors are sliced as native little-endian words; unrelated
weights, vision, DFlash2, frontend resources, norms and token embeddings are
replicated unchanged.

**Special case:** `auxiliary/000000` is an `int32[6144]` physical-input-column
permutation used by GDN output projections. Its 256-column physical GGUF
blocks can be interleaved between ranks, so the writer checks that every
physical block belongs entirely to one rank and that each rank gets an exact
3,072-column permutation. It gathers the corresponding GGUF blocks per row,
preserving their original bytes and physical order, and writes rank-local
`int32[3072]` indices by subtracting `rank * 3072`. Mixed-owner blocks,
duplicates and missing columns are errors. The trailing auxiliary lies at the
end of the real 15 GB source, so a partial download can be planned but not
sharded.

Rank outputs set `metadata.name` to `qwen3.8-27b-tp2-rank0` or `rank1` and
include `metadata.tensor_parallel.rank/world_size`. The current native
**single-rank** model registry admits only `qwen3.8-27b` and `qwen3.6-27b`;
therefore it rejects these rank artifacts before loading. A future TP2-aware
loader must consume the rank metadata and provide actual collective execution.
These tools alone provide **no two-GPU inference or performance evidence**.

`tp2_proxy.py` provides the HTTP supervisor and mirrored request dispatch
for a **TP2-enabled** pair of NInfer server processes. The native ranks must
support `NINFER_TP_RANK`, shared lockstep, rank-specific weights and the
all-reduce communication hooks; this fork's ordinary single-GPU server does
not become tensor-parallel merely by launching two copies through the proxy.

Before opening a port or launching either rank, the proxy now enforces two
**distinct physical CUDA GPUs**, two distinct rank ports, an executable server
binary, and both rank artifact files. For v3, it also checks the rank/world-size
metadata and complete *declared file length*, rejecting header-only shards.
Use `--preflight-only` to validate this without starting an HTTP server:

```bash
python3 -m tools.tp2.tp2_proxy \
  --binary /path/to/tp2-enabled/ninfer-serve \
  --model-prefix out/qwen38-tp2 \
  --gpus 0,1 --rank-ports 18940,18941 --preflight-only
python3 -m unittest -v tests.test_tp2_proxy_preflight
```

The file-length check is **not** a payload digest. Produce rank files from a
SHA-256 verified full source (the sharder supports `--sha256`) and use an
actual two-rank collective-capable executable. On a host with one GPU, the
preflight fails explicitly; it does not attempt to simulate TP2 by running two
non-communicating copies on that GPU. The proxy is a supervisor, **not** the
still-missing native v3/NCCL TP2 model executor in this repository.

The main tree now includes `src/runtime/tp2/nccl_collectives.{h,cpp}` as an
optional NCCL BF16 all-reduce/all-gather data plane and
`src/runtime/engine/tp_lockstep.h` with admission, prefill and decode host
lockstep. CMake auto-detects NCCL; specify `-DNINFER_NCCL_ROOT=/path/to/nccl`
when the headers and library live outside standard locations. Without NCCL the
existing engine still builds, but attempting `NcclCollectives::initialize()`
fails explicitly. Focused build/tests:

```bash
cmake --build build --target ninfer_tp2_nccl_test ninfer_tp2_nccl_mock_test ninfer_tp_lockstep_test -j2
ctest --test-dir build -R '^ninfer_(tp2_nccl(_mock)?|tp_lockstep)_test$' --output-on-failure
```

The mocked NCCL test checks production-call argument wiring, **not** transport
numerics. The native v3 rank-local target executor and two-GPU acceptance are
still missing; do not route TP2 shards into the ordinary single-rank backend.

For the remaining native TP2 implementation, consult the corresponding
`src/targets/qwen3_6_27b_tp2` and `runtime/engine/tp_lockstep` code in the
upstream reference. The original script handles v2 row-split/NVFP4/FP8
weights; the separate v3 script handles mixed GGUF metadata and physical
sharding. Multi-GPU execution and communication require
two actual CUDA devices and cannot be validated on a one-GPU machine.
