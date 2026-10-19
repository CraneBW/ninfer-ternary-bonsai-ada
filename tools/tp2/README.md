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

`tp2_proxy.py` provides the HTTP supervisor and mirrored request dispatch
for a **TP2-enabled** pair of NInfer server processes. The native ranks must
support `NINFER_TP_RANK`, shared lockstep, rank-specific weights and the
all-reduce communication hooks; this fork's ordinary single-GPU server does
not become tensor-parallel merely by launching two copies through the proxy.

For a native TP2 implementation, consult the corresponding
`src/targets/qwen3_6_27b_tp2` and `runtime/engine/tp_lockstep` code in the
upstream reference. The script currently handles v2 row-split/NVFP4/FP8
weights; the new v3 mixed-GGUF source needs format-aware block-aligned
sharding before TP2 can use it. Multi-GPU execution and communication require
two actual CUDA devices and cannot be validated on a one-GPU machine.
