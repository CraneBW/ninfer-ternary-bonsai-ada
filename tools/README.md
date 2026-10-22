# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, benchmark
orchestration, and serving smoke checks. These tools are not part of the public download-and-run
path; normal users should start with the [project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Build the Qwen3.8-27B artifact | [`convert/qwen3_8_27b/`](convert/qwen3_8_27b/) |
| Build the 35B-A3B artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

Qwen3.6-27B is retired as of 2026-10-21. Only Qwen3.8 is actively maintained;
legacy conversion code may still be imported by the shared Qwen3.8 conversion recipes.

## Artifact workflow

The converters consume their fixed local source checkpoints and write one complete `.ninfer`
artifact. The paths below are placeholders for the maintainer's local checkpoint checkouts:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect either result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

The exact source revisions, inventories, formats, and conversion recipes are recorded in
[`docs/maintainer/`](../docs/maintainer/). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.8-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```

## 固定长前缀快照基准

`python3 -m tools.bench.prefix_checkpoint create|prepare|restore --case 213k|256k --output <新目录>` 使用现有Qwen3.8固定长题，通过native Engine保存/新进程恢复完整q0前缀。日常测试默认200W；仅发布前最终验收显式 `--power-limit 300`，结束自动恢复200W。完整命令、数据边界与资格见 [前缀快照说明](../docs/maintainer/2026-10-22-long-prefix-checkpoint.md)。恢复token单独统计，不作为prefill吞吐。

固定快照的 `restore --checkpoint-stage raw --canonical-prepare` 可选择精确canonical回放准备候选（默认关闭、sync）；重现该候选的原对照须同时加 `--no-canonical-query-reuse`。不增加四槽Host预算。来源、200W同binary结果和未完成边界见 [维护记录](../docs/maintainer/2026-10-22-canonical-replay-prepare.md)。

快照 `restore`默认使用 `--checkpoint-io indexed`，复现旧纯I/O对照加 `--checkpoint-stage raw --no-canonical-query-reuse`，再切换 `--checkpoint-io legacy|indexed`。两种都全文件预验证，indexed消费时再验每个1MiB块；派生索引只在完整FNV通过后创建。200W成对首TTFT减少27.6%，[完整结果与来源](../docs/maintainer/2026-10-22-checkpoint-io-optimization.md)。

固定213K/256K基准现默认启用 `--canonical-query-reuse`；参考臂用 `--no-canonical-query-reuse`，重现上述旧纯I/O对照也加此项。200W下原213K三对36/36及256K6/6通过，六题会话成对减少75.49%；只有完整canonical状态后代、精确历史/选页一致、词法候选驻留且有容量才续问，其他情况重建。[实现与边界](../docs/maintainer/2026-10-22-canonical-query-reuse.md)。普通Serve仍须显式环境变量1，原位置模式保持0。

`prepare --case 213k|256k`从已有ABI1加载，只做一次32K重建生成独立 `canonical-ready.nkv`；`restore`现默认auto，有匹配ABI2且启用query reuse时优先加载，否则用原包。参考显式 `--checkpoint-stage raw`；强制新包用 `--checkpoint-stage canonical`。200W同binary213K三对首TTFT32.99→10.35s、会话35.54→13.11s，36/36及256K默认重启6/6，history replay归零；后五题合计有约0.28s回退。首次包/索引准备另计，不保存生成答案。页流水/P4和canonical-prepare测量需显式raw，以实际触发相应工作。[命令、成本及完整边界](../docs/maintainer/2026-10-22-canonical-ready-checkpoint.md)。

完整加载流水现于固定213K默认开启，256K仅功能资格、显式 `--checkpoint-load-pipeline`；参考用 `--no-checkpoint-load-pipeline`。双槽H2D、直接Host读与已验证私有COW首次K区间，213K三对session成对减少中位3.47%；原raw213K与256K各额外6/6。复现上述旧批次时两臂均显式关闭本项以冻结加载策略。[数据和生命周期](../docs/maintainer/2026-10-22-checkpoint-load-pipeline.md)。
