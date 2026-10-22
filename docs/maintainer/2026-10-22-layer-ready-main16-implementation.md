# OpenSpec S2/3.3 逐层消费者首个实际闭环（2026-10-22）

对应 `/home/ran/ninfer-work/openspec/changes/absorb-donor-techniques-and-optimize-sm89/` 的 3.3–3.6。本记录是 **Main16 resident rephase 的可运行增量**，不是完整 Cold/H2D→每层消费→所有分支退休、原 213K/P4 的阶段验收。当前 OpenSpec 勾选仍 **17/43**，3.3–3.6 保持未勾选。

## 本次直接接入的生产/消费路径

- `src/kvmem/kvmem_bridge.h`：定义源无关的 `LayerReadyConsumer` 依赖接口；`src/kvmem/kvmem_layer_ready.h` 实现有界 `LayerReadyFence`：一个非阻塞 producer stream、一个 producer 开始屏障、Main16 各 16 个 `ready` / `released` event，不分配额外 Device KV page。初始写与上个消费者均通过 compute-stream event 建立设备依赖。
- `src/kvmem/kvmem_rerope.cu`：保留原 `kvmem_rerope_int8_g64_planes<4>` 的 K 数学和 PageMajor 布局，Main16 的一个实体层重相位完成后才在 producer stream 记录此层 `ready[fidx]`。元数据仍是原先已预留的 pinned Host + Device workspace；下一轮生产前等待旧 producer 退休，不复用仍在进行 DMA 的来源。
- `src/targets/qwen3_6/impl/runtime/logical_kv_store.h`：只在显式 *prefill* compact、SM89/Main16/INT8-G64/PageMajor、非 original/非 canonical replay、非 legacy stream 的组合启用；原 source-K、Host/Cold stage-in 先沿用**原来有同步边界**的代码，resident rephase 的独立批次才送入 layer-ready producer。必要时以整层 sync 逻辑回退；generation-bound `KVAddressSpaceHandle` 绑定当前 producer，下一次 compact 撤销旧 ticket。地址 deactivate/release/truncate/重放之前 drain producer 与最后读者。
- `src/targets/qwen3_6/impl/runtime/{schedule.h,text_prefill_impl.h,text_context.h,text_context_impl.h,program_impl.h}`：仅在普通文本 **Prefill** 把符合身份的 producer 传给 `TextContext`，每个真实 full-attention 层在 `ops::causal_softmax_attention` 之前 `cudaStreamWaitEvent(ready[fidx])`，该 Op 最后一项缓存读取**提交后**立即在同 stream record `released[fidx]`。GDN 层不读该 KV；MTP1、Decode、Graph capture/replay 无新 event 插入，使用已有 sync 路由。
- 新功能初始关闭：`NINFER_KVMEM_LAYER_READY=1` 显式开启；`NINFER_KVMEM_LAYER_READY_TRACE=1` 额外输出 `KVMEM_LAYER_READY epoch=... layers=16 remap_pages=... source_restores=... producer=resident_rephase`。取消这些变量则保留已合格 sync 默认与源格式；无修改整个模型数学/存储 ABI/旧 checkpoint。

## 有针对性的真实验证（SM89 / RTX 4070 Ti SUPER / 200W）

```sh
cmake --build build --target ninfer_kvmem_compact_test ninfer-serve -j2
build/tests/ninfer_kvmem_compact_test --layer-ready
git diff --check
```

均通过。原生 GPU 夹具真实建立 16 层 PageMajor、64 token/page、INT8-G64 K+scale；每层等待生产完成，D2H 模拟消费者在末次读后记录释放；与原 synchronous GPU kernel 完整输出 byte-exact，另以独立 CPU Hadamard/RoPE/quant 计算选择的 head/token 数值（至多 2 个 int8 LSB）。同夹具还驱动真实 `KVAddressSpaceStore` 的 6→4 页 compact、实体生产 ticket、16 个逐层消费、旧票据撤销和 inflight pages=0。中途弃用 producer 的 drain、无效 physical page 请求的拒绝/不发布通过；`VERDICT: PASS`, `failures=0`。构建只有已存在的 `prefix_checkpoint_file.h` `FILE*` attribute warning。

短真实模型请求：`tmp-nvcc/opsx-layer-ready-smoke/{smoke.py,report.json,serve.log}`。现役 `qwen3_8_gsq_rco_iq3_s_v3.ninfer`、INT8 KV、单 GPU、`--max-context 4096 --kv-device-tokens 2048 --kvmem-budget 1024 --prefill-chunk 128 --no-cuda-graph`、`NINFER_KVMEM_LAYER_READY=1`；3659 prompt tokens，2 completion tokens，HTTP 200、返回 `READY`、serve exit 0。**22 次实际 `KVMEM_LAYER_READY` epoch**：大多数 resident remaps 13 页，部分 epoch 包含 5/1 条 source-K restore（该分支仍同步）。短样本含尾页，但这**不是**全图/全部 213K/所有 K/V 内容的验收，也不是 async 相对 sync 的成对性能对照。该自有测试进程正常退出。

追加一次同一短问题的 **MTP3 + CUDA Graph** 功能路由（`tmp-nvcc/opsx-layer-ready-smoke/{graph_mtp.py,graph-mtp/report.json,graph-mtp/serve.log}`），仍为 3659 prompt tokens / 输出 `READY`，HTTP200、22次 Main16 producer 命中，日志确认 `CUDA graphs ready` 且 `mtp accepted 3/3`，serve 正常退出。这只验证 graph/MTP 现有短执行路由与新 prefill hook 可并存；Graph replay 内 per-layer event、MTP 尾部最后一次 reader、长时连续 speculative round **未获**本测试覆盖。

受影响旧 Cold 生命周期两种最小回归 `build/tests/ninfer_kvmem_compact_test --cold-prefetch` 和 `--claim-prefetch` 均再次 `VERDICT: PASS`（0 failure；后者提交/完成3/3、资源最终退休），确认同步 fallback 与已有双槽源所有权未在此更改中明显回归。

## 不能提前标记完成的范围

1. **Cold/Host H2D 仍以整页同步/已完成事件为界**；source-K `restore_many()`/Cold I/O 没有完全拆成按目标层的 H2D ready/rebuild ready。22 个 epoch 证明的是 *resident rephase* 的真实 overlap，而不是 3.3 全部来源的零等待流水。
2. **MTP1** 的 cached KV 在独立对齐、verify/AR 多次读取；`released` 必须在该轮**最后** reader 后，不可在第一次读取后复用。**Graph decode** 的 Host hook 仅在 capture 时执行，不能假装每次 replay 重新打点；当前专门走旧同步路径。
3. **页尾和身份**：本次原生 64 列 PageMajor 与短真实请求的 partial tail 无明显错误；不能据此宣称跨地址撤销/所有 partial/Fork/Graph 被完整覆盖。全地址的物理页身份仍由旧 `LogicalKVPageStore` / reservation / generation 控制。
4. **异常**：定向取消/无效目标和当前 resource drain 通过，但尚未做实际 Cold pread I/O 失败、GPU copy 部分提交后失败、运行时 TERM 在飞以及 Graph 退出的 3.4 完整故障验收。
5. 3.5 的 **ABI1 raw 原始 213K 六题6/6、同 binary 200W sync/async 三对 P4 session wall** 均 **NOT_RUN**。按现有 P4 收益门，成对中位整体 session wall 至少改善 10% 才有资格讨论默认采用。3.1 已测 raw 样本更大的开销是 23.167s 历史 forward，不可将短样本吞吐或 CUDA 重相位与端到端收益混同。

**下一最小工作输入**：给 source-K/Cold Host H2D 按 layer-plane/目标 owner 建立 ready/最终 reader 契约；对 MTP last reader 和 Graph capture/replay 建立单独可重复依赖；增加一个实际 Cold I/O failure/部分 H2D drain 的原生夹具。全部路径通过后才用固定 ABI1 raw 213K 做一次质量臂并评估 P4，期间保持 sync default。

工作树中的其他未提交文件与 `src/ops/linear/q4/q4_dispatch.cpp` 未改；未 commit/push/tag、未下载权重、未改变200W、未启双卡。
