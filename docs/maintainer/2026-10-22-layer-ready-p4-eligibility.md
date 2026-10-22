# OpenSpec 3.5–3.6：逐层流水适用边界、显式命中和P4质量门（2026-10-22）

本记录与 `2026-10-22-layer-ready-s2-failure-drain.md` 配套，仅针对现役 Qwen3.8 GSQ / 单卡 RTX4070Ti SUPER 200W 和本次同一个生产二进制。3.3/3.4的代码与异常/TERM收据已完成；本页完成OpenSpec **3.6的命中与等待/回退说明**，不是3.5的性能验收。

## 已经进入 GPU stream 的可重叠阶段

在 `NINFER_KVMEM_LAYER_READY=1` 且SM89/Main16/INT8-G64/PageMajor/普通文本prefill的受限条件下，先验证 immutable source/Cold owner及generation，再按页准备最多2个现有 pinned source-K 槽、最多2个Cold槽；之后独立CUDA producer逐层发出 `Cold K/V H2D → immutable source-K H2D/rebuild → resident rephase → layer_ready[fidx]`。真正的 `TextContext::attn_mix` consumer在 cached-attention 操作前等待该事件，提交本层最后KV读取后记录 `released[fidx]`；下一轮覆盖物理页时还要等待该consumer stream。每层都有真实事件依赖，但尚未独立测量GPU实际重叠的比例，不能从event数量计算加速百分比。

## 必需等待、固定容量与保守回退

- CPU Cold pread和immutable source的完整读取与代际确认在enqueue前完成；实际Cold未来任务为最多2个，Cold staging总3页（第三页留给同步spill），source-K两页pin不无限增加。copy-ready不等于K重建后的layer-ready。
- Host bulk恢复、超过两页的source/Cold批次、非Main16 geometry、原位置R21、deferred canonical replay及其他未资格路径仍用原同步完成/原事件隔离；用户不设置开关则全局sync默认。
- MTP1 cache可被verify/AR多次读取，不能首个注意力read完成就释放；原CUDA Graph decode replay并未在Graph节点内植入可重臂的逐层事件，因此目前只使用已验证的sync fallback，不认领Graph层流水。
- partial tail仅复制immutable K实际已提交列，Host/Cold source在DMA结束前不得释放；报错时drain future/GPU source/Cold/readers并将地址标记为仅允许释放，避免错误publishing。这里没有更改原K/V数学、snapshot ABI或以前的R21失败。

## 真正会命中新路径的用户命令与收据

```bash
cd /home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada
python3 tmp-nvcc/opsx-layer-cold-ready-smoke/graph_mtp_cold.py
```

该脚本启动自有 `ninfer-serve`，仅显式在子进程传 `NINFER_KVMEM_LAYER_READY=1`、`NINFER_KVMEM_LAYER_READY_TRACE=1`、`NINFER_KVMEM_HOST_PAGES=1`；默认200W、单卡、INT8 KV、KVMem预算1024、短上下文4096、MTP3/原Graph。原真实模型短请求HTTP200，3659 prompt tokens/输出 `READY`，日志22次真实producer epoch，末轮含 `source_layer_async_pages=1 cold_layer_async_pages=1`。额外 `build/tests/ninfer_kvmem_layer_cold_test` 强制Host1页后从Cold恢复并显示2页source-K+1页Cold同epoch PASS。短样本仅验证功能，不是213K/P4性能结果。

## 原213K ABI1 raw P4质量门的实际结果和不可混淆的限制

固定原6题、ABI1 raw首次完整checkpoint、原 `--kvmem-budget 32768`、C1/MTP15/INT8、200W，最新binary启用layer-ready完成 **6/6 exact正确、全部terminal、serve退出0**，两组终态 `KVMEM_LAYER_FINAL pending_pinned_pages=0 inflight_pages=0 producer_active=0`。有效收据 `tmp-nvcc/opsx-layer-p4-20261022/quality-raw213k-layer-ready-port18118/{manifest.json,results.json,213k-layer-ready-raw-port18118.log}`。其中 **`KVMEM_LAYER_READY` 实际记录0次**：raw checkpoint在该用户场景经canonical选择/历史重建，尚未发生适用逐层生产的普通prefill compaction。此6/6只能证明新binary回归正确，**不能**声称新pipeline也在213K通过质量门。

第一份 `quality-raw213k-layer-ready/` manifest 错用Serve端口18151，而原 `kvmem_qualification` 客户端仍固定查询18118；结果为 `sent=0, ConnectionRefusedError`，不是错误答案，隔离保存供诊断。上述新鲜第二份manifest修正成18118并成功完成6次真实请求，二者不得混为6/6+一次失败。

由于P4的新异步分支在冻结六题中未实际命中，**不执行无效的同binary三对并计算虚假改善百分比**；3.5维持未完成，sync全局默认也不改变。下一个最小测量输入必须是真实ABI1 raw+相同业务条件下会在历史forward期间触发`KVMEM_LAYER_READY`的事件/页面选择，可另找固定长工具suffix作为新候选，但不可偷换原213K六题声称已达P4 10%门。当前原位置R21问题仍未解，canonical history成本依旧存在。
