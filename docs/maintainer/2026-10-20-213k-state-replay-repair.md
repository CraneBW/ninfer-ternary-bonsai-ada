# 213K 原题：选页视图与混合状态重建

> 本文保留213K修复阶段的原始记录。后续功能轮已补验原4K工作集32K、当前Graph/MTP/DFlash2短语义、保留分支及256K原题；当前结论见 [活跃验证记录](2026-10-20-kvmem-actkv-live-validation.md#本轮最终功能结果)。下文未运行声明仅指当时阶段，P4仍未重测。

2026-10-20，本轮原题修复执行记录。使用已有 Qwen3.8 GSQ 完整权重、原 `long-32k-window-fixed/212992-{0..5}.json` 六请求、32768-token 工作集、36864-token 物理池、INT8、chunk512、C1/base/no-Graph。不更改事实、问题、输出上限或模型，不升级256K。

## 定位与被否定的候选

此前原213K、不可变首写K修复后的213K均在Topaz题错答Saffron的 `copper-lotus-35`。同长完整KV及真正问题前511页视图从空状态的重放均正确；事实已在问题前入选，不能再解释为单纯漏选页。

优先适配参考源码：

- `ecosystem/kvmem/backends/patches/ninfer-kvmem.patch` 的 `memory_query_probe` / `capture_memory_prefix` / `rewind_memory_query`：探测完整Q、回退问题、恢复状态、冻结Q后回放。
- `ecosystem/melis-actkv/src/adapter/llama-memory-kvmem-hybrid.cpp` 的 GDN snapshot / replay 分支。
- 原生 `StateImageDevicePool`、KV `destructive_truncate` / `kvmem_compact` 及 `schedule::prefill_text_chunk`，不另写GDN数学或引入新模型。

第一候选实际保存完整StateImage（conv/recurrent/hidden/可选draft local）到固定Host槽，探测后恢复再回放。两个请求均完整结束：ACK正确，Topaz仍错答 `copper-lotus-35`。trace确认两个请求均执行 `query_replay`，故“仅补query snapshot/replay即可修复213K”被原题否定。该候选源码和收据保存在 `tmp-nvcc/diagnose-213k-20261020/query-replay-v1-ready/`；更早一次backing子目录缺失导致零请求启动失败，单独保存在 `query-replay-v1/`。最终代码移除了无效的153954304-byte/lane额外Host快照。

第二候选加入同组选页视图的完整GDN/KV重建：ACK和Topaz均正确，Topaz首个原反例已解除；随后首次执行到Indigo却错答Jade的 `teal-jasmine-61`，三个请求均完整结束。其trace与原tokenizer审计一致，词法候选是3316页的旧Topaz问题模板，真实Indigo事实在1183页。`window-replay-v2/`保留这次独立问题，不能将Topaz通过写成整档通过。

## 当前修复

词法补充按实际稀有token锚点签名保留最多四组，不能把DF/权重/命中数相同的不同名称当成同一事实。每组选择最新同签名页面及续页，交给原预算选择器；同签名旧版本仍软降序。原五题的真实tokenizer审计均能包含事实页，完整回答另验。

问题完整探测后不采样，先回滚探测KV，冻结本轮Q，然后在原预算内重新选页。根据每页原始token身份与当前packed位置构造同一有序历史视图，复用完整StateImage reset和原生prefill，重建该视图对应的GDN及所有层K/V，再回放原问题后采样。历史重放不包含未来/问题token，不使用预期标签，不加入新的聊天包装。

重建前通过已有每页回滚合同确认独占且无保护，保留逻辑页身份与设备配额；撤销这些页的旧不可变K来源、Host/Cold副本及mean-K有效计数，新来源从重建后的首次K捕获。其余归档历史仍保留，下一问题可以重新选入并重建。Main/backend由既有同组选页约束耦合。

内部重放按chunk512分步占用Engine服务单元，公开prompt token进度和usage只计原请求一次。usage中的cached_tokens仍描述原始逻辑复用前沿，不代表这些内部历史token免除了重算；实际TTFT包含完整重建成本，trace逐次记录window_replay_done token数。准入预留 `1 + ceil(working_tokens / prefill_chunk)` 个额外service units；重建阶段不发布检查点，结束后才发布真实状态。Host临时token IDs不超过工作集，继续使用原设备KV和workspace，无额外GPU模型或工作集。

适用条件：超工作集、可移动独占文本、query_begin不早于reuse_base、完整查询后缀不超过prefill_chunk。固定几何、共享历史及长工具/媒体后缀保持原有限制路径；本轮不对未运行组合作新资格声明。

联合重建改变了K/V和GDN条件化状态，不能据此单独判定某一个GDN算子或re-RoPE数学有错。此方案优先修复已证实的状态一致性问题，增加TTFT；旧P4不能作为新路径性能成绩，sync仍默认。

## 原题验收

原六请求在同步和异步均通过，**12/12正确且完整结束**。实际首轮212276 tokens，最后一题212441 tokens；所有请求与原冻结文件哈希一致，仍是原32K工作集/36864物理池。两个服务均exit0，输出、usage和373次逐次选页序列完全一致，最大选页工作集32768 tokens。

| 原题 | 同步 | 异步 | 原始期望 |
|---|---|---|---|
| ACK | PASS | PASS | ACK |
| Topaz Ledger | PASS | PASS | silver-maple-41 |
| Indigo Archive | PASS | PASS | amber-willow-92 |
| Saffron Register | PASS | PASS | copper-lotus-35 |
| Quartz Parcel 当前更新 | PASS | PASS | copper-lilac-28 |
| Jade Cabinet 工具结果 | PASS | PASS | teal-jasmine-61 |

两臂各373次压实、2471次Host→Cold spill，终态在飞页/字节均0。backing仍位于/home的NVMe文件系统。异步臂实际2批D2H提交/完成、0次async restore、386次fallback；Cold仍是同步文件I/O，不能说所有搬运都已异步或据此采用async默认。

每臂六轮实际历史重建为32676、32691、32659、32691、32662、32697 tokens，共196076；原后缀内部回放另162 tokens。公开prompt进度没有重复增加，但真实计算量明显增加。同步首轮TTFT为3m21.7s，后续五题24.1–26.1s；异步首轮3m15.2s，后续23.2–25.1s。该单对验收不是P4性能测量，不据此计算可采用收益。

验收证据：

- `tmp-nvcc/diagnose-213k-20261020/window-replay-v3/manifest.json`：原请求哈希、模型/二进制签名、完整参数与有界12调用。
- 同目录 `results.json`、`analysis.json`、逐请求SSE/receipt、两臂日志与metrics：回答、终态、计数、选页和真实回放成本。
- `tmp-nvcc/diagnose-213k-20261020/lexical-blocks-test.log`：现有CPU候选/预算回归及新增不同锚点同分/同名最新版本反例，failures=0。受影响服务构建成功；未扩展GPU测试矩阵。
- `query-replay-v1/`：1次零请求backing目录缺失启动失败；`query-replay-v1-ready/`：2完整请求、Topaz仍错；`window-replay-v2/`：3完整请求、Topaz正确但Indigo仍错。错误收据、部分源码快照和一次错误CMake测试目标名的零推理构建日志全部保留。

本轮追加17次模型调用全部有完整终态，其中两个中间候选错误保留；最终12次全正确。加上前期诊断，根ledger共41完整调用、4个历史错误，不是独立准确率样本。此前116次资格化尝试ledger保持原样。

结论仅覆盖以上原213K事实集及当前Qwen3.8 base/no-Graph同步/异步路径。新代码的4K工作集、Graph/MTP/DFlash2矩阵、任意语义文档及P4没有重跑，不能继承旧矩阵为新代码证据；256K仍NOT_RUN。没有下载新权重，没有提交/推送。
