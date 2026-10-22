# 2026-10-21：联合恢复与页流水第一批实施

对应 OpenSpec：`absorb-donor-techniques-and-optimize-sm89`，重点 S1/S2。主模型保持现有 Qwen3.8 GSQ v3，SM89 RTX 4070 Ti SUPER。新增候选默认关闭。本文不代表全部计划已实施。

## 来源与适配

Sync/Async 是同一仓库 [paicat1/Ninfer-KVMem-Async](https://github.com/paicat1/Ninfer-KVMem-Async)，不是两个待寻找的项目。本轮锁定：

- main：`f7e5c19180051ab5a0c997a17cba03ac6487e576`；
- engine-main：`a4f32e26deae7a49b7110c4c27e61f590b914a68`。

实际 main 补丁已保存于 `tmp-nvcc/opsx-state-pipeline/sources/`，文件摘要见其中 `manifest.json`。

| 上游实际文件 | 本地采用/待办 |
|---|---|
| [ledger46 inflight demand](https://github.com/paicat1/Ninfer-KVMem-Async/blob/f7e5c19180051ab5a0c997a17cba03ac6487e576/patches/ledger46_v2.1_inflight_demand_fix.patch) | 保留本地真实 free-page 判定；提交和退休分开，承诺释放的页不能分配。没有以在飞需求冲抵硬容量。 |
| [ledger46 multibatch](https://github.com/paicat1/Ninfer-KVMem-Async/blob/f7e5c19180051ab5a0c997a17cba03ac6487e576/patches/ledger46_v2.1_multibatch.patch) | 小幅适配为两个 16MiB 子批的 submit-before-wait。提交下一批后再退休上一批，末批沿用已有 PendingDemote 事务和 pin/epoch 校验。没有声称完整移植跨 compaction 的 N-slot ring。 |
| [ledger42 async demote](https://github.com/paicat1/Ninfer-KVMem-Async/blob/f7e5c19180051ab5a0c997a17cba03ac6487e576/patches/ledger42_v2_async_demote.patch) | 沿用本地完成事件、源 pin、成功后发布 Host 的所有权原则；新增流水不改变 native payload。 |
| [ledger42 cancel checkpoint](https://github.com/paicat1/Ninfer-KVMem-Async/blob/f7e5c19180051ab5a0c997a17cba03ac6487e576/patches/ledger42_cancel_checkpoint.patch) | 参考已提交工作先退休再退出。既有地址释放取消路径本轮复验；跨 forward 的逐页取消 hook 尚未移植。 |
| I3b submit/verify split | 补丁已冻结，尚未接入本机 Engine 的跨轮 step 流水。 |
| engine_registry.json | 登记了 Windows S13 和不同符号构建；源版本、dirty 文件和二进制布局均有差别，不能把注册的 384×SM120a 二进制成绩套到本机 SM89，不能认为 engine-main HEAD 就是注册生产件。 |

KVMem 的不可变原 K、完整 checkpoint、prefix identity，以及 ActKV 的 rewind/spill 所有权仍是本地恢复的依据。冻结版本继承旧维护记录，不重新计算其来源资格。Strata released/ready 是后续逐层 forward 的参照；本轮没有复制其完整 runtime 或页布局。

工作区 `/home/ran/ninfer-work/ecosystem/fusion-kvmem/LICENSE` 与其 engine 源许可证为 Apache-2.0；本轮实现是现有 Ninfer 生命周期上的小幅适配，保留以上具体补丁来源。V100X2 的 kernel/许可映射未在本轮完成。

## 相容改动和理论边界

1. `src/kvmem/kvmem_source_k.h`：新增 `NINFER_KVMEM_SOURCE_EVENT_PIPELINE=1`。两个固定 pinned 槽与两个读取 future；独立 copy stream，记录 ready 后 compute stream 等待，再运行原有 K 重建内核。上一槽的 H2D 完成后即可读下一页，不必等待其重建内核结束。要求目标物理页不重叠，并验证捕获的 owner/handle generation/columns/positions；异常会排空 GPU 和已有读取后返回。资源提前准备，避免在预期的重叠区间首次申请 pinned 缓冲。
2. `src/targets/qwen3_6/impl/runtime/logical_kv_store.h`：新增 `NINFER_KVMEM_TRANSFER_PIPELINE=1`。下一 D2H 子批先提交，再等待上一子批；单 store 最多两个 16MiB 批次在飞，末批保持原事务。页未完成就继续持有源 pin，不能用作空闲页。
3. `src/targets/qwen3_6/impl/runtime/program_impl.h`：单独的 `NINFER_KVMEM_PREFIX_TRACE=1` 只记录 canonical common prefix，便于区分“可复用历史真的改变”和“缺少更合适检查点”；不打开逐页/逐分数的大量 trace。

未修改算子数学、量化、RoPE、归档格式或原题。source-K 单页 Main16 为 1,081,344B，MTP1 为 67,584B；两个槽合计 2,297,856B，未由两槽扩为四槽。D2H 的双批只增加有限在飞 pin/event，不增加已经预留的 Host payload。原先 H2D 后等待整个重建再复用槽，候选以 ready event 缩短该依赖；收益上限来自传输/重建可以重叠的较小一项。读盘带宽、传输字节和 canonical 历史 token 数没有减少。当前没有本轮生产阶段时间比例 f，不能把局部流水的理论重叠当成整机收益；整体比较后才判断是否采用，低于 P4 10%门槛仍保留 sync 默认。

这批是 **page-copy → source-rebuild** 流水，不是逐层 forward-ready。函数完成边界仍有必要的 compute synchronize；取消/I/O 错误先排空。真正消费者层 ready/released 和 Graph 依赖还需继续接入。

## 恢复失败的新增证据

原位置实验只跑冻结原题的前两条诊断请求，日志/终态在 `tmp-nvcc/opsx-state-pipeline/original-diagnosis/`。ACK 正确；Topaz 应答仍把 `silver-maple-41` 错答为 `copper-lilac-28`。服务正常退出 0，已恢复 200W。

- q0 为 212275；完整 StateImage 字节校验通过。
- 查询证据页 99、100 和干扰页 3200 确实进入选页结果，不是遗漏目标 lexical 页。
- 首错位于 Main 首个输出，不能归因于后续 MTP 接受。
- 这些证据仍不足以证明原位置状态与 canonical 稀疏重建具有相同模型行为。R21 根因仍未解决，不能以 memcpy 验证代替原题质量，也不能删除 canonical 回放。

因此 `NINFER_KVMEM_ORIGINAL_POSITIONS=0` 保持不变，S1 原六题门保持未通过。

2026-10-22补充：[固定q0因果对照](2026-10-22-original-replay-causal-control.md)在完整状态及511页Main/MTP有效字节一致时绕过probe/restore，仍得到相同首Main token和错答，排除了该首错的恢复损坏假设。随后[canonical续问复用](2026-10-22-canonical-query-reuse.md)通过原213K三对36/36及256K6/6，大幅减少后续回放；它续用正确canonical状态，不认作R21原位置或逐层流水修复。

## 最少验证与失败保留

复用 R26 的独立 CPU Hadamard/RoPE/quant 参考夹具，未重跑全格式算子矩阵。Main16/MTP1、完整64列/部分17列共四例，新旧路径完整输出字节相同。新增三次真实 pread 场景：两槽已开始读取后改变 owner，必须拒绝；第三次在途读取必须在退出前排空，随后合法恢复可继续。收据：`tmp-nvcc/opsx-state-pipeline/fixture-results-v2/summary.json`。

首次定向 `--async-cancel` 在新旧开关下都触发“批次已经完成”的断言，失败记录保留在 campaign、campaign-v2、cancel-baseline。预分配不是该断言的充分修复。Nsight 小夹具记录在 `cancel-profile.nsys-rep` / `.sqlite`：约 996.7ms 的 stream wait，另有首次内核装载，原夹具只有 1s 的 host gate。仅将该夹具设置 `CUDA_MODULE_LOADING=EAGER` 后，未完成批次拒绝、取消排空、地址代际复用均通过，见 `cancel-eager-candidate/receipt.json`。未放宽断言，也没有给生产 Engine 统一打开 EAGER。

长题和吞吐比较：同一最终 binary、模型、原请求 hash、MTP15，300W 测试、finally 恢复 200W；原题候选通过后才做基准。首对比较单独标注，不把一次快慢当作三对稳定收益。不覆盖失败目录，不自动重试相同失败输入。

## 本轮结果

首对整体对照已结束，收据 `tmp-nvcc/opsx-state-pipeline/campaign-v3/summary.json`，300W，双臂原六题均6/6。新流水页事件路径Main/MTP各18498页，async拆批526次；history replay仍185836、reused仍10240，没有减少历史重建。首次TTFT 147.066→148.974s；后五题均值18.069→17.745s；六题TTFT和237.409→237.697s（增加0.12%）。这是首对观察，不是三对P4通过。没有足够整体收益，保持默认关闭，不扩大重复计时。

|输入|基准prefill/decode tok/s|候选prefill/decode tok/s|
|---|---:|---:|
|4096|2072.89 / 171.28|2069.48 / 168.60|
|32768|1894.56 / 205.73|1895.99 / 205.29|

每臂完整129个输出ID一致。4K/32K resident Bench未开启KVMem，属于未命中流水的对照，不能将波动归因于页流水。结束功耗已恢复200W。用户随后要求213K、256K固定输入的跨进程完整前缀快照，下一批先实现该项，避免以后反复支付整段prefill。

后续213K/256K完整q0快照已经完成，见 [快照实现与资格](2026-10-22-long-prefix-checkpoint.md)。后续同一ABI/输入的warm测试使用恢复包，cold-prefill仍独立保留；此项不改变上述页流水首对负结果。
