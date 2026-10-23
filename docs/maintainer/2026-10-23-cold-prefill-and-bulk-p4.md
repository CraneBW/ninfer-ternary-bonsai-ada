# 冷首轮宽批与新 Cold 流水 P4（200W）

本轮完成 OpenSpec 3.9、2.14、2.15。已有 R21 安全历史复用保持正确；新宽批候选降低真正冷首轮耗时，但没有消除冷预填充。新 Cold 流水正确性通过，整体小收益尚未与波动区分。全局默认仍为 MTP15 + sync，原位置及宽批均需显式开启。

## 配置与证据边界

- 本机 RTX 4070 Ti SUPER / SM89 / 66 SM / 16GiB，所有运行 200W。
- 已有 Qwen3.8-27B GSQ-RCO-IQ3_S v3，INT8-G64 KV，C1、CUDA Graph。
- 宽批对照使用 MTP7、32,768-token 总工作集、36,864-token 物理 KV 池。原 213K 实际首题 212,276 tokens；原 256K 实际首题 261,411 tokens。
- 每个长上下文臂都是全新服务、真实冷 prefill，未加载快照。冻结输入 SHA、预期答案、HTTP 终态及 usage 逐题保存；没有修改题目或把候选答案加入历史选择。
- 三对参考/候选共用同一二进制。宽批服务二进制为 402,735,304 bytes，mtime_ns=1792735824702694731；P4 使用此前已冻结的 E 二进制，402,729,896 bytes，mtime_ns=1792733297429052838。两组之间不能相减认领收益。
- [结构化宽批结果](../performance/2026-10-23-cold-prefill.json)、[结构化 P4 结果](../performance/2026-10-23-bulk-p4.json)。原始日志位于本地 `tmp-nvcc/opsx-cold-prefill-goal-20261023/`，未入库。

## 理论与批量适配

先复用已有 32K/200W trace：GPU kernel 合计 22.332s，其中 GGUF 矩阵 17.356s（77.72%）、prompt INT8 causal attention 2.165s（9.70%）。该比例不是 213K wall/关键路径占比，不直接乘 160s。

SM89 的现有 MMQ selector 在 N5120/T512/I128/J128 下有 160 tiles，tile-wave 判定为 160/(66×3)=80.8%，实际选 66-block Stream-K 及 fixup；T1024 为 320/(66×5)=96.97%，可走完整 K 并直接输出 BF16。这里的比例用于选择路径，不能当作 97/81 的速度提升，也不能说 baseline 有 160 个空跑 CTA。候选有机会减少 partial/fixup、FP32 中间结果及转换和 chunk 提交。

ActKV `182113cdb` 的 prefill-harvest 文档建议较宽 ubatch 并要求配套 write/fence；本机只取 1024，不复制其 2048 或近似 GDN。DeepInfer `774ce244` 的 chunked-prefill 调度预算仅作参考，不使用其 SM100 数字。沿用本机 GGML（MIT）及已有 R17/R19 kernel 数学和 R28 的动态 columns/blocks 组织；不重新启用已拒绝的 R28 down 路线。

实现包含两处配套变更：

1. `gguf_dispatch.cpp` 的启动固定 `NINFER_GGUF_LARGE_PREFILL=1` 同时把 SM89 内部 MMQ 批宽及 workspace 规划提高到 1024；默认仍为 512。
2. R17/R19 的 N17408/K5120 I64 只新增 T1024 准入，仍是同一 K256、INT8 MMA、FP32 fold、BF16 writer，38,400B shared，4 warps，实际 2,176 blocks。R29 的 T512 融合检查按实际宽度计算，本轮 R29 仍关闭。

第一版仅放宽 I64 准入时，Engine 仍被内部 cap 拆成两次 T512。实际命中 probe 发现后才补 dispatcher；旧收据保留为 `LEGACY_ROUTE_PASS_WIDE_NOT_HIT`，不用于完成验收。最终 probe 确认两个 I64 路由实际均为 T1024。

32K 总工作预算不变，但新块扩大后瞬时可保留历史从 32,256 减为 31,744 tokens。此差异经过原题验收，不声称选页逐位相同。workspace 从 106,914,816 增至 209,484,800 bytes，启动后可用显存从 1,870,790,656 变为 1,761,738,752 bytes，仍高于 256MiB 地板；未增加物理 KV 池。

## 必要资格与整体性能

实际 artifact 的 50 个唯一 linear shapes、9 种受影响编码，在 T1024 和 T1023 尾段各通过完整输出的 CPU 独立解码 + FP64 scaled row-sum 参考及前后 canary，共 100 组。IQ1_M 是未变的 vector 路线，不计入该矩阵。另对 IQ3_S、IQ3_XXS 随机输入逐字节比较全部 35,651,584-byte BF16 输出，参考/候选完全一致。正式 `ninfer-serve`、`ninfer_bench`、`ninfer_gguf_linear_test` 构建通过。

`ninfer_bench` 使用 pp4096/tg128 和 pp32768/tg128，Graph prime、warmup1、每臂 r1，三对交替；计时关闭 trace，完整生成 token 在计时之外保存。所有配对的 Graph prime、warmup、正式输出 token 一致。

| 指标：成对改善中位 | 4K | 32K |
| --- | ---: | ---: |
| prefill tok/s | +3.215% | +3.921% |
| decode tok/s | −0.107% | +0.049% |
| 整段耗时减少 | 2.405% | 3.645% |

decode 基本持平，不认领提速。

## 原 213K 三对与 256K

原位置安全复用候选启用 causal history 与 content anchors，MTP7；历史选择只取完整历史内容，问题提示只在 q0 之后安装。冻结问题和预期答案不参与历史预填充。三对共 36/36 精确正确，usage 对齐，各服务 exit0。

| 对 | chunk512 首轮 TTFT | chunk1024 首轮 TTFT | 成对减少 | 六问 session 减少 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 161.357s | 153.797s | 4.685% | 6.799% |
| 2 | 168.640s | 155.579s | 7.745% | 5.621% |
| 3 | 161.625s | 155.133s | 4.016% | 5.188% |

首轮 TTFT 成对减少中位 **4.685%**，六问 session 减少中位 **5.621%**。不能把两列中位相加。续问多数为 0.67–0.88s，Amber 题仍有 2.28–7.24s 波动；不宣称所有续问亚秒。

所有 213K 臂均记录 logical prompt 工作 212,476 tokens，`computed_prefill_tokens_total=212489`（独立引擎计数）、physical prefill 212,651，history replay **0**，query replay 162，prefix hit 1,061,650。首题之后保留历史复用，没有回到每题全历史 Root 重算；冷首题仍需处理完整输入。

候选原 256K **6/6**，exit0；首轮 TTFT **191.897s**、六问 session **203.026s**。后五题为 0.740/7.155/1.102/0.979/0.864s。history replay 0、query replay 162、prefix hit 1,307,325、logical prompt 261,611、computed 261,624、physical prefill 261,786。该轮仅用于正确性资格，没有参考臂，不能声称 256K 性能改善。

所有长臂终态 active/inflight 均为 0，层汇总 pending pinned/inflight/producer 均为 0。213K host VmHWM 约 17.08GB，256K 为 16.58GB；不是进程 RSS 与 GPU 内存的混合统计。完整启动内存和逐题原值保存在结构化报告。

## 新 bulk Cold 流水 P4

复用 KVMem/ActKV 的不可变 backing、已有固定事件/页 owner 与 Sync/Async 的层依赖，E 批次增加固定 I/O workers 和有界大批 Cold copy。对照为 canonical MTP15，Host 128 pages，32K 总预算/36K 物理池；候选同时启用 source-event、transfer、layer-ready、fixed-workers、cold-copy，参考同步关闭。它与上面的 original MTP7 宽批不是同一 profile。

三对交替共 36/36 精确正确，18 个对应输出与 usage 完全一致，全部 exit0、资源排空。真实 Cold 批次为 237–238 页，四个 worker pool 有实际读取，各 pool 两线程，staging 仍为 9,191,424 bytes（8.765625MiB）。未把零命中或小批 fallback 当作流水验证。

三对 session 耗时改善 **+0.849% / +0.386% / −0.672%**，中位 **+0.386%**。按原计划的“全部正收益且中位至少10%”规则，测量时记录为 `adopt_async=false`。用户随后明确“累计的小提升也是提升”，后续不再以10%作为必要门槛，小而可复现的正收益同样保留、可采用。本轮三对方向混杂，尚不能把0.386%与波动分开，故当前仍保持关闭和采用待定；不重复运行同一矩阵求正数。正确性/P4测量已完成，完整小收益收据保留，分项百分比不直接相加或相乘。

## 复现与采用

在仓库根目录运行（工具会检查固定模型、二进制和 200W，输出目录必须不存在）：

```bash
python3 -m tools.bench.cold_prefill_pairs --phase engine --output <新目录>
python3 -m tools.bench.cold_prefill_pairs --phase long --output <新目录>
python3 -m tools.bench.layer_ready_pairs --bulk --output <新目录>
```

256K 使用 `tools.bench.local_long_quality` 的 `--case 256k --passed-213k <已通过候选目录> --backend mtp --draft-tokens 7 --original --original-causal-prefill --original-content-anchors --prefill-chunk 1024 --large-prefill 1`。它验证前置 213K 的二进制、模型和相关配置，不跨档继承未验资格。

可显式使用本轮已验宽批 profile；没有翻转全局默认，也没有把尚未确认稳定收益的异步 bulk 叠加为“更快整包”。NVFP4 按用户要求未做；未做 300W 发布终测，未推送或打 tag。下一步仍需按 OpenSpec 评估实际 attention/GDN 热点及最终相容 profile，保留条件任务未完成状态。


## 后续小收益的实际预算

提交主批次后，补一份当前 MTP7/chunk1024/32K 的 Nsight Systems CUDA+NVTX/node trace，仅捕获已 warmup 的正式一轮。它用于区分热点，受探针影响，不代替前述无 trace 的整体对照；也不是 213K 的关键路径测量。详见 [MTP7 归因](../performance/2026-10-23-mtp7-attribution.json)。

| GPU kernel 时间和的分类 | prefill | MTP7 decode |
| --- | ---: | ---: |
| GGUF products | 17,680.261ms / 81.81% | 629.150ms / 82.51% |
| attention producer | 2,285.658ms / 10.58% | 87.094ms / 11.42% |
| GDN state | 738.782ms / 3.42% | 18.549ms / 2.43% |
| split reducer | 0.050ms / 0.00023% | 4.509ms / 0.591% |
| 外部 l2norm | 18.198ms / 0.084% | 已在 recurrent 内部执行，不能据外部零值声称无归一化成本 |

当前 MTP7 的 split reducer 确实比旧 MTP15 trace 更值得保留为小收益候选；若条件性地把它加速一倍，对这里的 decode kernel 和最多约改善 0.30%，并非冷 prefill 改善 0.30%。计划与 producer 适配一并验证，不因不足10%丢弃。此估算没有扣除新增 shared/barrier 的成本，也没有消除观察者效应，不能提前认领。

另外已拉取真正的 tuxKOH/ninfer-V100X2 稀疏源码，固定 `5c51513096a9fceb9bb1609e43b15efecac30f93`；它与已有 huangserva/ninfer-v100-tpx 目录不同。源端 `prepare_recurrent_qk_kernel` 用 FP32 Q/K 缓冲，只在 Volta 的 normalize/T≥64 顺序 recurrence 路线启用。本机 T≥64 已经是 BF16 pre-norm 加 chunked；直接搬过来不会消除冷 prefill 的重复归一化。短 T 内每 Q/K 行可能被 32 个 DV warp × 3 个 V heads 重复归一化，需适配本机 raw-BF16 records、FP32 state、部分接受和 fold，不能把“96 次归一化”说成“96倍 kernel 提速”。S4 的实现任务仍未勾选；新源码和预算将用于下一批，未把研究计为实现。
