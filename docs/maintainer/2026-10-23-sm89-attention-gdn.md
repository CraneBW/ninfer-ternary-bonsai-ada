# SM89 attention / GDN 适配与数值资格（2026-10-23）

本轮已实现两条默认关闭的候选：main24 attention 的 FP32 split-vector 读取，以及短序列 GDN 的 FP32 Q/K 预归一化。原生 Op 数学、记录和回放闭环验证通过；200W 整包三对及一次 attention-only 消融已完成，未确认稳定整体收益，两个候选继续默认关闭。目标为现有 Qwen3.8 GSQ、RTX 4070 Ti SUPER（SM89、66 SM、16 GiB）、日常 **200W**。未进行 NVFP4 或 300W 发布验收。

## 理论预算与选择

采用前一轮实际 MTP7 / Graph / chunk1024 / 32K 的 [Nsight 归因](../performance/2026-10-23-mtp7-attribution.json)：decode kernel 时间和中，attention producer 占 **11.42%**，GDN state 占 **2.43%**，split reducer 占 **0.591%**。这些比例受观察者影响，也不是 213K 冷首 TTFT 的关键路径比例。

- Split reducer 若局部快一倍，对该 decode kernel 时间和的理想改善约为 **0.30%**。本轮保留既有共享 softmax 统计和 split 顺序，只调整向量读取，不新增 shared staging 或同步。
- GDN 短序列每个 Q/K 行原本会被多个 DV warp 和映射的 value heads 重复归一化。27B 的几何下可出现 32 × 3 次相同归一化；这不是 96 倍 kernel 或模型提速。新准备 kernel、FP32 scratch 的读写与发射成本必须计入整体对照。
- Producer 已有 QK / probability 跨输出 warp 的复用。本轮没有把既有能力列作新增移植，也没有复制 Volta MMA、TP2 或其设备参数。稳定的小收益同样计入，不能按 10% 硬门槛提前否决。

## 固定来源、许可和本机改动

来源固定为 [tuxKOH/ninfer-V100X2@5c51513096a9fceb9bb1609e43b15efecac30f93](https://github.com/tuxKOH/ninfer-V100X2/tree/5c51513096a9fceb9bb1609e43b15efecac30f93)。[前次来源调查](2026-10-22-v100x2-source-map.md) 保留当时未移植、未验证的历史状态；下表记录本轮新增代码。

| 来源机制 | 本机落点与适配 |
| --- | --- |
| [`src/ops/kernel/gqa_attention_decode.cuh`](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/kernel/gqa_attention_decode.cuh) 的 split-vector 协同读取 | `src/ops/softmax_attention/dense/causal_cache/small_t.cuh`、`small_t.cu`。面向单卡 `Hq=24/Hkv=4/D=256`，使用 `float4` 读取连续 FP32 partial；保持每一维的 split 升序累加和最终 BF16 输出。源端 BF16 partial/shared staging 未照搬。其他注册几何维持原 reducer。 |
| [`src/ops/linear_attention/gated_delta_net/recurrent.cu`](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/linear_attention/gated_delta_net/recurrent.cu) 的 `prepare_recurrent_qk_kernel` | 本机同名 `recurrent.cu/.cuh`、`prepared_qk.h`、`gated_delta_net.cpp`、`replay.cpp`。源端用于 Volta 宽输入的思路，适配本机短序列、紧凑 batch 与 raw-record；保留本机归一化归约、FP32 state 和生命周期。 |

已保留源端 [Apache-2.0 LICENSE](../../third_party/ninfer_v100x2/LICENSE) 与 [NOTICE](../../third_party/ninfer_v100x2/NOTICE)，实现处注明固定来源与改动。本轮没有复制含其他来源许可的 Volta MMA、collective 或 vendored Flash kernel。设备和性能结论仍以本机资格为准。

## 数学、资源和恢复边界

`NINFER_SM89_ATTN_SPLIT_VECTOR=1` 和 `NINFER_SM89_GDN_PRENORM=1` 均为显式候选，默认 **0**；在启动时固定，使 planner、首次执行和 CUDA Graph 使用相同的路由及容量合同。

GDN 的 FP32 preparation 只用于直接 recurrence 的 **T=1..63**、width-one batch update，以及 ReplayRecord 的 **T=2..16/B=1..8**。**T≥64 的 BF16 预归一化加 chunked 路径及其 remainder 保持原实现**，没有用顺序 recurrence 替换大批 prefill，也没有新增 BF16 舍入点。

准备阶段从表示后的 raw BF16 Q/K 计算 FP32 normalized Q/K，scratch 不持久化；state 仍为 FP32，输出仍为 BF16。ReplayRecord 仍逐位保存原始 BF16 K/V 与 FP32 gate/beta；无效 suffix 的 records 不变、输出为精确零，fold 仍读取 raw records 并执行原来的归一化。部分接受、状态槽选择和提交语义不变。

新增两个 FP32 `[128,Hq,total_columns]` 临时缓冲；public capacity query 按包括 batch 在内的总列数计算，并由调用者 `WorkspaceArena` 和 target planner 显式供给，没有隐藏设备分配。`Hq=16/T=16/B=8` 时合计 **2 MiB**，T1/B1 为 16 KiB；本轮没有增加持久 state、KV 或 pinned staging。Record、BatchUpdate、Direct 的调用者和 benchmark 均使用对应容量查询。

## 必要验证与收据

汇总收据为 `tmp-nvcc/opsx-s4-sm89-20261023/qualification/summary.json`。下面只将直接覆盖的合同标为通过，不把数值测试的进程 wall 当作性能成绩。

| 验证 | 结果与范围 |
| --- | --- |
| GDN reference / candidate | `ninfer_gated_delta_net_test` 在 flag 0/1 下均 exit 0。由表示后的 BF16 输入与 FP32 初态直接对完整 FP64 recurrence；覆盖 Hq16/Hv32、48，短 T1/2/8/16/63、chunk 64/65/128、raw-QK、near-zero、in-place/distinct、B1..8、状态槽别名/分叉与边界 guard。 |
| Attention reference / candidate | `ninfer_softmax_attention_test --split-vector-only` 在 flag 0/1 下均 exit 0。公开 A1/A3/Batch 路径对独立 represented-cache FP64 oracle；覆盖 BF16/INT8-G64、T1/3/7/8/16、跨页与 split 层级、32K、碎片映射、非零 column offset、Graph mask/owner 变化及 zero-valid rows。main24 与另一注册几何均纳入。 |
| Attention 完整输出 | 两臂 **4,300,944 bytes** 完整输出逐位相同；候选完成文件位于 `qualification/resume-native/attention-candidate.bin`。初次被终止的 `qualification/attention-candidate.bin` 为不完整文件，不作为该结论依据。 |
| Raw-record / Graph | `ninfer_gated_delta_net_replay_record_test` candidate exit 0；raw K/V/gate 位模式、无效 suffix、输入/初态不变、Graph 重放及与直接 recurrence 输出精确等价通过。 |
| Fold / 部分接受闭环 | `ninfer_gdn_replay_fold_test` 最终 exit 0，见 `qualification/fold-fixed.log`；复用既有逐步 reference 和 projection→record→fold 闭环。首次 fixture 仍固定 256 B reference arena，已修为 public capacity query 后重跑通过；原失败收据保留。 |
| 实际 attention 路由 | 8 GiB scope 的 Nsight probe exit 0；捕获 **34 次** main24 `VectorLoads=true` reducer 发射、6 个模板变体，含 BF16/INT8 和 batch/mask/offset。证明候选在所测公开 Op 路径运行；不代替模型性能或长题验收。 |

初次带 Nsight 的 4 GiB scope 因 **`Result=oom-kill`** 被终止；已停止该次执行并保存 `qualification/report.json`，没有把它解释为数学失败。随后原生验证收据保存在 `qualification/resume-native/report.json`，该文件也保留修复前 fold 的非零退出；最终通过状态由 `summary.json` 连同 `fold-fixed.log` 给出，不覆写先前失败。

可复跑的最小数值命令为对上述测试按相应 flag 分别执行；构建和 GPU 工作必须与模型对照串行。本轮已执行所列命令，不要求再次重复整套测试。

## 编译证据与性能状态

`tmp-nvcc/opsx-s4-sm89-20261023/sass/report.json` 保存实际 SM89 object 的签名、命名 kernel、opcode 统计和原始 SASS。32 个 main24 reducer 构成 16 对其余模板参数相同的 scalar/vector 实例：vector 版全部含 `LDG.E.128`，scalar 版均无。典型实例为 scalar `LDG.E` 8 处，对应 vector `LDG.E` 3 处加 `LDG.E.128` 5 处。

GDN preparation 虽然源代码使用 `store_vec(float4)`，实际 SASS 是 **8 处 scalar `STG.E`，没有 `STG.E.128`**。8 个 prepared recurrence 变体均含 7 处 `LDG.E.128`，其中也包含 FP32 state 加载，不能全部称为 Q/K 加载。以上均为静态指令位置数，不能据此推断动态访存事务、带宽、占用率或性能收益；本次 SASS 检查只在 CPU 上执行。

## 整体性能与采用结论

[结构化结果](../performance/2026-10-23-sm89-attention-gdn.json)保留两组完整输出 token、实际 MTP 接受工作量、启动内存、各对速度与 binary 身份。共同配置为 GSQ v3、C1、INT8-G64、MTP7 optimized head、Graph、chunk1024、wide-prefill=1，pp4096/tg128 与 pp32768/tg128；Graph prime17、warmup1、正式r1。每组按 reference/candidate、candidate/reference、reference/candidate 交替，所有进程 exit0。

| 成对变化中位 | 整包 4K | 整包 32K | attention-only 4K | attention-only 32K |
| --- | ---: | ---: | ---: | ---: |
| prefill tok/s | −0.5354% | +0.0643% | −0.2423% | +0.0059% |
| decode tok/s | −0.0557% | +0.0693% | −0.0733% | −0.2621% |
| 整段耗时减少 | −0.3297% | +0.0627% | −0.2046% | −0.0190% |

整包三对的32K decode为 +0.1124%/+0.0693%/+0.0055%，4K为 −2.2391%/+0.3660%/−0.0557%；仅约0.07%的32K中位且没有跨档收益，尚不足以区分运行波动。按事先计划仅做一次有依据的消融，关闭 GDN preparation、单独开启 attention；未得到稳定改善。未再跑GDN-only矩阵，不因低于10%拒绝收益，也不靠重复测量寻找正数。

全部六对的 Graph-prime、warmup、正式输出 token 完全一致；每正式臂129个输出（128个decode）。实际工作也相同：4K为17轮、117个draft/111个accepted；32K为17轮、113个draft/111个accepted。差异不是提前结束或少做投机工作造成。

**结论：数值资格和本机移植已完成，两条候选均保持默认0，未并入最终性能profile。** 本页原测量结束时，原213K新S4组合为 **NOT_RUN（无胜出组合）**。后续在同日非NVFP4收口中已显式启用两候选，真实冷prefill原213K六题 **6/6精确正确**，资源排空与Serve退出正常，补齐长状态资格；详见[新收据及适用边界](2026-10-23-non-nvfp4-closeout.md)。该正确性验证不改变本页负性能结论，也没有新增256K资格。已验chunk1024与安全历史复用的正确性证据保持适用，不叠加历史百分比。
