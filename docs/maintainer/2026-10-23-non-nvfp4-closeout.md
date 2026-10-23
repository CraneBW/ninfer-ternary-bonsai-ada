# 非 NVFP4 收口：完整创作质量、AR 收束与有界调度重叠

本轮已完成 OpenSpec 5.5、6.2、12.2、12.3、12.9的实现、诊断与规定范围验收；NVFP4
7.2–7.4仍按用户要求暂缓。完整作品六次自然结束，核心画面五份通过、一份失败；S4原213K
六题通过；新调度功能合格但未获稳定性能收益，保持默认关闭。任务完成不等于所有作品质量
通过或所有候选已采用。所有运行使用本机 RTX 4070 Ti SUPER
（SM89、66 SM、16 GiB）和 200W，未下载模型、未做 300W 发布测试。

## 同条件完整作品

使用已有 `qwen3_8_gsq_rco_iq3_s_v3.ninfer`，本地 Pi 0.99.2 的标准 system prompt 与
read/write/edit/ls 工具，统一空工作目录、seed42、xhigh、262144 逻辑 context、131072 单轮
输出预算（包括思考）、32768 KVMem 工作集、36864 物理 INT8 KV、Graph、chunk1024。
KVMem Host 128 pages 与 context-cache Host 128 MiB 分开配置；磁盘 Cold 有界恢复。
每个 backend 串行完成“鹈鹕骑自行车”和“秦始皇骑北极熊”两道原题，模型实际写文件后结束。

所有正式臂采用模型推荐的 thinking 采样：temperature=1、top_p=0.95、top_k=20、min_p=0、
presence/frequency penalty=0。逐请求核对 xhigh、采样和输出预算。原 greedy 矩阵及其 128K
截断保留在[旧质量报告](2026-10-23-pi-xhigh-quality.md)，不将不同采样样本混合比较。
DFlash2 在共同的 32K 工作集/36K 物理池下不满足 256 MiB 余量门，标为容量拒绝；短容量的
生命周期验证不能替代此长档作品质量。

完整作品墙钟、首模型轮 decode、reasoning/output usage、工具续轮 TTFT 和作品审阅分别报告。
输出长度随采样改变，整作品耗时不是等工作量吞吐对照；同 seed 的 AR 与 MTP 使用不同采样
用途及随机数路径，首文字差异也不自动等于实现降质。现有混合 GSQ 不是纯 IQ3 或 BF16
参考，结论不能推广到所有 IQ 格式或量化前模型。

原始记录：`tmp-nvcc/opsx-pi-model-matrix-20261023/`，实际渲染及逐件记录位于 `visuals/`。

| 后端 / 题目 | 首轮输出（其中思考） | 输出 / decode 秒 | 整作品墙钟 | 实际作品审阅 |
|---|---:|---:|---:|---|
| MTP4 / 鹈鹕 | 33985（30540） | 95.234 tok/s | 360.663s | 核心场景通过 |
| MTP4 / 秦始皇 | 32211（23192） | 90.981 tok/s | 380.651s | 核心场景与动画通过 |
| MTP7 / 鹈鹕 | 28308（24100） | 90.303 tok/s | 319.442s | 核心场景通过 |
| MTP7 / 秦始皇 | 24271（16233） | 80.114 tok/s | 308.751s | 核心场景与动画通过，标题/页脚部分裁切 |
| AR / 鹈鹕 | 19375（15104） | 43.663 tok/s | 451.396s | 核心场景通过 |
| AR / 秦始皇 | 51440（42630） | 42.140 tok/s | 1267.386s | **核心场景失败：骑手与熊不可见** |

六次 Pi 终态均 stop，实际写文件，服务 exit0；两题首次请求分别1674/1678 token、cache0/Root。
上表速度以首轮全部输出除以 decode 时间，包含思考，不含工具后续轮或启动；不是作品墙钟的倒数。
AR 与两个 MTP 后端的首段思考都只共同到第19个字符，此后文本和长度不同。一个 seed、两道
开放式任务不足以证明全局优劣或分布完全一致。

AR 秦始皇在500/1500/8000ms的实际截图中均只有背景/标题；原 HTML 的主体组只有横向动画，
熊体、头部与骑手为负Y坐标，缺少移至地面的整体Y变换。没有修改模型作品来制造通过。
MTP7 的标题裁切同样保留。MTP 两臂本轮未出现这项核心场景错误，但不能据此把 AR 的作品
错误归因于 IQ3 或证明 MTP 普遍提升质量。DFlash2 继续是共同长档容量拒绝，而非质量失败。

## AR 循环的受控诊断

另设相同 Pi cwd、模型、xhigh、上下文、Host 和 seed 的 AR greedy 秦始皇诊断。
仍配置 131072 输出预算，只有检测到三个长窗口的严格重复才取消自有 Pi 请求，避免再次
消耗整段已知循环。该运行只记为 `DIAGNOSTIC_CYCLE_CANCELLED`，不是完整作品 PASS，
也不改变生产推理的停止策略。比较重点是循环开始是否早于 32K 回收，以及同条件官方采样
是否自然完成。固定量化模型上的采样对照不能区分原始模型与权重量化各自贡献。

本次同条件 greedy 诊断在281.122s复现旧轨迹，取消后服务正常退出。完整已观测思考前缀与
旧失败逐字相同；三个窗口起点22292/26762/31232字符，周期均4470字符，两个完整周期也
逐字相同。原生 tokenizer 重新编码到三个起点分别为7423/9760/12097 token；取消时全部
已观测思考12357 token，加1678-token输入仍不到工作集的一半，日志没有回收。因此，
本次循环的开始不依赖32K压缩或重相位，不能归因于MTP，因为此臂关闭了投机。

正式 AR 臂保持同权重、cwd、Host、seed和上下文，仅采用推荐随机采样，首轮自然结束于
51440输出/42630思考。它解决了此前不同cwd补充样本的混杂，证明固定量化模型上采样选择
可以改变此题的收束。修复是 harness 默认使用已注册模型采样，而非悄悄缩短思考或更改
模型权重；greedy仅保留显式诊断。不能据单次配对保证任意seed均收束，也不能进一步断言
“原始模型有错”或“量化完全无关”；量化在本对照中保持常量，没有新的BF16权重对照。
AR生成能够结束和其最终SVG画面错误是两个独立结论。

诊断原始记录：`tmp-nvcc/opsx-pi-ar-cycle-20261023/`。HTTP499是检测到重复后主动断开
诊断客户端的预期终态，不计为作品完成、模型崩溃或服务恢复失败。

## 投机质量的条件诊断

完整作品中未确认额外MTP质量缺陷，故没有为关闭任务而修改采样或量化数学。独立原生
`ninfer_speculative_round_test` 在本次生产构建后、真实GPU上通过；它以CPU FP64参考
分布检验248077-token实际词表域的top-k/top-p、温度和历史惩罚，覆盖K1–15、batch1–8、
接受/拒绝/残差/bonus、重复历史及完整整数变换，并检查缓冲区guard和受影响Graph路径。
这是采样算子的数值/状态证据，不能替代全部模型隐状态的分布等价证明。

结合四份MTP作品核心场景通过、现存真实模型生命周期与算子证据，本轮没有可复现的
“额外投机降质”需要修复；AR的SVG坐标错误不是MTP缺陷，MTP7的裁切也不能仅凭一个
随机作品归因于投机或IQ格式。此前确证的生成回收与endpoint发布问题已分别修复并保留
原失败收据。本项结论严格限制在现有混合GSQ及已执行样本，不称所有量化无损。

## DeepInfer 调度机制的本机适配

来源为本地 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的
`deepinfer/worker/worker.py`、`worker/zos_status.py` 和 `engine/llm_engine.py`。
采用上一轮状态明确归属、当前 GPU 提交先于上一轮 CPU 后处理的顺序；Apache-2.0 与改动
映射保留于 `third_party/deepinfer/`。没有移植 Python 框架、DeepSeek 数学或其 GPU 专用参数。

本机实现仅在 C1 下延后已提交轮次的 Frontend 输出处理和 response publication。
`DeferredPublication` 强持有 Request 和 timing observation，最多一个待处理轮次；Program
在实际 GPU enqueue 后、已有同步前调用同步借用的 `DecodeHostWork`。GPU accepted-prefix、
预算和模型状态提交顺序不变，metadata 不借给另一个线程。取消、admission/control/prefill、
资源事务及终态前先处理待发布输出；回调异常执行既有 GPU 排空清理。
AR/MTP/DFlash2 共用这份合同，C>1/TP 保持同步路径。详见[Engine 架构](engine-architecture.md)。

理论上限必须按可隐藏部分计算。正式 MTP4 鹈鹕第一轮 decode 为 356.856725s，整个 Engine
commit-output 只有 0.018650s，即约 **0.00523%**；真正可延后的部分更小。因此不能将旧 trace
中全部 host submit 的占比算作收益，也不能承诺 DeepInfer 异机加速比。候选开关
`NINFER_HOST_DECODE_OVERLAP` 默认 0，是否采用以同 binary、200W 的整体对照为准。

### 生命周期与整体结果

生产 `ninfer-serve`、`ninfer_bench` 和采样测试构建通过。真实GSQ、200W、C1、Graph、8K
短容量的开关0/1对照分别覆盖AR、MTP4和DFlash2：256-token普通输出、跨输出片段stop、
结构化工具参数，以及流式取消后同进程新请求。两臂正文、usage、停止前缀和工具参数一致；
取消后回答DELTA，active归零，六个Serve均exit0。MTP4普通输出实际接受157/388提议，
DFlash2为157/293，确实覆盖部分接受，AR没有草稿计算。

| 候选后端 | scheduled / processed | GPU提交后处理 | 边界处理 | pending / failures |
|---|---:|---:|---:|---:|
| MTP4 | 160 / 160 | 159 | 1 | 0 / 0 |
| AR | 435 / 435 | 434 | 1 | 0 / 0 |
| DFlash2（短容量） | 160 / 160 | 159 | 1 | 0 / 0 |

整体对照使用同一新binary、GSQ/MTP7、greedy、INT8、chunk1024、既有wide/SiLU、36K物理池，
固定语料4096/32768输入、128输出。三对交错off/on、on/off、off/on；每臂有Graph prime、
warmup和正式测量，完整输出IDs及MTP实际工作量逐对一致。高接受率固定语料与上文随机
xhigh创作是不同负载，不能互换吞吐数字。

| 负载 | prefill速度成对中位变化 | decode速度成对中位变化 | 整段耗时减少中位 |
|---|---:|---:|---:|
| 4K + 128 | +0.383% | **−0.335%** | +0.220% |
| 32K + 128 | −0.065% | **−0.710%** | −0.062% |

4K三对decode变化−1.510%/+0.421%/−0.335%，32K为−0.710%/+0.012%/−0.755%，未建立稳定
正收益。该候选不改变prefill数学，上表prefill波动也不能记为新prefill优化。所有候选性能臂
均scheduled=processed=overlapped=71，pending/failures=0。**功能实现与验证完成，性能不采用，
开关保持默认0**；不再加测追求正数。原始生命周期和三对记录分别为本轮收据目录中的
`host-lifecycle/`、`host-pairs/`。

## 工具续轮的精确匹配边界

[前轮修复](2026-10-23-generation-continuation.md)解决的是不可恢复旧检查点使有效最新 endpoint
被一并丢弃，原请求 TTFT 从 29.383s 降至 0.246s。本轮正式 model 采样 MTP4 秦始皇续轮
仍出现 22.052s：endpoint33888 有效，但新请求与已存 ledger 只匹配到32804，随后 token ID
分别为123177/95849。此前正确性检查确实拒绝了不匹配的 endpoint，回到共享前缀1671，
常规 prefill32279。不能通过放宽 token/state 匹配消除这项重算，也不能把前轮特定请求的
改善宣称为所有工具历史均无需重建。具体失配在生成历史内；尚不把它归因于某个 JSON
序列化字段。

AR 秦始皇也保留相同边界：有效endpoint53117，新输入只匹配到43643，下一token分别为
95779/145007；第一次工具续轮重算51509token、TTFT36.451s，之后精确续轮命中53283、
只算605token、TTFT0.635s。其他正式MTP续轮命中有效endpoint、只算21token，TTFT约0.160–0.202s。

## S4原213K长状态验收

新生产binary同时显式开启 `NINFER_SM89_ATTN_SPLIT_VECTOR=1`、`NINFER_SM89_GDN_PRENORM=1`
和本轮host-overlap候选，使用GSQ/MTP7、Graph、INT8、chunk1024、原位置/causal-history/
content-anchors、32K工作集、36K物理KV、Host128pages及sync。直接读取已冻结的原六题并
核对文本文件hash，**从冷prefill开始，没有加载测试KV快照**。物理池与工作集不等于全历史
驻留注意力，逻辑上限此臂为212992。

| 原题 | 期望与实际（完全相同） | 服务TTFT |
|---|---|---:|
| 1 | ACK | 154.144s |
| 2 | silver-maple-41 | 0.744s |
| 3 | amber-willow-92 | 1.453s |
| 4 | copper-lotus-35 | 1.463s |
| 5 | copper-lilac-28 | 2.130s |
| 6 | teal-jasmine-61 | 1.875s |

六次HTTP200、terminal与usage完整，服务exit0；首请求实际212276 token，续问常规prefill
34/39/40/42/45 token。history replay计数0，客户端session162.218s；这些是一次质量臂的
观测值，不能与历史异binary样本相减认领提速。峰值VmHWM17083650048B（约15.91GiB）。
Main/backend合计5617次Cold页回读、6452514816B，最终inflight pages/bytes、pending pinned、
producer active和active requests均0。host候选scheduled/processed/overlapped均1，pending与
failures为0。结束后GPU59MiB、200W。

这补齐[S4迁移](2026-10-23-sm89-attention-gdn.md)的原213K长状态资格，不推翻旧4K/32K
无稳定收益的结论。S4两个开关和host-overlap均保持默认0；没有新增256K或该组合P4资格，
没有改写原有已通过的P4/256K收据。

## 交付与证据范围

完整作品和greedy诊断先在同一旧生产binary上收集，然后串行构建本轮调度候选；之后的
oracle、生命周期、三对性能及原213K使用新binary，期间没有修改生产源码。新改动只调整
CPU输出publication顺序，GPU采样和数学未变；完整创作矩阵不冒称在新候选上重复运行。
完整指标、作品评价、采样和来源见[结构化收据](../performance/2026-10-23-non-nvfp4-closeout.json)。

原始命令、输出、失败与各阶段终态保留于 `tmp-nvcc/opsx-non-nvfp4-closeout-20261023/`；
Pi原始输出与渲染独立保存于前文目录。没有重复旧256K/P4/300W矩阵，没有修改默认推理
配置或下载NVFP4权重。真实历史token失配仍安全重建，AR作品缺陷及量化归因边界不隐去。
