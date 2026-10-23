# Qwen3.8 GSQ：投机、普通AR与并发（2026-10-23）

设备为RTX 4070 Ti SUPER（SM89/66 SM/16GiB），全部日常测量200W。模型为现有
`qwen3_8_gsq_rco_iq3_s_v3.ninfer`，包含IQ2_S/IQ3_S等混合GGUF块，使用原生C++/CUDA。
未下载模型，未运行NVFP4，未改变MTP15 + sync维护档及原位置OFF默认。

## 投机后端公平对照

固定C1、INT8 KV、Graph、chunk512、36,864物理/逻辑容量、greedy/no-thinking、
输出预算128；三类固定约4.6K输入、每档三次交替，均关闭前缀复用。
36个请求全部HTTP200、usage/单一ID/单一finish/[DONE]完整。
这些是冷prefill服务请求，不是快照加载；没有用短题替代长题资格。

下表为同任务三次中位decode tok/s；时间已包含实际接受前缀、修复和提交成本。

| 固定任务 | 普通AR | MTP4 | MTP15 | DFlash2 K3 |
|---|---:|---:|---:|---:|
| 对话 | 44.58 | 80.03 | 45.71 | 92.15 |
| 工具/代码 | 44.55 | 105.90 | 71.94 | 111.18 |
| 确定性记录 | 44.46 | 142.33 | 217.15 | 134.31 |

会话中位耗时（秒）分别为：对话5.749/4.502/5.707/4.319，
代码5.783/4.151/4.733/4.118，记录3.567/3.155/3.098/3.182。
三类TTFT约2.90–2.98s，投机主要改善decode，并不消除prefill。
记录题实际输出29tokens，其余128；相同任务各档输出长度相同。
MTP4与DFlash2各9/9完整文本与AR相同，MTP15为6/9相同；其余三次仍满足固定前缀和协议，
不能宣称所有后端逐字或bitwise相同。

| 实测启动预算 | 普通AR | MTP4 | MTP15 | DFlash2 K3 |
|---|---:|---:|---:|---:|
| 权重B | 11755300864 | 12481729536 | 12481729536 | 14360105472 |
| runtime B | 1526404352 | 1689359616 | 1714714880 | 2094188544 |
| KV B | 1245708288 | 1323700224 | 1323700224 | 1245708288 |
| Graph allowance B | 12582912 | 85983232 | 85983232 | 503316480 |
| 启动后空闲MiB | 3078.1 | 2298.1 | 2220.1 | 446.1 |

各档workspace均106914816B。DFlash2共同可行短档满足256MiB地板，但余量远小于MTP，
不能外推多并发或长档。这里实际Graph allowance不能套用别人的2.5–3GB图池结论。
普通AR不加载草稿权重，speculative_rounds/drafted/accepted全部0。

用实际整轮耗时估交叉点，而非未测的独立draft/verify kernel时间：
对话AR约22.41ms/token；MTP4每轮33.52ms，需产出>1.50tokens/轮，实际2.70；
MTP15每轮63.12ms，需>2.82，实际2.89；DFlash2每轮29.88ms，需>1.33，实际2.76。
细项计数、每题资源和输出在`docs/performance/2026-10-23-sm89-profile-budget.json`
与`tmp-nvcc/opsx15-20261023/spec-final/`。不将整轮代价伪装成独立草稿/验证计时。

建议：内存紧张或工具类请求可显式选择MTP4；有容量的已验C1短请求可选DFlash2 K3；
确定性高接受任务保留MTP15。没有通用赢家，默认不变。

## 普通AR相容批次

复用本仓`gguf_linear.cpp::gguf_project`的共享输入量化、已vendored GGML vector_product
以及现有GDN合并QKV布局，适配到实际编译的`gguf_dispatch.cpp`与27B variant。
四类合批为GDN QKV/Z、attention Q/gate/K/V、MLP gate/up以及严格canonical续接。
前3类由`NINFER_SM89_SHARED_VECTOR=1`控制，适用BF16输入、T=1..8、同K/同gather身份、2–4个投影；
任何产品不符合条件，整组在入队前回到原实现。没有抄异机tile或改变GDN FP32递推/BF16边界。

理论上限先按每token计算：48 GDN、16 full-attention、64 MLP，最多减少256次重复输入量化、
144次中间复制、96次投影launch；复制payload共983040B。权重字节与主乘加不变，
不能按“量化次数减少”认领相同比例的整模型提升。

实际shape独立NumPy/FP64 full-row oracle包含7种GDN QKV与6种MLP格式，T=1/3/8共39例；
两输出彼此、与旧生产实现逐字节相同，独立数值参考与输出canary均通过。
收据`tmp-nvcc/opsx15-20261023/shared-vector-oracle.{json,log}`。

普通AR同binary A、相同固定token ID、warmup1、tg128、Graph prime、200W，4K/32K各三对：

| 档位 | decode基线→候选 tok/s | 成对中位decode | prefill变化 | 总耗时减少 |
|---|---:|---:|---:|---:|
| 4K | 44.80→45.95 | +2.576% | −0.182% | 1.282% |
| 32K | 41.08→42.06 | +2.377% | −0.272% | 0.049% |

三对decode均提升，32K整体基本持平且单对有退步，因此共享量化为decode偏重的显式候选，
不作为prefill加速或全局默认。原始计时在`tmp-nvcc/opsx15-20261023/ar-pairs/summary.json`。
native bench保存完整token计数而未保存逐token IDs，Serve与长题另保留全文/usage。

普通AR的canonical续接还修复了完成阶段的重复选页：仅在原位置OFF、C1、显式query-reuse、
同owner、完整canonical谱系、几何前缀一致且窗口在预算内时保留视图；下一题仍须通过
候选页真实驻留、精确token/position和状态身份门。未命中时保留原canonical重建。
短样例未命中的事实保留，不以删除检查制造命中。

当前binary B普通AR + 共享量化 + canonical续接的真实cold-prefill资格：
213K和256K冻结原六题各6/6，完整SSE/usage/终态通过，Serve exit0、inflight pages/bytes为0。
213K实际输入212276–212441tokens；256K实际输入261411–261576tokens；
逻辑上限分别212992/262144、工作集32768、物理KV36864，并不等于整段KV同时驻留。
没有加载MTP或其他backend快照。

213K历史重放185836→32228、轮次6→1、复用10240→161544，后五题TTFT为359–382ms。
旧/新臂总wall306.749/189.388s，只是两次独立功能臂，构建与运行时刻不同，
不据此认领成对整体加速百分比。256K完整wall229.823s，后续题TTFT约0.4s。
首轮仍需要真实长prefill；不把热续接速度算作cold-prefill。
收据`tmp-nvcc/opsx15-20261023/{ar213-final,ar256-final}/`。

DFlash2长213K在planner阶段拒绝：runtime需要2290136832B，而实际可用2155097600B，
缺135039232B（128.78MiB），尚未加载权重、未prefill。
因此其原213K、256K质量均NOT_RUN，不能继承MTP或AR的长题通过。
这与无prefix-reuse的短档启动后446MiB余量并不矛盾：长档需要额外状态/元数据。
保留`tmp-nvcc/opsx15-20261023/df213/`的原始启动失败日志及容量分类，未重试或用UVM绕过。

binary A完成共享数学与三对计时；binary B新增普通AR完成阶段视图保留和显式诊断。
B不改变无KVMem性能臂的算子/调度数学；长题与生命周期收据明确记录B，
不把两次构建混称一个最终整包发布验收。

## 可复现命令与回退

从仓库根运行，模型路径为现有本地artifact；测量工具检查200W并仅对自有Serve发送TERM，
超时保留PID，不自动KILL。

```bash
python3 tools/bench/sm89_profiles.py spec \
  --artifact /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
  --output tmp-nvcc/local-spec-pairs
python3 -m tools.bench.shared_vector_pairs \
  --artifact /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
  --output tmp-nvcc/local-ar-pairs
```

单独Serve选择（沿用完整的已验环境PROFILE及本地监听配置）：
普通AR不传`--spec`；MTP为`--spec mtp --draft-tokens 4|15 --lm-head-draft`；
DFlash2为`--spec dflash2 --draft-tokens 3 --lm-head-draft`。
开启共享输入量化使用`NINFER_SM89_SHARED_VECTOR=1`，回退设0。
长canonical显式启用`NINFER_KVMEM_CANONICAL_QUERY_REUSE=1`，回退设0；
原位置保持`NINFER_KVMEM_ORIGINAL_POSITIONS=0`。
短比较使用`--kv-capacity`与`--no-prefix-reuse`；长KVMem使用`--kv-device-tokens`及前缀复用，
不能混搭这两类CLI参数。长质量不读取其他backend的MTP快照。

## 并发来源与本机适配边界

2026-10-23实际fetch `deepseek-ai/DeepInfer` HEAD到本地，固定为
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`；
worker/worker.py、worker/zos_status.py、csrc/engine/scheduler.hpp与mtp_varlen.hpp
相对前轮固定`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`无差分，许可Apache-2.0。
源码保留在`/home/ran/ninfer-work/ecosystem/deepinfer`，没有checkout覆盖已有固定源。

| 来源实际机制 | 本机处理 |
|---|---|
| DeepInfer WorkerStepState保活metadata、sample_end_evt | 对照本机PendingCandidate/Engine提交；本轮继续保留强owner/代际与完整提交后发布，不复制Python/Torch对象图 |
| 当前GPU forward先提交、再post-process上一轮 | C1 AR host约9μs/22ms，空间不足，不引入跨轮owned-step；并发另报告实际host暴露占比 |
| varlen维持processed≤scheduled≤processed+1，以N−2轮选batch预算 | 继承前轮本机MTP controller cohort/epoch修复及负性能结果；不复制DeepSeek confidence head阈值 |
| DeepInfer LCFS及active preemption | 不搬入本项目固定C≤8、有界FIFO、无活动请求抢占合同 |
| 5060Ti项目T1/T8分档、实际活跃lane成员、merged prefill | 固定`104af690541857a5d52f23ffd81ccac7eb82a675`，Apache-2.0；看过真实ternary_dispatch及advance_prefill_batch，而非只看README；保留本机精确B Graph/compact decode，给实际GSQ/GGML向量投影适配T=1..8共享量化 |
| 5060Ti ternary tile、FP8 GDN、C16–62与大并发投机宽度表 | 权重布局/状态数学/容量不同，不移植异机tile、FP8近似状态或高并发数字；其merged-prefill接口也未声称本机已经接入 |
| Strata/KVMem/ActKV | 继续用已适配的先验证后mutation、ready/released、owner/epoch与页/StateImage联合生命周期；未宣称复制了整个调度器 |

并发新候选是同一batch内GDN/attention/MLP的共享输入量化与合并QKV，
通过现有compact batch按真实T选择，不给inactive lane填充、不分配额外持久workspace。
单卡现有权重复用来自同一向量产品batch及其缓存行为；不将共享量化本身说成消除了全部重复权重流量。
是否采用由本机成对会话决定，不以“看过源码”代替功能/性能通过。

## 普通AR热点与理论边界

仅补一次普通AR NSYS（无投机、共享量化0、Graph node trace），按137个
`decode.ordinary_round` NVTX范围统计完全落在范围内的GPU kernel：

| 类别 | GPU kernel累计时间占比 |
|---|---:|
| GGUF vector products | 90.194% |
| GDN状态 | 2.635% |
| 激活Q8量化 | 2.605% |
| attention | 2.330% |
| 其他 | 2.237% |

这是GPU kernel时间之和的占比，包含短warmup，不是整请求wall或host空隙。
**profile下的42.28tok/s不用于性能对照**，性能采用未挂探针的三对。
完整名称和范围在`docs/performance/2026-10-23-ar-kernel-attribution.json`，原trace另保留。
实际每轮513次量化；本批最多去掉256次，对量化部分的kernel时间理想上限约省1.30%，
还存在合并96次投影launch和144次复制的空间；数学/权重工作量并未同时减半。
只把attention或GDN状态本身完全消除，kernel时间上限也分别只有2.33%/2.64%，
不能将MTP15的热点比例套到普通AR。
按静态main唯一payload扣除整表embedding、假设每轮线性权重读取一次，约11.348GB/token；
44.55tok/s对应约506GB/s的名义权重流量。这不是NCU测得的DRAM字节，也不保证每个权重只读一次，
用于说明继续优化批内权重复用与GSQ向量乘比增加CPU队列更有空间。

实际S10周期窗口（含边界与warmup）的decode host暴露占比：C2/C4/C8共享候选分别
0.0534%/0.0663%/0.0929%。即完全消去这部分host，理论收益也不足0.1%；
因此DeepInfer的先提交后post-process仅保留来源/合同对照，本批不移植owned-step。
不以host探针步耗时等同设备时间；没有新的CUDA算术kernel或盲目更换SM120 tile。

容量先按INT8 KV33792B/token、GDN StateImage153954304B/slot及最多2slots/lane上界预算；
实际no-prefix-reuse启动只需1device state/lane，Host KV和Host缓存StateImage均0，
其余Graph/workspace/runtime和256MiB地板从各档真实启动收据核对。
每请求逻辑8K、物理共享池8K在C1/2/4/8完全相同，**不保证8个8K满长请求同时驻留**；
实际有界工作负载依据prompt/output claim准入，保留容量拒绝。
最初16K物理池/8K C1被planner拒绝，未加载权重；已固定所有档8K池，未隐性缩每请求上下文。

定向C4生命周期：ALPHA/BETA/GAMMA三个不同长度请求精确、第四请求读首内容后断开，
随后DELTA精确；超8K输入HTTP400/context_length_exceeded；最终active、Main/backend occupied pages、
Host occupied bytes和KVMem inflight为0，Serve exit0。
收据`tmp-nvcc/opsx15-20261023/concurrent-lifecycle/`。
这里只验证所改批次的正常完成/取消/拒绝/排空，不外推真实设备故障或多lane内部异常隔离。

## 并发三对整体结果

同binary B、200W、INT8、Graph、greedy/no-thinking、chunk512、无prefix reuse；
所有C均8个请求同时到达，混合三种任务/长度，输出上限128。
C1初版逐个到达的延迟数据排除，只补C1同时到达三对；C2–8未重复。
最终24臂、192请求全部完整协议/独立CASE前缀通过；96/96个成对完整文本相同，
每臂实际完成829tokens。每C/每policy为3波、24请求；排队顺序取决于实际FIFO到达，
不是固定每个case总在同一lane，尾延迟只作这个小样本的本地描述。

下表是**共享量化关闭的参考配置**：整批吞吐/会话为三波中位，
单请求decode及p50/p95对24请求汇总。整批速率包含prefill与排队。

| C | 整批tok/s | 单请求decode tok/s | TTFT p50/p95 s | 请求平均TBT p50/p95 ms | 启动后最低空闲MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 39.19 | 45.42 | 9.75 / 18.36 | 22.02 / 22.05 | 4002 |
| 2 | 66.59 | 43.91 | 4.94 / 9.56 | 22.78 / 22.88 | 3838 |
| 4 | 87.82 | 38.00 | 2.18 / 6.43 | 26.32 / 26.71 | 3510 |
| 8 | 113.25 | 31.17 | 1.82 / 3.29 | 32.08 / 35.22 | 2852 |

TBT是**每请求平均token时间**在请求之间的分位数，不是逐token尾延迟或HTTP帧间隔。
队列p50/p95、每对原值、每个case、资源和公平性边界均在
`docs/performance/2026-10-23-sm89-concurrency.json`及其引用的完整收据。
C8相对C1的约2.89倍整批吞吐来自并发profile/已有compact batch的利用，并非本批新kernel的独立加速；
单请求decode较慢。积压短请求优先评估C4/C8，轻载低单请求延迟保留C1，默认不自动调整。

共享量化0→1的**成对整批tok/s变化中位**：C1 +2.264%、C2 −3.292%、C4 +0.198%、C8 +0.099%。
C4/C8没有稳定可辨的整体收益，C2有负结果，因此并发推荐使用共享量化0；
C1 decode偏重场景可显式选择1。没有为了正结果重跑或追加无依据消融。
真实T8路径已命中GDN QKV/Z、attention四产品和MLP两产品，而不是配置开关空转。

runtime随C1/2/4/8为557518336/725555200/1061628928/1733776896B；
Graph allowance为12/24/48/96MiB，workspace均106914816B，StateImage槽为1/2/4/8。
Host KV与缓存StateImage为0；进程VmHWM和最终RSS分别保存，不把VmPin=0误当作CUDA无pinned内存。

```bash
python3 tools/bench/sm89_profiles.py concurrency --profiles ar \
  --max-context 8192 --kv-tokens 8192 --prefill-chunk 512 \
  --artifact /home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer \
  --output tmp-nvcc/local-concurrency-pairs
```

实际服务的短请求候选：普通AR、`--max-concurrency 4`或8、`--max-context 8192`、
`--kv-capacity 8192 --no-prefix-reuse --kv-dtype int8 --prefill-chunk 512`，
`NINFER_SM89_SHARED_VECTOR=0`，其余遵守已验PROFILE；长会话/热prefix配置须单独选择，
不能外推此无cache短域。回退并发1即可恢复参考低单请求decode延迟。
无SLA承诺、无C>8测试、无多并发DFlash2推荐。

原R21错误、逐层页流水P4、NVFP4和300W发布终测各按独立门槛保留，
本批短性能结果不关闭这些缺口。本轮没有提交、推送或tag。
