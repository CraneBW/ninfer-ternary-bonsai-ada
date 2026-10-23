# 宽批SiLU量化融合与2048筛选（2026-10-23）

本轮继续旧OpenSpec 15.4的prefill性能任务，硬件为RTX4070 Ti SUPER/SM89/66SM/16GiB，已有Qwen3.8 GSQ v3，日常200W。先完成外层2048批次筛选，再适配当前1024主线遗漏的SiLU→MMQ激活量化融合；两个候选没有合成一个性能数字。NVFP4未做。

## 来源、理论和批次选择

复用本地ActKV `182113cdbc6bbdd9ef300a84289075e761a1c075` 的 `docs/prefill-harvest-optimization.md` Stage2大ubatch思路；其Stage1写线程/`write_spans`需要对应实际NVMe等待证据，本轮未直接移植。DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的分批预算概念也只用于容量判断，没有复制SM100 tile、Python worker或其异机数字。

当前MTP7/32K [Nsight归因](../performance/2026-10-23-mtp7-attribution.json)中GGUF产品占prefill kernel和81.807%，attention10.576%，GDN3.418%，other4.096%。该probe受观察者影响；21702.235ms host范围减21612.098ms kernel和不能当作准确CPU关键路径。仅扩大外层批次而保留内部MMQ1024，不减少矩阵计算本身；不能借donor大ubatch宣称本机多倍提速。

第二候选复用本仓已资格R29 SiLU→MMQ量化融合：GGML量化协议和本机BF16 producer，来源与许可仍见原R29/v1.0记录及第三方GGML许可。源码确认原入口仅支持T512，在chunk1024/R29=0主线未启用。现仅扩展既有kernel的column/grid到1024，保持所有CUDA算术、BF16-RN→amax/scale/code/sum→MMQ→BF16投影→残差的边界；没有重写量化或改变权重。

每层/1024 tokens可省17408×1024×4 = 71,303,168 bytes的中间activation写读；64层×32chunks为146.03GB（136GiB）逻辑访存，以及2048次独立SiLU launch。即使按672GB/s全部来自DRAM，所对应传输也约0.217s；实际L2、额外融合expf/寄存器成本都要扣除，不能将逻辑流量等同实测DRAM或直接乘成TTFT收益。小幅稳定收益同样可采用。

## 外层chunk2048筛选：未采用

同一已验binary、MTP7/Graph/INT8、物理KV36864、内部MMQ cap1024，只有外层chunk1024→2048。一对pp4096/tg128与pp32768/tg128、warmup1/r1、完整prime与输出捕获。结果：

| 输入 | reference prefill | candidate prefill | 变化 | reference decode | candidate decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1702.93 | 1655.32 | −2.796% | 180.13 | 175.43 |
| 32768 | 1511.39 | 1504.91 | −0.429% | 162.63 | 160.73 |

输出与实际MTP接受工作量相同，两个进程exit0；workspace209,484,800→371,598,336 bytes，候选启动可用2,118,254,592 bytes。该单对只是无收益筛选，受热状态/漂移影响，不能声称已证明稳定退步；按事先门停止，未扩三对或原213K/256K。完整记录见[2048筛选](../performance/2026-10-23-prefill2048-screen.json)。

## SiLU融合入口及资格

`NINFER_GGUF_SILU_MMQ=1`为启动固定的新opt-in，默认0；支持main Individual MLP的N5120/K17408、T512/1024和既有IQ3_S、IQ3_XXS、IQ4_XS、IQ2_XS、IQ2_S、Q2_K、Q4_K七格式。T1024要求内部MMQ cap足够，即已有`NINFER_GGUF_LARGE_PREFILL=1`；否则在enqueue前走原路。原R29仅T512的选择语义不变；MTP、其他shape、gather或未支持格式不新增融合。

保留原activation arena区间，因此没有新增workspace、持久state、KV或pinned空间，Graph地址和planner合同保持。新增target开关、共同eligibility helper、private dispatcher、bridge合法域及真实width trace统一构建；未扩大内部MMQ cap。新CUDA算术为零，复用原kernel的列独立索引。

`ninfer-serve`、`ninfer_bench`和新`ninfer_gguf_silu_down_test`构建exit0，8GiB/0swap限额实际峰值4.3G。原R29真实artifact七格式fixture纳入正式tests；只补T1024，不重复原T512资格。14次独立常量参考产品和14次random完整捕获共28个Op进程全部exit0，完整quant codes/scales/sums及guard、projected/residual BF16共574,622,720 bytes逐位一致；新增投影输出前后guard通过。random对照属于完整字节等价证据，不冒充独立随机FP64 matmul。

常规CTest无artifact参数时返回77并明确标为SKIP；实际28个Op资格由显式模型、独立参考与捕获参数运行。仅测试入口随后重建通过，生产binary未改变，未重复长题。

真实Engine pp1024/tg1探针exit0，观察到type10/12/17/18/21/22/23全部T1024 fused量化发射；仅证明路由，不用于性能结论。原始资格与探针分别在`tmp-nvcc/opsx-wide-silu-20261023/{qualification,route}/`。

## 整体性能与原题

同binary/200W、MTP7/Graph/INT8/chunk1024、内部wide=1，reference/candidate、candidate/reference、reference/candidate三对完成；只有新融合flag切换，所有进程exit0。完整Graph-prime17、warmup和正式129个输出token逐位一致，实际MTP rounds/drafted/accepted及各位置接受量一致，所有workspace capacity仍209,484,800 bytes。

| 指标 | 4K | 32K |
| --- | ---: | ---: |
| prefill参考独立中位 | 1645.81 tok/s | 1502.54 tok/s |
| prefill候选独立中位 | 1663.47 tok/s | 1512.14 tok/s |
| prefill成对提升中位 | +0.6924% | +0.6862% |
| 整段耗时成对减少中位 | +0.5076% | +0.6627% |
| decode伴随变化中位 | −0.1237% | +0.3307% |

32K prefill三对为+0.0006%/+0.7324%/+0.6862%，第一对近似持平；4K为−2.2446%/+1.3387%/+0.6924%，首对负值完整保留。按用户“小提升同样保留”的规则，作为显式候选采用，保持全局默认0；不认领普遍/强统计保证或新增decode收益，也不与之前chunk512→1024的百分比相加。绝对值的独立中位与成对比值中位不互相代算。

新融合flag1从零运行原213K与256K，各6/6精确、HTTP200/terminal/usage完整、server exit0；没有加载快照、改写问题或增大32K工作集/36K物理池。两档history replay0、query replay162、终态active/inflight0，所有layer摘要pending pinned/inflight/producer0。此次长题只作质量资格，缺少同轮对照，不将单臂TTFT称为收益。

| 单臂质量观测 | 213K | 256K |
| --- | ---: | ---: |
| 首次TTFT | 156.158s | 189.402s |
| 六题session | 170.437s | 200.608s |
| Amber续问TTFT | 10.576s | 7.444s |

其余续问约0.67–0.96s；Amber仍有明显延迟，不能宣称所有热续问亚秒或冷首轮问题已根除。最终新增一对300W发布前验收，完整输出/接受工作一致、进程exit0，并实际恢复200W；一对仅是该功耗样本，不是300W三对稳定收益。与200W同一binary/source/model/语料，未进行新发布或推送。

[结构化收据](../performance/2026-10-23-wide-silu-prefill.json)保留Op完整比较、Engine各臂输出/工作量、原题逐题实际答案和资源、300W两臂及恢复值。原始目录为`tmp-nvcc/opsx-wide-silu-20261023/`。

## 使用与下一瓶颈

在已验chunk1024/MTP7配置中显式加`NINFER_GGUF_SILU_MMQ=1`；关闭使用0。正常短tail、MTP decode及不满足格式/几何/cap的情况沿原路。工具复现：

```sh
python3 -m tools.bench.sm89_attention_gdn_pairs --candidate silu --output <新目录>
python3 -m tools.bench.local_long_quality --backend mtp --draft-tokens 7 \
  --original --original-causal-prefill --original-content-anchors --case 213k \
  --prefill-chunk 1024 --large-prefill 1 --silu-mmq 1 --output <213K新目录>
python3 -m tools.bench.local_long_quality --backend mtp --draft-tokens 7 \
  --original --original-causal-prefill --original-content-anchors --case 256k \
  --passed-213k <213K新目录> --prefill-chunk 1024 --large-prefill 1 \
  --silu-mmq 1 --output <256K新目录>
```

这轮已完成预算、采用/拒绝、数学/资源边界、整体性能和原题收据，不声称已经达到设备最优。下一瓶颈仍是占比约81.8%的GGUF矩阵主体，其余成本包括长窗口attention和NVMe写回；当前证据不足以把写线程或DeepInfer owned-step直接移入主线。继续研究需要新的具体收益假设，不重跑本轮已确认的候选矩阵。
