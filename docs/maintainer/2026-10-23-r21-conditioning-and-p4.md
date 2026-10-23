# R21选页条件与逐层P4续作

本轮使用现有Qwen3.8 GSQ v3、RTX 4070 Ti SUPER SM89/66SM、200W、INT8 KV、
32K工作集、36K物理池、chunk512。没有下载权重、修改冻结原题或扩大工作集。
NVFP4不启动；原位置仍为显式实验，默认OFF。

## 先算收益与成本

已验canonical热续问约0.4s，不能再次算作本轮收益。移除一次32K历史重建的旧实测上限
约22s，是R21最值得解决的成本。原位置资格拒绝后可能必须重算更长历史：213K完整prefill
约160s，因此“答对”和“降低TTFT”必须分别验收，不能将错误缓存命中当作快速基线。

旧快照加载路径的页transfer为56.642ms、session35.538s，其理想消除上限约0.16%。
这是该旧路径的上限，不能直接外推到本轮完整cold-prefill。新的P4固定Host128页、
Cold落在本地SSD目录，比较已有sync与source-event/async-D2H/读前/Main16逐层ready组合。
Main每页2,162,688B，Host128页264MiB，Cold三个槽6.1875MiB；Main immutable-K每槽
1.03125MiB，按实际分配槽数汇总staging，另报告MTP。没有无界pinned队列。
两页以上source/Cold batch、MTP与Graph decode继续原有同步回退。低于10%保持sync。

## 数值缺口：已修复，但不是R21首错的唯一原因

将Main24Q/4K、MTP16Q/2K、D256/R64在212275和261411位置与独立FP64 RoPE参考比较，
旧FP32位置乘频率存在超过原6.9e-3 BF16误差界的反例。共享text RoPE、mean-K逆旋转和
rephase现在在绝对位置>=65536时用double频率、乘积与range reduction，最后仍回到原
FP32系数/BF16输出/INT8编码边界。紧凑窗口<=36K保留旧算术。

新增256B只读频率常量，不增加KV或pinned。`NINFER_TEXT_ROPE_LONG_PRECISION=0`
是同binary旧数值参考；默认1。八个实际shape/native及自定义频率Graph用例通过，
既有KV rephase夹具通过，YaRN CPU用例通过。这是正确性修复，没有宣称kernel速度收益。

参考KVMem `faf3c225` Ninfer patch的固定YaRN参数，添加仅原位置显式
`NINFER_KVMEM_ORIGINAL_YARN_FACTOR=1..4`诊断入口；因子4没有修复已知错答，不推荐。
保留完整原始失败日志，不做更多因子扫描。

## 跨轮反例与修复

在相同binary、模型、功耗、输入及原位置模式下：

| 执行方式 | 首个Topaz问题 | 状态 |
|---|---|---|
| ACK后续问，长位置精度修复 | `copper-lilac-28` | FAIL |
| ACK后续问，固定YaRN4 | `copper-lilac-28` | FAIL |
| 新进程仅运行同一冻结第2条请求 | `silver-maple-41` | DIAGNOSTIC_PASS |

单题诊断使用原文件和SHA，不删除其中历史或改写问题；其结果不计六题验收。
输入自身包含不同实体的两个标签，Topaz早期事实与近期Quartz更新不是同一实体。

词法候选在完整prefill前安装，因而会影响随后历史token实际读取的注意力工作集，
也影响native KV和GDN。只比较文本前缀、位置和StateImage字节身份，不能证明由不同
词法条件生成的压缩历史可以复用。已有KVMem/ActKV的完整前缀联合恢复仍然必要，
但它们不替本地新增的词法选页政策提供这项证明。

新增`KvmemSelectionCondition`保存确切候选页及older-match页集合。在请求规划时只计算一次，
随实际执行状态保存。压缩原位置checkpoint的复用规划和mutation前preflight都要求条件相同；
条件不同拒绝该候选，允许仍合法的未压缩早期checkpoint或Root。
不清零/截短GDN，不改写native K，不缓存答案，不因缺资格继续使用旧混合状态。
这是补充必要条件，不宣称它取代已有owner/epoch/前缀/完整query边界校验。

### 原213K六题准确性通过

`original-conditioned213/report.json`为PASS：六个原始请求按原顺序运行，
`ACK / silver-maple-41 / amber-willow-92 / copper-lotus-35 / copper-lilac-28 /
teal-jasmine-61`均精确。6/6 HTTP200、终态与usage完整，Serve exit0，
Main/MTP层摘要pending/inflight/producer均0；全局inflight pages/bytes为0。
没有加载快照或转canonical。

六题均合法选择Root：实际分别计算212276/212303/212335/212368/212403/212441个prompt
token，prefix-cache hit均0；history replay=0、query replay=162。这里的history replay=0
不意味着没有计算历史，而是每题都从Root完整prefill，必须与主请求计算量一起解释。
服务TTFT依次158.922/159.449/159.790/159.011/158.514/159.842s。
首题API TTFT158.928s，后五题API TTFT158.521–159.848s。

该功能臂ready-to-complete wall1024.151s包括原runner的收据落盘和客户端间隙；
第4条HTTP wall193.363s中首内容后有34.349s间隙。它不是新的成对速度结论。
后续P4 runner将短响应收据延迟到计时结束后写出，单列session wall、HTTP wall和client gap，
不把落盘时间算作推理收益。

准确性收据binary大小402705904、mtime_ns1792699736339950412；随后P4构建仅新增
Cold读页/字节计数与退出摘要，无原位置数学或恢复变化。R21三对TTFT仍NOT_RUN，
OpenSpec2.6及旧16.4保持未完成；该慢路径不推荐，原位置默认OFF。

## 收据与入口

原始收据：`tmp-nvcc/opsx-r21-p4-20261023/`。

```bash
python3 -m tools.bench.local_long_quality --backend mtp --original \
  --draft-tokens 7 --case 213k --output /fresh/path/r21
python3 -m tools.bench.layer_ready_pairs --output /fresh/path/p4
```

P4不读取ABI1/ABI2快照；每臂新进程从完整原213K输入开始，保留已验canonical续问复用。
先做candidate六题资格，同一无密集trace的臂计入第一对，再按async/sync、sync/async、
async/sync完成三对。任何质量、实际consumer命中或排空失败即停止，不重复无关旧矩阵。
单条退出摘要记录层epoch/wait/release、source/Cold同步回退与staging，进程`/proc/io`
区分SSD物理读取和页缓存命中。SSD上的临时Cold文件不自动等于物理冷读。

### P4首臂资格（尚非三对性能结论）

`p4/pair-0-async`原六题6/6、exit0，351个Main层producer epoch、5664次consumer wait及
5664次最后读者release、154491个resident重相位页；退出两tier pending/inflight/producer均0。
source-K和Cold按层async页数均0，因为本长题batch超过双槽支持域；17062个source页仍经
已有whole-batch路径在forward前等待。不能把此臂称作长题所有H2D均已逐层重叠。

Main/MTP各238次真实Cold逻辑读，合计546889728B；已有两槽NVMe读前提交/完成476，
async demote提交/完成1052、split batch350。Main staging峰值8650752B，MTP540672B，
合计8.765625MiB。进程物理read_bytes3538976768B同时涵盖Cold、immutable-K及其他文件，
不能单独归给Cold；逻辑rchar20149899304B也不是SSD物理流量。

首臂session184.575s、首API TTFT182.118s、后五题0.392–0.409s，client gap仅0.038s；
history replay32228、1轮。它证明完整prefill原题的实际consumer路径命中；三对结果另行汇总。

### P4三对完成：正确性通过，性能不采用

同一binary、200W、完整原213K、MTP15/Graph、canonical续问复用、Host128页，
按冻结顺序完成三对，共36/36精确，18/18成对请求的全文与usage一致。
6个Serve全部exit0、running requests和inflight归零，退出producer/pinned均排空。

| 对次 | sync session（s） | async session（s） | 成对耗时减少 |
|---|---:|---:|---:|
| 1 | 186.435 | 184.575 | 0.998% |
| 2 | 186.543 | 186.445 | 0.053% |
| 3 | 187.912 | 186.779 | 0.603% |

**成对减少中位0.603%，未达10%，保留sync默认。** 这是每对百分比的中位数，
不等于两种模式各自session中位数的比值。首题API TTFT两模式中位分别184.013/183.986s，
后五题合并中位0.39975/0.39925s；本轮没有后续TTFT的实质改善。
所有臂history replay32228/1轮，client gap0.026–0.038s，staging峰值均9191424B。

三个候选均351个真实层epoch、5664次wait/release。Cold逻辑读合计542294016–546889728B；
全进程实际物理读取1223057408–7047245824B，受页缓存及其他文件访问影响。
source恢复/Cold计数在各臂有一两页差异，已逐臂公开，不作强行归一或组件独立归因。
没有全局清缓存、额外物理冷读矩阵、256K/媒体/多并发/300W验收，也不外推这些范围。

详细数值：`docs/performance/2026-10-23-layer-ready-p4.json`；
原始完整收据：`tmp-nvcc/opsx-r21-p4-20261023/p4/{manifest,results}.json`及六个arm目录。
新OpenSpec3.5按本支持域测量完成；2.6原位置三对TTFT仍未完成，NVFP4仍不启动。

本轮未提交、推送或打tag；外部q4修改保留。最终默认是否改变只能依据完整性能门。


2026-10-23后续进展：已新增query-independent历史内容锚点候选，原213K六题6/6；同binary/200W的ACK→Topaz三对续问TTFT由约160.4s降到0.667s，中位缩短99.5845%。冷首仍约160s，并非首次prefill已消除；原位置仍默认OFF，测量profile MTP7。旧文中的“逐题Root、三对未完成”是当时状态，当前收据及新大批source/Cold候选见[安全复用续作](2026-10-23-r21-safe-reuse-and-bulk-cold.md)。旧P4 0.603%结论仍有效，不用新R21结果替换它。
