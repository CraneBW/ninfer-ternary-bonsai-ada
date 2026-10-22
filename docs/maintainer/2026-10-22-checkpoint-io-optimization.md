# 2026-10-22：完整前缀快照校验与首TTFT优化

OpenSpec任务2.10，变更 `absorb-donor-techniques-and-optimize-sm89`。使用原有Qwen3.8 GSQ和213K/256K完整快照，日常测试200W，保留canonical/sync、MTP15及原问题。本轮只改快照IO，不改模型数学、上下文、选页、GDN状态、原K或快照状态ABI。

## 推进前的理论与实测

上一轮200W native213K：load独立中位26.443s、首TTFT49.962s、session wall164.583s。旧读取器先整文件FNV验证，再在消费时重复逐字节FNV。KVMem/ActKV冻结源码的snapshot.hpp采用同一串行hash，没有可直接复用的快校验实现，因此继续复用其有界codec，并复用本机已有xxHash源码实现快块验证。

对真实12,452,591,048B payload先跑CPU transport，2GiB内存/0swap，无GPU：验证11.014232s、消费10.446253s、合计21.460485s；CPU20.927s，说明主要是串行字节hash，不是GPU计算。

按上一轮整机时间，假设这21.46s完全消失，load理论上限约81%、首TTFT约43%、session约13%；这是上限，不是性能承诺。实现后CPU真实文件索引命中验证2.590954s、消费2.718610s，合计5.309565s（约4.04倍子路径速度）。Amdahl模型按f=21.46/49.96估算首TTFT约1.48倍速度上限，仍须整机验证，不能把CPU比例直接当GPU/会话结果。

首次建立213K索引的CPU样本13.967294s；256K首次建立18.948225s。原包只读，无额外长prefill；native测量前两索引已由完整FNV验证后生成。

## 复用与实现

- KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 的分块snapshot.hpp为现有codec基础，普通/memory SnapshotReader保持FNV默认。
- 本机 `/usr/include/xxhash.h` 的xxHash0.8.4头文件原样复制到 `third_party/xxhash/`，保留BSD-2-Clause与SHA说明；不装新系统包、不加动态依赖、不改CUDA kernel。
- PrefixCheckpointReader快速模式使用每1MiB的XXH3-128派生索引。首次必须整包FNV匹配原footer后才能原子发布 `.xxh3idx`；索引绑定长度、FNV、块长/版本并校验索引自身。
- 命中时仍先读整个payload、逐块比对，全部通过后才允许native状态恢复；消费时再次读每一块、校验后才提供字节。损坏payload在恢复前拒绝；校验后出现的修改在消费中拒绝。
- 缺失/陈旧/自身损坏的索引回退完整FNV后重建；只读目录可以使用当次验证生成的内存索引。mtime/文件存在不是放行条件。
- 快照文件及FNV footer不改；旧二进制仍能直接读同一包。新索引是可删除并重建的小文件，不能作为KV/模型状态使用。
- 内存固定1MiB缓冲和约0.2MiB索引；第二遍将碎片字段读取合并为大块，并避免重复串行FNV。实际RAM/磁盘吞吐仍单独约束。

## 最少正确性资格

`ninfer_prefix_checkpoint_io_test`为CPU-only：真实legacy文件与fragmented reads全字节比较；首次build/后续hit；损坏索引、截断索引、错误绑定；partial尾块损坏；无索引时坏FNV拒绝且不写索引；预验证后中途修改；footer/长度截断。CTEST仅该用例，0.05s通过。

受影响native Serve构建成功，8GiB/0swap、两worker限制下实际峰值2.6GiB，无GPU/数学变更。整模型仍需同binary/同原题/同功耗比较，数学证据沿用不变代码，未重复全算子或无变化4K/32K resident矩阵。

## 整机对照与采用

**已完成并默认采用indexed**。原213K三对每臂六题36/36，256K候选六题6/6，共42/42精确正确；完整输出相同，HTTP/terminal/usage完整、inflight页/字节与active请求清零，各服务退出0。

同binary200W成对结果（每格为legacy/indexed秒）：

|对次|load|首TTFT|六题session wall|
|---|---:|---:|---:|
|1|24.922/17.747|48.125/41.036|160.600/155.477|
|2|24.318/10.036|47.422/33.510|161.626/147.941|
|3|23.953/10.775|47.171/34.160|160.426/148.646|

|指标|legacy独立中位s|indexed独立中位s|成对用时减少的中位|
|---|---:|---:|---:|
|load|24.318|10.775|55.02%|
|首TTFT|47.422|34.160|27.58%|
|session wall|160.600|148.646|7.34%|
|后五题TTFT合计|112.802|113.974|-1.08%|

成对中位按每对减少比例后取中位，不以两臂独立中位的商替代。三对load/首TTFT/session wall均为正收益；后五题没有受益，不能将首轮优化说成持续decode加速。

六臂实际工作相同：restored212260、history replay185836、history reused10240、source-K rebuild3398。该收益没有减少原题或历史计算量；两份原快照与数学/state ABI1保持不变。256K恢复261395 tokens、load14.308s、首TTFT约37.3s；该档只做新模式质量，没有同功耗成对256K性能结论。

索引大小：213K 190080B，256K 233552B，约186/228KiB。每次仍验证所有payload块后才恢复，随后消费复验每块；不是仅凭小索引或mtime放行。

Native/CUDA fixture及CPU测试构建成功，实际峰值2.6GiB/0swap。三对Native cgroup峰值20GiB（含文件缓存）、swap407.2MiB，256K峰值20GiB/swap0；均在20GiB/2GiB限制内，无OOM，无KILL。退出恢复200W，300W本轮未运行。

按资格将native和CLI缺省值切到indexed；回退 `--checkpoint-io legacy` 或native环境 `NINFER_BENCH_CHECKPOINT_FAST_VERIFY=0`。默认提升只改变配置选择，并补临时索引清理；显式indexed实现、状态数学与布局不变。最终重新构建与完整CPU默认加载确认，复用已验GPU路径证据，没有重复长题矩阵。测量binary和最终安装binary的SHA分别保存在摘要中，不混称同一文件。

来源/理论/默认采用摘要 `tmp-nvcc/opsx-checkpoint-io/summary.json`；可随源码保存的秒数/工作量/二进制证据 [performance JSON](../performance/2026-10-22-checkpoint-io.json)。

## 边界

本项只减少新进程加载固定q0的等待；后五题canonical历史计算和后续decode不应借此认领加速。恢复token单独统计，不当作cold-prefill吞吐。新物理冷NVMe吞吐未验，OS页缓存与逻辑Cold层不混称。R21、逐层forward-ready和P4收益仍为独立未完成项；300W仅留发布最终验收。

后续ABI2默认可用后，复现本报告的纯I/O对照须显式 `--checkpoint-stage raw --no-canonical-query-reuse --no-checkpoint-load-pipeline`，仅切换 `--checkpoint-io legacy|indexed`，避免混入续问或首次重建消除收益。
