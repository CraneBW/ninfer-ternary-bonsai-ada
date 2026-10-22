# 完整快照加载流水：双槽、直接Host读取与私有首次K区间

2026-10-22；现有Qwen3.8 GSQ、RTX 4070 Ti SUPER /SM89，200W。

## Result

固定213K同binary三对36/36精确正确，load成对减少中位4.64%、首TTFT减少4.71%、六题会话减少3.47%。raw ABI1额外一次原六题通过，真实执行522次source-K恢复；256K新路径再通过一次原六题，共48/48。采用为固定工具213K默认；256K只有功能资格，显式opt-in。native实验环境变量全局仍0，普通sync与原R21关闭状态保持。

| 各臂独立中位 | 参考 | 新加载流水 |
|---|---:|---:|
| load | 10.886s | 10.380s |
| 首TTFT | 11.108s | 10.604s |
| 六题client session wall | 13.881s | 13.436s |
| 后五题TTFT合计 | 2.315s | 2.357s |
| 后续单题TTFT，15个样本中位 | 0.413s | 0.418s |

三对完整秒数保留：

| pair | load参考/候选 | session参考/候选 | session减少 |
|---|---:|---:|---:|
| 0 | 19.142 /11.406 | 22.428 /14.502 | 35.34% |
| 1 | 10.886 /10.380 | 13.881 /13.436 | 3.21% |
| 2 | 10.399 /10.027 | 13.520 /13.050 | 3.47% |

首对参考明显更慢，其整体额外成本尚未完全归因；不删除该样本，也不据此认领35%稳定收益。后两对会话各省约0.45s，后五题成对反而慢中位1.29%，不认领decode/后续TTFT提速。不与前两批百分比相加。[完整原值和收据](../performance/2026-10-22-checkpoint-load-pipeline.json)。

## Theory and source reuse

已有200W raw收据中，普通页transfer统计56.642ms，而会话35.538s；这项计数不包含所有source重建，不能当成GPU纯copy时间，也不足以证明逐层流水能省10%。首次canonical probe/选页/重建/forward的合计prefill仍约22s，不能靠一个copy开关消除。

转向本轮真实可删的载入工作：213K每次重新写约3.8GB首次K、Host页约7.65GB多一次bounce memcpy，Main/MTP各约504–512个resident页逐页等待，Main单页64个plane API。已有完整load约10s，其中约3s全文件预验证仍必需；优化上限受剩余load约束，不预认全部节省。COW新inode可能增加page-cache miss，作为实际成本一起计时，没有flush系统cache。

来源与小幅适配：

- KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 有界snapshot codec和完整验证；ActKV `182113cdbc6bbdd9ef300a84289075e761a1c075` 的 `kvmem/src/host/spill_io.cpp`，复用不可变extent、独立offset、owner/generation和退休原则。
- Strata `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4` `src/prefill/prefill.cpp:2832` 的released/ready协议，本机冻结件 `tmp-nvcc/perf-ada-round24/donors/strata-prefill.cpp`。这里只适配完整快照加载槽，不复制整个Strata runtime，不声称已接入层forward。
- CUDA13 batch-copy沿用本项目已有Infernix `a3edb450` /Don-Chad/ninfer-3090 `d8c29253f2c0845611a794d87057950107a4956b` 的stream-ordered source和独立range合同（Apache-2.0），不清错误后静默重放。
- 私有Btrfs reflink封装为本地实现；未在相关donor找到可原样移植的clone实现。不把它说成已复制某个上游完整缓存系统。

## Implementation and ownership

单个 `NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE=1` 合并四处相容改动，文件/数学ABI与原包保持：

1. 在原archive同一文件系统建立立即unlink的私有COW副本，再对**该副本**做完整预验证与逐块消费复验。首次K记录持有稳定FD/offset/长度，经原ColdKVStore handle/generation管理；Reader退出、原文件修改或删除均不能改变其数据。clone不支持时沿用原复制路径，不能静默引用可变原文件。
2. 后续capture或free-slot重用写入原内部文件；不覆盖另一条私有extent。read-ahead仍按既有owner约束保活FD及buffer。Private文件至最后owner退休才关闭。
3. 用现有Cold staging arena的两个槽交替读与H2D；事件表明最后DMA reader完成后才能复用slot。异常时先排空部分GPU提交，再销毁描述符和Host owner。全地址仍在全部页完成后publish。
4. Host页直接读入最终分配，避免bounce copy。兼容的PageMajor完整plane用既有CUDA13 batch-copy提交，最多64个独立range/页；descriptor由slot持有至完成，其余geometry沿用原copy函数。

没有新增Device或pinned容量。现有三页Cold arena中仅使用两页：Main共4,325,376B、MTP270,336B；每槽descriptor为有限64range普通Host metadata。私有副本逻辑上仍覆盖整个12–15GB文件，Btrfs共享原extents，不能把reflink逻辑大小、原source file-capacity计数当作物理新增空间或实际写盘字节。原始两档ABI1/ABI2文件均保留。

## Measured stages

新日志 `KVMEM_CHECKPOINT_PAGES` 每store一行，时间为**Host可见wall区间**，非纯GPU时间。213K三对的Main/MTP相加后独立中位：

| 阶段 | 参考 | 候选 |
|---|---:|---:|
| 首次K读取/恢复准备 | 1.299s | 1.010s |
| native页读取（候选包括写入最终Host页） | 2.551s | 2.535s |
| 额外Host allocation/copy | 0.730s | 0 |
| H2D提交 | 44.0ms | 29.4ms |
| 显式页完成等待 | 61.6ms | 1.31ms |
| 此页循环合计 | 4.660s | 3.586s |

完整load还包含预验证、arena/StateImage准备及读取、最终状态恢复等，不能仅相加这些阶段认领全部load差。两个store各504页走batch，共1008次；Host页每store2813，无native Cold页。source-K字节归档和后续raw恢复存在，并不因为native Cold=0就不存在NVMe来源。没有把热page cache读取称为物理NVMe冷读性能。

## Minimal qualification

只做相关项：

- CPU transport既有完整性用例，加私有clone在原文件修改/删除、Reader关闭、prefetch、代际/slot重用、私有数据损坏后的生命周期验证；本机private_clone=1。
- 既有native `--checkpoint` 一次：Device/Host/Cold/raw-K完整字节回环一致，代际与损坏拒绝仍通过。合计1.055s、峰值2GiB/0swap。CUDA数学未改，继承独立数值参考，不重跑resident算子矩阵、NSYS或NCU。
- raw213K六题一次：history32228、source rebuild522，load12.118s，正常退出。用于检查真的消费私有首次K，不充当性能样本。
- 两臂同binary、ABI2/indexed、query reuse开启、C1/MTP15/INT8、physical36864/budget32768/reserve512、sync、200W，三对交替；除本开关外一致。首对达到预先记录的load10%/session5%扩测条件后才补齐，后续未扩大矩阵。
- 256K显式新路径六题一次（最终选择策略将其保留为opt-in），load23.000s、index=hit，history/query replay/source rebuild为0，五次直接续问。没有成对256K速度结论，不从旧index-built样本计算收益。

测量binary `ee6d7e13d7c6bbaa32a6c6b9f362b6ea6ad8de6ab4757ebed332e2b767f8b194`。测量前构建47.658s、2.5GiB/0swap；raw资格45.093s、20GiB/256KiB swap；三对129.311s、20GiB/2GiB swap；256K34.127s、20GiB/1.9GiB swap。内存含文件缓存，受20GiB/2GiB约束，所有server正常退出0、HTTP/usage/终态完整，inflight pages/bytes和active requests归零。只TERM自有Serve，无KILL。GPU最后核实200W，300W未运行。

计时后收尾将prefetch的FD/offset在owner线程固定，后台worker不再读取可能增长的slots目录；并在CPU用例中覆盖其他记录增删/增长时的读取。该变更不改数据、数学或正常提交次序。交付binary SHA256 `b1354c4256fd2bd5e6c2857df615e826141a5964a055dd15b709238a59429504`，重新构建48.541s、2.6GiB/0swap；CPU与native相关小夹具再通过，1.050s、2GiB/0swap。按最小验证要求没有重复长题矩阵；上述48条长题与三对性能属于前述测量binary，两份身份在结构化报告分开记录。

## Usage and remaining work

```sh
# 固定213K默认开启；原包与canonical-ready都支持。
python3 -m tools.bench.prefix_checkpoint restore --case 213k --output <新目录>
# 本批参考，保持其余相同。
python3 -m tools.bench.prefix_checkpoint restore --case 213k --no-checkpoint-load-pipeline --output <新参考目录>
# 256K只有功能资格，显式选择；其默认仍用参考加载方式。
python3 -m tools.bench.prefix_checkpoint restore --case 256k --checkpoint-load-pipeline --output <新目录>
```

复现先前的ABI1/ABI2、FNV/indexed或canonical-prepare对照时，两臂都显式加 `--no-checkpoint-load-pipeline` 冻结旧加载策略。页流水/P4需要 `--checkpoint-stage raw` 以触发source/rebuild，不能用已省掉历史计算的ABI2测原流水收益。

OpenSpec 2.13完成；3.1的阶段调查得到本轮数据，但逐层source重建/forward之间的完整计时与消费者依赖还没全部落实，3.1–3.6仍保留未完成。下一步真正layer-ready须进入16个Main attention和1个MTP消费者，配对最后reader释放、Graph event与失败排空；现有归档仍PageMajor，不能直接删全局sync。R21原位置路线仍未通过。本轮3.47%是完整加载优化，**不是Async P4通过**。未发布、push或tag。
