# Canonical-ready快照：省掉重启后的首次历史重建

2026-10-22；已有Qwen3.8 GSQ，RTX 4070 Ti SUPER / SM89，200W。

## Scope and theory

上一轮已使213K后五题TTFT约0.4s，但ABI1初次加载后仍重建32228个历史token。首TTFT中位33.544s、load10.667s；移除剩余约22s计算是本轮上限，加载自身仍须付费。该路径是固定原题基准恢复，不是cold-prefill、任意新问题的前缀缓存或resident算子提速。

复用KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 的有界snapshot codec、本机已有全tier恢复和ActKV精确边界/状态恢复；参考本地 `ecosystem/ninfer-4080-32g/src/targets/qwen3_6/impl/runtime/session_snapshot_impl.h` 的完整StateImage及Host checkpoint配对。保留原32K canonical数学、现有查询续问门和所有旧ABI1包，不使用另一套推理runtime。

## Implementation

新增独立 `canonical-ready.nkv` / ABI2。在固定首题的query probe、选页及完整canonical重建完成后、计算最终问题后缀和生成答案之前保存：

- 全tier Main/MTP native K/V/scale、首次表示K/scale/birth与有效列、mean-K、LRFU时钟及页目录；沿用既有codec。
- 完整GDN recurrent、conv与continuation hidden；同一个q0的offset/frontiers与Main/MTP有效边界。
- 已完成probe的冻结Q sums，绑定Q形状；action/lexical元数据从精确匹配的完整首题重建。
- canonical positions/tokens、完整执行资格和四个有效Host回放检查点。句柄及StateImage generations在新进程重建，旧指针不落盘。

恢复绑定模型SHA256、精确完整首题tokens、容量/KV/chunk/draft/attention配置、canonical-query-reuse及canonical-prepare选择。先全文件校验，消费时再验1MiB块，拒绝形状/选页/边界不匹配；完整读取后再发布StateImage资格。恢复到已完成canonical的q0，跳过probe及历史回放，仍实际计算问题后缀和生成每个答案。

保存时四个Host槽均为活跃检查点，不能把槽0当作q0暂存覆盖。本实现增加一次仅限benchmark的153954304B pinned暂存，结束即释放；Device新增0，不复制整份12–15GB包进RAM。四个检查点最多增加615817216B文件载荷，但重建后的部分首次K记录已由既有生命周期撤销，净文件变化另以实际大小报告。生成前预检完整目标包、最大5GB不可变K临时文件和2GiB空闲地板；保留原包，不清理用户数据。

`prepare`加载已有ABI1，再做一次约32K重建并保存，避免重新prefill213K/256K。首次加载ABI2建立派生校验索引的成本单列，不混入已有索引的成对计时。恢复token仍单独计数；因新进程没有执行probe，问题后缀只计算并计数一次，不能沿用旧probe的hidden-replay扣减。

## Results

213K先完成派生保存6/6、新进程恢复6/6，再运行同binary200W三对，共36/36精确正确。参考与候选都开启上一轮已资格的canonical-query-reuse、indexed、C1/MTP15/INT8、physical36864/budget32768/reserve512、sync；R21及canonical-prepare关闭。未改变原题或期望答案。

| 指标（各臂独立中位） | 原ABI1+续问复用 | ABI2 canonical-ready |
|---|---:|---:|
| 首请求TTFT | 32.988s | 10.351s |
| 六题client session wall | 35.538s | 13.111s |
| load | 10.309s | 9.978s |
| 后五题TTFT合计 | 2.028s | 2.308s |
| 后续单题TTFT，15个样本中位 | 0.409s | 0.413s |
| 第一次续问TTFT | 0.394s | 0.681s |
| history replay / rounds | 32228 / 1 | 0 / 0 |
| query replay | 16 | 0 |
| source-K rebuild calls | 522 | 0 |
| history reused（后五题累计） | 161544 | 161544 |
| restored / computed prompt tokens | 212260 / 229 | 212260 / 229 |

三对首TTFT分别减少67.63% /68.62% /69.28%，中位68.62%；session减少62.22% /63.11% /63.72%，中位63.11%。load成对减少中位4.57%，不是本批主要收益。后五题合计反而增加中位13.25%，主要集中在第一次续问多约0.28s；剩余四题与单题中位接近原值，保留该小回退，不认领后续TTFT改善。没有把历史两轮优化百分比相加。

213K新包12,497,055,046B，原包12,452,591,064B保持，净增44,463,982B；生成中的save15.906s。第一次加载ABI2创建索引用11.561s、总load22.745s，属于一次性初始化；成对样本均记录index=hit。原数据、逐请求TTFT、完整响应及metrics在 `tmp-nvcc/opsx-canonical-ready/`，结构化报告随全部资格完成收录。

构建47.936s、2.5GiB/0swap；213K生成60.614s、20GiB/4.2MiB swap，新进程资格34.315s、20GiB/998.1MiB；三对186.136s、20GiB/11MiB。内存数字含文件缓存，均在20GiB/2GiB上限内，无OOM，server均正常退出0，成功终态active/inflight pages/bytes为0。原snapshot codec的损坏/截断/陈旧索引CPU资格继承，未改CUDA数学，不重复resident算子测试或NCU。

256K从已有ABI1派生保存6/6，并用工具新auto默认在新进程恢复6/6，五次直接续问，history/query replay和source-K rebuild均0。新包15,346,247,290B；恢复261395 tokens，computed229。生成61.307s、save14.349s、峰值17.7GiB/0swap；首次新包恢复41.315s、load29.283s（含13.403s索引创建）、峰值20GiB/1.9GiB swap，均正常退出且inflight归零。该档仅功能资格，未做成对性能，不能套用213K的68.62%或63.11%。

测量binary SHA256 `208ad1121dc3a55f3c03b6bcae4cc85fdeba92f63fea9d4cc9810030901aaa2e`；计时后未改native代码。工具auto选择在256K实际默认入口验证通过。采用为固定基准可用ABI2的默认恢复路径，原ABI1显式回退保留。[结构化报告](../performance/2026-10-22-canonical-ready-checkpoint.json)。全程200W，300W未运行，退出已核实200W。

## Usage

在源码仓执行；收据目录须为新目录。生成每档一次，不覆盖现有包：

```sh
python3 -m tools.bench.prefix_checkpoint prepare --case 213k --output <生成收据目录>
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --case 213k --checkpoint-stage canonical --output <恢复收据目录>
python3 -m tools.bench.prefix_checkpoint prepare --case 256k --output <生成收据目录>
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --case 256k --checkpoint-stage canonical --output <恢复收据目录>
```

日常 `restore`默认auto：有可用ABI2且启用query reuse时优先选择，否则加载原ABI1；显式 `--checkpoint-stage canonical`要求ABI2。重现本报告两臂均加 `--no-checkpoint-load-pipeline`；对照使用 `restore --checkpoint-stage raw`，两臂均保留已资格的canonical-query-reuse。若重现更早的逐题重建对照，另加 `--no-canonical-query-reuse`，不能用关闭续问的旧参考去认领本批单独收益。上述命令串行运行，建议20GiB RAM /2GiB swap、`SendSIGKILL=no`；工具只持有自身Serve并TERM/drain。日常200W；300W仅发布前最终验收，退出恢复200W。

逐层页流水/P4及canonical-prepare对照仍须显式 `--checkpoint-stage raw`：ABI2已没有首次history/source-K工作，不能拿它测出原页流水的收益。选用不同pipeline/prepare配置时须生成各自匹配的包或选raw，不静默套用错误状态。

## Boundaries

- 保存固定首题之前的完整计算状态，没有缓存生成答案；不同首题或数学/布局不兼容不能复用。改变prefill数学仍须重做cold资格；旧状态不能证明新kernel数学正确。
- 没有flush OS page cache，不认领物理NVMe冷读带宽；计时不含模型加载和服务启动，也不含一次性生成派生包。
- R21原位置仍关闭；真正逐层layer-ready/released流水、Async P4和300W发布终测仍未完成。
- 未改变resident prefill/decode算子，不新增该范围tok/s结论；本轮无发布、push或tag。
