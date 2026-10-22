# OpenSpec S3 / 4.4：MTP4 原长上下文质量门与快照资源边界（2026-10-22）

**最新状态：OpenSpec 4.4/4.5 均完成；MTP4不替换默认MTP15。** 原213K MTP4 save/独立restore各6/6、原256K MTP4主分区save/独立restore各6/6；额外256K Serve同盘三对36/36正确，正式4K/32K tg128各三对显示MTP4 decode回退。下文保留首次256K保存失败的真实原因与历史收据，再记录容量恢复后的成功续验。所有原生Serve质量/256K性能臂使用同一个 `ninfer-serve`（SHA256前缀 `53f0fc864254`），正式两档吞吐用同一个现有`ninfer_bench`（SHA256前缀 `92244e50b351`）；RTX 4070 Ti SUPER、200W、完整GSQ v3。未下载权重、修改数值算法、放宽snapshot兼容性或启用双卡。

## 固定测试规格和身份

| 维度 | 原213K | 原256K |
|---|---|---|
| 冻结输入 | `benchmarks/frozen-long-context/213k/manifest.json` 中六个原请求和SHA256 | `benchmarks/frozen-long-context/256k/manifest.json` 中六个原请求和SHA256 |
| 真实首请求 prompt | 212,276 tokens | 261,411 tokens |
| Context limit | 212,992 | 262,144 |
| 物理 INT8 KV | 36,864 tokens | 36,864 tokens |
| KVMem 工作集 / reserve | 32,768 / 512 tokens | 32,768 / 512 tokens |
| Speculative backend | MTP，**draft 4** | MTP，**draft 4** |
| 其他 | greedy、no-thinking、单并发、prefill chunk512、Host KV0、sync | 同左 |

两项均采用原题目、原 matcher 和原模型字节身份。DFlash2-K3 仅有短配置资源资格，不能将512-token档的约746MiB空闲直接扩至36K物理KV；长档**NOT_RUN**。

## 213K：真正匹配 MTP4 的 ABI1 快照，六题完整验证

**不能借用原 MTP15 快照。** `tmp-nvcc/opsx-s3-20261022/mtp4-213k-raw-quality/` 的第一份实验仅提交1条请求，在完整索引校验后明确报 `incompatible session snapshot`（HTTP500、无 terminal）。这是合法的安全拒绝，不是错误答案。原始MTP15包仍完整。

为了不跳过身份门，为 `tools/bench/prefix_checkpoint.py` 增加向后兼容的 `--draft-tokens 1..15`（默认15）：创建时按真实 draft-window 保存，读取时用匹配元数据预先拒绝不等width；不修改Serve校验器。选定 `--draft-tokens 4 --checkpoint-stage raw`，建立**新的独立**临时ABI1 checkpoint：

| 实际阶段 | 精确结果 | 原始收据 |
|---|---|---|
| 首次从冻结213K原请求计算、保存 | `SAVE_ORIGINAL6_PASS`，六条原答案全 exact、terminal、Serve退出0；archive 12,452,591,064 bytes，保存前缀212,260 tokens，保存事件 `complete=1` | `tmp-nvcc/opsx-s3-20261022/mtp4-213k-native-create/{checkpoint-receipt.json,results.json,213k-save.log}` |
| **新Serve进程**重新加载该专属 ABI1 包 | `RESTORE_ORIGINAL6_PASS`，六条原答案全 exact、terminal、Serve退出0；实际恢复212,260 tokens，加载事件 `complete=1` | `tmp-nvcc/opsx-s3-20261022/mtp4-213k-native-restore/{checkpoint-receipt.json,results.json,213k-load.log}` |

主工作盘原本仅8.4GiB空闲，无法保存新的11.6GiB包。本次专属原始包曾位于 `/tmp/ninfer-openspec-mtp4-213k-20261022/prefix.nkv`（tmpfs）；save+restart restore全部通过后，**只删除本次自己创建的临时包、派生索引及临时元数据**来释放空间，测试收据永久保留在上述工程路径。若未来需再次恢复MTP4 213K，必须重新生成该包或获得容量合适且授权的存储位置；旧MTP15包不能替代。

## 256K：首次 tmpfs 保存失败及容量门（历史记录）

- 尝试通过完全相同的严格 `create --case 256k --draft-tokens 4 --checkpoint-stage raw`，新专属 checkpoint 位于临时内存文件系统，**首个请求在保存阶段**报 `checkpoint write failed`、HTTP500、无 terminal（`tmp-nvcc/opsx-s3-20261022/mtp4-256k-native-create/{checkpoint-receipt.json,requests.jsonl,256k-save.log,results.json}`）。未产生有效 checkpoint，未得到任何可用于判定答案对错的输出，故不是 MTP4 算法/事实召回失败。
- 原256K冻结归档长度15,301,783,444 bytes（14.25GiB），而 `/tmp` 总16GiB，写入时还需临时页和内存。记录到的错误仅为通用 `checkpoint write failed`，没有细化底层errno；**不擅自断言是ENOSPC还是其他I/O资源失败**。机器工作盘剩余8.4GiB，不满足目标文件大小，不能改用工作盘重复试错，更不能擅自清理模型或旧归档。
- 针对此次浪费的完整 prefill，在基准 `create` 路径加入实质性**写前容量门**：依据冻结manifest的原归档字节数，要求目的位置余量至少 `archive_bytes + 2 GiB`。同位置最小纯CPU预检实际拒绝：需要17,449,267,092 bytes、实际可用16,441,274,368 bytes，`NO_NEW_ARCHIVE`，200W保持；不会再次消耗几分钟GPU后才在落盘阶段失败。此容量门不影响正常restore或原格式。

## 256K：主工作盘容量恢复后完成原生新包与独立进程质量门

2026-10-22后续阶段工作盘 `df -B1` 可用54,186,577,920 B，超过原256K冻结包15,301,783,444 B和额外2 GiB暂存余量的合计门槛17,449,267,092 B；单张RTX4070 Ti SUPER处于200W，原模型及二进制SHA未变化。于**原冻结256K目录同一个NVMe分区**新建独立MTP4 ABI1 raw包 `benchmarks/frozen-long-context/256k/mtp4-raw-prefix-20261022.nkv`（真实路径位于 `/home/ran/ninfer-work/`），未覆盖原MTP15 `prefix.nkv` 与任何canonical-ready包。

| 新增阶段 | 原冻结6问精确匹配 | 完整归档 | Serve/资源 | 原始收据 |
|---|---|---|---|---|
| MTP4 `create --case 256k --checkpoint-stage raw --draft-tokens 4` | **6/6 exact，六次均HTTP200及terminal**，原输入261,411 tokens | 新建归档15,301,783,444 B，保存前缀261,395 tokens；`complete=1`，native写入14.263136s | exit0、终态requests_active=0 / inflight_pages=0 / inflight_bytes=0，200W | `tmp-nvcc/opsx-s3-20261022/mtp4-256k-ssd-create/{checkpoint-receipt.json,results.json,256k-save.log}` |
| **新Serve进程** `restore` 上述新MTP4包 | **6/6 exact，六次均HTTP200及terminal** | 恢复相同261,395 tokens；`complete=1`，native加载20.954858s | exit0、同样三个终态0，200W | `tmp-nvcc/opsx-s3-20261022/mtp4-256k-ssd-restore/{checkpoint-receipt.json,results.json,256k-load.log}` |

两阶段二进制SHA256均为 `53f0fc8642546971f23c1d739883810e9c4b1bedd66136b6e15f829444328a8a`，模型SHA256均为 `29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`。这里验证的是整个原生 GSQ+MTP4+KVMem 256K 质量门，**不是**256K完整历史全部驻留GPU或原位置R21修复。

可复现入口为 `PYTHONPATH=. /usr/bin/python3 -m tools.bench.prefix_checkpoint {create|restore} --case 256k --draft-tokens 4 --checkpoint-stage raw --checkpoint /home/ran/ninfer-work/benchmarks/frozen-long-context/256k/mtp4-raw-prefix-20261022.nkv --output <新的唯一收据目录>`；已有包不可覆盖，重做create须使用另一个通过容量门的全新包路径。两档宽度的恢复均必须读取**各自匹配**的完整包。**OpenSpec 4.4符合通过条件**；整体对照结果和默认决策见下节。全局默认依旧MTP15 + sync，无提交、推送、双卡或300W发布测试。

## 4.5：200W同二进制三对整体对照与保守采用结论

原256K MTP4质量门完成后先运行了真实Serve的**额外原256K整会话对照**：同一个上述`ninfer-serve`、同模型、同冻结六问、Greedy/no-thinking、物理36,864 INT8 KV / KVMem32,768 + reserve512、原生sync、同一NVMe分区上的两份各自合格RAW ABI1归档，三对顺序MTP15→MTP4 / MTP4→MTP15 / MTP15→MTP4。每个臂独立Serve进程，共36/36原答案exact、完整HTTP200/terminal、exit0、终态inflight/active均0，未共享不同介质的快照。

| 256K原六题，真实API第一个请求发送至第六个请求结束 | 第1对 | 第2对 | 第3对 |
|---|---:|---:|---:|
| MTP15 session wall（秒） | 39.330439 | 39.175543 | 38.425775 |
| MTP4 session wall（秒） | 37.672258 | 37.330543 | 37.417459 |
| MTP4成对减少 | 4.2160% | 4.7096% | 2.6241% |

上述成对中位节省**4.2160%**，仅对这六个极短确定答案的固定续问成立；每臂历史重放32,211 tokens、restore 261,395 tokens，MTP4实际接受42/提议44、MTP15接受54/提议90；不将较高接受比例推断为通用更高吞吐。原始逐轮SSE、TTFT、metrics、子进程命令与汇总位于`tmp-nvcc/opsx-s3-20261022/mtp4-vs-mtp15-256k-200w-three-pairs/`，可复现入口`PYTHONPATH=. /usr/bin/python3 -m tools.bench.s3_mtp_paired --baseline /home/ran/ninfer-work/benchmarks/frozen-long-context/256k/prefix.nkv --candidate /home/ran/ninfer-work/benchmarks/frozen-long-context/256k/mtp4-raw-prefix-20261022.nkv --output <新的唯一收据目录>`。

按OpenSpec design的**正式整体吞吐口径**，又固定4K与32K输入、各tg128、C1、INT8-G64、Graph已prime、warmup1/repetition1、同一真实GSQ模型/65,536-token corpus、prefill chunk512、200W，用**同一个**现成`build/bench/ninfer_bench`二进制SHA256 `92244e50b35109039e91738c330d493b5edc236beacc5246ab675cc460a01fa7`运行MTP15与MTP4三对交替。每个进程一次测试两个上下文，六进程总计4K/32K各6次measured（共12条），每条报告generated129 / decode128，Graph prime已确认，Serve生产代码未因S5正在修改的源重新构建。每档分别计算三次候选/参考成对比例的中位，正值表示MTP4更好。

| 正式200W两档（MTP4相对MTP15，成对中位） | 4K/tg128 | 32K/tg128 |
|---|---:|---:|
| prefill吞吐变化 | +1.1528% | +0.2631% |
| decode吞吐变化 | **−14.0031%** | **−33.4194%** |
| 单例完整测量time节省（含prefill+decode，不含load/prime/warmup） | **−2.7889%** | **−1.1939%** |
| TTFT计算代理值改善（prepare+prefill，不等于Serve线上TTFT） | +1.1399% | +0.2627% |

两档三对全部decode负增长，MTP4提议/接受量在4K为103/102、32K为102/102，而MTP15分别为164/116和124/119：MTP4的高比例接受无法补偿缩小草稿窗口造成的额外round数。原JSON逐臂prefill/decode/total秒数、接受量、Graph prime、命令和三对原值在`tmp-nvcc/opsx-s3-20261022/mtp4-vs-mtp15-4k32k-200w-three-pairs/`；两组机器可读完整聚合另存`docs/performance/2026-10-22-s3-mtp4-200w-pairs.json`。正式两档重现命令为`PYTHONPATH=. /usr/bin/python3 -m tools.bench.s3_mtp_bench_pairs --reference-report <已有的合格首个MTP15报告> --output <新的唯一目录>`；对应底层命令为`build/bench/ninfer_bench --weights <GSQ完整模型> --corpus bench/fixtures/bench_corpus.ids -pg '4096,128;32768,128' --max-ctx 36864 -r 1 --warmup 1 --kv-dtype int8 --prefill-chunk 512 --spec mtp --draft-tokens {4|15} --lm-head-draft --output json --output-file <报告.json>`。

**证据限制：** `ninfer_bench`记录生成129/token decode128及完整吞吐/投机统计，但该旧可执行程序的JSON没有逐token输出IDs，因此正式4K/32K输出IDs相等项为`NOT_RECORDED`，不能伪称全token bitwise相同；实际256K API三对独立验证了36个精确原问答案，213K/256K MTP4新包两阶段质量各通过。4K/32K吞吐用Engine基准可执行程序、256K API用Serve，两者是分别冻结的可执行文件，不把跨可执行文件绝对速度直接相减；以上百分比仅各自**同binary内**比较。4.5对照与命令交付完成，胜出候选**只作为显式可选MTP4 profile**，全局默认继续**MTP15 + sync**；未做300W发布终测、双卡、DFlash2长档或旧R21修复。
