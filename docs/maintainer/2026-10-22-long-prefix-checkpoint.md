# 固定213K / 256K测试的跨进程前缀快照

用户要求：已有Qwen3.8固定长题保存计算状态，之后不再在每次调优测试中从零处理整段长前缀。测试输入没有修改。当前实现用于单路、纯文本、MTP、canonical KVMem的固定基准。

## 输入与保存位置

|档位|逻辑context|首次请求实际token|长user正文字符数|目录|
|---|---:|---:|---:|---|
|213K|212992|212276|686767|`/home/ran/ninfer-work/benchmarks/frozen-long-context/213k/`|
|256K|262144|261411|840316|`/home/ran/ninfer-work/benchmarks/frozen-long-context/256k/`|

`input.txt` 是可读长user正文；完整请求的system、assistant tool-call、tool result、最后user与采样参数以 `manifest.json` 所指原始JSON为准。内容是重复的 `Note N: the record is unchanged and archived.`，其中埋入Topaz等固定事实、Quartz的旧值/更新以及Jade工具结果。先回复ACK，再问五个固定问题，六个期望输出依次为ACK、silver-maple-41、amber-willow-92、copper-lotus-35、copper-lilac-28、teal-jasmine-61。

快照文件为同目录的 `prefix.nkv`，资格与模型/二进制绑定为 `prefix.nkv.json`。这些大文件位于工作区基准目录，不进入源码仓库。

## 保存的边界和数据

在初始长请求的q0保存，即开始执行最后user问题之前。保存：

- Main/MTP的完整native K/V及scale，所有Device/Host/Cold页和逻辑目录；
- 不可变首次表示K、尺度、birth positions及有效列；
- 完整StateImage的GDN recurrent/conv及continuation hidden；
- mean-K、有效token数、LRFU/profile及评分时钟、窗口位置与复用边界；
- 精确完整token序列、q0、模型SHA256，以及context/KV/投机/页布局/注意力缩放绑定。

新进程加载时重建句柄、页引用和物理映射，不复用旧进程地址。快照的二进制哈希保留创建来源；恢复绑定状态ABI、模型和布局/计算配置，允许不改变状态语义的调度/搬运构建继续使用。改变数学或布局时须升级ABI并重做cold资格，不能用旧快照验证新prefill数学。Native格式的身份和布局验证也保留。当前不是任意聊天的服务端会话导入API；只命中同一冻结首请求。初次选页会读取完整query/lexical，因此不能把首问不同但长正文相同视为同一状态。

加载ABI1跳过大段初始前缀的GPU计算。该原包初问仍完成选页及一次canonical工作集回放；固定基准工具现默认在资格成立时保留这个精确canonical状态续问，未满足条件仍重建，参考臂可用 `--no-canonical-query-reuse`。原213K三对36/36及256K6/6通过，见[续问复用及性能边界](2026-10-22-canonical-query-reuse.md)；没有将R21失败的原位置模式打开。`ninfer:restored_prefill_tokens_total` 单独计数，恢复token不加入 `computed_prefill_tokens_total`。HTTP `cached_tokens` 和进度计数包含已恢复前缀；资源计划的reuse path仍表示目录选出的原路径。不能用跳过计算的前缀认领prefill吞吐。

## 来源及有界写入

- 分块codec直接适配 `kvmem/kvmem@faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 的 `kvmem/include/kvmem/snapshot.hpp`（保留来源；该上游根目录未见独立LICENSE，未据此认领许可审查完成），保留1MiB分段I/O与滚动校验。
- 参考本地Ninfer的 `ninfer-4080-32g/.../session_snapshot_impl.h` 和 `ninfer-all/.../storage/session_snapshot.cpp` 的完整递归状态及配置绑定原则。
- 原KVMem NVMe会话目录只在同一服务运行中有效；本实现额外保存本地超池目录，并恢复所有进程内所有权，支持服务正常退出后新进程加载。

不建立整个12–15GB快照的RAM副本。逐页写入，复用已有固定Host staging与StateImage槽。写临时文件、fsync后原子发布；加载先完整验证长度/校验和，再构造状态，完成时再次验证消费完整性。此合同为可信本地文件的跨进程恢复，不承诺跨架构、跨数学版本或突然断电恢复。

## 命令

在 `/home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada` 执行。生成只需每档一次，不覆盖已有包；输出收据目录须为新目录。默认工具功耗200W。下方所引用的初始快照资格在300W完成，属于历史记录；后续日常测试统一200W，只有发布前最终验收使用300W并恢复200W。

```sh
python3 -m tools.bench.prefix_checkpoint create --case 213k --output tmp-nvcc/prefix-213k-create --power-limit 200
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 213k --output tmp-nvcc/prefix-213k-restore --power-limit 200
python3 -m tools.bench.prefix_checkpoint create --case 256k --output tmp-nvcc/prefix-256k-create --power-limit 200
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 256k --output tmp-nvcc/prefix-256k-restore --power-limit 200
```

这四条命令串行运行；本机执行用systemd用户scope/service约束20GiB RAM / 2GiB swap，`SendSIGKILL=no`。native Serve由原资格工具持有，结束只TERM并等待自身退出。每次restore会重新启动Serve，使用同一原始六请求；不会取保存好的答案作回复。

工具是固定基准入口，显式设置当前已验GSQ MTP15/sync档。Native实验开关为 `NINFER_BENCH_PREFIX_CHECKPOINT`、`NINFER_BENCH_PREFIX_CHECKPOINT_MODE=save|load`、`NINFER_BENCH_CHECKPOINT_MODEL_SHA256`；一般直接用上述工具，以免绕过模型/原题/二进制绑定。

日常调优与回归统一使用200W；仅发布前最终验收把上述命令的功耗参数改为300，结束自动恢复200W。下方历史资格样本是在300W采集，保留原标注。

## 证据与测量口径

小用例：`build/tests/ninfer_kvmem_compact_test --checkpoint`。真实保存Device/Host/Cold、释放旧地址后以新代际加载，重新导出所有native/raw-K/metadata字节一致；修改真实文件一个字节，读取在mutation之前拒绝；之后仍通过原冷页K/V数学参考。收据 `tmp-nvcc/opsx-checkpoint/unit-2/receipt.json`。

正式资格和时长收据在 `tmp-nvcc/opsx-checkpoint/campaign-receipt.json` 及四个case/action子目录。生成臂的首TTFT包含保存开销，不当作纯cold-prefill基线；恢复时长、剩余query/replay、TTFT分别记录。已有文件的OS页缓存与真实物理冷读不同，本轮不做物理冷盘收益认领。

两档均完成保存后的原六题和新进程恢复后的原六题，各6/6正确，HTTP/usage/终态完整、在飞页清零。结果汇总 `tmp-nvcc/opsx-checkpoint/summary.json`。

|档位|包GiB|跳过初始token|保存秒|恢复秒|恢复后首TTFT秒|
|---|---:|---:|---:|---:|---:|
|213K|11.60|212260|11.37|23.93|42.33|
|256K|14.25|261395|20.35|32.17|51.02|

这些是本次加载样本，不是三对稳定性能结论，也不是cold-prefill速度。213K最终计数补丁又用原前两请求验证，恢复27.06s、首TTFT45.59s；首次cached_tokens=212260，续问=212269，computed_prefill未将恢复前缀重复计入。状态/原题资格继承两档完整六题的相同数学路径，统计补丁没有再重复整套长题。所有测试300W，最终恢复200W。资源限制20GiB RAM/2GiB swap下完成；两档包合计25.85GiB。

213K的canonical回放仍为185836 tokens、复用10240；这项快照消除了新进程的大段初始prefill，不等于解决每题的历史回放。R21原位置实验和真正逐层forward-ready仍未完成、未开启。

恢复入口可显式 `--page-pipeline` 选择前一批source-event/async-D2H候选，以便后续使用同一恢复包做对照；默认仍为已验sync。该候选此前首对整体收益未达门，本次两档快照资格为sync，不将新组合认领为P4通过。

## 2026-10-22 快速完整校验默认

`restore`现默认 `--checkpoint-io indexed`（native `NINFER_BENCH_CHECKPOINT_FAST_VERIFY=1`）；可显式legacy/0回退。每份prefix.nkv旁新增约190/234KB的 `.xxh3idx`。第一次没有索引时仍需完整FNV，之后每次全文件分块预验证、消费时逐块复验；不是仅凭mtime命中。原包、FNV footer及state ABI1不变，不用重新prefill生成。索引可删除重建，索引损坏/陈旧时回退旧验证；只读目录使用当次内存索引。`create`仍写相同格式，IO选项只影响读取。

200W已完成213K三对36/36和256K6/6。首TTFT成对中位减少27.58%，session减少7.34%；详见 [完整IO优化记录](2026-10-22-checkpoint-io-optimization.md)。canonical-prepare实验仍0，未因此解决R21或逐层流水。

## Canonical-ready后续默认

新增独立 `canonical-ready.nkv` /ABI2，保存完整重建完成q0，包括冻结查询Q、canonical位置/tokens及四个有效Host回放检查点。`prepare`从现有ABI1生成，不再cold-prefill整段；`restore`auto优先已有ABI2。两档均保存与重启各6/6，213K同binary三对36/36、history replay32228→0；首TTFT32.99→10.35s。原ABI1保留，上述历史重现命令已显式raw。[完整说明与成本](2026-10-22-canonical-ready-checkpoint.md)。改变数学/布局仍须重做cold资格；逐层页流水/P4须选raw，不用已经省掉replay的入口测流水收益。
