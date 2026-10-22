# NInfer 工程大文件与可清理空间清单（2026-10-22）

**本次仅盘点，不删除或移动任何原有文件。** 扫描范围 `/home/ran/ninfer-work`；扫描时所在 Btrfs 分区约 478 GiB、已用约 467 GiB、空闲约 8.4 GiB（99% 已用）。本项目的 `du` 占用约 132.11 GiB；分区其余已用空间不归因于本项目。

完整清单：`docs/maintainer/2026-10-22-project-large-files.tsv`（**201 个不小于 50 MiB 的普通文件**，含相对路径、大小、磁盘 blocks、修改时间、保留/清理建议）。不扫描/清理项目之外的目录。

## 大目录（`du -x -B1`，单位 GiB）

| 目录（相对 `ninfer-work/`） | 占用 GiB | 判定 |
|---|---:|---|
| `benchmarks/frozen-long-context/` | 51.78 | **保留**：213K、256K 两套 ABI1 raw + ABI2 canonical-ready 冻结检查点 |
| `publish/ninfer-ternary-bonsai-ada/` | 49.39 | 当前主仓；含 `build/` 26.17、`tmp-nvcc/` 23.12，绝大部分为可重新编译或历史复制的二进制 |
| `real-models/` | 14.62 | **保留**：唯一现役 Qwen3.8 GSQ 以及 Bonsai2 模型 |
| `ecosystem/` | 8.67 | 第二工程组；其中 `ninfer-all/` 约8.03；暂不跨工程清理 |
| `hf-xet-gsq/` | 6.90 | 单个旧 `.incomplete` 下载文件，疑似中断缓存，须确认不需要断点续传 |

主仓的 `build/tests/` 目录占约22.74 GiB；其中84个大于100MiB的测试可执行文件合计22.31 GiB、**81个在10月22日前修改**，占约21.54 GiB。当前编译生成的测试程序建议保留，旧程序重新编译有时间代价。

主仓的 `tmp-nvcc/` 占23.12 GiB；其中按修改时间与体积筛出的73个较旧大文件合计18.51 GiB，包含大量 `ninfer-serve`/benchmark/test 的历史二进制副本。**不能整目录删除**：`*.json`、`*.jsonl`、`*.md`、`*.log`、`*.sse`、`*.nsys-rep`、原测试 manifest、资格收据和结果会用于 OpenSpec/P4 对照；即使二进制可重新编译，SHA和精确复现资格仍可能失效。

## 最大的六个独立文件

| 文件 | GiB | 建议 |
|---|---:|---|
| `benchmarks/frozen-long-context/256k/canonical-ready.nkv` | 14.29 | **保留**：256K ABI2 |
| `benchmarks/frozen-long-context/256k/prefix.nkv` | 14.25 | **保留**：256K ABI1 |
| `real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer` | 13.99 | **保留**：当前真实权重 |
| `benchmarks/frozen-long-context/213k/canonical-ready.nkv` | 11.64 | **保留**：213K ABI2 |
| `benchmarks/frozen-long-context/213k/prefix.nkv` | 11.60 | **保留**：213K ABI1 |
| `hf-xet-gsq/.cache/huggingface/download/NOw2NqaIf9x4zzeM_F_-85PcsNI=.29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750.7a5bba05.incomplete` | 6.90 | **第一清理候选**：最后修改2026-10-19；扫描时没有发现正在打开它的进程。若需要断点续传，则不得删除。 |

针对上述四个快照及主模型的 `btrfs filesystem du -s` 显示为独占 extent，非 reflink 共享：这表示它们确实使用大额物理空间，但重建成本高，不建议作为清理对象。

## Btrfs 共享程度与物理回收限制

`btrfs filesystem du -s` 查询结果：

| 目录 | Total | Exclusive | Set shared |
|---|---:|---:|---:|
| `build/tests/` | 22.74 GiB | 22.38 GiB | 323.33 MiB |
| `tmp-nvcc/` | 23.11 GiB | 20.57 GiB | 1.44 GiB |
| `ecosystem/ninfer-all/tmp-nvcc/` | 5.61 GiB | 4.87 GiB | 726.27 MiB |
| 单个 `.incomplete` | 6.90 GiB | 6.90 GiB | 0 |

以上不应用 `du` 不加区分地累加求“删除后必定增加多少可用空间”：CoW共享extent、文件系统元数据和同时运行的编译活动均会影响实测净释放。实际每一批清理后以 `df -h` 和 `btrfs filesystem du` 重新确认。

## 建议清理批次（先确认，再执行）

1. **低风险缓存，约6.90 GiB**：仅删除已确认不需要续传的上述 Hugging Face `.incomplete` 文件，不删除最终模型、不操作其余HF cache。
2. **旧测试可执行文件，约21.54 GiB**：按明确列表仅处理 `build/tests/` 中10月22日前生成的81个 >100MiB 可执行文件；保留新测试及 `build/apps/` 当前 Serve。若需旧二进制精确复现实验，先单独保留对应文件和SHA收据。此组可由 CMake/Ninja 重建，但需要重新编译时间。
3. **历史 benchmark 二进制拷贝，最大约18.51 GiB候选**：仅在逐个对应 manifest/性能资格之后移除 `tmp-nvcc/` 中过时的可执行/静态库副本；**完整保留实验收据、冻结请求、Nsight report 和 OpenSpec 文档**。Btrfs共享extent使实际净释放可能少于18.51 GiB。
4. **其他工程，需单独确认**：`ecosystem/ninfer-all/tmp-nvcc/` 约5.61 GiB、`build-ada-companion/` 约2.33 GiB，属于另一个工作树，不建议在主工程清理中顺带删除。

第一、二批合计账面约28.44 GiB，**实际释放额需执行后实测**；这已经足以成为解除16GiB级新快照容量门的重点候选。当前仍保留 213K/256K 固定包、完整GSQ模型、工作树dirty修改、所有Git文件及新 OpenSpec收据。

**盘点结束状态：没有执行删除或移动；未提交、未推送。**

## 用户批准后的实际清理执行补记

上述段落描述的是**只读盘点阶段**。其后用户明确批准删除三类文件，已依锁定清单逐项检查原 inode、dev、大小、mtime 和可执行文件格式，再执行永久删除；未删除任何清单外文件。

| 实际处理范围 | 已删除数量 | 文件原占用（按 blocks） |
|---|---:|---:|
| Hugging Face 旧 `.incomplete` 下载 | 1 | 6.899 GiB |
| `build/tests/` 旧大型 ELF 测试程序 | 81 | 21.539 GiB |
| `tmp-nvcc/` 历史大型 ELF / `.a` 复制件 | 71 | 18.199 GiB |
| **合计** | **153** | **46.637 GiB** |

在 `sync -f` 后实测该 Btrfs 分区可用空间由 **8.186 GiB 增至 50.430 GiB，实际净回收 42.243 GiB**；`df -h` 显示约 **51G 空闲、90% 已用**。文件原账面块数不能当作Btrfs实际净释放量。153条目标均经二次检查确认不存在；最新3个 KVMem 测试、当前Serve、主模型/Bonsai模型、213K/256K四份完整冻结checkpoint、OpenSpec tasks、真实MTP4与NSYS证据、Git HEAD及dirty源码均保留。`tmp-nvcc/` 的2个大型非二进制CUDA中间源码按范围保留。

**审核收据**：`docs/maintainer/2026-10-22-approved-storage-cleanup-manifest.json`（冻结删除前的目标及inode等信息）与 `docs/maintainer/2026-10-22-approved-storage-cleanup-result.json`（全部153个目标及实际删除结果、Btrfs同步后空闲空间、受保护文件二次检查）。执行 `git diff --check` 无错误；**未提交或推送**。
