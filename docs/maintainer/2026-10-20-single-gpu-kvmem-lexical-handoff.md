# NInfer Ada 单卡优化暂停交接（2026-10-20）

> 状态：用户明确要求“先暂停一下，写一个交接文档”。本文件创建后停止开发、编译、推理、提交与推送，直到用户指示恢复。**双卡 TP2 持续暂停，只有用户明确要求“弄双卡”时才恢复。**

## 1. 工程现场与保护范围

- 主仓库：`/home/ran/ninfer-work/publish/ninfer-ternary-bonsai-ada`；CoS 对应 `/ninfer-work/publish/ninfer-ternary-bonsai-ada`。
- 分支 `master`，最新已推送 HEAD **`2138637 docs: record kvmem historical recall regression`**；在本次未完成的词法补救尝试之前，`master` 与 `origin/master` 同步。
- 单 GPU：RTX 4070 Ti SUPER，CUDA `sm_89`、16 GiB 显存；现有 27B `.ninfer` 完整权重已下载并经过 SHA256 校验。**禁止无意义地重下或重复对整个权重做哈希**。
- 本次暂停核对时，没有运行中的 `ninfer-serve` 进程；没有开展新的构建或推理。
- **外部会话预留修改**：`src/ops/linear/q4/q4_dispatch.cpp` 一直是脏文件；绝对不可修改、暂存、覆盖、提交或清理。
- Companion checkout：`/home/ran/ninfer-work/ecosystem/ninfer-all`，基线 `b2913b6`，约 44 个未提交实验修改；**不得提交、推送、清理该 checkout**。Bonsai2B 实验补丁在主仓库 `tools/companion/bonsai2b-smallshape-experimental.patch`。
- 其他边界：禁止 `git reset`、`git clean`、破坏性清理、动无关 `/home/ran/zatfung`、GitHub issue `#7/#4`；避免相同 build 目录的并发 Ninja。不要未经授权使用私人 HF 凭据。
- 用户已明确要求减少不必要的测试和审查，优先真实代码、必要的定向测试；**任何真实双卡推理、NCCL 安装或 TP2 功能开发都暂停**，已有 TP2 代码与提交保留不回退。

## 2. 最近已完成、验证并推送的主仓库提交

在 `2d99508` 之后，按时间顺序：

| 提交 | 内容 | 验证边界 |
|---|---|---|
| `0a1bd26` | `perf: amortize kvmem cold free-list growth` | ColdKVStore 改为几何扩容；4096 页写入、回收、旧句柄失效的定向 CPU 测试 PASS |
| `97fb892` | `fix: retain unconsumed ffmpeg video packets` | FFmpeg `avcodec_send_packet` 返回 `EAGAIN` 时先 drain 后重新提交未消费包；media 解码测试 PASS，实际 MP4 4/4、40/40 帧检查 PASS；未直接强制注入 `EAGAIN` |
| `fa78e64` | `fix: count actual frames in vfr video without metadata` | 当容器没有 `nb_frames`，不再用 duration × fps 推测帧数；9 KiB VFR MKV 回归 6/6 帧 PASS，原 MP4 4/4、40/40 帧仍 PASS |
| `2138637` | `docs: record kvmem historical recall regression` | 增加**失败的真实 KVMem 检索**及完整 KV 控制对照文档；不是修复完成声明 |

此前已经完成的真实单卡成果（不可因本轮暂停被误认为丢失）：Qwen3.8 GSQ v3 base/MTP/DFlash2/Vision、Vision+MTP、Vision+DFlash2、多图和短 MP4；Qwen3.6 Groupwise/NVFP4 的 base、MTP、Vision、Vision+MTP；Host/Cold KVMem 路径、Action-LRFU 256K 页级调度；Companion Bonsai2B 512 点 Hadamard 补丁及真实生成。具体证据见：

- `docs/maintainer/2026-10-20-Ada-full-feature-status.md`
- `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md`
- `docs/maintainer/2026-10-20-v3-vision-multi-image-video-single-gpu.md`
- `docs/maintainer/2026-10-20-qwen36-v3-extra-single-gpu-validation.md`
- `tools/companion/HEADER_CONFIG_ORACLE.md`

**不要把 HTTP 200、元数据解析、部分 ModelView 绑定或单个正确输出当成完整功能验收。**

## 3. 可复现的当前主问题：多轮历史事实丢失

2026-10-20 真实 GSQ 27B 单 GPU 成对对照（完整提示约 2968 tokens）：第一轮将 171 条备忘录送入模型，位于中部的一条是 `The Indigo Archive display label is cobalt-orchid-73.`，第一轮返回 `ACK`；第二轮问 Indigo Archive 的精确 display label。

| 路径 | 第二轮结果 | 说明 |
|---|---|---|
| `--kvmem-budget 512 --kvmem-action-lrfu --kvmem-lrfu-lambda 0.5 --kvmem-lrfu-top-frac 0.3`，设备 KV 1024 tokens | **`Indigo Archive`，错误** | 2968 prompt tokens；98.8% 的 prompt-cache 命中，选中页 `0,40,41,42,43,44,45,46`；中部事实所在页未保留 |
| 不开启 KVMem，设备 KV 3072 tokens | **`cobalt-orchid-73`，正确** | 相同第二轮请求，同一真实模型权重及贪心解码 |

压缩路径 `KVMEM_TRACE`：13 次 compactions、`cold bytes=0`、`host_to_cold_spills=0`；故该反例指向**查询相关历史页检索/保留质量缺口**，不是 NVMe 文件丢失。单样本反例不能推算全局回忆率或证明任何启发式一定改善。

原始本地证据（在主仓库 `tmp-nvcc/`，不需要再次下载模型）：

```
kvmem-multiturn-first-request.json
kvmem-multiturn-first-response.json
kvmem-multiturn-second-request.json
kvmem-multiturn-second-response.json
kvmem-multiturn-fullkv-control-response.json
gsq-v3-kvmem-multiturn-20261020.log
gsq-v3-fullkv-control-20261020.log
```

关键源代码：`src/kvmem/kvmem_retrieval.h`（mean-K 与 Query GQA-max cosine）、`src/kvmem/kvmem_blocks.h`（sink/mandatory/action/recent 与 soft score quota）、`src/targets/qwen3_6/impl/runtime/logical_kv_store.h`（页记账、评分、compact/host-cold）、`src/targets/qwen3_6/impl/runtime/program_impl.h`（从 staged prompt 捕获 query 与分块 prefill）。

## 4. **暂停时未完成、未编译、未测试、未提交**的代码

最新 `git status -sb` 已有以下本轮试验性改动：

```
 M src/kvmem/kvmem_blocks.h
 M src/ops/linear/q4/q4_dispatch.cpp            # 外部所有，禁止碰！
 M src/targets/qwen3_6/impl/runtime/logical_kv_store.h
 M src/targets/qwen3_6/impl/runtime/program_impl.h
 M tests/kvmem_blocks_test.cpp
?? src/kvmem/kvmem_lexical.h
```

**这些内容只是进行中的实现草稿，不能声称 PASS，更不要直接提交：**

1. `src/kvmem/kvmem_lexical.h` 新增 `kvmem_lexical_recall_pages(tokens, query_begin, query_end, page_tokens)`：对最多 128-token 的最新短问题，统计之前每页匹配的低 document-frequency token ID；若某页命中至少两种罕见 token，则候选选择该页和后续一页。这只匹配**token IDs**，不是字符子串；尚未拿真实 tokenizer 与上述失败请求验证候选能否选到目标事实。
2. `src/kvmem/kvmem_blocks.h` 为 `kvmem_select_blocks` / `preview_select` 增加可选 lexical page 列表，优先级落在 pinned sink、mandatory、action、recent 之后，score quota 之前，保留预算上限。
3. `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` 在每个 address 内维护 lexical pages，address 释放时清空，并在非 `sink_only` compaction 中送入 preview；只有开启 `action_lrfu` 才会使用。
4. `src/targets/qwen3_6/impl/runtime/program_impl.h` 在 `staged.query_captured` 首次置位前，从 `staged.prompt.token_ids` / `query_begin..query_end` 构造候选并写入 text/backend KV address；带媒体输入时禁用，开启 `NINFER_KVMEM_TRACE` 可打印 lexical IDs。
5. `tests/kvmem_blocks_test.cpp` 添加了 **`test_lexical_history_rescue` 函数但尚未在 `main()` 中调用**；故即使当前独立 test 可执行也不会检查新逻辑。

**恢复前必须核查的实现问题：**

- 新测试中 `kvmem_lexical_recall_pages(tokens, 48*64, 49*64).empty()` 使用 64-token 查询，虽然测试注释称“long”，但实现阈值是**超过 128 tokens** 才跳过，断言可能不成立；修改为真正大于阈值的输入或者测试真实 common-token 场景。务必让测试在 `main()` 中实际被调用。
- 当前 lexical 规则能否命中真实问题的 `Indigo Archive` 文本 token，以及保留页是否包含完整 `cobalt-orchid-73` 值，尚未验证；若未命中，不能为追求单样本结果而无依据叠加权重。
- 受窗口大小限制，`lexical` 页不能挤掉系统/工具前缀、当前动作、强制页及新近页；候选页一旦被插入，会减少语义检索配额，需测试语义/长文场景的负面作用。
- `kvmem_lexical_recall_pages` 扫描整个历史 prompt（逐页构造 `unordered_set`），这是每轮额外 CPU 成本。对于最大 256K 上下文及较高并发须在语义有价值后量化开销；可考虑一次性预建 sparse token→page 倒排，但**不要在证明方案有效前扩大架构**。
- 页 ID 是逻辑 KVMem repository ID，注意 prefill 期间多次 compact 和 ledger / packed KV 索引偏移；若 prompt 缓存恢复、fork、MTP backend 位置不一致，必须保证不存在错误映射或请求间污染。
- 严格确认 protected page 的集合未随 candidate 增加超过预算；后续测试应有单轮正常输入、两轮事实成功、无关键词情况下不影响基线的独立对照。

这一批尚未执行任何新 Ninja build、单测、全模型推理；**不要把现有未验证草稿当作功能完成**。如果继续研究时发现策略不可靠，允许只回退**上述本轮试验性文件/片段**（保护外部 `q4_dispatch.cpp`），另选更稳健办法，但不可执行全局 `git reset` / `git clean`。

## 5. 恢复时的明确执行顺序（须等待用户说继续）

1. `git status -sb` 和 `git log -3 --oneline` 快速确认现场；不要重复扫描整个仓库。确认 `q4_dispatch.cpp` 仍归外部所有。
2. 审查 `src/kvmem/kvmem_lexical.h` 的统计逻辑、测试边界，并**调用**新增 `test_lexical_history_rescue()`；先单独编译并执行 `tests/kvmem_blocks_test.cpp` 的 CPU standalone。运行失败则先纠正逻辑，不能掩饰。
3. 仅在 CPU 检查通过后，用 `TMPDIR=$PWD/tmp-nvcc cmake --build build --target ninfer-serve -j2` 做必要单卡路径编译；编译前确认没有其他 Ninja 占用相同 build 目录。
4. 用既有验证过的 GSQ 权重运行**原样** `tmp-nvcc/kvmem-multiturn-{first,second}-request.json`：保留上一轮配置（3072 context、1024 device KV、512 KVMem budget、128 MiB Host KV、Action-LRFU λ 0.5/top fraction 0.3、no graph、greedy、no thinking）。在 trace 中验证 lexical 页**确实选中事实所在页**，且最终答复与 full-KV 控制一致；必要时仅做一个以上负例/非精确问题对照。
5. 如果真实推理仍答错，记录事实，不要将 HTTP 200 作为成功。不要为了让单例通过而悄悄改变用户请求或权重。先检查完整值跨页、prefill 早期丢页、末尾评分覆盖、查询 token span、cache reuse；若机制无效，废弃草稿或替换为有可验证边界的方案。
6. 如确实得到稳健增量，更新 `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md` 和 `docs/maintainer/2026-10-20-Ada-full-feature-status.md`，记录成功/失败与可复现命令；仅 stage 本次相关文件，以**小写 Conventional Commit** 推送 `origin master`。**先暂停的当前指令不授权立即执行本步骤**。

## 6. 更长期剩余目标与不能夸大的范围

- KVMem：真实 256K、多轮、复杂工具事实的检索准确率，Host→NVMe 多轮转移/恢复，跨 compaction recall 质量仍需要证明。此前 4096 页测试只证明 host 选择与 ColdKVStore I/O 边界，并非 256K 完整推理。
- Vision/视频：GSQ、Groupwise、NVFP4 已有短图、双图、短视频、与 MTP/DFlash2 的部分组合验收；短视频对未出现的计时器有幻觉，高分辨率/长视频准确率并未完成。Qwen3.6 已验证的 Groupwise/NVFP4 artifact **没有 DFlash2 实际物理组件**，不能简单移除拒绝条件伪造支持。
- Companion：Bonsai2B 修复已验证真实推理；Bonsai 4B/9B/27B、Gemma4 26B-A4B 和 Flash-Next 125B 还缺完整可信权重，不应把 `.part` header 配置解析当成 CUDA/推理成功。Companion checkout 不提交不推送。
- TP2：之前已有 sharder、rank View、NCCL transport 和 lockstep 等已提交代码，但本机只有单 GPU，未进行真实双卡 numerical allreduce / inference。**用户最新明确指示双卡以后再说，现阶段持续暂停。**
- 不运行全 124 项测试，仅做对所改代码负责的目标验证；未证明整体全功能前，不调用 `session_finish`。

## 7. 当前暂停点结论

当前已可靠完成并推送的是 KVMem 冷页容量扩容、FFmpeg 未消费包和 VFR 帧数修复，以及一项真实、可复现的 KVMem **失败**对照。正在试验的词法历史页补救**尚未经过 build/test/GPU 验证**，保留在未提交工作树供恢复时检查。

本交接文件本身仅记录工程状态，不包含进一步代码优化；**写完即暂停，不提交不推送。**

## 2026-10-21 后续执行记录

上述为10月20日历史暂停点；用户已多次授权继续，当前没有新的暂停时限或关机安排。当前按“先算理论、合并多项、统一验证，失败再拆分”推进，原题、已有GSQ权重和失败证据保持。

R24 source read-ahead原六题6/6；R26整包原36题36/36但性能不稳定，默认0、不采用。R27三格式规约/合作写回resident decode三对均正，中位+1.601048%，完整129 IDs相同；source四warp部分因恢复配对不稳定已撤回。最终新Serve decode-only原213K六题6/6精确正确、完整终态、server0，显式GSQ profile使用R27=1/R26=0、全局默认0。未宣称TTFT/P4或设备最优。当前结论、资源及入口见[活跃验证记录](2026-10-20-kvmem-actkv-live-validation.md#r24r26r27先理论批量修改统一验收2026-10-21)和目标仓`tmp-nvcc/perf-ada-round27-bundle/`。未提交、未推送，外部`q4_dispatch.cpp`保持不动。


R28批量筛选（2026-10-21）：先计算真实热shape与流量/寄存器/shared成本，再一次合入四个I64/J128完整K MMQ shape和GDN Q/K/V直接打包输出，共七文件，NINFER_R28_BUNDLE缺省0。八个公共MMQ独立constant oracle、GDN七种实际格式/T512/17/8的独立constant oracle与完整153,990,144B输出/guard/Graph比较通过；down规约改变产生少量BF16差异，未承诺bitwise数学不变。

同一binary统一做两组整体ABBAAB，每组六次：短pp4096+tg128的prefill/decode吞吐中位分别−1.114718%/−1.574450%；长pp32768+tg16分别+0.873695%/−0.404652%，长prefill三对有正有负。短129和长17个完整输出IDs各自六次一致，投机工作量各自相同。状态MATH_PASS_REJECTED_NO_STABLE_OVERALL_GAIN，R28=0，R27=1/R26=0基线保留；不为负组合追加原213K、TTFT或P4矩阵，也不逐格式做长性能筛选。构建256.935s/4G/0swap；统一数学5.731s/912.7M/0；短56.021s/13.5G/0、长236.425s/2.1G/0（服务内存峰值不可当作进程RSS比较），全部terminal0。精确实测binaries、配对原值、失败采用门与收据保存在tmp-nvcc/perf-ada-round28-bundle；原题质量NOT_RUN，不能继承旧6/6作本批验收。没有提交推送。

后续R29仍按用户“理论先行、多项相容改动后统一构建/测试”推进：优先复用现成T16 MMQ尝试固定MTP15，并只在原fullK/no-fixup几何合法时直写BF16；同时准备保持BF16舍入边界的SiLU/down激活量化融合。当前仅研究和CPU overlay，尚未生产构建、数学/Graph/质量或性能验收。R21原位置模式仍失败且默认0，15.4及16.1–16.4不勾选；不存在新的暂停期限或关机授权。


R29 / v1.0.0 发布验收（2026-10-21）：按用户要求先算理论与资源成本，将 fixed MTP15/T16、fullK/J16 BF16 writer 与保持BF16舍入的SiLU/down量化融合一次合入19文件后统一构建与数学验收。50次Op（T16十格式含完整Q4_K head22次，融合七格式constant/random28次）和K15混合接受前缀/guards/frame通过。初次eager129IDs正确，Graph在拓扑切换失败；只对明确topologyChanged先实例化替代exec后换，旧失败保留，修复前后Ops/GGML/Engine库字节未变，不重复数学矩阵。

最终同binary A显式六flags0/K7、B六flags1/K7、C六flags1+R29/K15，ABC/CBA/ABC，4K与32K各9进程、warmup1/r1/tg128；每档完整129IDs九次一致，短C Graph与已验eager投机工作一致。B/A短prefill吞吐中位+5.232890%（三对正）、decode+.761975%（两正一负）；C/A长prefill+9.582095%、decode+11.779491%（均三对正）。C/B短decode−15.183451%，长+10.219626%，故采用两个显式profile：短K7、长32K档K15；无通用自动阈值或普遍K15收益。

最终K15/Graph/sync/W32768/物理36864/HostKV0原213K六题6/6精确、HTTP/terminal/usage完整、server/service0；248.360s、cgroup峰值16.9G/swap329.5M（20G/2G，非OOM）。整组history replay185836/reused10240/query162、718压实/3010Coldspill/36998sourceK恢复、staging2297856B保持。单臂API TTFT首题152.592s、后五题中位17.715s，仅作观测，不认领配对TTFT或新P4改善；原位置实验仍失败且默认0，15.4和16.1–16.4仍未完成。

全新README、CMake软件版本1.0.0、完整整体原值与原题/数值证据纳入源代码版本；公开报告见 docs/releases/v1.0.0.md 与 docs/releases/v1.0.0/。本机完整失败和精确二进制保存在 perf-ada-round29-bundle。权重、二进制、Cold后备、profiler数据和外部所有权q4诊断改动不纳入源码提交。发布推送/tag状态以实际收据为准。
