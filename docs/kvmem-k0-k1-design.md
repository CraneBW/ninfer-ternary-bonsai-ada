# KVMem K0+K1 设计：宿主块仓库与窗口组装（zatfung）

状态：K0/K1a/K1b/K2 已实现（K1b 的 serve 语义见 §6）；K3 见 §7。基线见
`perf-baseline-sm86.md`；机制评估见 `kvmem-reference-assessment.md`。参考实现：
`laamaafung/kvmem/`（v26 线）。

## 0. 本文档修正一处早前判断

`kvmem-reference-assessment.md` 曾写"K1 = 选择结果 → block_table 组装（保持原位置），
完全不需要动内核"。**这个判断不成立**。因果链：

- 注意力内核把 **table 槽位下标当作 token 绝对位置**
  （`prompt_i8.cuh:194-233`：`key = tile_k0 + key_l`、`physical_page = block_table[key >> 6]`、
  因果掩码 `key <= base_pos + q0`，tile 迭代数由 `max_query_abs` 决定）。
- 因此"选块但保持原始位置"无法减少内核迭代量（要迭代到 `base_pos` 为止），也就没有
  计算与显存收益；而"紧凑化窗口"（选中块按序打包到 `[0, W)`、query 位置 = W..）才是
  KVMem 的本意 —— 但紧凑化改变了每个块的槽位 → 槽位即位置 → **K 的 RoPE 相位必须
  重铸**。laamaafung 为此做的正是原地 de-rotate/re-rotate（`rope.hpp`）。
- KV 存的是已烘焙 K（`ops::rope` in-place 施加后写入 cache），所以重铸不可绕过。

**修正后的分层**（K 编号沿用评估文档，K1 内容修订）：

| 层 | 内容 | 改动面 |
|---|---|---|
| K0 | 宿主块仓库元数据 + 选择算法 + 窗口差分计划（纯宿主逻辑，可单测） | 新增，零内核风险 |
| K1 | 选择结果 → `block_table` 组装 + 新 token 位置 = 窗口位置 | 引擎序列规划（logical_kv_store 层） |
| K2 | K 相位重铸：驻留块原地 re-RoPE；冷块从 raw-K 镜像重铸 | 新增一个小 GPU kernel + 宿主编排 |
| K3 | 每步重选（retrieval/attention 热度信号接入） | K0 的信号源 |

## 1. zatfung 对接面盘点（已核实）

| 需要的能力 | 现成设施 | 位置 |
|---|---|---|
| GPU 页池（64 token/页） | `DeviceKVPagePool`（lease/reservation/zero/copy） | `core/paged_kv_cache.h` |
| **宿主↔GPU 页拷贝** | `copy_to_host` / `copy_from_host`（配 `HostKVAllocationView`） | 同上 :234-238 |
| 宿主页内存 | `HostKVArena`（同几何布局 `plan_host_kv_page_layout`） | `core/host_kv_arena.h` |
| block_table 发布 | `KVExecutionTablePool::publish(row, logical_begin, pages)` | `paged_kv_cache.h:345` |
| 表装配调用点 | `logical_kv_store.h` 四处 `tables_->publish(...)`（:1169/:1366/:1467/:1852） | qwen3_6 runtime |
| 新 token 位置 | 序列规划产 positions（MRoPE 3 轴 Text） | `decode_impl.h` / `layouts_impl.h` |
| 页内容纪元 | `HostKVPageReplica`（content_epoch / membership_node） | `logical_kv_store.h` |

结论：**搬运与发布的原语全在**，缺的是（a）决策层 K0、（b）序列规划接受"窗口计划"、
（c）重铸 kernel。

## 2. K0：纯宿主决策层（本次实现）

自包含（不依赖引擎头文件），落位 `src/kvmem/`，之后原样进引擎。

### 2.1 数据结构

```
KvBlockMeta   { id, token_begin, n_tokens(=64), tier{Gpu,Host}, in_working_set,
                baked_pos, attn_score, retrieval_score,
                remap_count, remap_abs_delta }        // 双阈值精度账本
KvSelectConfig{ budget_tokens, sink_tokens(默认 max(1%,1024..2048)),
                recent_tokens(默认 max(8%,4096..16384)),
                retrieval_blocks, profile_blocks }    // Quota 双信号配额
KvWindowPlan  { stage_in[], stage_out[],
                remaps[{block_id, from_base, to_base, skip, raw_refresh}],
                total_window_tokens,
                retained_position_stable, retained_position_moved }
```

### 2.2 算法（移植 laamaafung `pick_topk_ungrouped` + `set_selection` 差分）

1. **频段**：sink（头部，抗稀释）+ recent（尾部，保连续）恒选；mandatory（媒体组等）
   最新优先、装不下丢最旧并告警。
2. **top-k**：剩余配额按 retrieval/profile 双信号配额分配（`nth_element`），
   兜底用合并分数填满预算。
3. **差分**：未选中的 GPU 驻留块 → stage_out（保留其 baked_pos，供未来重铸）；
   选中块升序紧凑打包 → 每块生成 remap{from=baked_pos, to=window_pos}；
   `skip = (baked_pos == to) && K驻留`；`raw_refresh` 由
   `remap_count ≥ N` ∨ `remap_abs_delta ≥ T` ∨ `baked 越界` 触发（fp16 累积漂移对策）；
   非 skip 的块更新计数器，`baked_pos = window_pos`。

### 2.3 验收（单元测试，`_probe/test_kvmem_k0.cpp`）

- 覆盖不变量：预算不超、频段恒在、选中升序、窗口位置连续无重叠、
  `skip ⇔ baked==to && K驻留`、raw_refresh 计数器重置语义。
- 场景：全量装得下（直通）、超预算淘汰、mandatory 修剪、连续多轮重选的
  位置稳定性（多数块 retained_position_stable）。

## 3. K1：窗口计划 → 引擎（下一步，先直通后分叉）

分两个里程碑，风险递增：

**K1a（直通，验收 = 与基线逐位一致）**：选择 = 全部块。此时窗口 = 完整上下文，
`baked_pos == window_pos` 恒成立、零重铸，唯一变化是 block_table 组装与位置生成
走 K0 的计划对象。改 `logical_kv_store.h` 的四处 publish 调用收敛到一个
"窗口计划"来源 + 序列规划的新 token 位置从 `total_window_tokens` 起算。
**这一步只验证管道，不改任何数值行为。**

**K1b（真选择）**：GPU 页池缩到 `budget`；未选中块在 stage_out 时
`copy_to_host` 归档；新会话轮开始时按 K0 计划 stage_in（`copy_from_host`）+ 重映射。
直通模式作为常开的 A/B 臂（env `NINFER_KVMEM_MODE=off|on`）。

## 4. K2：K 相位重铸（K1b 的前置，可并行开发）

- 驻留块 `baked_pos == to`：零成本跳过（多数选择轮的大多数块）。
- 移动块：原地 de-rotate(from) + re-rotate(to)。int8-group64 KV 的完整链：
  反量化 → 去 Hadamard(H64) → 去 RoPE(from) → RoPE(to) → Hadamard → 再量化。
  新增一个独立小 kernel（每 token 每头一次），**不改任何现有内核**。
- 冷块（stage-in）：raw-K 镜像（宿主存未旋转 bf16 K）→ 上传 → RoPE(to) → Hadamard →
  量化。上传量 2×（bf16 vs int8），仅对当轮新入场块发生。
- 精度账本沿用 K0 的 remap_count/remap_abs_delta 双阈值。

## 5. 与 8GB 显存的现实约束

PQ2_0 权重 6.70 GiB，启动后仅剩 ~12 MiB —— **在 PQ2_0 上做 K1b 需要 cago 权重空间**：
要么降低 kv-capacity（现有 auto 会按剩余显存算）、要么用 PTQ1_0（5.52 GiB 权重，
1.0 GiB 可用于 KV+工作区，int8 KV 4096 token 仅 132 MiB）。
**建议 K1b 的性能对比用 PQ2_0、功能开发用 PTQ1_0**，与基线文档口径一致。

## 6. K1b serve 语义：窗口 + 窗口边缘锚点（已实现）

**问题**：`finish()` 处的轮末压实把 token 账本截到头窗口，却没有同步序列的
`rebuild_work`，于是 `checkpoint rebuild work does not match its frontier` 被吞掉，
**整条 continuation 根本没进目录** —— 表现为每轮 base=0、全量重预填充，压实只换来显存封顶。

**契约**：continuation 的每个 checkpoint 都带 `required_kv.main_frontier/pages`，页数与
frontier 一一对应，且 checkpoints 的 StateImage 必须真的处于那个 frontier。窗口若截到
`W < frontier`，全上下文 endpoint 就不可物化，必须连同 turn-closure rewrite 一起丢弃。

**解法**（窗口边缘的真实复原点）：

1. `plan_request` 为 KVMem 地址追加一个 frontier = `kvmem_window_edge(prompt)` 的私有
   **long anchor** capture（= 页对齐后的 `NINFER_KVMEM_BUDGET`；会切开 media consumer 的
   边界直接放弃，本轮 KVMem 不生效）。prefill 走到该 frontier 时快照出的 state 就是窗口
   边缘的 state，代价只有一次状态快照。
2. `finish` 只在**该锚点存在**时才允许压实（`kvmem_compaction_edge`）。压实成功后
   `kvmem_apply_window` 截断 ledger/prefix_identity/prefix_digests/`text_kv_valid`/
   `execution_frontier`，把 `rebuild_work` 重挂到锚点的 build work，并**丢弃 endpoint**
   （连同其 StateImage）——从此锚点是这条 continuation 唯一诚实的复原点。
3. 下一轮 `inspect_lane` 走 `PrivateLongAnchor`，base = 窗口长度。窗口是连续头前缀，
   位置天然对齐，**无需 K2 重铸**；新 token 位置从 `total_window_tokens` 起算（K1a 约定）。
4. 该锚点**每轮重新捕获**：reuse 时以 `ConsumeToActive` 消费它（锚点 image 直接成为
   active state），于是在 `cursor == base` 处捕获不会落在未结算的 state fork 后面
   （引擎禁止未结算 fork 上开 capture transaction）。因此锚点自续，不需要跨轮继承。
5. tracker 账本跟随地址：resume/分叉带来的既有页在 `ensure_mapped_to_tokens` 里按
   born-placed 补登记；`destructive_truncate*` 走 `kvmem_forget_window_tail` 忘掉窗口外的
   块并交还宿主副本。

**验收**（RTX 3060 Ti / PTQ1_0 / int8 KV / `NINFER_KVMEM=1 NINFER_KVMEM_BUDGET=512`
`--max-shared-prefixes 0`）：turn 2/3 的 `cache` = 512（路径 `private_long_anchor`），
prefill token 1335/1368 → 823/856，TTFT 34.6s → 22.4s/23.3s，回答不变；KVMem 关闭时
默认路径仍复用整段 turn-closure 前缀（97%）。

## 7. K3 仍待做

窗口现在是**连续头前缀**，因此省下的是"预算那么大"的 prefill，而不是 KVMem 论文里
"丢中段、留尾部"的收益。要拿到后者需要在注意力里表达非连续子序列，并让 continuation
契约支持 `KV 页数 ≠ frontier`（位置原点另行携带）；查询驱动的尾部保留与对应的重铸记账
都在那一层。

## 8. 窗口化 continuation 的契约扩展（设计，未实现）

目标：窗口 = `[0, S)`（sink/头部）+ `[TB, T)`（最近尾部），T = 本轮 prompt+生成的结束。
中段 `[S, TB)` 从 KV 里彻底移除（既不是"保持原位置做稀疏注意力"，也不做零页稀释）。
尾部块由 K2 重铸到紧凑位置 `[S, S+Wr)`，于是窗口位置 = `window_tokens = S + Wr`，
新 token 的位置从 `window_tokens` 起算。

**必须扩展的契约**（`ref.frontier` 与 KV 几何解耦）：

| 量 | 含义 | 谁用 |
|---|---|---|
| `prompt_frontier = T` | 本轮已"消费"的 prompt token 数（用于匹配与下一轮计数） | 复用匹配、digest |
| `kv_frontier = window_tokens` | 地址 frontier / 页数 / pin 范围 | `text_kv_valid`、`trim_sequence_kv`、`set_checkpoint_requirement`、`ensure_mapped_to_tokens` |
| `position_origin = window_tokens` | 新 token 的 RoPE 位置起点 | prefill 位置推导、`rope_delta` |
| `WindowGeometry{S, TB, Wr}` | 窗口几何，用于匹配 | catalog 精确校验 |

- 尾部重铸不需要新内核：`kvmem_compact(handle, budget, stream, mandatory, sink_only=false)`
  的 K0 选择（sink/recent/quota）+ `kvmem_rerope_page` 已经就位，`remap_count/remap_abs_delta`
  双阈值账本也已在 `set_selection` 里维护（K3 的"重铸记账"由此覆盖）。
- `prefix_matches` 无法表达"跳段"：匹配改为 **几何相等 + `prefix_digests.at(T)` 相等**。
  这要求 finish 时**保留** digest 到 T（不要截到 `window_tokens`），并把该 digest 显式存进
  检查点（`checkpoint_summary` 目前从 `sequence.prefix_digests.at(frontier)` 取，窗口化
  情况需要覆写来源）。resident `ledger`/`prefix_identity` 则按窗口顺序重排（= 抽掉中段），
  它们描述的是"当前驻留的 token 序列"，正好也是下一轮追加的起点。
- `inspect_lane` 对窗口化 continuation 走 `PrivateEndpoint` 形态：`reuse_base = T`（prompt
  切片起点），同时把 `kv_frontier`/`position_origin` 带进 plan；`start_sequence` 用
  `kv_frontier` 做地址/state 操作、用 `reuse_base` 做 prompt 切片。
- prefill 位置：suffix 的位置必须是 `window_tokens + (index - T)`，即对 prompt 的
  positions 施加 `window_tokens - T` 的偏移；`io.pos`/`rope_positions` 的 staging 站点
  （`schedule::prefill_text_chunk` 一路到 `TextContext`）是唯一需要改的落点，
  生成段则用 `append_generated(count, rope_delta = window_tokens - T)`。

**验收**（与 §6 同一套环境）：第二轮 `cache` 应 ≈ T（而非窗口长度）、窗口页数 =
`(S+Wr)/64`、回答质量应接近全量注意力（尾部可见）；对比 §6 的头窗口（中段与尾部都不可见）。
