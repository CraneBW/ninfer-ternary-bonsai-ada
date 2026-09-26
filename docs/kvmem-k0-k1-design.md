# KVMem 分层设计：宿主块仓库、窗口组装与查询信号（zatfung）

状态：K0/K1a/K1b/K2 已实现（K1b 的 serve 语义见 §6）；K3 的查询信号已接入（§7）、
窗口化 continuation 契约已落地（§8）。基线见 `perf-baseline-sm86.md`；机制评估见
`kvmem-reference-assessment.md`。参考实现：`laamaafung/kvmem/`（v26 线）。

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

结论：**搬运与发布的原语全在**，缺的只有三件——（a）决策层 K0、（b）序列规划接受
"窗口计划"、（c）重铸 kernel。三件均已落地（§2/§3/§4）。

## 2. K0：纯宿主决策层（已实现）

自包含（不依赖引擎头文件），落位 `src/kvmem/`。

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

### 2.3 验收（单元测试，`tests/kvmem_blocks_test.cpp`）

- 覆盖不变量：预算不超、频段恒在、选中升序、窗口位置连续无重叠、
  `skip ⇔ baked==to && K驻留`、raw_refresh 计数器重置语义。
- 场景：全量装得下（直通）、超预算淘汰、mandatory 修剪、连续多轮重选的
  位置稳定性（多数块 retained_position_stable）、检索配额是否真按分数取块
  （且分数全 0 时与无信号版选出一致的窗口）。

## 3. K1：窗口计划 → 引擎（K1a 直通与 K1b 真选择均已实现）

分两个里程碑，风险递增：

**K1a（直通，验收 = 与基线逐位一致）**：选择 = 全部块。此时窗口 = 完整上下文，
`baked_pos == window_pos` 恒成立、零重铸，唯一变化是 block_table 组装与位置生成
走 K0 的计划对象。改 `logical_kv_store.h` 的四处 publish 调用收敛到一个
"窗口计划"来源 + 序列规划的新 token 位置从 `total_window_tokens` 起算。
**这一步只验证管道，不改任何数值行为。**

**K1b（真选择）**：GPU 页池缩到 `budget`；未选中块在 stage_out 时
`copy_to_host` 归档；被重新选中的块 stage_in（`copy_from_host`）+ 重铸到新槽位。
开关是 `NINFER_KVMEM=1` 加 `NINFER_KVMEM_BUDGET`（token 数，页对齐）；不开时整条
路径不参与，默认复用行为不受影响。`kvmem_compact_round_boundary` 是喉管，窗口的
sink/recent/middle 三段由 K0 的 `preview_select` 决定，未选中块留在账本里（tier=Host）
而不是被遗忘，将来可以被重新选中。

## 4. K2：K 相位重铸（已实现）

- 驻留块 `baked_pos == to`：零成本跳过（多数选择轮的大多数块）。
- 移动块：原地 de-rotate(from) + re-rotate(to)。int8-group64 KV 的完整链：
  反量化 → 去 Hadamard(H64) → 去 RoPE(from) → RoPE(to) → Hadamard → 再量化。
  落位 `src/kvmem/kvmem_rerope.cuh`（每 token 每头一次），**不改任何现有内核**。
  MRoPE rotary_dim=64 ⇒ 只有 group 0 带位置相位，groups 1..3 逐位不动。
- 冷块（stage-in）：**尚未做 raw-K 镜像**。宿主存的是"旧相位页"而非未旋转 bf16 K，
  所以还原块走的是与驻留块同一条重铸链，代价是多一次 int8 量化。真 raw-K 重建
  仍未接线。
- 精度账本沿用 K0 的 remap_count/remap_abs_delta 双阈值；当前阈值恒 0（不限），
  所以原地重铸是常规路径而非省路。

## 5. 与 8GB 显存的现实约束

PQ2_0 权重 6.70 GiB，启动后仅剩 ~12 MiB —— **在 PQ2_0 上做 K1b 需要 cago 权重空间**：
要么降低 kv-capacity（现有 auto 会按剩余显存算）、要么用 PTQ1_0（5.52 GiB 权重，
1.0 GiB 可用于 KV+工作区，int8 KV 4096 token 仅 132 MiB）。
**建议 K1b 的性能对比用 PQ2_0、功能开发用 PTQ1_0**，与基线文档口径一致。

## 6. K1b serve 语义：窗口 + 窗口边缘锚点（已实现）

**问题**：`finish()` 处的轮末压实起初把 token 账本直接截到窗口长度，却没有同步序列的
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
   `kvmem_apply_window` **不截断账本**：`ledger`/`prefix_identity`/`prefix_digests` 连同
   `execution_frontier` 留在 prompt 空间（复用匹配、prompt 切片、endpoint 自身的 state
   都靠它们），只把 KV 侧的 `text_kv_valid`/`mtp_kv_valid`/`dflash_context_frontier` 与
   位置平移收到窗口长度，用一个 `kv_offset` 表达差值。同时**丢弃 endpoint**（连同其
   StateImage）——从此锚点是这条 continuation 唯一诚实的复原点。
3. 下一轮 `inspect_lane` 走 `PrivateLongAnchor`，base = 窗口长度。窗口**不是**连续头
   前缀：它是 sink 头 + 被重铸到紧凑槽位的尾部（见 §8），所以窗口前沿 = 尾段最后一个
   有效列，而不是打包后的页数——新 token 续写在最后一页尚未填满的列里，避免触发
   `rebuild_checkpoint_protection` 的 terminate。重铸由 K2 承担。
4. 该锚点**每轮重新捕获**：reuse 时以 `ConsumeToActive` 消费它（锚点 image 直接成为
   active state），于是在 `cursor == base` 处捕获不会落在未结算的 state fork 后面
   （引擎禁止未结算 fork 上开 capture transaction）。因此锚点自续，不需要跨轮继承。
5. tracker 账本跟随地址：resume/分叉带来的既有页在 `ensure_mapped_to_tokens` 里按
   born-placed 补登记；`destructive_truncate*` 走 `kvmem_forget_window_tail` 忘掉窗口外的
   块并交还宿主副本。

**验收**（RTX 3060 Ti / PTQ1_0 / int8 KV / `NINFER_KVMEM=1 NINFER_KVMEM_BUDGET=512`
`--max-shared-prefixes 0`）：三轮对话 `cache` = 1297/1328（97%），TTFT 1.2s；5 个 KVMem
单测 + 默认路径冒烟为回归基线。KVMem 关闭时默认路径同样复用整段 turn-closure 前缀
（97%，TTFT 1.2s）。

## 7. K3：查询信号（检索打分已实现）

**已实现**：窗口 middle quota 原来没有任何信号——所有块分数恒 0，配额退化成"新块优先"，
于是请求真正在问的事实只要落在上下文中段就会被丢掉。现在每个块按
"请求 query 与块 mean-K 的余弦"打分，接落 `KvBlockRepository::add_scores` →
`preview_select` 的 retrieval 配额。

- **必须比较在 raw 域**（去 RoPE 之前）。key 与位置无关，所以跨距离的块只有去掉相位
  才可比；这也是参考实现 harvest 的是 pre-RoPE K/Q 的原因。在已烘焙相位上直接算余弦，
  跨块会退化成近似正交——实测 19 个块分数全为 0。
- 缓存 K 是 int8-group64，所以打分链是：反量化 → 撤销 H256（对合）→ 在块实际烘焙的
  位置 un-RoPE → 与 raw Q 比余弦。落位 `src/kvmem/kvmem_score.cu`。
- Q 来源：prefill 时在 `ops::rope` **之前**按列捕获进一个启动期分配的 device 缓冲
  （decode graph 捕获之前装好，地址才不会漂）。

**验收**（与 §6 同一环境）：中段埋唯一事实的两轮问答，无信号时第二轮答不出来（中段块
不在窗口），有信号时答对；cache 与 TTFT 与无信号版完全一致。新增
`tests/kvmem_score_test.cu` 用真实页池 + CPU 模型逐位对照。

**仍未做**：

- query 只取 prompt 最后一个 token。参考实现是把**整个 query 区间**（本轮最后一条用户
  消息）的 Q 求和再平均。区间边界在 frontend 已经算好
  （`chat_template` 的 `message_boundaries` + `last_real_user_query`），只是在
  `frontend.cpp` 被丢弃、没进 `PreparedPromptData`；kernel 侧只需由覆写改成累加。
  真正的工件是区间落在复用前缀里时的情形：参考实现另有一套查询状态冻结/导入机制
  来跨请求复用上一轮的 Q，我们还没有。
- mean-K 在打分时由驻留 device 页现算，没有持久 mean-K store，所以已挤出到宿主 arena
  的块恒为 0 分（"无证据"）。参考实现保留每块每层的 mean-K，挤出块也有分数。
- 分数只在压缩时刻按当时的 query 算一次，不是每步重选。attention 热度（profile）信号
  仍未接入，`profile_score` 恒 0。

## 8. 窗口化 continuation 的契约扩展（已实现）

目标：窗口 = `[0, S)`（sink/头部）+ `[TB, T)`（最近尾部），T = 本轮 prompt+生成的结束。
中段 `[S, TB)` 从 KV 里彻底移除（既不是"保持原位置做稀疏注意力"，也不做零页稀释）。
尾部块由 K2 重铸到紧凑位置 `[S, S+Wr)`，于是窗口位置 = `window_tokens = S + Wr`，
新 token 的位置从 `window_tokens` 起算。

**落地方式：账本留在 prompt 空间，用一个标量 `kv_offset` 平移 KV 坐标。**

```
kv_tokens(offset, ledger) = ledger + offset        // 缓存槽位与 RoPE 位置
kv_offset = window_tokens - prompt_frontier        // 由 kvmem_apply_window 设定
```

| 量 | 实现落点 | 谁用 |
|---|---|---|
| prompt 账本（`ledger`/`prefix_identity`/`prefix_digests`/`execution_frontier`） | 保持 prompt 空间，不被压缩改写 | 复用匹配、prompt 切片、endpoint 自身的 state |
| `kv_offset` | `SequenceState`（序列级），prefill 位置与 KV 槽位均按它平移 | `text_kv_valid`、地址/pin 范围、`ensure_mapped_to_tokens`、RoPE 位置 |
| 窗口前沿 | `Address.committed_frontier` = 尾段最后一个**有效列** | 新 token 续写、`kv_reuse_floor` 换算 |
| resume 下限 | `SequenceState::kv_reuse_floor`（prompt 空间） | 拒绝会泄漏未来 token 的 checkpoint |

**设计预测被实现推翻的四处**（原设计不再适用）：

- **不需要 `WindowGeometry{S, TB, Wr}`**，也不需要让 `prefix_matches` 认识"跳段"。
  因为账本与 digest 留在 prompt 空间，`prefix_matches` 原样可用；窗口几何由地址自身
  （页数 + `committed_frontier`）加 `kv_offset` 隐含表达。原设计担心的"catalog 精确校验"
  改由 resume 下限承担。
- **`position_origin` 不是一个新字段**，就是 `kv_offset` 加到 `text_kv_base` 上
  （`TextContext` 的 `text_kv_offset_`），prefill 与生成段共用同一条平移。
- **`inspect_lane` 走的是 `PrivateLongAnchor`（§6 的窗口边缘锚点），不是 `PrivateEndpoint`**。
  endpoint 路径是通用形态；serve 实际命中的是长锚点，且该锚点用 `ConsumeToActive` 消费，
  以免在未结算的 state fork 上开 capture transaction。两条路径都带 `reuse_base = T` 与
  `kv_frontier`，这一点与原设计一致。
- **多了一条视觉护栏**：窗口化 continuation 在 `kv_frontier != reuse_base` 且 prompt 含
  vision items 时被直接拒绝——Vision 位置按轴绝对值给定，MRoPE 的绝对位置不是我们能平移的。

**验收**（与 §6 同一套环境）：第二轮 `cache` ≈ T（而非窗口长度）、窗口页数 =
`(S+Wr)/64`、回答质量接近全量注意力（尾部可见）；对比头窗口版本（中段与尾部都不可见）。
