# KVMem K0+K1 设计：宿主块仓库与窗口组装（zatfung）

状态：K0 实现中；K1 为下一步。基线见 `perf-baseline-sm86.md`；机制评估见
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
