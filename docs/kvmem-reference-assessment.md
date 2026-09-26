# KVMem → zatfung 参考价值评估

> 评估对象：`G:/Agents/ninfer-works/laamaafung`（自有推理引擎，KVMem 已在 v17 → v26 多分支移植）
> 评估目标：`G:/Agents/ninfer-works/zatfung`
> 结论先行：**参考价值很高，但价值在设计层面，不在代码层面** —— 两者底层抽象不同
> （llama.cpp/ggml vs NInfer 自有 Tensor/paged-KV），不能直接搬文件；但 KVMem 的
> **机制、数据结构、精度策略和一堆实战参数**正是 zatfung 缺失的那一层。

---

## 1. KVMem 的真实机制（以 laamaafung 源码为准）

> 这一节比 `kvmem-llama.cpp` 的 README 更精确 —— README 说 "retrieves blocks …
> 放置到有界的 GPU 工作集"，但源码里的实现比这细致得多。

### 1.1 位置压缩，而不只是"挑块"

关键数据结构 `KvMemBlock`（`kvmem/include/kvmem/kvmem_store.hpp`）：

| 字段 | 作用 |
|---|---|
| `orig_pos_start` / `n_tokens` | 块在**原始序列**中的位置与长度（默认 128 token） |
| `baked_pos` | 该块的 K **当前被烘焙到的窗口位置**；prefill 后 == `orig_pos_start` |
| `remap_count` / `remap_abs_delta` | 非 no-op 的原地 re-RoPE 次数 / 累积位移量 |
| `attn_score` / `profile_score` / `retrieval_score` | 三层选择信号（累积注意力 / 窗口局部热度 / 全局检索分） |
| `tier` / `gpu_slot` / `cpu_slot` / `nvme_slot` / `ssd_clean` / `dirty_gpu` / `in_flight` | 分层与生命周期 |

**no-copy 设计**：源码注释写得很直白 —— *"the stored cache IS the repository"*。
GPU cache 本身就是仓库，选中块**原地重排**到窗口 `[0, total_window_tokens)`，
并把 query 位置设为 `total_window_tokens`。**不是"保持原始位置做稀疏注意力"** ——
这与我最初的理解不同，位置是被**压缩**的，所以必须 re-RoPE。

### 1.2 re-RoPE 的精度工程（最有价值的部分）

`KvMemRemap` 的设计揭示了真正的难点：

```
de-rotate(baked_pos) + re-rotate(new)   // rope_block_remap_paged，原地
```

于是产生一个累积误差问题，源码注释把它写透了：

> *"With fp16 KV, however, every non-noop remap rounds the result again; a long multi-turn
> trace can therefore accumulate error."*

对策是**双阈值触发的 raw-K 重建**：

- 保留一份**未旋转的 CPU raw-K 镜像**（`raw_kv_store.cpp`，1179 行）
- `remap_count` 或 `remap_abs_delta` 超过阈值 → 置 `raw_refresh = true`，
  从 CPU 镜像**重新烘焙**，而不是再叠加一次有损旋转
- 冷 stage-in 的块**必须**走 raw-K 重建（`raw_refresh` 恒为 true）

> ⚠️ 注意 `raw_kv_store.hpp` 顶部注释里的坑：生产路径其实**不需要**未旋转 K
> （"Restore is memcpy; orig pos does not need unrotated K"），raw K 是
> **immutable-source 模式下的刷新源**。别被这句注释误导而以为可以省掉它 ——
> 一旦开启位置压缩，它就是精度兜底。

### 1.3 嫁接点：最终产物是一个 page-index 列表

`KvMemPlan` 的输出形态：

```cpp
struct KvMemPlan {
    std::vector<uint32_t>  stage_in;   // 新进入工作集的块
    std::vector<uint32_t>  stage_out;  // 不再选中、需换出的块
    std::vector<KvMemRemap> remaps;    // 按窗口序的完整重排计划
    uint32_t total_window_tokens;
    uint32_t selection_overlap_blocks; // 与现有工作集的重叠
    uint32_t gpu_reused_blocks;        // GPU 页仍驻留、可免物理传输
};
```

注释点明了 executor 的用法：**"assemble the kernel page-index list from these blocks
in this order"**。

---

## 2. 对 zatfung 的适配性

### 2.1 好消息：结构与 zatfung 同构

| KVMem 概念 | zatfung 对应物 | 评价 |
|---|---|---|
| `baked_pos`（KV 里是已旋转 K） | `ops::rope()` 是 **in-place** 施加到 K 再写 cache（`include/ninfer/ops/rope.h`：*"mutates only dimensions [0,rotary_dim)"*） | ✅ 模型一致，"baked" 概念可直接借用 |
| executor 组装的 **page-index list** | zatfung 的 paged KV 本就按 `block_table[logical_page] → physical_page` 驱动 | ✅ **天然嫁接点** |
| `KvTier{GPU,CPU,SSD}` | `host_kv_arena` + `HostKVExtentStore` + `LogicalKVPageStore`/`KVAddressSpaceStore` | ✅ 主机 tier 的存储与传输规划已存在 |
| 块的生命周期/副本 | `HostKVPageReplica`（含 `content_epoch`、`membership_node`）、`KVPrefixForkReservation` | ✅ 甚至比 KVMem 的 `ssd_clean/dirty_gpu` 更细 |
| ReplaySSM | `gdn_replay_records` + `linear_attention_state` | ✅ 齐备（术语都同源） |

### 2.2 坏消息：三处真实复杂度，KVMem 没覆盖

**① zatfung 的 RoPE 是多轴 MRoPE，不是 1-D NeoX**

`rope.h` 明确列了三种域：

- Text 1-D：`positions [T]`
- **Text MRoPE：`positions [T,3]`，pair i 用 axis `i%3`**
- Vision 2-D：`positions [T,2]`

而 `kvmem/rope.hpp` 只有一个 1-D 配置（`n_rot/n_embd_head/freq_base`，注释写
*"Matches ggml GGML_ROPE_TYPE_NEOX"*）。**re-RoPE 必须按 axis 分别 de-rotate/re-rotate**，
这是 KVMem 的 47 行 `rope.cpp` 完全没考虑的情形。

**② zatfung 的 KV 量化自带 Hadamard 旋转，与位置 RoPE 是两层独立的旋转**

- `kv_cache/int8_g64_codec.cuh`：`rk8v4/rk4v4/rk4v4-e8/rk2v4-e8` 用 **H64 变换**按 64 组旋转
- `kv_cache/append/k8v4_kernel.cuh`：固定 **FP32 D256 Hadamard rotation**

这层旋转是**量化用的**（与 token 位置无关）。做 remap 时必须分清：

```
位置域旋转（NeoX/MRoPE，要重做）
   ↕  两者不可混淆
量化域旋转（H64 / D256 Hadamard，remap 时不能动）
```

KVMem 的 `q8_0/q4_0` 没有这层，所以它的设计里没有这个区分。
**这是 zatfung 移植时最容易写错的地方。**

**③ 量化 KV 上的 remap 误差比 fp16 更严重**

KVMem 是 fp16 KV，重旋转一次舍入一次，靠 raw-K 刷新兜底。
zatfung 的 KV 是 int8/fp8 量化格式 —— 如果 re-RoPE 走
`反量化 → 重旋转 → 再量化`，每轮多一次量化误差，比 fp16 的舍入恶劣。
**必须直接复用 KVMem 的"阈值触发 raw-K 重建"作为主路径**，
而不是把原地重旋转当常规手段。

### 2.3 不可复用部分

- `src/adapter/llama-memory-kvmem*.cpp` —— 接的是 llama.cpp 的 `llama_memory` 接口
- `kvmem/` 的存储/量化实现（`block_q8_0`/`block_q4_0` 等 ggml 格式）
- `rope.cpp` 的 1-D NeoX 实现（zatfung 要用自己的 `ops::rope` 语义重写）

**可复用的是设计，不是代码。**

---

## 3. 值得直接借鉴的清单（按价值排序）

| # | 借鉴点 | 参考位置 |
|---|---|---|
| 1 | **原地 remap + 双阈值 raw-K 重建**（精度兜底的核心） | `kvmem/include/kvmem/kvmem_store.hpp` 的 `KvMemBlock`/`KvMemRemap` 注释 |
| 2 | **三层选择信号**：`attn_score`（#40 内置累计注意力 top-k）、`profile_score`、`retrieval_score` | `KvMemBlock` 字段 |
| 3 | **`KvMemPlan` 的 diff 结构**（stage_in/out + remaps + overlap + gpu_reused） | 同上 |
| 4 | **inclusive / exclusive tier 模式**（是否保留低层干净副本） | `KvMemBlock::ssd_clean` 注释 |
| 5 | **"冷块必须从 raw-K 重建，即使位置相同"** | `KvMemRemap::raw_refresh` 注释 |
| 6 | `working_k_resident` 与 `stage_in` **刻意区分**（前者物理驻留，后者逻辑进出） | `KvMemRemap` 注释 |

### 3.1 从 git 历史学到的实战参数（省掉反复试错）

这些都是 laamaafung 在多个版本里踩出来的，直接看提交信息即可：

| 提交 | 经验 |
|---|---|
| `21b7f6819` | `--kvmem-gen-reserve` 默认值 **256 → 8192**（256 是测试值，不是生产值） |
| `f66c4b109` | `--kvmem-gpu-ratio` 统一为 **0.50**（早期提交曾用 0.90，现已过时 —— 以此为准） |
| `12f996318` | **gen-reserve 环形缓冲换出**：`alloc_slot` 耗尽时逐块落盘 GPU 块；gpu-ratio 默认 0.50 与官版对齐 |
| `8ff6d8389` | **启动前检（pre-flight）**：显存不足时拒绝启动并给出参数建议 |
| `0ce856fd2` | **Q 捕获只 pin query 行**，并加捕获视图守卫 |
| `b4882eb06` | 图像超预算时**在 token 化之前**自动缩小（对齐 DeepSeek Harness 归一化标准） |
| `bd43d67bc` | 首次移植进 v21 的入口提交 —— 判断移植工作量时可作为对照 |

> 特别提醒第 3 条：`gen-reserve 环形缓冲换出` 说明"预留的生成槽位耗尽"是
> 真实会发生的边界情况（README 也承认单次生成不能超过 `--kvmem-gen-reserve`）。
> zatfung 若照搬这套机制，**必须同时实现这个换出路径**，否则长回答会直接失败。

---

## 4. 建议的移植路线（分层，由易到难）

| 阶段 | 内容 | 依赖 | 验收 |
|---|---|---|---|
| **K0** | 宿主侧块仓库 + 选择算法移植（纯逻辑，可单测，不碰 GPU） | 无 | 单元测试：给定伪造的注意力分数，选出正确块集 |
| **K1** | 选择结果 → zatfung `block_table` 组装（"live page-index list"） | K0 + paged KV | 用固定块集跑通一次推理，**保持原始位置**（不做压缩） |
| **K2** | 精度保底：raw-K 镜像捕获 + 双阈值重建策略 | K1 | 多轮对话下与"全量 KV"的输出差异在阈值内 |
| **K3** | **位置压缩 + 原地 re-RoPE**（真正的难点） | K2 + MRoPE 支持 | 窗口位置紧凑化后 ppl/召回不退化 |
| **K4** | NVMe tier（可选） | K3 | 参照 `nvme_kv_tier.hpp`（895 行）；注意 laamaafung 侧 NVMe 也不完整 |

**关键判断**：**K1 单独就有价值**。因为 zatfung 已有 paged KV + 前缀复用，
只做"选择哪些块参与注意力、保持原始位置"就能拿到大部分收益（等价于
KVMem 论文里"32K 工作集 ≈ 256K 全量"的那部分效果），**且完全不需要动内核**。
K3 的位置压缩是锦上添花，但它是内核级改动 + MRoPE 适配，风险与成本都高一个量级。

> 建议：**先做 K0+K1，验证召回与速度**，再决定是否投入 K3。

---

## 5. 与已有文档的关系

- 本文回答"KVMem 有无参考价值"
- `docs/zatfung-sm86-sm75-port.md` 回答"SM86/SM75 架构移植"
- 两者的交汇点：**K3 的 re-RoPE 内核需要按架构分别实现**（sm_75 无 cp.async、
  sm_86 的 smem 档位），且要绕开为 sm_75/sm_86 剔除的 FP8 路径
