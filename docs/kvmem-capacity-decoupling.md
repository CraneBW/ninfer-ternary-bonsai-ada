# KVMem 容量解耦设计：令 CTX 脱离显存上限（zatfung）

状态：第一、二步已落地（第二步只剩生成期回收），第三步未实现。目标分支 vD（从 vC 派生）。
前置：K0/K1a/K1b/K2 已实现、K3 检索打分已接入（见 `kvmem-k0-k1-design.md`）。

## 1. 要解决的问题

KVMem 的核心价值是「**用一个固定大小的显存区域反复复用，让 CTX 可以远超显存**」，
从而在小显存卡上跑长上下文而不掉速。参考实现（laamaafung）的做法是：

    GPU attention cache size = min(n_ctx, budget + gen_reserve)

即 GPU 缓存大小与 `n_ctx` **解耦**：`n_ctx` 只是逻辑上限，其余 KV 住在 host（raw-K 镜像 +
pinned/NVMe 层），只有工作集上屏。实测配方：9B/4.88 GiB 模型，CTX 256K + 工作集 32K +
生成预留 16K。

**zatfung 目前没有这个能力**：`--kv-capacity` 被强制 ≥ `--max-context`，所以 device 页池
必须装得下整个 CTX。用 int8 KV 实测（`--max-context 4096 --kv-dtype int8` → 64 页、
runtime 520.8 MiB，约 8.1 MiB/页 ≈ 127 KiB/token），CTX 256K 需要约 4096 页 ≈ **32.6 GiB
显存**，8 GB 卡上不可能。

结果：zatfung 现时从 KVMem 拿到的只有**速度**收益（窗口化续写、97% 前缀复用、TTFT 34.6s →
1.2s），**没有显存**收益。而显存收益才是 KVMem 的卖点。

## 2. 现状盘点（已核实的接入面）

好消息：**逻辑／物理页分离的骨架已经在**，缺的只是约束层与溢出路径。

| 环节 | 现状 | 证据 |
|---|---|---|
| 逻辑容量 | `capacity = max_context`，与物理页数**无关** | `layouts_impl.h:807`、`layouts_impl.h:733` |
| 物理页数 | `main_page_groups`，来自 `--kv-capacity` | `layouts_impl.h:734`、`kv_capacity.cpp:76-131` |
| 布局层 | `logical_pages = page_count(max_context)`、`physical_pages = main_page_groups`，两者独立传入 | `layouts_impl.h:104-105` |
| 逻辑页上限 | `page_capacity_ = logical_page_capacity = page_count(max_context)` | `program_impl.h:840-842` |
| 激活预留 | **一次性预留整个 entitlement**（`growth = entitlement - required_pages`） | `logical_kv_store.h:950-957` |
| 增长超限 | 抛 `invalid_argument`（不是换出） | `logical_kv_store.h:1485-1492` |
| 预留失败 | `resize_reservation` 抛 `bad_alloc` | `paged_kv_cache.cpp:283` |
| 压力换出机制 | **已有** `PressureKVDecisionKind::DemoteToHost` / `DropDeviceDuplicate` | `program.h:128-141`、`program_impl.h:452-503` |
| 但活跃页被排除 | 换出要求 `!addresses.has_active_reference(logical)` | `program_impl.h:464-467` |
| KVMem host arena | `capacity = page_stride * page_capacity_ + 1MiB`，**上限仍是逻辑页数** | `logical_kv_store.h:2230-2235` |
| 窗口压缩 | `kvmem_compact` 已做 select → stage_out（停 host）→ stage_in → 重铸 | `logical_kv_store.h:2117-2340` |

**真正的枷锁只有三处校验：**

1. `serve_options.cpp:363-366` — `kv_capacity >= max_context`
2. `layouts_impl.h:676-678` — 同一条（第二次）
3. `layouts_impl.h:668` / `:821` — `minimum_pages = max(logical_pages, max_concurrency)`

把物理页下限从 `logical_pages` 解掉，池就可以小于 CTX。**但只解校验会立刻撞到预留模型**：
激活时 `growth = entitlement - required_pages`，而长 prompt 的 entitlement 可以到 3000+ 页，
远超声明的物理页 → `resize_reservation` 抛 `bad_alloc`。

## 3. 目标与非目标

**目标**：`--max-context` 可以远超 `--kv-capacity`，显存占用由 `--kv-capacity` 决定；
超出工作集的 KV 停在本机 host arena，可按需换回。

**非目标（本轮不做）**：

- 不照抄 laamaafung 的 ring。ring 解决的是「生成期把预留耗尽」这个**症状**；本设计的
  第一步到第三步落地后症状才会出现，届时再按实测决定（见 §5 第四步）。
- 不做 NVMe 层（`nvme_kv_tier` 在参考实现里也是独立一层）。
- 不改 MRoPE 位置语义。视觉 prompt 已经在 `request_plan_impl.h:586-588` 被排除在窗口化
  续写之外，本设计沿用同一条护栏。

## 4. 要改的契约（按步拆分）

### 第一步：容量校验与预算解耦（已实现，仅到启动与单轮）

- 物理页下限改为「**每个并发序列的工作集 + 生成余量**」，不再是 `logical_pages`。
  新增 `--kv-device-tokens N` 表达上屏预算，`--kv-capacity` 语义与既有约束不变。
- 去掉 `kv_capacity >= max_context` 校验，改为 `--kv-device-tokens` 走独立分支。
- 逻辑容量 `capacity`／`page_capacity_` 保持 `max_context` 不变。
- 新增启动期校验：页面存储的描述符数（设备页 + host 页）必须覆盖逻辑地址空间，
  否则报出「需要多少 host 页」而不是在深层物化时才失败。

**实测**（`--max-context 8192 --kv-device-tokens 2048`）：

- 启动成功，容量行显示 `KV 2,048 tokens | pages 32/128` —— **逻辑 128 页、物理 32 页，
  解耦成立**。
- 单轮问答正常。
- 默认路径零回归：不加新选项时仍是 `pages 64/64`、cache 97.2%、TTFT 1.2s、答案一致；
  5 个 KVMem 单测全过。

**仍未通过：第二轮。** 报错是准入层的
`isolated-feasible request is blocked in an idle Engine`
（`engine_core.h:1710`）：资源巡检认为该请求在**空闲**引擎中仍不可行。原因是计划层仍然按
「池要装得下整个 prompt」计算物理需求——`text_kv_page_entitlement =
pages_for_tokens(prompt_tokens + effective_output - 1)` 与 `prepare_activation` 的
一次性 `growth = entitlement - required_pages`（见第二步）。所以第一步是**使能改动**，
真正的可用性要等第二步把「按需预留 + 耗尽时换出」落地。

**风险**：低。新选项独立，默认路径逐位不变。

### 第二步：物理页耗尽时换出，而非抛错

- 激活与增长不再预留整个 entitlement，改为预留「当前执行前沿 + 工作余量」。
- `ensure_mapped_to_tokens` 在物理页不足时走 **已有的** `PressureKVDecision` 路径，
  把非工作集驻留页停到 host，而不是抛 `invalid_argument`。
- 复用 `kvmem_compact` 已有的 stage_out/stage_in/重铸三段，让它在需要时被触发，
  而不是只在 `finish`／轮边界跑。

**验收**：先单测验证「换出 → 换回」闭环后 KV 逐位一致（沿用 `tests/kvmem_compact_test.cu`
的 CPU 对照模型）；再跑真实 serve，`--max-context` 远超 `--kv-capacity` 时多轮对话不崩、
答案与物理充足时一致。

**风险**：中。引入「页可能不在 device」的运行态，所有读页路径都要能处理。

**已落地部分（prefill 侧）**：记账把「逻辑 entitlement」与「device 声索」分开（后者取
`min(entitlement, 池页数)`，差额记 Host KV 字节）；prefill 不再一次映射整个 prompt，改为按
块（piece）映射，并在每块之前判断「该块末端需要的页数是否超过池」，超了就调用现成的
`kvmem_compact` 把窗口外的页停到 host（保留窗口 = 池 − 本块页数，因此块上限 =
池 − 最小窗口，且尽量维持 `--prefill-chunk`）。换出后补回增长预留。实测 `--max-context 8192`
下池 20 页与 10 页两种配置三轮对话均通过（池 10 页时单轮 prefill 内换出 3 次），缓存命中
97.2%/97.1%，答案与物理充足时一致。

**未落地**：生成期回收。单请求的输出若长到超出池的剩余空间，仍会失败——需要第四步。

### 第三步：工作集固定

- 给块加「已固定」标记：sink、recent、以及检索命中（K3 已实现打分）的块不可换出。
- 放松 `has_active_reference` 保护：改为「不在固定工作集内」才允许换出。
- 参考 laamaafung 的约束精神（跳过 pinned 检索集、跳过未写满的尾块），但按其自身语义重写。

**验收**：中段唯一事实的两轮问答应答对（沿用 K3 的 needle probe）；被换出的块不得是
sink/recent/检索命中块。

**风险**：中高——这一步改变**注意力可见性语义**：中段会在同一轮内就不可见，而不只是跨轮。
这是质量权衡，需要用户确认可接受。

### 第四步：（暂缓）生成期回收

只有当第一至第三步落地后，实测出现「单请求生成长过预留 → 失败」时再做。
届时可参考 laamaafung 的 `reclaim_generation_slot`（`llama-memory-kvmem.cpp:1006-1051`
与调用点 `:1671-1685`）的约束形式。

## 5. 顺序与回退

每一步一个提交，独立可回退。每步完成后跑：

- 5 个 KVMem 单测（`tests/kvmem_{blocks,rerope,executor,compact,score}_test.*`）
- 关闭 KVMem 的默认路径冒烟（确认无回归）
- 大 CTX 冒烟（`--max-context 8192 --kv-capacity 2048`）

## 7. host 预算：实测口径与更正

早前把「256K 需要 33 GiB」当作 host 的硬性要求。**这个数字是错的**：那是把
device 页池要装下整个 context 时的显存需求，误当成了 host 侧开销。

参考实现在本机（RTX 3060 Ti 8 GB + **31.8 GB RAM**，见其 `perf-tests`）跑通 256K 的配置：

| 模型 | 参数 | 显存峰值 |
|---|---|---|
| 27B | `-c 262144 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048` | 6956 MiB |
| 35B | `-c 262144 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 16384` | 6101–6207 MiB |

它做得到，靠三件事：

1. **KV 激进量化**：K 用 `q8_0`（≈1.06 B/权重），V 用 `turbo4`（4 bit）。zatfung 目前的
   int8 把 V 也存 int8，V 这一半的开销约是它的两倍。
2. **淘汰写的是打包后的量化 K/V**，不是 raw fp16/bf16（参考实现的设计记录：stage-out
   D2H 的是 packed K 与 V）。所以 host 副本与设备副本**同尺寸**，不会膨胀。
3. **host 只需装下整个 context 的量化 KV**，对这些模型是数 GB 级，31.8 GB 内存绰绰有余。

按 zatfung 现在的表示（27B：16 层 full attention、4 个 KV head、head_dim 256、K/V 均 int8
＋每 64 维一个 fp16 scale）解析估算：每 token 每层约 2112 B，16 层约 33.8 KiB/token，
**256K ≈ 8.6 GiB host**——仍然可行。注意 zatfung 现有的「127 KiB/token」实测值包含了
KV 平面之外的分配，不宜直接外推；落实 host 预算前应当先量出纯平面的边际字节数。

**配额口径**：参考实现的 KVMem 溢出盘**独立于前缀缓存**，有自己的开关
（`--kvmem-cpu-gb` → `cpu_bytes`、`--kvmem-nvme-gb` → `nvme_bytes`、
`--kvmem-raw-k-nvme`）。所以本设计第 8 节第 3 条「单独配额」与参考实现一致。

注意本机 RAM 是共享的：参考实现的 27B/35B 测试同时把 MoE 专家放在 CPU
（`--n-cpu-moe`），所以 host 预算必须在「CPU 常驻权重」与「KV 溢出盘」之间分。

## 8. 已确认的决策

1. **CLI 形态**：新增独立选项表达上屏预算（`--kv-capacity` 语义与既有约束保持不变），
   既有配置不需要迁移。
2. **质量权衡**：分期做。先落地跨轮收益（池只需装「压缩后的窗口 + 本轮新增 + 生成余量」），
   实测后再决定是否加「单轮内也换出中段」，后者作为独立提交。
3. **host arena 配额**：给 KVMem 淘汰块单独一个配额，不与已缓存前缀共用 `--host-kv-mib`。

因此本设计的落地顺序调整为：第一步（容量校验与预算解耦）→ 第二步（物理页耗尽时换出）→
第三步（工作集固定，服务跨轮收益）→ 若实测需要，再加「单轮内换出」与「生成期回收」。