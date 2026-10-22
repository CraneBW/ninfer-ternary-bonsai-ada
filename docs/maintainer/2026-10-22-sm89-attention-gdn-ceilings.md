# OpenSpec 5.1：SM89 attention / GDN 有界理论上限（2026-10-22）

**状态：THEORY_ONLY / SOURCE_MAPPED / PERFORMANCE_NOT_MEASURED。** 这是 [V100X2 固定来源与许可映射](2026-10-22-v100x2-source-map.md) 的后续独立估算，只使用已核对的 [tuxKOH/ninfer-V100X2@5c51513096a9fceb9bb1609e43b15efecac30f93](https://github.com/tuxKOH/ninfer-V100X2/tree/5c51513096a9fceb9bb1609e43b15efecac30f93)、本机已存在的源码和 **200W** 收据。本轮没有编译、GPU、Profiler、候选实现、性能或质量新实测；公式中的“上限”指已列条件下的理论可移除工作，不是实际优化收益或允诺百分比。只考察单卡 RTX 4070 Ti SUPER / SM89、现役 Qwen3.8 GSQ、完整上下文、原 INT8-G64 KV 与可见数值边界。

## 可用的 200W 时间证据与几何

既有 [200W 功耗/整模型配对报告](../performance/2026-10-21-power-cap.md) 与 [原始记录](../performance/2026-10-21-power-cap.json) 用相同 GSQ、INT8 KV、chunk512、C1、Graph；每进程一次 warmup/一次 measured run，两档各三对，阶段值不包含 load/prime/warmup：

| 现有 200W 参考档 | Prefill tok/s（中位） | Decode tok/s（中位） | 用途 |
|---|---:|---:|---|
| pp4096/tg128、ctx8192、MTP7（verify 上限 T8） | 1554.61 | 172.55 | 短档整体参考；不能得出 QK/GDN kernel 占比 |
| pp32768/tg128、ctx36864、MTP15（verify 上限 T16） | 1431.27 | 197.57 | 长档整体参考；T16 会经现有 ChunkedSmallT 分段，不能当一个 T16 small-T CTA |

同一报告的 300W 数据仅用于该报告原有功耗配对，本估算**不把 300W kernel 归因移作 200W**。该表仅有阶段吞吐，**没有这三项 kernel 在 200W、这两个 profile 上的时间份额**。曾有早期 Nsight 显示 MTP5 decode 的 GGUF vec 占 kernel sum 87.438%、prefill MMQ 占 86.956%（[已有 R12 记录](2026-10-20-kvmem-actkv-live-validation.md)），但其 binary、窗口和功耗资格不同，故不当作本轮的热点系数。

本机静态几何来自 `src/targets/qwen3_6_27b/impl/config.h:56–74,126–128`，且 Qwen3.8 仍复用该 family/27B 实现：64 主层中 **16 attention / 48 GDN**；attention `Hq=24, Hkv=4, D=256, GroupSize=6`；GDN `Hqk=16, Hv=48, Dg=128`，每 QK head 映射三个 value heads。实际入口为 `src/ops/softmax_attention/dense/causal_cache/{causal_softmax_attention.cpp,small_t.cu,small_t.cuh,small_t_i8.cuh}` 和 `src/ops/linear_attention/gated_delta_net/{gated_delta_net.cpp,recurrent.cu,recurrent.cuh,chunked/launch.cu}`。上述 16/48 是主模型层数，不能简单乘作含 MTP 全部真实 kernel 调用次数。

## Amdahl 计算规则（未知时间份额保留变量）

固定同一 measured phase（prefill 或 decode），定义 `f_i = t_i / t_phase`，其中 `t_i` 是基线该阶段内**互不重叠**且确实受候选影响的工作耗时，`s_i = t_i/t_i'` 是其独立加速比，`e` 是新增准备、同步、缓存与调度成本除以原阶段耗时，则：

`S_phase = 1 / (1 - Σ[f_i × (1 - 1/s_i)] + e)`；对单项、忽略新增成本的绝对理想上限是 `S_phase ≤ 1/(1-f_i)`，对应节省阶段时间比例 `≤ f_i`。如果只删掉某组件原工作 `1-1/R` 的重复次数，则 `S_phase ≤ 1/[1-f_i(1-1/R)]`（忽略其他不可删的计算与新成本）。同一 partial kernel 内 QK/softmax、V load、PV 不能将 kernel 全时长分别算作多个 `f_i`，合并只取时间并集，不能相加三份猜测收益。

**本轮 `f_QK`、`f_reducer`、`f_GDN-norm`、`s_i`、`e` 均无可对应的当前 200W 收据**，所以没有可证的整模型提速百分比。以已知 200W tok/s 代入 `f` 或假定全部 QK 可删均无根据。

## A. QK / softmax 跨 PV warps 复用：源端思路，本机该重复项已消除

固定 donor 的 [INT8 Volta TC kernel](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh) 与 [launcher](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/launcher/gqa_attention_decode.cu) 将 QK score/softmax 供不同 PV 维度 warp 复用。本机的 **SM89** `small_t_i8.cuh:571–775` 也已明确让 `warp < RowTiles` 的 producer 对每个 `(query row,key tile)` **计算一次** QK、online softmax，写 `p_s`/ `alpha_s`；各 `consumer_slice` 读取它们来做 PV。因此此种“原先每个 PV warp 都重新算一遍”的当前重复因子 **`R_QK,duplicate = 1`**，可删的这部分算术为 **0**。存在的数据搬运、V 解码、shared 概率读和在线归约仍可能优化，但需另测，不计入“QK 重复消除”。

同一个 `T=8, K=32768`、每个 attention 层的**完整可见 K 上界**：需评估的 query-key score 至多 `Hq×T×K = 6,291,456`，QK `D=256` 点积即 `1,610,612,736` 个乘累加对（粗略 `3,221,225,472` 算术操作等价；生产实际是 INT8 IMMA + FP32 scale，**不是 FP32 FLOPS 实测**）。这些独立 query/head/key 点积本身不能因 PV warp 共享而消失。若未来实证某新路径在相同 `(head,query,key)` 真重复 `R>1` 次，可删重复部分上界 `1-1/R`，再仅对该片段的实测 `f` 使用上式；**当前生产路径不能代入 `R=2/4/6` 领取收益**。

按真实 INT8-G64 编码，每个 **KV head、每个 key** 的最低逻辑有效 cache payload 为 `256B K-code + 256B V-code + (4+4)×2B FP16 scales = 528B`，四个 KV heads 即 **2112B/key**，`K=4096` 为 **8,650,752B（8.25MiB）/attention 层**，`K=32768` 为 **69,206,016B（66MiB）/attention 层**。它只说明唯一 KV 的代码/尺度字节；不同 query chunks、split CTAs、页表、读前和 L2 命中可重复读取，**不是实际 DRAM bytes 或完整 Op traffic**。新方案如不改 codec/页布局，也不能将这份完整历史下限消成零。

**SM89 producer/consumer 资源风险（源码的静态数组算术，非 ptxas/NCU）：** `T8` 时 `RowTiles=ceil((8×6)/16)=3`。`K>8198` 的 launcher 选 **6 warps / CTA、Bc32、MinBlocksPerSm=2**（`small_t.cu:178–189`）；静态 shared 为 `q_s 48×256B + r_s 4×32×256B + p_s 48×32×2B + alpha_s 48×4B + 双 scale 2×32×4×2B = 48,832B/CTA`。两个 CTA 需 **97,664B/SM**，尚未加任何新 staging。若实际 per-SM 可用 shared 预算为 `S_SM`，欲保住两个 CTA，新增 shared 须满足 `ΔS ≤ floor(S_SM/2)-48,832`；**仅假设 `S_SM=102,400B` 时为 2,368B/CTA**，真实 carveout/分配粒度须以后续查询为准。PV consumer 在这档 `PVNtPerWarp=256/((6/3)×8)=16`，源码 `float acc[16][4]` 至少 **64 个 FP32 accumulator 元素/consumer thread**，尚有 QK fragments、地址与其他临时量；新增 score/register 缓存可能 spill 或降驻留，寄存器数、spill bytes、occupancy 当前 **未知**。

比较小窗口 `2054<K≤8198` 的 T8 launch 为 **12 warps、Bc64、动态 arena**：仅本 kernel 声明的静态+动态 shared 总计 **85,200B/CTA**（动态 `4×64×256=65,536B`）；`float acc[8][4]` 为至少 32 个 accumulator/thread，配置目标为一 CTA/SM。两种窗口显存、warp 数与计算/共享压力不同，不能把 donor **SM70 QPN/mma.sync.m8n8k4** 或其 12-Q-head TP2 tile 原样用于 SM89。

## B. Split-vector reducer：逻辑 global 字节下限不变，风险在 staging/同步

本机 `small_t.cuh:169–197,200–269` 已在**每 CTA** 把 `partial_m/l` 合成一份 `weights`；对 `Hq=24`，`small_t.cu:512–525` 设置 **`DChunk=256`**，每 `(qhead,token)` 一 CTA，以 `split=0..S-1` 顺序载入 FP32 `partial_acc` 并累加。固定 donor [reducer](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/kernel/gqa_attention_decode.cuh#L220-L350) 用 shared `values` 协同预读 8 个（某 profile 为 16 个）split，保持升序相加；但 donor 的 `partial_acc` 是 **BF16**，本机是 **FP32**，不能复制 cast/中间舍入。

设 `N=Hq×T`，`D=256`，`S=active_split_count`（依实际 `window`、dtype、T、launch capacity；不是 `K/64` 固定），则本机**一个 small-T 调用/一个 attention 层**：

| 逻辑数据/内存级 | 字节公式（没有其他压缩） | 条件实例 `T8, K32768, S69, Hq24` |
|---|---|---:|
| FP32 partial 向量 `partial_acc` | `4×N×S×D` | 13,565,952B |
| FP32 `partial_m + partial_l` | `8×N×S` | 105,984B |
| Producer 至少写一次上述 partial，reducer 至少读一次 | `2×N×S×(4D+8)` | **27,343,872B** |
| Reducer 最终 BF16 output | `2×N×D` | 98,304B |
| 上述两阶段合计必需逻辑访问 | `2×N×S×(4D+8)+2×N×D` | **27,442,176B（约 26.17MiB）** |

这里 `S69=ceil(32768/480)` 源自当前 `small_t.cuh:81–124` 在该长窗的 INT8 分段规则，前提为 launch capacity 足够且不计当前新 token 对 K 的微小增量；若取 T16 必须对**实际两个 T8 chunk** 分别计数。上述是生产 kernel 的**逻辑 global loads/stores（和必须可寻址的中间结果）**，不是 200W DRAM 测量或峰值 scratch：实际工作区按 **launch capacity** 预留，可能大于活跃 `S`，L2/事务合并也改变外部带宽。

仅重排 split 向量的 cooperative staging **不减少这份 FP32 partial 最低 global 读写字节**，所以“纯靠减少必需 bytes 的带宽收益”在这项候选下为 **0**；可能争取的是内存访问并发、延迟隐藏和不同 reducer 并行度，需以 `f_reducer`、等待/访存 stalls 和 clean Op 时间证实。当前一个 CTA 的明确静态 shared 为 `weights[256] + warp_sums[8] + scalars[2]`，即 **1064B**。若保留 `DChunk256`、用 `SplitTile8` 暂存所有 `float` 维度，需另 **`8×256×4=8192B/CTA`**，合计至少 **9256B/CTA**（不含新对齐/编译器额外资源），`S69` 时需 `ceil(69/8)=9` 轮；donor 的每轮写读 staging 还加两次 `__syncthreads()`，不能默认为免费。若改 `DChunk32`，暂存仅 1024B，但相同 D256 要**8 个 CTA**，`m/l` 和指数权重会被八组 CTA 重算/重读；仅统计量 `8NS` 的逻辑读取便由 **105,984B→847,872B**，新增 **741,888B**（缓存命中未知），全 kernel CTA/grid/调度也改变。两档是否有任何正收益均**未知**。

于是可证的 speedup 边界只有：若实测 reducer 占目标阶段 `f_reducer`，且独立新实现让 reducer 节省其时间的 `β_reducer∈[0,1]`，则 `S_phase ≤ 1/[1-f_reducer×β_reducer]`（不计附加同步），绝对理想极限 `1/(1-f_reducer)`。**不能用 26.17MiB/66MiB 推算 200W kernel ms 或百分比**；本机 256 线程已按维连续读取 FP32 partial，coalescing 缺陷也尚无 NCU 证明。

## C. GDN Q/K 预归一化：T≥64 已有，T<64 才有重复计算空间

固定 donor [`recurrent.cu:53–70,199–210`](https://github.com/tuxKOH/ninfer-V100X2/blob/5c51513096a9fceb9bb1609e43b15efecac30f93/src/ops/linear_attention/gated_delta_net/recurrent.cu) 在长输入预先计算 Q/K 的**FP32** 规范化数组，重用相同 QK head 的各 value/state tile；本机 `gated_delta_net.cpp:178–195,241–289` **已经**在 `T_full=floor(T/64)×64>0 && normalize_qk` 时调用两次 `l2norm`，产生独立的 **BF16** normalized Q/K，并供 chunked 和 tail 重用，因此 **T≥64 的“反复计算原始 Q/K norm”理论去重空间为零**。来源 FP32 写回改变本机中间 BF16 舍入界，且 donor 宽输入沿顺序 recurrent，不能移植成原数学同速的替代。

`T<64`（特别 T1 direct/batch-update/record，是否为该轮真实热核仍待验证）沿本机 `recurrent.cuh:63–75,119–136,151–165,615–675` 对每个 value-head/state-tile/warp 的 Q 与 K 分别做 norm。实际 `kNumWarps=4, kDvPerWarp=4, kStateDim=128`，每 value head 有 `128/(4×4)=8` 个 state tiles；每 qk head 对应 `Hv/Hqk=3` 个 value heads。因此一个 `(qk head,token,Q 或 K)` norm 在理论 warp 级执行 **`R=3×8×4=96` 次**，而只预计算一份可把 norm 数目降到 1；本机 16 QK heads 的 T1 路径，**Q+K norm warp 实例数 `2×16×96=3072`，预计算后为 `2×16=32`**，删除 **3040 次 warp 范围内 norm 工作**，仅对这个运算子项最多去除 `95/96`。它不改变 48×128×128 FP32 recurrence state 转移、必要的独立状态演化、输出或终态依赖，也不保证 GPU 整核加速。

若 `f_norm,short` 是目标 **200W** 阶段中实际花在上述重复 norm 算术上的可归属时间，则忽略准备、加载、同步的**极端**上限 `S_phase ≤ 1/[1-f_norm,short×(95/96)]`；实际需要新的预计算 kernel/暂存、Graph/工作区路由，可能更慢。对于 `T≥64` 现有已去重的场景，只可能讨论把 Q、K 的两次 BF16 `l2norm` 合为一次发射或改善读取（保持两者各一次归一化及 BF16 RN）；最多省**一条归一化 kernel launch/主 GDN 层/该输入调用**，当前 launch 耗时和占阶段比例未知，不能将 48 层直接乘成毫秒收益。

**Q/K 的字节条件**（每层一次 `T`、不含 `gdn_input_proj`、conv、chunked W/U/state 或 allocator 对齐）：原始 BF16 Q+K 为 `2×Hqk×Dg×T×2 = 8192T B`。单独 BF16 pre-norm 至少读取 `8192T`、写出 `8192T`；若换 donor FP32 输出，至少读取 `8192T`、写出 `16384T`，下游每次完整消费亦由 `8192T` 增至 `16384T`（若路径真的使用该 FP32 输入）。`T512` 时当前 BF16 Q/K scratch 为 **4,194,304B（4MiB）**，FP32 scratch **8,388,608B（8MiB）**，额外 **4MiB/当前活跃层 workspace**；仅准备阶段输入+输出流量下限由 **8MiB→12MiB/层**，消费端仍另计。48 层不能按各层同时分配而未经 planner 就推断峰值乘 48。

`T1` 未预归一化的原始 Q/K 唯一输入仅 **8192B**，96 次 warp 读取的**逻辑访问数**为 `96×8192=786,432B/层`（这 **不代表 768KiB DRAM**：重复数据可能命中 cache）。新增独立 BF16 pre-norm 仍需 8192B 原始读取、8192B 写回，并由状态 warps 消费归一化输入；因此删掉 norm 归约不自动节约 HBM，可能增加至少一轮 global 读写。若改用 FP32 暂存，重复 consumer 读取还会按元素宽度翻倍。direct kernel 每线程的 `state[4][4]` 已至少持有 **16 FP32 state 元素**，再加 Q/K lane、gate、partial 与索引；真实 registers/thread、spill、smem、受 launch 开销影响程度均**未知**。不得直接照抄 donor 的 FP32 normalized buffer 或顺序长 prefill。

## 每项证据、许可与下一最小测量输入

| 项目 | 已证的真实源码/工作量事实 | **未知的 200W 热点事实** | 固定许可与可做的最小下一测量 |
|---|---|---|---|
| QK + softmax sharing | SM89 `small_t_i8.cuh:571–775` 已 producer 计算一次并共享 P/alpha；T8 long 静态 shared 48,832B、PV accumulator 64 floats/consumer；unique KV 2112B/key | 4K/K7、32K/K15 当前实际 partial kernel 占比；QK/softmax 在 partial 内部的可归属时长；register/spill、cache/stall、T16实际子路由 | 固定 donor `gqa_attention_decode_i8_tc_volta.cuh` 按仓库 **Apache-2.0 + NOTICE**；依赖的 `src/ops/common/volta_mma.cuh` 明示改写自 **llama.cpp MIT**，还须 MIT 原版权/许可。下次只需同 binary **200W 32K/K15** 一次轻量 Nsight Systems kernel 分类和实际 route/width（必要时补 4K/K7），判断 partial 是否值得开独立 Op；不运行整模型 NCU。 |
| split-vector reducer | `small_t.cuh:169–269` 与 `small_t.cu:512–525`，partial FP32，DChunk256，T8/K32768/S69 最低两阶段逻辑访问 27,442,176B；staged8 要新增8192B shared/CTA | 同 200W profile reducer kernel 的绝对 ms、`f_reducer`、memory/stall、L2/DRAM sectors、真实活跃 S 分布，stage barrier 代价 | donor `gqa_attention_decode.cuh` 为仓库 **Apache-2.0 + NOTICE**，未在该子项发现必须复制的 MIT helper。若 baseline `f_reducer` 有分量，再用一个 **H24/D256/T8、S 为实际采集值、FP32 partial、相同位置/掩码** 的独立小 Op 做 NCU 及 clean 性能，先不改生产。 |
| GDN Q/K pre-norm | `gated_delta_net.cpp:178–289` 已为 T≥64 BF16 预归一化；`recurrent.cuh:615–675` 的短 T warp 重复因子 R96；T512 FP32 对比 BF16 多 4MiB workspace | 两个 `l2norm` kernel 的 200W 阶段占比、direct/record/fold T1调用数/占比、实际是否 norm-bound、state dependency stalls、融合后资源峰值 | donor `recurrent.cu`/`gated_delta_net.cpp` 为仓库 **Apache-2.0 + NOTICE**；该子项不依赖 Volta MIT MMA，其他嵌套来源待实施时复查。只需 **200W 一次现役 Graph profile** 区分 `l2norm` 与 recurrent（T1、T512），记录 kernel 次数、目标 T/H、主请求类型；确认 `f_norm` 后再做 1 个独立数值 Op。 |

上表来自当前代码与既有资料，**不代表那些下一步测量已执行**。两档的最小采集输入应固定已验权重和模型身份、`200W/C1/GSQ/INT8/chunk512/Graph`、相同 prompt/tg128、K7 或 K15、真实 `width/column_begin/full_width/valid_columns`、`visible_keys` 与 launch capacity、每次 kernel 名/调用数/GPU 时间，并区分 trace 与关闭 trace 的 clean 整体时间。**先用一个 32K/K15 的只读 trace 补 `f`；若 profile 实际落入多个 T8 子调用才按真实形状建理论表**。若任一项没有可见热点、被现有路径完全复用或新增 shared/register 容量无法承受，就以“不采用，缺可争取空间”收口。任何后续实现均必须独立 FP32/FP64 oracle 验证实际 cache codec/FP32 partial/GDN BF16 中间舍入及状态转移，并经过同 binary、200W 的匹配整体对照；本次没有执行这些步骤。

**明确排除：** 不移植 SM70 专用 QPN/`mma.sync.m8n8k4`、SM70 CUTLASS tile、TP2/P2P 通信、QUASAR NVFP4 权重或实验性 `--volta-long-attention` 归约顺序；不混用 donor BF16 `partial_acc` 与本机 FP32；不改任何 `src/`、`tests/`、OpenSpec `tasks.md`、功耗、仓库提交或远端。未经重新验证，本记录不产生任何 kernel 热点百分比、整体速度或资格通过声明。
