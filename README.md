[![English](https://img.shields.io/badge/Language-English-2ea44f?style=flat-square)](README.en.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-d73a49?style=flat-square)](README.md)

# 三元 Bonsai 2 27B 跑在 **NINFER** 上

> 单卡 C++20/CUDA 推理引擎 **NINFER** 的 Ada（`sm_89`）移植，跑三元量化的 Bonsai 2 27B。
> 目标机器是一张 **RTX 4070 Ti SUPER（16 GB）**，本文所有性能数字都在它上面实测。

三元权重按 `{−1, 0, +1}` 打包，每码 2 bit，外加每 128 权重一组一个 fp16 scale。
存储代价是**每权重 2.125 bit**（常说的"1.58-bit"是 log₂3，即一个三元符号的信息量——
两个数都对，量的是不同的东西，**内存系统付出的是前者**）。这大约是 4-bit 格式一半的权重流量，
也正是下面所有数字的来源：模型每验证轮要读 7.12 GiB 权重，而这张卡的实测读带宽是 637 GB/s。

**本仓库是衍生作品，不是 NINFER 上游。** 谱系与致谢见 §5。

---

## 1. 性能

### 1.0 读这些数字之前：口径与精度

**跑测的固定条件**（不对齐这些，下面的数字互相不可比）：

| 项 | 取值 |
|---|---|
| KV 精度 | **`bf16`**（三版都认的档位；`rk4v4` 上游三元版的 **CLI 不认**，只有 serve 认） |
| `--max-context` | 8192（长提示场景另设） |
| 采样 | **`--greedy`** —— 只为对账用，它改变输出分布，**不是生产默认** |
| 思考 | **`--no-thinking`** |
| 生成长度 | prefill 场景 `--max-new 8`；decode 场景 300 token |
| 轮次 | **真交替 3 轮的中位数**（每轮把三版都跑一遍，避免 GPU 温度漂移整段落在某一版头上） |

> **⚠️ 绝对值有 ±1~2% 的时段漂移，比值没有。** 同一天、同一台机器、同一份夹具、同一个二进制，
> 间隔 45 分钟的两轮全表测出来上一版的长提示从 1.25k 掉到 1.22k（−2.4%）、
> decode 从 146.9 掉到 145.4（−1.0%），而**三版同幅下移**，所以表里的百分比稳定、绝对值不稳定。
> 结论：**拿这张表比版本，不要拿它比时段**；要跟别的时段对账，**当场把两种配置都跑一遍**。

**★ 精度有三条互相独立的轴，先说清楚哪一条在变：**

| 轴 | 三版是否相同 | 说明 |
|---|---|---|
| **权重** | **三版相同** | 固定三元 `PQ2_0_G128`，每权重 2.125 bit。**永远不构成变量** |
| **激活** | **三版不同** ← 唯一在变的那条 | 走哪个 GEMM 档（见下） |
| **KV 缓存** | **三版相同** | 本节所有表都钉在 `bf16`（见上面固定条件） |

所以"某一版是什么精度"这个问法本身太粗。**会变的只有激活那条轴**，具体是：

| 版本 | 激活走哪条路（`--max-context 8192` 下）|
|---|---|
| **上一版** | 只有两条 bf16 路（小 T 的 tile 路 + 大 T 的 MMA 路）。**它那时还没有 s8 档** |
| **上游三元版** | **prefill 多一条 int8 路**（激活按 token 做 absmax 量化）；decode 仍走 bf16 |
| **本版** | 同上 |

**那条 int8 路只在 prefill 上生效**——它的门槛是 T ≥ 17（`kTernaryS8MinTokens`），
而 decode 的验证宽度上界是 16（`kMaximumVerifyTokens`）。**所以 §1.1 那座 decode 表三版其实是同精度的**
（都走 bf16），不一样的只是轨迹——prefill 的数值差会顺着 KV 传下来，于是接受率那一格会动。

**要同精度比 prefill，三版都关掉 int8：**

```bash
NINFER_TERNARY_S8=0 <binary> <model> ...      # 上一版没有这个开关，设了也无害
```

§1.2 的表里因此给出两行：**「各自默认」**（现实部署的样子）与
**「同为 bf16」**（剥掉激活差异后的纯内核对比）。

### 1.1 三版对比：上一版 / 上游三元版 / 本版

三个版本，**同一台机器、同一个模型文件、同一批夹具、同一套参数**
（`--greedy --no-thinking`，prefill 取 `--max-new 8`，decode 取 300 token），
每格是**真交替 3 轮的中位数**（交替是为了让 GPU 温度漂移不整段落在某一个版本头上）。

**三个版本各自是什么，以及本表用的到底是哪个二进制**（三份都能按 sha256 核对）：

| 版本 | 具体是什么 | 本表用的二进制 |
|---|---|---|
| **上一版** | 本仓库 **`ad6cb46`**（2026-09-21）—— 上一轮结束时的状态，也是本轮开工前 `origin/master` 的位置。**不是"上游的某个版本"，是本仓库自己的上一个提交** | `~/ninfer-work/bin-pre-s8/ninfer`<br><sub>2026-09-26 07:06 构建，sha256 `cea0e193…`</sub> |
| **上游三元版** | 三元移植的出处：魔搭 **[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary)**（引擎侧 `patches/` + 打包/验证工具）；源码树的直接基座是 **[Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)** v1.0.8 线。**上游发布的是 Windows `.exe` 包（MSVC）** | `~/ninfer-off-build/build/apps/ninfer`<br><sub>sha256 `b89c77e9…`，**本机用 GCC 16.2.1 重编的 Linux 版**</sub> |
| **本版** | 本仓库当前 HEAD | `~/ninfer-build/build/apps/ninfer`<br><sub>sha256 `70ecd5ca…`</sub> |

> **三份都是本机用同一个 GCC（16.2.1）编的 Linux 二进制**（`readelf -p .comment` 可核）。
> 上游那份尤其要说清楚：**它的发布物是 Windows 包，表里跑的是把它在这台机器上重编出来的 Linux 版**——
> "上游发布的正式版"和"表里那个二进制"因此不是同一个东西。
> 好处是这条路只比内核、不比工具链；代价是它不代表上游发布物的行为。
> 三份的构建时间分别是 07:06 / 05:59 / 16:33（2026-09-26）。

**Prefill · 各自默认精度**（tok/s，越高越好）：

| 提示规模 | 上一版<br>(bf16) | 上游三元版<br>(int8) | **本版**<br>(int8) | 对上一版 | 对上游三元版 |
|---|---:|---:|---:|---:|---:|
| 长（T=1759） | 1.22k | 1.96k | **2.53k** | **+107%** | **+29%** |
| 中（T=123） | 769.9 | 1.33k | **1.58k** | **+105%** | **+19%** |
| 短（T=38） | 280.6 | 664.1 | **734.7** | **+162%** | **+11%** |
| 极小（T=15） | 144.4 | 333.8 | **388.4** | **+169%** | +16% |

**Decode**（tok/s，`en-code.json`，300 token）：

| 场景 | 上一版 | 上游三元版 | **本版** | 对上一版 | 对上游三元版 |
|---|---:|---:|---:|---:|---:|
| MTP draft 3 | 145.4 | 122.5 | **147.3** | **+1.3%** | **+20%** |
| └ 轮净耗时（ms） | 20.77 | 24.65 | **20.30** | **−2.3%** | **−18%** |
| 无投机 | 64.8 | 57.4 | **64.4** | −0.6% | **+12%** |

> **注意 MTP 的接受率**：上一版与上游三元版都是 67.6%，本版是 **66.3%**。
> 差异来自 split-K（见 §1.4 末）：它在 T=17..64 的 prefill 上是 fp32 重结合，数值不完全逐位相同，
> 于是这一题的输出轨迹分叉，落到一条接受率略低的路径上。**它不是"每轮算得慢"，是"换了条轨迹"**
> （KSPLIT=0 时接受率立刻回到 67.6%）。同一二进制内这个选择是确定的、可复现的，
> 跨版本比较时才需要注意；`NINFER_TERNARY_S8_KSPLIT=0` 可以退回逐位相同的那条。

> **decode 那一行的来历**（两处，都在验证路 T ≤ 8 的 small_t 上，都不进 prefill）：
> **① 逐形状选 row block**（`n ≤ 5120` 用 16，否则 32）——响应非单调、一个全局值不可能对，
> 真交替双向 A/B **+1.8%**，接受率/直方图/生成文本全不变。**② 小形状的 split-K**——
> 只对 `n ≤ 6144` 且每片还剩 ≥4 个 K 步的形状切（本模型只有 5120×17408 满足），
> 双向 A/B **+1.3~1.4%**，六夹具 +0.2~1.8% 且接受长度无一下降。
> ②**不是逐位相同**（K 的求和顺序变了），回退开关 `NINFER_TERNARY_SMALL_T_KSPLIT=0`；
> ①是逐位相同的，覆盖开关 `NINFER_TERNARY_SMALL_T_ROWS`。

**一句话**：长提示从"追平"变成**反超 29%**，其余 prefill 场景全面领先，
decode 保持 20% 领先（接受率那一格的差别见上）。

### 1.2 长提示，与"同精度"到底是什么样

§1.1 那张表是**各自默认精度**——上游三元版与本版吃 int8，上一版吃 bf16。
把 int8 关掉（`NINFER_TERNARY_S8=0`）才能看出**内核本身**谁强。

`./three-way-long.sh`，KV 显式钉 `--kv-dtype bf16`（见 §1.3：不钉的话本仓库在这个长度上会
自动选 fp8，另外两版不会，于是量到的是精度差而不是内核差），真交替 3 轮中位数：

| 提示规模 | 精度 | 上一版 | 上游三元版 | **本版** |
|---|---|---:|---:|---:|
| **28,199**<br><sub>`--max-context 32768`</sub> | 各自默认 | 1.17k <sub>(bf16)</sub> | 1.75k <sub>(int8)</sub> | **2.18k** <sub>(int8)</sub> |
| | **同为 bf16** | 1.17k | **864** | **1.17k** |
| **61,636**<br><sub>`--max-context 65536`</sub> | 各自默认 | 1.06k <sub>(bf16)</sub> | 1.51k <sub>(int8)</sub> | **1.81k** <sub>(int8)</sub> |
| | **同为 bf16** | 1.06k | **804** | **1.06k** |

三个结论，**两个长度上都成立**：

1. **剥掉精度差异后，本机的 bf16 prefill 内核快上游三元版 35%（30k）/ 32%（64k）**。
   上游三元版的 prefill 优势**全部**来自它的 int8 档——这一条在 T=1.7k、28k、62k 上都成立。
2. **加回精度，本仓库反超 25%（30k）/ 20%（64k）**（2.18k 对 1.75k；1.81k 对 1.51k）。
   上一版这里还是"持平"——差的那一截是 **token tile 铺进 `blockIdx.y`（G2，+26%）**，
   而它**只作用于 int8 那条路**，所以下面"同为 bf16"那一行不动。
3. **同为 bf16 时，上一版与本版仍然持平**（1.17k / 1.17k；1.06k / 1.06k），
   上游三元版则被拉开 35%/32%。**本轮的长提示提速全部走 int8 档**：bf16 那条路上本轮没有改动，
   而它也不需要——它本来就在前面。

> **超过 64k 的三方对比用 CLI 做得到，只是不能用 `rk4v4`。** 上游三元版的 CLI 不认 `rk4v4`
> （只有它的 serve 认），但 **`fp8` 三版都认**，且实测能撑到 **233k**——124k 语境只需 3.9 GiB。
> 所以 >64k 走 `fp8` 即可，不需要 serve，也就不需要任何口径对齐。**本仓库尚未补测。**

### 1.3 上下文与 KV 档

**不传 `--kv-dtype` 时按 `--max-context` 自动选档：16383 及以下 `bf16`，16384 及以上 `fp8`。**
阈值不是拍的——下面第一张表就是它的依据。想要别的档显式写 `--kv-dtype` 即可，五档都认。

> **可达上下文随"启动时还剩多少显存"移动，不是模型的常数。** 表里的值是本机（16 GB 卡、
> 桌面不占独显）实测的上限；`--query-gpu=memory.used` 若已有几百 MiB 被占，这几个数会等比下降。
> 复现方法：从大往小试 `--max-context`，起不来时会明确报差多少字节
> （`but only N bytes are available after weights`），从差值除以每 token 字节数就能算出边界。
> 2026-09-26 复测：`fp8` 请求 233,000 起得来（规划 233,024 / 3,641 页 / runtime 7.66 GiB / 余 1.01 GiB），
> 234,000 起不来；`int8` 213,000 起得来。**反推每 token 32,994 字节（32.2 KiB）**，与表里那列一致。

| KV 精度 | 实测可达上下文 | 每 token | 说明 |
|---|---:|---:|---|
| `bf16` | ~120k | 64 KiB | `--max-context` ≤ 16383 时的默认 |
| `fp8` | **233,024** | 32.3 KiB | ≥ 16384 时的默认；**长上下文最快** |
| `int8` | ~213k | 33.0 KiB | 三档里数值最准，但比 `fp8` 慢 |
| **`rk4v4`** | **262,144（吃满原生上限）** | 17.0 KiB | 4-bit，运行时 8.54 → **4.71 GiB** |

**实测 decode**（`--spec mtp --draft-tokens 3`，真交替 3 轮取中位数，每一种配置都显式写 `--kv-dtype`，
所以量的是内核差不是精度差）：

| KV 档 | 30k 提示 | 64k 提示 | 长口径 PPL 代价 |
|---|---:|---:|---:|
| `bf16` | 142.9 tok/s | 96.7 tok/s | —（锚） |
| `int8` | +6.2% | +13.8% | **+0.077%** |
| **`fp8`** | **+7.1%** | **+16.4%** | +0.143% |
| `rk4v4` | +6.9% | +13.4% | +0.213% |

（64k 那一轮的 `rk4v4` 接受率是 61.2%，另外三档都是 60.1%——KV 档改的是 attention 的数值，
轨迹可能分叉，所以它的 +13.4% 略微被偏高地衬托了。
另：这两个锚测于验证路两处改动（§1.1 的 decode 注）之前，绝对值偏低约 1.3%，
**百分比不受影响**——row block 按形状选、split-K 只在 `n ≤ 6144` 且 K 够深的形状上切，
两条都与 KV 档无关，**四个 KV 档一视同仁**。）

读法三条：

1. **收益随上下文近线性**（约每 3.7k token 一个百分点），16k 以下进不了单次 decode 测量的
   run-to-run 噪声——这正是阈值取 16384 的原因。PPL 那一列用的是长口径门
   （`--context 65536 --stride 32768`，12.4 万 token 语料，3 窗口，bf16 锚 2.099109），
   **判据 ≤0.30%，四档全过。**
2. **`rk4v4` 的 KV 字节只有 `fp8` 的一半，decode 却更慢。** 它的 decode 端每层每次 attention
   多一个 `kv_cache_inverse_rotate_output_kernel`（16 层 ⇒ 图里多 16 个节点），那个固定代价
   吃掉了带宽收益。⇒ **KV 字节数不是长上下文 decode 的约束。** 早先按"KV 读 = 权重读"的算术
   把 `rk4v4` 当成 110k 以上的必然选择，那张模型表没有算进这一项，结论因此是错的。
3. 所以 **`rk4v4` 的定位是容量档，不是速度档**：它让 262,144（吃满原生上限）成为可能——
   那一行 `bf16`/`fp8`/`int8` 会先 OOM，只有它跑得起来。已实测 **247,646 token 深埋召回
   答对**。要速度用 `fp8`，要最准用 `int8`。

**服务端侧复核**（`ninfer-serve` + `/v1/chat/completions`，30k 提示，`--max-context 40960`——
四档共同装得下的深度，`bf16` 是 64 KiB/token，200K 要 12.8 GB 放不进 16 GB 卡。
真交替 2 轮，**每一种配置都重启服务，所以前缀是冷的**）：

| KV 档 | prefill | TTFT | decode | MTP 接受率 |
|---|---:|---:|---:|---:|
| `bf16` | 2.22k / 2.21k | 12.7 / 12.8 s | **146.6 / 146.6** | 94.4% |
| `int8` | 2.28k / 2.27k | 12.4 / 12.4 s | **157.1 / 157.0** | 94.4% |
| **`fp8`** | 2.25k / 2.24k | 12.6 / 12.6 s | **157.1 / 157.1** | 94.4% |
| `rk4v4` | 2.20k / 2.20k | 12.8 / 12.8 s | **156.1 / 155.9** | 94.4% |

**两条**：① **prefill 与 KV 档基本无关**（极差 3.6%）——prefill 是**权重带宽**受限
（要读 7.12 GiB 权重），KV 写入只占零头，换档不会让长提示更快进上下文。
② **decode 只有 `bf16` 明显慢**，其余三档约 +7%，`int8` 与 `fp8` 打平、`rk4v4` 落后 0.6%。
**排序与上面 CLI 那张表一致**，所以那张表的口径是可靠的。

> ⚠️ 两个口径陷阱：**每一种配置都必须冷前缀**（第二次起命中 99.9%，`prefill X tok/s` 那行要么是残量
> 要么不打）；**深度必须四档都装得下**。

### 1.4 数值

| 配置 | PPL |
|---|---:|
| 本机基线（`NINFER_TERNARY_S8=0`） | **9.69192** |
| 本机默认（int8 档开启） | **9.6934** |

口径：`ninfer-perplexity --text wiki-slice.txt --context 512 --stride 256`（110 窗口 / 28,160 token）。

**split-K 与 T=17..64 这个窗口**（`NINFER_TERNARY_S8_KSPLIT`，默认开）：这个窗口里 token 轴
只够一个 64 的分片、行轴又填不满卡（`gridX = div_up(n,64) < 198`），于是 s8 沿 K 切片补 CTA，
在 T=28 上 prefill **+7.1%**。代价是 K 的累加被重结合成 fp32 的分片再求和——**确定，但不逐位相同**：
`--context 32` 的 PPL 因此 +0.041%，而**上表两个数不受影响**（`--context 512` 的 T=508 早出了窗口，
G3 前后读到的是同一个值）。

> **它不改 decode 的速度，但会改 decode 的结果。** decode 的验证宽度上界是 16
> （`kMaximumVerifyTokens`），低于 s8 自己的门槛 17（`kTernaryS8MinTokens`，见
> `ternary_rowsplit_gemm.cu` 的 `x.ne[1] >= ternary_s8_min_tokens()`），**所以 decode 这一步
> 根本不走 s8**。变的是 prefill 的数值，于是这一题的输出轨迹可能分叉：`en-code.json` 上分叉了
> （接受率 67.6% → 66.0%），而同样落在窗口里的 `bash.json`(T=31) 和
> `gap-sm.json`(T=27) **逐字节相同**。**分叉是任务相关的事件，不是必然。** 退回逐位相同的那条：
> `NINFER_TERNARY_S8_KSPLIT=0`。

**验证路上的第二处 split-K**（`NINFER_TERNARY_SMALL_T_KSPLIT`，默认开，2026-09-26）：
它和上面这条**方向相反**——切的是 decode 的 T ≤ 8，而正是 s8 够不到的那一段。
只对「行网格填不满卡 **且每片还剩 ≥4 个 K 步」的形状切（本模型只有 `5120×17408` 满足，
4 片、1280 CTA），因为实测**每片只剩 1~2 步的形状反而变慢**（prologue 摊不掉）：
不分这道门时**整体 −1%**，加了之后 +1.3~1.4%。六夹具 +0.2~1.8%、接受长度无一下降，
**两种配置的 PPL 逐位相同**（prefill 根本不切，所以上表两个数不受影响）。同样不是逐位相同，
回退开关是 `NINFER_TERNARY_SMALL_T_KSPLIT=0`。

> **★ 这两个数是 `fp8` KV 的数。** `ninfer-perplexity` 的 KV 显式钉在 `fp8`（源码里写死，
> 见 `apps/perplexity/main.cpp`），**而 `ninfer` CLI 按 `--max-context` 自动选档**
> （≤16383 用 `bf16`，≥16384 用 `fp8`，见 §1.3）。两者协议不同、**数不要混**——
> §1.1 的性能表在 `--max-context 8192` 下跑，是 bf16 KV；这一节是 fp8 KV。

int8 档的偏移是 **+0.0015%**，量级正是 int8 激活量化误差本身。

**KV 轴**（同一口径，五档，2026-09-26 首测）：

| KV 档 | PPL | 相对 `bf16` | 每 token | 说明 |
|---|---:|---:|---:|---|
| **`bf16`** | **9.688451** | — | 64.00 KiB | `--max-context` ≤ 16383 时的 CLI 默认。无损基准 |
| `int8` | 9.691663 | +0.033% | 33.00 KiB | **比 `fp8` 更准**，但 decode 比 `fp8` 慢 |
| `fp8` | 9.693396 | +0.051% | 32.25 KiB | ≥ 16384 时的 CLI 默认，也是 `ninfer-perplexity` 的默认 |
| `rk4v4` | 9.702774 | **+0.148%** | **17.00 KiB** | 跑满 262,144 的**唯一**档位，decode 比 `fp8` 慢 |
| `rk4v4-e8` | 9.726728 | **+0.395%** | 17.00 KiB | **比 `rk4v4` 更差，且容量/速度相同** |

三条结论：

1. **`int8` 比 `fp8` 准**（+0.033% vs +0.051%），字节只多 2.3%。原因是编码方式：
   `int8-g64` 是每 64 维一个 scale 的 8 位**定点**，而 `fp8-e4m3-r256` 只有 **3 位尾数**、
   每 256 维才一个 scale。**逐元素精度差约 6 倍。** 但"更准"不等于"更好的默认"：
   decode 实测 `fp8` 更快（64k 下 +16.4% 对 +13.4%，见 §1.3），所以长上下文的默认是 `fp8`，
   `int8` 是精度优先时的选择。
2. **`rk4v4` 的代价是 +0.148%**，落在 KV 档的"软门"（0.05%~1%）低端——作为 16 GB 卡上
   跑满 262,144 上下文的**唯一**档位，这个代价是站得住的。它是本仓库第一次被量化。
3. **`rk4v4-e8` 比 `rk4v4` 差 3.5 倍，而容量、速度、每 token 字节完全相同。**
   原因是它的解码端是个**刻意的半陪集近似**（代码注释原文：*"the D8+0.5 E8 coset is
   collapsed … **not an exact E8**"*）——E8 的整形增益有一部分被丢掉了。
   **在当前实现下 `rk4v4-e8` 没有存在理由。**

> ⚠️ **这是 `--context 512` 的短口径。** 逐元素量化误差与上下文长度无关（KV 一次写入、
> 只读不重量化），所以这张表在数值上有效；但越长上下文注意力越选择性，代价**可能更高**
> （不是更低）。**长口径已做**（`--context 65536 --stride 32768`，12.4 万 token 语料，3 窗口，
> bf16 锚 2.099109）：`int8` +0.077% / `fp8` +0.143% / `rk4v4` +0.213%，判据 ≤0.30% 四档全过。
> **`rk4v4` 长口径是短口径的 1.44 倍**，方向与预期一致。

> ⚠️ **不要拿这个 9.69 去和别处的数字比。** PPL 是协议的产物：换语料、换 context、换 stride 都是另一个数。
> 上游文档里的金判据是 ≈6.445，那是它自己的协议。对账要跟本机自己的历史基线比。

---

### 1.5 服务端的两条行为（都会咬人）

**① `reasoning_effort` 的默认值是模型自带的 `xhigh`。**
`chat_template.cpp:432` 的 `result.reasoning_effort.default_effort = ReasoningEffort::XHigh;`。
**请求里开思考而不指定档位，就是最高档**——不是调用方设的。要降档得显式传
`reasoning_effort: low|medium|high`（`request.h:142` 认 none/minimal/low/medium/high/xhigh）。

**② `--default-thinking-budget` 是许可证，不是刹车。**
思考/正文的切分靠模型吐控制 token（`frontend.cpp:557`）；预算只喂给一个**默认整个不启用**的
语义追踪器（`semantic.in_reasoning = starts_in_reasoning && thinking.budget.has_value()`，
注释说明默认不启用是为了不把每个 token 解码两遍），超预算的动作是**抛
`model output exceeded the licensed thinking budget`——报错，不替模型收尾。**
⇒ **它不解决"想不停"。**

**给使用者的三条**（都是上面两条的直接推论，不含模型行为）：

| 想要 | 怎么做 |
|---|---|
| 思考的档位可控 | 请求里显式传 `reasoning_effort`，别依赖默认值 |
| 思考有终点 | **走 agent 循环**——每一步以工具调用收尾，思考因此有终点（可直接跑的例子见 §2.4）。`--default-thinking-budget` 是另一条路，但**它会在越界时报错，别把它当"自动收尾"用** |
| 只要速度 | `--no-thinking`（取舍见 §2.5） |

**"思考有终点"实测**（同一份二进制、同一台机器，服务端 `--max-context 200000 --kv-dtype fp8
--spec mtp --draft-tokens 3`，**不传 `--no-thinking`、不传 `--default-thinking-budget`**
⇒ 思考不限、档位是默认的 `xhigh`）：

| | 客户端上限 32,768 | **客户端上限放开到 199,000** |
|---|---|---|
| 首次拿到可用产出 | 3 分 22 秒 / 7 步 | **3 分 27 秒 / 8 步** |
| 第 1 步思考 | 56,783 字符 | **34,844 字符** |
| 思考合计 | 58,689 字符 | **47,487 字符** |
| 产出 | 58 行单文件 HTML | **49 行单文件 HTML** |

**放开上限之后它并没有想得更多（反而少 19%），照样收尾并产出。**
⇒ **终端条件是"每一步有个必须落地的动作"，不是输出上限。**
把上限调大既不是让长思考收敛的办法，也不是它不收敛的原因。

## 2. 部署

### 2.1 硬件与前置

| 项 | 要求 |
|---|---|
| GPU | NVIDIA Ada `sm_89`（RTX 40 系）。本仓库在 **RTX 4070 Ti SUPER 16 GB** 上调优与验证 |
| 显存 | ≥ 16 GB（权重 6.70 GiB + MTP 0.42 GiB + KV） |
| 系统 | Linux（本仓库在 Arch 上验证）。**路径必须纯 ASCII** |
| 工具链 | CUDA 13.x、GCC 15+、CMake ≥ 3.24 |

**把显示器接在核显上（如果主板有）。** 独显跑桌面会抢显存带宽，本机实测差约 **10%**。
这不是调参，是环境状态。

### 2.2 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j"$(nproc)"
```

产物：`build/apps/{ninfer, ninfer-serve, ninfer-perplexity}`。

> **两个已知的编译坑**（都是 `-rdc=true` 下多路 nvcc 并发引起的，与代码无关）：
> 1. **`TMPDIR` 在 tmpfs 上会导致 GCC 段错误。** 把 `TMPDIR` 指到磁盘：
>    `export TMPDIR=$PWD/tmp-nvcc`
> 2. 编译偶发报 ICE（`cc1plus` 段错误）。**重试即可**，不是代码问题。
>    并发降到 `-j6` 能显著降低概率。

### 2.3 模型

模型是 **Ternary Bonsai 2 27B** 的 `.ninfer` 制品。**本仓库不包含、也不重新分发权重**——
从官方渠道获取，并遵守其许可（可能不是 Apache-2.0）。详见 §5。

### 2.4 跑起来

**命令行**（单次问答）：

```bash
./build/apps/ninfer <model.ninfer> \
  --messages prompt.json \
  --max-context 8192 --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --no-thinking
```

`prompt.json` 是 OpenAI 风格的消息数组：

```json
[{"role": "user", "content": "你好"}]
```

> 中文提示**必须**走 `--messages`。走 `--prompt "中文"` 会报 `failed to normalize UTF-8 text as NFC`。

**服务端**（推荐用于多轮对话）：

```bash
./build/apps/ninfer-serve <model.ninfer> \
  --host 127.0.0.1 --port 8084 \
  --kv-dtype rk4v4 --max-context 262144 --kv-capacity auto \
  --max-concurrency 1 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --no-thinking
```

```bash
curl -s http://127.0.0.1:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":512}' | jq -r '.choices[0].message.content'
```

> **多轮对话优先用 `ninfer-serve` 而不是 `ninfer`。** CLI 侧的前缀缓存是**硬编码关闭**的，
> 而 serve 默认开启——同一段历史在 CLI 上每轮都会从头 prefill 一遍。

**接 agent harness：一个能验的例子**

上面两条只证明"服务起来了"。**工具调用 + 多轮**这条链路要单独验，用 agent harness 跑一个具体任务：

```bash
# 终端 1：把服务起起来（§2.5 的推荐配置，但**要去掉 --no-thinking**——见下）
# 终端 2：
BONSAI_API_KEY=local dsh --profile headless \
  "Draw a 2D animation of a pelican riding a bicycle as ONE self-contained HTML \
   file, inline SVG only, under 120 lines. Output only the code."
```

harness 侧的模型配置（`~/.dsh/settings.yaml`）指向 `http://127.0.0.1:8080/v1`、model id
`Ternary-Bonsai-2-27B`，**`contextWindow` 要和 `--max-context` 对齐**——给大了客户端会发超长请求，
服务器直接拒掉。

**实测**：**3 分 22 秒**结束，过程中发出 **8 次请求**——agent 的工具循环：自己写文件、
自己 `wc -l` 数行数、自己解析 SVG 验证结构。产出是一个 **58 行、零外部引用、能渲染的单文件 HTML**。

> **两条前提，缺了它这个例子不会自己结束**：
> ① **服务端不要加 `--no-thinking`**——agent 循环靠模型自己收尾，关掉思考链它就只会一遍遍重写同一个文件；
> ② 采样不要用 `--greedy`（那是给对账用的，会加剧这一点）。
>
> **为什么用 agent 循环而不是一次性生成**：每一步都以一个工具调用收尾，**思考因此有终点**。
> 另跑一次对照，把客户端输出上限从 32,768 放开到 199,000：**3 分 27 秒 / 8 步**，
> 它**并没有想得更多**（思考量反而少 19%），照样收尾并给出成品。
> **终端条件是"每一步有个必须落地的动作"，不是上限。**

### 2.5 推荐配置与每一项的代价

```bash
--kv-dtype rk4v4          # 上下文 120k → 262,144；运行时 8.54 → 4.71 GiB
--max-context 262144 --kv-capacity auto
--max-concurrency 1       # 交互延迟优先；**批处理吞吐**见下
--prefill-chunk 1024      # 默认值已是最优（扫过 128~2048）
--spec mtp --draft-tokens 3   # decode +2.3×。**K 的最优点随任务移动**，见下
--no-thinking             # decode +31%。**用法取舍**：关掉思考链
```

**`--max-concurrency` 有两种口径，别只看一半**：decode 是**真正的批**（权重读被摊掉），
但 prefill 是**独占**的（一次只服务一条 lane）。所以

| | 单路延迟 | 总吞吐 |
|---|---|---|
| `--max-concurrency 1` | ✅ 113 t/s | 1.00× |
| `2 ~ 3` | 降到 70% / 51% | **约 1.40× / 1.53×** |

注意 **KV 池按并发线性放大**（`max_concurrency × page_count(max_context)`），
所以 **16 GB 上 `N>1` 与长上下文（>64k）互斥**。

**`--draft-tokens` 的最优点随任务移动**（2026-09-26 实测，真交替 3 轮中位数）：

| 任务 | 每位置存活率 | 最优 K | K=3 | K=5 | K=7 |
|---|---:|---:|---:|---:|---:|
| `en-code`（写代码，低可预测性） | ≈0.76 | **5** | 143.4 | **151.0** | 136.6 |
| 数数 / 结构化输出 | ≈0.95 | **7** | 191.3 | 232.4 | **231.8（+21%）** |
| 模板复读（上界） | ≈0.995 | **≥7** | 197.0 | 248.4 | **289.3（+47%）** |

**接受率不是常数，它随位置衰减，而衰减速度是任务的属性**，`accepted by pos` 直方图直接
把它印出来（K=7 时）：

```
en-code        56,45,28,23,17,11, 7    每位置 ~0.76，第 4 位起就亏
mtp-count      40,39,39,32,30,22,13    前四位 ~0.97，后面掉
mtp-template   32,32,32,32,32,32,31    几乎不衰减，接受长度 7.97（上限 8）
```

所以"draft 更大只会更慢"只对低可预测性任务成立：**同一个二进制上，`en-code` 的最优是 K=5
而 K=7 反而比 K=3 差，模板复读却在 K=7 上快 47%。** 成本侧是固定的（每个 draft 步
≈1.94 ms，与任务无关），收益侧是存活概率，交叉点在每位置 ≈0.75。

**K 的上限是 7**（`T=K+1≤8`，再高会掉出 `small_t`，验证轮从 13.3 ms 跳到 40 ms），
`--draft-tokens` 在这个范围内按任务选。上限写在四个地方，各自是不同类型的东西
（捕获图的宽度、模型配置、CLI 校验、运行时不变量），彼此手工对齐，改的时候要一起改——
见 `round_state.h` 里那段注释。

`--greedy` 能让 decode 再快约 12%（接受率 36% → 44%），但它**改变输出分布**——
**只在做性能对账时用**，不要当成生产默认。

**16 GB 上的取舍**：`--max-context 262144` 配 bf16 KV 起不来，必须配 4-bit KV。

`--vision` 的显存代价约 **0.5 GiB**（视觉塔 0.27 + workspace 0.24）。
另有 media 子系统的**预算上限 3 GiB**（`--media-cache-mib` 1024 + `--media-live-mib` 2048）——
那是**宿主侧**额度、按需分配、不进显存预留。要省它不必关视觉，调这两个参数即可。

**`--kv-capacity auto` 在 `--max-concurrency 1` 时等价于显式写 `max_context`**——
它不是"自适应"，只影响"放不下就报错"那条分支。

---

## 3. 调优

### 3.1 档位

三元线性层按 token 数 T 选内核。**T 不是提示长度**——实测规律（七个夹具一致）：

```
线性层看到的 T = prompt_tokens − 4
```

当前默认：

| T | 档位 |
|---|---|
| 1 | `small_t`（tensor core） |
| 2..8 | `small_t`（验证轮） |
| 9..16 | `small_t_tiled`（8 宽 tile，每 8 token 重入一次） |
| 17 及以上 | `s8`（int8 激活 × int8 权重张量核） |
| （s8 关闭时）≥32 | `short_mma` / `wide_mma`（bf16） |

### 3.2 全部可调旋钮

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `NINFER_TERNARY_S8` | 开 | `=0` 回到 bf16 档（**同二进制 A/B 用这个**） |
| `NINFER_TERNARY_S8_MIN_TOKENS` | 17 | s8 起效的 T |
| `NINFER_TERNARY_GAP` | `auto` | T=9..31 用哪档：`auto`/`small_t`/`s8`/`gemv` |
| `NINFER_TERNARY_GAP_SMALL_T_MAX` | 16 | `auto` 的分界 |
| `NINFER_TERNARY_MMA_MIN_TOKENS` | 32 | bf16 MMA 起效的 T |
| `NINFER_TERNARY_SHORT_TILE_TOKENS` | 64 | 64 宽 / 128 宽 tile 的分界 |
| `NINFER_TERNARY_VERIFY_CAP` | 8 | 验证轮接受的 T 上限 |
| `NINFER_TERNARY_SMALL_T_ROWS` | 32 | small_t 的 rows/CTA（16/32/48） |
| `NINFER_TERNARY_DECODE` | `small_t` | `=gemv` 回退 |
| `NINFER_TERNARY_PREFILL` | `mma` | `block`/`ref` 为 A/B 对照 |
| `NINFER_TERNARY_HADAMARD` | 开 | `=0` 关旋转（**输出无意义**，仅诊断） |
| `NINFER_TERNARY_S8_DEBUG` | 关 | `=1` 打印**每次调用实际走的档位** |

**改 dispatch 之后第一件事是开探针确认分支真的被走到了**——只看速度看不出走没走对分支。
这条是本仓库最贵的一课，详见 `.claude/skills/ninfer-perf-tuning/SKILL.md`。

### 3.3 本机实测常量

换卡要全部重测。

| 量 | 值 |
|---|---|
| 只读带宽天花板 | 637 GB/s（实测；**单位是 GB/s 还是 GiB/s 尚未钉死**，见下） |
| 权重每趟 | **6.80 GB = 6.33 GiB**（64 层 6461.8 MB + `output_head` 337.7 MB，逐张量累加；不含 MTP） |
| `s8` 代价 | T=15 → 44.3 ms，T=28 → 51.3 ms。**不是"近平"**——mma 条数与 T 无关（tile 固定 64 宽），T=15 时 **76% 的 mma 算在零上** |
| `small_t` 代价 | T=9..16 约 **20.5 ms × ceil(T/8)**；**T=1..8 是完全平的**（T=1 与 T=4 实测 13.75 / 13.3 ms，同一个 kernel、同一份权重） |
| `small_t` / `s8` 交叉点 | **T=17** |
| 64 宽 / 128 宽 tile 交叉点 | **T=64** |
| decode 带宽 | **6.80 GB / 13.75 ms = 494 GB/s**（约占 637 的 78%） |
| 每个 decode token 的 kernel 数 | **1224 个**（全在一张 CUDA graph 内，帧间间隙 **64 ns**，真空闲仅 3.7%） |

> ⚠️ **上面几行的单位口径不一致，是已知问题。** `637` 这个天花板在有些推导里当 GB/s 用
> （6.80 GB ÷ 637 = 10.7 ms），在另一些里当 GiB/s 用（6.65 ÷ 637 = 10.4 ms）。两者差 7%。
> **在钉死之前不要引用百分比。** 一次干净的单 kernel 读数就能定。

---

## 4. 这个分支改了什么

在上游基础上，本分支做了三类改动。**每一处的实测数据都写在代码注释里**。

**内核路径**（预览版阶段）
- 三元 tensor-core prefill 路径，取代 blocked GEMV
- 投机验证轮的 small-T tensor-core 路径
- 两轮靠读 SASS 找出来的 decode 优化
- 旋转内核的发射打包

**本轮（移植 + 调优）**
- **int8（`s8`）档移植** —— 上游正式版的 int8 激活 × 张量核档位。同口径 **1.61×**，PPL 偏移 +0.0015%
- **prefill token tile 按 T 分档** —— 128 宽 tile 原本没有 token 循环，T 不满 128 时整块 K 扫描照付。
  改为 T≤64 走 64 宽档（shared 43264 → 26880 B，2 → 3 CTA/SM）。T=42/50 **+42%**
- **填 T=9..31 的空洞** —— 这一带原本两个阈值的覆盖面之间有个缝，而普通短问题（prompt 13~35）
  正好落在里面。现在按代价模型分段：T≤16 走 8 宽张量核、T≥17 走 int8。**+131% ~ +266%**
- **s8 阈值重拟合 33 → 17** —— 上游的 33 是对着它自己的 bf16 档量的；本机两条线的交叉点在 16/17。
  顺带消掉了 T=32 上一个 74% 的台阶

**明确试过但无效的**（负面结论同样记在代码里）
- 加宽 small_t 的 tile —— 是回归（45.2 → 19.0 t/s），累加器吃掉了驻留 CTA
- 用 `__launch_bounds__` 换 occupancy —— 三处独立实测都单调变慢
- **收窄 s8 的 token tile 让它跟 T 走** —— 也是回归（六个夹具 **+1%**）。机制：kernel 不是 mma 受限
  （张量核只 17~35% 活跃），砍一半 tile 只把每 CTA 的固定开销摊到一半的工作量上。
  **强制 32 在 T=38 上更是 +19%**——`tok_base` 循环跑两遍，权重读两遍
- 拆 LUT —— −12.6%，机制是同地址广播
- `--lm-head-draft` —— 噪声内；它的头 356.5 MB 比主 output head 337.7 MB 还大，流量反而 +6%

---

## 5. 谱系与致谢

这项工作完全建立在 **NINFER** 及其周边分支之上。

| 项目 | 贡献 |
|---|---|
| **[Neroued/ninfer](https://github.com/Neroued/ninfer)** | **NINFER 规范上游** —— C++20/CUDA 架构、DFlash2、ReplaySSM、Paged KV Cache。Apache-2.0 |
| [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | 最早的 RTX 4090 分支；E8 格 `rk4v4-e8` KV 存储 |
| [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) | Ada `sm_89` 内核优化、GDN 协作启动修复 |
| [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) | Win32/MSVC 可移植层 |
| [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) | 原生 Windows MSVC 编译基座 |
| [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) | Ampere 早期工作 |
| **[Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)** | **本分支源码树的直接基座** |
| **[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary)**（魔搭） | **三元移植本身的出处**：`patches/` 引擎侧改动、`tools/` 打包与验证工具、`docs/` 技术记录 |

**方法参考**：三元编解码语义对齐 llama.cpp 生态的 `ggml-quants.c`；折叠 Hadamard 基参考 PrismML 的
公开运行时与其 `prism.hadamard.*` 元数据契约；张量核 FWT 的设计思路受公开的 HadaCore / TurboQuant 工作启发。
以上仅为方法参考，本仓代码为独立实现。

### 模型权重

模型是 **Ternary Bonsai 2 27B**，底座 `Qwen/Qwen3.8-27B`，架构未改，权重为三元量化 + Hadamard 旋转基。
**权重版权归其原作者与发布方所有（PrismML 及其上游 Qwen 体系）——本仓库不包含、也不重新分发任何模型权重。**
由权重产出的 `.ninfer` 制品属于权重派生品，其再分发义务**以权重原许可为准，与代码许可无关**。

上游的 `NOTICE` 与 `LICENSE` 原样保留。本分支修改过的每个文件都在顶部带上显著声明，
这是 Apache-2.0 §4(b) 的要求。

---

## 附：其它文档

| 文档 | 内容 |
|---|---|
| `.claude/skills/ninfer-perf-tuning/SKILL.md` | **调优方法论**：测量纪律、档位探针、A/B 设计、数值门、会静默骗人的坑 |
| [README.en.md](README.en.md) | 英文版（含上游 NINFER 的完整 README 原文） |
| `docs/` | 上游的产品指南（CLI、服务、性能、评测、维护者文档） |
