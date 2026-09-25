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

### 1.1 三版对比

三个版本，**同一台机器、同一个模型文件、同一批夹具、同一套参数**
（`--greedy --no-thinking`，prefill 取 `--max-new 8`，decode 取 300 token），
每格是**真交替 3 轮的中位数**（交替是为了让 GPU 温度漂移不整段落在某一个版本头上）。

| | 说明 |
|---|---|
| **预览版改良** | 本仓库改良过的预览版（本轮之前的状态） |
| **正式版** | 上游发布的正式版，本地 Linux 构建 |
| **本机当前版** | 本轮移植 + 调优之后（即本仓库当前 HEAD） |

**Prefill**（tok/s，越高越好）：

| 提示规模 | 预览版改良 | 正式版 | **本机当前版** | 对预览版 | 对正式版 |
|---|---:|---:|---:|---:|---:|
| 长（T=1759） | 1.26k | 1.98k | **1.99k** | **+58%** | 持平 |
| 中（T=123） | 796.6 | 1.34k | **1.37k** | **+72%** | +2% |
| 短（T=38） | 284.8 | 659.3 | **696.7** | **+145%** | +6% |
| 极小（T=15） | 146.2 | 335.7 | **391.7** | **+168%** | +17% |

**Decode**（tok/s，`en-code.json`，300 token）：

| 场景 | 预览版改良 | 正式版 | **本机当前版** | 对预览版 | 对正式版 |
|---|---:|---:|---:|---:|---:|
| MTP draft 3 | 148.6 | 125.6 | **148.5** | 持平 | **+18%** |
| └ 轮净耗时（ms） | 20.32 | 24.04 | **20.34** | 持平 | **−15%** |
| 无投机 | 64.6 | 57.5 | **63.7** | −1% | **+11%** |

> MTP 三版的**接受率完全相同**（67.6%）。这不是巧合——**接受率是任务的属性，不是引擎的属性**。
> 所以 decode 的差距全部来自"每轮算得多快"，不来自"每轮多产几个 token"。

**一句话**：长提示已经追平正式版，其余 prefill 场景全面反超，decode 保持 18% 领先。

### 1.2 上下文

| KV 精度 | 实测可达上下文 | 说明 |
|---|---:|---|
| `bf16` | ~120k | 默认 |
| `fp8` | ~238k | 精度与速度几乎不动 |
| **`rk4v4`** | **262,144（吃满原生上限）** | 4-bit，运行时 8.54 → **4.71 GiB** |

`rk4v4` 是 16 GB 卡上的关键档位。已实测 **247,646 token 深埋召回答对**。
代价是 4-bit KV 有精度损失，**本仓库还没有对它做过 PPL 门**——这是已知缺口，不是已验结论。

### 1.3 数值

| | PPL |
|---|---:|
| 本机基线（`NINFER_TERNARY_S8=0`） | **9.69192** |
| 本机默认（int8 档开启） | **9.6934** |

口径：`ninfer-perplexity --text wiki-slice.txt --context 512 --stride 256`（110 窗口 / 28,160 token）。
int8 档的偏移是 **+0.0015%**，量级正是 int8 激活量化误差本身。

> ⚠️ **不要拿这个 9.69 去和别处的数字比。** PPL 是协议的产物：换语料、换 context、换 stride 都是另一个数。
> 上游文档里的金判据是 ≈6.445，那是它自己的协议。对账要跟本机自己的历史基线比。

---

## 2. 部署

### 2.1 硬件与前置

| | |
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

### 2.5 推荐配置与每一项的代价

```bash
--kv-dtype rk4v4          # 上下文 120k → 262,144；运行时 8.54 → 4.71 GiB
--max-context 262144 --kv-capacity auto
--max-concurrency 1       # 交互延迟优先（每加一路，单路掉约 28%）
--prefill-chunk 1024      # 默认值已是最优（扫过 128~2048）
--spec mtp --draft-tokens 3   # decode +2.3×；draft 4/5 接受率不再涨，只花时间
--no-thinking             # decode +31%。**用法取舍**：关掉思考链
```

`--greedy` 能让 decode 再快约 12%（接受率 36% → 44%），但它**改变输出分布**——
**只在做性能对账时用**，不要当成生产默认。

**16 GB 上的取舍**：`--max-context 262144` 配 bf16 KV 起不来，必须配 4-bit KV。
`--vision` 固定吃约 3 GB，不用就别开。

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
| `NINFER_TERNARY_PREFILL` | `mma` | `block`/`ref` 为 A/B 臂 |
| `NINFER_TERNARY_HADAMARD` | 开 | `=0` 关旋转（**输出无意义**，仅诊断） |
| `NINFER_TERNARY_S8_DEBUG` | 关 | `=1` 打印**每次调用实际走的档位** |

**改 dispatch 之后第一件事是开探针确认分支真的被走到了**——只看速度看不出走没走对分支。
这条是本仓库最贵的一课，详见 `.claude/skills/ninfer-perf-tuning/SKILL.md`。

### 3.3 本机实测常量

换卡要全部重测。

| 量 | 值 |
|---|---|
| 只读带宽天花板 | 637 GB/s（实测） |
| 权重每趟 | 6.65 GiB（不含 MTP） |
| `s8` 代价 | 近平：T=15→28 只从 44.3 涨到 51.3 ms |
| `small_t` 代价 | 约 **20.5 ms × ceil(T/8)** |
| `small_t` / `s8` 交叉点 | **T=17** |
| 64 宽 / 128 宽 tile 交叉点 | **T=64** |
| decode 带宽 | 6.70 GiB / 13.75 ms = 487 GB/s（规格峰的 71%） |

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
