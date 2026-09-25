---
name: ninfer-perf-tuning
description: 在本仓库（Ternary Bonsai 2 27B on NINFER / Ada sm_89）上做性能调优与 kernel 改动时使用。包含测量纪律、档位探针、A/B 设计、数值门，以及一系列会静默骗人的坑。改动任何 dispatch 阈值、kernel schedule 或量化档位之前先读这个。
---

# NINFER 三元引擎性能调优

这个仓库是在别人的基础上做的 Ada 移植 + 本机调优。调优的难点不在改代码，在**别把噪声当结论**。
下面每一条都是踩过之后写的，不是泛泛的原则。

## 一、动手之前：四项必须对齐

比较任何两个配置之前，先确认这四项一致。不对齐的话量到的不是引擎。

| 变量 | 默认值 | 影响 |
|---|---|---|
| 思考模式 | **开**（`--no-thinking` 关闭） | decode **+31%** |
| 采样 | temp 1.0（`--greedy` 关闭） | decode **+12%** |
| 任务 | — | 接受率 **44% ↔ 95.7%** |
| 生成长度 | — | decode 速度随 `--max-new` 变 |

另外两条：
- **显示器必须在主板核显上**（不在独显）。这条值约 10%，且是环境状态不是参数。
- **GPU 温度会漂**：同一夹具 p2048 在 1.12k~1.22k 之间晃。所以**必须真交替**——
  每一轮把 A、B 都跑一遍取中位数，而不是先跑完 A 再跑 B。后者的漂移会整段记到 B 头上。

## 二、先问"走了哪一档"，再问"快不快"

**这是本仓库最重要的一条。** 档位由一串 env 可覆盖的阈值决定，阈值配错时速度差异看起来像"内核不行"。

改动 dispatch 之后，第一件事是开探针确认分支：

```bash
NINFER_TERNARY_S8_DEBUG=1 NINFER_TERNARY_S8_DEBUG_MIN_T=5 \
NINFER_TERNARY_S8_DEBUG_BUDGET=100000 \
  <bin> <model> --messages <fixture> --max-new 4 --no-thinking 2>&1 \
  | grep -o 'rung=[a-z0-9_]*' | sort | uniq -c
```

**每一条可走的分支都必须有探针。** 缺一条的代价是真金白银的：分块 GEMV 兜底路径曾经没有探针，
于是一个走那条路的夹具打不出任何 prefill 档位，我把它误判成"夹具异常"查了很久。
现在 `small_t` / `gemv_tile` / `s8` / `small_t_tiled` / `short_mma` / `wide_mma` /
`block_gemv` / `reference` 八条都会报出自己。

### 档位图（当前默认）

| T | 档位 | 阈值 / 开关 |
|---|---|---|
| 1 | `small_t`（T=1 入口） | `NINFER_TERNARY_DECODE=gemv` 回退 |
| 2..8 | `small_t` | `verify_token_cap()`（钳在 8） |
| 9..16 | `small_t_tiled` | `NINFER_TERNARY_GAP_SMALL_T_MAX`（16） |
| 17..∞ | `s8` | `NINFER_TERNARY_S8_MIN_TOKENS`（17）；`NINFER_TERNARY_S8=0` 关闭 |
| （s8 关时）≥32 | `short_mma` / `wide_mma` | `NINFER_TERNARY_MMA_MIN_TOKENS`（32）、`NINFER_TERNARY_SHORT_TILE_TOKENS`（64） |
| 兜底 | `block_gemv` / `reference` | `NINFER_TERNARY_PREFILL`、`NINFER_TERNARY_ROWS`、`NINFER_TERNARY_TILE` |

**注意 T 不是 prompt 长度。** 实测规律（七个夹具一致）：

```
线性层看到的 T = prompt_tokens − 4
```

所以 `mma_min_tokens = 32` 实际覆盖的是 **prompt ≥ 36**。把阈值和 prompt 长度混起来会算错覆盖面。

## 三、A/B 怎么做

- **同一个二进制、只用环境变量切**。两个二进制比会混进编译差异。
- 新加的档位必须**从第一天就做成 env 可切**，否则每个变体都要重编。
- **交替跑**（见上）。
- **同时记绝对值**。短提示（T<64）的 prefill t/s 被固定开销淹没——T=35 时 154 t/s 意味着整趟
  0.23 s，而 6.65 GiB 权重在 637 GB/s 下只要 10.4 ms。**比值会骗人，用 `model elapsed` 的差值看。**
- 短提示场景要**专门造夹具**。现成夹具的 T 与预期经常差很远，造完先用探针量一遍真实 T。

## 四、数值门

改内核 / 改量化档**必须**过数值门。标准口径：

```bash
ninfer-perplexity <model> --text fixtures/wiki-slice.txt --context 512 --stride 256
# → 110 窗口 / 28,160 token
```

| 判据 | 期望 |
|---|---|
| 旧二进制与 `NINFER_TERNARY_S8=0` | **逐位相同**（9.69192）——移植惰性的硬证据 |
| 改数值的档位（s8 等） | 偏移小且**可解释**（s8 是 +0.0015%，量级 = int8 激活量化误差） |
| 任何情况 | **不是几百**。"几百" = 数值坏了，不是"慢" |

**两条容易搞错的地方**：

1. **标准口径用的是 `--context 512`，线性层看到 T=508。** 要验小 T 段的数值（比如 T=9..31 的空档），
   必须用 `--context 32 --stride 16` 单独跑一次，否则那个带根本没被覆盖。
2. **正式版文档写的金判据是 PPL ≈ 6.445**，那是**它的语料和协议**下的值。本机同口径是 9.69。
   逐位对账要跟本机自己的历史基线比，不要跟它的数字比。

## 五、会静默骗人的坑

按"骗过我的次数"排序。

1. **两个本该不同的臂给出同一个数 → 先怀疑它没跑，而不是它没用。**
   我加 `GAP=s8` 这条入口时忘了 scratch 的分配条件是**策略阈值**（33）而不是"可被请求的下界"，
   于是 T<33 上永远拿不到 scratch、静默落回 GEMV，读数正好等于基线臂。
   **不报错、不崩、PPL 也不动。** 唯一的线索就是那个"一模一样"。
2. **缺探针的分支 = 隐形的分支。** 见 §二。
3. **基准跑着的时候不要重编二进制、不要起第二个基准实例。** 两个实例争 GPU 会让两边都废；
   中途换二进制会让同一轮里前后用的不是同一个东西。**改任何东西之前先确认基准已停**
   （`ps -eo pid,args | grep <脚本名>`）。
4. **zsh 的 `noclobber` 会让 `> file` 静默失败**（报「文件已存在」）。脚本里一律用 `>|`。
   不显眼是因为它看起来像"这一格没跑"。
5. **管道与重定向是块缓冲的**，后台任务的进度看起来会卡住。写独立日志文件，或等完成通知。
6. **`pkill -f "ninfer"` 会杀掉自己刚起的命令。** 用 `pkill -x ninfer`。
7. **`pgrep` 匹配不上长进程名**（comm 截断到 15 字符，`ninfer-perplexity` 变 `ninfer-perplexi`）。
   用 `ps -eo pid,args | grep` 更可靠。
8. **arena 容量必须同步。** 任何从 workspace 分配的新 scratch 都要计入
   `ternary_rotation_workspace_bytes()`（它是三元线性的容量声明）。漏了就在 CUDA graph 里越界。
   溢出会抛 `std::bad_alloc`（响亮失败），但别依赖这个兜底。

## 六、本机（4070 Ti SUPER 16 GB / 66 SM）实测的常量

换卡要全部重测。这些值是**这台机器**的。

| 量 | 值 |
|---|---|
| 只读带宽天花板 | **637 GB/s**（实测）；权重每趟 6.65 GiB |
| 三元 prefill 各档代价 | `s8` 近平（T=15→28：44.3→51.3 ms）；`small_t` 约 **20.5 ms × ceil(T/8)**；`wide_mma`(128宽) T=1024 时约 149 ms |
| small_t / s8 交叉点 | **T=17** |
| 128 宽 / 64 宽 tile 交叉点 | **T=64**（T=65 时 64 宽要第二趟） |
| decode 带宽 | 6.70 GiB / 13.75 ms = 487 GiB/s = 规格峰的 71% |
| 每个 decode token 的非矩阵乘开销 | 约 2.4 ms（其中旋转约 1.1 ms） |

## 七、改 kernel 时的硬约束

- **静态 shared 上限 48 KiB**（sm_89）。超了要用动态 shared + `cudaFuncSetAttribute`，
  而**那个调用在 CUDA graph 捕获里是非法的**。所以新 schedule 先算 `kSharedBytes`。
- **不要用 `__launch_bounds__` 的 min-blocks 换 occupancy。** 三处独立实测都指向"单调反向"
  （正式版 wide_t +34%、本机 GDN record 单调变慢、本机分块 GEMV 4 CTA 时崩到 68.2 t/s）。
  先减 shared，再看寄存器。
- **不要加宽 small_t 的 tile**——正式版实测是回归（45.2 → 19.0 t/s，累加器吃掉了驻留 CTA）。
  要覆盖面就用 token 循环。
- **`TernaryLaunch` 是一个函数指针类型**，两个入口点（t1/t8）签名必须一致。加参数要同时改。
- **prefill 是 eager 执行的**（只有三个 decode-batch 捕获入口），所以选档可以按当次 T 走，
  不需要在捕获时冻结；但 decode 路径上的选择**必须进程级冻结**（env 只读一次）。
