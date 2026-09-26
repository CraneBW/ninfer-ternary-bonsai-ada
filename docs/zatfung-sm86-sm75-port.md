# zatfung：SM86 / SM75 移植方案

> 目标：让 zatfung 在同一份源码树上支持 `75` (Turing) / `86` (Ampere) / `89` (Ada) / `120a` (Blackwell)，
> 其中 **SM86 是本机 RTX 3060 Ti 的主力目标**，SM75 作为最保守档位一并打通。
>
> 基线仓库：`G:/Agents/ninfer-works/zatfung`（原 `ninfer-ternary-bonsai-ada`）
> —— 唯一同时具备 **Windows/MSVC 工程**与**完整三元 Bonsai 实现**的源码树。

---

## 1. 为什么是这个基线

| 候选仓库 | Windows | 三元 Bonsai | SM86 | 结论 |
|---|---|---|---|---|
| **zatfung**（bonsai-ada） | ✅ MSVC + CMake | ✅ 全量（`src/ops/linear/ternary/`，14 文件） | ❌ 待补 | **基线** |
| ninfer-3090 | ✅ | ❌ 0 处引用 | ✅ 原生 | 版本(0.6.1 线)落后，搬三元子系统代价高 |
| ninfer-4090-windows | ✅ 100% native | ❌ | ❌ sm_89 | 三元缺失 |
| ninfer-2080ti-22g | ❌ 纯 Linux | ❌ | ✅ | **SM75/SM86 算子来源**（见下） |
| ninfer-ternary-…-3090（may-has-bug） | ✅ | ⚠️ 仅补丁 | ✅ | `src/ninfer` 子模块未检出，只能当参考 |

**关键发现**：`ninfer-2080ti-22g` 是一份已经打通的 **Turing port**，其 CMake 明确支持
`75 / 86 / 120a` 三档，并已具备一整套编译期降级机制。它的 `src/core/arch_traits.cuh`
就是为这件事写的架构特性表。这让 SM75/SM86 的移植从"从零适配"变成"门控合并"。

---

## 2. 已完成：构建与门控层（Phase 0）

### 2.1 CMake 骨架（已完成）

`CMakeLists.txt` 的改动：

1. 架构白名单 `^(120a|89)$` → `^(120a|89|86|75)$`
2. 新增架构宏，与 2080ti 线**同名**（这是后续合并门控代码的前提）：

   | 架构 | 宏 | 共享内存档位 |
   |---|---|---|
   | 75 | `NINFER_SM75` | 64 KiB/SM，无 `cp.async`，无 bf16 MMA |
   | 86 | `NINFER_SM86` | 100 KiB/SM，有 `cp.async`/bf16 MMA |
   | 89 | `NINFER_SM89` | 100 KiB/SM，capped w8 变体 |
   | 120a | （不定义） | 高内存调度，NVFP4 TMA |

3. `project(ninfer)` → `project(zatfung)`
4. **CUDA 版本逃生舱**：新增 `option(NINFER_ALLOW_LEGACY_CUDA)`。上游硬要求 ≥ 13.1；
   非 120a 目标放宽到 12.8，120a 仍然强制 ≥ 13.1。硬下限 12.8。
   > **⚠️ 更正一处早期判断**：起初认为 13.1 是"纯策略性门槛"，首次全量编译证明**不完全是** ——
   > 代码里确有 4 处真正需要 CUDA 13 的写法（`constexpr dim3`，见 §7.3）。
   > 已把这 4 处改为 `const`，使 12.8 可用；但若有更多 CUDA 13 专有 API 未暴露，
   > 升级到 CUDA 13.x 仍是更彻底的路线。
5. **宿主编译器逃生舱**：新增 `option(NINFER_CUDA_ALLOW_UNSUPPORTED)`，
   透传 nvcc 的 `-allow-unsupported-compiler`（应对 nvcc 白名单比已装 MSVC 旧的情况）。
6. **ffmpeg 能力探测**：源码树里没有 `ffmpeg/{include,lib}` 时自动关闭媒体解码
   （`decode.cpp` 全身由 `#ifdef NINFER_HAVE_FFMPEG` 包住，关闭后可编译），
   避免 `LNK1104: 无法打开 avcodec.lib`。

`src/CMakeLists.txt` **无需改动**：它的三处架构判断都是 `MATCHES "^120"` 形式，
75/86/89 自动落到"架构通用内核集 + stub"分支。

### 2.2 构建器（已完成）

`builder.go` 作为生产标准构建器，已实测完成本机 `-configure-only` 全流程
（含自建 MSVC 环境、Ninja 生成、CUDA 12.8 放宽、ffmpeg 自动降级）。
详见文件头注释。

---

## 3. 待办：内核层门控合并（Phase 1–3）

### 3.1 实测结论：SM86 的缺口远比预期小

zatfung 的源码**已经为 SM86 预留了门控**。全量统计 `NINFER_SM89`（30 处）：

| 形态 | 处数 | 状态 |
|---|---|---|
| `defined(NINFER_SM86) \|\| defined(NINFER_SM89)` | 21 | ✅ 已兼容，无需改动 |
| 纯 `defined(NINFER_SM89)` | 9 | ⚠️ 需逐处判断 |

9 处纯 SM89 中：

- `src/core/device.h:24` —— 下一行已有 `#elif defined(NINFER_SM86)`
  （`kTargetSmCount = 82`，RTX 3090），**实际已兼容**，不用改。
- 其余 8 处集中在 `kv_cache/append/{kernel.cuh,launch.cu}` 与
  `softmax_attention/dense/causal_cache/{prompt.cu,prompt_i8.cuh,small_t.cu,small_t_i8.cuh}`，
  语义完全一致：**Ada 线（sergiuszm/ninfer-4090）引入的可选 i8 KV 量化内核**
  （`rk4v4` / `rk4v4-e8` / `Int8Group64` 存储模式），属**增强路径，不是基础路径**。

> ⚠️ **配对陷阱（9 处里唯一需要小心的地方）**
> `prompt.cu:28` 是 `#if !defined(NINFER_SM89)` —— 它在 SM86 构建下**会被启用**，
> 而它设置 smem attribute 的目标 kernel 却定义在 `#if defined(NINFER_SM89)` 分支里。
> 两处条件必须同步修改，否则会报"引用了未定义的 kernel"。

**因此 SM86 的移植策略是「先保功能、后保增强」：**

1. 基础路径（bf16 KV + 三元权重）在 SM86 下应当直接可用 —— 这也是首次全量编译
   到 158/527 仍零错误的原因；
2. i8 KV 增强存储模式在 SM86 下临时关闭（保留枚举与接口，不编译其内核）；
3. 待端到端跑通后，再评估把 `rk4v4`/`rk4v4-e8` 这批 i8 内核移植到 SM86
   （它们是 MMA + 位运算 + 格点编解码，理论上不依赖 Ada 专有指令）。

### 3.2 待合并的门控清单（来自 2080ti 线）

2080ti 侧共 13 个文件、42 处 `NINFER_SM75`/`NINFER_SM86` 门控，与 zatfung 重叠的核心文件：

| 文件 | 移植内容 | 风险 |
|---|---|---|
| `src/core/arch_traits.cuh` | **整文件新增**。提供 `ArchTraits<SM75/SM86/SM120A>` 与 `CurrentArchTraits` 别名 | 低（纯编译期常量表） |
| `src/ops/common/memory.cuh` | SM75 无 `cp.async` 的同步回退路径（`#if !defined(NINFER_SM75)` 包裹 cp.async 用法） | 中（性能路径） |
| `src/ops/common/mma.cuh` | SM75 的 mma tile 形状降级（fp16 m16n8k8 / int8 m8n8k16）、bf16 MMA 回退 | 中 |
| `src/ops/common/math.cuh` | 精度/指令降级辅助 | 低 |
| `src/ops/linear/w8/w8_config.h` | w8 调度档位表：把 `NINFER_SM89` 判定扩为受限档位 | 低（与 zatfung 结构同源） |
| `src/ops/linear/w8/w8_rowsplit_gemm_splitk.cu` | 同上，splitk 档位 | 低 |
| `src/ops/linear_add/w8/…`、`src/ops/linear_pair/w8/…` | 同上 | 低 |
| `src/core/pdl.cuh` | PDL 已有 `prop.major >= 9` 运行时判定 + `__CUDA_ARCH__ >= 900` 编译期判定，**SM75/86 自动退化为普通流内 launch，无需改动** | 无 |

> `arch_traits.cuh` 需要**增加 SM89 特化**（2080ti 线没有 Ada），
> 数值可比照 SM86 档位（Ada 同为 100 KiB/SM 档）。

### 3.3 需要新排查的 zatfung 侧文件

这些文件在 2080ti 线里没有对应物（两线 attention 组织方式不同：zatfung 是
`src/ops/softmax_attention/dense/causal_cache/`，2080ti 是 `src/ops/kernel/gqa_attention_*.cuh`），
SM75 下需逐个确认是否存在隐式 sm80+ 依赖（`cp.async`、`ldmatrix` 变体、`__nv_fp8`）：

- `src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh`（含 7 处 `__CUDA_ARCH__` 门控）
- `src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh`
- `src/ops/softmax_attention/dense/causal_cache/prompt.cu`、`small_t.cu`
- `src/ops/kv_cache/append/kernel.cuh`、`launch.cu`
- `src/ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu`、`…_splitk.cu`
- `src/ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_materialized.cu`
- `src/core/device.h`

### 3.4 SM75 比 SM86 难一个量级

SM86 能"捡便宜"是因为 zatfung 源码本就以 `SM86 || SM89` 成对判断。SM75 不同：
现有代码里**完全没有 SM75 的概念**，且有几处硬门槛必须先拆掉：

| 门槛 | 位置 | 处理 |
|---|---|---|
| `#error "NInfer requires NINFER_SM86 or NINFER_SM89"` | `src/core/device.h:29` | 加 `NINFER_SM75` 分支（`kTargetSmCount`：2080 Ti 为 68 SM；若目标卡非 2080 Ti，宜改为运行时 `device_sm_count()`） |
| `arch_traits.cuh` 缺失 | 需从 2080ti 线引入 | 整文件新增（含 SM89 特化），提供 `has_cp_async=false` / `has_bf16_mma=false` 等判据 |
| `cp.async` 无条件使用 | `src/ops/common/memory.cuh` 及各 attention 内核 | 走 2080ti 线的 `#if !defined(NINFER_SM75)` 回退分支 |
| bf16 MMA | `src/ops/common/mma.cuh` | SM75 无 bf16 tensor core，需回退到 fp16 MMA（`m16n8k8`） |
| 静态/动态 smem 上限 | 全内核 | 64 KiB/SM（vs SM86 的 100 KiB）—— 现有 capped 档位按 48 KiB 设计，SM75 下需重算 occupancy |
| 三元内核 tile 形状 | `src/ops/linear/ternary/*` | 确认是否假定 `m16n8k16`；SM75 的 fp16 MMA 是 `m16n8k8`、int8 是 `m8n8k16` |

**结论**：SM86 是本轮主要目标且已接近可用；SM75 建议作为独立阶段推进，
以 `ninfer-2080ti-22g` 为对照物逐文件比对（它是唯一已知可用的 Turing 实现）。

### 3.5 三元内核（Bonsai 本体）的架构兼容

`src/ops/linear/ternary/` 的 14 个文件走 `mma.sync` + `ldmatrix`，**SM75 起即支持**，
从指令集角度看无阻塞。但要注意：

- `ternary_rowsplit_mma.cuh` / `…_mma_small_t.cuh` 的 tile 形状是否匹配 SM75 的
  `m16n8k8`（fp16）/`m8n8k16`（int8）—— 若假定了 `m16n8k16`，需要走 `ArchTraits` 分支
- 三元格式跨 128 权重一组带 fp16 scale，与 `cp.async` 的搬运粒度耦合，
  SM75 走同步回退路径时需要确认没有隐藏的异步依赖

---

## 4. 实施顺序与验收

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P0** ✅ | CMake 门控 + builder.go | `builder.exe -configure-only` 通过（已实测） |
| **P1** | 引入 `NINFER_SMEM_CAPPED`，完成类别 A/B 拆分 | sm_86 全量编译通过 |
| **P2** | 合并 `arch_traits.cuh`（含 SM89 特化）+ 公共原语层门控 | sm_86 编译通过 + `ninfer --help` 可跑 |
| **P3** | 三元内核 tile/寻址的 SM86 分支 | sm_86 下加载三元权重并出 token |
| **P4** | SM75 全链路（同期排查 §3.3 清单） | sm_75 编译通过 + 2080 Ti 实机验证 |

**验收口径**（每个架构档都要过）：

1. `builder.exe -arch <id>` 全量构建无错误
2. `ninfer-perplexity` 在 Bonsai 2 27B 三元权重上跑出合理 ppl（对标 dense 27B 基线）
3. 对照 `ninfer-ternary-…-3090` 工作区的实测常量（预填 1.34~1.36k tok/s @ 2×3090）
   判断 SM86 路径是否退化

---

## 5. 已知风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **本机显存只有 8 GB** | RTX 3060 Ti 8GB，而三元 27B 权重约 7.12 GiB —— 权重几乎占满，KV 几乎没有余量 | 用短上下文（`--max-context 4096`）+ `--kv-dtype int8` 先打通功能，性能验证另找 24 GB 卡 |
| `mma.cuh` 两线已分叉 201 行 | 直接覆盖会破坏 Ada/Windows 线既有优化 | 只取门控分支做 `#elif` 合并，不整文件替换 |
| `w8_config.h` 分叉 129 行 | 同上 | 同上 |
| CUDA 12.8 + MSVC 19.44 | 不经上游验证的组合（已实测可编译 sm_86 目标文件） | 保留 `-cuda-allow-unsupported` 与升级 CUDA 13.x 两条路 |
| `reg.exe` 被安全策略拦截 | nvcc 内部调用，不影响产物（实测 .obj 正常） | builder.go 已在日志里识别并解释该噪声 |

---

## 6. 实测记录（2026-09-25，本机）

| 项目 | 结果 |
|---|---|
| GPU | RTX 3060 Ti，compute_cap 8.6，8192 MiB，驱动 616.92 |
| CUDA | 12.8.93（`/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8`） |
| MSVC | VS2022 Community，工具集 14.44.35207，宿主编译器 19.44.35228.0 |
| Windows SDK | 10.0.26100.0 |
| CMake / Ninja / Go | 3.31.12 / VS 自带 / 1.27.0 |
| `nvcc -arch=sm_86 -std=c++20` 编译试探 | ✅ 产出 40 KB `.obj`（CUDA 12.8 + MSVC 19.44 组合可用） |
| `builder.exe -configure-only` | ✅ 100 s configure 完成，产出 `_build_86/` |
| VS 生成器 | ❌ `No CMAKE_C_COMPILER could be found`（cmd.exe 被禁 → 无法走 vcvars 通道）→ 故默认 Ninja |

---

## 7. 已实施的代码改动

### 7.1 FP8 A8 / FP8 KV 的架构剔除（首次编译的实证结果）

首次 sm_86 全量编译推进到 FP8 内核时停机，错误非常明确：

```
ptxas ... error : Feature 'mma with FP8 floating point type' requires .target sm_89 or higher
ptxas fatal   : Ptx assembly aborted due to errors
small_t_fp8.cu
```

FP8 tensor core 自 Ada (sm_89) 起才有，sm_75/sm_86 没有这条通路。

**新增**：`src/ops/fp8_legacy_stubs.cpp`（10 个入口的替身，全部抛精确的运行时错误），
并在 `src/CMakeLists.txt` 加入剔除块。

剔除的 7 个翻译单元（**均为 FP8 A8 激活路径**，共两条正则）：

| 正则 | 命中文件 |
|---|---|
| `fp8/.+_a8\.cu$` | `linear/fp8/fp8_a8.cu`、`attn_input_proj/fp8/fp8_attn_input_a8.cu`、`gdn_input_proj/fp8/fp8_gdn_input_a8.cu`、`linear_add/fp8/fp8_linear_add_a8.cu`、`linear_swiglu/fp8/fp8_linear_swiglu_a8.cu` |
| `causal_cache/(prompt\|small_t)_fp8\.cu$` | `prompt_fp8.cu`、`small_t_fp8.cu` |

**保留**（同为 FP8 但走 A16 反量化，解成 bf16 后做 bf16 MMA，可正常编译）：
`ops/linear_topk/fp8.cu`、`ops/linear_topk/fp8_m64.cu` —— 证据是它们调用
`fp8_e4m3x2_to_bf16x2_bits()` 而非 FP8 MMA。

> ⚠️ **与 `ninfer-3090` 的关键差异（容易照抄错）**
> 3090 的门控是 `^(86|89)$`，连 sm_89 一起剔除，因为那条线从 Blackwell 上游派生，
> FP8 用的是 `mma.sync...kind::f8f6f4`（Blackwell-only 限定符）。
> zatfung 源自 **Ada 线**，用的是 sm_89 档的 FP8 MMA（ptxas 的报错措辞
> "requires .target **sm_89** or higher" 就是证据），所以 **sm_89 必须保留**。
> → zatfung 的门控是 `^(75|86)$`。

副作用（可接受，需在文档与 CLI 提示里保持一致）：sm_75/sm_86 上
`--kv-dtype fp8` 会在启动/首次调用时抛运行时错误，应改用 `--kv-dtype int8` 或 `bf16`。

### 7.2 arch_traits.cuh 落地

新增 `src/core/arch_traits.cuh`（自 `ninfer-2080ti-22g` 引入并补上 **SM89 特化**），
提供 `ArchTraits<SM75|SM86|SM89|SM120A>` 与 `CurrentArchTraits` 别名。

当前是"已就位、随门控合并逐步接入"的状态：现有代码继续用 `NINFER_SM*` 宏工作，
两者并存不冲突。它是后续把所有 `#if defined(NINFER_SMxx)` 收敛成
`if constexpr (CurrentArchTraits::has_cp_async)` 的判据来源 —— 语义化的写法能让
"这档架构有没有某能力"一眼可见，而不是靠宏名猜。

### 7.3 `constexpr dim3` → `const dim3`（4 处）

剔除 FP8 之后，编译撞上第二类错误，而它**与架构无关**：

```
error: a constexpr variable must have a literal type or a reference type
      constexpr dim3 block(Storage::kGroupK, 1u, 1u);
note: cannot call non-constexpr function "dim3::dim3(unsigned, unsigned, unsigned)"
```

`dim3` 的构造函数**自 CUDA 13 起才标记为 `constexpr`**，CUDA 12.8 的不是。
这就是上游 CMake 里 `CUDA >= 13.1` 门槛的真实来源之一 —— 它不是空穴来风的策略要求。

改动：4 处 `constexpr dim3 block(...)` → `const dim3 block(...)`，语义不变 ——
`block` 只作为 kernel launch 配置（`kernel<<<grid, block, smem, stream>>>`），
从不参与常量表达式求值；同文件里紧邻的 `grid` 本来就写作 `const dim3`。

| 文件 | 所属路径 |
|---|---|
| `src/ops/linear/ternary/ternary_rowsplit_gemm.cu` | **三元/Bonsai 核心** |
| `src/ops/linear/q4/q4_rowsplit_gemv.cu` | Q4 量化 |
| `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu` | Q4/Q5 量化 |
| `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu` | Q4/Q5 量化 |

> 取舍说明：4 处局部改动 vs 安装 6 GB 的 CUDA 13.x。选前者 —— `const` 在 CUDA 13 下
> 同样合法，所以这个改动**不牺牲未来升级路径**；而升级工具链则是一次性大动作，
> 留待出现"无法用局部改动绕过"的 CUDA 13 依赖时再做。

### 7.4 补齐 `decode.cpp` 的无 ffmpeg 分支（链接缺口）

编译推进到 **223/225**，只剩最后一步链接时失败：

```
ninfer_engine.lib(processor.cpp.obj) : error LNK2019: 无法解析的外部符号
  ninfer::media::decode::inspect_image(std::span<unsigned char const,-1>, Policy const&)
  ninfer::media::decode::inspect_video(std::span<unsigned char const,-1>, Policy const&, double, int, int)
apps\ninfer.exe : fatal error LNK1120: 2 个无法解析的外部命令
```

根因：`decode.cpp` 的无 ffmpeg 分支（`#else`）**只补了 `decode_image` / `decode_video`
两个 stub，漏了 `inspect_image` / `inspect_video`**。这两个是被 qwen3_6 前端的
`Processor::count_tokens`（token 计数）与视觉项探测引用的，而它们的定义整段位于
`#ifdef NINFER_HAVE_FFMPEG` 内。上游发布包总是自带 ffmpeg，所以这个缺口从不暴露 ——
恰好被"ffmpeg 目录缺失 → 自动关闭媒体解码"这条路踩了出来。

**修复**：在 `#else` 分支补上这两个 stub（抛出一条说明"本构建未启用 FFMPEG"的运行时错误），
使 `ninfer_media_decode` 在关闭媒体解码时仍然链接完整。

> 这类缺口的通用形态值得记住：**同一个 `#else` 分支里，stub 的覆盖面必须与头文件声明的
> 公开接口完全一致**。上游漏了两个，只有在"真的没有 ffmpeg"的构建里才会以 LNK2019 现形。
