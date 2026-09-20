[![English](https://img.shields.io/badge/Language-English-2ea44f?style=flat-square)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-d73a49?style=flat-square)](README.zh-CN.md)

# 三元 Bonsai 2 27B 跑在 **NINFER** 上（Ada / `sm_89`，Linux）

> **三元量化 —— 每权重 2.125 bit —— 的 Bonsai 2 27B 移植**，落在 **NINFER** C++20/CUDA 推理引擎上，
> 目标平台是 Ada Lovelace（`sm_89`）下的原生 Linux。
>
> **本仓库是衍生作品，不是 NINFER 上游。** 英文 README 里横线以下的部分是上游 README 原文，未作改动。

---

## 这是什么

**NINFER** 是一个单卡 C++20/CUDA 推理引擎。这个分支把 Bonsai 2 27B 的三元（PQ2_0）量化落到
**NINFER** 的 Ada 线上，补上三元格式需要的 tensor-core 路径，并且每一处改动都在实机
（RTX 4070 Ti SUPER）上量过。

三元权重按 `{−1, 0, +1}` 打包成每码 2 bit，外加每 128 权重一组一个 fp16 scale。

这里有两个容易混的数：这类三元权重通常被称作 **1.58-bit**，那是 **log₂3** —— 一个三元符号的
信息量；而**实际存储**代价（把码和分组 scale 都算上）是 **每权重 2.125 bit**。
两个数都对，量的是不同的东西——**内存系统真正付出的是后一个**。

这大约是 4-bit 格式一半的权重流量，也正是下面那些数字的来源：
模型每验证轮要读 7.12 GiB 权重，而这张卡的实测读带宽是 637 GB/s。

| | |
|---|---|
| 引擎 | **NINFER** — v1.0.8 Ada 线 |
| 权重 | 三元 Bonsai 2 27B，`PQ2_0_G128` |
| 硬件 | RTX 4070 Ti SUPER（16 GiB，`sm_89`，实测读上限 637 GB/s） |
| 工具链 | CUDA 13.4、GCC 15、Linux |

## 项目谱系与致谢

这项工作完全建立在 **NINFER** 及其周边分支之上。按谱系顺序：

| 项目 | 贡献 |
|---|---|
| **[Neroued/ninfer](https://github.com/Neroued/ninfer)** | **NINFER 规范上游** —— C++20/CUDA 架构、DFlash2、ReplaySSM、Paged KV Cache。Apache-2.0。 |
| [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | 最早的 RTX 4090 分支；WDDM 可驱逐预算绕过、E8 格 `rk4v4-e8` KV 存储 |
| [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) | Ada `sm_89` 内核优化、`rk4v4-e8` 适配、GDN 协作启动修复 |
| [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) | Win32/MSVC 可移植层、无缓冲异步 I/O |
| [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) | 原生 Windows MSVC 编译基座 |
| [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) | Ampere 早期工作与兼容桥接 |
| **[Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows)** | **源码树的直接基座** —— 本分支的引擎代码从这里开始 |

上面两条是本仓库的**直接输入**，应当排在其余之前单独点名：

- **[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary)（魔搭 ModelScope）**
  —— 标题《NInfer on Ada · 三元 Bonsai 2 27B 实战移植》，Apache-2.0。**三元移植本身的出处**：
  它的 `patches/` 是引擎侧改动、`tools/` 是打包器与验证工具、`docs/` 是技术记录。
- **Ambolio/ninfer-4090-windows** —— 引擎源码树的来源。

### 模型权重：本仓不分发

模型是 **Ternary Bonsai 2 27B**，底座 `Qwen/Qwen3.8-27B`，架构未改，权重为三元量化 + Hadamard 旋转基。
**权重版权归其原作者与发布方所有 —— PrismML 及其上游 Qwen 体系 —— 本仓库不包含、也不重新分发任何模型权重。**
请自行从官方渠道获取，并遵守其各自的许可条款（可能并非 Apache-2.0）。由权重产出的 `.ninfer`
制品属于**权重派生品**，其再分发义务**以权重原许可为准，与代码许可无关**。

### 方法参考

三元编解码语义对齐 llama.cpp 生态的 `ggml-quants.c`；折叠 Hadamard 基的语义参考 PrismML 的公开运行时
与其 `prism.hadamard.*` 元数据契约；张量核 FWT 的设计思路受公开的 HadaCore / TurboQuant 工作启发。
以上仅为**方法参考**，本仓代码为独立实现。

上游的 `NOTICE` 与 `LICENSE` 原样保留。本分支修改过的每个文件都在顶部带上显著声明，
这是 Apache-2.0 §4(b) 的要求。

## 这个分支改了什么

在上游基线上叠了 7 个提交，每个提交的信息里都带着它自己的实测数据：

- **三元 tensor-core prefill 路径** —— 同等权重下比 blocked GEMV 快 6.8 倍，再经 NCU 指导的调优
  又快了 3.51 倍（对应 `prefill-3.2x`、`prefill-3.51x` 两个 tag）。
- **投机验证轮的 small-T tensor-core 路径。** prefill 内核把 token 轴按 128 切，在 T=3 时有 97% 是空的；
  验证路径改为把整个 K 放进一个 CTA，于是所有草稿 token 只读一遍权重。decode 速度大部分来自这里，
  它也是让 MTP 从净亏转为净赚的那一步。
- **两轮靠读 SASS 找出来的 decode 优化**（不是靠推理）：一处被编译器展开成两个分支加一条 SEL 的三元选择，
  以及一个更早版本 magic 常量残留的 bias 前缀。两者进出都逐位一致。
- **旋转内核的发射打包** —— 旋转是「一个 warp 负责一个 (K块, token) 对」，grid 由数据定死，
  唯一的自由量是这些 warp 怎么打包；按 8 个一块时，一个 decode 形状的旋转会把 20 个 warp
  压在 66 个 SM 里的 3 个上。

## 规格速查

```
设备        NVIDIA GeForce RTX 4070 Ti SUPER · 16 GB (16376 MiB) · sm_89 (Ada) · 驱动 615.71.09
模型        三元 Bonsai 2 27B · 每权重 2.125 bit · 原生上下文上限 256k

解码速度    100.8 t/s      MTP draft 3 · 300 token · en-code
填充速度    1230  t/s      prefill · tensor-core 路径
显存占用    7.12 GiB       权重（其中 MTP 层 0.42 GiB；不开 --spec 则为 6.70 GiB）
上下文      120k token     bf16 KV 实测上限（128k 起不来）
            238k token     fp8 KV 实测上限（240k 起不来）

长上下文召回  128k 6/6 · 160k 5/6 · 200k 3/6
              （6 针大海捞针；200k 两次复现）
```

**以上每一个数都是本机实测。** 上下文上限来自 `--kv-capacity auto`——它按显存自算并在放不下时
明确报错；每个上限两侧的值我都试过。

**KV 格式买到什么、没买到什么**：fp8 把可达上下文翻倍，但上面的召回数据是 **bf16** 下测的，
**fp8 KV 是数值上的取舍，本机没有测过它对召回的影响。**

## 本机实测

Decode，`en-code.json`，300 token，MTP draft 3，取两轮中较好的一次，且各臂的接受率统计完全一致：

| 配置 | decode |
|---|---:|
| 验证轮走 SIMT tile 内核 | 43.2 t/s |
| 验证轮走 small-T tensor-core 路径 | 89.9 t/s |
| + draft 窗口调优与 SASS 驱动的 decode 工作 | 99.0 t/s |
| + 旋转发射打包、行块复用 | **100.8 t/s** |

Prefill：tensor-core 路径 **1.23k t/s**（blocked GEMV 177 t/s，参考内核 50 t/s）。
数值等价性：相对参考 prefill 内核的 PPL 差为 0.015%。

## 试过但无效的

记在这里，因为这几个负面结论花的工夫不比正面结论少，而且其中三条是结构性的、不是调参问题：

- **靠加宽行块降低验证内核的激活流量。** 按搬运比例推算，激活占 L2 流量的三分之二；
  实际把激活降了三分之一，有效带宽只从 453 涨到 480 GB/s（对 637 的上限而言）。这个内核不是 L2 带宽受限的。
  它确实带来的那 0.45%，是靠引擎级 A/B 保留下来的，不是靠那套理论。
- **提高 GDN record 内核的驻留 warp 数。** 16/48 warp 的占用率看着就是瓶颈；把寄存器压下去
  以塞进 32 个 warp，两轮结果都是**单调变慢**。它受限于对已接受 token 的串行递推，不是驻留数。
- **融合旋转调用。** **NINFER** 本来就这么做 —— 一次旋转同时服务四个注意力投影，
  因为它们的激活宽度相同。

最大的一笔开销 —— 验证轮 13.3 ms 对 10.7 ms 的地板 —— 我没能找到可用的杠杆。
这不是因为没量：达到 80% 吞吐效率需要 117 t/s。

---

> 上游 NINFER 4090 Windows 的完整英文 README（引擎特性、构建方式、Windows 相关内容）
> 请见 [README.md](README.md)。本页是中文版导览，只覆盖本仓库自身的内容。
