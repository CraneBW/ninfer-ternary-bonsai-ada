# SM86 性能基线（KVMem K0+K1 动工前测定）

- 日期：2026-09-25 10:20–10:26
- 机器：RTX 3060 Ti 8GB / CUDA 12.8 / MSVC 19.44 / commit `94bb6d2`
- 采样：`--greedy --no-thinking`，每项两轮；原始日志 `_bench/baseline_*.log`（不入库）

## 数字

| 配置 | PTQ1_0 (Q1) | PQ2_0 (Q2) |
|---|---|---|
| decode (t/s) | **4.8** / 4.8 | **26.4** / 26.4 |
| prefill ~2000 tok (t/s) | 14.9 / 14.7 | **396.8** / 396.4 |
| prompt tokens | 2017 | 2017 |
| engine ready | 4.1 s | 4.1 s |
| 权重驻留 | 5.52 GiB | 6.70 GiB |
| free after startup | 1.00 GiB | 12.0 MiB |

- PTQ1_0：`--max-context 4096 --kv-dtype int8`（默认 prefill-chunk 1024）
- PQ2_0：`--max-context 2048 --prefill-chunk 256 --kv-dtype int8`
- KV int8-group64；decode 短样本（greedy 20 token 即 EOS），方差 <2%
- 对照（laamaafung 同机，llama.cpp 体系，KVMem 开启）：PQ2_0 gen 38.6 t/s（4K 窗）、PTQ1_0 30.3 t/s（256K）

## 为什么 PTQ1_0 慢 5~26 倍（已定位，非 SM86 移植回归）

判定链在 `src/ops/linear/ternary/ternary_rowsplit_gemm.cu`：

1. `:135-151` —— 快速 kernel 家族（`launch_pq2_gemv` warp-per-row GEMV、
   `launch_pq2_gemv_tile`/`_tile_block`）的门槛是
   `w.qtype == PQ2_0_G128 && w.qhigh == nullptr && padded_shape[1] == k`。
2. `:119-124` —— PTQ1_0 与 PQ2_0 的通用路径都走 `launch_gemm<..., SimtDecodeAtom>`
   标量解码，但只有 PQ2_0 能进上面的快速家族。
3. 该条件只看格式，不看架构 —— 上游 Ada 构建同样如此。**PTQ1_0 慢是上游优化缺口，
   不是我们 SM86 移植引入的。**

结论：**后续 KVMem 对比实验以 PQ2_0 为基准模型**（它同时是 laamaafung 侧跑得最快的格式，
可比性最好）。PTQ1_0 的快速 kernel 家族是独立优化项，与 KVMem 无关，暂不阻塞。

## KVMem 改动后的验收口径（预记）

- decode t/s：PQ2_0 @2048ctx 相对 26.4 的变化（K1 直通模式应 ≤3% 损耗）
- prefill t/s：相对 396.8 的变化（每步 stage-in 会拉低首 token 延迟，另计）
- 功能：长对话（>32K 累积 token）场景下 KV 显存占用与生成质量对比
