# tools/verify — 独立数值验证

这棵树的判据原来只有两类：**PPL** 与**贪心 token ids 的 sha**。两者都对
"权重在旋转基里被放错了位置"这类错误**不够敏感**——402 个三元矩阵全部以旋转基存储，
所以**相关性检验对它们无效**，只有逐字节对账、解码往返相等、以及引擎的加载测试能用。

这里补上第三类：**独立 oracle**（用 numpy 显式重建文档化的变换，与设备代码的元素逐对比较）
和**结构检查器**。

## 来源与改动

来自 `shensanshu/ninfer-ada-ternary`（ModelScope）的 `tools/verify/`。
逐文件改动如下——**没有一处是"照着抄"**，每处都注明了理由：

| 文件 | 改动 |
|---|---|
| `check_signs.py` | 去掉作者硬编码的 Windows 源码根，改成由本文件位置推出（树可搬） |
| `oracle_rot.py` | **改 token 布局约定**：作者用 `reshape(k, tokens)`（C 序，连续内存 = 连续 token），本引擎是 **token-major**（`token * k + column`，见 `ternary_rotation_kernels.cuh` 的 LAYOUT 注释）。**两者在 T=1 时恰好重合**，所以作者的版本在它的两个 T=1 用例上通过、三个 T>1 用例上全部失败——那是约定差，不是内核缺陷 |
| `harness/rot_test.cu` | 输出目录从编译期常量改成 `argv[1]`；去掉 MSVC 专用的 `/wd4819` |

## 怎么用

```bash
# 1) 符号表：只需 .ninfer + hadamard 元数据，秒级
python3 tools/verify/check_signs.py <artifact.ninfer> <bonai2-hadamard-meta.json>
#    期望：signs identical to meta: True   max|diff|=0.0

# 2) 旋转内核 oracle：编译 + 跑六个用例 + numpy 比对
bash tools/verify/run-rot-oracle.sh [dump-dir]
#    期望：RESULT: 6/6 cases pass，且每条 WRONG 变体都 differs OK
```

**依赖**：`numpy`；oracle 那一步还要 `nvcc` 与一块能跑 sm_89 的卡。

## 为什么 `oracle_rot.py` 值得优先跑

它的六个用例里 **`plain_t3` / `perm_t2` / `inv_t2` 是 T>1 的**。
T=1 时 token-major 与 row-major 布局重合，所以**只测 T=1 的话，真实 prefill 全错也能全绿**——
这正是本仓库此前没有任何手段能发现的那类失效（`ternary_rotation_kernels.cuh` 的
LAYOUT 注释把这条陷阱写在了代码里，但一直没有测试去钉它）。

判据是**双侧**的：文档化的变换必须匹配（`rel_l2 <= 0.01`），**并且**每一个"看似合理但错"的变体
都必须被分离（`> 10x`）。单侧判据分不清"内核是对的"和"oracle 太松"。

## 还没搬的

| 文件 | 为什么没搬 |
|---|---|
| `check_payload_order.py` | 逻辑可用，但它的 `first_code_zero_share` 是 **1/4 子样估计器**，在 `gdn/value_z` 上给出 0.3273、落到它自己的"正确区间"之外（全量值是 0.327792）。搬之前要把抽样改成全量。另：作者文档里的 `high_byte_kinds = 11~32` 判据**实测与理论都是 8**，别照抄 |
| `check_row_order.py` / `check_assembly.py` | 依赖未发布的 `_ternary_ref.Gguf` 模块。逻辑有价值（尤其 `repeated`/`missing` 那套对非双射置换的正面检测），但要配一个自己的 GGUF reader 才能跑 |
| `check_embedding.py` | 需要引擎侧 dump 钩子（要改源码） |
| `gemm_oracle.py` | 只覆盖 SIMT 兜底内核；本引擎的 prefill 走 MMA、verify 走 small_t，都覆盖不到。要补得按同样方法学新写 |
