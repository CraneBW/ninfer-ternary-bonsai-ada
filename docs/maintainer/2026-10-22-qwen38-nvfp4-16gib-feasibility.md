# OpenSpec 7.1：Qwen3.8 NVFP4 在单卡 SM89 的真实注册格式与16GiB容量门

**日期：2026-10-22。** 主机 RTX 4070 Ti SUPER，物理显存 **16,376 MiB = 15.992 GiB**，现役功耗 **200W**。本项仅核对已经注册的 Qwen3.8 NVFP4 **text** 形状、执行格式与GPU/Host/PCIe预算；目前没有已确认完整 Qwen3.8 NVFP4 的实际权重和模型启动收据，以下是**源码形状推出的下界**，不是实际某一 `.ninfer` 文件大小、装载成功或模型输出资格。已有 Qwen3.6 NVFP4 真实单卡结果不能作为 Qwen3.8 替代。

## 1. 格式与几何：不能用全参数 4bit 估算这个 profile

实际注册 `WeightsProfile::Qwen38Nvfp4` 使用 `src/targets/qwen3_6_27b/impl/load/bindings.cpp:606–656` 的 `bind_qwen38_nvfp4_text_layers`；配置 `config.h` 固定 **64 层、hidden 5120、intermediate 17408、48 GDN、16 full attention、词表 padded 248320**。注册绑定恰为：

| 层/张量 | 注册物理编码 | 数量 × 形状（以元素数） | 仅 codes/值的最小字节量 |
|---|---|---|---:|
| 前56层 MLP gate/up 与 down | NVFP4, 4bit code + 16元素一个FP8 scale | 56 × [(34816×5120)+(5120×17408)] | **7.844 GiB**（每元素 9/16 B；不含末尾divisor/对齐） |
| 最后8层 MLP | FP8 E4M3FN, BF16 row scales | 8 × 同上 | **1.992 GiB**（按每元素1B下界） |
| 48层 GDN input/output | FP8 E4M3FN, BF16 row scales | 48 × [(16384×5120)+(5120×6144)] | **5.156 GiB** |
| 16层 full attention input/output | FP8 E4M3FN, BF16 row scales | 16 × [(14336×5120)+(5120×6144)] | **1.563 GiB** |
| GDN 的 A/B 控制投影 | BF16 | 48 × (96×5120) | **0.044 GiB** |
| token embedding + vocabulary output head | FP8 E4M3FN, BF16 row scales | 2 × (248320×5120) | **2.368 GiB**（若两个独立 parent） |

仅上述大矩阵，**两份独立 endpoint 时 ≥18.967 GiB**；即使为避免误判而**保守假定 embedding/output head 完全同一物理 parent，只计一份**，下界仍 **≥17.783 GiB**，超过 GPU **15.992 GiB** 的全部物理显存 **至少1.791 GiB**。还没计入 FP8 row scales、NVFP4额外 divisor 与对齐、per-layer norm / conv、GDN state、KV、MTP/Vision、Graph 和 scratch。共享父权重/稀疏非物化可能影响实际装载方式，**不能据此声称特定缺失包的文件大小**；但对“以上所有注册 tensor 都在卡上完整常驻”假设，当前几何已足以排除。

`src/targets/qwen3_6_27b/impl/variant.cpp:53–61` 在 `NINFER_SM89` 分支选择 **`kNvfp4TextPolicy=A16Only`**，本机消费 NVFP4 codes 使用 BF16/A16 计算，`AllowA4` 是 Blackwell 路线，SM89 无真实 NVFP4 W4A4 Tensor Core资格。本机 `src/ops/linear/nvfp4/nvfp4_format.cpp` 对 parent 确认 `N%128=0, K%64=0, 16元素/FP8 scale` 和整包有效性；现有 `nvfp4` 是**权重格式**。`--kv-dtype nvfp4` 又是独立的**KV cache格式**，与权重编码无自动联系；当前 **KVMem 明确只支持 `--kv-dtype int8`**（`layouts_impl.h`），不能依靠 NVFP4 KV 获得相同的 213K KVMem 功能。W4A4 是第三项独立的激活/算子路线，本卡不适合宣称原生执行。

## 2. 已验证 GSQ 资源作为 runtime 下界参考（非新权重实测）

| 预算项 | 实际GSQ 213K MTP15 + INT8 KVMem 冻结记录 | Qwen3.8 NVFP4 的允许结论 |
|---|---:|---|
| Model text + MTP 常驻权重 | 11.62 GiB；`docs/maintainer/2026-10-22-canonical-replay-prepare.md` | 本 NVFP4 注册 text 权重**至少17.783GiB**，是否带额外 MTP/Vision 还需相应包 + feature 检查，不能从 GSQ 扣减假装已经适配 |
| Runtime reservation（**含**KV/Graph/GDN与workspace） | 2.09 GiB，**不是**可重复相加的第二组独立组件总额 | NVFP4 单卡无启动测量；即便预留降到0也容不下上述完全常驻权重 |
| 其中36,864 physical KV pool / INT8 | 1.58 GiB (=1,695,682,560 B)，KVMem 32K budget+512 gen reserve | 32K工作集所需的主物理KV不能因为缩小逻辑模型格式而消失；减少KV仅改变这一项，不能解决 text weight 的超额 |
| 其中 CUDA Graph allowance | 82 MiB | 仅 GSQ 已验图预算；NVFP4 Graph capture、Workspace、重放形状和并发需单独执行/测量 |
| 每个 GDN StateImage 的理论payload | 48×48×128×128×4B 的 FP32 recurrence =144MiB；48×10240×3×2B BF16 conv =2.813MiB，**合计约146.813MiB / slot** | 该简式源于固定拓扑，尚不含 offset、pad、Main/MTP state extra；实际 slot pool 的字节数必须由 `plan_state_image_device_pool` 查得，避免与既有 runtime总额双计 |
| 4个 Host replay checkpoints | 615,817,216 B（587.29MiB） | 属于主机 RAM，在 Host/StateImage tier 使用；不额外算作 GPU weight，重放仍要 H2D/BF16状态验证 |
| Source-K+Cold staging | 此次 OpenSpec 3.1/3.2 库存上界9,191,424 B（8.77MiB） | 属于 pinned Host 资源，不足以容纳/替代缺失的设备 text weight；见同日 layer-ready 报告 |
| 程序 RAM / swap 上限 | 本机日常实验预算 **20 GiB RAM / 2 GiB swap**，见已有 S0 清单 | 是否满足 NVFP4 代码、权重源包、Host extent、checkpoint 并存尚未实测；不能让页换入推理被称为合格全驻留 |
| 额外桌面/驱动及安全保留 | 既有容积门约0.86GiB + 0.25GiB 发布地板 | GPU 总额15.992GiB本就小于 text 最低17.783GiB，任何正保留都会扩大不足；不尝试 sudo/调高功耗规避 |

再重述先前 S0 的**假设值**：如果把27B参数每个都以 **0.5625B** 存放，重量约14.14GiB，配上 GSQ runtime2.09GiB +桌面0.86GiB+安全0.25GiB≈17.34GiB，也超16GiB；但**现役 Qwen38Nvfp4 注册的前56层NVFP4+其余FP8/BF16混合映射更大**，所以实际注册映射必须采用上面的17.783GiB保守下界，而不能继续引用14.14GiB当真。

## 3. Host/PCIe、允许的降档与拒绝条件

本轮 `nvidia-smi` 表明卡空闲时 PCIe 当前协商 **gen2×16**（可能是空闲省电协商，**不是有负载时的持续吞吐带宽**）。因此**没有可引用的真实 PCIe GB/s 或 NVFP4 weight offload 吞吐**。任何候选 Host offload 必须按实际重放需经 Host↔Device 的 payload 及 `measured_effective_BW` 计算搬运时间 `bytes / bandwidth`，逐层运行与首次加载分别计；只用 Btrfs reflink/checkpoint 逻辑大小不能代表物理读取字节。当前至少1.791GiB设备空间缺口是在**零 runtime**的极端宽松假设下，offload真实需求更大。以托管权重/UVM实现可启动也不意味着达到单卡常驻或可交付的吞吐/TTFT。

**容量门：** 现有 `src/runtime/engine/kv_capacity.cpp` 先用权重后可用设备字节 + 最小 runtime 计算可装下的工作集，并在预留不足时抛出明确的 `invalid_argument`；KVMem下 `--kvmem-budget` 也必须不大于 `--kv-device-tokens`。`src/targets/qwen3_6/impl/runtime/layouts_impl.h` 禁止非 INT8 KV 的 KVMem，对本 profile 开 `--kv-dtype nvfp4 --kvmem` 不能通过配置资格。保持拒绝即可，不能删除安全门换取表面启动。

| 目标配置 | 截至本轮资格 | 后续有意义条件 |
|---|---|---|
| 本注册 Qwen3.8 NVFP4，所有 text weights **全设备常驻** + GSQ 32K/KVMem | **静态不可行**（17.783GiB权重下界 > 15.992GiB总VRAM） | 换更大容量卡、真实不同物理编码/量化覆盖或真正有预算的卸载架构；不要以减小KV独立挽救 |
| 同模型但 8K/4K 设备 KV 工作集 | **仍无法全设备常驻**（weight自身超额） | 需要重量测量及明确的权重 offload吞吐/状态所有权方案；仅减小 KV 非完整方案 |
| Qwen3.8 NVFP4 + MTP/Vision/Graph | **NOT_RUN**，特征消耗只能更多 | 真实完整 model artifact、合法 MTP/Vision物理父层及独立显存/数值门后才测 |
| Qwen3.8 NVFP4 作为**CPU/Host托管**候选 | 有代码通路、**尚无实物/吞吐资格** | 用户提供完整权重后先分析页驻留/Host RAM、带宽、PCIe H2D工作量，再决定是否值得启用 |
| SM89 NVFP4 weight + BF16 A16 推理 | 软件路线存在，Qwen3.6 另模型已验 | Qwen3.8 全格式仍无数据；不得混淆成 W4A4 或 NVFP4 KV |

**本项仅 7.1**：以后用户提供或授权取得完整 Qwen3.8 NVFP4 包时才进入 `7.2` 的模型/codec/实物资格；`7.3–7.4` 的数值与性能测试保持 **NOT_RUN**，GPU功耗维持200W，300W只留最终发布验收。没有下载权重、建替代包、修改数学或启动不存在的模型。
