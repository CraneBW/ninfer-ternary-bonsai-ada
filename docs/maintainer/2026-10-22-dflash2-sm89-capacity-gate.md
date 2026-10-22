# S3 / OpenSpec 4.1：Qwen3.8 GSQ DFlash2 在 16 GiB SM89 的容量门

**结论范围：** 复用 2026-10-20 已运行的同一完整 GSQ v3 模型单卡短请求和已验 213K KVMem 资源记录，仅做 16GiB 静态门；没有重新加载模型或重复已经通过的短模型测试。本项不把 DFlash2 小上下文资格扩大为 32K working-set / 213K / Graph 资格，后续真实 profile 仍按 4.2–4.5 单独验收。

## 数据与代码路径

| 已有实物/规格 | 数量与来源 | 容量含义 |
|---|---:|---|
| GPU 物理容量 | 16,376 MiB（本轮 `nvidia-smi`，RTX 4070 Ti SUPER / SM89，200W） | 不是可用的独占空闲内存；显存还有桌面和驱动开销 |
| 无 Vision 的 GSQ v3 + DFlash2 | **13.0 GiB weights**、**526.1 MiB runtime**、**180.2 MiB free**；`tmp-nvcc/gsq-v3-full-dflash2-smoke.log` | 512-token KV、`--spec dflash2`、短请求真实输出，2/2 draft accepted；非长上下文基线 |
| Vision + DFlash2 | **13.3 GiB weights**、**461.4 MiB runtime**、**53.8 MiB free**；`tmp-nvcc/gsq-v3-vision-dflash2-combined-20261020.log` | 256-token KV、Vision、无 CUDA Graph、一槽 Host state，实际 HTTP 200；后续复合操作显存极紧 |
| 当前合格 MTP15 canonical 213K 长档 | **11.62 GiB GPU weights**，**2.09 GiB runtime**，其中 **1.58 GiB physical KV**、**82 MiB Graph**；`docs/maintainer/2026-10-22-canonical-replay-prepare.md` | physical 36,864 / KVMem budget 32,768 / reserve512 / C1 / INT8；是另一权重和 speculative 组合，不能将其 2.09 GiB 视为 DFlash2 精确测量 |
| DFlash2 本机 cyclic state 下限 | **40 MiB / state**（5 local layers × 2,048 tokens × 8 KV heads × 128 dims × K/V × BF16） | `src/targets/qwen3_6_27b/impl/config.h` 与 `src/targets/qwen3_6/impl/runtime/layouts_impl.h`；真实分配已包括在相应 runtime/state pool，不重复加到现成内存日志 |
| DFlash2 全历史额外 KV | **0 full layers** (`full_layers=0`) | DFlash2 使用 local cyclic state，不能把旧 DFlash full-cache 容量假设套入；Main 的 KVMem KV 仍需保留 |
| CUDA Graph / 新请求分支 | 在 Vision+DFlash2 短资格中 `--no-cuda-graph`，长档尚无同配置显存实测 | 新 Graph 占用不能凭旧 MTP 的82MiB照搬，不能默认0 |

按相同 Main+MTP 原长档保守比较，DFlash2 **短 profile 仅余约 180MiB**，而现役 36K physical INT8 KV payload 单项已经 **1.58GiB**，尚未计入扩大的 KVMem metadata/索引、Graph workspace、prefill scratch 和可回收桌面预算。因此**旧短档不能原样加大为 32K 工作集**；哪怕在短档已验过 2/2 接受，也不能将溢出转由 UVM swap 当作正确或可用默认。能否由缩小工作集、关闭 Graph/双状态或改变显存预算得到可用的新小档，需要真实重新规划及资源合格证据，仍是后续任务。

默认 KV `automatic` 方案通过 `src/runtime/engine/kv_capacity.cpp` 的容量 resolver，要求预留 **1GiB 自动 headroom**；不足时明确报 `automatic KV headroom requires ... bytes`，且在运行前验证最低 runtime bytes。显式/DeviceBudget 模式也在 `reservation > capacity_budget` 时拒绝并报告所需/现有字节；`layouts_impl.h` 还拒绝 `--kvmem-budget > --kv-device-tokens`、未满足 Main window 的设备预算。应以这些**真实预留门**筛选配置，不能通过 `--managed` 将 UVM 换页假装足够显存。

**格式与调用边界：** `src/targets/qwen3_6_27b/impl/load/bindings.cpp` 的 v3 `dflash2/*` 物理父张量、91个语义定位以及 native `DFlash2Plan` 已有资格；`DFlash2` 与 `MTP` 是互斥 speculative backend，不把两套权重/状态叠加预算。该 GSQ v3 的主输出头格式无法用于 DFlash2 的 fused full-head `linear_topk`，现有 loader 已有明确错误提示；使用 `--spec dflash2` 必须同时指定 `--lm-head-draft`。已有 DFlash2 短请求及 Vision+DFlash2 检查不因此重复。这里的 **DFlash2 五层 local cyclic state** 不等于有 full history 的原 `DFlash`，命令帮助仍支持后者但不能借同名假定执行/容量相同。

**门后工作：** 4.2 保留已验 native backend，不动无证缺口；4.3 仅选 4K/小 working set 且有实际容量的 MTP0/4/7/15 与 DFlash2 三种固定短输入，区分接受率、Prefill 和 Decode；若有新的 DFlash2 profile 能保持至少256MiB空闲、整模型启动/Graph实际可用、无隐藏 UVM pressure，再进入长上下文质量门。当前不将短档 DFlash2 改为生产默认、不运行 213K DFlash2 变体，也不下载其他权重。全程 200W，不改功耗/系统设置。

## 4.2：现役 backend、Graph 与提交生命周期复用核验

本次没有发现新的 DFlash2 数学或 Graph/commit 缺口需修改。已验生产调用链包括 `src/targets/qwen3_6_27b/impl/load/bindings.cpp` 的本模型 v3 物理父张量与 `DFlash2Plan`/`materialize_dflash2_weights`、`src/targets/qwen3_6/impl/runtime/dflash_impl.h` 和 `dflash_context_impl.h` 的五层 cyclic StateImage、`src/targets/qwen3_6/impl/runtime/program_impl.h` 的 proposal/verify/commit 路由。相关代码对本项复核无差分；本轮实际变更仅是默认关闭的历史阶段诊断及 Cold 所有权检查，未改变 draft 数学、提交/验证语义、Graph capture 或 state layout。

已有单卡实际验收 `docs/maintainer/2026-10-20-kvmem-actkv-live-validation.md` 的 P2/P3（特别是 `274–332` 行及 `605–616` 行），固定 native token 三次请求 `tmp-nvcc/p3-20261020-dflash2-final-results.json`：

| 已验路径 | 本机观察 | 承接界限 |
|---|---|---|
| DFlash2 K3 / Graph / sync + async | 两臂三次 24-token 请求，草稿接受依次 **17/17、16/18、17/17**，output IDs / committed decode / reuse 相同；每臂 page-boundary `PASS`，async submitted/completed **20/20**，末端 inflight **0** | 旧冻结请求、短 Graph 与投机生命周期；不证明 213K+Graph 或新候选 profile |
| MTP3 / Graph / sync + async | 既有固定 native 三次请求接受依次 **16/18、15/21、16/18**；提交、输出和释放相同，末端 inflight **0** | 继承当前 MTP 控制器既有能力，未修改其 width 选择 |
| 语义双轮回放 | base / Graph / MTP3 / DFlash2 K3 每路四请求，共 **16/16** 答案正确；第二轮复用 2,933 prompt tokens，DFlash2 accepted/drafted **5/10**，末端 in-flight **0** | 已有短请求真实服务路径的提交和接受前缀核验，不是所有输入与所有 speculative 档位证明 |

早期 MTP speculative pool 资金不足、边界同步异常已在 P3 修复后由上述最终收据覆盖；旧失败仍在原报告中且未重算。对本轮未触动的 backend/Graph/math 沿用该实际证据，OpenSpec **4.2 完成**；若在 4.3 或新 profile 中出现新 SM89/Graph/提交失败，须独立处理并重新资格。**4.3–4.5 保持未完成，DFlash2 不进入当前默认 213K 档**。
