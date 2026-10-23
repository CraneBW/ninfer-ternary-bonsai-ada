# OpenSpec S5：本机 MTP host / GPU 轮次与验证宽度（2026-10-22）

2026-10-23后续已实现有界C1 owned publication与GPU提交重叠，AR/MTP4/短容量DFlash2
生命周期通过；同binary200W三对未获稳定收益，默认关闭。实现范围、实际可隐藏上限和
完整结果见[非NVFP4收口报告](2026-10-23-non-nvfp4-closeout.md)。以下保留当时的trace判断和
adaptive controller原始测量，不将旧未实现状态当作当前状态。

## 6.1：现有证据下的可迁移性判定

来源固定于 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的
`deepinfer/worker/worker.py:113–187`、`worker/zos_status.py:10–19`、
`engine/llm_engine.py:67–119`、`csrc/engine/mtp_varlen.hpp:160–207`；
许可证 Apache-2.0。映射到本机：

| DeepInfer 机制 | 本机真实对应路径 | 结论 |
|---|---|---|
| 本轮先提交 GPU，处理上一轮 CPU | `program_impl.h::decode_mtp_batch` 同步返回 host egress；`engine_core.h::run_decode_round/commit_pending` 紧接提交 Engine 决策 | 下一轮 GPU 依赖本轮 accepted 数、KV/StateImage commit 与后续预算；若未重构 owned pending-step/event/metadata，则不能提前读写其共享 host ingress/egress |
| scheduled/processed 差 0 或 1 | `ProgramImplCore::decode_mtp_batch` / `resolve_pending_raw` / Engine `commit_pending` | 现役单 owner 串行：差为 0；预执行下一步需要跨 Program、Engine 的显式保活和取消/结束屏障，不能直接删 `device.synchronize()` |
| varlen acceptance 管理 | `mtp_adaptive.h`、`program_impl.h`、`mtp_impl.h` | 现有 controller 根据每一列前缀生存率和同批执行耗时动态选宽；每轮 GPU 实际做 `min(draft_window, verify_window+1)` 草稿，值得修复 cohort 生存期边界 |

复用此前一次**真实** GSQ v3 / SM89 / C1 / MTP15 / 200W / 31,987
input + 128 output Nsight Systems CUDA/NVTX trace：`tmp-nvcc/opsx-s4-20261022/trace.sqlite`；
聚合方式：`NVTX_EVENTS` 有结束时间的区间，按 `StringIds` 名称归类；
不可将父子嵌套 NVTX 时长相加。

| NVTX 区间 | 观测次数 | 累计 wall |
|---|---:|---:|
| `decode.mtp_round` | 38 | 3,069.780 ms |
| `decode.mtp.wait` | 38 | 3,028.684 ms |
| `decode.mtp.submit` | 38 | 40.902 ms |
| `engine.commit_output`（该 trace 所有请求） | 212 | 0.616 ms |
| `program.post`（该 trace 所有执行单元） | 304 | 1.536 ms |

MTP decode 的 `submit` 只占 `40.902 / 3069.780 = 1.332%`，
`wait` 为 `98.661%`。即使将已观测的全部 submit（再加整条 trace
累计 Engine commit 0.616 ms）理想完全隐藏，按这段 MTP decode
的 Amdahl 下界也只有约 `1.014x` 理论加速。这里的 `wait`
表示主机阻塞等待设备完成，**不**当作全部是有效 GPU 计算，更不推断
当前存在可消除的 98.7% CPU 空转。且这些数据是单次 trace，不能作为
200W 整模型成对性能结果。

因此 **6.2 的迁移条件不成立**，不实现跨层 owned-step 队列和
额外 event/metadata 生命周期，保持该任务未完成；未来若新 profile
显示 `submit/post/Engine commit` 的主机占比明显升高，再决定是否重构。

## 6.3：按真实接受率和当前图成本修复宽度控制器

已通过的三类固定短题筛选（`docs/maintainer/2026-10-22-s3-spec-profile-screen.md`）
显示 MTP4 为 `128/235=54.5%` 提议被接受，MTP15 为
`147/541=27.2%`；平均 Serve wall 分别 `0.916s` 和 `1.300s`。
它们是不同固定 draft 深度的整体图轮次，**不能**直接等同
同一 K 下仅改变验证宽度的独立节省；已有本机 adaptive
controller 的 cost model 才用于真实 width 选择。

本次定位到 `decode_mtp_batch` 原 `cohort_key ^= lane`，
不同集合 `{0,1}` 与 `{2,3}` 都得到同一个值，
且原签名不包含 `lane_epochs`，在相同 lane 的下一位请求会复用
前一请求的已学习验证宽度和窗口滞留计数。
这种错误最容易在跨请求/部分接受后使新低接受请求运行前一请求的
过宽 CUDA Graph，正是 controller 应重启短档 probe 的场景。

改动：

* `mtp_adaptive_cohort_key(lanes, lane_epochs)` 对 distinct lane mask +
  各 lane epoch 做与行顺序无关的散列；这是性能选择标识，不作为
  内存所有权/安全校验；源头已有原生 lane/epoch 的生命周期边界。
* `MtpAdaptiveBatchController::select` 仅在新 cohort 的启动宽度
  确定后开始记录本 cohort 的 window transition，避免把上一
  请求切换成新请求的初始宽度计为一次自适应收缩。
* 保留现有定量 `MtpRoundCostModel`、生存率、draft AR/verification
  数值、Graph宽度容量、原样默认 MTP15，不复制 DeepSeek confidence
  head 或修改 Qwen 采样行为。

定向 CPU C++20 原生回归：
`c++ -std=c++20 -O2 -Isrc tests/test_mtp_adaptive.cpp -o /tmp/openspec-s5-mtp-adaptive`，
`/tmp/openspec-s5-mtp-adaptive` → `mtp adaptive tests passed`。
新测试覆盖跨 lane 集合碰撞、同组换序、lane epoch 变化、强接受
宽档切至新低证据 cohort 重启窄档、不会误计宽档 transition；
已有 controller 宽档/缩档与边界行为同次通过。此处为 CPU
功能资格；实际模型/整会话 benchmark 另列 6.4，尚未据此
宣称速度提升。

## 6.4：真实单卡同二进制、200W、两个上下文档、三对整体性能

**已完成且结果为负：** 采用本次修复后的正式 `ninfer-serve`，
二进制 SHA256
`354149af6e6dd7034e2bd24bd34937063490c828cd2bde141fcb17d8df28b158`，
源代码原生 `cmake --build build --target ninfer-serve -j2` 成功。
同一现役 Qwen3.8 GSQ v3 权重、单 RTX4070 Ti SUPER、
200W、C1、INT8 KV、Graph、greedy/no-thinking、
物理KV36,864 / KVMem工作集32,768 / reserve512、
prefill chunk512 / `--kvmem-transfer-mode sync`。两个模式都显式
`--spec mtp --draft-tokens 4 --lm-head-draft`；
候选额外 `--adaptive-mtp`，无其他模型/采样差异；
启用 `--request-log-jsonl` 留存原生 Engine request_done
时间、host/device分解及 MTP 接受/窗口行为。

每档各三对（交错为 fixed/adaptive、adaptive/fixed、fixed/adaptive），
每臂独立创建自身Serve进程并正常SIGTERM/exit0；
所有12臂HTTP200，输出文本SHA256和完整usage在每对**一致**。
4K档真实prompt 3,697 tokens，32K档31,987 tokens，均产生
**128 completion tokens**；两个上下文档因此可对照原Design
的 `tg128` 规格。启动Graph准备时间单独留存，不叠到请求的
Engine测量time。额外早期4K普通列表prompt仅自然完成35 tokens，
其原始 `results-4k.json` 保留作探索，不纳入以下正式表或通过门。

下表数值均按**每对**先算 `adaptive/fixed-1`，再取三对中位；
同列固定/候选原值另为三次独立样本中位（其相除不等于成对中位）。
所有单位为秒，变化正号表示时间增加。

| 指标 | 4K固定 → adaptive | 4K成对变化 | 32K固定 → adaptive | 32K成对变化 |
|---|---:|---:|---:|---:|
| 完整Engine请求time | 3.170245 → 3.204266 | **+1.127%** | 22.806161 → 22.896830 | **+0.289%** |
| TTFT | 2.330639 → 2.342992 | +0.460% | 21.283704 → 21.330285 | +0.090% |
| prefill | 2.320853 → 2.334858 | +0.467% | 21.267807 → 21.311308 | +0.086% |
| decode | 0.838326 → 0.861361 | **+2.985%** | 1.534259 → 1.566005 | **+1.954%** |
| Engine Host exposed | 1.372644 → 1.378768 | +0.446% | 12.552550 → 12.567691 | +0.121% |
| Engine Device wait exposed | 1.786633 → 1.814311 | +1.631% | 10.236510 → 10.310942 | +0.488% |
| decode-class Host exposed | 0.000914 → 0.001507 | +63.109%（绝对微小） | 0.001475 → 0.002226 | +39.251%（绝对微小） |
| decode-class Device wait exposed | 0.837504 → 0.858992 | +2.714% | 1.533092 → 1.564460 | +1.923% |

32K三对的 fixed 每臂 `164 proposed / 84 accepted`，
adaptive 每臂 `148 proposed / 81 accepted`，
adaptive窗口3与4各运行31与15轮，发生2次width transition；
4K fixed `101/101`，adaptive `100/101`，
adaptive窗口3与4各6与21轮，发生1次width transition。
因此新控制器实际执行了宽度变更，但节省草稿量**没有**换来
本机整模型的稳定吞吐提升；不能把相对更少draft当作更快。

原生收据与完全复现命令：

```bash
python3 tools/bench/mtp_adaptive_pairs.py --case 4k128
python3 tools/bench/mtp_adaptive_pairs.py --case 32k
```

主数据是
`tmp-nvcc/opsx-s5-20261022/results-{4k128,32k}.json`、
`tmp-nvcc/opsx-s5-20261022/{4k128,32k}-{1,2,3}-{fixed,adaptive}.jsonl`
与各臂Serve日志；Git跟踪的汇总副本为
`docs/performance/2026-10-22-s5-mtp-adaptive-200w-pairs.json`；两份JSON的 `state` 均为
`THREE_PAIRS_SAME_TEXT_AND_USAGE`。
静态C++20 `test_mtp_adaptive.cpp` 和正式Serve构建通过，
`git diff --check`通过。新的本机cohort生存期隔离是有效修复，
但DeepInfer异步队列6.2仍不适用且未移植，
本次自适应profile只保留 `--adaptive-mtp` **显式opt-in**；
整体正式长档默认继续固定MTP15 + sync，未做300W/双卡测试。
