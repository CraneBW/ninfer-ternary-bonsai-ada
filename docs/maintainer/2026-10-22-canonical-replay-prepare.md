# 2026-10-22：200W canonical 回放准备批次

OpenSpec `absorb-donor-techniques-and-optimize-sm89` 的 S0 来源/预算收口，以及 S1/S2 中保留 canonical 正确性的准备工作优化。本轮没有启用未通过的 R21 原位置模式。本轮功能通过，整体收益不稳定，候选保持默认关闭。

## 功耗与冻结输入

日常测试、分析和优化统一 **200W**；仅发布前最终验收使用 **300W**，结束恢复200W。根与目标仓AGENTS、当前OpenSpec及快照工具说明均按此更新；以前的300W收据保持历史功耗，不能重新标注为200W。

- 当前模型：`/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer`，SHA256 `29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`，15,017,456,896B。维持Qwen3.8主线，不恢复Qwen3.6权重。
- 原213K六请求：`/home/ran/ninfer-work/benchmarks/frozen-long-context/213k/manifest.json`中的六个绝对路径和SHA不变。逻辑context212992、首输入212276、物理KV36864、选页工作集32768；这不是full-history attention。
- 已资格q0快照：`prefix.nkv`，212260 tokens。仅省初始prefill，load独立计时；不读取保存答案。
- 本轮同binary对照：Qwen3.8 GSQ、INT8 KV、chunk512、C1、Graph、greedy/no-thinking、MTP15/optimized head、sync；其他source-event/D2H候选均0。二进制与完整执行记录见 `tmp-nvcc/opsx-canonical-prepare/receipt.json`。
- 两个日常resident参考profile：4K pp4096/tg128/ctx8192/MTP7，32K pp32768/tg128/ctx36864/MTP15。既有200W记录的decode/prefill分别172.55/1554.61、197.57/1431.27 tok/s，来源 `docs/performance/2026-10-21-power-cap.json`。属于旧版本实测导航；本批仅命中KVMem query-replay，不能声称这些resident吞吐因本批而提高。

复现本轮基准（普通200W，不生成新快照）：

```bash
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 213k --output <新参考目录> --power-limit 200 --no-canonical-query-reuse
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 213k --output <新候选目录> --power-limit 200 --no-canonical-query-reuse --canonical-prepare
```

两命令必须串行、同binary；本机以20GiB RAM/2GiB swap用户service运行。候选开关 `NINFER_KVMEM_CANONICAL_PREPARE=1`；默认0。输出准确计入restored token，不能作为cold-prefill吞吐。只有发布终测才把功耗改为300。

## Sync/Async：来源、引擎和部署的实际映射

用户指定的来源为同一个 [paicat1/Ninfer-KVMem-Async](https://github.com/paicat1/Ninfer-KVMem-Async)。main固定 `f7e5c19180051ab5a0c997a17cba03ac6487e576`；engine-main固定 `a4f32e26deae7a49b7110c4c27e61f590b914a68`。两分支已按实际文件核对，不再从单个kv_store.h推断功能。

| 项目层/补丁 | 对应实际引擎或部署 | 本机采用边界 |
|---|---|---|
| `main/build/README.md`、`build/s6_1208a.bat` | SRC=`engine-src/src-tree/fusion-engine-src`，本机开发branch line-coder390；README推荐engine-main完整快照且已含补丁。CMake sm_120a、MSVC、Windows，输出 `.temp/build-1208a/apps/ninfer-serve.exe` | 本机Linux/SM89不运行这个构建脚本；仅映射来源，不搬SM120a配置 |
| `main/deploy_engine.ps1` | 接受新exe，部署到 `engine/self-built-20261008/ninfer-serve.exe` | 不复制其Force停止方式；本机保持自有TERM/drain |
| `scripts/start_engine_default.ps1` | 源件 `engine/lineage/I3b_step13_fourlayer_20261015.exe`，交给engine_ctl复制并启动 | 部署入口对应特定谱系件，不等于engine-main HEAD |
| `ninfer_launcher.py`、`engine_registry.json` | 下拉读取registry，S13登记sha前缀322A972D、官方链+4个dirty文件；带符号件另登记3543AD18/9453FFC6，sym3源be2c0f9 | 这些是上游登记信息，未取得其完整dirty源码与生产exe；不声称本机复现其二进制/速度 |
| ledger42 async demote、ledger46 multibatch | engine-main `src-tree/fusion-engine-src/src/models/qwen3_5/program/program_impl.h`、`storage/context.cpp`：`PendingDemoteRing`、`retire_pending_demote`、`demote_kv_pages_to_host` | 先前本地双16MiB submit-before-wait是小幅适配；没有完整移植其N-slot跨step ring |
| ledger46 inflight demand | 同一context.cpp的`ensure_ring_room`：demand扣除claimed_pages；硬容量仍读真实free，必要时退休再判定 | 保留现有本地真实容量/pin；不能把pending当已空闲 |
| ledger42 cancel checkpoint | Program的decode_cancel_probe以及逐页循环采样取消 | 本地现有事务异常排空已具备；跨forward逐页取消hook未全部移植 |
| I3b S2 submit / S9 verify split | `speculative/mtp.cpp`、`step_graph_profile.h`、`target_verification.cpp`、`ops/launcher/speculative_round.cu`中的计时stamp | 主要是测量拆分；没有据此宣称跨轮异步Engine step已实现，不照搬插桩时长 |

额外关键差分：engine-main的`async_demote_depth()`已将默认从补丁初版2改为 **1**，保留显式1..8。源码记录depth>=2曾导致重复demote/严重回退。不能因为main/multibatch.patch写默认2，就假定其当前默认或本机安全默认为2。

engine-main根LICENSE为Apache-2.0，具体本地适配仍保留来源说明。main补丁冻结摘要位于 `tmp-nvcc/opsx-state-pipeline/sources/manifest.json`；本轮补充入口和engine-main实际文件在 `tmp-nvcc/opsx-canonical-prepare/sources/manifest.json`。未重新审查已冻结KVMem/ActKV全树。

## 16GiB预算与NVFP4初门

| 项目 | 当前GSQ实例或保守初算 | 含义 |
|---|---:|---|
| GSQ GPU常驻weight | 12,481,729,536B = 11.62GiB | 来自已资格Serve startup，不是15.02GB容器大小 |
| 当前runtime reservation | 2,240,713,472B = 2.09GiB | 已含物理KV、状态/图等；不能再把KV分项相加 |
| 其中KV payload | 1,695,682,560B = 1.58GiB | 36864物理slots，Main/MTP布局依既有profile |
| 其中Graph allowance | 85,983,232B = 82MiB | 本机当前profile数值，不借用他人2–3GB图池 |
| Host replay checkpoints | 615,817,216B = 587.29MiB | 四槽不增加；与普通8个Host StateImage槽分开 |
| 调优进程预算 | RAM20GiB / swap2GiB | 避免先前内存耗尽；GPU构建与推理串行 |
| GPU地板 | 256MiB | 发布资格继续保留 |
| NVFP4权重极乐观初算 | 27e9 × (0.5+1/16)B ≈14.14GiB | 假定所有参数4bit、每16权重1B scale，尚不含额外MTP、BF16 embedding/head或其他元数据 |
| 同当前runtime+桌面+地板 | 14.14+2.09+约0.86+0.25≈17.34GiB | 当前全驻留配置超过16GiB；不能承诺NVFP4替换GSQ后直接运行 |

这是容量初门，不是完整Qwen3.8 NVFP4实测。没有已确认完整权重，不下载、不恢复Qwen3.6；真实Op/长题/速度仍NOT_RUN。实际量化覆盖率、非量化张量、MTP及更小KV/Graph配置需在S6详细核算，不能用本表勾选7.2–7.4。

## 本批改动与理论上限

复用KVMem checkpoint完整边界、prepare/restore分离与ActKV归档所有权原则，适配本地页布局；不改CUDA数学/原问题/选页评分/页排列/量化/RoPE语义。

1. 四个既有Host槽覆盖预算1/32、1/16、1/8、1/4，而非1/16、1/8、1/4、最后整块。32K档即1024/2048/4096/8192，不按原题答案设置。旧公共前缀1920/3200/3200/6848/2880，理论复用10240→11264，仅减少1024个历史token，约0.55%。
2. query probe冻结后，马上要canonical回放的分支，将最大可复用检查点作为保守保留范围。之后的stage-in只申请原有物理页，不读取马上被覆盖的Cold/Host载荷，不提交有效列；完整source owner一直保留到begin_replay撤销。前缀V仍准备，前缀K仍从immutable码/scale恢复。整窗的临时重定相也被跳过，因为prefix随后会恢复K、suffix随后重算。

旧snapshot restore6收据有2936次stage-in、3398次source-K rebuild、3.11s传输计时。理论可减少数GiB载荷及相关同步，纯PCIe节省量仅亚秒到秒级；不能由此承诺整体10%。相容两项一个开关、一个binary验证。先一对，后五题TTFT总和改善不足1%即停止；有机会才补三对。计时前计算和完整执行依据见 `tmp-nvcc/opsx-canonical-prepare/theory.md`。

## 结果与未完成边界

本轮结论：**功能通过，未取得稳定整体收益，候选默认关闭**。213K同binary三对、每臂原六题均精确正确，36/36；256K候选单次原六题精确正确，6/6，共42/42。HTTP/terminal/usage完整，inflight页/字节与active请求归零，各Serve退出0。没有重做长前缀prefill，两个原有快照未改。

|对次|参考后五题TTFT合计s|候选后五题TTFT合计s|后五题成对减少|参考session wall s|候选session wall s|
|---|---:|---:|---:|---:|---:|---:|
|1|117.497|112.140|+4.56%|167.923|161.487|
|2|113.816|114.200|-0.34%|164.583|163.880|
|3|114.681|115.904|-1.07%|163.860|165.313|

后五题TTFT的成对中位减少 **-0.34%**（即退步0.34%）；session wall成对中位减少 **0.43%**，不足以认领稳定整体加速。各臂独立中位TTFT合计114.681/114.200s不能替代成对差值。初始第一对4.56%的收益没有稳定复现，完整三对保留，无额外消融或长题重试。

首次TTFT独立中位：参考49.962s、候选48.953s；snapshot-load参考26.443s、候选25.729s。load源码没有修改，这些差异不认领为本批加速。整个会话包含加载，但不包含服务启动；不等于cold-prefill成绩。

|实际工作/资源|参考|候选|
|---|---:|---:|
|history replay tokens|185836|184812|
|history reused tokens|10240|11264|
|source-K rebuild页数|3398|1384|
|native transfer runs|2136|1083|
|Host replay checkpoint bytes|615817216|615817216|

每个候选213K会话跳过1648个Main/MTP stage-in载荷准备。调试日志的rephase_skipped是整窗计划页数（含原本noop页），不能当实际kernel调用数。transfer_us是含等待的阶段计时，三对中位参考2.166s、候选4.005s，亦无稳定改善；不能用copy次数下降推导PCIe吞吐提升。本轮未清OS页缓存，不能声称真实物理冷NVMe提速。

256K逻辑context262144、首次实际输入261411、物理KV36864、工作集32768，恢复261395 tokens，load32.112s；只补候选质量，不与历史300W记录混算性能。candidate+page-pipeline组合未验（NOT_RUN），本轮仅sync。

构建成功，实际峰值2.6GiB、swap0；三对campaign cgroup峰值20GiB（含文件缓存）、swap9.9MiB；256K峰值20GiB、swap0。未增加四槽Host预算，无OOM、无KILL。最终200W。完整收据 `tmp-nvcc/opsx-canonical-prepare/summary.json` 与同目录各pair/256k目录。

已有CUDA数学/codec未改，继承此前独立oracle证据；本批必要检查为受影响构建、真实原题与完整终态/资源、Python入口及OpenSpec校验，不跑无变化resident吞吐或完整算子矩阵。保留源码中的opt-in机制，不提高默认性能声明。


R21仍默认关闭；没有声称query-only与canonical具有同一GDN条件。逐层forward-ready/released、Main/MTP混合接受边界、P4新流水、V100X2 kernel/许可及NVFP4实测均不因本批而完成。已有256K包保留；本轮没有改写其资格。
