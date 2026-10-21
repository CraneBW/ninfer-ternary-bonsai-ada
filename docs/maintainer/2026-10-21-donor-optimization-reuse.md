# 5060 Ti / DeepInfer 源码吸收与本机实施顺序

2026-10-21。目标仍是现有 Qwen3.8 GSQ、4070 Ti SUPER、原 213K 题目正确，优先缩短 TTFT 和历史重建。研究期间没有执行 donor 脚本、构建 donor、下载权重或启动另一项 GPU 工作；R22 已冻结的 TTFT 对照继续串行执行。

## 已读的真实源码

| 来源 | 冻结版本 | 适用边界 |
|---|---|---|
| [5060 Ti Qwen3.8 项目](https://github.com/XuanzhengZhou/qwen3.8-27b-ternary-5060ti/tree/104af690541857a5d52f23ffd81ccac7eb82a675) | `104af690541857a5d52f23ffd81ccac7eb82a675`，Apache-2.0 | 与本项目同 NInfer 血统，但其主档是 PQ2 三元权重、Blackwell、DFlash2；本机为 Ada、混合 IQ 格式 GSQ、MTP7 |
| [DeepSeek 官方 DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01) | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，Apache-2.0 | 当前支持 DeepSeek-V4.1-Flash、SM100/Ascend950；完整 KV cache system 未开源，不是 Qwen/SM89 现成后端 |

本地分别位于 `ecosystem/qwen3.8-27b-ternary-5060ti` 和 `ecosystem/deepinfer`。
逐文件映射、社区补丁与冻结记录位于本仓库
`tmp-nvcc/perf-ada-donor-20261021/`：`5060ti-source.md`、
`deepinfer-source.md`、`community-service-source.md`、`community-primary/manifest.json`。
这些是本机调查证据；在线固定 commit 链接才是外部可复查来源。

## 四条最相关线索的实际结论

### GDN：有现成 dtype 实现，但并非精确状态压缩

5060 Ti donor 的 `engine-patches/src/core/linear_attention_state.*`、
`ops/linear_attention/gated_delta_net/recurrent.*`、chunked state-passing、
`replay.cpp`、`state_image.cpp` 和 layouts 已贯穿 FP16/FP8 recurrent storage。
计算寄存器仍是 FP32，**每次存态有损舍入，下一步读取该舍入状态**。
它会改变后续推理，不能用“镜像压缩”暗示与当前 FP32 完全等价。

本机当前 recurrent payload 为 `48×48×128×128×4 = 144 MiB`/完整 frontier；
已记录的整体 StateImage 为 153,954,304 B。若只改变 recurrent 存储，保持其他组成，
纯布局计算为 FP16 78,456,832 B、FP8 40,708,096 B。后两者均未实现、未实测，
不能写成已节省的显存/RSS。donor 的短 PPL 结果也不替代本机原 213K 题目。

若后续测量证明状态容量/传输是瓶颈，先完整移植 **FP16 opt-in**，保留默认 FP32，
再单独评估 FP8；至少涉及 12 个现有文件，必须贯穿 Host/device 布局、record/fold、
chunked prefill 和恢复身份。本轮没有开启有损 dtype，也没有把它当成 R21 错题修复。

### KV materialize：没有找到所猜测的 PDL 修复

[donor materialize](https://github.com/XuanzhengZhou/qwen3.8-27b-ternary-5060ti/blob/104af690541857a5d52f23ffd81ccac7eb82a675/engine-patches/src/ops/context_kv_materialize/context_kv_materialize.cpp)
与本项目的差异仅是增加头文件、将两个 batch 上限从 8 改为并发常量。
没有 launch、PDL、调度或数学修改；对当前 C=1 没有可直接采用的性能补丁。
多个 `gdn_input_proj` plan 也主要扩大并发门限。

DeepInfer 部分 TileLang 算子使用 PDL。NVIDIA 的
[PDL 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
将该能力限定为 CC≥9.0；本机 SM89 应使用普通 stream 顺序和 event 依赖。

### CUDA Graph：保留本机测量边界

donor 的并发阈值和 2.5–3 GB 图池数字属于其模型/布局；本机现有 R22 启动记录的
`cuda_graph_allowance_bytes` 是 85,983,232 B（82 MiB），不是同一数字。
allocation allowance 也不是完整实际图内存测量。当前没有据此关闭 C=1 Graph。

采用其测量经验：trace 用来确认机制，clean 同输入 A/B 用来确认收益；
步探针设备时间、主机 launch 窗口、假设 MTP 全接受都不能代替实际 API TTFT 或发出的 token 数。

另需纠正转述：donor 总表的“FP8 KV −4.9%”对应冷 prefill 143.3→136.3 秒，
是用时缩短约 4.9%；代价是相对其 k8v4 显存增加、256K 无法启动。
这不是“FP8 必然慢”的证据。继续保留该负结果的原始条件，不将其转成跨架构禁令。

### 四层缓存和 HTTP：先分清缓存了什么

| donor 工艺 | 实际保存/改变的对象 | 本机处理 |
|---|---|---|
| stable_prefix | 将可变 TASK 移到稳定前缀后，改变 prompt 位置 | 不改冻结原题；未来若适配 DSH，以新的产品输入合同单独评估 |
| 0.63 ms 精确回放 | 已生成 HTTP/SSE 响应字节 | 不是 Engine KV 命中；不能用于原题推理资格或宣称冷 prefill 加速 |
| 前缀落盘 | prompt/tools/model JSON | 不是 GDN/KV 二进制持久化；模型/布局/采样身份必须参与失效 |
| 启动预热 | 重新提交前缀进行一次真实预填 | 可缩短之后的匹配请求，不能省掉该次重算成本 |
| HTTP 分帧收流 | 零 chunk/trailers 或 Content-Length 到达即结束 | 本机 `tools/streaming_http/client.py:151` 已用 `read1()`；未发现同类 5 秒 EOF 等待，不重复增加代理层 |

98.9–99.1% 命中率和 0.13–0.40 秒 TTFT 是 donor 工作负载的自报数据，
不能推广为当前 213K KVMem 历史重建耗时。其 maxTokens 无 adapter clamp 的结论也不能
覆盖本机剩余上下文/输出容量计算；配置输出预算不会自动增加 `--max-context` 或物理容量。

## DeepInfer 真正值得吸收的实现

1. **异步 step 所有权**：`worker/worker.py:113–187` 先 enqueue 本轮 GPU，再处理上一轮 CPU 输出；
   `zos_status.py:10–19` 和 `llm_engine.py:67–119` 显式保活 metadata、输出、event、clone pin。
   `mtp_varlen.hpp:160–207` 区分 schedule/process 轮次，计数差限定为 0/1。
   这些是未来异步拆分的实现参考。本机 `BatchedGeneratedRound` 当前返回已经完成的 host spans，
   下一轮依赖接受数和 Engine commit，不能只删除 `synchronize()`。
2. **SWA bounded replay 的边界**：官方
   [Q&A 179–191](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L179-L191)
   明确它是截窗后的近似重算，重建结果只用于 decode、不进入 prefix cache。
   140-token 策略属于 YoCo consumer，不能替代 Qwen GDN 完整递归状态，也没有解释 R21 首个 Main bonus 错误。
3. **按层重叠搬运与计算**：公开设计将 fetch 与 Host→GPU swap-in 分开、按层流水；完整缓存服务未公开。
   本机应继续从 KVMem/ActKV 抄 native 页归档、pin 和恢复，采用 Cold-read completion、H2D-ready event、
   source-K rebuild completion 各自的真实依赖，避免把 API 返回当作数据已就绪。
4. **真实接受率驱动验证宽度**：吸收 graph 档位成本与前缀生存率的建模，复用本机已有 MTP controller。
   不复制 DeepSeek confidence-head 阈值，不假设 Qwen artifact 有相同 head；当前固定 MTP7 保持不变。

## 社区单卡优化：发现了组件工作，尚无本机完整后端

本次限定检索没有验证到可运行的 SM89 消费卡完整 DeepInfer LLM 移植。
以下均为调查时的未合并 PR，保留源码快照，未执行：

| 原始来源 | 实际做了什么 | 本机可吸收部分/限制 |
|---|---|---|
| [#54](https://github.com/deepseek-ai/DeepInfer/pull/54) | 1×SM86 构建和组件微基准，profiler 空闲快路、event 复用 | 有真实单卡组件证据；没有完整模型吞吐证明；本机原生 C++ 不承受同样 Python `record_function` 成本 |
| [#87](https://github.com/deepseek-ai/DeepInfer/pull/87) | streaming-prefill RFC | 仅文档，不是可直接复制的实现 |
| [#99](https://github.com/deepseek-ai/DeepInfer/pull/99) | 4×B200 单节点 Engram/NVLink 路线 | 多卡单节点不等于单卡；作者未隔离质量差异、未提供性能测量 |
| [#101](https://github.com/deepseek-ai/DeepInfer/pull/101) | SM90 FP8 checkpoint loader | DeepGEMM/FlashMLA fallback 仍缺，不是完整 SM90 后端，更不是 SM89 后端 |
| [#111](https://github.com/deepseek-ai/DeepInfer/pull/111) | 大队列输出分组/clone_grouped | 借鉴批处理减少 host 访问；其 1024-sequence 场景不支持 C=1 收益结论 |

## 本次落地与下一步

已经完成：

- 新增[本机 CUDA 服务规程](cuda-service-operations.md)：精确 owner SIGTERM、超时仍视为存活、禁止主动 SIGKILL、
  D-state 不自动重启、256 MiB post-warmup 显存地板；新 unit 要有 `SendSIGKILL=no`。
  donor unit 缺少这个设置，因此没有原样复制。规则不是新部署的 watchdog，也不是已编译进引擎的显存门。
- 新增 `tools/bench/cuda_process.py`，复用 donor 的优雅停止思路：只操作自身 Popen child，
  超时抛出“仍存活”错误，保留进程，绝不升级 KILL。两个 CPU 子进程分别验证正常 TERM 退出和忽略 TERM 后仍存活；
  两个 fixture 最后均正常退出，没有跑 GPU。
- 准备 `qualification-graceful.pending.patch`，在存活超时时仍先保存请求 receipts/失败 ledger。
  **当前未接入正在执行的 R22 runner**，避免在同一对照的中途更换 launcher 行为；现有 legacy fallback 仍存在。
- 修正 `docs/serving.md` 已过时的 global Host KV descriptor/admission 说明，明确 R22 的实际 owner 分账。

继续执行顺序：

1. 完成当前 R22 同 binary、HostKV8192/0、ABBAAB 六进程/36 个原请求；任何失败立即保留，不追加样本。
   终态前不重建、不改生产算子、不开始第二项 GPU 工作。完成后只分析现有 receipts 一次，分别报告初次完整预填和后五题 TTFT。
2. 确认现有服务/子进程全部终止后，应用上述停止补丁；新 GPU unit 使用显式 no-KILL 设置。
   不覆盖原 matrix 的 runner/manifest/结果。该接入是已授权的后续实现，不需要用户再次确认。
3. 性能方向仍优先减少 canonical 历史重复重建、NVMe read-ahead/逐层 wait-ready；源码优先 KVMem/ActKV。
   R21 原位置模式继续默认关闭且质量未通过，不能用 donor 近似重放替原题放宽标准。
4. 算子继续优先 GSQ 实际热点；IQ2_S、IQ3_S、IQ3_XXS 可共同开启已有合格实现，但组合收益实测，
   不叠加单项中位数。新 tile 候选先确认资源约束和已有负结果，再做最小 Op 对照。
5. 有损 GDN dtype、异步 step 合同改写、自适应验证宽度保持为有明确触发条件的后续工作；
   本次没有因此换模型、改原题、宣称 256K/P4 通过或翻转全局默认。

目前没有从上述 donor 得到可直接粘贴、保持现有数学且已证明提升 C=1 GSQ 的引擎补丁。
可以直接复用的运维实现已落实，其余机制已定位到文件与生命周期，按实际瓶颈推进。

## 执行续记

R22矩阵已终态36/36、所有server/service0，后五题TTFT合计三对均正、中位减少2.262371%/2126.038ms；首题中位-0.110288%无稳定改善。generic qualification runner已在矩阵终态后接入owned-child优雅停止helper，原R22 runner/manifest/结果未改。上述“未接入/等待矩阵”是准备阶段记录，已由本续记替代。R23 J64两格式正确性通过但完整Op吞吐均下降，拒绝，未额外长测；继续推进默认关闭的source-K有界读前。
