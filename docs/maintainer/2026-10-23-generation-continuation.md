# Qwen3.8 AR 长思考与生成后工具续轮

本轮只处理 Pi xhigh 的 AR 收束诊断和生成回收后的历史复用。设备为 RTX 4070 Ti SUPER（SM89、66 SM、16 GiB），全部运行在 200W；使用已有 GSQ mixed-IQ v3，未下载权重、未修改量化或 CUDA 数学。

后续已完成相同cwd/Host条件的官方采样与greedy诊断，具体因果证据及仍需重建的真实token失配边界见[非NVFP4收口报告](2026-10-23-non-nvfp4-closeout.md)。本页保留当时不同cwd补充样本和原续轮前后对照的原始条件。

## AR：先纠正测试采样，再解释失败

[Qwen3.8 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)的 thinking 推荐为 temperature=1.0、top_p=0.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1；xhigh 和 preserve_thinking 均是官方默认。本仓 `Package::sampling_defaults` 已正确注册这些可用参数，乘法重复惩罚保持恒等；上轮 Pi harness 的 `--greedy` 将 temperature 强制成了 0。

旧 AR 秦始皇题输出 131072 token，全部为 reasoning，没有作品。CPU 分析原日志发现同一段落起点出现 54 次，重复周期约 4439–4470 字符；第二次出现前的文本用 artifact 内原生 tokenizer 编码为 9731 token，连同 1678-token prompt 仍远低于 32K。因而工作集回收不是循环开始的必要条件。该证据不排除模型或低位量化对 greedy 循环的影响。

补充运行仅重跑受影响的秦始皇原题，使用本地 Pi、xhigh、262144 逻辑上下文、32768 工作集、36864 物理 KV、131072 单轮输出预算，未设置 thinking_budget。实际请求日志核实 temperature=1.0/top_p≈0.95/top_k20/min_p0/presence0/frequency0/seed42。harness 默认改为模型采样，`--sampling greedy` 保留为诊断配置，并逐请求检查实际参数。

| AR 秦始皇 | 旧 greedy | 官方采样补充 |
|---|---:|---:|
| 首轮输出 | 131072 | 48598 |
| 首轮 reasoning | 131072 | 38554 |
| 首轮 decode | 40.066 tok/s | 42.229 tok/s |
| Pi 总墙钟 | 3273.61 s | 1192.02 s |
| 终态 | length，无文件 | stop，实际写出 index.html |

补充样本随后自然完成工具收尾，全部 HTTP 正常、Serve exit 0。实际 HTML 在 Chrome 中渲染出帝王装束人物骑白熊的雪地场景，包含 SVG 动画；属于风格化作品，不能作为严谨历史形象或所有量化无损的证明。新样本确实仍需约 19 分钟完成首轮，这是正常长思考和旧持续循环需要分开的地方。

边界：新工作目录改变了 Pi system prompt 中的 cwd（首轮 prompt 1677，而旧为1678），Host 从旧默认全逻辑历史预留改成显式128pages/磁盘Cold。此补充证明当前配置能自然交付，不能将全部差异归因于 temperature，不能替代同输入随机种子矩阵或同模型 BF16 对照。原失败不删除；不把不同采样的 AR 新作品加入旧 greedy 的公平三臂质量结论。官方关于更大 reasoning/output 上限的建议处于其 1M 配置下，本轮仍使用用户已确认的 262K/128K预算。

原始补充收据：`tmp-nvcc/opsx-pi-ar-sampling-20261023/`。该运行使用修复前二进制，工具续轮仍重算48625 token、TTFT34.095s，为同类历史复用缺口提供了 AR 实例。

## 工具续轮：旧检查点导致最新状态丢失

原 MTP4 秦始皇首轮在13次生成回收后完成42072 token。下一模型轮只命中1671-token共享前缀，重算42099 token，TTFT29.383s，其中prefill29.353s。

问题位于终态发布：生成回收推进 `kv_offset` 与 `kv_reuse_floor`，原 prompt 的 ResponseReplay/早期 anchor 已没有匹配 KV 覆盖，却仍参与 `populate_continuation_summary`。其 frontier 加压缩偏移会成为负数，`kv_tokens` 抛异常；ResourceManager 按安全回退丢弃整条 continuation，最新有效 endpoint 一并消失。

修复复用现有 `inspect_lane` 的恢复下界与 checkpoint-drop 引用所有权规则：终态排空 Main/backend 搬运，退休不可恢复的旧私有检查点，再发布最新 endpoint。共享引用、别名检查点与 primary StateImage 继续受保护。原始 token/position/execution-boundary、Main/MTP/GDN、代际和覆盖验证不放宽。新增 `NINFER_PREFIX_TRACE=1` 仅用于诊断，输出 token ID 和边界，不输出请求正文。

理论上可避免旧续轮29.353s的历史计算，约占旧完整作品417.8s的7%；没有新增GPU常驻缓冲，不是decode内核提速。实际对照见下方收据，不将这个上限直接当成实测。

源码参考：5060ti donor `104af690541857a5d52f23ffd81ccac7eb82a675` 的 `server/dsh_adapter.py` 保持各轮稳定布局，并将HTTP回复精确回放与引擎KV复用分开；其缓存回复不适合修复这里的状态丢失。ActKV `182113cdbc6bbdd9ef300a84289075e761a1c075` 的 `follow_retrieval` 继续作为既有Main单选/backend跟随依据。DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的采样接口仅供语义核对。本次直接适配本仓既有checkpoint引用释放实现，没有移植外部设备tile或替换生成算法。

## 验证记录

| 场景 | 修复前 | 修复后 |
|---|---:|---:|
| 原Pi MTP4第二模型轮缓存命中 | 1671 | 43749 |
| 同一43770-token续轮的常规prefill | 42099 | 21 |
| 同一续轮TTFT | 29.3833 s | **0.2461 s** |
| 同一续轮prefill耗时 | 29.3531 s | 0.2214 s |
| 完整Pi作品流程墙钟 | 417.818 s | 382.943 s |
| 小工作集MTP续轮缓存命中/常规prefill | 55 / 3100 | 3135 / 20 |
| 小工作集MTP续轮TTFT | 3.4366 s | 1.4815 s |

原Pi复现使用同一旧工作目录；system sections、工具定义、用户输入一致。首轮文本与工具参数逐字相同（排除传输层生成的call ID），输出均42072 token，accepted/drafted均31217/43376，生成回收均13次。两份实际HTML的SHA256均为 `1f3ac64dd5ae350ad19eab316a187efc68b8d1bb2a45e77eae467fc1c1d6e8be`，复用已审阅作品，不重复渲染。未捕获完整native token IDs，不能把文本一致写成IDs逐位对比。

实际诊断先退休 `frontier=1673, floor=24128, offset=-10816` 的旧rewrite，再发布endpoint43749；下一请求与已存ledger前43750个token全部匹配，位置/执行identity通过，走private_endpoint。没有Pi JSON或思考重序列化失配证据。目标续轮TTFT减少 **99.16%**；这是同原请求的实际历史重建消除，不是decode内核加速。

完整流程墙钟观察值减少8.35%，但首轮运行时刻/Host配置与后续read参数、返回长度不同（旧第三轮532个常规prefill，新为1707），因此不当作等工作量吞吐结论，不相加分项百分比。第二轮响应由旧重建状态生成85token变成恢复真实endpoint后73token，第三轮均自然完成；原作品相同，HTTP正常，Serve exit0。

短回归使用2048工作集/4096物理KV、3072-token无思考生成，专门触发回收，不替代创作质量验收。MTP两版首轮正文6826bytes相同；修复后MTP与AR均命中endpoint3135、常规prefill20、回答ACK。修改首条system后两后端均拒绝旧endpoint，cache0/Root，回答ACK并正常退出。AR修复后续轮TTFT1.4564s。短回归追加新的user问题，仍可能需要窗口选页/恢复，不能把“20个常规prefill”解释为全部内部计算只有20token；真正工具续轮以上表0.2461s为准。

生产Serve构建通过；首次诊断代码引用了后声明的helper，修正作用域后重新构建成功（失败journal保留）。Python语法和受影响diff检查通过。全部GPU运行200W、串行、正常TERM或自然退出；未复跑旧213K/256K、完整创作矩阵、300W或NVFP4。

完整指标：[JSON收据](../performance/2026-10-23-generation-continuation.json)。原始数据：`tmp-nvcc/opsx-generation-continuation-20261023/{before,after,ar-after}/`、`tmp-nvcc/opsx-pi-continuation-fixed-20261023/`。前者可由 `python3 -m tools.bench.generation_continuation --output <new-dir> --expect-reuse` 复现；AR增加`--backend ar`。Pi harness默认采用model采样；复现旧greedy工具续轮须显式`--sampling greedy --profiles mtp4 --cases emperor --prefix-trace`，并用`--workspace`指定原空目录，以保持system prompt一致。
