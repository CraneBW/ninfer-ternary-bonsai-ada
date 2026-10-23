# 最终 SM89 宽批配置对照（2026-10-23）

本轮选定显式 **Qwen3.8 GSQ v3 / MTP7 / optimized head / INT8-G64 / Graph / chunk1024**。`NINFER_GGUF_LARGE_PREFILL=1`；新attention split-vector和GDN prenorm均为0，bulk异步未合入，全局默认仍不变。200W三对确认整包收益后完成300W一对发布前验收，并恢复200W；未推送、打tag或发布新版本。

参考为同binary、同模型/语料、相同MTP7与Graph、chunk512/wide=0；候选为chunk1024/wide=1。resident物理KV均36864，pp4096/tg128与pp32768/tg128，Graph prime17、warmup1、正式r1。32K resident测量没有触发超池R21，不能将其称作213K TTFT。长期原题证据单独继承，单项百分比没有相加。

## 200W三对

按reference/candidate、candidate/reference、reference/candidate执行。速度单位tok/s；绝对列为各臂独立中位，改善列为对应三对的比值中位，二者不能互相代算。

| 输入 | reference prefill | candidate prefill | 成对改善 | reference decode | candidate decode | 成对变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1603.70 | 1666.16 | +3.461% | 176.44 | 175.76 | -0.213% |
| 32768 | 1454.14 | 1512.55 | +4.233% | 161.25 | 160.85 | +0.345% |

完整Graph prime、warmup与正式输出逐token一致，全部正式臂输出129 tokens（含128 decode），实际MTP rounds/drafted/accepted及各位置接受量也相同。decode没有跨档稳定改善，主要采用依据是prefill及完整段耗时。

首次运行在第二对候选期间以143中断，原因未确定；systemd记录Result=success、2.1G peak，没有oom-kill。第一对完整成功结果保留，未完成的候选不进入统计；同一binary/source/model/语料续跑余下两对，原始中断收据和续跑来源均保留。未重复第一对或删除失败尝试。

## 300W发布前终验

仅一对reference/candidate，功能、完整输出、接受工作量和资源验收；以下为该次样本，**不认领300W三对稳定收益**。此轮期间未改源码/二进制；finally及systemd退出动作恢复200W，收据记录实际恢复值。

| 输入 | reference prefill | candidate prefill | 成对改善 | reference decode | candidate decode | 成对变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2114.24 | 2164.75 | +2.389% | 205.37 | 205.00 | -0.179% |
| 32768 | 1929.97 | 1978.18 | +2.498% | 187.67 | 187.32 | -0.189% |

## 长上下文、P4与资源

最终未开启新S4长状态或bulk流水，复用未变路径的[冷首轮与P4收据](2026-10-23-cold-prefill-and-bulk-p4.md)，本轮未重复原213K/256K。该长题证据来自前一冻结binary，**不是本轮最终binary新跑的长题**：original+causal-history+content-anchors/MTP7/chunk1024，32768总工作集、36864物理池；原213K三对36/36、原256K6/6。213K冷首TTFT候选153.797/155.579/155.133s，成对减少中位4.685%；session减少中位5.621%。后续多数0.67–0.88s，Amber仍2.28–7.24s。256K单臂首TTFT191.897s、session203.026s，仅质量资格；两档history replay0/query replay162，终态active/inflight/pinned/producer均排空。

canonical MTP15的新bulk P4是独立配置：36/36、真实237–238页Cold、四store固定worker命中，staging8.765625MiB不增、终态pending0；三对session +0.849%/+0.386%/−0.672%，尚未区分噪声，不将其叠加到original MTP7 profile。

resident workspace从106,914,816增至209,484,800 bytes。每臂memory与workspace峰值见[结构化收据](../performance/2026-10-23-final-sm89-profile.json)，不能将resident启动余量当作超池Serve余量。已有长Serve wide配置可用显存1,761,738,752 bytes；实际两种域均高于256MiB地板。300W仅此次终验，系统默认200W。

## 复现与未完成边界

日常200W在仓库根运行：

```sh
python3 -m tools.bench.sm89_attention_gdn_pairs --candidate final --output <新目录>
```

发布前才临时设置300W，并使用相同工具的`--release-acceptance`做一对；调用方必须在退出时恢复200W，工具自身只核验功耗。长题命令继续使用[既有宽批入口](2026-10-23-cold-prefill-and-bulk-p4.md#复现与采用)，保持原题不改。

[S4实现](2026-10-23-sm89-attention-gdn.md)已有完整Op数值资格，但无胜出组合，新S4原213K为NOT_RUN。DeepInfer owned-step因未发现足够host等待机会仍未移植；NVFP4按用户要求暂停，缺完整权重且当前绑定超16GiB。以上均未勾选为完成，不将工具/报告通过当作功能通过。
