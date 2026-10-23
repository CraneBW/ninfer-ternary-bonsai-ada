# Qwen3.8 工具续轮的原始分词回放

本轮针对正式 Pi xhigh 样本仍有的 MTP4 22.052s、AR 36.451s 历史重建，沿用已有
Qwen3.8 GSQ v3、RTX 4070 Ti SUPER、200W，恢复已完成输出的原始 token 切分。
CUDA 算子、量化、KV 选页策略与原位置配置不变。

## 原因与收益预算

MTP4 的续轮输入 33950 token，原 ledger 33889、有效 endpoint 33888；首个分歧在 32804，
新输入 token 123177 是“下雪”，原生成 token 95849 是“下”。AR 对应输入 53180、ledger
53118、endpoint 53117；首分歧在 43643，新输入 95779 是“大”，原生成 145007 是“大秦”。
MTP4 分歧位于工具参数正文，AR 位于工具调用前的末段思考。保存的文字相同，BPE 重新编码可以改变分词；完整状态检查
因此拒绝旧 endpoint，原样本退回共享前缀 1671，分别重算 32279 / 51509 token。

保留已提交的原 token 有机会把上述重算降为工具结果及下一轮 opener 的几十个 token，
理论可省掉几乎全部 22 / 36 秒 prefill。新增记录无 GPU 分配；8 条，每条最多 262144 IDs、
2MiB 文本及 4096 个 added-token 标记，持久有效载荷约 24.5MiB 上限，另有容器和瞬态映射。
每个活动文本请求还保留一份 prompt IDs，262K 上限约 1MiB。

## 实现与安全条件

复用家族 `EncodedHistoryCache` 的有界存储、互斥与字节前缀规则，但将已生成 token 来源
独立保存，避免混入要求 cold-encode 完全一致的普通 Host 编码缓存。Engine 完成模型、输出
提交和 terminal retention 后，OutputSession 才发布已提交的 prompt/generated IDs。

仅接受自然 im_end、非 raw、无媒体的完整文本。续轮必须满足完整前缀逐字节一致、所有
added-token ID 及字节位置相同；literal 形式的特殊标记不能冒充结构。完整重映射 rewrite、
message、cache frontier；某个所需边界落在旧 token 内就整体放弃替换。随后重建文本位置、
查询/retention span 和预算，Program 的 token/位置、GDN 执行分界、StateImage、KV 和 epoch
资格检查继续生效。没有匹配 KV 时，记录本身不会制造 cache 命中。

默认开启；回退为启动 Engine 前设置 `NINFER_GENERATED_HISTORY_REPLAY=0`。
`NINFER_PREFIX_TRACE=1` 输出 `generated_encoding canonical=... original=...`，不打印用户正文。

## 验证范围与复现

使用保存的两份 Pi `message_end` 记录，调用已安装 Pi provider 的 `convertMessages`，按 Serve
相同顺序生成 tool definition。思考和 HTML 原文完整保留，xhigh、262K 逻辑上下文、32K 工作集、
36K 物理 KV、INT8、chunk1024、Host128pages、sync、C1、Graph、200W。

冻结历史导入在已观察位置使用字节完全相同的原分词选择，保留对应 GDN 执行 frontier，再
真实 GPU prefill，生成自然 im_end，随后从公共 Engine 运行结构化工具续轮。三臂为同一 binary
关闭回放的普通编码、开启回放、关闭回放但显式输入原 token 的正对照；每臂最多生成 64 token、
greedy/seed42，后两臂核对完整生成 IDs。候选另改动首条历史，确认旧状态被拒绝。
这是保存历史的续轮回归；没有重新运行 9–20 分钟首轮创作，不作为新的作品质量或采样评测。

内部 fixture 入口只包装家族 PreparedPrompt 的完整元数据，不提供 KV 或资格访问。初版裸 token
导入遗漏 GDN 分界、identity=0，被 Program 正确拒绝；该失败保留在 r4 收据，未放宽检查。
最初几次准备失败发生在加载模型前，分别为 systemd PATH、v3 资源读取、缺预处理配置及
device-budget API 配置，原日志保留。

```sh
cmake --build build --target ninfer-serve ninfer_qwen3_6_generated_history_test -j2
python3 -m tools.bench.generated_history --output tmp-nvcc/generated-history-new
```

脚本自动从现有 v3 模型提取 frontend 资源；需已安装 pi 和 node 在 PATH。资源目录亦可通过
`NINFER_OFFICIAL_TOKENIZER_DIR=<输出目录>/frontend` 用于 CPU 边界测试。原始证据在
`tmp-nvcc/opsx-generated-history-20261024-r5/`（MTP4）和 `r6/`（AR），性能汇总见相邻 performance JSON。

## 实测结果

| 后端 | 原 Pi 失配 TTFT（历史） | 本轮回放关闭 | 本轮开启 | 复用 / 新算 token | 原 token 正对照 |
|---|---:|---:|---:|---:|---|
| MTP4 | 22.052s | 23.042s | **0.159s** | 33888 / 63 | 64/64 IDs 一致 |
| AR | 36.451s | 37.857s | **0.247s** | 53117 / 63 | 64/64 IDs 一致 |

本轮开关对照分别减少 99.31% / 99.35% TTFT；每档一组必要功能/性能对照，不推导 p95 或
跨负载稳定 SLA。关闭回放臂没有首轮原始共享前缀，Root 从 0 重建33950 / 53180 token，
所以其 23.042 / 37.857s 与历史退回1671共享token的 22.052 / 36.451s 分列。

MTP4 恢复原分词后 prompt 33950→33951（多1个原始token），AR仍53180；实际预算同步更新。
正确性正对照为延续原 token，而非把旧的重编码输入作为相同数学输入：两种 token 切分的后续
文字允许不同。本轮没有改写已有作品。两候选修改首条历史后复用均为0，六个有效三臂进程均
exit0；官方 tokenizer 字节、结构标记、frontier、编辑历史和自然结束边界检查及既有 Frontend
测试 PASS，生产构建 PASS。默认启用回放，功耗仍200W。

MTP4 完整三臂收据在 r5，AR 原始第43643位置完整三臂在 r6（r5另一次 AR 试验定位于更晚
的同文分词，未用于本表；确定原始位置后主动 TERM，未用 kill -9）。
[机器可读结果](../performance/2026-10-24-completed-output-replay.json)。

## 来源

- 本仓家族 Host 编码缓存源自 dylanbrodiefafard `48d1857` 的增量编码适配，本轮复用其存储边界，
  新生成记录不改变旧缓存的 cold-encode oracle。
- ActKV `182113cdbc6bbdd9ef300a84289075e761a1c075`，
  `tools/llama-kvmem-driver.cpp` 的 `common_token_prefix`、GDN catch-up/restore 及 server 的
  token-prefix 身份规则；本轮保留现有 Main/MTP/StateImage 资格，不引入文本近似匹配。
- 5060Ti 项目 `104af690541857a5d52f23ffd81ccac7eb82a675`，`server/dsh_adapter.py` 的
  stable-prefix、原请求字节精确缓存及工具副作用边界作为对照；本轮实际复用 KV，没有回放旧 HTTP
  响应或跳过工具执行。现有来源许可映射继续保留，新代码使用本仓许可。

NVFP4 仍延后；未重复 213K/256K、算子数学、长作品矩阵或 300W 发布验收。
