# Canonical续问复用：减少历史重复重建

2026-10-22；已有Qwen3.8 GSQ，RTX 4070 Ti SUPER / SM89，日常200W。

## Result

**固定213K三对36/36、256K六题6/6精确正确，采用为固定快照基准工具的默认续问策略。** 同binary、相同原题与32K总工作预算，213K六题会话用时成对减少中位75.49%，后五题TTFT合计减少98.18%。保持native运行时全局默认0；普通Serve需显式启用，资格限本次C1/MTP15事实续问配置。

| 指标 | 参考 | 续问复用 |
|---|---:|---:|
| 六题client session wall，中位 | 145.232s | 36.110s |
| 后五题TTFT合计，中位 | 111.160s | 2.027s |
| 后续单题TTFT，15个样本中位 | 22.061s | 0.404s |
| 首请求TTFT，中位 | 33.819s | 33.544s |
| snapshot load，中位 | 10.733s | 10.667s |
| 历史回放token / 轮数 | 185836 / 6 | 32228 / 1 |
| 历史复用token累计 | 10240 | 161544 |
| query replay token | 162 | 16 |
| source-K恢复调用 | 3398 | 522 |

表中秒数为各臂独立中位；三对成对session减少76.0404% / 75.4912% / 74.9600%，后五题合计减少98.1813% / 98.2000% / 98.1764%。load成对有升有降，中位仅减少0.6182%，不认领加载收益。首请求仍包含完整快照加载与一次canonical工作集重建，本批主要解决后续重复计算。

## Theory and source reuse

优化前工作量是185836历史token。已有trace中后五题约110s，若历史状态和所选页可合法保持，每次命中最多可省约30–32K回放、约22s计算及相关搬运。与仅重叠几次copy相比，这才是本轮主要空间。没有把这些上限直接当作性能结果。

优先查看KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 的 `memory_checkpoint_frontier`、`capture_memory_checkpoint`、`restore_memory_prefix`，以及ActKV对应patch；复用其“精确前缀状态可合法续算”的结构。实际实现小幅适配本机已有StateImage、canonical replay cache、Main/MTP coupled selection和generation reserve，没有迁入另一套runtime。具体来源为 `ecosystem/kvmem/backends/patches/ninfer-kvmem.patch` 与 `ecosystem/melis-actkv/backends/patches/ninfer-kvmem.patch`。

实现是相容批次：

1. **完整canonical状态资格。** 历史回放完全结束后，记录当前精确选页位置、token序列、frontier和StateImage generation。只有已知fork的真实后代可继承；保留仍合法的immutable source资格，清除已释放代际。较早写入、owner或工作集改变使资格失效。初次从ABI1快照加载没有这份资格，必须先重建一次。
2. **严格续问门。** 仅C1、纯文本、MTP；当前完整状态须属同一canonical后代，Main/MTP位置一致，选中历史prefix的位置与tokens精确一致，所有词法候选页均有已提交的驻留KV，且当前后缀与`2*draft+1`能放进工作预算。未满足时沿用原probe/选择/重建，不把“没有请求restore”误当成“证据已经驻留”。
3. **实际留出已有reserve。** 完整canonical选页使用现有 `--kvmem-gen-reserve 512` 留出续问空间。原511页几乎填满32K，只够一小段后缀；改为历史及512预留合计不超过同一32768预算。没有扩大工作集、物理KV或模型上下文，没有新增参数。
4. **命中后直接续算。** 不probe、不重选、不清零或复制GDN、不回放历史，仍只捕获一次当前问题Q供正常LRFU提交；沿用原CUDA计算、MTP/Graph提交和生成回收。

资格metadata只有有限的live StateImage句柄，位置核对暂存两个至多32K的vector；没有新增Device/pinned分配。四Host replay槽仍615817216B，source staging仍2297856B。

## Qualification and whole-session comparison

固定配置：native v3 GSQ混合GGUF块、INT8 KV、C1、Graph、prefill512、MTP15、physical KV36864、KVMem budget32768、gen-reserve512、HostKV0、action-LRFU、sync、indexed snapshot、R27/R29=1、R26/R28/canonical-prepare/original-positions=0。原213K逻辑上限212992，首请求212276 tokens，恢复212260；原256K上限262144、首请求261411、恢复261395。没有修改请求、期望答案或模型。

最终候选先通过213K原六题，trace关闭的该资格臂复用为首个候选样本，再完成C/B、B/C、C/B三对；所有36个响应精确正确、HTTP/terminal/usage完整，成功终态active requests、inflight pages/bytes为0，所有server正常退出0。三次候选均五次直接续问，工作量计数完全一致。所有参考继续六轮canonical重建。

256K再用**工具新默认**验证一次，6/6精确，五次直接续问；load14.479s，只有一次canonical历史回放32211 tokens。该256K样本用于功能资格，没有做三对性能，不把213K百分比外推为256K性能。

所有GPU工作200W；原213K paired整体9m2.964s、20G RAM /169.7M swap峰值；256K46.864s、20G/3.6M。各在20GiB/2GiB限额内，正常TERM/drain，无OOM；最终构建47.510s、2.6G/0swap。未改CUDA数学，继承已有数值证据，不重复resident4K/32K或整模型NCU。

最终实测binary SHA256 `4c7bc7edc2c4b084cee29344bc43e041cd5b011bf927ef1d32c03685f1f208b7`。native代码在计时后没有修改；随后仅将已通过开关设为固定基准工具默认，并通过实际256K默认入口验证。完整秒数、各臂配置、工作量与收据索引见[结构化报告](../performance/2026-10-22-canonical-query-reuse.json)；本机原始资料在 `tmp-nvcc/opsx-canonical-query/`。

CPU诊断记录的roundtrip和prefix、完整状态、q0、容量不匹配拒绝通过，1.066s /276.2M/0swap；其结果属于R21诊断记录，不充当新的模型质量或性能样本。

## Usage

固定快照日常测试默认启用；参考臂用 `--no-canonical-query-reuse`：

```sh
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 213k --output <新目录>
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 256k --output <新目录>
python3 -m tools.bench.prefix_checkpoint restore --no-checkpoint-load-pipeline --checkpoint-stage raw --case 213k --no-canonical-query-reuse --output <参考新目录>
```

普通native Serve使用原有显式长档命令，并添加 `NINFER_KVMEM_CANONICAL_QUERY_REUSE=1`；设0回退。保留 `--kvmem-budget 32768 --kvmem-gen-reserve 512 --spec mtp --draft-tokens 15` 及上述已验配置。没有通用的“词法页驻留即语义充分”保证：这是保持已有canonical视图的选页策略，不保证与每题重新选择的输出逐token相同；本次固定原题逐条精确正确。未验工作负载和并发/投机组合不自动继承。

## Boundaries

- 加载原ABI1包，首问仍重建一次；不是把snapshot load当cold-prefill速度。client session wall包含load及六题，排除模型/服务启动。没有flush OS page cache，不报告物理NVMe冷读带宽。
- 原位置R21仍默认0并保留错答。[q0因果对照](2026-10-22-original-replay-causal-control.md)已排除其首错的probe/restore字节损坏，但原时间线conditioning/位置的细分原因仍未定位。本优化续用已经正确的canonical状态，不将R21原位置路线冒充通过。
- 本轮sync，无新增真正layer-ready/released流水；75.49%不是Async P4结果。逐层页流水和原16.x验收仍未结项。
- resident prefill/decode算子未改，没有新的resident tok/s结论；300W发布终测NOT_RUN。无新发布、push或tag。
