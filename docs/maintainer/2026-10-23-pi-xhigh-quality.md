# Pi xhigh 作品质量与长生成（2026-10-23）

当前状态：正式矩阵执行结束，5份作品完成，AR秦始皇耗尽输出预算未交付。RTX 4070 Ti SUPER / SM89 / 66 SM / 16 GiB / 200W。

## 固定范围

使用本机完整 Qwen3.8 GSQ v3（15,017,456,896 bytes），其中混用 IQ3_S、IQ3_XXS、IQ4_XS、IQ2_XS、IQ2_S、Q2_K、Q4_K；不是纯 IQ3。没有下载其他权重，也没有 BF16 同模型基准，不能将结果推广为所有量化无损。

用户指定本机 Pi coding harness，标准 system prompt 和 read/write/edit/ls，关闭额外 skills/extensions/context，独立配置，不改个人认证。所有正式臂固定相同空工作目录，先 MTP4/MTP7 后 AR，greedy/seed42/presence0；xhigh 确实进入服务 requested/resolved effort，262144 逻辑 context、131072 单次模型输出上限（含思考）、32768 KV 工作集、36864 物理 KV、INT8 KV、Graph、chunk1024。一次 Pi 作品可调用多个模型轮次，131072 不是整个 Pi 会话累计上限。使用默认前缀缓存处理工具续轮。

来源及固定原句：

- [Simon Willison](https://simonwillison.net/2024/Oct/25/pelicans-on-a-bicycle/)：`Generate an SVG of a pelican riding a bicycle`。
- [LINUX DO 原帖](https://linux.do/t/topic/2758684)：`创建一个html，使用SVG绘制秦始皇骑北极熊的2D动画`。使用其核心句，没有附加帖中后三条风格要求。

两题统一追加保存到当前工作目录的要求，分别为 result.svg / index.html，自包含、无外部依赖。保存完整 Pi JSONL、真实文件、每题服务请求日志、错误/终态、usage 及耗时。TTFT计到首个模型token，可能是思考，不是作品可用时间；作品可用延迟另报Pi总wall。进程退出 0 不能替代 assistant 自然 stop 和实际作品；接受率不等于作品正确率。

## 已发现的长生成故障

旧正式目录 `tmp-nvcc/opsx-pi-xhigh-v2-20261023`：MTP4 鹈鹕自然结束，约99.24s、11147+233输出token，其中Pi/API报告7850 reasoning token。秦始皇约286.97s后在生成中HTTP500，报 Main KV speculative batch 无法映射设备 claim，未写出作品。Pi仍返回进程0；这是运行时容量故障，不是IQ3降质证据。旧日志保留。

原因：原生成回收只在普通AR执行；MTP路径在映射前检查claim，所有speculative backend被视为几何固定，未利用轮次之间的安全边界。放大gen_reserve只能推迟失败，不能提供持续长生成。

修复复用本仓chunked-prefill的Main/MTP耦合压缩：只在完整提交、无pending及无发布endpoint的边界，Main选择后MTP跟随相同block IDs，恢复两地址增长entitlement，清空旧布局产生的未验证draft，重新计算下一轮envelope和Graph profile。逻辑ledger和已提交GDN StateImage保留，fold与后续copies在同一stream排序；异常仍交给原cleanup，不发布部分恢复。DFlash/DFlash2未套用此路径。每次压缩新增一轮anchor-only验证及搬运，没有增加常驻GPU页池。

这是有损工作集选择下的长生成，不是131K输出全attention；相同工作集的AR也采用生成压缩。新MTP4/7均已通过原题跨边界完成；仅C1/当前GSQ及配置范围，不以此外推所有长思考质量。

来源映射：ActKV `src/adapter/llama-memory-kvmem-mtp.cpp::follow_retrieval`（182113cdbc6bbdd9ef300a84289075e761a1c075）提供Main单选/MTP跟随原则，本仓现有 `kvmem_offload` / `kvmem_compact_generation` 提供原生INT8、页所有权、重相位和GPU生命周期。KVMem faf3c22582adf9d99ff072c543ea30ca03a8e2ca 与ActKV服务本身将generation_limit限制在gen_reserve，后者标注retrieval-pinned耗尽为v1限制，不能声称直接继承了128K生成。

## 为什么本轮没有直接改用DFlash2

本地已有DFlash2，现有GSQ中也有draft权重；短8K、无思考裸API鹈鹕已生成完整SVG，约125.4 tok/s。这个探索样本不能代替Pi xhigh，也不能当作公平长上下文速度排名。

同一短配置启动测量：MTP4权重12,481,729,536 bytes；DFlash2权重14,360,105,472 bytes，多1,878,375,936 bytes。DFlash2 8K规划slack1,095,900,928 bytes。把Main物理KV从8192增加到36864，按33,792 bytes/token至少增加968,884,224 bytes，只剩127,016,704 bytes（约121.1MiB），尚未扣其他长context开销，低于256MiB地板。因此共同32K工作集的DFlash2是静态容量拒绝，不是假称实际长测失败。未为了放进对照偷偷缩减KV或改权重量化。

DeepInfer（774ce244b87cc0eb8d4b94ab9f5de2ab22049f01）的confidence head有自身训练/架构要求，不直接移植到本Qwen权重。参考ninfer-4090（5c60b7c9b455231795c09da21a9fbb6aa53f08e5）的固定K对照方法，保留本机功耗、KV和资源预算。

## 判断边界与后续性能

[精确投机采样论文](https://arxiv.org/abs/2211.17192)的目标是保持目标模型分布；这里目标已经是量化GSQ，并不等于BF16。实际single-column与batched kernel的浮点归约也可能使greedy边界不同。作品不同本身不是降质，需观察完整完成、画面与原题要求，且两道作品只构成样本资格。

[llama.cpp IQ3/MTP issue](https://github.com/ggml-org/llama.cpp/issues/27373)的接受率/速度问题不是所有IQ3输出降质的证明；[DFlash/MLX issue](https://github.com/z-lab/dflash/issues/159)说明其他实现也可能出现greedy差异，不能用换后端替代本机验证。

既有Nsight阶段归因：GGUF乘法约占prefill81.8%、decode82.5%，不是全213K关键路径比例。AR短样本约45tok/s、约11.62GiB主权重流量意味着约561GB/s有效读带宽；相对标称672GB/s空间有限，且不包含其他开销，不能承诺单靠搬运让AR翻倍。候选prefill工作为把已验SiLU→MMQ融合扩到T256，避免64层共约1.063GiB逻辑激活往返和64launch；这不是实测DRAM或速度收益，需受影响域独立数值资格及真实整体对照后再决定。

正式结果如下；T256优化采用另记独立性能收据。裸API/no-thinking、被中止Pi小上下文记录仅作探索；未重跑旧213K/256K，没有300W、推送或tag。

## 已交付的恢复修复

代码、Pi harness及KV合同已在本地提交 `da62b0c`，未推送；本地AGENTS原则和OpenSpec未入提交。新正式目录 `tmp-nvcc/opsx-pi-xhigh-reclaim-20261023` 中MTP4/7两题均自然stop、工具写出实际SVG/HTML、服务exit0。MTP4秦始皇跨13次回收，MTP7跨4次；MTP4动画两个不同时间截图显示腿/头等位置改变。MTP4长工具续轮仅复用1671token，重新prefill42099token/TTFT29.383s，已记录OpenSpec12.8；可能涉及工具消息重序列化或checkpoint资格，尚未定位，不能称为新回收算法的已知根因，也不能放宽prefix/state匹配来修复。

## 正式结果

整个矩阵耗时75分21秒。所有臂均为同binary、同问题、Pi xhigh和相同预算；但生成轨迹、思考长度、工具轮数不同，所以这不是等输出工作量的加速比。decode列仅第一模型轮的服务decode吞吐（包含思考和实际回收）；wall包含Pi工具与续轮。

| 模式 | 题目 | Pi总耗时s | 总输出token | 其中思考token | 首轮decode tok/s | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MTP4 | 鹈鹕 | 101.77 | 11125 | 7223 | 111.21 | SVG完成、核心场景通过 |
| MTP7 | 鹈鹕 | 122.39 | 13236 | 9737 | 109.99 | SVG完成、核心场景通过 |
| AR | 鹈鹕 | 222.92 | 9814 | 6288 | 44.33 | SVG完成、核心场景通过 |
| MTP4 | 秦始皇 | 417.82 | 42347 | 35143 | 109.83 | HTML完成、核心场景和动画通过 |
| MTP7 | 秦始皇 | 363.53 | 34347 | 26899 | 102.03 | HTML完成、核心场景和动画通过 |
| AR | 秦始皇 | 3273.61 | 131072 | 131072 | 40.07 | 输出上限，仍在思考，无文件 |

AR最后请求HTTP正常完成、finish为output_limit，Pi stop_reason=length且退出0；harness正确返回失败码1，Serve日志显示正常stopped，随后GPU释放至59MiB。harness旧版没有在异常finally持久化这个Serve子进程退出码，因此该字段保持null，不能用父进程1冒充服务崩溃，也不补造子进程exit0。随后harness已加finally记录。

三个鹈鹕均实际渲染，能识别喙囊、双轮自行车、鸟身和脚踏；没有观察到MTP版本新增的核心场景失败。两份秦始皇HTML均自包含，帝王装束位于白熊背上；实际浏览器渲染并查看不同时间帧，MTP4含明确腿/头变化，MTP7含CSS动画及画面变化。两者造型、历史形制和骑姿仍有简化，非盲评、非精细美术评分。AR没有秦始皇作品，缺完整三臂该题质量对照。

**本轮没有建立“IQ3+MTP导致降质”的证据，也没有证明所有IQ量化或所有MTP输出都无损。** 实测仅一个混合GSQ；AR的长思考失败可能涉及greedy、有限工作集/重相位及模型本身，当前未做因果分离。不能把它归因于量化，也不能仅凭MTP更早结束宣称质量完全等价。后续优先以适配该模型的采样和原位置/压缩控制定位AR收束问题，不在本轮为取得成功而重抽长样本。

按当前两道Pi xhigh样本，MTP4是较保守的吞吐选择，第一轮约110tok/s；MTP7秦始皇wall更短主要伴随输出更少，并非decode更快。保持原全局默认不变，既有MTP7固定语料profile继续保留，不宣称一个宽度适合全部任务。

## Host预算缺口及修正

本轮harness遗漏 `NINFER_KVMEM_HOST_PAGES`，仅设置了context-cache的 `--host-kv-mib 128`。KVMem懒分配Main Host arena采用4096pages，全量约8.25GiB；该配置对所有本轮臂一致。实际进程组峰值21,489,479,680 bytes、swap峰值561,532,928 bytes，观察到oom/oom_kill=0。这些内存压力和一次短暂吞吐波动留在原结果，不作为干净稳定性能基准。

后续harness默认显式128pages，Main Host上限276,824,064 bytes（264MiB），MTP Host约16.5MiB；另保留context-cache128MiB，两个预算不能混淆。相比原默认，Main Host预留按代码公式减少约7.99GiB（理论容量差，不是新轮峰值实测）。Cold目录显式放在结果目录的backing子目录，所在分区 `/dev/nvme0n1p4` / Btrfs，复用既有P4的有界Host/Cold实现。没有回写或重跑本轮长矩阵；本轮质量对应旧Host配置，新配置的存储数学/排空资格继承已有同实现收据。

[结构化逐请求收据](../performance/2026-10-23-pi-xhigh-quality.json)包含实际输出/思考量、接受量、TTFT、停止原因、文件路径/哈希与失败。报告中的核心质量判断来自实际查看生成作品，不由接受率推导。
