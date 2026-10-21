# 一小时优化截止交接与恢复记录

以下为01:04暂停时的历史状态；当前恢复后的结果见文末各轮续记。

截止：2026-10-21 01:04:03 Asia/Shanghai；闹钟ninfer；用户明确要求到点暂停，不自动恢复。优化工作已结束；主线程在截止收尾时暂停目标，时间 2026-10-21T01:04:30.946962+08:00。所有本会话构建/GPU/Serve句柄已终态，不自动恢复。

- 已有Qwen3.8和原题保持；不下载模型/不重新量化。
- R5：MTP3与MTP5分别完成真实ID成对测量和各自原213K6/6。K3常用显式配置，K5适合高接受率；全局spec保持off。见[报告](../../tmp-nvcc/perf-ada-round5/report.md)。
- R6：孤立K x4算子慢13.5%–14.2%，已恢复原内核并重建；工具入口保留。见[拒绝记录](../../tmp-nvcc/perf-ada-round6/report.md)。
- R7：Engine chunk1024未配对差值仅0.22%、workspace多81.4MiB，保留512，不扩大验收。
- R8：完整INT8 producer/worker流水，16例oracle/arm及全部字节一致，32K历史完整Op +47.11%/+50.21%；新Serve原213K6/6通过。候选默认关闭，NCU计数器及整模型配对仍缺，见[报告](../../tmp-nvcc/perf-ada-round8/report.md)。

恢复后的最短工作：用只实例化当前INT8内核的小profiling harness减少NCU自身主机占用，维持4GiB隔离上限；运行已准备的round8/compare-engine.py（真实pp32768+tg16、ctx36864、三对ABBAAB）。当前内核未再改变时不重跑已过原六题；只在证据完整后决定默认采用。未运行项保留NOT_RUN。

当前生产原内核仍为默认；R8只有NINFER_PROMPT_I8_PIPELINE=1可进入。R5实际测量二进制保存在round5/measured-bin；R6拒绝候选和R8baseline/patch都已保存，无提交推送。OpenSpec15.4继续未完成，strict结构验证通过，不宣称设备最优。

闹钟配置后来出现额外关机动作，与本会话暂停授权不一致；未执行关机，已向用户单独询问。提醒现已PAUSED，避免重复触发。

## 用户恢复

2026-10-21T07:38:29.069998+08:00 用户明确要求继续，目标已恢复。先前一小时暂停为已结束的历史阶段，不自动续设新闹钟；用户明确仅暂停优化，关机已排除。R9复用未改数学的R8候选，补齐整模型配对和小型独立NCU，不重跑已通过原题。


## R9 恢复后采用结果

完整INT8 prompt流水已成为SM89 plain INT8默认，`NINFER_PROMPT_I8_PIPELINE=0`可回退。32K输入的Engine ABBAAB三对prefill中位+4.764359%，六进程完整17 IDs相同；512输入补充没有稳定收益。小harness已在4GiB/0swap内完成NCU，最大峰值689.4MiB，保留local traffic上升的计数事实；不声称无spill。最终构建和env未设的真实launch通过，R8原213K6/6质量证据保持、不重跑。详见[第九轮报告](../../tmp-nvcc/perf-ada-round9/report.md)。原闹钟仍暂停；当前无新的用户暂停时限，不关机。15.4继续未完成。


R10/R11恢复后续筛均已拒绝：runtime-K两大MLP桶全负，部分Q scale缓存的kernel早筛全负。候选与失败证据保留；生产仍为R9，两个候选均恢复。参见[第十轮](../../tmp-nvcc/perf-ada-round10/report.md)与[第十一轮](../../tmp-nvcc/perf-ada-round11/report.md)。未为失败候选追加整模/原题矩阵。


R13续记：当前MTP5真实NSYS已澄清IQ3_S为T6；复用MTP7在固定pp4096+tg128三对decode中位+11.133378%，完整129 IDs不变。新发现backend compaction后漏恢复增长预留，已复用bundle resize修复，K7新Serve原213K六题6/6完整，serverexit0，在飞0；首次ACK后失败/余5未发的arm保留。K7只作高接受率显式配置，全局spec仍off，没有新增短档/TTFT/P4收益。R14 source-K层合并已通过独立oracle及闭合restore计时，正在原题质量验收，尚未默认采用。参见round13/report.md及round14/status.json。当前无新暂停时限，不关机，15.4未完成。


R14最终采用：Main16 source-K层合并发射已默认1，NINFER_KVMEM_SOURCE_LAYER_BATCH=0回退；完整restore latency三对中位减少23.075908%，不是整条TTFT收益。新Serve/K7/Graph/sync原213K6/6完整，在飞0；最终构建和env未设的实际生产TU路线全部终态通过。首次夹具漏RDC的失败已保留并修正，没有放宽oracle。报告在round14/report.md。R15 IQ2_S独立CPU fixture已生成、GPU未跑；R16已有3090 copy_batch复用方案、尚未实现。所有自有GPU/build/Serve句柄终态，无新OOM；15.4及持续优化goal仍active，未续闹钟、不关机、无提交推送。


R15/R16续记：用户再次明确优先现有Qwen3.8 GSQ。R15 IQ2_S完整Op三对中位+6.630609%，但Engine三对中位仅+.273332%且一对负，保留已资格化opt-in，NINFER_GGUF_IQ2_S_MMQ_BF16默认0，不追加原题。R16 source-K批次H2D在layer-batch=1上完整restore latency中位进一步减13.529041%，独立oracle/18,384,064B全等，原213K新binary6/6完整、在飞0；已默认1，NINFER_KVMEM_SOURCE_COPY_BATCH=0回退。最终构建与env未设的实际batch+layer路线均终态通过。GPU仍32段copy，不外推整条TTFT或乘跨轮中位。R17只有CPU准备的GSQ IQ3_S I64/J128/4warp候选，未构建/GPU。报告分别round15/report.md与round16/report.md；所有GPU/build/Serve句柄终态，goal仍active、15.4未完成，无新暂停时限、不关机。


R17最终续记：IQ3_S I64完整Op +7.181572%，但单独Engine mixed，不默认采用。用户询问组合后，IQ2_S BF16+IQ3_S I64同binary BOTH0/BOTH1三对prefill均正、中位+1.275941%/25.991701ms，完整129 IDs相同、两Engine hook实际命中。最小TU宏方案已接入生产，两全局default仍0；显式GSQ组合both1。正常构建、生产oracle/17,825,792B与scratch exact通过，新生产原213K6/6完整、server/service exit0、263.980s/20G/2Gswap、在飞0，无新OOM。只作显式组合资格，不加单Op中位数、不报长TTFT收益。R18实际MTP7/T8热点已定位，逐列消费候选只CPU准备、尚未build/GPU。所有自有句柄终态，目标active/15.4未完成；原闹钟继续PAUSED、不关机。详见round17/report.md和round18/README.md。


R18/R19终态：R18 IQ3_S/T8逐列消费虽然182→80regs且数学全字节通过，但完整Op三对全负、中位-1.097509%，拒绝且生产不动。R19 IQ3_XXS I64完整Op+6.258693%，原R17双flag1固定后，仅新flag切换的Engine三对均正、中位增量+.849449%/17.13358ms；六次完整129 IDs一致。已最小TU-local表hook接生产，三个globaldefault均0，显式三格式profile均1。正常build/probe通过，原213K新生产6/6完整、server/service0，262.524s/20G/2Gswap、在飞0，非OOM。没有跨轮合成总收益或长TTFT加速声明。报告见round18/report.md与round19/report.md。

下一优先是R21参考KVMem的完整original-position KV/pre-query完整StateImage/query-only replay路线，先完善当前设计与生命周期/identity边界再实现，不直接停canonical replay或复用旧条件化state。R20 T16现有MMQ/vec8+8仅CPU准备，MTP上限7和FP32头边界未变；作为随后decode组件实验。所有自有GPU/build/Serve句柄已终态，goal active、OpenSpec15.4未完成，无新闹钟、不关机、无提交推送。


R22实际修复及验收（2026-10-21）：已解开KVMem逻辑descriptor下限与global Host payload的耦合，并修复root admission将独立KVMem归档虚报为global Host debt；真实global Host副本/demotion/restore及KVMem自己Host/Cold配额不删。HostKV0的当前canonical三格式GSQ原213K六题6/6、完整终态、server0；248.378s/20G峰值/swap776.3M（20G/2G限制），非OOM，global Host容量与占用0。history185836/reuse10240/query162、compaction718/Coldspills3010/sourceKrebuild36998、在飞0，与R19工作量相同。初次metadata-only版本的准入拒绝保留，未改题也未增大工作集。

R22同一最终binary、仅global HostKV8192/0的ABBAAB三对已全部终态，6fresh processes/36原请求全部正确、所有server/service exit0；unit ninfer-r22-ttft-paired/session59798结束，26min27.490s/整矩阵20G峰值/swap2G。首题长TTFT三对内减少中位-0.110288%，没有稳定改善；后五题TTFT合计三对均正、中位减少2.262371%/2126.038ms。当前C1/canonical/三格式GSQ显式长上下文profile采用--host-kv-mib0，全局默认不变，不宣称首次长prefill加速或RSS少8GiB；原KVMem归档/StateImage保持。R21新原位置模式仍失败且默认0，16.3/16.4和15.4继续未完成。详见目标仓tmp-nvcc/perf-ada-round22/report.md及ttft-paired/ttft-summary.json。

用户补充的5060 Ti/DeepInfer来源已完成源码比对，见[吸收报告](2026-10-21-donor-optimization-reuse.md)及[CUDA服务规程](cuda-service-operations.md)。没有现成保持当前数学的SM89/C1性能补丁；GDN FP16/FP8写回有损，SWA bounded replay近似、不可替代完整GDN恢复；materialize只有并发上限差异。HTTP read1本机已有，不额外建代理。禁止主动SIGKILL的owned-child helper已完成两个CPU子进程行为检查；generic runner接入patch保存在perf-ada-donor-20261021/qualification-graceful.pending.patch，**等当前R22矩阵实际终态后应用**，当前冻结runner不改。新GPU服务显式SendSIGKILL=no；不关机、不恢复闹钟、不提交推送。

R22终态后generic qualification已应用优雅停止helper，原frozen runner不改；新R23服务SendSIGKILL=no。R23 I64/J64/128线程两格式168regs/APImax3CTA，独立oracle与17,825,792B/arm random bytes通过，但完整Op吞吐三对全负：IQ3_S中位-12.894310%、IQ3_XXS-9.924623%，均拒绝，不跑NCU/Engine/长题。初次误把40B stack当spill的仪表失败保留，新旧ptxas均零spill，已据此修正断言。R24源K有界双slot读前正在default0最小实现；R25 decode符号LUT仅CPU准备。
