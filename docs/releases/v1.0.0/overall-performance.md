# v1.0.0 整体推理性能对比

同一最终 `ninfer_bench` 二进制、现有 Qwen3.8 GSQ、RTX 4070 Ti SUPER 上完成18个独立 Graph Engine 进程。每个工作负载按 `A B C / C B A / A B C` 运行三组，每进程 warmup1、测量1次；每次检查完整129个输出 IDs（prefill首个输出加128个decode输出），同工作负载九次全部一致。原始数值、每次完整 IDs、配置与重算字段均保存在 [overall-performance.json](overall-performance.json)。

**结果有上下文差异：C/MTP15 相对 B/MTP7，在32K工作负载的decode吞吐三对全正、中位数 +10.2196%；在4K工作负载三对全负、中位数 −15.1835%。因此保留B作为所测短工作负载配置，C作为所测32K上下文配置，不宣称普遍更快。**

## 配置与分母

| 配置 | MTP drafts | 六个性能开关 | R29组合开关 | 含义 |
|---|---:|---:|---:|---|
| A |7|全部0|0|同一个v1.0.0二进制的显式保守回退；不是产品默认值，也不是旧版本|
| B |7|全部1|0|此前已资格化的MTP7性能配置|
| C |15|全部1|1|本轮组合及MTP15；按上下文选择|

六个开关为：`NINFER_GGUF_MMQ_BF16`、`NINFER_GGUF_IQ2_S_MMQ_BF16`、`NINFER_GGUF_IQ3_S_MMQ_I64`、`NINFER_GGUF_IQ3_XXS_MMQ_I64`、`NINFER_PROMPT_I8_PIPELINE`、`NINFER_R27_BUNDLE`。总组合开关为 `NINFER_R29_BUNDLE`。

所有配置固定 `NINFER_R26_BUNDLE=0`、`NINFER_R28_BUNDLE=0`、source read-ahead/layer batch/copy batch=1、original positions/verify state=0；INT8 KV、chunk512、optimized proposal head、Graph、单并发。所有trace关闭。4K为pp4096+tg128/ctx8192，32K为pp32768+tg128/ctx36864；没有把32K工作集写成213K性能测试。

测量取Engine报告的prefill/decode阶段秒数，load/upload、Graph prime和warmup不计入这两个阶段。每个Graph进程另有prime：K7为17个输出，K15为33个输出；它们不是测量输出。decode吞吐分母为128，不是129。进程wall只作为操作receipt，包含准备工作，不能解释为TTFT。

## 各配置绝对性能中位数

下表每项分别取三个进程的中位数。

| 工作负载 | 配置 | Prefill秒 | Prefill token/s | Decode秒 | Decode token/s |
|---|---|---:|---:|---:|---:|
| 4K |A|2.175362074|1882.905|0.685999942|186.589|
| 4K |B|2.075467749|1973.531|0.674581819|189.747|
| 4K |C|2.054183914|1993.979|0.795342219|160.937|
| 32K |A|19.468513840|1683.128|0.738127469|173.412|
| 32K |B|17.896225330|1831.001|0.731307284|175.029|
| 32K |C|17.826882430|1838.123|0.663500060|192.916|

## 同组三对比较

每一组三个进程内部计算B/A（已资格配置相对保守回退）、C/B（本轮增量）、C/A（组合相对回退）。吞吐变化=`100×(baseline秒/candidate秒−1)`；耗时减少=`100×(1−candidate秒/baseline秒)`，正值表示改善。报告的是三对百分比的中位数，**不是两个配置绝对中位数之比**，也不跨历史轮次相加或相乘；两种百分比的数值不可混用。

| 工作负载 | 比较 | Prefill吞吐变化 | Prefill耗时减少 | Decode吞吐变化 | Decode耗时减少 | Prefill/Decode正向对数 |
|---|---|---:|---:|---:|---:|---|
| 4K |B/A|+5.2329%|+4.9727%|+0.7620%|+0.7562%|3/3，2/3|
| 4K |C/B|+1.0663%|+1.0550%|-15.1835%|-17.9015%|2/3，0/3|
| 4K |C/A|+5.7672%|+5.4528%|-14.6850%|-17.2127%|3/3，0/3|
| 32K |B/A|+9.1575%|+8.3892%|+1.5607%|+1.5367%|3/3，3/3|
| 32K |C/B|+0.5130%|+0.5104%|+10.2196%|+9.2721%|3/3，3/3|
| 32K |C/A|+9.5821%|+8.7442%|+11.7795%|+10.5382%|3/3，3/3|

三个配对给出这批固定工作量的实际方向，不构成置信区间或跨模型结论。B/A的短decode为混合结果，即使中位数为正，也不称稳定改善；长工作负载的B/A与C/B decode则三对全正。没有做组件消融，因此不将C/B收益归给单个kernel或单项优化。

## 原始时间与投机工作量

以下保留18次测量秒数。完整每位置接受计数、workspace、剩余显存和每次129 IDs见同目录JSON。

| 工作负载 | 顺序 | 组 | 配置 | Prefill秒 | Decode秒 | rounds | drafted / accepted |
|---|---:|---:|---|---:|---:|---:|---|
| 4K |1|1|A|2.184074690|0.685999942|17|117 / 111|
| 4K |2|1|B|2.075467749|0.674581819|17|117 / 111|
| 4K |3|1|C|2.047807729|0.795342219|12|164 / 116|
| 4K |4|2|C|2.054183914|0.790552583|12|164 / 116|
| 4K |5|2|B|2.052021139|0.669359200|17|117 / 111|
| 4K |6|2|A|2.164226240|0.674459548|17|117 / 111|
| 4K |7|3|A|2.175362074|0.686352444|17|117 / 111|
| 4K |8|3|B|2.078675638|0.687458623|17|117 / 111|
| 4K |9|3|C|2.056744777|0.808593573|12|164 / 116|
| 32K |1|1|A|19.535071210|0.743766940|17|113 / 111|
| 32K |2|1|B|17.896225330|0.731392661|17|113 / 111|
| 32K |3|1|C|17.826882430|0.665387662|9|124 / 119|
| 32K |4|2|C|17.938898530|0.663500060|9|124 / 119|
| 32K |5|2|B|18.030925230|0.731307284|17|113 / 111|
| 32K |6|2|A|19.468513840|0.738127469|17|113 / 111|
| 32K |7|3|A|19.397257470|0.737748809|17|113 / 111|
| 32K |8|3|B|17.729131010|0.726411513|17|113 / 111|
| 32K |9|3|C|17.634797470|0.644688884|9|124 / 119|

4K下A/B均17round、111/117接受；C为12round、116/164接受。32K下A/B均17round、111/113接受；C为9round、119/124接受。较大draft窗口的收益取决于真实接受长度和验证成本，不能只用round数减少推算加速。每位置计数与完整输出足够从JSON复算这些量。

## 身份、工具链与失败记录

- 模型：`qwen3_8_gsq_rco_iq3_s_v3.ninfer`，15,017,456,896 bytes；本次发布未再hash整份15GB模型。
- 同一最终Bench：`build/bench/ninfer_bench`，399,384,728 bytes，SHA256 `bbb55ee579126bffea1495b3d4ba6ed2a79719c580a203bce2803c280609bf05`。
- Corpus：`bench/fixtures/bench_corpus.ids`，65536 tokens，SHA256 `27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002`。
- GPU：NVIDIA GeForce RTX 4070 Ti SUPER，sm_89/66 SM；CUDA compiler13.4.92、runtime/driver API13.4，GNU C++16.2.1，CMake4.4.4，Release。driver API版本不冒充驱动软件包版本。

初次MTP15 Graph在启动阶段出现 `cudaErrorGraphExecUpdateFailure / topologyChanged`，进程exit1（21.570秒）。该失败保留，未计入性能样本；不能因之前Eager正确就声称Graph通过。窄修复仅在已确认的拓扑变化下先成功实例化新executable再退休旧对象，其它CUDA错误仍上报。之后这份最终binary的18个Graph进程全部正常退出、完整输出门通过，包含短C与原有Eager完整speculative字典相同及真实partial-prefix rejection。没有重复Eager计时。

此前50次public Op的数值资格通过；Graph修复只改变Core主机管理及软件版本metadata，Ops/GGML/Engine archives及数学owner未变，因而继承已有数值证明，未重复运行oracle。其SHA与关联身份在JSON中记录。18个Engine输出检查补充了真实Graph路径证据，不能替代通用数学证明。

## 可移植复现接口与边界

在仓库根目录使用JSON中每次的 `portable_command`，模型以明确basename提供。各配置环境值与所有固定开关都在JSON内；按声明顺序串行运行。构建需明确本卡架构，例如：

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_SM_COUNT=66 \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --target ninfer_bench -j2
```

JSON只发布派生标量、完整的公开benchmark生成token IDs和可移植参数，未附权重、tensor dump、二进制、冷层数据、日志或数学row-sum oracle。独立原六题质量由主发布说明另行给出，不属于这18次性能测量分母；本页不认领API TTFT、213K性能或P4性能提升。
