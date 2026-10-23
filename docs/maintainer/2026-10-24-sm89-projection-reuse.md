# 200W decode / prefill 投影复用

本轮以 Qwen3.8 GSQ v3、RTX 4070 Ti SUPER（SM89、66 SM、16GiB）、200W 为范围，
批量实现两个复用候选，再用同一生产二进制比较。NVFP4、权重下载、GPU 数学和原题均未修改。
两个候选均已完成实现、资格和整体测量，但未确认稳定收益，开关保持默认0，不纳入性能profile。

## 来源、预算与不变量

decode 复用本仓 R27 的 canonical FP32 butterfly、协同输出和 persistent-row mapping。
原始分布式输出来源为 `Don-Chad/ninfer-3090@d8c29253f2c0845611a794d87057950107a4956b`，
`src/ops/linear/ggml/ggml_mmvq.cuh:137–146`，MIT/原 Strata 映射保留在既有代码。
本轮扩展的是已经编译的 K5120/T8 kernel，不改变其归约和解码。

prefill 复用本仓 `gguf_linear.cpp::gguf_project` 的 layout/gather 分组规则，并对照固定
`ninfer-all@b2913b65e1e09819047638a5beeb1365f7003880` 的同源实现。保留各产品原始矩阵尺寸、
full-K/fixup 路由、FP32 累加及 BF16 物化，按本机单产品 scratch 上界改为有界栈上产品组。
未照搬大规模分组、SM70 dequant/FP16 slab 或其他 GPU tile 参数；没有引入新依赖或外部框架。

已读取 5060Ti 项目 `104af690541857a5d52f23ffd81ccac7eb82a675` 的实际补丁目录与投影包装，
其中不同量化和设备专属实现没有直接替换 GSQ 路径。新增实现由本仓已有归约、量化与产品函数
组成，继续使用现有许可映射。

既有 MTP7/32K Nsight 归因是受观察者影响的 kernel 时间和，不等于 clean wall 或213K关键路径：

- GGUF 主体占 prefill 81.81%、decode 82.51%；本轮涉及的非旧R27 T8/K5120三格式投影约占
  decode 10.61%。若局部减少10%–20%，整体条件速度收益约1.1%–2.2%，不是实测承诺。
- 矩阵量化326.289ms / prefill kernel21612.098ms，约1.510%；D2D共有4618次、32.21GB、
  41.853ms，约0.193%。共享只省其中同输入产品组的重复量化和GDN复制，预计prefill约0.5%–1.2%。
- 未盲扩I64 down：I128/8warp→I64/4warp没有增加总warp，也不能把CTA翻倍当占用翻倍。
  没有新的收益机制依据时，优先消除实际重复工作。

## 实际范围及资源

`NINFER_SM89_R27_PROJECTIONS=1` 仅作用于三格式 IQ3_S/IQ3_XXS/IQ4_XS 的 SM89、K5120、
T8、plain BF16 store、连续输出，支持行数1024/2048/6144/10240/16384/34816。
原N17408的 `NINFER_R27_BUNDLE` 独立保持不变。FP32、fused、其他stride/K/T及设备走原路。

`NINFER_SM89_SHARED_MATRIX=1` 仅用于 text Variant 的 K5120、T512/1024、上述三格式的
2–4产品组。所有产品的 gather 指针和 MMQ activation layout 相同才进入；全部目标资格在
任何enqueue前验证。GDN Q/K/V直接写原packed缓冲区，Z、attention、MLP gate/up保持各自
独立输出；不合并矩阵行数、不引入新的归约树。无法共享或不支持的格式、tail回到原路。

一次共享量化复用单个已有activation buffer，每个产品分别复用原最大plane/fixup容量。
无新增权重、KV、StateImage、pinned或持久GPU分配，Graph和Main/MTP事务不变。
实测两臂workspace容量均209,484,800B；物理KV、权重和sequence容量完全相同。

## 必要资格与实际路由

生产Serve、Bench、GGUF Op构建PASS，两个并行worker、8GiB RAM/0swap限制；GPU任务与构建串行。
既有独立NumPy0.19 CPU解码+FP64完整row-sum参考按本轮几何筛选，2048行视图仅截取既有parent
独立结果；未用另一个CUDA kernel冒充独立oracle，也未重复其他格式/路径的旧矩阵。

| 范围 | 验证 |
|---|---|
| decode新增18个实际/派生几何 | T8、constant/random、开关0/1，共72进程；完整6,782,976B逐位一致，输出guard正常 |
| prefill三格式、N1024/2048/6144/17408 | T512/1024、constant/random，共48进程；完整两个输出与原产品一致，第二输出不同column stride及逐列guard通过 |
| 独立常量参考 | 各受影响产品由实际artifact与完整CPU row-sum参考验证；random另作完整原路等价证明，不把它称为随机独立FP64 matmul |
| 原生Engine短探针 | pp1024/tg16、MTP7、Graph；9条三格式N1024/2048/6144 checked发射及2/4产品组单次量化实际命中，进程exit0 |

合计84个case、120个Op进程exit0。短探针只证明实际路由，不作为性能成绩；性能臂关闭trace。
原始资格在 `tmp-nvcc/opsx-projection-reuse-20261024/qualification/`，路由在同目录 `route/`。

## 整体对照

统一GSQ、C1、MTP7 optimized head、INT8、Graph、chunk1024、已有wide和SiLU融合、36K物理池。
每臂pp4096/tg128、pp32768/tg128，Graph prime17、warmup1/r1，三对交错 A/B、B/A、A/B。
在计时之外保留所有prime/warmup/measured IDs及实际MTP drafted/accepted/rounds，不叠加历史百分比。

| 整包成对变化中位 | 4K | 32K |
|---|---:|---:|
| prefill | +0.0942% | +0.4339% |
| decode | −0.3555% | −0.3724% |
| 整段耗时减少 | +0.3109% | +0.4045% |

decode没有确认收益，R27扩展不纳入性能profile。该整包结果保留在 `pairs/comparison.json`；
只追加一次有依据的shared-matrix单独消融，结果如下；没有增加重复矩阵寻找正数。

## 单独消融与最终决定

| 仅共享量化，成对变化中位 | 4K | 32K |
|---|---:|---:|
| prefill | −0.0568% | +0.2407% |
| decode | −0.7695% | −0.7047% |
| 整段耗时减少 | −0.1686% | +0.1669% |

方向混杂且未达到预计prefill收益，两开关均不采用。decode数学未改，prefill-only仍出现decode
波动，不能把全部计时差因果归于共享量化；数据不足以确认稳定改善，也没有统计显著性结论。
累计小收益可以采用，但本轮没有确认这样的收益。已有旧R27、wide/SiLU及历史复用配置不变。

以下是仅共享量化消融中各臂三次绝对速度的中位数；它与“成对变化中位”是不同统计量，
不能用两臂中位数相除替代成对结果。

| context | 参考prefill tok/s | 候选prefill tok/s | 参考decode tok/s | 候选decode tok/s |
|---|---:|---:|---:|---:|
| 4K | 1629.81 | 1633.04 | 173.21 | 172.29 |
| 32K | 1481.14 | 1486.26 | 159.45 | 159.12 |

整包及消融合计12臂、6对，所有Graph prime/warmup/measured完整token IDs一致，实际
MTP drafted/accepted/rounds及逐位置接受量一致。两种候选的短程等价资格通过，
**不等于新增213K/256K资格**。没有胜出profile，按计划不重复长题；本轮213K/256K均
NOT_RUN，原冻结问题未修改，默认走原已验收路径。没有重新跑Pi创作，也不改写既有AR画面失败。

机器可读结果含逐对数据、完整模型/二进制/源码签名、理论条件、采用决定与边界：
[性能收据](../performance/2026-10-24-sm89-projection-reuse.json)。原始完整报告和token IDs留在
`tmp-nvcc/opsx-projection-reuse-20261024/{pairs,matrix-only}/`。

后续优化应优先针对GGUF主体访存及实际热点；这些小比例重复工作的理论上限已受限，
不再仅靠扩大相同共享域或相同R27映射反复加测。NVFP4继续延后。

## 复现

```sh
cmake --build build --target ninfer-serve ninfer_bench ninfer_gguf_linear_test -j2
python3 -m tools.bench.sm89_projection_qualification --output <新资格目录> --oracle <已有完整CPU参考>
python3 -m tools.bench.sm89_attention_gdn_pairs --candidate projections --output <整包新目录>
python3 -m tools.bench.sm89_attention_gdn_pairs --candidate matrix --output <消融新目录>
```

开关必须在进程启动前固定；回退分别设置0。日常保持200W，本轮不做300W发布终测。
