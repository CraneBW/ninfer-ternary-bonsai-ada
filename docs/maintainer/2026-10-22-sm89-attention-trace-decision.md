# S4 / 5.2–5.3：一次 SM89 32K/K15 Nsight Systems 热点门与暂不移植决策

2026-10-22，**TRACE_ONLY / NO_KERNEL_CHANGE / NO_GAIN_CLAIMED**。这是 `2026-10-22-sm89-attention-gdn-ceilings.md` 中所要求的最小实测输入；没有回头重复旧GPU小算子测试、没有重写已有的 QK/softmax sharing，也没有把SM70 donor 的Volta IMMA、BF16 reducer或FP32 GDN暂存拷贝到SM89。

## 一、采集事实与可复现输入

仅一次现役全GSQ v3、RTX 4070 Ti SUPER / 200W、单并发、Graph、INT8 KV、max-context 36,864、物理KV 36,864、KVMem预算 32,768、prefill chunk512、MTP15。使用已有冻结32K输入的前五条消息，仅将末轮固定说明换成128-token编号生成，以保证decode段实际执行；**实际 prompt 31,987 tokens，output 128 tokens，HTTP200、Serve exit0**。非原213K/256K任务，也不是生产性能配对样本。

可复现的测试驱动（只启动和SIGTERM终止该脚本拥有的Serve子进程）及原始Nsight收据：

```
python3 tmp-nvcc/opsx-s4-20261022/nsys_one_request.py
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --force-overwrite=true \
  -o tmp-nvcc/opsx-s4-20261022/trace \
  -- python3 tmp-nvcc/opsx-s4-20261022/nsys_one_request.py
```

第二条是真实Nsight观测命令；第一条是同一输入的直接干净执行方式，**本次没有另跑配对clean性能**。原始文件：`tmp-nvcc/opsx-s4-20261022/{trace.nsys-rep,trace.sqlite,kernels_cuda_gpu_kern_sum.csv,request-report.json,serve.log}`；机器可读的分类结果：`docs/performance/2026-10-22-s4-attention-hotspot-nsys.json`。Nsight Systems v2026.3.2；所有下面的GPU时间是此跟踪中已记录的**核函数持续时间求和**，不是阶段wall或关键路径，也没有带资源争用的多轮误差条。

## 二、量化时间归因

总计166,196次CUPTI kernel records；按Nsight CUDA kernel summary去重合并有195个kernel名称，核函数持续时间合计 **22,332.020ms**：

| 内核类别（本trace实际出现） | kernel持续时间和 | 次数 | 相对总GPU kernel持续时间和 |
|---|---:|---:|---:|
| GGUF quantized matrix `mul_mat_q<...>` | **17,356.191ms** | 33,931 | **77.72%** |
| INT8 prompt causal attention | 2,165.302ms | 1,057 | 9.70% |
| Main/MTP small-T tiled attention（不含reducer） | 9.771ms | 95 | 0.0438% |
| **split-vector output reducer** | **0.463ms** | 95 | **0.0021%** |
| GDN kernel名称中 `gated_delta_net` | 719.273ms | 9,494 | 3.22% |
| GDN `l2norm_warp_bf16x2` | 19.067ms | 6,048 | 0.0854% |

真实用户请求/Serve生命周期约32.891s（含startup、TTFT、prefill、decode和Nsight开销），不能把22.332s的kernel持续时间之和错认成整会话wall，更不能将被L2缓存命中的logical流量等同DRAM吞吐。Tiled名称包含多种模板形状，不能未经再解析直接宣称每个CTA都是同一`T8/S69`。

## 三、对OpenSpec 5.2 / 5.3 的实施门结论

**5.2 暂不引入8KiB shared split-vector staging。** 当前生产路径已经在producer/consumer之间复用QK/online softmax，并非Volta donor可消去的重复算术；新增 shared staging只改变reduce的内存访问并发，不删除需写读的FP32 partial global字节，也引入同步/occupancy风险。此真实trace的reducer合计仅0.463ms，对整个33秒请求来说即使将其理想化为零、且假定完全在关键路径，也只对应约0.0014%的wall上界（纯反事实、不是观测加速）。不存在足以合理期待发布档 10% 或者甚至稳定可分辨整体收益的该分量。因此**不修改现役kernel、不伪造优化、不勾选5.2**。

**5.3 同样暂缓 T1 Q/K pre-normalize**：当前观测所有GDN BF16 `l2norm`合计约19.067ms且包含已在T>=64复用的归一化；T1 direct recurrent整个集合仅占很小部分。即使数学上的R96存在，不能把理论计算重复直接代入整会话速度增益，更不能用donor FP32暂存绕过本机BF16舍入。维持原状态转移/Graph/BF16边界，不勾选5.3。

主热点是GGUF量化矩阵(17.356s/核时间之和的77.72%)，已超出本次V100X2 attention/GDN三组件迁移题面。若要继续优化，应先在OpenSpec另开有证据的GGUF MMQ专项，并做相容性/独立数值与同binary对照；不可在S4 5.2名下偷换目标。**5.4–5.5继续NOT_RUN**，本轮为对不值得改的候选做实证拒绝，而不是宣布5.2已完成。
