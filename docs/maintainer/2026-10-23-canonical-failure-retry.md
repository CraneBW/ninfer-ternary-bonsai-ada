# 原位置软件恢复失败：同请求canonical重启

2026-10-23，Qwen3.8 GSQ v3、RTX4070 Ti SUPER、200W、C1、MTP3、CUDA Graph。

## 实现与边界

沿用Strata 9de04cd的完整checkpoint配对、KVMem faf3c225的代际撤销和本机已有Program清理顺序。原位置恢复失败后，只有设备同步成功、Main/MTP旧owner与StateImage已退休、无context transaction，才发出DrainedRequestFailure；不能验证的设备/CUDA错误继续走Engine-wide failure。

在首token尚未发布、原请求未取消、非TP且单lane时，Engine将同一请求最多重新准入一次。Program保留的是原始owning PreparedPrompt，不是答案或旧KV；先退休所有原位置continuation/catalogue，验证两个KV store为空，再启动新的canonical archive epoch，并销毁旧Graph后重新捕获。常驻模型权重不重复加载。切换后服务保持canonical，不在旧archive内热切换位置语义。

Engine在物理退休后清空逻辑catalogue，保留原请求身份、输出session、TTFT起点和队列容量；不重复发送generation-start，重算期间prompt progress不倒退，累计计算计入两次尝试。已经取消或开始输出的请求不重试；没有隐含无限重试。原位置实验启动默认仍OFF，正常canonical路径不受影响。

## 实测

运行tools/bench/original_recovery_smoke.py，各场景一个Serve进程、两条流式请求：

| 注入位置 | 原请求 | 随后不同请求 | 退出 |
|---|---|---|---|
| after_main_rewind | HTTP200、ACK、完整usage及DONE | HTTP200、BETA | 0 |
| after_state_h2d | HTTP200、ACK、完整usage及DONE | HTTP200、BETA | 0 |

每进程只有一个fault marker及一个canonical_restart marker；每条流只有一个响应id和一个finish_reason。首次输入7659 tokens，下一问题7660 tokens。模型、原问题文本和输出预算不通过恢复流程修改。前缀重算是故障恢复的真实代价，不能认领为TTFT加速。

完整收据：tmp-nvcc/opsx15-20261023/recovery-{main,h2d}/report.json、requests.jsonl及serve.log。两次总进程时间约28.07s/26.73s。

这是两种指定软件故障后的同请求回退资格，不是实际硬件DMA故障容错。没有解决R21原213K原位置准确性，也没有将该C1机制推广为多lane故障隔离。原213K与256K旧质量套件未为这项恢复补丁重复运行。
