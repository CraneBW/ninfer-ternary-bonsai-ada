# R21首次实测状态分歧与原位置生命周期

## 首次分歧诊断

使用冻结213K六题中首个已知错答的原始请求（第2题，期望`silver-maple-41`，
q0=212275），没有改写输入或缩短请求。分别在canonical与original模式下执行到首次真实压缩，
捕获完整153954304B StateImage、选页目录和位置元数据，下一块完成后定向终止。
这是200W下两个约32K前缀的数值诊断，HTTP500为诊断的刻意终止，**不是长题质量臂**。

| 捕获边界 | 两模式完整StateImage | 选页ID | cursor / KV有效长度 |
|---|---|---|---|
| 首次压缩前 | 字节相同 | 相同 | 32768 / 32768 |
| 压缩后立即 | 字节相同 | 相同 | 32768 / 32256 |
| 下一512token块结束 | 出现分歧 | 相同 | 33280 / 32768 |

offset从0变为−512，原位置标志分别为0/1；其他算子与R21原始bundle环境相同，
包括R29=0、共享量化=0、MTP7、INT8、chunk512、32K预算及sync。
最早出现差异的持久层是GDN ordinal3（零起算模型layer4，紧随第一full-attention layer3）。
GDN ordinal0–2的conv与FP32 recurrent均相同。

- ordinal3 conv：26939个元素不同，首元素canonical=0.042236328125，original=0.0390625。
- ordinal3 recurrent：786430个FP32元素不同，首元素canonical=0.00039882882265374064，
  original=0.0004126066924072802。

这是首次**实测分歧块**`[32768,33280)`和最早持久层，不声称定位了块内确切首token、
首CUDA instruction或某一个RoPE算子的独立因果效应。主变量是压缩后原位置/重相位的执行语义；
压缩并未立即篡改完整StateImage，差异在随后真实续算中出现。
结合旧probe/no-probe两个臂都给出相同错误首token、Main/MTP/StateImage恢复摘要相同，
首次状态分叉已明确早于q0/probe/restore/Main bonus；继续只修q0字节恢复无法消除这个早期分叉。
它不证明单一层就是最终错答的唯一原因，也不把旧R21改为通过。

完整原始文件在`tmp-nvcc/opsx15-20261023/first-compaction-r21/`；
`report.json`包含每一持久层/区域的差异数量、首位置和值。
首次脚本曾误选开场ACK并在检查通用错误日志时停止，保留于`first-compaction/`，
不作为本结论的成对证据。修正后两臂都使用首个已知错答的相同冻结输入、各正常退出0。

复现（新目录，完整镜像不会覆盖）：

```bash
python3 -m tools.bench.first_compaction_diagnostic \
  --output tmp-nvcc/local-r21-first-compaction
```

## 原位置MTP/Graph接受与生命周期

现有原位置模式只支持C1；真实GSQ、MTP3、Graph、200W短长输入沿生产Strata配对路径执行：

| 实际提交 | produced / accepted / width | state_columns | 结果 |
|---|---|---:|---|
| 完整接受 | 4 / 4 / 4 | 4 | 正常继续 |
| 终止前缀 | 4 / 1 / 4 | 1 | repair=1，ACK精确 |
| 终止前缀 | 4 / 2 / 4 | 2 | repair=1，DELTA精确 |
| 客户端读取首内容后关闭 | transport HTTP499 | — | 同进程新请求DELTA、exit0 |

终态active requests、KVMem inflight pages/bytes均0。实际取消发生在一个正常提交之后，
没有观察到本轮`cancelled[row]=1`的GPU提交；该规划分支继承已有CPU混合接受/取消fixture，
不外推GPU多lane原位置混合batch（当前不支持）。此前真实GPU双tier故障/代际退休夹具的代码未变，
直接继承其三场景证据，不重复运行。

收据`tmp-nvcc/opsx15-20261023/original-lifecycle/{report.json,requests.console.log,metrics.txt}`。
这些关闭配对/接受前缀生命周期的已支持范围，原213K准确性、三对TTFT及旧change16.x仍未完成；
原位置默认OFF。普通canonical AR的213K/256K通过属于另一路径。
