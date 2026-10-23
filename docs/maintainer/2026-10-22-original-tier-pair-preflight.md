# OpenSpec S1 2.4：原位置 Main/MTP 联合预校验（2026-10-22）

本增量针对当前 Qwen3.8 GSQ v3、RTX4070 Ti SUPER、单卡200W的原位置
KVMem opt-in 恢复。严格保留 `NINFER_KVMEM_ORIGINAL_POSITIONS=0`
默认与已冻结213K/256K问题和数学ABI。原R21事实首问仍错，
本项修复不会把旧错误记录改成质量通过。

## 复制/适配来源

- KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca`：
  `ecosystem/kvmem/backends/patches/ninfer-kvmem.patch`
  的 `capture_memory_checkpoint`/`restore_memory_prefix`，
  明确前缀页所有权、首次写入修订、部分尾页和代际验证。
- Strata `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`（MIT）：
  `src/core/conversation_state.cpp` 的整状态和KV检验后恢复；
  `src/core/conversation_snapshot.cpp` 的布局与前缀匹配。
  适配的是验证顺序和逻辑，未复制其GPU专用kernel或引入第二套runtime。
- 本机R21既有 `tmp-nvcc/perf-ada-round21/native-prefix-probe.cu`
  提供完整原生 INT8 KV、Host/Cold、partial及单页首写K CPU fixture；
  新定向夹具沿用这些已写过的物理初始化方法，只执行本次**联合失效**场景。

## 本机新增代码

`src/targets/qwen3_6/impl/runtime/logical_kv_store.h`：
从既有 `rewind_original_prefix()` 提取
`validate_original_prefix()`，复用同一校验过程：先完成在飞归档传输退休，
再检查地址代际、checkpoint归属、frontier、每页唯一Device/Host/Cold
原生所有权、首次写入revision和partial mean形状。原rewind入口也继续
调用这个检查，避免两套验证策略漂移。

`src/targets/qwen3_6/impl/runtime/program_impl.h`：
在 q0 probe 恢复Main以前，先检查Main/MTP的logical frontier都等于
q0、完整StateImage可独占恢复、Main+MTP分别通过上述原生
`validate_original_prefix`。若草稿层有损坏或代际失效，**Main还未做任何
destructive truncate**，也尚未冻结本轮查询选页。两tier均验证通过之后，
才依次做原Main/MTP rewind、StateImage原位置恢复与后续原行为。
所有现存resume/Graph/数学代码与snapshot ABI不变。

Strata 的 `conversation_snapshot_validate` 还要求整会话的相同前缀、
每层KV计数与完整状态布局先同时匹配。本机针对带MTP的原位置q0在
上述原生owner检查之前又调用
`kvmem/strata_checkpoint_pairing.h::validate_strata_original_pair`，
核对checkpoint prefix与当前实际token ledger完全相同、
Main/MTP页修订数与q0覆盖一致、Host StateImage字节数和真实
`StateImageDevicePool::host_layout().image_bytes`一致，并禁止
尚未完成Host复制或未结算Fork的状态参加恢复。该适配使用C++静态
协议，不复制Strata模型kernel，也不分配额外GPU内存。

`ProgramImplCore::resolve_pending_raw`另在**原位置MTP实验模式**下接入
Strata启发的全batch mixed accept预校验，位置在原ReplaySSM GPU
fold提交之前。对同一个record width下的full continuation、partial
terminal、cancelled分别要求Main/MTP base一致及许可的状态commit列数；
仅使用全部row预验证通过后的结果。输出由调用方提供固定数组，不引入
热路径堆分配或默认MTP15模式的新开销。相关CPU fixture及许可边界见
`2026-10-22-strata-original-checkpoint-pairing-port.md`和
`third_party/strata/LICENSE`。

之后收敛了实际恢复事务，直接调用
`restore_strata_original_pair`完成一次Main/MTP/StateImage
联合预校验、两层rewind、窗口一致性和整态copy；代码不再维护
一份独立的手写联合恢复次序。新
`src/kvmem/original_recovery_guard.h`按Strata/KVMem原有资源生命周期
设计：preflight失败时保持两层原位置epoch不变；从Main第一次
rewind前`begin_mutation()`开始，发生后续CUDA/state/后缀
异常时，撤销q0请求专属恢复资格以及Main/MTP的原位置代际；
全程成功并完成必要stream同步后才`commit()`。guard
不销毁lane，不释放Native/Host/Cold页；外层原有
`device.synchronize()`与`clear_execution_failure_lanes`
负责请求资源退休。**这仍不是同一次请求的canonical自动重试。**

`tests/kvmem_original_pair_preflight_test.cu`：
复用旧原生Rig构造两个独立KVMem store，
分别建立完整 q0=337 token checkpoint，并写入到448 token的probe后继。
故意修改MTP checkpoint第0页revision，模拟
Program的验证顺序，要求报错且**Main和MTP仍保持448 token frontier**；
随后分别通过两tier合法校验，完整rewind回相同 q0=337，
退休并验证两个KV store的Device/Host/Cold所有权归零。
无模型、无伪造推理响应、不用已冻结原213K长题测试验收。

## 实际最小验收

```bash
cmake --build build --target ninfer_kvmem_original_pair_preflight_test ninfer-serve -j2
build/tests/ninfer_kvmem_original_pair_preflight_test
```

**构建两目标 exit0，原生GPU夹具 exit0：**

```json
{"case":"original_pair_preflight","verdict":"PASS","bad_draft_rejected_before_main_mutation":true,"matched_main_mtp_restored":true,"whole_branch_poison_rejected":true,"released":true}
```

本次guard接线后的最小新增验证：
`cmake --build build --target ninfer-serve ninfer_original_recovery_guard_test -j2`
exit0，
`build/tests/ninfer_original_recovery_guard_test` →
`ORIGINAL_RECOVERY_GUARD CPU PASS`。
独立CPU用例涵盖成功commit、preflight拒绝、Main已开始但MTP/StateImage失败、
取消幂等、双tier代际撤销、无MTP单层和generation复用拒绝；没有重新运行
原213K长事实集或既有模型性能对照。

进一步只用本次修改的Serve二进制做**一次真实原位置短请求**
（本地测试文件 `tmp-nvcc/opsx-original-recovery-20261023/smoke.py`；
收据 `report.json`、`serve.log`），单卡200W，
`NINFER_KVMEM_ORIGINAL_POSITIONS=1`、MTP3、CUDA Graph、
物理KV 8192 tokens、KVMem预算4096、单并发、INT8 KV。
原冻结测试语料的同类前缀缩短到7659输入，q0=7643，
新Serve实记`KVMEM_ORIGINAL checkpoint q0=7643 cache=4059`和
`KVMEM_ORIGINAL query_replay q0=7643 end=7659 cache=3995 history_tokens=0`；
Graph ready、`mtp accepted 3/3`，HTTP200、实际输出`ACK`、
usage完整、Serve exit0、无fatal/error。
首次试探性配置预算1024在启动时被既有`budget>=4096`校验拒绝，
没有发出任何模型请求；之后只在符合已有门槛的配置跑了上述一次模型验证。
这个结果证明联通了**真实Program q0恢复与MTP Graph解码**，
不包含跨batch部分终止、取消或原213K事实准确率；后者原R21仍失败，
且不因短样本成功而改写。

这只验证恢复前的**联合完整性门**，不是Graph捕获、MTP1 verify/AR
混合接受前缀的全部生命周期协议，也没有证明原213K首错已修复。
OpenSpec **2.4保持未完成**，2.1/2.5/2.6/2.7亦保持开放。
新失败不会发布不配对的原生状态，但如后续硬件I/O错误发生在
**验证后**的任一实际rewind，当前外层捕获仍需清理整个失败请求；
真正自动canonical重试和三对质量/性能门尚未实现，不能冒充完成。
