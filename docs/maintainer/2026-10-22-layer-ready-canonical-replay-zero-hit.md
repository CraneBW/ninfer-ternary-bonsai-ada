# S2 / OpenSpec 3.5：原213K raw零layer-ready命中的调用链与最小安全增量（2026-10-22）

**结论：3.5继续未完成。** 已有原213K raw的6/6 exact、Serve exit0与资源归零收据真实有效，但现有`NINFER_KVMEM_LAYER_READY=1`逐层生产者在该工作负载上命中**零次**。根因是历史选页后的**canonical重算路径**未进入普通prefill compaction的逐层生产端，并非 CUDA layer-ready event丢失或异步线程未完成。本次只依据生产调用链及已有收据，不重新执行213K/GPU/Serve、不试图用软件开关伪造命中或收益。

## 原题实证与准确边界

- 原始收据：`tmp-nvcc/opsx-layer-p4-20261022/quality-raw213k-layer-ready-port18118/{manifest.json,results.json,213k-layer-ready-raw-port18118.log}`。200W单卡Qwen3.8 GSQ，实际物理INT8 KV **36,864 tokens**，KVMem budget **32,768**、reserve **512**，C1/MTP15，ABI1 raw原213K六题，全部HTTP200/exact/terminal，Serve exit0。两个`KVMEM_LAYER_FINAL`均为`pending_pinned_pages=0 inflight_pages=0 producer_active=0`；日志不存在`KVMEM_LAYER_READY epoch=`。
- 首问使用`KVMEM_BENCH_CHECKPOINT action=load prefix_tokens=212260 ... complete=1 phase=raw abi=1`，记录两次`KVMEM_CHECKPOINT_PAGES pipeline=1`、512页GPU/2805页Host与真正的恢复读写成本。这是**完整快照加载流水**，不是逐层K恢复与attention消费者的`LayerReadyFence`，不能把`pipeline=1`写成3.5命中。首问TTFT约44.1s，后五问约0.43–0.51s。
- 后五问逐一记录`KVMEM_CANONICAL_QUERY_REUSE`，q0为212275/212307/212339/212374/212409，canonical history resident分别为32243/32275/32307/32342/32377 tokens，明确不再进行首问的整段历史重算或普通prefill compaction。

## 第一个跳过逐层producer的确切代码条件

1. `src/targets/qwen3_6/impl/runtime/program_impl.h` 的首问KVMem query probe完成后，在约`13066–13161`进入canonical重选，调用`kvmem_offload(staged.hidden_replay_tokens, true, true)`，第三个实参标识 **canonical_replay=true**。
2. 同文件`12795–12797`将这一模式转为`text_kv_addresses->kvmem_compact(..., replay_from, !canonical_replay && !staged.vision)`；因此普通纯文本也强制传入`prefill_layer_consumer=false`。Store的`logical_kv_store.h:3177–3181`又要求`prefill_layer_consumer && layer_ready_requested() && !original_positions && !replay_from`及合法Main16/PageMajor/独立stream，故本路径压根不执行`3693–3740`的CUDA producer/event绑定。
3. 重选后，`program_impl.h:13097–13149`取出选中的历史token IDs，调用`begin_kvmem_window_replay(handle, reuse, stream)`并将`window_replay_cursor=reuse`。Store的`logical_kv_store.h:4410–4460`会撤销不再有效的历史K/V，恢复可复用前缀的immutable K；**必须用新canonical历史的因果状态重新forward**，不能直接把前一轮probe选中的原始K/V当成新历史读取。
4. `program_impl.h:12511–12554`的`window_replay_ids`专用分支逐chunk调用`schedule::prefill_text_chunk`，其`text_layer_ready`只查询`kvmem_layer_ready_for(address)`；该分支在历史forward期间**没有**发起新的`kvmem_offload`或逐层生产者。`logical_kv_store.h:2140–2146`只有先前**同地址、active**的compaction ticket才会返回非空；当前没有ticket。因此32K级别历史forward有真实计算，但0个`KVMEM_LAYER_READY epoch`。
5. 请求2–6的`reuse_canonical_query()`在`program_impl.h:12458–12509`通过后直接复用已完成的canonical view，后续仅处理很短的真实query suffix；这里也没有符合条件的普通prefill重压实。旧短4096-token、budget1024、Host仅1页的收据之所以能出现22个epoch，是它确实进行了带source/Cold的普通prefill compaction；不能将那条路径的短证据外推到原213K。

**不可采用的伪修复：** 将`!canonical_replay`直接改为`true`不等于使32K replay forward获得层准备。一方面这会在旧选窗/重相位阶段生成ticket，但`begin_kvmem_window_replay`随即使这些KV列的原先conditioning失效或清空；另一方面新32K历史计算来自同一forward stream，原本没有独立H2D/rephase producer可与其叠加。把旧V或其提前发布事件跨conditioning epoch使用会破坏事实召回和Main/MTP语义，违反原题精确质量门。原位置R21与deferred canonical同样不能因此绕过现有fallback。

## 本轮可独立安全落地的源码增量

`src/kvmem/kvmem_layer_ready.h`把`wait_before_layer`、`release_after_layer`严格绑定到`begin(compute)`登记的同一个consumer stream。之前两个方法只要求非空且不同于producer，允许等待事件排在流A而最终release事件排在无关流B：旧KV真实读取尚未完成时，retire/reuse有机会被错误宣告安全。修复阻止跨流等待和释放而不改变正常Main16同流生产、任何数值算法、选页或ABI。`tests/kvmem_layer_source_test.cu`在实际Main16 source-K producer上新增**另一条CUDA流**的负例：已发布ready但无资格的流不可wait，正确流wait后另一条流仍不可release，然后按原同流D2H+release走完。没有修改`program_impl.h`、`logical_kv_store.h`、`text_context_impl.h`等非独占调用方。

这是为后续引入真实canonical producer准备的**生命周期安全补丁**，不是已完成3.5或新的213K layer-ready性能结果。worker阶段仅做静态检查；主代理在GPU与Serve排他释放后，于同一200W单卡上完成最小受影响目标的构建和GPU fixture：

```text
$ cmake --build build --target ninfer_kvmem_layer_source_test -j2
[1/2] Building CUDA object tests/CMakeFiles/ninfer_kvmem_layer_source_test.dir/kvmem_layer_source_test.cu.o
[2/2] Linking CXX executable tests/ninfer_kvmem_layer_source_test
$ build/tests/ninfer_kvmem_layer_source_test
SOURCE_LAYER_READY failures=0 VERDICT: PASS
```

该PASS验证本次新增的不相关CUDA流wait/release必须拒绝，以及已有同一consumer流的source-K字节读取、释放、退役仍正确；仅证明新的定向生命周期修补。没有启动长题Serve、没有重新执行此前213K/3.3/3.4验证，也没有新的原213K pipeline有效命中或P4整会话收益数据。

## 真正完成3.5尚需哪项新增工作和测试

要在原冻结213K六问中让逐层流水产生有价值的命中，必须先证明**当前真实热阶段**存在准备旧K/V而仍会由新历史forward消费的GPU内核或I/O；若`reuse=0`且全部32K都由forward新生成，预先rephase旧K/V不会带来正确的生产者/消费者重叠，不能仅追求`KVMEM_LAYER_READY`行数。进一步需要在由主代理独占的`program_impl.h`的`query_probe`→`window_replay_ids`接续，及`logical_kv_store.h`的`begin_kvmem_window_replay`/地址生命周期明确一个**新canonical有效性epoch**及真正的逐层source/Cold准备操作，和`text_context_impl.h`同一consumer stream的first-read/release绑定。只有在选中且将被消费的有效源数据确实存在时，才能新增层事件；已有`replay_from`、Graph、MTP1、原位置R21及地址失效继续同步退回。此跨文件变更已向prime报告协调，worker未擅自修改。

先复用S2 3.1既有一次trace的上限证据：该原213K raw历史forward约23.167s，而522个source-K恢复的GPU H2D和rebuild事件分别合计约18.951ms、8.564ms（详见`2026-10-22-layer-ready-dependency-profile.md`）。即使理想隐藏上述约27.515ms，也仅占历史forward约**0.119%**；这只是对那一批source-K事件的Amdahl边界，**不是**整会话预测，更不能宣称满足3.5要求的**≥10%**。新方案若要过门，应识别并实际缩短其他秒级瓶颈，不能把cold/Host等待、完整checkpoint读取或新增replay本身混算为source-K层流水的理论收益。

完成资格仍按OpenSpec：先得到**实际命中新层事件**且同模型/物理36K/工作集32K的**原213K raw六题6/6 exact、终态完全健康**；再对**有效**的旧sync与候选async同binary/同介质200W三对，分别记录first/later TTFT、session wall、真正Cold与Source页数/Host staging与同步等待、MTP接受数、正确性，候选整体收益需**≥10%**才可勾选3.5，否则保留未完成与默认**MTP15 + sync**。不追加旧213K或3.3/3.4重复测试、未运行300W、未提交推送。
