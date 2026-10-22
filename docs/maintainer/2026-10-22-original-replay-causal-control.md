# R21 原位置恢复：固定 q0 的因果对照

2026-10-22，Qwen3.8 GSQ / RTX 4070 Ti SUPER / 200W。

## Result

针对原213K首事实错答，**probe/restore导致有效KV或完整q0状态字节损坏的假设已被排除**。控制臂完全不执行该问题的probe和restore，仍得到相同首Main token和相同错答。原位置模式的质量根因尚未完全定位，仍保持默认0；这不是修复或性能验收。

| 边界 | 参考臂 | 控制臂 |
|---|---|---|
| 输入 | 原前两条请求，字节不变 | 相同 |
| q0 / packed frontier | 212275 / 32691 | 相同 |
| 问题前完整StateImage | 保存并经probe后恢复 | 保持原状态，不执行probe/restore |
| 所选原页 | 511页，含99/100/3200 | 强制同一组页 |
| Main/MTP有效K/V/scale | 恢复后逐页读取并记录128位摘要 | 计算问题前逐页读取，必须完全匹配摘要 |
| 完整153954304B状态 | 恢复后memcmp保存的q0状态 | 必须匹配同一摘要；页搬运后再次核对 |
| 首Main token | 66，`c` | 66，`c` |
| 首事实期望 / 实际 | silver-maple-41 / copper-lilac-28 | 相同错答 |
| server退出 | 0 | 0 |

完整prefix tokens采用逐token比较；状态和KV使用XXH3-128，不靠mtime或页号断言内容相同。KV读取覆盖每页每层每head的有效列，包含K/V及scale，不包含未写列或布局padding。原位置坐标、量化、选页、初始状态与计算二进制相同；两臂均为原MTP7配置，R29=0，保留旧R21触发条件。

## Implementation

优先复用KVMem `faf3c22582adf9d99ff072c543ea30ca03a8e2ca` 的 `capture_memory_checkpoint` / `restore_memory_prefix` 合同及本机已有的native单owner搬运、Main/MTP coupled selection。来源原文件为 `ecosystem/kvmem/backends/patches/ninfer-kvmem.patch`；ActKV对应实现位于 `ecosystem/melis-actkv/backends/patches/ninfer-kvmem.patch`。没有新增推理数学或GPU allocation。

- `logical_kv_store.h::visit_original_window`：只读访问实际工作集，沿用已有单页pinned staging；原K码/scale/位置不改变。
- `src/kvmem/original_replay_diagnostic.h`：独立的诊断记录，存精确输入prefix、q0、选页与摘要，不存生成输出；不与已有canonical `.nkv` ABI混用。
- `ProgramImplCore::advance_prefill`：显式控制臂在q0验证身份后，用已有coupled compaction建立同一工作集，核对字节，再直接计算未修改的问题；参考臂仍走原路径。

仅以下显式环境变量启用，默认没有诊断I/O：

```text
NINFER_KVMEM_ORIGINAL_DIAG_Q0=212275
NINFER_KVMEM_ORIGINAL_DIAG_SAVE=<fresh reference-selection.r21diag>
# 或控制臂：
NINFER_KVMEM_ORIGINAL_DIAG_BYPASS=<reference-selection.r21diag>
```

记录绑定当前输入与状态；不匹配在计算问题前拒绝。控制臂没有恢复前缀状态，因此“恢复后自比较”不是此对照的唯一依据。

## Evidence

`tmp-nvcc/opsx-original-causal/receipt.json` 状态 `CAUSAL_CONTROL_COMPLETE`。两臂各2个原请求，ACK均正确，首事实均错误；按原质量门停止后四题。每个响应HTTP200且流终态、usage完整；正常TERM退出。整段服务5m50.525s，20GiB RAM / 2GiB swap限额内，峰值20G/2G，非OOM。构建47.030s，2.5G/0swap。

实际二进制 SHA256：`9281b4e4b5c61269ab6b5c77472d265c2cd095e8f21df668ce332155a72a7110`。控制记录851240B：

```text
state XXH3-128 7562f1b2655a155073b6c5b7d2fc7edb
Main  XXH3-128 dd27defa0c99d51d69708c737581a03f
MTP   XXH3-128 f76ee9fc932fbd3e21397a4961eaabb8
```

原storage组件证据可继承：当前 `capture_original_prefix`、`rewind_original_prefix`、`commit_original_native_columns`、`require_original_native_owner` 与已发布HEAD逐函数完全相同，记录见 `inherited-storage-contract.json`。旧独立CPU字节/mean fixture已覆盖2,061,312B K/V/scale、24,576个mean分量、future撤销、partial追加、revision/lineage/generation拒绝、guard和释放；不重复无变化的GPU fixture。该fixture未覆盖高位置非零rotary数值，不能因此认领完整原位置质量通过。

## Remaining boundary

目前只能把该首错收窄到：**真实原时间线q0状态与原位置native工作集作为输入时，问题的Main计算已经给错**；不能继续把后续MTP接受、probe或restore字节损坏当作该首错的解释。原时间线conditioning、RoPE距离和canonical重算历史之间的具体因果尚未分离，OpenSpec 2.1、2.4–2.7及旧16.x验收仍未完成。

后续性能候选沿用已经正确的canonical状态，只在同一工作集和真实StateImage后代资格成立时续问；这是独立的选页政策优化，不把本次错答改写为通过。

## 继续定位：冻结选择的第一个输入分叉（无需重复GPU/长题）

使用新增的只读CPU工具
`tools/bench/selection_gap_audit.cpp`，
以已有`reference-selection.r21diag`经完整校验绑定原q0的exact prefix、
全StateImage摘要、Main/MTP native摘要及选页目录，再逐个扫描原token页号：

```bash
c++ -std=c++20 -O2 -Isrc -Ithird_party \
  tools/bench/selection_gap_audit.cpp \
  -o tmp-nvcc/opsx-original-causal/selection_gap_audit
tmp-nvcc/opsx-original-causal/selection_gap_audit \
  tmp-nvcc/opsx-original-causal/reference-selection.r21diag
```

完整结果留于`tmp-nvcc/opsx-original-causal/selection_gap_audit.json`：
Git中对应摘要为`docs/performance/2026-10-22-r21-selection-gap.json`。

| 固定量 | 数值 |
|---|---:|
| full q0 | 212275 tokens |
| native compact frontier | 32691 tokens |
| 原prefix页总数 | 3317 |
| 实际保留页 | 511 |
| 被省略页 | 2806 |
| **第一个未保留的源页** | **50（起始原token 3200）** |
| 不连续历史区段 | 177 |
| 绑定完整原q0状态、Main/MTP摘要 | 与已冻结no-probe记录完全匹配 |

这把原q0全历史计算与仅对所选历史重算之间的**最早可证输入集合分叉**
限定在原token 3200所在页：原时间线的StateImage继续处理全体后续token，
canonical恢复则通过`begin_kvmem_window_replay`和zero/重建处理选中历史。
结合同一q0的probe与no-probe臂均首token66，说明继续追查输入条件化、
native KV原位置与canonical rephase的语义差异有必要；**页缺口本身没有证明
token3200就是首个数值状态不相等位置，也不能证明错答只由GDN或某一层attention导致**。
它不是2.1的最终因果验收，2.1仍保持未勾选，未作任何新的GPU/模型请求、
没有修改原位置的生产数值或默认模式。
