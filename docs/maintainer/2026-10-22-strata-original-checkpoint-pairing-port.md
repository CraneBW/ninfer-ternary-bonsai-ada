# S1 2.4：Strata 完整 State/KV 配对的可移植增量（2026-10-22）

**状态：CPU-only可复用原语和定向测试已实现；OpenSpec 2.4仍不勾选。** worker没有运行GPU/Serve或原213K；主代理已将本header的 q0 元数据校验接入真实Program原位置恢复入口，并在`resolve_pending_raw`的原位置MTP分支上接入固定容量的mixed accept全批次预校验。该增量未取得完整Graph/异常后canonical降级或冻结长题资格。

## 固定上游与许可

- 通过 git clone --filter=blob:none 将 https://github.com/Niko1221/Strata.git 克隆到 /home/ran/ninfer-work/ecosystem/strata，独立检出 **9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4**，没有使用浮动HEAD。
- 上游 LICENSE 为 MIT，Copyright (c) 2026 Niko1221 and the Strata contributors，完整许可原样保存至本仓 third_party/strata/LICENSE（字节级cmp通过），新增头文件明确记录来源。本轮按NInfer的SM89、StateImage、PageMajor及KV owner接口重写局部协议，没有复制整个Strata runtime或CUDA文件。
- src/core/conversation_state.cpp:74–94 要求留存checkpoint的tokens/image keys是真live前缀；:143–152检查GDN/indexer tail、dead、block_pos运行态字节完整；:301–326把running state、每层KV及draft写入临时SavedConversation，所有成功后才发布；:382–410先统一validate几何、层范围、运行态与所有main/draft KV，然后才restore。
- src/core/conversation_snapshot.cpp:183–201 校验每层五段KV payload布局/长度/真实目标，:266–284将完整合法captured image按合法prefix裁剪后restore。src/prefill/prefill.cpp是GPU prefill staging与stage callback入口，在固定SHA中不是完整conversation snapshot/接受事务的唯一owner。

## 实际可复用新代码

新增 src/kvmem/strata_checkpoint_pairing.h 和 tests/strata_checkpoint_pairing_test.cpp，均无GPU依赖：

1. validate_strata_original_pair：核对q0精确token前缀、Main/MTP原生OriginalPrefixCheckpoint的相同logical_frontier、非零lineage_epoch、page64数量对应的prefix_revisions，以及完整Host StateImage字节数、已完成copy、真正prefix identity匹配、独占state状态。物理first-write owner/Host/Cold正确性仍必须由原KVAddressSpaceStore::validate_original_prefix检查。
2. restore_strata_original_pair：先依次完成Main、MTP、StateImage全部真实preflight，任何一处失败都**不发生第一次mutation**。全部通过才运行Main rewind、MTP rewind（结果physical window必须相等）、StateImage restore。第一处mutation后任何DMA/restore/窗口不匹配异常都强制调用调用方提供的 **noexcept poison_branch**，使整个原位置branch只能排空/释放，不能消费部分恢复内容。该CPU helper不假称CUDA copy原子性。
3. plan_strata_mixed_accept：先检查全batch同一个GDN record_width及每row Main/MTP base、produced/accepted、partial terminal、cancelled、prefix_execution_split、duplicate lane和frontier溢出，然后给出各row最终Main/MTP/ledger frontiers、应fold的GDN列数、partial-terminal hidden选择和是否retire。一个row可满接受，另一个row部分终止，第三个取消，不允许将拒绝的draft计入StateImage。**使用调用方std::span固定数组作输出，双遍校验先于任何输出写入，零热路径堆分配**。生产端仍拥有ReplayFold和CUDA执行权。

## 真实NInfer生产接口的待协调差分

主代理已在 src/targets/qwen3_6/impl/runtime/logical_kv_store.h 中抽出validate_original_prefix，并在 program_impl.h 完成 Main/MTP 双tier preflight，现已接入本header的 q0 metadata pairing 校验。本任务严格没有改这两文件；后续完整闭环适配点为：

1. q0 capture（program_impl.h约12371–12405）：两份OriginalPrefixCheckpoint、冻结token前缀和已同步完成的Host StateImage归为一个身份。若prefix、StateImage完整性或owner不符，不能发布original_query_checkpoint_ready。
2. q0 probe→rewind（约12975–13025）：在当前 Main/MTP 物理preflight **之前**运行新增pairing元数据门和StateImage目标/host identity检查；两者全过再调用Main rewind、MTP rewind及Host StateImage恢复。可直接通过 restore_strata_original_pair 包裹三组真实preflight/restore callback，并在正确提交后发布执行frontier。不可因为某单tier的验证成功就开始truncate另一tier。
3. 生产入口随后直接复用`restore_strata_original_pair`执行整组preflight→Main/MTP restore→StateImage copy；`OriginalRecoveryGuard`从首次Main rewind到成功同步使用RAII管理双tier与request-local q0 proof，异常先撤销原位置epoch和专属q0凭据，外围`device.synchronize`/request cleanup排空资源。CPU guard定向测试PASS、生产Serve重构后构建PASS。真实GPU I/O/Graph故障和同请求canonical自动重试尚未实现，不将其当作完成2.5。
4. MTP混合接受现已由主代理在`program_impl.h::resolve_pending_raw`的原位置MTP分支，于原CUDA GDN ReplaySSM fold之前实际调用`plan_strata_mixed_accept`。固定`std::array`存储和`std::span`输出无堆分配；在所有row验证后才使用计划中的状态commit列数、partial terminal hidden selector与资源终止标志。默认`NINFER_KVMEM_ORIGINAL_POSITIONS=0`不会进入该额外分支；原既有CUDA数学与提交顺序不变。

## 证据与未完成门槛

在仓库根目录单独执行（没有改tests/CMakeLists.txt，也不占共享编译目录）：

    g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc tests/strata_checkpoint_pairing_test.cpp -o /tmp/ninfer-strata-checkpoint-pairing-test
    /tmp/ninfer-strata-checkpoint-pairing-test

结果：**STRATA_CHECKPOINT_PAIRING CPU PASS**（g++ -std=c++20 -O2 -Wall -Wextra -Werror）；主代理又将其注册为正式CTest目标：`cmake --build build --target ninfer_strata_checkpoint_pairing_test -j2`及`build/tests/ninfer_strata_checkpoint_pairing_test`均exit0，并完成最新`ninfer-serve`生产构建exit0。附加固定Strata MIT LICENSE字节比较PASS、git diff --check PASS。测试用真实原KV checkpoint相同成员形状，覆盖跨页1列tail、完整StateImage字段占位、MTP preflight stale拒绝且Main零变更、state容量/prefix身份/fork拒绝、Main/MTP恢复frontier分歧与中途异常whole-branch poison，另用3行不同accepted数验证full continuation、partial terminal、cancelled，失败row必须使整批无改动且调用方输出数组保持不变。

这验证的是新CPU原语和生产接线能构建，**不是**NInfer原位置完整恢复、R21质量或GPU故障退休。主代理的现役原生 bad-draft preflight 测试已独立通过，原位置mixed accept有全批次新校验，异常分支也增加Main/MTP双tier epoch poison。2.4完整门仍缺故障后真实GPU drain、Graph/混合接受前缀GPU语义和原长题质量；2.5还缺实际canonical重试。**2.4维持未完成、默认MTP15+sync不变**。Strata本地克隆是离线固定来源，不作为发布库；仅MIT许可和适配代码进入主仓。本增量无新的提交/推送，外部q4脏文件未碰。
