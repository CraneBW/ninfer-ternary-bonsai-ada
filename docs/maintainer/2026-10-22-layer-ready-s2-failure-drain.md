# OpenSpec S2 3.3–3.4：Main16 layer-ready 生产、失败排空及地址注销（2026-10-22）

原 change：`/home/ran/ninfer-work/openspec/changes/absorb-donor-techniques-and-optimize-sm89`。这是 `2026-10-22-layer-ready-main16-implementation.md` **之后**的有效实施更新；之前文档保留初始 resident-only 验收的历史时点，不代表当前 source/Cold 已完成代码。实际工作在现有未提交树上追加，绝不重置其他 dirty 文件。

## 1. 已实装、严格限定的工作

- 显式 `NINFER_KVMEM_LAYER_READY=1` 才启用；单卡 SM89、PageMajor、Main16/INT8-G64、普通纯文本 prefill 且 non-original/non-ABI2 replay。生产流在 compute 流确认上一读者已释放后才开始。真实 full-attention `TextContext::attn_mix` 通过 `layer_ready[fidx]` 等待，提交该层最后一项 cached-KV 读之后记录 `released[fidx]`。GDN 不消费这些页。整轮 sync 默认不翻转。
- 生产端每一层按 `Cold K/V H2D → 首写 immutable source-K H2D/rebuild → resident K rephase → layer_ready` 顺序；Cold 至多两页（原三页 staging 中另一页留作同步 spill）、source-K 至多两页（现有双 pinned buffer），超过额度、特殊 geometry、MTP1 和 decode CUDA Graph replay 仍使用原有 sync 路由。Host bulk restore 和 CPU pread 完成/代际验证仍有真实等待；不宣称零等待。
- 源码改动在 `src/kvmem/{kvmem_bridge.h,kvmem_layer_ready.h,kvmem_rerope.cu,kvmem_source_k.h,kvmem_cold_store.h}` 与 `src/targets/qwen3_6/impl/runtime/{logical_kv_store.h,schedule.h,text_context.h,text_context_impl.h,text_prefill_impl.h,program_impl.h}`，定向 GPU 用例在 `tests/kvmem_layer_{source,cold}_test.cu`。`NINFER_KVMEM_LAYER_READY_TRACE=1` 输出 per-epoch source/Cold 页数与 final 资源记录。

## 2. 3.4 本轮新增失败事务合同

1. `ColdKVStore::prefetch` 新增仅供测试的 `NINFER_KVMEM_TEST_COLD_PREAD_FAILURE=1`：在真正 `std::async` 的工作线程向其 future 保存故障，消费者 `future.get()` 抛出，不更改正常 Cold I/O。future 成员早于 Host buffer 析构，异常后先 join 本次 file reader。
2. `enqueue_cold_layer` 在第一项真实 `cudaMemcpy2DAsync` 已提交后，可通过 `NINFER_KVMEM_TEST_PARTIAL_LAYER_H2D_FAILURE=1` 触发错误；事件 `ready` 尚未记录，不能假装数据完成。发生任何此类异常，`kvmem_compact` 的事务局部 guard 排空 producer 和 consumer、source-K DMA、Cold pinned staging，撤销 layer ticket；失败计数单调递增。
3. Compaction 的 block selection/stage-out 已可能改变当前 address membership。失败时只重建其**资源注销目录**（旧地址仍有效的引用 + stage-in 真实新页引用，去重；不发布错误 K/V），并标记 `failed_compaction`。这个地址 **只允许正常 deactivate/release，不允许下一次 activate/compact**，下一请求必须得到新的地址代际。失败后的 `store.release(handle)` 正常归还所有实际物理页；旧句柄不可能复用。
4. Store 销毁时主动完成 producer、consumer、source-K、Cold staging 的 stream draining 及 async demote retirement；在 trace 时输出 `KVMEM_LAYER_FINAL pending_pinned_pages=... inflight_pages=... producer_active=... failure_drains=...`。使用原有固定空间上限，没有新增后台 worker、GPU page 或第二套 snapshot 格式。

## 3. 新增、仅针对改动范围的实测收据

在 NVIDIA RTX 4070 Ti SUPER / 200W 单卡，实际 CMake `ninfer_kvmem_layer_cold_test` 与 `ninfer-serve` 构建成功。构建仍可见先前 `prefix_checkpoint_file.h` 的 `FILE*` attributes warning；非本轮失败。

| 测试 | 命令/收据 | 结果 |
|---|---|---|
| Cold read-ahead 的异步 I/O 失败 | `build/tests/ninfer_kvmem_layer_cold_test --pread-failure` | **PASS，0 failure**；明确捕获 injected pread 错误，失败 drain=1、layer pending=0、inflight=0；release 原地址成功、Device pool allocated=0 |
| 实际首个 DMA 提交后中断 | `build/tests/ninfer_kvmem_layer_cold_test --partial-h2d-failure` | **PASS，0 failure**；没有虚报 ready，失败 drain=1、Host/source pinned 退休；release 原地址成功、Device pool allocated=0 |
| 自有 Serve 进程 TERM | `python3 tmp-nvcc/opsx-layer-error-drain/term_serve.py` | **PASS**。只向该脚本启动的 pid 发送 SIGTERM；发送时已记录3个真实 `KVMEM_LAYER_READY` epoch，终态共22个。Qwen3.8 GSQ + MTP3 + CUDA Graph，HTTP200、server logged stopped、exit code0。两份 `KVMEM_LAYER_FINAL` 均为 `pending_pinned_pages=0 inflight_pages=0 producer_active=0` |

TERM 原始数据：`tmp-nvcc/opsx-layer-error-drain/{term-report.json,term-serve.log}`；脚本只向自己创建的 child 发送 TERM，未触及其他 Serve 进程。故障测试是 **确定性故障注入**，不称为发生过真实 NVMe 物理 EIO 或 GPU 硬件故障。

## 4. 合格范围及后续门槛

本轮证明当前 **Main16 普通文本 prefill** 的真正 layer-ready / last-reader 事件闭环、Source/Cold/Resident 实际 GPU 生产路径、17 列尾页字节保护与经允许的 MTP3+Graph **同步 fallback** 可共存；没有声称在每次 Graph replay 内重新挂载 host callback 或提前退休 MTP1 多次 AR reader。3.3–3.4 的这个受约束模式具备完成收据；不满足模式的配置不进新路径。

**3.5、3.6 尚未验收**：下一次仅使用固定 ABI1 raw 原213K六题做新二进制的 6/6 质量臂；若失败绝不进行或宣传 P4 性能。若通过，则在同 binary、200W、明确 `NINFER_KVMEM_LAYER_READY=0/1` 的 sync/async 三对中报告首/后续TTFT、wall、实际 Cold/热 cache 分类、staging 峰值与终态。P4 session wall 成对中位增益未到10%之前始终 sync 默认。S1 R21 original-position 首错仍不因本轮得到修复。

没有提交、推送、tag、下载已退役权重或开启双卡；保持200W日常功耗。
