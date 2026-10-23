# R21安全历史复用与大批Cold续作（2026-10-23）

本轮目标是原213K不再逐题Root且精确，随后优化大批source-K/Cold。日常RTX4070TiSUPER/SM89/66SM/16GiB、200W；Qwen3.8 GSQ v3，NVFP4未开展。默认原位置OFF、MTP15+sync不变。未提交、推送或打tag，外部q4工作保留。

## 先算收益，再限定实现

旧质量臂六题都Root，实际computed prompt合计1274126tokens、各约159s；若相同历史只完整prefill一次，理论上可少算约106万tokens，六题wall上限缩短约83%。冷首轮本身仍要处理212K输入，本批历史复用不应被说成消除冷首轮159s。

source-K旧臂约17060页、Main K为1081344B/页，约17.18GiB；按20GiB/s仅H2D理想时间约0.86s。Cold旧臂约0.51GiB，纯DMA约0.026s。相对约187s canonical会话，不能先承诺10%整体收益。固定worker减少逐页std::async创建，Cold submit-before-retire隐藏部分pread/H2D等待；收益需独立量，不与R21重复重算收益混计。

## 来源及实际适配

| 来源 | 固定版本/文件 | 本机改动及边界 |
|---|---|---|
| KVMem | faf3c22582adf9d99ff072c543ea30ca03a8e2ca；kvmem_store.cpp::pick_prefill_ungrouped | sink/recent预填充与query检索分离；本机新增history-content锚点，不能归给donor准确性 |
| KVMem | 同版spill_io.cpp::StorageIoQueue | 固定worker/队列及immutable work合同，复用本机HostWorkerPool(2,2)；不是完整拷贝donor运行时 |
| ActKV | 182113cdbc6bbdd9ef300a84289075e761a1c075；tools/llama-kvmem-cli.cpp:510起 | 完整q0 GDN/StateImage后probe/restore/query replay；本机Main/MTP联合原生恢复不变 |
| Sync/Async | main f7e5c191、engine-main a4f32e26；ledger46/context.cpp | 复用submit与retire分离，不能把pending当已完成；本机Cold仍三页arena，未搬SM120a tile/ring默认 |
| DeepInfer | 774ce244；worker/cache_engine.py，Apache-2.0 | 128token落盘和rollback/state边界作参考，不替换本机64token INT8页/GDN状态，不搬SM100内核 |
| 本机既有archive loader | logical_kv_store.h::LoadSlot | 复用双slot最后H2D读者退休；新增Cold大批路径不做CUDA算术修改 |

## 安全复用实现

`NINFER_KVMEM_ORIGINAL_POSITIONS=1`下显式启用`NINFER_KVMEM_ORIGINAL_CAUSAL_PREFILL=1`；历史预填充忽略未来问题的lexical/action/retrieval/profile选页元数据，到完整q0捕获后才安装当前问题词法检索。完整StateImage、native原坐标K/V/scale、未来列撤销和联合恢复仍走既有验证路径。同请求发生恢复故障转canonical后，也同步停用本候选的lexical安装条件。

只用sink/recent的候选失败：原序列ACK正确后Topaz错答copper-lilac-28；后续TTFT虽0.664s也不得采用。fresh进程直接完整Topaz同错，排除仅为跨轮污染。两次均在首错停止、exit0。

增加独立`NINFER_KVMEM_ORIGINAL_CONTENT_ANCHORS=1`：只从历史内容旧页计算token页DF，至少两种DF<=4的token才评分；最多32页含下一续页，预填充锚点最多占工作预算1/4。锚点不使用当前问题、模型输出或期望答案，也不硬编码实体。完整锚点ID及policy2进入复用条件；变动则拒绝压缩复用，Root/合法早期checkpoint保留。GPU工作预算仍32K，不截断/清零GDN。

中间binary B已完成原六题6/6，冷首轮158.717s，后五题0.664/2.934/0.840/0.866/0.800s，session165.113s。此为单次质量臂，不与不同binary旧收据算正式百分比。收据tmp-nvcc/opsx-r21-safe-reuse-20261023/content-213k/。

## 大批source-K/Cold候选

`NINFER_KVMEM_COLD_COPY_PIPELINE=1`只在async设备可用、Cold>2页、非默认stream、未走小批layer stage时启用。限定canonical Main16普通prefill，含canonical回放准备；bulk资格独立于按层ticket。MTP/original/Graph内部沿用回退。保持3页pinned arena、read/H2D owner合计<=3且各<=2；先发新copy，再退休旧buffer，空出的槽发下一pread。最终drain再发布，异常在部分enqueue已发生时整stream排空后才释放pin。这里是大批读取/H2D流水，不是全部大批forward已逐层重叠。

`NINFER_KVMEM_FIXED_IO_WORKERS=1`让source-K和Cold已有prefetch共用各store固定2worker/2排队上限，FD/offset在owner线程捕获。新增ReadFuture析构、移动覆盖均join，避免packaged_task future不join造成buffer过早释放；store关闭FD前先join。最多四store共8worker；不新增pinned/GPU字节，默认0。退出摘要`KVMEM_IO_WORKERS`报告实际reads/pending。

新增CPU检查验证不依赖问题的选页、锚点改变拒绝资格、未来问题不改变锚点；Cold future移动/析构join及并发元数据增长通过。原生GPU>2 Cold整V字节检查、真实H2D提交后异常以及真实异步pread异常通过，退出pending/inflight均0。最终worker版收据bulk-worker-{gpu,fault,pread}.log。未用CPU mock代替这些GPU验证。

## 最终对照与采用范围

同binary三对与新大批组合质量已完成，最终收据见下方。原位置/新开关暂不翻默认。256K、媒体、多并发原位置、原位置跨进程archive、300W及新大批P4三对未在本轮预认。


## 复现入口

原题/权重冻结位置`/home/ran/ninfer-work/benchmarks/frozen-long-context/213k/manifest.json`；以下命令由仓库根执行，日常保持200W，使用20GiB/2GiB上限运行即可。现有ABI1/ABI2 archive是canonical模式，本原位置候选不加载或覆盖这些pack。

```bash
python3 -m tools.bench.r21_reuse_pairs --output <新的三对收据目录>
python3 -m tools.bench.local_long_quality --backend mtp --original --original-causal-prefill --original-content-anchors --draft-tokens 7 --case 213k --output <新的六题质量收据目录>
python3 -m tools.bench.local_long_quality --backend mtp --draft-tokens 15 --case 213k --canonical-query-reuse 1 --layer-ready 1 --transfer-mode async --host-pages 128 --fixed-io-workers 1 --cold-copy-pipeline 1 --output <新的大批组合收据目录>
```

三对的首候选完整跑六题，其余臂只用冻结前两条；subset只能产生DIAGNOSTIC_PASS，不能用于256K前置资格。报告逐请求API TTFT、真实request_done的computed/reused/path、cold逻辑读与进程物理IO；后采集的VmHWM在请求计时外，首两个已启动臂未采集的不补造。既有4份replay HostStateImage与8份Host cache StateImage容量也保持：单份153954304B；staging_peak只算source/Cold arena，不能当作总pinned内存。


## R21三对最终收据：通过

测量binary C：402729864B、mtime_ns1792731064591048173。副本与SHA256保留于本轮tmp目录r21-paired-serve{,.sha256}。第一候选完整6/6，另五臂2/2；共16/16请求精确且完整，三对六组对应请求全文及prompt/completion/total token数一致（cached_tokens按路径不同），六个Serve均exit0，结束pending/inflight/producer均0。后两个pair进程峰值RSS15.906–15.909GiB；首两个臂未采样，不补造。启动GPU空闲1784MiB，未增大32K/36K预算。

| 对 | Root首题TTFT s | 候选首题TTFT s | Root续问TTFT s | 候选续问TTFT s | 两请求wall之和缩短 |
|---|---:|---:|---:|---:|---:|
|1|166.179381|160.039450|160.403266|0.664627|50.7814%|
|2|161.695649|162.359104|160.411515|0.670235|49.3737%|
|3|163.004299|161.825791|160.649553|0.667445|49.7815%|

续问成对缩短中位99.5845%，两条request wall之和成对缩短中位49.7815%。这是相同冻结ACK→Topaz的三对；不能冒称整六题session成对缩短83%。冷首轮未消除，约160–166s；三对首题只有小幅波动，没有独立冷首轮性能结论。

最终C六题qual computed_prefill共212476（首212276、后34/39/40/42/45），相同六prompt逐题Root需1274126。五个续问复用212269/212296/212328/212361/212396tokens，真实private_turn_closure；history replay0/0轮、query replay162。指标相互核对，没有把Root重算藏进history_replay0。该六题质量臂session173.941s含client gap0.033s，单臂不与旧1024s作正式时延百分比。

OpenSpec2.6及旧16.4按六题质量+上述三对范围完成。新大批3.9仍须独立P4；R21结果不替它认领性能。原位置/新开关全局默认仍0，推荐范围仅已测GSQ v3/C1/MTP7/INT8/Graph/200W原213K显式profile。


D大批组合原六题6/6、session185.612s，固定worker四store实际source各17062、Cold各238、全部pending0；但Cold H2D ring为0命中，原因是恢复在canonical回放准备（不具备layer-ready资格）。因此D只作worker功能收据，不能作新大批H2D性能证明。E把plain-prefill bulk资格与layer-ready分开，canonical回放准备允许有界全batch ring，并同时启用既有失败目录/owner排空guard；仍禁止original/MTP/Graph内部消费。新增无layer-ready真实GPU>2页成功及部分H2D异常再验证，原六题命中门仍待E结果。


## 最终E：生产大批入口命中且六题通过

最终binary E402729896B、mtime_ns1792733297429052838；正常R21路径与测量C无算术或控制变化（bulk默认0），C保留独立副本。D新增原位置→canonical词法gate已用真实7.6K/MTP3/Graph软件故障通过：after_main_rewind一次、同请求200/ACK、下一请求200/BETA、exit0。E只再改变plain-prefill bulk资格/错误disposal，与R21正常路径无关，未重复其三对。

E新增无layer-ready的GPU bulk成功和真实H2D提交后失败均通过：全V字节、source/Cold固定worker路由、失败目录修复、地址释放后physical pool0。实际plain-prefill命中不能用原小批layer-ready成功替代；对应日志final-bulk-no-layer{,-failure}.log。

E真实cold-prefill原213K六题6/6，HTTP/terminal/usage完整，exit0；Main原16层351epochs、5664wait/release；大批Cold一批237页实际H2D，MTP仍全batch旧回退。四store固定2worker、队列2：Main/MTP source各17061读，Cold各237读，pending全0。staging峰值Main8650752B+MTP540672B=9191424B（8.765625MiB），未增加；所有inflight/pin/producer最终0。

原始IO：Cold逻辑544591872B；全进程物理read_bytes1481199616B、逻辑rchar20146452520B，不能把全部物理IO归给Cold。进程VmHWM17328128000B（约16.14GiB），没有突破20GiB上限。单臂session184.775227s（client gap0.029409s）只作功能及工作量记录；与D185.612s、旧P4不同binary/顺序/页缓存的单次数字不计算采用收益，页数一页差异原样保留。

OpenSpec47/58：本轮关闭2.6、3.7、3.8；旧change80/81（16.4关闭）。3.9新组合同binary P4三对仍NOT_RUN；纯DMA理论收益很小，未新增该长测矩阵，不据单臂翻默认。下一重点仍是冷首轮约160s的实际热点和新组合P4。默认原位置OFF/MTP15+sync、200W；NVFP4未做，未提交、推送或打tag。


## 新大批组合P4三对完成（2026-10-23续作）

同binary E402729896B/mtime1792733297429052838，200W/C1/GSQ v3/MTP15/INT8/32K总工作集/36K物理池/Host128、canonical-query-reuse1。完整冷prefill、原题六条、无archive，按async/sync、sync/async、async/sync交替。参考sync+固定worker0+bulk0，候选async+layer-ready1+source-event1+固定worker1+bulk1；环境和语料在manifest内。

| 对 | sync session s | 新async session s | 缩短 |
|---|---:|---:|---:|
|1|188.145535|186.548550|0.8488%|
|2|188.390076|187.662761|0.3862%|
|3|187.665959|188.927760|-0.6724%|

36/36精确、18组对应请求全文和usage一致、六Serve exit0/最终inflight、pin、producer全0。各候选Cold237–238页/1batch实际H2D、四store固定worker（source各17061–17062、Cold各237–238），Main351epochs/5664wait-release；staging仍9191424B。逻辑Cold与全进程物理read_bytes分开，OS页缓存未强制清理；一页计数差异保留。此为新组合实际命中三对，不借用旧3.5。

成对中位缩短0.3862%，第三对反而更慢，未达到全部正收益且中位>=10%的门槛。3.9测量完成，默认sync和新开关0不翻；不能据源码功能把bulk流水说成大幅加速。完整数值与Host峰值见docs/performance/2026-10-23-bulk-p4.json，原始收据tmp-nvcc/opsx-cold-prefill-goal-20261023/p4/。新增冷首轮2.8/2.9继续，不把本项负性能结论改写成160s已解决。
