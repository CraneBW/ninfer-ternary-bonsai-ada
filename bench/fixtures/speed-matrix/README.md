# 速度矩阵夹具与结果

`README.md` §1.6 那张表的数据就在这里。跑法：

```bash
export NINFER_MODEL=/path/to/bonsai.ninfer
python3 tools/bench/run_k_sweep.py 1                    # 先定每个任务的 K
python3 tools/bench/run_speed_matrix.py 短 1            # 短上下文 18 组
python3 tools/bench/run_speed_matrix.py 长 1            # 长上下文 18 组
ONLY_MODE=KVMem-int8 python3 tools/bench/run_speed_matrix.py 长 1   # 只补一个模式
```

结果追加到 `results.jsonl`（每行一组），K 扫描写到 `ksweep.json`。

## 夹具

| 文件 | 任务 | 说明 |
|---|---|---|
| `pelican.json` | 鹈鹕骑自行车 | 要求生成 SVG，长输出、中高可预测性 |
| `novel.json` | 小说撰写 | 中文短篇小说开头，长输出、低可预测性（接受率 25~38%） |
| `tool-weather.json` | 带工具查询（**纯 tool_call**） | 两个城市、两个工具调用，输出结构化、接受率 80~95% |
| `tool-weather-long.json` | 带工具查询（**查到结果之后写建议**） | 与上一条是**两种任务**，见下 |
| `long-pelican.json` / `long-novel.json` | 长上下文版 | 任务 prompt 接在 124k 正文之后 |
| `long-tool-call.json` | 长上下文版（纯 tool_call） | 同上 |
| `long-tool.json` | 长上下文版（写建议） | 同上 |

长版的正文来自 `bench/fixtures/` 之外的 124k 文档，用同一条 prompt 拼在不同任务前面；
三份的 token 数因此一致（123,867~124,133）。

## 两个坑（都值得单独说）

**① "工具查询"有两种任务，速度差一倍。** 纯 tool_call 是结构化输出，接受率 80~95%，
K=7 到 232 tok/s；"查到结果之后写建议"是自然语言，接受率 40% 上下，只有 112。
造夹具时为了让输出够长（短输出的 decode 测不稳）而把前者改成后者，会把任务性质一起
换掉 —— 那样量到的不是同一个任务。两个夹具都留在这里，就是为了让这个区别可见。

**② K 必须按任务取。** 见 `run_k_sweep.py` 的文件头：同一个 `--greedy` 下，
结构化输出 K=3→7 是 +26%，自然语言写作 K=3→7 是 −30%。固定一个 K 跑遍所有任务，
一部分任务必然跑在次优点上。

## 结果字段

`results.jsonl` 每行一组，字段：
`len`（短/长）、`mode`（resident-fp8 / resident-int8 / KVMem-int8）、`task`、`think`、
`k`、`kv`（KV 档）、`reps`、`decode`（tok/s）、`acc`（MTP 接受率 %）、`prefill`（tok/s）、
`prompt`、`gen`（生成 token 数）、`host_gib`（host KV 驻留量，KVMem 才有）、`wall`（秒）。

## 已知缺口

- 长上下文每组只跑 1 轮，`acc` 那列混着单次采样噪声（KV 量化会改变输出，接受率是真实
  差异；但单次值不是精确值）。
- **长上下文的 K 沿用了短上下文扫出来的值**，没有在 124k 上重扫 —— 每个点 90 秒，
  三档就是 4.5 分钟/任务，本轮没做。
- prefill 那一列在短上下文下被固定开销主导（69 token 只有 852 tok/s，340 token 有 1970），
  **不能用它衡量引擎的 prefill 能力**；要比就在同一长度上比。
