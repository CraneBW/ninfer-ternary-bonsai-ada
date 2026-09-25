# 三元 Bonsai → NInfer(.ninfer) 转换手册（zatfung 线）

> 目标：把 `C:\WorkModels\Qwen3.8-27B\` 下的两个三元 GGUF 转成 zatfung 能加载的
> `.ninfer` 产物，**输出回 C 盘原始目录**（SSD 快盘；G 盘是机械盘，测试会拖慢）。
>
> 状态：✅ **已完成**。两个产物均已生成并通过容器验证（见 §7）。

---

## 1. 资产与工具

| 项 | 位置 |
|---|---|
| 三元模型（PQ2_0） | `C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf`（6.8 GB，type 142） |
| 三元模型（PTQ1_0） | `C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.6 GB，type 143） |
| vision mmproj | `C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-mmproj-BF16.gguf`（889 MB） |
| 转换器（原始） | `G:\Agents\ninfer-works\ninfer-ternary-qwen3.8-27b-3090--may-has-bug\pack\pack.py` |
| 映射规则 | 同目录 `pack/MAPPING.json`（每条规则都有实测证据） |
| 转换器（zatfung 版） | `C:\WorkModels\Qwen3.8-27B\_pack\pack_zatfung.py`（路径环境变量化） |
| 模板 artifact | `C:\WorkModels\Qwen3.8-27B\_pack_template\qwen3_8_27b_v2.ninfer`（19.03 GiB） |

`pack.py` 的三种模式：

```bash
python pack_zatfung.py check          # 几何 + 解码 + 字节往返证明（不写文件）
python pack_zatfung.py layer3 <out>   # 最小产物：frontend + sign 表 + 一个 full-attention 层
python pack_zatfung.py build  <out>   # 完整文本模型
```

---

## 2. 三个非显然的坑

### 坑 1（最重要）：容器版本 v3 vs v2

| | zatfung / ninfer-3090 | HF 官方（当前） |
|---|---|---|
| magic | `NINFER\x00\x02` | `NINFER\x00\x03` |
| 头部 | `<8sQ>` = 8B magic + 8B JSON 长度 | 8B magic + 8B 长度 + **16B 额外字段** |
| JSON 偏移 | **16** | **32** |
| JSON schema | `{"identity":..., "objects":[...]}` | `{"components":{...}}` |

**直接拿 HF 当前的 artifact 当模板会失败**（`load_template()` 在 offset 16 读到的是二进制字段，
`json.raw_decode` 报 `Expecting value: line 1 column 1`）。

**解法**：HF 仓库里 v2 仍然存在于历史 revision。

| revision | 日期 | 说明 |
|---|---|---|
| `1cbd84e7221e` | 2026-09-15 | Update the built-in Qwen chat template（当前 main） |
| `51630a0c0f4f` | 2026-09-15 | **Publish v3 artifact** ← v3 起点，不要用 |
| **`dc370fb6295a`** | **2026-09-06** | **Update artifact with DFlash2 companion weights** ← **用这个（v2）** |
| `18dfc887423f` | 2026-08-19 | docs(eval) |
| `3526913004b1` | 2026-08-14 | Add Qwen3.8-27B NInfer artifact |
| `6925b5541b49` | 2026-08-06 | initial commit |

下载 v2 模板：

```bash
curl -L -C - -o qwen3_8_27b_v2.ninfer \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/dc370fb6295a/qwen3_8_27b.ninfer"
```

验证拿对了版本（`magic` 必须是 `...02`，JSON 必须在 offset 16）：

```python
raw = open(p, "rb").read(48)
assert raw[:8] == b"NINFER\x00\x02", "拿成 v3 了"
assert raw.find(b"{") == 16
```

### 坑 2：模板必须**全量** 19.03 GiB —— 无法只取需要的部分

模板只用来**借用** 353 个对象（vision/mtp/dflash2/frontend/draft_head），
`text/*` 的 16.25 GiB 会被完全替换、根本不需要。但这 16.25 GiB **跳不过去**，
因为借用对象横跨文件首尾：

| 组 | 数量 | 载荷窗口 |
|---|---|---|
| frontend | 6 | `0 .. 12,837,177`（文件开头） |
| draft_head | — | `17,106,328,064 .. 17,463,368,192` |
| mtp | 12 | `17,463,368,192 .. 17,914,635,776` |
| vision | 333 | `17,914,635,776 .. 18,210,355,200` |
| dflash2 | 66 | `18,210,355,200 .. 20,437,148,160`（文件末尾） |

即借用区从 offset 0 一直延伸到 20,437,148,160 = **19.034 GiB = 全文件**。
（理论上可以造稀疏文件 + 分段下载省 84% 带宽，但 `Artifact.open()` 会 mmap 整文件，
为稳妥直接全量下载；下载实测 13.4 MB/s，约 25 分钟。）

### 坑 3：zatfung 的 `tools/artifact` 需要 PyTorch

`zatfung/tools/artifact/layouts.py` 第 19 行即 `import torch`（全文 138 处），
而 `pack.py` 本身是纯 numpy 的。缺 torch 时 `from tools.artifact import ...` 直接失败：

```
ModuleNotFoundError: No module named 'torch'
```

装 CPU 版即可（不需要 CUDA 版）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. 已验证的兼容性（实测数据）

**两个 GGUF 与 pack.py 的预期完全一致** —— 这不是推测：

| 指标 | 实测值 | pack.py 期望 |
|---|---|---|
| tensors | 851 / 851 | — |
| header_end | **11,120,982** | 注释里点名的就是这个数 |
| 三元类型 | 142(PQ2_0) / 143(PTQ1_0)，各 402 个 | `T_PQ2_0=142, T_PTQ1_0=143` |
| 层数 | 64（0..63） | — |
| full-attention 层 | 16 | MAPPING.json 的 `full_attention` 列表长度 |
| `blk.64.*` | 不存在 | 注释："Bonsai's GGUF has NO MTP head" |

**模板对象清单**（v2，实测解析）：1190 个 = text 773 / vision 333 / dflash2 66 / mtp 12 / frontend 6
→ 借用 353，生产 837。

**API**：`pack.py` 需要的 7 个符号 zatfung 全有
（`Artifact` `ArtifactIdentity` `ArtifactWriter` `ResourceSpec` `TensorSpec`
`encode_direct` `row_split_geometry`）。

---

## 4. 执行步骤

```bash
# 0) 环境
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu   # torch 见坑 3

# 1) 模板（见坑 1，必须用 v2 revision）
cd C:\WorkModels\Qwen3.8-27B\_pack_template
curl -L -C - -o qwen3_8_27b_v2.ninfer \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/dc370fb6295a/qwen3_8_27b.ninfer"

# 2) 先跑 check（不写文件，验证几何/解码/字节往返）
cd C:\WorkModels\Qwen3.8-27B\_pack
set ZATFUNG_GGUF=C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf
python pack_zatfung.py check

# 3) 完整打包（输出到 C 盘原始目录）
set ZATFUNG_GGUF=C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf
python pack_zatfung.py build C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.ninfer

set ZATFUNG_GGUF=C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf
python pack_zatfung.py build C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PTQ1_0.ninfer
```

脚本读这些环境变量（默认值已指向 zatfung 与 C 盘）：

| 变量 | 默认 |
|---|---|
| `ZATFUNG_ROOT` | `G:/Agents/ninfer-works/zatfung`（提供 `tools.artifact`） |
| `ZATFUNG_TEMPLATE` | `C:\WorkModels\Qwen3.8-27B\_pack_template\qwen3_8_27b_v2.ninfer` |
| `ZATFUNG_GGUF` | `C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf` |

> `pack.py` 原版硬编码 `H:/ninfer-ternary/...`、`H:/ninfer-3090/models/...`、
> `H:/Ternary-Bonsai-.../` 三个路径。`prepare_pack.py` 已把它们环境变量化，
> **原仓库文件未改动**。

---

## 5. 空间与时间预算（C 盘实测 51 GB 可用）

| 项 | 大小 |
|---|---|
| 模板（用完可删） | 19.03 GiB |
| 产物 ×2（PQ2_0 + PTQ1_0） | ≈ 7.45 GiB 各 → ≈ 14.9 GiB |
| **峰值占用** | **≈ 34 GiB**（转换完删模板后 ≈ 15 GiB） |

---

## 6. 产物使用

```bash
G:\Agents\ninfer-works\zatfung\_build_86\ninfer.exe ^
  C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PQ2_0.ninfer ^
  --prompt "你好" --max-context 4096 --kv-dtype int8
```

> 本机 RTX 3060 Ti 只有 8 GB 显存，而三元 27B 权重约 7.12 GiB。
> 首次验证请用 `--max-context 4096`，性能基准另找 24 GB 卡。

---

## 7. 实际结果（2026-09-25 完成）

### 产物

| 文件 | 大小 | text 生产 | 借用 | objects |
|---|---:|---:|---:|---:|
| `Ternary-Bonsai-2-27B-PQ2_0.ninfer` | **9.810 GiB** (10,533,732,876 B) | 6.696 GiB | 3.114 GiB | 1192 |
| `Ternary-Bonsai-2-27B-PTQ1_0.ninfer` | **8.637 GiB** (9,274,212,876 B) | 5.523 GiB | 3.114 GiB | 1192 |

两者都在 `C:\WorkModels\Qwen3.8-27B\`（按 SSD 要求，未放 G 盘）。
`objects = 1192` = 模板 1190 + `text/hadamard_signs` + `text/hadamard_widths`（后两者由 packer 追加）。
借用构成：`{frontend: 6, text: 2, mtp: 12, vision: 333, dflash2: 66}` ——
其中 `text: 2` 正是 `draft_head` 与 `draft_head_token_ids`。

### 验证

- 两个模型 `check` 模式：**RESULT: OK**（CHECK 1/2/3 全过）
- 产物用 **zatfung 自己的 `Artifact.open()`** 打开：1192 个对象可见
- 抽样 payload 与 check 阶段的计算值逐一吻合：
  `query_key` 9,748,480 B / `token_embedding` 337,715,200 B /
  `hadamard_signs` 114,688 B / `vision/patch_embedding` 1,382,400 B

### 踩到并修掉的一个 pack.py bug

`check` 模式的 offset 自检硬编码了 PQ2_0 的组布局（`reshape(-1, 34)` + scale 取 `[:, 0:2]`），
PTQ1_0 是 **28 B/组且 scale 在组尾 `[26:28]`** → `cannot reshape array of size 13762560 into shape (34)`。
修复：按 `fmt` 分派组大小与 scale 列（已改在 `pack_zatfung.py`）。
**仅是自检逻辑缺陷，不影响 payload 产出** —— 修完 PTQ1_0 的 CHECK 2 全项 PLAUSIBLE。

### 一个省时间的技巧

`check` **不需要完整模板**：`mode_check` 只经 `Packer.__init__` → `load_template()`，
后者仅读模板前 8 MB 的 JSON。所以 19 GB 还在下载时就能用 8 MB 头部先跑 check。

### 磁盘与模板

- C 盘：234 G 总 / **13 G 可用**（95%）—— 模板 19 GB + 两产物 18.4 GB 占了大部分
- 模板 `qwen3_8_27b_v2.ninfer`（19.03 GiB）**转换后不再需要**，
  删掉可回收 19 GB：`del C:\WorkModels\Qwen3.8-27B\_pack_template\qwen3_8_27b_v2.ninfer`
  若打算转换别的模型则保留。

### 未做：端到端试跑

产物已被容器层面验证，但**尚未在 zatfung 里实际加载推理**。
注意显存：产物 9.81 GiB > 本机 8 GB 卡，与 laamaafung 能跑（llama.cpp mmap + KVMem 卸载）
不是一回事 —— zatfung 目前没有 KVMem，也没有权重 offload。
第一次试跑建议：

```bash
G:\Agents\ninfer-works\zatfung\_build_86\ninfer.exe ^
  C:\WorkModels\Qwen3.8-27B\Ternary-Bonsai-2-27B-PTQ1_0.ninfer ^
  --prompt "你好" --max-context 4096 --kv-dtype int8
```

（PTQ1_0 更小，8.64 GiB，先用它试。）

---

## 8. 模板瘦身备份（19.03 GiB → 3.114 GiB，省 83.6%）

原模板已删除，替换为只含"有用部位"的备份。

### 为什么可以瘦身

`pack.py` 对模板只用到两样东西：**头部**（16 B 前缀 + 185 KB JSON 对象目录，
决定产物要写哪些对象）与 **419 个借用对象的载荷**。其余 771 个 `text/*` 对象的载荷
由 packer 自己 `produce()` 生成，**从不读取**。

> 顺带纠正一个数字：pack.py 注释说借用 353 个（vision 333 + mtp 12 + frontend 6 + draft_head 2），
> 但那是 dflash2 加入之前的注释。按 pack.py 的实际判据（`非 text/*` 即借用），
> **真实数量是 419** = 353 + dflash2 66。

而在文件布局上，这 419 个对象**恰好集中在两端**：

| 文件位置（字节） | 内容 |
|---|---|
| `0 .. 13,025,792` | header + frontend(6) |
| `13,025,792 .. 17,106,516,480` | 771 个 `text/*`（**无用**，15.920 GiB） |
| `17,106,516,480 .. 20,437,336,576` | draft_head(2) + mtp(12) + vision(333) + dflash2(66) |

### 备份构成（`_pack_template\`）

| 文件 | 大小 | 用途 |
|---|---:|---|
| `qwen3_8_27b_v2_slim_extract.bin` | 3,343,845,888 B（3.114 GiB） | 两段原始字节顺序拼接 |
| `qwen3_8_27b_v2_slim_meta.json` | 835 B | 区段映射（源 offset / 长度 / 元数据） |
| `restore_v2_template.py` | 1.5 KB | 还原脚本 |

### 还原

```bash
cd C:\WorkModels\Qwen3.8-27B\_pack_template
python restore_v2_template.py                      # -> qwen3_8_27b_v2.ninfer
python restore_v2_template.py D:\somewhere\v2.ninfer   # 或指定路径
```

还原出的文件 `20,437,336,576 B`，中间被跳过的 15.92 GiB 为零填充。
**功能上与原模板完全等价** —— `Artifact.open()` 的 `_validate_ranges` 只校验
offset 落在文件范围内、不校验内容，而 pack.py 从不读那 771 个对象。

### 验证与恢复来源

- 覆盖性：脚本校验 **419/419 个借用对象**的文件范围都被备份区段覆盖（`0 missing`）
- 还原脚本实测产出了正确大小的文件（20,437,336,576 B）
- 若备份也丢失，可从源头重取：
  `neroued/Qwen3.8-27B-NInfer` @ revision **`dc370fb6295a`** 的 `qwen3_8_27b.ninfer`（19.034 GiB）

### 试过的两条"更正规"的路，都走不通

1. **精简 artifact（只含 419 个对象的合法容器）也没成**：`container._parse_object` 有两条硬校验 ——
   tensor 的 `bytes` 必须 **positive**，且必须**精确等于** `encoded_size(layout, format, shape)`。
   所以无法把非借用对象声明成 0 或 1 字节的"占位"来绕过。
   （要真做，得让 pack.py 从外部清单读 specs、从精简模板借 payload —— 改动更大。）
2. **NTFS 稀疏文件**：实测可行但 Git Bash 下占用读数不可靠（10 MB 文件写 4 KB 后 `du` 报 2 MB），
   且拷到别的介质会展开，故弃用。

---

## 9. Range 下载器（仓库只存源码，不存 3 GB 二进制）

上面的 extract 是 3.114 GiB 的本地备份。更好的做法是**连这份备份也不进仓库** ——
改为让用户用 HTTP Range 直接从 HF 抓那两段。

### 交付

`zatfung/tools/template-fetch/`（Go，≈476 行 + README）

```bash
go build -o template-fetch.exe main.go
./template-fetch.exe            # 默认：产出可直接使用的完整模板
./template-fetch.exe -slim      # 只产出 3.11 GiB extract
./template-fetch.exe -j 8       # 并发块数
./template-fetch.exe -no-resume # 丢弃进度重下
```

仓库里因此**只有源码 + 内嵌区段表**，没有二进制。

### 关键实现点

| 点 | 做法 |
|---|---|
| Range 正确性 | 显式要求 `206 Partial Content`；若服务端返回 `200`（会灌进整个 19 GiB）直接报错，不默默写坏数据 |
| 断点续传 | 进度记在 `<out>.tplfetch.json`，重跑跳过已完成块 |
| 分块 | 默认 32 MB/块、4 并发；**块不跨区段**，故 URL 映射是单段线性的 |
| 校验 | 全部就绪后对整份 extract 算 SHA-256，与内置基准比对 |
| revision 防漂移 | HEAD 比对 `Content-Length`；HF 的 `main` 已换成 v3，URL 钉死在 `dc370fb6295a` |

### 验证

- **HTTP Range 中间位置可精确取数**：从 `17,106,516,480` 取 1 MB，
  与 extract 对应区段比对 —— `sha256 6370c3d4077f176f7e57818ee0d50fe36a2c2053` **完全一致**
- 内置基准：`97437007c19310a5f3204e594a6fb1acc190e2ffade9acc568b8c9310ea2a72c`（3,343,845,888 B）
- 完整 3.11 GiB 实跑校验：见下方"实跑结果"

### 实跑结果（2026-09-25，全部通过）

| 环节 | 结果 |
|---|---|
| `-slim` 下载 | 101 块 × 32 MB、8 并发，**1 分 23 秒**拿到 3,343,845,888 B |
| SHA-256 | **通过**（`97437007c19310a5…` 与基准一致） |
| revision 防漂移 | HEAD 比对 `Content-Length` = 20,437,336,576 **通过** |
| `-from-extract` 展开 | 产出 **20,437,336,576 B**，与原模板尺寸完全一致 |
| `Artifact.open` | objects=**1190**，按前缀分布与原始一致（frontend 6 / text 773 / mtp 12 / vision 333 / dflash2 66） |
| 载荷抽样比对 | frontend / vision / dflash2 / mtp / draft_head **逐字节一致**（`match=True`） |

> 一个容易写错的地方（我自己的校验脚本先踩了一次）：`o.offset` 是
> **payload 相对偏移**，文件绝对位置 = `payload_offset + o.offset`，其中
> `payload_offset = align_up(16 + json_bytes, 4096) = 188,416`。
> 直接用 `o.offset` 当文件偏移会读到错位数据。

### 结论

**仓库只需存这份源码**（`main.go` + `README.md`，区段表内嵌），不必存任何 GB 级二进制。
用户克隆后一条命令即得模板：

```bash
go build -o template-fetch.exe main.go && ./template-fetch.exe -out <模板路径>
```
