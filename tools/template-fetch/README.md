# template-fetch —— 模板抓取器

从 HuggingFace **只下载 3.11 GiB**，拿到 `pack.py` 所需的 19.03 GiB 模板 artifact。

这样仓库里只需要这份源码，**不必存 3 GB 级的二进制**；用户一条命令即可自建模板。

---

## 为什么可以只下 3.11 GiB

`pack.py`（三元 Bonsai → `.ninfer` 的打包器）需要一个"模板 artifact"来借用
**419 个对象**的载荷：vision(333) + dflash2(66) + mtp(12) + frontend(6) + text/draft_head(2)。
其余的 **771 个 `text/*` 对象**由打包器自己 `produce()` 生成，**从不读取模板里的它们**。

而在文件布局上，那 419 个对象**恰好集中在两端**：

| 文件偏移 | 内容 | 是否需要 |
|---:|---|---|
| `0` .. `13,025,792` | header（16 B 前缀 + 185 KB JSON 对象目录）+ frontend(6) | ✅ |
| `13,025,792` .. `17,106,516,480` | 771 个 `text/*` | ❌ 15.92 GiB，跳过 |
| `17,106,516,480` .. `20,437,336,576` | draft_head(2) + mtp(12) + vision(333) + dflash2(66) | ✅ |

所以用 HTTP Range 取这两段即可：**3,343,845,888 B = 3.114 GiB，省 83.6%**。

> 顺带一个容易踩的点：这条 419 的口径与 `pack.py` 注释里的 353 不一致 ——
> 后者写于 dflash2 加入之前。以代码的实际判据（`非 text/*` 即借用）为准。

---

## 用法

```bash
go build -o template-fetch.exe main.go   # 或 go run main.go

./template-fetch.exe                     # 默认：下载并产出可直接使用的完整模板
./template-fetch.exe -slim               # 只产出 3.11 GiB 的 extract
./template-fetch.exe -out D:\tpl.ninfer  # 指定输出路径
./template-fetch.exe -j 8                # 并发块数（默认 4）
./template-fetch.exe -no-resume          # 丢弃进度重新下载
```

得到模板后，把它设为 `pack.py` 的模板路径：

```bash
set ZATFUNG_TEMPLATE=<上面 -out 的路径>
python pack_zatfung.py build <输出>.ninfer
```

### 两种输出模式

| 模式 | 产出 | 磁盘占用 | 说明 |
|---|---|---|---|
| 默认 | 完整模板（`20,437,336,576 B` 逻辑尺寸） | ≤ 22.1 GiB | 中间 15.92 GiB 区段留零。**可直接使用** |
| `-slim` | extract（`3,343,845,888 B`） | 3.11 GiB | 需自行按区段表还原成完整模板 |

中间那段为什么可以是零：`Artifact.open()` 的 `_validate_ranges` 只校验每个对象的
`offset` 落在文件范围内、**不校验内容**；而 pack.py 对 `text/*` 走 `produce()`，
从不去读模板里那些字节。

---

## 可靠性

- **HTTP Range 精确抓取**，并显式要求 `206 Partial Content`；若服务端忽略 Range
  返回 `200`（会灌进整个 19 GiB），直接报错而不是默默写坏数据。
- **断点续传**：进度记在 `<out>.tplfetch.json`，中断后重跑即接着下，已完成的块跳过。
- **分块并发**：默认 32 MB/块、4 并发；块不跨区段，因此到 URL 的映射是单段线性的。
- **完整校验**：全部块就绪后对整份数据算 SHA-256，与内置基准比对：
  ```
  97437007c19310a5f3204e594a6fb1acc190e2ffade9acc568b8c9310ea2a72c
  ```
  不符即报错并提示 `-no-resume` 重下。
- **revision 固定**：URL 钉在 `dc370fb6295a`（v2 容器）。
  ⚠️ **不要用 `main`** —— HF 上的 `main` 已换成 **v3** artifact
  （magic `NINFER\x00\x03`、JSON 在 offset 32、schema `{"components":...}`），
  与 zatfung / ninfer-3090 的 **v2** 读取器不兼容。程序会 HEAD 一次并比对
  `Content-Length`，若源文件被替换会直接报错而不是产出坏模板。

---

## 区段表（内嵌于 `main.go`）

```go
var segments = []segment{
    {BlobOffset: 0,          FileOffset: 0,           Bytes: 13025792},   // header + frontend
    {BlobOffset: 13025792,   FileOffset: 17106516480, Bytes: 3330820096}, // draft_head..dflash2
}
const sourceBytes  = 20437336576
const extractBytes = 3343845888
```

来源：
`https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/dc370fb6295a/qwen3_8_27b.ninfer`
（revision `dc370fb6295a`，2026-09-06，"Update artifact with DFlash2 companion weights"）
