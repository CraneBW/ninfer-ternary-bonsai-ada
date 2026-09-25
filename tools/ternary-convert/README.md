# ternary-convert —— 三元 Bonsai → `.ninfer` 转换工具链

把 **Ternary-Bonsai-2-27B**（PQ2_0 / PTQ1_0 两种三元打包）的 GGUF 转成 zatfung 能直接加载的
`.ninfer` artifact。

## 组件

| 文件 | 角色 |
|---|---|
| **`pack_zatfung.py`** | **主转换器**。由上游 `pack.py` 改造而来：三个路径环境变量化 + 修掉 PTQ1_0 的自检 bug |
| `prepare_pack.py` | 从原 `pack.py` 生成 `pack_zatfung.py`，把改造点固定下来（可复核、可重放） |
| **`CONVERSION_NOTES.md`** | 完整操作记录：四个坑、实测数据、模板瘦身、Range 下载器。**先读这份** |
| `make_slim_backup.py` | 生成"区段表"的脚本 —— 记录 3.11 GiB 瘦身备份是怎么从完整模板算出来的 |
| `restore_v2_template.py` | 从瘦身 extract 还原完整模板。**该功能已被 `tools/template-fetch` 覆盖**，保留作独立参考 |
| `qwen3_8_27b_v2_slim_meta.json` | 区段表快照。实际使用时以 `tools/template-fetch/main.go` 内嵌的表为准 |

## 完整复现流程

```bash
# 1) 取模板 —— 只下载 3.11 GiB，自己拼出 19 GiB 的完整模板
cd ../template-fetch && go build -o template-fetch.exe main.go
./template-fetch.exe -out D:\tpl\qwen3_8_27b_v2.ninfer

# 2) 取 GGUF（外部，上游发布）
#    Ternary-Bonsai-2-27B-PQ2_0.gguf    type 142
#    Ternary-Bonsai-2-27B-PTQ1_0.gguf   type 143

# 3) 转换
cd ../ternary-convert
set ZATFUNG_ROOT=<zatfung 仓库根>            # 提供 tools.artifact
set ZATFUNG_TEMPLATE=D:\tpl\qwen3_8_27b_v2.ninfer
set ZATFUNG_GGUF=<GGUF 路径>
python pack_zatfung.py check                  # 先验证几何/解码/字节往返，不写文件
python pack_zatfung.py build <输出>.ninfer    # 产出
```

`pack_zatfung.py` 接受的环境变量（都有默认值，见文件头）：

| 变量 | 含义 |
|---|---|
| `ZATFUNG_ROOT` | zatfung 仓库根，用于 `import tools.artifact`（默认本仓库） |
| `ZATFUNG_TEMPLATE` | 完整模板 artifact 路径 |
| `ZATFUNG_GGUF` | 要转换的 GGUF |

## 运行前提

`tools/artifact/layouts.py` 模块级 `import torch`（138 处），所以需要：

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

CPU 版 torch 即可，不需要 CUDA 版。

## 上游与许可

- 原 `pack.py` 及 `MAPPING.json`（张量映射规则）来自三元移植工作区
  `ninfer-ternary-qwen3.8-27b-3090`（Apache-2.0，NInfer 派生）。
  `prepare_pack.py` 记录了相对它的全部改动，因此这里不重复分发原文件。
- 模型权重与三元打包格式归其原作者（prism-ml / Hikari07jp / Qwen）；
  本工具链只做格式转换，不修改权重数值。

## 已知约束

- 模板必须是 **v2 容器**。HF 上的 `main` 已是 **v3**（magic `NINFER\x00\x03`、JSON 在 offset 32），
  与 zatfung 的读取器不兼容 —— 所以 URL 钉死在 revision `dc370fb6295a`。
- `check` 模式的 offset 自检在原版里硬编码了 PQ2_0 的组布局（34 B/组、scale 在组首），
  对 PTQ1_0（28 B/组、scale 在组尾 `[26:28]`）会崩 —— `pack_zatfung.py` 已按 `fmt` 分派修正。
