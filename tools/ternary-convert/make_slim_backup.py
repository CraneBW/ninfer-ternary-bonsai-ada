#!/usr/bin/env python3
"""把 19.03 GiB 的模板 artifact 瘦身成 ~3.11 GiB 的"有用部位"备份 + 还原脚本。

原理
----
`pack.py` 对一个模板 artifact 只用到两样东西：
  1. **头部**：16 B 前缀 + JSON 对象目录（185 KB）—— 决定产物要写哪些对象；
  2. **353 个借用对象的载荷**（vision/mtp/dflash2/frontend + text/draft_head）。

其余 771 个 `text/*` 对象的载荷由 packer 自己 `produce()` 生成，从不读取。

而在文件布局上，这 353 个对象**恰好集中在两端**：

    文件位置（字节）                                 用途
    0            .. 13,025,593                      header + frontend(6)
    13,025,593   .. 17,106,516,480                  771 个 text/* 对象（无用）
    17,106,516,480 .. 20,437,336,576                draft_head(2) + mtp(12)
                                                    + vision(333) + dflash2(66)

所以只需备份这两段，还原时把中间区段留空（零填充）即可 —— `Artifact.open()`
的 `_validate_ranges` 只校验 offset 落在文件范围内，不校验内容，因此还原后的文件
在功能上与原模板**完全等价**（pack.py 行为一致）。

输出
----
- `qwen3_8_27b_v2_slim_extract.bin`  ~3.11 GiB，两段原始字节顺序拼接
- `qwen3_8_27b_v2_slim_meta.json`    区段映射，还原用
- `restore_v2_template.py`           还原脚本（产出 byte-equivalent 模板）
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ZATFUNG = r"G:/Agents/ninfer-works/zatfung"
if ZATFUNG not in sys.path:
    sys.path.insert(0, ZATFUNG)

from tools.artifact import MAGIC, PAYLOAD_ALIGNMENT, get_layout  # noqa: E402
from tools.artifact.container import PREFIX, PREFIX_BYTES  # noqa: E402
from tools.artifact.layouts import align_up  # noqa: E402

SRC = Path(r"C:\WorkModels\Qwen3.8-27B\_pack_template\qwen3_8_27b_v2.ninfer")
OUT_DIR = SRC.parent
EXTRACT = OUT_DIR / "qwen3_8_27b_v2_slim_extract.bin"
META = OUT_DIR / "qwen3_8_27b_v2_slim_meta.json"
RESTORE = OUT_DIR / "restore_v2_template.py"
CHUNK = 8 << 20


def is_borrowed(name: str) -> bool:
    return (not name.startswith("text/")) or name.startswith("text/draft_head")


def main() -> int:
    if EXTRACT.exists():
        raise SystemExit(f"refusing to overwrite {EXTRACT}")

    with open(SRC, "rb") as f:
        magic, json_bytes = PREFIX.unpack(f.read(PREFIX_BYTES))
        if magic != MAGIC:
            raise SystemExit(f"source magic mismatch: {magic!r}")
        doc = json.loads(f.read(json_bytes))
        file_bytes = SRC.stat().st_size
    payload_off = align_up(PREFIX_BYTES + json_bytes, PAYLOAD_ALIGNMENT)

    objects = doc["objects"]
    borrowed = [o for o in objects if is_borrowed(o["name"])]
    kept = [o for o in objects if not is_borrowed(o["name"])]
    print("objects: %d total = %d borrowed + %d produced-by-packer"
          % (len(objects), len(borrowed), len(kept)))

    # 借用对象在文件中的绝对位置
    segs_abs = sorted((payload_off + o["offset"], payload_off + o["offset"] + o["bytes"])
                      for o in borrowed)
    # 只保留头 + frontend 这一段（第一段之前还有 header）
    first_kept_start = min(payload_off + o["offset"] for o in kept)
    first_kept_end = max(payload_off + o["offset"] + o["bytes"] for o in kept)

    seg_a = (0, first_kept_start)                                  # header + frontend
    seg_b = (first_kept_end, file_bytes)                           # draft_head..dflash2
    segments = [seg_a, seg_b]
    print("kept(text/*) occupies %s .. %s (%.3f GiB, skipped)"
          % (format(first_kept_start, ","), format(first_kept_end, ","),
             (first_kept_end - first_kept_start) / 2**30))

    total = 0
    with open(SRC, "rb") as src, open(EXTRACT, "wb") as dst:
        for off, end in segments:
            n = end - off
            src.seek(off)
            left = n
            while left:
                b = src.read(min(CHUNK, left))
                if not b:
                    raise EOFError("unexpected EOF")
                dst.write(b)
                left -= len(b)
            total += n
            print("  copied %s .. %s  (%s B)" % (format(off, ","), format(end, ","), format(n, ",")))

    meta = {
        "_doc": "模板瘦身备份：只有这些区段含 pack.py 真正会读取的数据。还原见 restore_v2_template.py",
        "source_file": str(SRC),
        "source_bytes": file_bytes,
        "extract_file": EXTRACT.name,
        "extract_bytes": total,
        "payload_offset": payload_off,
        "json_bytes": json_bytes,
        "objects_total": len(objects),
        "objects_borrowed": len(borrowed),
        "objects_skipped": len(kept),
        "skipped_range": [first_kept_start, first_kept_end],
        "copied_segments": [{"blob_offset": sum(s[1] - s[0] for s in segments[:i]),
                             "file_offset": s[0], "bytes": s[1] - s[0]}
                            for i, s in enumerate(segments)],
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    RESTORE.write_text(RESTORE_TEMPLATE.format(
        extract=EXTRACT.name, meta=META.name, out="qwen3_8_27b_v2.ninfer"), encoding="utf-8")

    print("\nextract: %s  (%s B = %.3f GiB)"
          % (EXTRACT.name, format(total, ","), total / 2**30))
    print("meta   : %s" % META.name)
    print("restore: %s" % RESTORE.name)
    print("saving : %.3f GiB (%.1f%% smaller)"
          % ((file_bytes - total) / 2**30, 100.0 * (file_bytes - total) / file_bytes))
    return 0


RESTORE_TEMPLATE = '''#!/usr/bin/env python3
"""从瘦身备份还原模板 artifact（与原文件功能等价）。

用法：python restore_v2_template.py [输出路径]
默认输出到本目录下的 {out}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = HERE / "{meta}"
EXTRACT = HERE / "{extract}"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "{out}"
CHUNK = 8 << 20


def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {{OUT}}")
    total = meta["source_bytes"]
    with open(EXTRACT, "rb") as src, open(OUT, "wb") as dst:
        # 先按最终大小建立文件（中间被跳过的区段留作零）
        dst.truncate(total)
        for seg in meta["copied_segments"]:
            src.seek(seg["blob_offset"])
            dst.seek(seg["file_offset"])
            left = seg["bytes"]
            while left:
                b = src.read(min(CHUNK, left))
                if not b:
                    raise EOFError("extract truncated")
                dst.write(b)
                left -= len(b)
    print("restored %s (%s B)" % (OUT, format(OUT.stat().st_size, ",")))
    print("skipped range %s kept as zeros -- 与原模板功能等价"
          % meta["skipped_range"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

if __name__ == "__main__":
    raise SystemExit(main())
