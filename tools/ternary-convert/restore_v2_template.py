#!/usr/bin/env python3
"""从瘦身备份还原模板 artifact（与原文件功能等价）。

用法：python restore_v2_template.py [输出路径]
默认输出到本目录下的 qwen3_8_27b_v2.ninfer
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = HERE / "qwen3_8_27b_v2_slim_meta.json"
EXTRACT = HERE / "qwen3_8_27b_v2_slim_extract.bin"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "qwen3_8_27b_v2.ninfer"
CHUNK = 8 << 20


def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
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
