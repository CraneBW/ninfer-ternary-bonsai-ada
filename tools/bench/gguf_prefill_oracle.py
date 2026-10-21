"""Decode the existing GSQ sample blocks on CPU for the native GGUF Op test.

Uses GGML's independent NumPy decoder (gguf package), never the CUDA bridge.
Reads 128-row samples by default. --full-shapes covers existing up/down
projection shapes, decoding at most 128 rows at a time; never loads the model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import struct
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--gguf-source", type=Path, required=True)
    p.add_argument("--full-shapes", action="store_true",
                   help="decode complete existing up/down projections for each format")
    a = p.parse_args()
    sys.path.insert(0, str(a.gguf_source))
    import numpy as np
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    from gguf.quants import dequantize

    formats = ["iq4_xs", "iq3_s", "iq3_xxs", "iq2_xs", "q2_k", "q4_k"]
    result = {"artifact": str(a.artifact.resolve()), "decoder": "gguf CPU NumPy",
              "decoder_version": importlib.metadata.version("gguf"), "formats": {}}
    if a.full_shapes:
        result["token_counts"] = [1, 8, 9, 512, 1024, 2048, 2049]
        result["timing_tokens"] = [1, 512, 1024, 2048]
    with a.artifact.open("rb") as f:
        header = f.read(32)
        assert header[:8] == b"NINFER\0\x03"
        length = struct.unpack_from("<Q", header, 8)[0]
        directory = json.loads(f.read(length))
        data_begin = (32 + length + 4095) // 4096 * 4096
        for fmt in formats:
            format_name = "gguf_" + fmt
            candidates = [x for x in directory["objects"]
                          if x.get("format") == format_name and x.get("kind") == "tensor"
                          and len(x["shape"]) == 2 and x["shape"][0] >= 128
                          and x["shape"][1] % 256 == 0]
            if a.full_shapes:
                selected = []
                for shape in ([34816, 5120], [5120, 17408], [17408, 5120]):
                    obj = next((x for x in candidates if x["shape"] == shape), None)
                    if obj is not None:
                        selected.append(obj)
            else:
                selected = [next(iter(candidates))]
            qt = GGMLQuantizationType[fmt.upper()]
            block, size = GGML_QUANT_SIZES[qt]
            for obj in selected:
                n, k = obj["shape"]
                rows = n if a.full_shapes else 128
                row_bytes = k // block * size
                if k % block or obj["bytes"] < rows * row_bytes:
                    raise ValueError(f"invalid tensor geometry: {obj['id']}")
                f.seek(data_begin + obj["offset"])
                row_sums = []
                for first_row in range(0, rows, 128):
                    count = min(128, rows - first_row)
                    payload = f.read(count * row_bytes)
                    if len(payload) != count * row_bytes:
                        raise EOFError(f"incomplete tensor: {obj['id']}")
                    packed = np.frombuffer(payload, dtype=np.uint8).reshape(count, row_bytes)
                    decoded = dequantize(packed, qt)
                    row_sums.extend(decoded.astype(np.float64).sum(axis=1).tolist())
                key = f"{format_name}_n{n}_k{k}" if a.full_shapes else format_name
                result["formats"][key] = {
                    "format": format_name, "object_id": obj["id"],
                    "offset": obj["offset"], "n": n, "k": k, "rows": rows,
                    "row_sums": row_sums}
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
