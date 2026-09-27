"""Compare the artifact's folded sign table against the authoritative GGUF metadata.

The runtime maps an activation as y = H * (s . (P * x)) using `text/hadamard_signs`, a flat F32
array of +-1 partitioned by `text/hadamard_widths`. A wrong width prefix sum, a wrong sign order
or a wrong dtype conversion would silently rotate every folded weight by the wrong signs -- which
produces fluent-looking nonsense everywhere, exactly the symptom under investigation.

Usage: check_signs.py <artifact.ninfer> <bonai2-hadamard-meta.json>

Origin
------
Ported from the ternary patch set published by shensanshu at
https://modelscope.cn/models/shensanshu/ninfer-ada-ternary  (`tools/verify/check_signs.py`).
The check itself is unchanged; this copy drops the author's hard-coded Windows source root in
favour of one derived from this file's own location, so the tree is relocatable.

Why it is worth keeping
-----------------------
This is the only check in that tool set that needs no GGUF source, no build, and no engine dump:
an `.ninfer` plus the Hadamard metadata is enough. It compares the artifact's sign table against
the authority element-wise (`np.array_equal`, not a correlation), which is the strongest form of
agreement available here -- the 402 ternary matrices are all stored in the rotated basis, so
correlation cannot verify them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# This file lives at <repo>/tools/verify/check_signs.py; the container reader is at <repo>/tools.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from tools.artifact import container  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: check_signs.py <artifact.ninfer> <bonai2-hadamard-meta.json>")
        return 64
    artifact_path, meta_path = sys.argv[1], sys.argv[2]

    with open(meta_path, encoding="utf-8") as f:
        raw = json.load(f)
    P = "prism.hadamard."
    meta = {k[len(P):]: v for k, v in raw.items() if k.startswith(P)}
    widths_meta = meta.get("sign_widths")
    values_meta = meta.get("sign_values")
    print(f"meta: sign_widths={widths_meta} len(sign_values)={len(values_meta)}")
    print(f"meta: block_size={meta.get('block_size')} sign_mode={meta.get('sign_mode')} "
          f"transform={meta.get('transform')}")
    print(f"meta: n_weight_names={len(meta.get('weight_names', []))} "
          f"inverse={meta.get('inverse_weight_names')} gdn_v_grouped={meta.get('gdn_v_grouped')}")

    meta_values = np.asarray(values_meta, dtype=np.float32)
    print(f"meta values: unique={np.unique(meta_values)[:5]} sum={meta_values.sum()}")

    with container.Artifact.open(artifact_path) as art:
        signs = np.frombuffer(bytes(art.payload("text/hadamard_signs")), dtype=np.float32)
        widths = np.frombuffer(bytes(art.payload("text/hadamard_widths")), dtype=np.int32)
    print(f"artifact: signs n={signs.size} unique={np.unique(signs)[:5]}")
    print(f"artifact: widths={widths.tolist()} sum={int(widths.sum())}")

    if signs.size != meta_values.size:
        print("MISMATCH length")
        return 1
    same = np.array_equal(signs, meta_values)
    diff = np.abs(signs - meta_values).max()
    print(f"signs identical to meta: {same}  max|diff|={diff}")
    if not same:
        # locate the first divergence: a shifted prefix sum shows up as a block-wise disagreement
        bad = np.nonzero(signs != meta_values)[0]
        print(f"  divergent count={bad.size} first={bad[:8].tolist()}")
    if widths.tolist() != list(widths_meta):
        print(f"  WIDTH MISMATCH artifact={widths.tolist()} meta={widths_meta}")
    else:
        print("widths match (prefix sums 5120->0, 6144->5120, 17408->11264)")
    return 0 if same else 2


if __name__ == "__main__":
    raise SystemExit(main())
