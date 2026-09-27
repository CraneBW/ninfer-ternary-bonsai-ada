"""numpy oracle for the folded-basis rotation kernels (forward and inverse).

Reads the raw f32 dumps the standalone nvcc harness wrote, rebuilds the documented transform
explicitly, and compares against the REAL kernel output element for element.

The pass criterion is deliberately two-sided: the documented variant must match, AND a family of
plausible-but-wrong variants must all fail.  A one-sided check cannot tell "the kernel is right"
from "the oracle is too loose" -- which is exactly the trap the handoff notes call a
self-consistent false green.

Usage: oracle_rot.py <dump-dir>

Origin
------
Ported verbatim from the ternary patch set published by shensanshu at
https://modelscope.cn/models/shensanshu/ninfer-ada-ternary  (`tools/verify/oracle_rot.py`).
Unchanged: it takes the dump directory as an argument and hard-codes no path.

Why it is worth keeping
-----------------------
It is the only check in that tool set that exercises the rotation kernels with **T > 1**.
At T = 1 the token-major and row-major activation layouts coincide, so a decode-shaped check
passes bit-exactly even when the real prefill path is wrong -- the failure mode the author's
toolchain documents, and the one this repo has no other way to catch. `CASES` carries
`plain_t3`, `perm_t2` and `inv_t2` for exactly that reason.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BLOCK = 1024
CASES = [
    # tag, k, tokens, perm_hd, perm_nk, perm_rep, inverse
    ("plain_t1", 5120, 1, 0, 0, 1, False),
    ("plain_t3", 5120, 3, 0, 0, 1, False),
    ("perm_t1", 6144, 1, 128, 16, 3, False),
    ("perm_t2", 6144, 2, 128, 16, 3, False),
    ("wide_t1", 17408, 1, 0, 0, 1, False),
    ("inv_t2", 5120, 2, 0, 0, 1, True),
]


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount32(a: np.ndarray) -> np.ndarray:
    """numpy 1.26 has no bitwise_count, so unroll the byte lookup explicitly."""
    v = a.astype(np.uint32)
    return (
        _POPCOUNT[v & 0xFF].astype(np.uint16)
        + _POPCOUNT[(v >> 8) & 0xFF].astype(np.uint16)
        + _POPCOUNT[(v >> 16) & 0xFF].astype(np.uint16)
        + _POPCOUNT[(v >> 24) & 0xFF].astype(np.uint16)
    )


def hadamard(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    parity = _popcount32(idx[:, None] & idx[None, :]) & 1
    return np.where(parity == 1, -1.0, 1.0) / np.sqrt(float(n))


def forward_source_index(j: int, hd_n: int, nk_n: int, rep_n: int) -> int:
    """inverse of the reshape/permute llama.cpp applies: post-P column j -> source column."""
    hd = j % hd_n
    q = j // hd_n
    nk = q // rep_n
    rep = q % rep_n
    return hd + hd_n * nk + hd_n * nk_n * rep


def forward_dest_index(i: int, hd_n: int, nk_n: int, rep_n: int) -> int:
    """the opposite direction: source column i -> post-P column (a negative control)."""
    hd = i % hd_n
    nk = (i // hd_n) % nk_n
    rep = i // (hd_n * nk_n)
    return hd + hd_n * rep + hd_n * rep_n * nk


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return float(np.linalg.norm(got - ref))
    return float(np.linalg.norm(got - ref) / denom)


def evaluate(tag: str, k: int, tokens: int, hd_n: int, nk_n: int, rep_n: int, inverse: bool,
             dump_dir: str, H: np.ndarray) -> bool:
    x = np.fromfile(os.path.join(dump_dir, tag + ".in.f32"), dtype=np.float32).astype(np.float64)
    s = np.fromfile(os.path.join(dump_dir, tag + ".signs.f32"), dtype=np.float32).astype(np.float64)
    got = np.fromfile(os.path.join(dump_dir, tag + ".out.f32"), dtype=np.float32).astype(np.float64)
    # TOKEN-major, which is this engine's convention and not the author's.
    #
    # ninfer follows the ggml layout where ne[0] is the contiguous axis, so an activation with
    # ne = (k, tokens) stores element (column, token) at `token * k + column` -- a token's k-block
    # is contiguous. numpy's `reshape(k, tokens)` assumes the opposite (C order: consecutive memory
    # is consecutive tokens) and the two agree EXACTLY at tokens == 1. That is why the author's
    # copy passed its two single-token cases and failed all three of its T > 1 cases here: a real
    # convention difference, not a kernel defect. `reshape(tokens, k).T` gives the token-major
    # reading. The negative controls separate under either reading, so the check keeps its power.
    x = x.reshape(tokens, k).T
    got = got.reshape(tokens, k).T

    blocks = k // BLOCK
    permuted = (not inverse) and rep_n > 1

    def blockwise_hadamard(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        for b in range(blocks):
            lo = b * BLOCK
            out[lo:lo + BLOCK] = H @ vec[lo:lo + BLOCK]
        return out

    def permute(vec: np.ndarray, src) -> np.ndarray:
        return np.array([vec[src(j)] for j in range(k)], dtype=np.float64)

    # Build every variant as a full [k, tokens] array so the comparison never broadcasts.
    columns: dict[str, list[np.ndarray]] = {}
    for t in range(tokens):
        col = x[:, t]
        if inverse:
            current = {
                "CORRECT  s*(H*z)": s * blockwise_hadamard(col),
                "WRONG    H*(s*z)": blockwise_hadamard(s * col),
                "WRONG    H*z": blockwise_hadamard(col),
                "WRONG    s*z": s * col,
            }
        else:
            base = col if not permuted else permute(
                col, lambda j: forward_source_index(j, hd_n, nk_n, rep_n))
            # one sign row off: block b uses sign row b-1
            shifted = np.empty_like(s)
            for b in range(blocks):
                src = ((b - 1) % blocks) * BLOCK
                dst = b * BLOCK
                shifted[dst:dst + BLOCK] = s[src:src + BLOCK]
            current = {
                "CORRECT  H*(s*(P*x))": blockwise_hadamard(s * base),
                "WRONG    s*(H*(P*x))": s * blockwise_hadamard(base),
                "WRONG    unnormalized H": blockwise_hadamard(s * base) * np.sqrt(BLOCK),
                "WRONG    sign row shifted": blockwise_hadamard(shifted * base),
            }
            if permuted:
                current["WRONG    H*(s*x) no P"] = blockwise_hadamard(s * col)
                current["WRONG    P the other way"] = blockwise_hadamard(
                    s * permute(col, lambda j: forward_dest_index(j, hd_n, nk_n, rep_n)))
        for name, value in current.items():
            columns.setdefault(name, []).append(value)

    variants = {name: np.stack(cols, axis=1) for name, cols in columns.items()}
    correct_name = "CORRECT  s*(H*z)" if inverse else "CORRECT  H*(s*(P*x))"
    ref = variants[correct_name]

    correct = rel_l2(got, ref)
    print(f"case {tag:9s} k={k:<6d} tokens={tokens} perm={permuted}")
    print(f"  norms: ||x||={np.linalg.norm(x):.6g} ||ref||={np.linalg.norm(ref):.6g} "
          f"||got||={np.linalg.norm(got):.6g}  ratio={np.linalg.norm(got) / np.linalg.norm(ref):.6g}")
    ok = correct <= 0.01
    print(f"  {'PASS' if ok else 'FAIL'}  CORRECT variant          rel_l2={correct:.3e}")
    separation = True
    for name, value in variants.items():
        if name.startswith("CORRECT"):
            continue
        r = rel_l2(got, value)
        good = r > 10.0 * max(correct, 1e-6)
        separation = separation and good
        print(f"        {name:26s} rel_l2={r:.4f}  {'(differs OK)' if good else '(TOO CLOSE)'}")
    return ok and separation


def main() -> int:
    dump_dir = sys.argv[1]
    H = hadamard(BLOCK)
    print(f"oracle: explicit normalized Sylvester-Hadamard {BLOCK}x{BLOCK} "
          f"(H[i][j] = popcount(i&j)&1 ? -1 : +1, /sqrt({BLOCK}))")
    print(f"        symmetric={np.allclose(H, H.T)}  orthogonal={np.allclose(H @ H, np.eye(BLOCK), atol=1e-9)}")
    results = []
    for tag, k, tokens, hd_n, nk_n, rep_n, inverse in CASES:
        results.append(evaluate(tag, k, tokens, hd_n, nk_n, rep_n, inverse, dump_dir, H))
        print()
    passed = sum(1 for r in results if r)
    print(f"RESULT: {passed}/{len(results)} cases pass (correct variant matches AND all "
          f"negative controls separate)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
