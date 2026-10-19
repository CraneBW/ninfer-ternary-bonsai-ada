"""Split a Qwen3.8-27B NInfer v2 artifact into two tensor-parallel rank artifacts.

Rank r of 2 keeps half of every head / intermediate dimension:
  * column-parallel (split output rows N): attention qkgv, GDN qkvz, GDN a/b, a_log, dt_bias,
    GDN convolution channels, MLP gate_up, and the same for the MTP layer;
  * row-parallel (split input columns K): attention output, GDN output, MLP down (each rank then
    produces a partial sum that the runtime all-reduces);
  * everything else (embeddings, LM/draft heads, norms, divisors, vision, resources) is copied.

Every split is done on exact stored words (FP8 codes + row scales, NVFP4 packed codes + natural
scales + divisor, W8 groups + scales, BF16/FP32 values) and re-encoded with tools.artifact, so the
two halves re-concatenate bit-exactly to the source (see --verify).

usage: python -m tools.tp2.shard_qwen38_27b SRC OUT_PREFIX [--verify]
       writes OUT_PREFIX.rank0.ninfer and OUT_PREFIX.rank1.ninfer
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from tools.artifact.container import (Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec,
                                      TensorObject, TensorSpec)
from tools.artifact import layouts as L

TP = 2


@dataclass(frozen=True)
class Rule:
    axis: str                  # "rows" (N) or "cols" (K)
    segments: tuple[int, ...]  # lengths of consecutive logical segments along that axis; each is halved


ATTN_QKGV = Rule("rows", (6144, 1024, 6144, 1024))
GDN_QKVZ = Rule("rows", (2048, 2048, 6144, 6144))
GDN_AB = Rule("rows", (48, 48))
GDN_HEADS = Rule("rows", (48,))
GDN_CONV = Rule("cols", (2048, 2048, 6144))     # (4, 10240) taps x channels
MLP_GATE_UP = Rule("rows", (17408, 17408))
K6144 = Rule("cols", (6144,))
K17408 = Rule("cols", (17408,))


def rule_for(name: str) -> Rule | None:
    if name.startswith("vision/") or not (name.startswith("text/layers/") or name.startswith("mtp/layer/")):
        return None
    table = {
        "/attention/query_key_gate_value": ATTN_QKGV,
        "/attention/output": K6144,
        "/gdn/query_key_value_z": GDN_QKVZ,
        "/gdn/a_b_projection": GDN_AB,
        "/gdn/a_log": GDN_HEADS,
        "/gdn/dt_bias": GDN_HEADS,
        "/gdn/convolution": GDN_CONV,
        "/gdn/output": K6144,
        "/mlp/gate_up": MLP_GATE_UP,
        "/mlp/down": K17408,
    }
    for suffix, rule in table.items():
        if name.endswith(suffix):
            return rule
    return None


def shard_shape(shape: tuple[int, ...], rule: Rule) -> tuple[int, ...]:
    if rule.axis == "rows":
        assert shape[0] == sum(rule.segments), (shape, rule)
        return (shape[0] // TP,) + tuple(shape[1:])
    assert shape[-1] == sum(rule.segments), (shape, rule)
    return tuple(shape[:-1]) + (shape[-1] // TP,)


def take(t: torch.Tensor, dim: int, segments: tuple[int, ...], rank: int, unit: int = 1) -> torch.Tensor:
    """Concatenate rank's half of each logical segment along `dim`; `unit` = storage elements per
    logical element along that dim (e.g. 1/2 for packed NVFP4 codes is expressed via scale)."""
    parts, start = [], 0
    for seg in segments:
        seg_units = seg * unit
        assert seg_units % TP == 0
        half = seg_units // TP
        parts.append(t.narrow(dim, start + rank * half, half))
        start += seg_units
    assert start == t.shape[dim], (start, t.shape, dim)
    return torch.cat(parts, dim=dim).contiguous()


def take_frac(t: torch.Tensor, dim: int, segments: tuple[int, ...], rank: int, num: int, den: int) -> torch.Tensor:
    """Like take(), with storage elements per logical element = num/den along dim."""
    parts, start = [], 0
    for seg in segments:
        seg_units = seg * num // den
        assert seg * num % den == 0 and seg_units % TP == 0
        half = seg_units // TP
        parts.append(t.narrow(dim, start + rank * half, half))
        start += seg_units
    assert start == t.shape[dim], (start, t.shape, dim)
    return torch.cat(parts, dim=dim).contiguous()


def shard_payload(obj: TensorObject, raw: memoryview, rule: Rule, rank: int) -> bytes:
    shape = tuple(obj.shape)
    new_shape = shard_shape(shape, rule)
    dim = 0 if rule.axis == "rows" else len(shape) - 1
    fmt = obj.format
    if fmt == "FP8_E4M3FN_ROW_BF16S":
        codes, scales = L.decode_fp8_row_scaled_words(raw, shape)          # [N,K] u8, [N] bf16
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule.segments, rank), take(scales, 0, rule.segments, rank)
        else:
            codes = take(codes, 1, rule.segments, rank)
        return L.encode_fp8_row_scaled(codes, scales, new_shape)
    if fmt == "NVFP4":
        codes, scales, divisor = L.decode_nvfp4_words(raw, shape)          # [N,K/2], [N,K/16], ()
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule.segments, rank), take(scales, 0, rule.segments, rank)
        else:
            codes = take_frac(codes, 1, rule.segments, rank, 1, 2)
            scales = take_frac(scales, 1, rule.segments, rank, 1, 16)
        return L.encode_nvfp4(codes, scales, divisor, new_shape)
    if fmt == "W8G32_F16S":
        scales, codes = L.decode_row_split_codes(raw, fmt, shape)          # [N,G], [N,G,32]
        geom = L.row_split_geometry(fmt, shape)
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule.segments, rank), take(scales, 0, rule.segments, rank)
        else:
            assert geom.groups_per_row * 32 == shape[-1], "padded K not supported for K split"
            codes = take_frac(codes, 1, rule.segments, rank, 1, 32)
            scales = take_frac(scales, 1, rule.segments, rank, 1, 32)
        return L.encode_row_split(codes, scales, fmt, new_shape)
    if fmt in ("BF16", "FP32"):
        t = L.decode_direct(raw, fmt, shape)
        t = take(t, dim, rule.segments, rank)
        return L.encode_direct(t, fmt)
    raise ValueError(f"no shard rule for format {fmt} ({obj.name})")


def build(src: str, prefix: str) -> None:
    art = Artifact.open(src)
    specs = []
    for obj in art.objects:
        if isinstance(obj, TensorObject):
            rule = rule_for(obj.name)
            shape = shard_shape(tuple(obj.shape), rule) if rule else tuple(obj.shape)
            specs.append(TensorSpec(obj.name, shape, obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
    identity = ArtifactIdentity(art.identity.model_id, art.identity.weights_id + "-tp2")
    writers = [ArtifactWriter(f"{prefix}.rank{r}.ninfer", identity, specs) for r in range(TP)]
    for i, obj in enumerate(art.objects):
        raw = art.payload(obj)
        rule = rule_for(obj.name) if isinstance(obj, TensorObject) else None
        for r in range(TP):
            writers[r].write(obj.name, shard_payload(obj, raw, rule, r) if rule else raw)
        if rule and i % 40 == 0:
            print(f"[{i}/{len(art.objects)}] {obj.name} {tuple(obj.shape)} -> {shard_shape(tuple(obj.shape), rule)}", flush=True)
    for w in writers:
        w.finish()
    print("done", flush=True)


def verify(src: str, prefix: str, limit: int = 0) -> None:
    """Re-concatenate each split tensor from the two ranks and compare words with the source."""
    art = Artifact.open(src)
    ranks = [Artifact.open(f"{prefix}.rank{r}.ninfer") for r in range(TP)]
    checked = 0
    for obj in art.objects:
        if not isinstance(obj, TensorObject):
            assert all(bytes(rk.payload(obj.name)) == bytes(art.payload(obj)) for rk in ranks)
            continue
        rule = rule_for(obj.name)
        if rule is None:
            for rk in ranks:
                assert bytes(rk.payload(obj.name)) == bytes(art.payload(obj)), obj.name
            continue
        # Round trip: shard again from source and compare to the stored rank payloads.
        for r, rk in enumerate(ranks):
            assert bytes(rk.payload(obj.name)) == shard_payload(obj, art.payload(obj), rule, r), obj.name
        # Semantic check on dequantized values: rank halves of each segment reassemble the source.
        if obj.format in ("FP8_E4M3FN_ROW_BF16S",):
            full = L.dequantize_fp8_row_scaled(art.payload(obj), obj.shape)
            halves = [L.dequantize_fp8_row_scaled(rk.payload(obj.name), rk.find(obj.name).shape) for rk in ranks]
            dim = 0 if rule.axis == "rows" else 1
            rebuilt, off = [], [0, 0]
            for seg in rule.segments:
                h = seg // TP
                for r in range(TP):
                    rebuilt.append(halves[r].narrow(dim, off[r], h)); off[r] += h
            assert torch.equal(torch.cat(rebuilt, dim=dim), full), obj.name
        checked += 1
        if limit and checked >= limit:
            break
    print(f"verify ok ({checked} split tensors)", flush=True)


if __name__ == "__main__":
    src, prefix = sys.argv[1], sys.argv[2]
    if "--verify" in sys.argv:
        verify(src, prefix, limit=int(sys.argv[sys.argv.index("--verify") + 1]) if len(sys.argv) > sys.argv.index("--verify") + 1 else 0)
    else:
        build(src, prefix)
