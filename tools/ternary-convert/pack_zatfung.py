#!/usr/bin/env python3
"""Bonsai 2 27B (arch qwen35) GGUF -> ninfer `.ninfer` ternary artifact  (M1-D packer).

Mapping rules are NOT derived here; they come from MAPPING.json (which records the evidence
for each).  This script owns: a seek-based GGUF reader; an INDEPENDENT reference decoder for
the two Prism ternary types ported from ggml/src/ggml-quants.c; byte-for-byte ternary plane
assembly plus its exact inverse (used as the losslessness proof); the inventory built from
the template artifact's own object list; and a streaming writer with borrowed payloads.

Borrowed payloads (deliberate, reported in the run summary):
  vision/*    333 -- Bonsai ships its vision tower as a separate mmproj GGUF; the template's
                     tower is the same official one and is not read during text decode.
  mtp/*        12 -- Bonsai's GGUF has NO MTP head (851 tensors, zero `blk.64.*`).  The
                     template's head is the same official Qwen3.8-27B head: all twelve
                     tensors matched at cosine >= 0.99966 with the seven norms BIT-IDENTICAL.
  frontend/*    6 -- tokenizer and friends.
  text/draft_head(+_token_ids) -- frequency shortlist; same tokenizer => same shortlist.

Modes
  check            geometry + decode + byte-round-trip proofs (no writes)
  layer3 <out>     minimal artifact: frontend + sign table + one full-attention layer
  build  <out>     full text model

Interpreter: <PYTHON>\\python.exe
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- zatfung 改造：路径全部环境变量化 -------------------------------------
# 原版硬编码 H:/... 路径。改为从环境变量取，便于同一脚本处理两个模型、
# 且不把 zatfung 的源码树和模型耦合进脚本。
import os
NINFER_ROOT = os.environ.get("ZATFUNG_ROOT", r"G:/Agents/ninfer-works/zatfung")
if NINFER_ROOT not in sys.path:
    sys.path.insert(0, NINFER_ROOT)

from tools.artifact import (  # noqa: E402
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
    encode_direct,
    row_split_geometry,
)

TEMPLATE = os.environ.get("ZATFUNG_TEMPLATE",
            r"C://WorkModels//Qwen3.8-27B//_pack_template//qwen3_8_27b_v2.ninfer")
GGUF = os.environ.get("ZATFUNG_GGUF",
            r"C://WorkModels//Qwen3.8-27B//Ternary-Bonsai-2-27B-PQ2_0.gguf")

T_PQ2_0, T_PTQ1_0, T_F32, T_BF16 = 142, 143, 0, 30
FMT = {T_PQ2_0: "PQ2_0_G128", T_PTQ1_0: "PTQ1_0_G128"}
HIDDEN, V_HEADS, V_HEAD_DIM = 5120, 48, 128
QK_ROWS, V_ROWS = 4096, 6144
SIGN_WIDTHS = [5120, 6144, 17408]
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


# ---------------------------------------------------------------------------
class Gguf:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        self.size = self.f.tell()
        self.f.seek(0)
        if self._raw(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()
        self.kv: dict[str, object] = {}
        for _ in range(self.n_kv):
            k = self._str()
            self.kv[k] = self._value(self._u32())
        self.kv_end = self.f.tell()
        self.tensor: dict[str, tuple[list[int], int, int]] = {}
        order = []
        for _ in range(self.n_tensors):
            name = self._str()
            ne = [self._u64() for _ in range(self._u32())]
            tt = self._u32()
            off = self._u64()
            self.tensor[name] = (ne, tt, off)
            order.append((off, name))
        # header_end is the end of the TENSOR INFO LIST, not the end of the KV block.
        # Recording it before this loop (as an earlier revision did) shifts data_start down
        # by the size of the list -- ~50 KB in this file -- so every payload read comes from
        # the wrong offset, while names/shapes/types still parse perfectly and a
        # self-consistent byte round trip still passes.  That is a silent, maximally
        # misleading failure: probe_gguf_types.py puts this file's header end at
        # 11,120,982 and the reader must agree.
        self.header_end = self.f.tell()
        self.alignment = int(self.kv.get("general.alignment", 32))
        self.data_start = -(-self.header_end // self.alignment) * self.alignment
        order.sort()
        self.tbytes: dict[str, int] = {}
        for i, (off, name) in enumerate(order):
            nxt = order[i + 1][0] if i + 1 < len(order) else (self.size - self.data_start)
            self.tbytes[name] = nxt - off
        self._cache: dict[str, bytes] = {}

    def _raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read of {n}")
        return b

    def _u32(self):
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._raw(8))[0]

    def _str(self):
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        if t in (0, 1, 7):
            return self._raw(1)[0]
        if t in (2, 3):
            return struct.unpack("<h", self._raw(2))[0]
        if t == 4:
            return struct.unpack("<I", self._raw(4))[0]
        if t == 5:
            return struct.unpack("<i", self._raw(4))[0]
        if t == 6:
            return struct.unpack("<f", self._raw(4))[0]
        if t == 8:
            return self._str()
        if t in (10, 11):
            return struct.unpack("<q", self._raw(8))[0]
        if t == 12:
            return struct.unpack("<d", self._raw(8))[0]
        if t == 9:
            et, n = self._u32(), self._u64()
            if et == 8:
                return [self._str() for _ in range(n)]
            if et == 6:
                return list(struct.unpack("<%df" % n, self._raw(4 * n)))
            if et in (0, 1, 7):
                return list(self._raw(n))
            if et in (2, 3):
                return list(struct.unpack("<%dh" % n, self._raw(2 * n)))
            if et == 4:
                return list(struct.unpack("<%dI" % n, self._raw(4 * n)))
            if et == 5:
                return list(struct.unpack("<%di" % n, self._raw(4 * n)))
            self.f.seek(FIXED[et] * n, 1)
            return f"<{n} values>"
        raise ValueError(f"gguf value type {t}")

    def payload(self, name: str) -> bytes:
        if name not in self._cache:
            _ne, _tt, off = self.tensor[name]
            self.f.seek(self.data_start + off)
            self._cache[name] = self._raw(self.tbytes[name])
        return self._cache[name]

    def tinfo(self, name):
        ne, tt, _ = self.tensor[name]
        return ne, tt

    def row_shape(self, name) -> tuple[int, int]:
        """(n_rows, k) in GGUF row order, plus the number of weights."""
        ne, _tt = self.tinfo(name)
        if len(ne) != 2:
            raise SystemExit(f"{name}: expected rank 2, got {ne}")
        return ne[1], ne[0]

    def blocks(self, name):
        """(n_rows, groups_per_row, block_bytes, row_bytes, raw)."""
        n, k = self.row_shape(name)
        ne, tt = self.tinfo(name)
        block = 28 if tt == T_PTQ1_0 else 34
        gpr = k // 128
        raw = self.payload(name)
        if len(raw) != n * gpr * block:
            raise SystemExit(f"{name}: {len(raw)} != {n}*{gpr}*{block}")
        return n, gpr, block, gpr * block, raw

    def row_fn(self, name):
        n, gpr, block, rb, raw = self.blocks(name)
        return lambda i: raw[i * rb:(i + 1) * rb]


# ---------------------------------------------------------------------------
# Reference decoders -- independent ports of ggml/src/ggml-quants.c
# ---------------------------------------------------------------------------
def dq_pq2_0(raw: bytes) -> np.ndarray:
    qs = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = qs[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    j = np.arange(128)
    q = (qs[:, 2:34][:, j // 4] >> (2 * (j % 4))) & 0x03
    return ((q.astype(np.int32) - 1) * d).reshape(-1)


def dq_ptq1_0(raw: bytes) -> np.ndarray:
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    pow3 = (1, 3, 9, 27, 81, 243)
    g = blk.shape[0]
    d = blk[:, 26:28].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs, qh = blk[:, 0:24], blk[:, 24:26]
    vals = np.empty((g, 120), dtype=np.float32)
    col, j = 0, 0
    for c in (32, 16, 8):                      # c=32 emits nothing: 0 + 32 > 24
        while j + c <= 24:
            for n in range(5):
                prod = (qs[:, j:j + c].astype(np.uint16) * pow3[n]) & 0xFF
                vals[:, col:col + c] = ((prod.astype(np.uint16) * 3) >> 8).astype(np.float32)
                col += c
            j += c
    if col != 120:
        raise SystemExit(f"ptq1_0 qs walk emitted {col}, expected 120")
    tail = np.empty((g, 8), dtype=np.float32)
    col = 0
    for n in range(4):
        prod = (qh.astype(np.uint16) * pow3[n]) & 0xFF
        for h in range(2):
            tail[:, col] = ((prod[:, h].astype(np.uint16) * 3) >> 8).astype(np.float32)
            col += 1
    out = np.concatenate([vals, tail], axis=1)
    return ((out - 1.0) * d[:, None]).reshape(-1)


DEQUANT = {T_PQ2_0: dq_pq2_0, T_PTQ1_0: dq_ptq1_0}


def read_direct(g: Gguf, name: str) -> np.ndarray:
    ne, tt = g.tinfo(name)
    raw = g.payload(name)
    if tt == T_F32:
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif tt == T_BF16:
        a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        raise SystemExit(f"{name}: type {tt} is not F32/BF16")
    return a.reshape(tuple(reversed(ne))) if len(ne) > 1 else a.astype(np.float32)


# ---------------------------------------------------------------------------
# Ternary plane assembly, byte for byte, and its inverse
# ---------------------------------------------------------------------------
def block_to_planes(fmt, blk):
    if fmt == "PTQ1_0_G128":
        return blk[0:24], blk[24:26], blk[26:28]
    if fmt == "PQ2_0_G128":
        return blk[2:34], b"", blk[0:2]
    raise ValueError(fmt)


def planes_to_block(fmt, base, high, scale):
    return base + high + scale if fmt == "PTQ1_0_G128" else scale + base


def assemble_ternary(fmt, shape, row_fn) -> bytes:
    """row_fn(i) -> the raw GGML block bytes for destination row i."""
    n, k = shape
    geo = row_split_geometry(fmt, shape)
    gpr, bb, hb = geo.groups_per_row, geo.base_bytes_per_group, geo.high_bytes_per_group
    block = bb + hb + 2
    out = bytearray(geo.payload_bytes)
    for i in range(n):
        row = row_fn(i)
        if len(row) != gpr * block:
            raise SystemExit(f"row {i}: {len(row)} != {gpr * block}")
        for sel in range(gpr):
            base, high, scale = block_to_planes(fmt, row[sel * block:(sel + 1) * block])
            o = geo.base_offset + i * geo.base_row_bytes + sel * bb
            out[o:o + bb] = base
            if hb:
                o = geo.high_offset + i * geo.high_row_bytes + sel * hb
                out[o:o + hb] = high
            o = geo.scale_offset + i * geo.scale_row_bytes + sel * 2
            out[o:o + 2] = scale
    return bytes(out)


def disassemble_ternary(fmt, shape, payload: bytes) -> bytes:
    """Inverse of assemble_ternary: rebuild the contiguous GGML block stream."""
    n, k = shape
    geo = row_split_geometry(fmt, shape)
    gpr, bb, hb = geo.groups_per_row, geo.base_bytes_per_group, geo.high_bytes_per_group
    out = bytearray()
    for i in range(n):
        for sel in range(gpr):
            o = geo.base_offset + i * geo.base_row_bytes + sel * bb
            base = payload[o:o + bb]
            high = b""
            if hb:
                o = geo.high_offset + i * geo.high_row_bytes + sel * hb
                high = payload[o:o + hb]
            o = geo.scale_offset + i * geo.scale_row_bytes + sel * 2
            out += planes_to_block(fmt, base, high, payload[o:o + 2])
    return bytes(out)


# ---------------------------------------------------------------------------
# Non-ternary transforms
# ---------------------------------------------------------------------------
def tiled_to_grouped(t: np.ndarray, groups: int = 3) -> np.ndarray:
    """GGUF TILED (3,16) -> ninfer GROUPED (16,3) on the leading 48-head axis."""
    shape = t.shape
    return (t.reshape(groups, shape[0] // groups, *shape[1:])
             .transpose(1, 0, *range(2, len(shape) + 1))
             .reshape(shape))


def bf16_payload(a: np.ndarray) -> bytes:
    import torch
    return encode_direct(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                         .to(torch.bfloat16), "BF16")


def fp32_payload(a) -> bytes:
    import torch
    return encode_direct(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), "FP32")


def i32_payload(a) -> bytes:
    import torch
    return encode_direct(torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)), "I32")


# ---------------------------------------------------------------------------
# Inventory (built from the template, never assumed)
# ---------------------------------------------------------------------------
def load_template():
    with open(TEMPLATE, "rb") as f:
        f.seek(16)
        blob = f.read(8 << 20)
    obj, _ = json.JSONDecoder().raw_decode(blob.decode("utf-8", "replace"))
    return obj["identity"], obj["objects"]


def layer_kind(objs) -> dict[int, str]:
    kind: dict[int, str] = {}
    for o in objs:
        p = o["name"].split("/")
        if len(p) > 3 and p[0] == "text" and p[1] == "layers":
            l = int(p[2])
            kind.setdefault(l, "attn" if p[3] == "attention" else "gdn")
    return kind


# Target formats for the NEW artifact.  The template's own formats describe the
# groupwise-int artifact and must NOT be reused: the packer replaces all 402 ternary
# matrices and re-encodes the norms, so the writer would otherwise validate every produced
# payload against the wrong geometry (CHECK 3 caught exactly this).
TERNARY_SUFFIXES = frozenset((
    "mlp/gate_up", "mlp/down",
    "attention/query_key", "attention/gate_value", "attention/output",
    "gdn/query_key", "gdn/value_z", "gdn/output",
))
BF16_SUFFIXES = frozenset((
    "input_norm", "post_attention_norm",
    "attention/query_norm", "attention/key_norm",
    "gdn/norm", "gdn/convolution", "gdn/a_projection", "gdn/b_projection",
))
FP32_SUFFIXES = frozenset(("gdn/a_log", "gdn/dt_bias"))

SIGN_OBJECTS = (
    ("text/hadamard_signs", (28672,), "FP32"),
    ("text/hadamard_widths", (3,), "I32"),
)


def build_specs(p):
    """Ordered specs for the NEW artifact.

    Template objects keep their order, shapes and borrowed formats, but every produced
    tensor takes its TARGET format; the two Hadamard sign-table objects are appended.
    """
    specs = []
    for o in p.objects:
        if o["kind"] == "tensor":
            fmt, layout = p.target_format(o["name"])
            specs.append(TensorSpec(o["name"], tuple(o["shape"]), fmt, layout))
        else:
            specs.append(ResourceSpec(o["name"], o["encoding"], o["bytes"]))
    for name, shape, fmt in SIGN_OBJECTS:
        specs.append(TensorSpec(name, shape, fmt, "contiguous-le-v1"))
    return specs


# ---------------------------------------------------------------------------
class Packer:
    def __init__(self, g: Gguf):
        self.g = g
        raw_identity, self.objects = load_template()
        self.identity = ArtifactIdentity(raw_identity["model_id"], raw_identity["weights_id"])
        self._by_name = {o["name"]: o for o in self.objects}
        self.kind = layer_kind(self.objects)
        self._signs = None

    # -- helpers ---------------------------------------------------------
    def rel(self, name: str, l: int) -> str:
        return name.replace("{l}", str(l))

    def fmt_of(self, gguf_name: str) -> str:
        _ne, tt = self.g.tinfo(gguf_name)
        if tt not in FMT:
            raise SystemExit(f"{gguf_name}: type {tt} is not ternary")
        return FMT[tt]

    def target_format(self, name: str) -> tuple[str, str]:
        """(format, layout) this object carries in the NEW artifact."""
        if name == "text/hadamard_signs":
            return "FP32", "contiguous-le-v1"
        if name == "text/hadamard_widths":
            return "I32", "contiguous-le-v1"
        if name == "text/token_embedding":
            return self.fmt_of("token_embd.weight"), "row-split-k128-v1"
        if name == "text/output_head":
            return self.fmt_of("output.weight"), "row-split-k128-v1"
        if name == "text/final_norm":
            return "BF16", "contiguous-le-v1"

        parts = name.split("/")
        if len(parts) > 3 and parts[0] == "text" and parts[1] == "layers":
            suffix = "/".join(parts[3:])
            pre = f"blk.{int(parts[2])}."
            if suffix in TERNARY_SUFFIXES:
                if suffix == "mlp/gate_up":
                    return self.fmt_of(pre + "ffn_gate.weight"), "row-split-k128-v1"
                if suffix == "mlp/down":
                    return self.fmt_of(pre + "ffn_down.weight"), "row-split-k128-v1"
                if suffix in ("attention/query_key", "attention/gate_value"):
                    return self.fmt_of(pre + "attn_q.weight"), "row-split-k128-v1"
                if suffix == "attention/output":
                    return self.fmt_of(pre + "attn_output.weight"), "row-split-k128-v1"
                if suffix in ("gdn/query_key", "gdn/value_z"):
                    return self.fmt_of(pre + "attn_qkv.weight"), "row-split-k128-v1"
                if suffix == "gdn/output":
                    return self.fmt_of(pre + "ssm_out.weight"), "row-split-k128-v1"
            if suffix in BF16_SUFFIXES:
                return "BF16", "contiguous-le-v1"
            if suffix in FP32_SUFFIXES:
                return "FP32", "contiguous-le-v1"

        # everything else (mtp/*, vision/*, draft_head*) is BORROWED: keep the template's
        o = self._by_name.get(name)
        if o is None:
            raise SystemExit(f"no template object and no producer for {name}")
        return o["format"], o["layout"]

    def fused(self, entries, shape) -> bytes:
        """entries: list of (gguf_name, row_index); all sources must share one format."""
        fmts = {self.fmt_of(n) for n, _ in entries}
        if len(fmts) != 1:
            raise SystemExit(f"fused tensor mixes formats: {fmts}")
        fmt = fmts.pop()
        srcs = {}
        for n, _ in entries:
            if n not in srcs:
                nrows, _gpr, _blk, rb, raw = self.g.blocks(n)
                srcs[n] = (rb, raw)

        def row_fn(i):
            n, r = entries[i]
            rb, raw = srcs[n]
            return raw[r * rb:(r + 1) * rb]

        return assemble_ternary(fmt, shape, row_fn)

    def direct_ternary(self, gguf_name: str, shape) -> bytes:
        return assemble_ternary(self.fmt_of(gguf_name), shape, self.g.row_fn(gguf_name))

    def signs(self):
        if self._signs is None:
            vals = self.g.kv["prism.hadamard.sign_values"]
            widths = list(self.g.kv["prism.hadamard.sign_widths"])
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size != 28672 or not np.all(np.abs(arr) == 1.0):
                raise SystemExit("sign table is not 28672 strictly-+-1 values")
            if widths != SIGN_WIDTHS or sum(widths) != arr.size:
                raise SystemExit(f"sign_widths mismatch: {widths}")
            self._signs = (arr, widths)
        return self._signs

    @staticmethod
    def payload_bytes(spec) -> int | None:
        """Expected payload size for a template object spec, or None when not computable."""
        if spec["kind"] != "tensor":
            return None
        layout = spec.get("layout")
        if layout == "row-split-k128-v1":
            return row_split_geometry(spec["format"], tuple(spec["shape"])).payload_bytes
        if layout == "contiguous-le-v1":
            wb = {"BF16": 2, "FP32": 4, "I32": 4}[spec["format"]]
            return int(np.prod(spec["shape"])) * wb
        return None

    # -- producers -------------------------------------------------------
    def produce(self, name: str):
        """Return the payload bytes for a text/* object, or None to borrow it."""
        p = name.split("/")

        if name == "text/hadamard_signs":
            return fp32_payload(self.signs()[0])
        if name == "text/hadamard_widths":
            return i32_payload(self.signs()[1])

        if name == "text/token_embedding":
            return self.direct_ternary("token_embd.weight", (248320, HIDDEN))
        if name == "text/output_head":
            return self.direct_ternary("output.weight", (248320, HIDDEN))
        if name == "text/final_norm":
            return bf16_payload(read_direct(self.g, "output_norm.weight") - 1.0)

        if len(p) > 3 and p[0] == "text" and p[1] == "layers":
            l = int(p[2])
            suffix = "/".join(p[3:])
            g = self.g
            pre = f"blk.{l}."

            if suffix == "input_norm":
                return bf16_payload(read_direct(g, pre + "attn_norm.weight") - 1.0)
            if suffix == "post_attention_norm":
                return bf16_payload(read_direct(g, pre + "post_attention_norm.weight") - 1.0)
            if suffix == "mlp/gate_up":
                ng, _ = g.row_shape(pre + "ffn_gate.weight")
                nu, _ = g.row_shape(pre + "ffn_up.weight")
                ent = [(pre + "ffn_gate.weight", i) for i in range(ng)]
                ent += [(pre + "ffn_up.weight", i) for i in range(nu)]
                return self.fused(ent, (ng + nu, HIDDEN))
            if suffix == "mlp/down":
                return self.direct_ternary(pre + "ffn_down.weight", (5120, 17408))

            if suffix.startswith("attention/"):
                sub = suffix.split("/", 1)[1]
                if sub == "query_key":
                    q = self.deinterleave(pre + "attn_q.weight", 0, (6144, HIDDEN))
                    nk, _ = g.row_shape(pre + "attn_k.weight")
                    return self.concat_streams(q, 6144, pre + "attn_k.weight", nk,
                                               (6144 + nk, HIDDEN))
                if sub == "gate_value":
                    gt = self.deinterleave(pre + "attn_q.weight", 1, (6144, HIDDEN))
                    nv, _ = g.row_shape(pre + "attn_v.weight")
                    return self.concat_streams(gt, 6144, pre + "attn_v.weight", nv,
                                               (6144 + nv, HIDDEN))
                if sub == "output":
                    return self.direct_ternary(pre + "attn_output.weight", (5120, 6144))
                if sub == "query_norm":
                    return bf16_payload(read_direct(g, pre + "attn_q_norm.weight") - 1.0)
                if sub == "key_norm":
                    return bf16_payload(read_direct(g, pre + "attn_k_norm.weight") - 1.0)
                raise SystemExit(f"unmapped attention object {name}")

            if suffix.startswith("gdn/"):
                sub = suffix.split("/", 1)[1]
                if sub == "query_key":
                    ent = [(pre + "attn_qkv.weight", i) for i in range(QK_ROWS)]
                    return self.fused(ent, (QK_ROWS, HIDDEN))
                if sub == "value_z":
                    return self.gdn_value_z(l)
                if sub == "output":
                    return self.direct_ternary(pre + "ssm_out.weight", (5120, V_ROWS))
                if sub == "convolution":
                    return bf16_payload(self.conv1d(l))
                if sub == "norm":
                    return bf16_payload(read_direct(g, pre + "ssm_norm.weight"))  # RAW
                if sub == "a_projection":
                    return bf16_payload(tiled_to_grouped(read_direct(g, pre + "ssm_alpha.weight")))
                if sub == "b_projection":
                    return bf16_payload(tiled_to_grouped(read_direct(g, pre + "ssm_beta.weight")))
                if sub == "a_log":
                    a = read_direct(g, pre + "ssm_a")
                    return fp32_payload(tiled_to_grouped(np.log(-a.astype(np.float64))
                                                         .astype(np.float32)))
                if sub == "dt_bias":
                    return fp32_payload(tiled_to_grouped(read_direct(g, pre + "ssm_dt.bias")))
                raise SystemExit(f"unmapped gdn object {name}")

            raise SystemExit(f"unmapped layer object {name}")

        return None   # borrow

    # -- GDN / attention fusion helpers ----------------------------------
    def deinterleave(self, gguf_name: str, parity: int, shape):
        """query (parity 0) or output-gate (parity 1): every other 256-row chunk."""
        n, k = self.g.row_shape(gguf_name)
        if n % 512 != 0:
            raise SystemExit(f"{gguf_name}: {n} rows is not a multiple of 512")
        heads = n // 512
        fmt = self.fmt_of(gguf_name)
        nrows, gpr, block, rb, raw = self.g.blocks(gguf_name)
        ent = [(gguf_name, (2 * h + parity) * 256 + r)
               for h in range(heads) for r in range(256)]
        return assemble_ternary(fmt, (len(ent), k),
                                lambda i: raw[ent[i][1] * rb:(ent[i][1] + 1) * rb])

    def concat_streams(self, first: bytes, first_rows: int, gname: str, gn: int, shape):
        """Concatenate a pre-assembled ternary payload with another tensor's rows."""
        n, k = shape
        fmt = self.fmt_of(gname)
        geo = row_split_geometry(fmt, shape)
        gpr, bb, hb = geo.groups_per_row, geo.base_bytes_per_group, geo.high_bytes_per_group
        block = bb + hb + 2
        _n2, gpr2, block2, rb, raw = self.g.blocks(gname)
        if gpr2 != gpr or block2 != block:
            raise SystemExit("concat_streams: block geometry differs")
        geo_a = row_split_geometry(fmt, (first_rows, k))

        def row_fn(i):
            if i < first_rows:
                out = bytearray()
                for sel in range(gpr):
                    o = geo_a.base_offset + i * geo_a.base_row_bytes + sel * bb
                    base = first[o:o + bb]
                    high = b""
                    if hb:
                        o = geo_a.high_offset + i * geo_a.high_row_bytes + sel * hb
                        high = first[o:o + hb]
                    o = geo_a.scale_offset + i * geo_a.scale_row_bytes + sel * 2
                    out += planes_to_block(fmt, base, high, first[o:o + 2])
                return bytes(out)
            j = i - first_rows
            return raw[j * rb:(j + 1) * rb]

        return assemble_ternary(fmt, shape, row_fn)

    def gdn_value_z(self, l: int) -> bytes:
        """concat(tiled_to_grouped(attn_qkv[4096:10240]), tiled_to_grouped(attn_gate[0:6144]))."""
        pre = f"blk.{l}."
        qkv, gate = pre + "attn_qkv.weight", pre + "attn_gate.weight"
        f1, f2 = self.fmt_of(qkv), self.fmt_of(gate)
        if f1 != f2:
            raise SystemExit("value_z: mixed formats")
        fmt = f1

        def perm48(i):
            # tiled (3,16) -> grouped (16,3) for a 48-entry HEAD axis:
            # destination head g*3+t comes from source head t*16+g
            return (i % 3) * 16 + (i // 3)

        def perm_row(i):
            # perm48() permutes a 48-entry axis, but this tensor's V/z rows are 48 heads of
            # V_HEAD_DIM rows each, so the permutation has to be applied at HEAD granularity.
            # Applying perm48() straight to the 6144-row index is not a bijection: perm48(48) and
            # perm48(1) are both 16, so rows repeat and source rows are dropped while every size,
            # row count and byte total stays exactly right -- which is why byte accounting, the
            # row-count cross-check and the payload round-trip all stayed green over it.
            head, inner = divmod(i, V_HEAD_DIM)
            return perm48(head) * V_HEAD_DIM + inner

        shape = (12288, HIDDEN)
        geo = row_split_geometry(fmt, shape)
        gpr, bb, hb = geo.groups_per_row, geo.base_bytes_per_group, geo.high_bytes_per_group
        block = bb + hb + 2
        gpr1 = self.g.blocks(qkv)[1]
        gpr2_ = self.g.blocks(gate)[1]
        if gpr1 != gpr or gpr2_ != gpr:
            raise SystemExit("value_z: groups_per_row mismatch")
        rb1 = self.g.blocks(qkv)[3]
        rb2 = self.g.blocks(gate)[3]
        raw1, raw2 = self.g.payload(qkv), self.g.payload(gate)

        def row_fn(i):
            if i < V_ROWS:                      # attn_qkv rows 4096..10240, tiled->grouped
                src = QK_ROWS + perm_row(i)
                return raw1[src * rb1:(src + 1) * rb1]
            src = perm_row(i - V_ROWS)           # attn_gate rows 0..6144, tiled->grouped
            return raw2[src * rb2:(src + 1) * rb2]

        return assemble_ternary(fmt, shape, row_fn)

    def conv1d(self, l: int) -> np.ndarray:
        """(4,10240) <- ssm_conv1d: keep channels 0:4096, reorder the V channels, transpose."""
        pre = f"blk.{l}."
        t = read_direct(self.g, pre + "ssm_conv1d.weight")       # (10240, 4)
        if t.shape != (10240, 4):
            raise SystemExit(f"conv1d shape {t.shape} != (10240, 4)")
        head = t[0:QK_ROWS]
        v = t[QK_ROWS:QK_ROWS + V_ROWS].reshape(V_HEADS, V_HEAD_DIM, t.shape[1])
        v = (v.reshape(3, V_HEADS // 3, V_HEAD_DIM, t.shape[1])
              .transpose(1, 0, 2, 3)
              .reshape(V_ROWS, t.shape[1]))
        return np.concatenate([head, v], axis=0).T                # (4, 10240)


# ---------------------------------------------------------------------------
def mode_check(g: Gguf) -> int:
    p = Packer(g)
    print("=" * 78)
    print("CHECK 1  geometry over every ternary shape present in the GGUF")
    print("=" * 78)
    combos = {}
    for name, (ne, tt, _off) in g.tensor.items():
        if tt in FMT:
            combos.setdefault((FMT[tt], ne[1], ne[0]), []).append(name)
    for (fmt, n, k), names in sorted(combos.items()):
        geo = row_split_geometry(fmt, (n, k))
        src = n * (k // 128) * (28 if fmt == "PTQ1_0_G128" else 34)
        pad = geo.payload_bytes - src
        print(f"  {fmt:<14} n={n:<7} k={k:<6} groups/row={geo.groups_per_row:<4} "
              f"src={src:>13,} payload={geo.payload_bytes:>13,} pad={pad:>5} "
              f"({len(names)} tensors)")
    print(f"  distinct (format,shape) combos: {len(combos)}")

    print("\n" + "=" * 78)
    print("CHECK 2  byte round trip + decode equality on real tensors")
    print("=" * 78)
    sample = ["blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.3.ffn_down.weight",
              "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "token_embd.weight"]
    ok = True
    for name in sample:
        if name not in g.tensor:
            print(f"  {name}: absent, skipped")
            continue
        ne, tt = g.tinfo(name)
        n, k = ne[1], ne[0]
        fmt = FMT[tt]
        payload = p.direct_ternary(name, (n, k))
        back = disassemble_ternary(fmt, (n, k), payload)
        src = g.payload(name)
        byte_ok = back == src
        a, b = DEQUANT[tt](src), DEQUANT[tt](back)
        dec_ok = bool(np.array_equal(a, b))
        zeros = float(np.mean(a == 0.0))

        # OFFSET SELF-CHECK -- the decisive one.  bytes_equal only proves the assembly is
        # self-consistent; reading a wrong file offset would ALSO give bytes_equal, because
        # the same wrong bytes go in and come out.  So validate the CONTENT: a genuine group
        # scale is a tight positive cluster (few distinct high bytes, all finite, none
        # negative), whereas a mis-placed read draws scale words from arbitrary positions and
        # looks like a uniform uint16 sample with many NaN/inf and negative values.
        # 组大小与 scale 在组内的位置都随格式而异，不能硬编码：
        #   PQ2_0_G128: 34 B/组，scale 在组首 [0:2]   (见 dq_pq2_0)
        #   PTQ1_0_G128: 28 B/组，scale 在组尾 [26:28] (见 dq_ptq1_0)
        # 原版在此硬编码 reshape(-1, 34) + [0:2]，因此 PTQ1_0 会抛
        #   ValueError: cannot reshape array of size 13762560 into shape (34)
        # （13762560 = 491520*28，能被 28 整除、不能被 34 整除）。
        # 这只是自检逻辑的缺陷，不影响 direct_ternary 产出的 payload 本身。
        if fmt == "PTQ1_0_G128":
            arr = np.frombuffer(src, dtype=np.uint8).reshape(-1, 28)
            sc = arr[:, 26:28]
        else:
            arr = np.frombuffer(src, dtype=np.uint8).reshape(-1, 34)
            sc = arr[:, 0:2]
        words = sc.copy().view(np.uint16).reshape(-1)
        hi_distinct = int(np.unique((words >> 8).astype(np.uint8)).size)
        d = sc.copy().view(np.float16).astype(np.float32).reshape(-1)
        fin = np.isfinite(d)
        n_nonfinite = int(np.sum(~fin))
        n_neg = int(np.sum(fin & (d < 0)))
        med = float(np.median(d[fin])) if fin.any() else float("nan")
        plaus = (n_nonfinite == 0 and n_neg == 0 and hi_distinct <= 64 and 0.0 < med < 1.0)

        print(f"  {name:<28} {fmt:<14} n={n:<6} k={k:<6} bytes_equal={byte_ok} "
              f"decode_equal={dec_ok} zero_share={zeros:.4f}")
        print(f"  {'':<28} scale: hi_distinct={hi_distinct:<4} nonfinite={n_nonfinite} "
              f"negative={n_neg} median={med:.5f}  "
              f"{'PLAUSIBLE' if plaus else 'IMPLAUSIBLE -- offset bug?'}")
        ok &= byte_ok and dec_ok and plaus

    print("\n" + "=" * 78)
    print("CHECK 3  producer smoke test (shapes + sizes only, no artifact written)")
    print("=" * 78)
    for name in ["text/layers/3/attention/query_key", "text/layers/3/attention/gate_value",
                 "text/layers/3/mlp/gate_up", "text/layers/3/mlp/down",
                 "text/layers/3/attention/output", "text/layers/0/gdn/value_z",
                 "text/layers/0/gdn/query_key", "text/layers/0/gdn/output",
                 "text/token_embedding", "text/output_head", "text/final_norm",
                 "text/hadamard_signs", "text/hadamard_widths"]:
        base = p._by_name.get(name)
        if base is not None:
            shape = base["shape"]
        else:
            shape = dict((n, list(s)) for n, s, _f in SIGN_OBJECTS).get(name)
            if shape is None:
                print(f"  {name}: neither in template nor a sign object, skipped")
                continue
        fmt, layout = p.target_format(name)
        spec = {"kind": "tensor", "shape": shape, "format": fmt, "layout": layout}
        data = p.produce(name)
        if data is None:
            print(f"  {name:<44} {fmt:<14} {'(borrowed)':>13}  SKIP")
            continue
        need = p.payload_bytes(spec)
        got = len(data)
        flag = "OK" if (need is None or got == need) else f"MISMATCH want {need}"
        print(f"  {name:<44} {fmt:<14} {got:>13,}  {flag}")
        if need is not None and got != need:
            ok = False

    print("\nRESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def mode_build(g: Gguf, out: str, only_layer: int | None = None) -> int:
    p = Packer(g)
    keep = None
    if only_layer is not None:
        names = {n for n, _s, _f in SIGN_OBJECTS}      # the sign objects are appended later
        for o in p.objects:
            n = o["name"]
            if n.startswith("frontend/") or n.startswith("text/hadamard_"):
                names.add(n)
            elif n.startswith(f"text/layers/{only_layer}/"):
                names.add(n)
        keep = names

    specs = []
    chosen = []
    for o in p.objects + [
        {"name": "text/hadamard_signs", "kind": "tensor", "shape": [28672],
         "format": "FP32", "layout": "contiguous-le-v1"},
        {"name": "text/hadamard_widths", "kind": "tensor", "shape": [3],
         "format": "I32", "layout": "contiguous-le-v1"},
    ]:
        if keep is not None and o["name"] not in keep:
            continue
        if o["kind"] == "tensor":
            fmt, layout = p.target_format(o["name"])
            specs.append(TensorSpec(o["name"], tuple(o["shape"]), fmt, layout))
        else:
            specs.append(ResourceSpec(o["name"], o["encoding"], o["bytes"]))
        chosen.append(o)

    out_path = Path(out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    borrowed = Counter()
    borrowed_bytes = 0
    produced_bytes = 0
    with Artifact.open(TEMPLATE) as tpl, \
            ArtifactWriter(out_path, p.identity, specs) as w:
        for o in chosen:
            name = o["name"]
            if name.startswith("text/") and not name.startswith("text/vision"):
                data = None if name.startswith(("text/draft_head",)) else p.produce(name)
                if data is not None:
                    w.write(name, data)
                    produced_bytes += len(data)
                    continue
            obj = tpl.find(name)
            if obj is None:
                raise SystemExit(f"template has no object {name} to borrow")
            mv = tpl.payload(obj)
            borrowed_bytes += len(mv)
            w.write(name, mv)
            del mv          # release the mmap view, else Artifact.close() raises BufferError
            borrowed[name.split("/")[0]] += 1

    size = out_path.stat().st_size
    print(f"\nwrote {out_path}")
    print(f"  total file        : {size:>15,} B = {size / 2**30:.3f} GiB")
    print(f"  text part produced: {produced_bytes:>15,} B = {produced_bytes / 2**30:.3f} GiB")
    print(f"  borrowed payloads : {borrowed_bytes:>15,} B = {borrowed_bytes / 2**30:.3f} GiB "
          f"{dict(borrowed)}")
    print(f"  objects           : {len(chosen)}")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    g = Gguf(GGUF)
    if args[0] == "check":
        return mode_check(g)
    if args[0] == "layer3":
        return mode_build(g, args[1], only_layer=3)
    if args[0] == "build":
        return mode_build(g, args[1])
    print(f"unknown mode {args[0]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
