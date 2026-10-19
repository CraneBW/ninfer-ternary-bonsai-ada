"""Byte-exact TP2 sharder for Qwen3.8-27B v3 mixed-GGUF NInfer artifacts.

Reads the 32-byte header and JSON independently of the (potentially incomplete)
weight payload. Planning never reads or invents missing weight bytes. Writing
requires the complete single-file artifact and valid physical input-column map.
The result is intentionally *not* loadable by the current single-GPU engine.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import BinaryIO, Iterator


MAGIC = b"NINFER\0\x03"
HEADER_BYTES = 32
PAYLOAD_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 256
CHUNK_BYTES = 4 * 1024 * 1024

# As defined by src/artifact/storage_layouts.cpp (GGUF logical elements, stored
# bytes per independent GGML block). All column slicing uses groups of 256
# logical elements; supported 32/64-element formats copy complete block groups.
GGUF_BLOCKS = {
    "gguf_q8_0": (32, 34),
    "gguf_q2_k": (256, 84),
    "gguf_q3_k": (256, 110),
    "gguf_q4_k": (256, 144),
    "gguf_q5_k": (256, 176),
    "gguf_q6_k": (256, 210),
    "gguf_iq2_xxs": (256, 66),
    "gguf_iq2_xs": (256, 74),
    "gguf_iq2_s": (256, 82),
    "gguf_iq3_xxs": (256, 98),
    "gguf_iq3_s": (256, 110),
    "gguf_iq1_s": (256, 50),
    "gguf_iq1_m": (256, 56),
    "gguf_iq4_nl": (32, 18),
    "gguf_iq4_xs": (256, 136),
    "gguf_q4_0": (32, 18),
    "gguf_q5_0": (32, 22),
    "gguf_q2_0": (64, 18),
}
DIRECT_WORD_BYTES = {"bf16": 2, "fp32": 4, "int32": 4}

SPLIT_ROWS = {
    "attention/query", "attention/gate", "attention/key", "attention/value",
    "gdn/query", "gdn/key", "gdn/value", "gdn/z",
    "gdn/a_projection", "gdn/b_projection", "gdn/a_log", "gdn/dt_bias",
    "mlp/gate", "mlp/up",
}
SPLIT_COLUMNS = {"attention/output", "gdn/output", "mlp/down"}
CONVOLUTION = "gdn/convolution"
REPLICATED_LAYER = {
    "input_norm", "post_attention_norm", "attention/query_norm",
    "attention/key_norm", "gdn/norm",
}
LAYER_PATTERN = re.compile(r"^(?:text|mtp)/layers/\d+/(.+)$")


class ShardError(ValueError):
    """An artifact cannot be safely sharded or its payload is incomplete."""


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def positive_int(value: object, label: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < 0 or (not zero and value == 0):
        raise ShardError(f"{label}: expected {'nonnegative' if zero else 'positive'} integer")
    return value


def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ShardError(f"duplicate v3 JSON key {key!r}")
        result[key] = value
    return result


def read_directory(path: str | Path) -> tuple[dict, bytes, int, int]:
    """Return v3 JSON, source UUID, payload start and physical file length.

    The only I/O here is the header and complete declared JSON directory; the
    caller may supply a header-only prefix or a partially downloaded file.
    """
    source = Path(path)
    size = source.stat().st_size
    with source.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:8] != MAGIC:
            raise ShardError("source must have a complete 32-byte NInfer v3 header")
        directory_bytes = struct.unpack_from("<Q", header, 8)[0]
        if not 0 < directory_bytes <= 64 * 1024 * 1024:
            raise ShardError("invalid v3 directory length")
        raw = handle.read(directory_bytes)
        if len(raw) != directory_bytes:
            raise ShardError("source has an incomplete v3 JSON directory")
    try:
        root = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ShardError(f"invalid v3 JSON directory: {exc}") from exc
    if type(root) is not dict:
        raise ShardError("v3 directory root must be an object")
    payload_start = align_up(HEADER_BYTES + directory_bytes, PAYLOAD_ALIGNMENT)
    # A header-only prefix need not include the zero padding between its JSON
    # and the first payload object. Planning never dereferences that region.
    return root, header[16:32], payload_start, size


def object_elements(obj: dict) -> int:
    elements = 1
    for size in obj.get("shape", []):
        elements *= positive_int(size, f"{obj['id']} shape dimension")
    return elements


def stored_row_bytes(obj: dict) -> int:
    shape = obj.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ShardError(f"{obj['id']}: split matrix requires rank two")
    columns = positive_int(shape[1], f"{obj['id']} width")
    fmt, layout = obj.get("format"), obj.get("layout")
    if layout == "gguf_blocks_v1" and fmt in GGUF_BLOCKS:
        elements, nbytes = GGUF_BLOCKS[fmt]
        if columns % elements:
            raise ShardError(f"{obj['id']}: incomplete GGUF physical block in row")
        return columns // elements * nbytes
    if layout == "contiguous_le_v1" and fmt in DIRECT_WORD_BYTES:
        return columns * DIRECT_WORD_BYTES[fmt]
    raise ShardError(f"{obj['id']}: split layout/format {layout}/{fmt} unsupported")


def binding_parts(binding: dict, name: str) -> list[dict]:
    if type(binding) is not dict:
        raise ShardError(f"{name}: invalid binding")
    if set(binding) == {"object"} and isinstance(binding["object"], str):
        return [binding]
    if (set(binding) != {"parts"} or not isinstance(binding["parts"], list)
            or len(binding["parts"]) != 1):
        raise ShardError(f"{name}: only whole-object or single-part bindings supported")
    part = binding["parts"][0]
    if (type(part) is not dict or set(part) != {"object", "range"}
            or not isinstance(part["object"], str) or not isinstance(part["range"], list)
            or len(part["range"]) != 2):
        raise ShardError(f"{name}: invalid logical-element part range")
    return [part]


def role_mode(name: str) -> str | None:
    match = LAYER_PATTERN.fullmatch(name)
    if not match:
        return None
    suffix = match.group(1)
    if suffix in SPLIT_ROWS:
        return "rows"
    if suffix in SPLIT_COLUMNS:
        return "mapped_cols" if suffix == "gdn/output" else "cols"
    if suffix == CONVOLUTION:
        return "conv_cols"
    if suffix in REPLICATED_LAYER:
        return None
    raise ShardError(f"unknown layer tensor role {name!r}: refusing implicit replication")


@dataclass(frozen=True)
class Operation:
    source_id: str
    mode: str
    shape: tuple[int, ...]
    source_offset: int
    source_bytes: int
    output_bytes: int
    # All groups are ranges of physical rows (not serialized bytes).
    row_groups: tuple[tuple[int, int], ...] = ()
    # Ranges of logical physical columns, in their original segment order.
    column_segments: tuple[int, ...] = ()


@dataclass
class ShardPlan:
    source: Path
    source_uuid: bytes
    source_payload_start: int
    source_file_size: int
    expected_file_size: int
    operations: list[Operation]
    directories: tuple[dict, dict]
    auxiliary_id: str | None

    @property
    def complete(self) -> bool:
        return self.source_file_size == self.expected_file_size

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for op in self.operations:
            counts[op.mode] = counts.get(op.mode, 0) + 1
        return {
            "format": "ninfer-v3-gguf-tp2-plan-v1",
            "source": str(self.source),
            "source_bytes_present": self.source_file_size,
            "source_bytes_expected": self.expected_file_size,
            "source_complete_by_length": self.complete,
            "physical_objects": len(self.operations),
            "partitions": counts,
            "input_columns": "requires full source payload" if self.auxiliary_id else None,
            "rank_payload_bytes": [d["files"][0]["payload_bytes"] for d in self.directories],
            "rank_metadata_names": [d["metadata"]["name"] for d in self.directories],
            "native_single_rank_loader": "rejected by rank-specific metadata.name",
        }


def make_plan(path: str | Path) -> ShardPlan:
    root, source_uuid, payload_start, actual_bytes = read_directory(path)
    expected_top = {"components", "objects", "bindings", "uses", "files"}
    if not expected_top.issubset(root) or set(root) - (expected_top | {"metadata", "provenance"}):
        raise ShardError("unsupported v3 root directory schema")
    if (not isinstance(root["components"], dict)
            or not isinstance(root.get("metadata"), dict)
            or root["metadata"].get("name") != "qwen3.8-27b"):
        raise ShardError("expected unsharded v3 qwen3.8-27b metadata")
    text = root["components"].get("text")
    if not isinstance(text, dict) or not isinstance(text.get("config"), dict):
        raise ShardError("expected v3 text component config")
    text_config = text["config"]
    if text_config.get("model_type") != "qwen3_5_text":
        raise ShardError("expected Qwen3.8 text component topology")
    files = root["files"]
    if (not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict)
            or set(files[0]) != {"path", "payload_bytes"} or files[0]["path"] is not None):
        raise ShardError("v3 TP2 sharder currently requires a single embedded payload file")
    payload_bytes = positive_int(files[0]["payload_bytes"], "declared payload size", zero=True)
    expected_size = payload_start + payload_bytes
    if actual_bytes > expected_size:
        raise ShardError("source has trailing bytes beyond its declared v3 payload")
    objects = root["objects"]
    bindings = root["bindings"]
    uses = root["uses"]
    if not isinstance(objects, list) or not objects or not isinstance(bindings, dict) or not isinstance(uses, list):
        raise ShardError("invalid v3 objects/bindings/uses")
    by_id: dict[str, dict] = {}
    source_cursor = 0
    for obj in objects:
        if type(obj) is not dict or not isinstance(obj.get("id"), str) or not obj["id"]:
            raise ShardError("invalid v3 physical object id")
        name = obj["id"]
        if name in by_id:
            raise ShardError(f"duplicate physical object {name}")
        if obj.get("kind") not in {"tensor", "resource"}:
            raise ShardError(f"{name}: unsupported physical kind")
        required = ({"id", "kind", "shape", "format", "layout", "offset", "bytes"}
                    if obj["kind"] == "tensor"
                    else {"id", "kind", "encoding", "offset", "bytes"})
        optional = {"divisors"} if obj["kind"] == "tensor" else set()
        if not required.issubset(obj) or set(obj) - (required | optional):
            raise ShardError(f"{name}: unsupported v3 physical object descriptor")
        offset = positive_int(obj.get("offset"), f"{name} offset", zero=True)
        size = positive_int(obj.get("bytes"), f"{name} byte length", zero=True)
        alignment = TENSOR_ALIGNMENT if obj["kind"] == "tensor" else 1
        if offset < source_cursor or offset % alignment or offset + size > payload_bytes:
            raise ShardError(f"{name}: overlapping, misaligned, or out-of-range byte span")
        if obj["kind"] == "tensor":
            if not isinstance(obj.get("shape"), list) or len(obj["shape"]) > 16:
                raise ShardError(f"{name}: missing tensor shape")
            elements = object_elements(obj)
            fmt, layout = obj.get("format"), obj.get("layout")
            if layout == "gguf_blocks_v1":
                if fmt not in GGUF_BLOCKS or len(obj["shape"]) != 2:
                    raise ShardError(f"{name}: unknown GGUF physical storage")
                block_elements, block_bytes = GGUF_BLOCKS[fmt]
                if obj["shape"][1] % block_elements or size != elements // block_elements * block_bytes:
                    raise ShardError(f"{name}: GGUF physical byte count differs from shape")
            elif layout == "contiguous_le_v1" and fmt in DIRECT_WORD_BYTES:
                if size != elements * DIRECT_WORD_BYTES[fmt]:
                    raise ShardError(f"{name}: contiguous physical byte count differs from shape")
        elif obj["encoding"] not in {"raw_bytes_v1", "raw-bytes-v1"}:
            raise ShardError(f"{name}: unknown v3 resource encoding")
        source_cursor = offset + size
        by_id[name] = obj

    for name, component in root["components"].items():
        if not isinstance(component, dict):
            raise ShardError(f"{name}: invalid v3 component")
        resources = component.get("resources", {})
        if not isinstance(resources, dict):
            raise ShardError(f"{name}: invalid v3 resources")
        for resource_id in resources.values():
            if (not isinstance(resource_id, str) or resource_id not in by_id
                    or by_id[resource_id]["kind"] != "resource"):
                raise ShardError(f"{name}: unresolved physical resource {resource_id!r}")

    # Resolve all bindings before mapping bytes. The v3 range is a *logical
    # element* interval, including for GGUF matrices, never an encoded-byte
    # interval. Only row-aligned physical intervals may be row-sharded.
    references: dict[str, list[tuple[str, dict]]] = {name: [] for name in by_id}
    for name, binding in bindings.items():
        for part in binding_parts(binding, name):
            obj = by_id.get(part["object"])
            if obj is None or obj["kind"] != "tensor":
                raise ShardError(f"{name}: missing referenced tensor {part['object']}")
            if "range" in part:
                begin, end = part["range"]
                begin = positive_int(begin, f"{name} begin", zero=True)
                end = positive_int(end, f"{name} end")
                if begin >= end or end > object_elements(obj):
                    raise ShardError(f"{name}: out-of-range logical-element binding")
            references[part["object"]].append((name, part))

    auxiliary_id: str | None = None
    mapped_uses: set[str] = set()
    for use in uses:
        if type(use) is not dict or use.get("parameter") not in bindings:
            raise ShardError("v3 Use refers to missing parameter")
        auxiliaries = use.get("auxiliaries", {})
        if type(auxiliaries) is not dict:
            raise ShardError("invalid v3 Use auxiliaries")
        if "input_columns" in auxiliaries:
            if role_mode(use["parameter"]) != "mapped_cols":
                raise ShardError("input_columns encountered outside GDN output")
            mapped_uses.add(use["parameter"])
            ref = auxiliaries["input_columns"]
            if type(ref) is not dict or set(ref) != {"object"} or ref["object"] not in by_id:
                raise ShardError("invalid GDN input_columns auxiliary reference")
            if auxiliary_id is None:
                auxiliary_id = ref["object"]
            elif auxiliary_id != ref["object"]:
                raise ShardError("different GDN output mappings cannot share one TP2 column plan")
    mapped_bindings = {name for name in bindings if role_mode(name) == "mapped_cols"}
    if mapped_uses != mapped_bindings:
        raise ShardError("every GDN output must have one input_columns Use mapping")
    if auxiliary_id is not None:
        if auxiliary_id != "auxiliary/000000":
            raise ShardError("unsupported GDN input_columns physical auxiliary id")
        aux = by_id[auxiliary_id]
        if (aux.get("kind") != "tensor" or aux.get("format") != "int32"
                or aux.get("layout") != "contiguous_le_v1"
                or aux.get("shape") != [6144] or aux.get("bytes") != 6144 * 4):
            raise ShardError("GDN input_columns must be contiguous int32[6144]")

    operations: list[Operation] = []
    # For a fused row parent, each binding's pre-shard range maps to its rank's
    # half within the new parent. The map is independent of the rank because
    # both ranks keep the same segment lengths and ordering.
    rewritten_ranges: dict[tuple[str, int, int], tuple[int, int]] = {}
    for obj in objects:
        name = obj["id"]
        roles = references[name]
        modes = {role_mode(role) for role, _part in roles}
        if len(modes) > 1:
            raise ShardError(f"{name}: incompatible fused row/column/shared roles")
        mode = next(iter(modes)) if modes else None
        if name == auxiliary_id:
            if roles:
                raise ShardError("GDN input_columns cannot also be a regular weight binding")
            mode = "auxiliary"
        if mode == "mapped_cols" and auxiliary_id is None:
            raise ShardError(f"{name}: GDN output needs a validated input_columns auxiliary")
        if mode is None:
            operations.append(Operation(name, "copy", tuple(obj.get("shape", [])),
                                        obj["offset"], obj["bytes"], obj["bytes"]))
            continue
        if obj["kind"] != "tensor" or "divisors" in obj:
            raise ShardError(f"{name}: split is unsupported for this physical descriptor")
        shape = tuple(obj["shape"])
        groups: tuple[tuple[int, int], ...] = ()
        segments: tuple[int, ...] = ()
        if mode == "rows":
            if not shape or shape[0] % 2:
                raise ShardError(f"{name}: row split requires an even first dimension")
            if len(shape) == 2:
                row_bytes = stored_row_bytes(obj)
                if row_bytes * shape[0] != obj["bytes"]:
                    raise ShardError(f"{name}: row encoding byte size differs")
                row_elems = shape[1]
            elif (len(shape) == 1 and obj["layout"] == "contiguous_le_v1"
                  and obj["format"] in DIRECT_WORD_BYTES):
                row_elems = 1
                if shape[0] * DIRECT_WORD_BYTES[obj["format"]] != obj["bytes"]:
                    raise ShardError(f"{name}: vector byte size differs")
            else:
                raise ShardError(f"{name}: unsupported row storage")
            physical_groups = set()
            for role, part in roles:
                if "range" not in part:
                    if len(roles) != 1:
                        raise ShardError(f"{name}: whole-object binding mixed with row parts")
                    physical_groups.add((0, shape[0]))
                else:
                    begin, end = part["range"]
                    if begin % row_elems or end % row_elems:
                        raise ShardError(f"{role}: part range must align to logical rows")
                    physical_groups.add((begin // row_elems, end // row_elems))
            groups = tuple(sorted(physical_groups))
            cursor = 0
            local_cursor = 0
            for begin, end in groups:
                if begin != cursor or end <= begin or (end - begin) % 2:
                    raise ShardError(f"{name}: row parts do not cover complete even groups")
                half = (end - begin) // 2
                rewritten_ranges[(name, begin * row_elems, end * row_elems)] = (
                    local_cursor * row_elems, (local_cursor + half) * row_elems)
                local_cursor += half
                cursor = end
            if cursor != shape[0]:
                raise ShardError(f"{name}: fused row parts leave unowned rows")
        elif mode == "auxiliary":
            if shape != (6144,):
                raise ShardError("invalid input_columns auxiliary geometry")
        else:
            if len(shape) != 2 or len(roles) != 1 or "range" in roles[0][1]:
                raise ShardError(f"{name}: column split requires one whole matrix binding")
            if mode == "conv_cols":
                segments = (2048, 2048, 6144)
                if shape[1] != sum(segments):
                    raise ShardError(f"{name}: GDN convolution must be [taps,10240]")
            else:
                segments = (shape[1],)
            if any(segment <= 0 or segment % 2 for segment in segments):
                raise ShardError(f"{name}: column segments must have even widths")
            if obj["layout"] == "gguf_blocks_v1" and obj["format"] in GGUF_BLOCKS:
                if any(segment % 256 for segment in segments):
                    raise ShardError(f"{name}: column cuts must align to 256 logical GGUF elements")
                if mode == "mapped_cols" and shape[1] != 6144:
                    raise ShardError(f"{name}: mapped GDN output must have 6144 physical columns")
            elif obj["layout"] != "contiguous_le_v1" or obj["format"] not in DIRECT_WORD_BYTES:
                raise ShardError(f"{name}: unsupported column-split codec")
            if mode == "mapped_cols" and obj["layout"] != "gguf_blocks_v1":
                raise ShardError(f"{name}: mapped GDN output must use GGUF physical blocks")
            if stored_row_bytes(obj) * shape[0] != obj["bytes"]:
                raise ShardError(f"{name}: physical byte size mismatches column geometry")
        if obj["bytes"] % 2:
            raise ShardError(f"{name}: cannot evenly split encoded bytes")
        operations.append(Operation(name, mode, shape, obj["offset"], obj["bytes"],
                                    obj["bytes"] // 2, groups, segments))

    rank_directories = []
    for rank in (0, 1):
        directory = copy.deepcopy(root)
        directory["metadata"]["name"] = f"qwen3.8-27b-tp2-rank{rank}"
        directory["metadata"]["tensor_parallel"] = {
            "rank": rank, "world_size": 2, "source_model": "qwen3.8-27b",
            "partition": "exact-physical-v3-v1", "requires_native_tp2": True,
        }
        cursor = 0
        for obj, op in zip(directory["objects"], operations):
            cursor = align_up(cursor, TENSOR_ALIGNMENT if obj["kind"] == "tensor" else 1)
            obj["offset"] = cursor
            obj["bytes"] = op.output_bytes
            if op.mode == "rows":
                obj["shape"][0] //= 2
            elif op.mode in {"cols", "conv_cols", "mapped_cols"}:
                obj["shape"][1] //= 2
            elif op.mode == "auxiliary":
                obj["shape"][0] //= 2
            cursor += op.output_bytes
        directory["files"] = [{"path": None, "payload_bytes": cursor}]
        for name, value in directory["bindings"].items():
            for part in binding_parts(value, name):
                if "range" in part:
                    key = (part["object"], *part["range"])
                    if key in rewritten_ranges:
                        part["range"] = list(rewritten_ranges[key])
        rank_directories.append(directory)

    return ShardPlan(Path(path), source_uuid, payload_start, actual_bytes,
                     expected_size, operations, (rank_directories[0], rank_directories[1]),
                     auxiliary_id)


@dataclass(frozen=True)
class PhysicalColumnMap:
    # Physical 256-column block positions, in their original order, per rank.
    blocks: tuple[tuple[int, ...], tuple[int, ...]]
    # Rank-local activation indices corresponding to each selected block.
    columns: tuple[tuple[int, ...], tuple[int, ...]]


def input_column_map(source: BinaryIO, plan: ShardPlan) -> PhysicalColumnMap | None:
    if plan.auxiliary_id is None:
        return None
    op = next((item for item in plan.operations if item.source_id == plan.auxiliary_id), None)
    if op is None or op.mode != "auxiliary" or op.source_bytes != 6144 * 4:
        raise ShardError("missing physical input_columns object")
    source.seek(plan.source_payload_start + op.source_offset)
    raw = source.read(op.source_bytes)
    if len(raw) != op.source_bytes:
        raise ShardError("incomplete input_columns payload: TP2 cannot infer physical block owners")
    values = struct.unpack("<6144i", raw)
    blocks: list[list[int]] = [[], []]
    columns: list[list[int]] = [[], []]
    for block in range(6144 // 256):
        group = values[block * 256:(block + 1) * 256]
        owners = {value // 3072 for value in group if 0 <= value < 6144}
        if len(owners) != 1 or any(not 0 <= value < 6144 for value in group):
            raise ShardError(f"input_columns physical 256-block {block} crosses TP2 rank boundary")
        rank = owners.pop()
        blocks[rank].append(block)
        columns[rank].extend(value - 3072 * rank for value in group)
    for rank in (0, 1):
        if len(blocks[rank]) != 12 or sorted(columns[rank]) != list(range(3072)):
            raise ShardError(f"input_columns rank {rank} must own a permutation of 3072 columns")
    return PhysicalColumnMap((tuple(blocks[0]), tuple(blocks[1])),
                             (tuple(columns[0]), tuple(columns[1])))


def _copy_spans(op: Operation, rank: int,
                column_map: PhysicalColumnMap | None) -> Iterator[tuple[int, int]]:
    """All offsets are relative to source object, lengths are encoded *bytes*."""
    if op.mode == "copy":
        yield 0, op.source_bytes
        return
    if op.mode == "auxiliary":
        return
    shape = op.shape
    if op.mode == "rows":
        row_bytes = op.source_bytes // shape[0]
        for start, stop in op.row_groups:
            half = (stop - start) // 2
            yield (start + rank * half) * row_bytes, half * row_bytes
        return

    if len(shape) != 2:
        raise ShardError(f"{op.source_id}: unsupported column shape")
    rows, columns = shape
    if op.mode == "mapped_cols":
        if column_map is None:
            raise ShardError("mapped GDN output requires complete physical input_columns")
        if columns != 6144 or op.source_bytes % (rows * 24):
            raise ShardError(f"{op.source_id}: invalid mapped GGUF page geometry")
        physical_block_bytes = op.source_bytes // rows // 24
        row_bytes = op.source_bytes // rows
        for row in range(rows):
            for block in column_map.blocks[rank]:
                yield row * row_bytes + block * physical_block_bytes, physical_block_bytes
        return

    if op.mode not in {"cols", "conv_cols"}:
        raise ShardError(f"{op.source_id}: unsupported partition mode {op.mode}")
    row_bytes = op.source_bytes // rows
    if columns == 0 or row_bytes % columns == 0:
        element_bytes = row_bytes // columns
    else:
        # GGUF is a packed per-row format. Segment cuts were checked to be
        # multiples of 256 logical values, so physical byte offsets scale
        # exactly (each segment starts at a whole GGML block group).
        element_bytes = None
    for row in range(rows):
        start = 0
        for segment in op.column_segments:
            begin = start + rank * (segment // 2)
            end = begin + segment // 2
            if element_bytes is not None:
                yield row * row_bytes + begin * element_bytes, (end - begin) * element_bytes
            else:
                if begin * row_bytes % columns or end * row_bytes % columns:
                    raise ShardError(f"{op.source_id}: packed column cut is not byte aligned")
                yield (row * row_bytes + begin * row_bytes // columns,
                       (end - begin) * row_bytes // columns)
            start += segment


def _coalesce_spans(spans: Iterator[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    """Avoid redundant seeks/reads when row runs are adjacent in source."""
    pending: tuple[int, int] | None = None
    for begin, length in spans:
        if length == 0:
            continue
        if pending is not None and pending[0] + pending[1] == begin:
            pending = (pending[0], pending[1] + length)
        else:
            if pending is not None:
                yield pending
            pending = (begin, length)
    if pending is not None:
        yield pending


def _write_bytes(source: BinaryIO, dest: BinaryIO, offset: int, length: int) -> None:
    source.seek(offset)
    while length:
        data = source.read(min(CHUNK_BYTES, length))
        if not data:
            raise ShardError("physical source payload ended during shard copy")
        dest.write(data)
        length -= len(data)


def _source_has_holes(source: BinaryIO, plan: ShardPlan) -> bool:
    """Linux local sparse range downloads must not masquerade as complete input.

    Some filesystems conservatively report all ranges allocated. If they do
    not support SEEK_HOLE, the caller still verifies full size and every read.
    """
    if not hasattr(os, "SEEK_DATA") or not hasattr(os, "SEEK_HOLE"):
        return False
    fd = source.fileno()
    try:
        for op in plan.operations:
            if not op.source_bytes:
                continue
            begin = plan.source_payload_start + op.source_offset
            end = begin + op.source_bytes
            if os.lseek(fd, begin, os.SEEK_DATA) != begin:
                return True
            if os.lseek(fd, begin, os.SEEK_HOLE) < end:
                return True
    except OSError as error:
        if error.errno in (getattr(os, "ENOTSUP", 95), 22):
            return False
        if error.errno == 6:  # ENXIO: no allocated data after offset
            return True
        raise
    return False


def check_full_source(source: BinaryIO, plan: ShardPlan) -> PhysicalColumnMap | None:
    length = os.fstat(source.fileno()).st_size
    if length != plan.expected_file_size or not plan.complete:
        raise ShardError(
            f"incomplete v3 weights: have {length} bytes; require {plan.expected_file_size}; "
            "use --plan-only until the full source is available")
    if _source_has_holes(source, plan):
        raise ShardError("input has sparse/unallocated object ranges: weights are incomplete")
    return input_column_map(source, plan)


def encode_directory(directory: dict) -> bytes:
    return json.dumps(directory, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _write_rank(source: BinaryIO, dest: BinaryIO, plan: ShardPlan, rank: int,
                column_map: PhysicalColumnMap | None) -> None:
    directory = plan.directories[rank]
    directory_bytes = encode_directory(directory)
    rank_uuid = uuid.uuid5(uuid.UUID(bytes=plan.source_uuid), f"qwen38-tp2-rank-{rank}")
    dest.write(MAGIC + struct.pack("<Q", len(directory_bytes)) + rank_uuid.bytes)
    dest.write(directory_bytes)
    padding = align_up(dest.tell(), PAYLOAD_ALIGNMENT) - dest.tell()
    dest.write(b"\0" * padding)
    payload_start = dest.tell()
    for obj, op in zip(directory["objects"], plan.operations):
        absolute = payload_start + obj["offset"]
        if dest.tell() > absolute:
            raise ShardError(f"{op.source_id}: output objects overlap")
        if absolute > dest.tell():
            dest.write(b"\0" * (absolute - dest.tell()))
        if op.mode == "auxiliary":
            if column_map is None:
                raise ShardError("input_columns cannot be reconstructed from missing source")
            for index in column_map.columns[rank]:
                dest.write(struct.pack("<i", index))
        else:
            copied = 0
            for relative, length in _coalesce_spans(_copy_spans(op, rank, column_map)):
                if relative < 0 or relative + length > op.source_bytes:
                    raise ShardError(f"{op.source_id}: source copy range invalid")
                _write_bytes(source, dest, plan.source_payload_start + op.source_offset + relative,
                             length)
                copied += length
            if copied != op.output_bytes:
                raise ShardError(f"{op.source_id}: output encoded byte count differs from plan")
        if dest.tell() != absolute + op.output_bytes:
            raise ShardError(f"{op.source_id}: output cursor differs from planned size")
    if dest.tell() != payload_start + directory["files"][0]["payload_bytes"]:
        raise ShardError("rank output payload length differs from v3 directory")


def build(plan: ShardPlan, prefix: str | Path) -> tuple[Path, Path]:
    """Preflight both shards, then write into two temporary files and publish."""
    base = Path(prefix)
    destinations = (Path(f"{base}.rank0.ninfer"), Path(f"{base}.rank1.ninfer"))
    if destinations[0] == plan.source or destinations[1] == plan.source:
        raise ShardError("rank outputs cannot overwrite source")
    if any(path.exists() for path in destinations):
        raise ShardError("rank output already exists; choose a new output prefix")
    if not base.parent.is_dir():
        raise ShardError(f"output directory does not exist: {base.parent}")
    temporary: list[Path] = []
    with plan.source.open("rb") as source:
        column_map = check_full_source(source, plan)
        try:
            for rank in (0, 1):
                fd, name = tempfile.mkstemp(prefix=".tp2-v3-", suffix=".ninfer.tmp",
                                            dir=base.parent)
                temporary.append(Path(name))
                with os.fdopen(fd, "wb") as output:
                    _write_rank(source, output, plan, rank, column_map)
                    output.flush()
                    os.fsync(output.fileno())
            if os.fstat(source.fileno()).st_size != plan.expected_file_size:
                raise ShardError("source changed during TP2 writing")
            # Atomic no-replace publication, with rollback if the second
            # rank cannot be published. Hard links work because each temp is
            # created in the same destination directory as its final name.
            published: list[Path] = []
            try:
                for temp, output in zip(temporary, destinations):
                    os.link(temp, output)
                    published.append(output)
            except BaseException:
                for output in published:
                    output.unlink(missing_ok=True)
                raise
        finally:
            for temp in temporary:
                temp.unlink(missing_ok=True)
    return destinations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="single-file v3 NInfer artifact (or header prefix for planning)")
    parser.add_argument("output_prefix", nargs="?", help="rank files: PREFIX.rank{0,1}.ninfer")
    parser.add_argument("--plan-only", action="store_true",
                        help="validate metadata and report exact shard geometry without reading weights")
    parser.add_argument("--plan-json", metavar="PATH",
                        help="optionally save both rank JSON directories in plan mode")
    args = parser.parse_args(argv)
    if not args.plan_only and not args.output_prefix:
        parser.error("output_prefix is required unless --plan-only is set")
    if args.plan_json and not args.plan_only:
        parser.error("--plan-json requires --plan-only")
    try:
        plan = make_plan(args.source)
        if args.plan_only:
            if args.plan_json:
                Path(args.plan_json).write_text(json.dumps({
                    "summary": plan.summary(), "rank_directories": plan.directories,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(plan.summary(), indent=2))
        else:
            outputs = build(plan, args.output_prefix)
            print(json.dumps({**plan.summary(), "written": [str(p) for p in outputs]}, indent=2))
        return 0
    except (ShardError, OSError) as error:
        print(f"TP2 v3 sharder: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
