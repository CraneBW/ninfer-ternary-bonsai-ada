"""Independent raw-byte oracle for the v3 mixed-GGUF TP2 artifact sharder."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from tools.tp2 import shard_v3_qwen38 as tp


GGUF = {
    "gguf_iq2_s": 82,
    "gguf_iq3_s": 110,
    "gguf_iq4_xs": 136,
    "gguf_q4_k": 144,
}


def fixture_bytes(name: str, shape: list[int], fmt: str) -> bytes:
    if fmt in GGUF:
        rows, width = shape
        block_bytes = GGUF[fmt]
        return b"".join(
            bytes(((sum(map(ord, name)) + row * 29 + column * 67 + byte * 13) & 255)
                  for byte in range(block_bytes))
            for row in range(rows) for column in range(width // 256)
        )
    word_bytes = {"bf16": 2, "fp32": 4, "int32": 4}[fmt]
    count = 1
    for dimension in shape:
        count *= dimension
    return bytes(((sum(map(ord, name)) + index * 19) & 255)
                 for index in range(count * word_bytes))


def make_fixture(path: Path) -> tuple[dict, dict[str, bytes], tuple[int, ...]]:
    """A minimal but valid physical v3: fake bytes, exact real segment widths."""
    source: dict[str, bytes] = {}
    objects: list[dict] = []
    bindings: dict[str, dict] = {}
    uses: list[dict] = []
    cursor = 0

    def add(name: str, shape: list[int] | None, fmt: str, *, encoding: bool = False) -> None:
        nonlocal cursor
        if encoding:
            raw = b'{"tokenizer":"synthetic"}'
            obj = {"id": name, "kind": "resource", "encoding": "raw_bytes_v1"}
        else:
            assert shape is not None
            raw = fixture_bytes(name, shape, fmt)
            obj = {"id": name, "kind": "tensor", "shape": shape, "format": fmt,
                   "layout": "gguf_blocks_v1" if fmt in GGUF else "contiguous_le_v1"}
        cursor = tp.align_up(cursor, 1 if encoding else 256)
        obj.update(offset=cursor, bytes=len(raw))
        objects.append(obj)
        source[name] = raw
        cursor += len(raw)

    def whole(name: str, object_id: str) -> None:
        bindings[name] = {"object": object_id}

    def parts(layer_prefix: str, obj: str, roles: list[tuple[str, int]], columns: int) -> None:
        start = 0
        for suffix, rows in roles:
            bindings[layer_prefix + suffix] = {
                "parts": [{"object": obj, "range": [start * columns, (start + rows) * columns]}]
            }
            start += rows

    add("resource/text/tokenizer.json", None, "", encoding=True)
    add("weight/qkgv", [8, 256], "gguf_iq3_s")
    parts("text/layers/3/", "weight/qkgv", [
        ("attention/query", 2), ("attention/gate", 2),
        ("attention/key", 2), ("attention/value", 2),
    ], 256)
    add("weight/qkvz", [12, 256], "gguf_iq2_s")
    parts("text/layers/0/", "weight/qkvz", [
        ("gdn/query", 2), ("gdn/key", 2),
        ("gdn/value", 4), ("gdn/z", 4),
    ], 256)
    add("weight/conv", [4, 10240], "bf16")
    whole("text/layers/0/gdn/convolution", "weight/conv")
    add("weight/gdn_out", [4, 6144], "gguf_iq3_s")
    whole("text/layers/0/gdn/output", "weight/gdn_out")
    uses.append({"parameter": "text/layers/0/gdn/output", "input": "gdn/gated_output",
                 "auxiliaries": {"input_columns": {"object": "auxiliary/000000"}}})
    add("weight/attn_out", [4, 6144], "gguf_iq4_xs")
    whole("text/layers/3/attention/output", "weight/attn_out")
    add("weight/mlp_gate_up", [8, 256], "gguf_q4_k")
    parts("text/layers/0/", "weight/mlp_gate_up", [
        ("mlp/gate", 4), ("mlp/up", 4),
    ], 256)
    add("weight/mlp_down", [4, 17408], "gguf_iq2_s")
    whole("text/layers/0/mlp/down", "weight/mlp_down")
    add("weight/a_projection", [4, 256], "bf16")
    whole("text/layers/0/gdn/a_projection", "weight/a_projection")
    add("weight/b_projection", [4, 256], "bf16")
    whole("text/layers/0/gdn/b_projection", "weight/b_projection")
    add("weight/a_log", [4], "fp32")
    whole("text/layers/0/gdn/a_log", "weight/a_log")
    add("weight/dt_bias", [4], "fp32")
    whole("text/layers/0/gdn/dt_bias", "weight/dt_bias")
    add("weight/vision_copy", [3, 4], "bf16")
    whole("vision/layers/0/norm1_weight", "weight/vision_copy")

    # 256 physical GGUF columns belong exclusively to rank 0 or 1, but
    # blocks are interleaved in storage and each block permutes its local inputs.
    input_columns = []
    for block in range(24):
        rank = block % 2
        ordinal = block // 2
        local = list(range(ordinal * 256, (ordinal + 1) * 256))
        if ordinal % 2:
            local.reverse()
        input_columns.extend([rank * 3072 + index for index in local])
    add("auxiliary/000000", [6144], "int32")
    source["auxiliary/000000"] = struct.pack("<6144i", *input_columns)

    root = {
        "components": {"text": {"config": {"model_type": "qwen3_5_text", "hidden_size": 5120}}},
        "objects": objects,
        "bindings": bindings,
        "uses": uses,
        "metadata": {"name": "qwen3.8-27b"},
        "provenance": {"recipe": "synthetic raw-byte TP2 unit fixture"},
        "files": [{"path": None, "payload_bytes": cursor}],
    }
    write_fixture(path, root, source)
    return root, source, tuple(input_columns)


def write_fixture(path: Path, directory: dict, source: dict[str, bytes]) -> None:
    raw_json = json.dumps(directory, separators=(",", ":")).encode()
    prefix = tp.MAGIC + struct.pack("<Q", len(raw_json)) + bytes.fromhex("00112233445566778899aabbccddeeff")
    with path.open("wb") as output:
        output.write(prefix)
        output.write(raw_json)
        output.write(b"\0" * (tp.align_up(output.tell(), 4096) - output.tell()))
        start = output.tell()
        for obj in directory["objects"]:
            output.write(b"\0" * (start + obj["offset"] - output.tell()))
            output.write(source[obj["id"]])
        assert output.tell() == start + directory["files"][0]["payload_bytes"]


def rank_contents(path: Path) -> tuple[dict, dict[str, bytes]]:
    root, _identity, payload_offset, file_bytes = tp.read_directory(path)
    assert file_bytes == payload_offset + root["files"][0]["payload_bytes"]
    values: dict[str, bytes] = {}
    with path.open("rb") as handle:
        for obj in root["objects"]:
            handle.seek(payload_offset + obj["offset"])
            values[obj["id"]] = handle.read(obj["bytes"])
    return root, values


def oracle_spans(name: str, obj: dict, rank: int,
                 original_map: tuple[int, ...]) -> list[tuple[int, int]]:
    """Standalone reference partitions; operates on logical 256-value groups."""
    shape = obj.get("shape", [])
    if name in ("weight/qkgv", "weight/qkvz", "weight/mlp_gate_up"):
        groups = {
            "weight/qkgv": (2, 2, 2, 2),
            "weight/qkvz": (2, 2, 4, 4),
            "weight/mlp_gate_up": (4, 4),
        }[name]
        bytes_per_row = obj["bytes"] // shape[0]
        row = 0
        spans = []
        for count in groups:
            spans.append(((row + rank * count // 2) * bytes_per_row,
                          count // 2 * bytes_per_row))
            row += count
        return spans
    if name in ("weight/a_projection", "weight/b_projection", "weight/a_log", "weight/dt_bias"):
        return [(rank * obj["bytes"] // 2, obj["bytes"] // 2)]
    if name == "weight/conv":
        spans = []
        width_bytes = 10240 * 2
        for row in range(4):
            col = 0
            for segment in (2048, 2048, 6144):
                spans.append((row * width_bytes + (col + rank * (segment // 2)) * 2,
                              segment))
                col += segment
        return spans
    if name in ("weight/attn_out", "weight/mlp_down"):
        rows, width = shape
        block_bytes = GGUF[obj["format"]]
        blocks = width // 256
        row_bytes = blocks * block_bytes
        return [(row * row_bytes + rank * (blocks // 2) * block_bytes,
                 blocks // 2 * block_bytes) for row in range(rows)]
    if name == "weight/gdn_out":
        rows = shape[0]
        block_bytes = GGUF[obj["format"]]
        return [(row * 24 * block_bytes + block * block_bytes, block_bytes)
                for row in range(rows) for block in range(24)
                if original_map[block * 256] // 3072 == rank]
    if name == "auxiliary/000000":
        return []
    return [(0, obj["bytes"])]


class TestV3TP2Sharder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.source = self.dir / "source.ninfer"
        self.root, self.source_bytes, self.input_columns = make_fixture(self.source)

    def test_complete_shards_match_independent_byte_oracle(self):
        plan = tp.make_plan(self.source)
        self.assertTrue(plan.complete)
        self.assertEqual(plan.auxiliary_id, "auxiliary/000000")
        self.assertEqual(plan.operations[1].row_groups, ((0, 2), (2, 4), (4, 6), (6, 8)))
        rank_files = tp.build(plan, self.dir / "model")
        original = {obj["id"]: obj for obj in self.root["objects"]}
        for rank, rank_path in enumerate(rank_files):
            directory, values = rank_contents(rank_path)
            self.assertEqual(directory, plan.directories[rank])
            self.assertEqual(directory["metadata"]["name"], f"qwen3.8-27b-tp2-rank{rank}")
            self.assertEqual(directory["metadata"]["tensor_parallel"]["rank"], rank)
            self.assertEqual(directory["metadata"]["tensor_parallel"]["world_size"], 2)
            for name, obj in original.items():
                if name == "auxiliary/000000":
                    expected = [self.input_columns[block * 256 + index] - rank * 3072
                                for block in range(24)
                                if block % 2 == rank for index in range(256)]
                    self.assertEqual(values[name], struct.pack("<3072i", *expected))
                    self.assertEqual(len(set(expected)), 3072)
                    continue
                slices = oracle_spans(name, obj, rank, self.input_columns)
                expected = b"".join(self.source_bytes[name][begin:begin + length]
                                    for begin, length in slices)
                self.assertEqual(values[name], expected, name)
            self.assertEqual(directory["bindings"]["text/layers/0/gdn/value"]["parts"][0]["range"],
                             [2 * 256, 4 * 256])
            self.assertEqual(directory["bindings"]["text/layers/0/gdn/z"]["parts"][0]["range"],
                             [4 * 256, 6 * 256])
            self.assertEqual(directory["objects"][-1]["shape"], [3072])
            self.assertEqual(directory["objects"][-1]["bytes"], 12288)

    def test_independent_inverse_restores_every_original_physical_object(self):
        plan = tp.make_plan(self.source)
        output_files = tp.build(plan, self.dir / "inverse")
        ranks = [rank_contents(path)[1] for path in output_files]
        for obj in self.root["objects"]:
            name = obj["id"]
            rebuilt = bytearray(obj["bytes"])
            if name == "auxiliary/000000":
                for physical_block in range(24):
                    rank, ordinal = physical_block % 2, physical_block // 2
                    values = struct.unpack("<3072i", ranks[rank][name])
                    for index in range(256):
                        original_value = values[ordinal * 256 + index] + rank * 3072
                        struct.pack_into("<i", rebuilt, (physical_block * 256 + index) * 4,
                                         original_value)
            else:
                for rank in (0, 1):
                    spans = oracle_spans(name, obj, rank, self.input_columns)
                    cursor = 0
                    for offset, size in spans:
                        rebuilt[offset:offset + size] = ranks[rank][name][cursor:cursor + size]
                        cursor += size
                    self.assertEqual(cursor, len(ranks[rank][name]))
            self.assertEqual(bytes(rebuilt), self.source_bytes[name], name)

    def test_plan_only_on_header_prefix_and_refuse_incomplete_write(self):
        root, _source_uuid, base, size = tp.read_directory(self.source)
        header_only = self.dir / "prefix.ninfer"
        with self.source.open("rb") as handle:
            header = handle.read(16)
        directory_end = 32 + struct.unpack_from("<Q", header, 8)[0]
        header_only.write_bytes(self.source.read_bytes()[:directory_end])
        plan = tp.make_plan(header_only)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.source_file_size, directory_end)
        self.assertLessEqual(directory_end, base)
        self.assertEqual(plan.expected_file_size, size)
        self.assertEqual(len(plan.operations), len(root["objects"]))
        with io.StringIO() as stdout, redirect_stdout(stdout):
            self.assertEqual(tp.main([str(header_only), "--plan-only"]), 0)
            message = json.loads(stdout.getvalue())
            self.assertFalse(message["source_complete_by_length"])
            self.assertEqual(message["partitions"]["mapped_cols"], 1)
        rank_prefix = self.dir / "should-not-exist"
        with io.StringIO() as stderr, redirect_stderr(stderr):
            self.assertEqual(tp.main([str(header_only), str(rank_prefix)]), 2)
            self.assertIn("incomplete v3 weights", stderr.getvalue())
        self.assertFalse(Path(f"{rank_prefix}.rank0.ninfer").exists())
        self.assertFalse(Path(f"{rank_prefix}.rank1.ninfer").exists())

    def test_reject_mixed_owner_in_physical_gguf_block(self):
        values = list(self.input_columns)
        values[0], values[256] = values[256], values[0]
        broken = dict(self.source_bytes)
        broken["auxiliary/000000"] = struct.pack("<6144i", *values)
        write_fixture(self.source, self.root, broken)
        plan = tp.make_plan(self.source)
        with self.assertRaisesRegex(tp.ShardError, "crosses TP2 rank boundary"):
            tp.build(plan, self.dir / "invalid")
        self.assertFalse(list(self.dir.glob("invalid.rank*.ninfer")))

    def test_reject_unsupported_split_codec_and_misaligned_logical_range(self):
        incorrect = json.loads(json.dumps(self.root))
        next(obj for obj in incorrect["objects"] if obj["id"] == "weight/qkvz")["format"] = "gguf_future"
        write_fixture(self.source, incorrect, self.source_bytes)
        with self.assertRaisesRegex(tp.ShardError, "unknown GGUF physical storage"):
            tp.make_plan(self.source)

        incorrect = json.loads(json.dumps(self.root))
        incorrect["bindings"]["text/layers/0/gdn/query"]["parts"][0]["range"] = [0, 82 * 2]
        write_fixture(self.source, incorrect, self.source_bytes)
        with self.assertRaisesRegex(tp.ShardError, "align to logical rows"):
            tp.make_plan(self.source)

    def test_existing_rank_output_is_not_overwritten(self):
        plan = tp.make_plan(self.source)
        destination = self.dir / "already"
        rank0 = Path(f"{destination}.rank0.ninfer")
        rank0.write_bytes(b"owner data")
        with self.assertRaisesRegex(tp.ShardError, "already exists"):
            tp.build(plan, destination)
        self.assertEqual(rank0.read_bytes(), b"owner data")

    @unittest.skipUnless(hasattr(os, "SEEK_DATA") and hasattr(os, "SEEK_HOLE"),
                         "sparse allocation checks need POSIX SEEK_HOLE")
    def test_sparse_file_with_full_declared_length_is_not_complete_weights(self):
        root, _source_uuid, payload_start, file_size = tp.read_directory(self.source)
        name = self.dir / "sparse.ninfer"
        with name.open("wb") as output:
            output.write(self.source.read_bytes()[:payload_start])
            last = root["objects"][-1]
            output.seek(payload_start + last["offset"])
            output.write(self.source_bytes[last["id"]])
        self.assertEqual(name.stat().st_size, file_size)
        with name.open("rb") as handle:
            first = root["objects"][1]
            at = payload_start + first["offset"]
            try:
                contains_hole = os.lseek(handle.fileno(), at, os.SEEK_DATA) != at
            except OSError:
                contains_hole = True
        if not contains_hole:
            self.skipTest("filesystem cannot report sparse ranges")
        plan = tp.make_plan(name)
        self.assertTrue(plan.complete)
        with self.assertRaisesRegex(tp.ShardError, "sparse/unallocated"):
            tp.build(plan, self.dir / "sparse-output")


if __name__ == "__main__":
    unittest.main()
