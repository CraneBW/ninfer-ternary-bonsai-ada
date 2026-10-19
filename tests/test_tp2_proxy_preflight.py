"""Fail-safe TP2 topology/rank-artifact preflight (no CUDA device required)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tools.tp2.tp2_proxy import preflight, validate_rank_artifact


class Tp2ProxyPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "ninfer-tp2"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self.prefix = self.root / "model"
        for rank in (0, 1):
            self.make_rank(rank)
        self.args = SimpleNamespace(
            gpus=[0, 1], rank_ports=[19000, 19001],
            max_inflight=1, max_waiting=4, binary=str(self.binary),
            model_prefix=str(self.prefix))

    def make_rank(self, rank, *, metadata_rank=None, declared_name=None):
        descriptor = {"metadata": {
            "name": declared_name or f"qwen3.8-27b-tp2-rank{rank}",
            "tensor_parallel": {"world_size": 2, "rank": rank if metadata_rank is None
                                else metadata_rank, "source_model": "qwen3.8-27b"}},
                      "objects": [], "bindings": {},
                      "files": [{"path": None, "payload_bytes": 128}]}
        payload = json.dumps(descriptor).encode()
        prefix = b"NINFER\0\3" + len(payload).to_bytes(8, "little") + b"\0" * 16 + payload
        data = prefix + b"\0" * ((-len(prefix)) % 4096) + b"\0" * 128
        (self.root / f"model.rank{rank}.ninfer").write_bytes(data)

    def test_two_distinct_devices_and_rank_artifacts(self):
        self.assertEqual(preflight(self.args, devices={0, 1}), {0, 1})

    def test_single_gpu_must_not_launch_as_fake_tp2(self):
        with self.assertRaisesRegex(ValueError, "two physical GPUs"):
            preflight(self.args, devices={0})
        self.args.gpus = [0, 0]
        with self.assertRaisesRegex(ValueError, "DISTINCT"):
            preflight(self.args, devices={0, 1})

    def test_invalid_rank_ports_and_missing_binary(self):
        self.args.rank_ports = [19000, 19000]
        with self.assertRaisesRegex(ValueError, "distinct valid rank ports"):
            preflight(self.args, devices={0, 1})
        self.args.rank_ports = [19000, 19001]
        self.binary.unlink()
        with self.assertRaisesRegex(ValueError, "binary is missing"):
            preflight(self.args, devices={0, 1})

    def test_mislabelled_rank_and_truncated_header(self):
        self.make_rank(1, metadata_rank=0)
        with self.assertRaisesRegex(ValueError, "not the expected two-rank"):
            preflight(self.args, devices={0, 1})
        bad = self.root / "model.rank1.ninfer"
        bad.write_bytes(b"NINFER\0\3")
        with self.assertRaisesRegex(ValueError, "not a valid NInfer artifact"):
            validate_rank_artifact(bad, 1)

    def test_missing_rank_is_not_silently_replicated(self):
        (self.root / "model.rank1.ninfer").unlink()
        with self.assertRaisesRegex(ValueError, "rank1 source artifact is missing"):
            preflight(self.args, devices={0, 1})

    def test_header_only_rank_must_not_launch(self):
        path = self.root / "model.rank1.ninfer"
        with path.open("r+b") as stream:
            stream.truncate(4096)
        with self.assertRaisesRegex(ValueError, "truncated or incorrect rank payload length"):
            preflight(self.args, devices={0, 1})


if __name__ == "__main__":
    unittest.main()
