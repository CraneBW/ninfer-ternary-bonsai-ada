import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.model_router import choose_backend, inspect, main


class ModelRouterTest(unittest.TestCase):
    def test_versions_and_model_routes(self):
        cases = (
            (b"NINFER\0\2", {"identity": {"model_id": "qwen3.6-27b", "weights_id": "nvfp4"}},
             "native"),
            (b"NINFER\0\3", {"metadata": {"name": "qwen3.8-27b"},
                                "components": {"text": {"config": {"model_type": "qwen3_5_text"}}}},
             "native"),
            (b"NINFER\0\3", {"metadata": {"name": "qwen3.6-27b"},
                                "components": {"text": {"config": {"model_type": "qwen3_5_text"}}}},
             "native"),
            (b"NINFER\0\3", {"metadata": {"name": "bonsai2-4b"},
                                "components": {"text": {"config": {"model_type": "qwen3_5_text"}}}},
             "companion"),
            (b"NINFER\0\3", {"metadata": {"name": "gemma4-26b-a4b"},
                                "components": {"text": {"config": {"model_type": "gemma4_text"}}}},
             "companion"),
            (b"NINFER\0\3", {"metadata": {"name": "flash-next"},
                                "components": {"text": {"config": {"model_type": "qwen4_exp_text"}}}},
             "companion"),
        )
        with TemporaryDirectory() as scratch:
            for index, (magic, directory, expected) in enumerate(cases):
                path = Path(scratch) / f"case_{index}.ninfer"
                body = json.dumps(directory).encode("utf-8")
                path.write_bytes(magic + len(body).to_bytes(8, "little") +
                                 (b"\0" * 16 if magic[-1] == 3 else b"") + body)
                found = inspect(path)
                self.assertEqual(choose_backend(found, []), expected)
                self.assertEqual(main([str(path), "--dry-run"]), 0)
                if index in (1, 2):
                    self.assertEqual(
                        choose_backend(found, ["--spec", "mtp"]), "native"
                    )
                    self.assertEqual(
                        choose_backend(found, ["--spec", "dflash2", "--vision"]), "native"
                    )
                    self.assertEqual(
                        choose_backend(found, ["--spec", "dflash"]), "companion"
                    )


if __name__ == "__main__":
    unittest.main()
