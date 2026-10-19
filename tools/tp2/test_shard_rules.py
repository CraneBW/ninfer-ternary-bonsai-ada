"""Focused host checks for Qwen3.8 two-rank partition geometry."""

import unittest

import torch

from tools.tp2.shard_qwen38_27b import Rule, rule_for, shard_shape, take, take_frac


class ShardRulesTest(unittest.TestCase):
    def test_rows_preserve_each_segment(self):
        rule = Rule("rows", (2, 4))
        weights = torch.arange(12, dtype=torch.int32).reshape(6, 2)
        ranks = [take(weights, 0, rule.segments, r) for r in range(2)]
        self.assertEqual(shard_shape(tuple(weights.shape), rule), (3, 2))
        rebuilt = torch.cat(
            [ranks[0][:1], ranks[1][:1], ranks[0][1:], ranks[1][1:]], dim=0
        )
        self.assertTrue(torch.equal(weights, rebuilt))

    def test_columns_preserve_each_segment(self):
        rule = Rule("cols", (4, 8))
        weights = torch.arange(24, dtype=torch.int32).reshape(2, 12)
        ranks = [take(weights, 1, rule.segments, r) for r in range(2)]
        self.assertEqual(shard_shape(tuple(weights.shape), rule), (2, 6))
        rebuilt = torch.cat(
            [ranks[0][:, :2], ranks[1][:, :2],
             ranks[0][:, 2:], ranks[1][:, 2:]], dim=1
        )
        self.assertTrue(torch.equal(weights, rebuilt))

    def test_packed_nvfp4_split_on_code_boundaries(self):
        segments = (8, 16)
        codes = torch.arange(12).reshape(1, 12)
        ranks = [take_frac(codes, 1, segments, r, 1, 2) for r in range(2)]
        self.assertEqual([tuple(x.shape) for x in ranks], [(1, 6), (1, 6)])
        rebuilt = torch.cat(
            [ranks[0][:, :2], ranks[1][:, :2],
             ranks[0][:, 2:], ranks[1][:, 2:]], dim=1
        )
        self.assertTrue(torch.equal(codes, rebuilt))

    def test_model_roles(self):
        self.assertEqual(
            rule_for("text/layers/0/mlp/gate_up").segments, (17408, 17408)
        )
        self.assertEqual(
            rule_for("text/layers/5/gdn/output").axis, "cols"
        )
        self.assertIsNone(rule_for("text/token_embedding"))


if __name__ == "__main__":
    unittest.main()
