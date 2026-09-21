import unittest
from pathlib import Path

from cachepilot.config.baseline import load_model_baseline
from cachepilot.runtime.kv_planner import (
    ContextLimitExceededError,
    KVModelSpec,
    KVPlanner,
    KVPlannerError,
    UsableKVCapacityRequiredError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPOSITORY_ROOT / "config" / "model.json"


class KVPlannerTests(unittest.TestCase):
    def setUp(self):
        self.spec = KVModelSpec(
            num_hidden_layers=24,
            num_key_value_heads=2,
            head_dim=64,
            dtype="bfloat16",
            block_size=16,
            context_limit=8192,
        )
        self.planner = KVPlanner(self.spec)

    def test_matches_hand_calculated_qwen_kv_sizes(self):
        self.assertEqual(12_288, self.planner.bytes_per_token)
        self.assertEqual(196_608, self.planner.bytes_per_block)
        self.assertEqual(512, self.planner.context_blocks)
        self.assertEqual(100_663_296, self.planner.theoretical_context_bytes)

    def test_request_rounds_up_to_complete_blocks(self):
        plan = self.planner.plan_request(
            prompt_tokens=100,
            expected_output_tokens=60,
        )

        self.assertEqual(160, plan.total_tokens)
        self.assertEqual(10, plan.logical_blocks)
        self.assertEqual(160, plan.allocated_tokens)
        self.assertEqual(0, plan.padding_tokens)
        self.assertEqual(1_966_080, plan.theoretical_bytes)

        rounded = self.planner.plan_request(100, 61)
        self.assertEqual(11, rounded.logical_blocks)
        self.assertEqual(176, rounded.allocated_tokens)
        self.assertEqual(15, rounded.padding_tokens)

    def test_request_cannot_exceed_service_context_limit(self):
        with self.assertRaises(ContextLimitExceededError):
            self.planner.plan_request(8000, 193)

    def test_capacity_requires_an_explicit_usable_kv_budget(self):
        with self.assertRaises(UsableKVCapacityRequiredError):
            self.planner.plan_capacity(None)

        capacity = self.planner.plan_capacity(1_000_000)
        self.assertEqual(5, capacity.logical_blocks)
        self.assertEqual(80, capacity.token_capacity)
        self.assertEqual(983_040, capacity.allocated_bytes)
        self.assertEqual(16_960, capacity.unused_bytes)

    def test_model_baseline_supplies_architecture_and_context(self):
        baseline = load_model_baseline(MODEL_CONFIG)
        spec = KVModelSpec.from_model_baseline(
            baseline,
            dtype="bf16",
            block_size=16,
        )

        self.assertEqual(24, spec.num_hidden_layers)
        self.assertEqual(2, spec.num_key_value_heads)
        self.assertEqual(64, spec.head_dim)
        self.assertEqual("bfloat16", spec.dtype)
        self.assertEqual(8192, spec.context_limit)

    def test_invalid_or_incomplete_values_are_rejected(self):
        with self.assertRaises(KVPlannerError):
            KVModelSpec(24, 2, 64, "int8", 16, 8192)
        with self.assertRaises(KVPlannerError):
            KVModelSpec(24, 2, 64, "float16", 0, 8192)
        with self.assertRaises(KVPlannerError):
            self.planner.plan_request(-1, 1)
        with self.assertRaises(KVPlannerError):
            self.planner.plan_request(0, 0)


if __name__ == "__main__":
    unittest.main()
