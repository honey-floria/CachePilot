import unittest

from workloads.generator import WORKLOAD_NAMES, generate_workload, validate_trace


class WorkloadGeneratorTests(unittest.TestCase):
    def test_all_workloads_are_valid_and_reproducible(self):
        for name in WORKLOAD_NAMES:
            with self.subTest(workload=name):
                first = generate_workload(name, seed=41, count=8)
                self.assertEqual(first, generate_workload(name, seed=41, count=8))
                self.assertEqual(first, validate_trace(first, name))

    def test_workload_shapes_encode_intent(self):
        burst = generate_workload("burst", seed=1, count=8)
        self.assertEqual({0, 100}, {record["arrival_ms"] for record in burst})
        shared = generate_workload("shared-prefix", seed=1, count=8)
        self.assertIn("prefix_group", shared[0])
        cancelled = generate_workload("cancellation-heavy", seed=1, count=8)
        self.assertGreaterEqual(sum("cancel_after_ms" in r for r in cancelled), 6)
        long_context = generate_workload("long-context", seed=1, count=8)
        self.assertGreaterEqual(min(r["prompt_tokens"] for r in long_context), 384)

    def test_validation_rejects_duplicate_ids_and_non_monotonic_arrivals(self):
        records = list(generate_workload("uniform", seed=1, count=2))
        records[1]["request_id"] = records[0]["request_id"]
        with self.assertRaises(ValueError):
            validate_trace(records)
        records = list(generate_workload("uniform", seed=1, count=2))
        records[1]["arrival_ms"] = -1
        with self.assertRaises(ValueError):
            validate_trace(records)


if __name__ == "__main__":
    unittest.main()
