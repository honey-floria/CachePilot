import unittest

from cachepilot.executor_capabilities import (
    EXECUTOR_CAPABILITY_MATRIX,
    IncomparableMetricError,
    require_comparable_metric,
)
from cachepilot.executors import SimExecutor, TorchExecutor, VllmExecutor


class ExecutorCapabilityTests(unittest.TestCase):
    def test_matrix_matches_executor_declarations(self):
        expected = {
            "SimExecutor": SimExecutor,
            "TorchExecutor": TorchExecutor,
            "VllmExecutor": VllmExecutor,
        }

        self.assertEqual(set(expected), set(EXECUTOR_CAPABILITY_MATRIX))
        for name, executor_type in expected.items():
            self.assertIs(EXECUTOR_CAPABILITY_MATRIX[name], executor_type.capabilities)

        self.assertEqual("simulated_continuous", SimExecutor.capabilities.batch)
        self.assertEqual("single_request", TorchExecutor.capabilities.batch)
        self.assertEqual("vllm", VllmExecutor.capabilities.batch_owner)
        self.assertFalse(VllmExecutor.capabilities.physical_kv_observable)

    def test_latency_rejects_simulated_and_measured_results(self):
        with self.assertRaisesRegex(
            IncomparableMetricError,
            "incompatible semantics",
        ):
            require_comparable_metric(
                "ttft_ms",
                ("SimExecutor", "TorchExecutor"),
            )

    def test_throughput_rejects_different_batch_semantics(self):
        with self.assertRaisesRegex(
            IncomparableMetricError,
            "incompatible semantics",
        ):
            require_comparable_metric(
                "throughput_completion_tokens_per_s",
                ("TorchExecutor", "VllmExecutor"),
            )

    def test_unobservable_physical_metrics_are_rejected(self):
        for metric in ("physical_kv", "physical_hit"):
            with self.subTest(metric=metric), self.assertRaisesRegex(
                IncomparableMetricError,
                "unobservable",
            ):
                require_comparable_metric(
                    metric,
                    ("SimExecutor", "VllmExecutor"),
                )

    def test_shared_logical_semantics_remain_comparable(self):
        self.assertEqual(
            "cachepilot_logical_reservation_blocks",
            require_comparable_metric(
                "reserved_blocks_peak",
                ("SimExecutor", "TorchExecutor", "VllmExecutor"),
            ),
        )
        self.assertEqual(
            "measured_monotonic_clock",
            require_comparable_metric(
                "ttft_ms",
                ("TorchExecutor", "VllmExecutor"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
