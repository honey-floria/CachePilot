import unittest

from benchmarks.compare import ComparisonError, compare_summaries


class MetricComparisonTests(unittest.TestCase):
    def test_same_semantics_are_preserved_without_aggregation(self):
        result = compare_summaries(
            "ttft_ms",
            (
                self.summary("torch", "TorchExecutor", False, 10.0),
                self.summary("vllm", "VllmExecutor", False, 2.0),
            ),
        )

        self.assertEqual("measured_monotonic_clock", result["semantic_signature"])
        self.assertEqual(
            ["torch", "vllm"],
            [run["run_id"] for run in result["runs"]],
        )

    def test_simulated_and_measured_latency_are_rejected(self):
        with self.assertRaisesRegex(ComparisonError, "incompatible semantics"):
            compare_summaries(
                "ttft_ms",
                (
                    self.summary("sim", "SimExecutor", True, 10.0),
                    self.summary("torch", "TorchExecutor", False, 10.0),
                ),
            )

    def test_different_batch_throughput_is_rejected(self):
        with self.assertRaisesRegex(ComparisonError, "incompatible semantics"):
            compare_summaries(
                "throughput_completion_tokens_per_s",
                (
                    self.summary("torch", "TorchExecutor", False, 10.0),
                    self.summary("vllm", "VllmExecutor", False, 10.0),
                ),
            )

    @staticmethod
    def summary(run_id, executor, simulated, value):
        sha = "a" * 40
        return {
            "artifact_type": "cachepilot_experiment_summary",
            "schema_version": 1,
            "run_id": run_id,
            "metrics": {
                "ttft_ms": {
                    "count": 1,
                    "p50": value,
                    "p95": value,
                    "p99": value,
                }
            },
            "throughput_completion_tokens_per_s": value,
            "cancellation_rate": 0.0,
            "resource_peaks": {"reserved_blocks_peak": 1},
            "simulation": {
                "is_simulated": simulated,
                "executor": executor,
                "label": "simulated" if simulated else "measured",
            },
            "control_variables": {
                "trace_id": "trace-1",
                "seed": 7,
                "model_id": "model",
                "model_revision": sha,
                "tokenizer_revision": sha,
                "executor": executor,
            },
        }


if __name__ == "__main__":
    unittest.main()
