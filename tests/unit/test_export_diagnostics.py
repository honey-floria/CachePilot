import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.export_diagnostics import ExportError, export_bundle, fetch_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload.encode("utf-8")


class DiagnosticExportTests(unittest.TestCase):
    def test_single_host_bundle_contains_snapshot_summary_and_report(self):
        metrics = """# TYPE cachepilot_gateway_up gauge
cachepilot_gateway_up 1
cachepilot_executor_healthy 1
cachepilot_active_sequences 2
cachepilot_reserved_kv_blocks 12
cachepilot_kv_capacity_blocks 30
cachepilot_estimated_gpu_seconds_total{model="m",tenant="t"} 1.5
cachepilot_estimated_cost_total{currency="USD",model="m",tenant="t"} 0.25
"""

        def opener(request, timeout):
            self.assertEqual("http://127.0.0.1:8000/metrics", request.full_url)
            self.assertEqual(2.0, timeout)
            return FakeResponse(metrics)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "bundle"
            paths = export_bundle(
                base_url="http://127.0.0.1:8000",
                run_dir=REPOSITORY_ROOT / "tests" / "fixtures" / "diagnostics",
                output_dir=output_dir,
                timeout_seconds=2.0,
                opener=opener,
            )

            self.assertEqual(metrics, paths["metrics"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")

        self.assertEqual("demo-sim-mixed-001", summary["run_id"])
        self.assertEqual(
            "cachepilot_operational_snapshot", snapshot["artifact_type"]
        )
        self.assertIn("KV utilization | 40.00%", report)
        self.assertIn("estimated cost total | 0.25", report)
        self.assertIn("ttft_ms", report)

    def test_non_cachepilot_metrics_are_rejected(self):
        def opener(request, timeout):
            del request, timeout
            return FakeResponse("other_metric 1\n")

        with self.assertRaises(ExportError):
            fetch_metrics(
                "http://127.0.0.1:8000",
                timeout_seconds=1.0,
                opener=opener,
            )


if __name__ == "__main__":
    unittest.main()
