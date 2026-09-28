import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from benchmarks import formal_gpu_run
from benchmarks.analyze import ProtocolError, validate_request
from benchmarks.colab_phase1 import sim_run
from cachepilot.gateway.api import GatewaySettings, create_app
from tests.contract import test_experiment_protocol


ROOT = Path(__file__).resolve().parents[2]


class ColabEvidenceTests(unittest.TestCase):
    def test_notebook_refuses_exit_without_prefix_evidence(self):
        notebook = json.loads(
            (ROOT / "notebooks/colab_phase1_matrix.ipynb").read_text()
        )
        sources = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for source in sources:
            ast.parse(source)
        source = next(source for source in sources if "def read_optional" in source)
        strategies = [
            ("strict", "fcfs", "blind"),
            ("adaptive", "fcfs", "blind"),
            ("strict", "wfq", "blind"),
            ("strict", "wfq", "aware"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def save(name, value):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            for label in (
                "matrix-validation",
                "chaos-control",
                "chaos-gpu",
                "phase0",
                "api-regression",
                "phase1-gate",
            ):
                save(label + ".status.json", {"returncode": 0})
            save(
                "chaos-gpu.json",
                {
                    "status": "PASS",
                    "cases": [
                        {"name": name}
                        for name in (
                            "cancel",
                            "disconnect",
                            "timeout",
                            "exception",
                            "oom",
                        )
                    ],
                },
            )
            rows = []
            for strategy in strategies:
                name = "-".join(strategy)
                save("progressive-" + name + ".status.json", {"returncode": 0})
                save(
                    "progressive/" + name + "/progressive_report.json",
                    {"matrix_points": [{"safe": True}]},
                )
                rows.extend(
                    [
                        {
                            "strategy": name,
                            "logical_hits": 0,
                            "failed": 0,
                            "ttft_p95_ms": 10,
                            "tpot_p95_ms": 1,
                            "total_p99_ms": 20,
                            "wall_tokens_per_s": 10,
                            "fairness_jain": 1,
                            "rejection_rate": 0,
                            "request_kv_peak_blocks": 8,
                        }
                        for _ in range(3)
                    ]
                )
            exec(
                compile(source, "notebook-conclusions", "exec"),
                {
                    "OUT": output,
                    "json": json,
                    "STRATEGIES": strategies,
                    "REPETITIONS": 3,
                    "SESSION": "test",
                    "rows": rows,
                    "save": save,
                    "checkpoint": lambda: None,
                    "display": lambda text: None,
                    "Markdown": str,
                },
            )
            conclusion = json.loads((output / "conclusions.json").read_text())
            self.assertEqual(conclusion["必要对照"], "INCONCLUSIVE")
            self.assertEqual(conclusion["Phase1出口"], "NOT_PASSED")

    def test_formal_recorder_exports_analyzable_gateway_ledger(self):
        app = create_app(GatewaySettings(model_id="Qwen/Qwen2.5-0.5B-Instruct"))
        baseline = json.loads((ROOT / "config/model.json").read_text())
        environment = {
            "gpus": [
                {
                    "name": "test-gpu",
                    "memory_total_bytes": 16 * 1024**3,
                    "driver": "test-driver",
                }
            ],
            "gpu_count": 1,
            "cuda": "test-cuda",
            "pytorch": "test-torch",
            "python": {"version": "3.13.15"},
            "model": {
                "id": baseline["model_id"],
                "revision": baseline["model_revision"],
                "tokenizer_revision": baseline["tokenizer_revision"],
                "context_limit": 8192,
            },
        }
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            output = Path(directory) / "formal"
            args = argparse.Namespace(
                root=ROOT,
                output=output,
                base_url="http://testserver",
                tenant="team-a",
                tenants=["team-a", "team-b"],
                contexts=[8, 16, 8, 16],
                concurrency=2,
                wave_size=2,
                workload_profile="mixed-policy-v2",
                admission="strict",
                scheduler="fcfs",
                prefix_mode="blind",
                repetition_index=0,
                run_id="record-test",
                seed=7,
                max_tokens=256,
                timeout_ms=30000,
                arrival_spacing_ms=0,
                dtype="float16",
                model=baseline["model_id"],
                executor="TorchExecutor",
                warmup=False,
                trace_id="test-trace",
            )

            def post(url, payload, headers, timeout):
                response = client.post(url, json=payload, headers=headers)
                return response.status_code, response.json()

            def get(url, tenant):
                return client.get(url, headers={"X-Tenant-ID": tenant}).json()

            with (
                patch.object(
                    formal_gpu_run, "collect_environment", return_value=environment
                ),
                patch.object(formal_gpu_run, "validate_environment", return_value=[]),
                patch.object(
                    formal_gpu_run.metadata, "version", return_value="test-version"
                ),
                patch.object(formal_gpu_run, "_post", side_effect=post),
                patch.object(formal_gpu_run, "_get", side_effect=get),
            ):
                self.assertEqual(formal_gpu_run.run(args), 0)
                with self.assertRaises(FileExistsError):
                    formal_gpu_run.run(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["terminal_counts"], {"FINISHED": 4})
            observations = json.loads((output / "observations.json").read_text())
            self.assertGreater(observations["wall_throughput_tokens_per_s"], 0)
            requests = observations["requests"]
            self.assertGreaterEqual(
                min(row["actual_arrival_ms"] for row in requests[2:]),
                max(row["completed_ms"] for row in requests[:2]),
            )
            self.assertEqual(
                [row["input"]["max_tokens"] for row in requests], [256, 256, 128, 128]
            )
            self.assertEqual(observations["load_mode"], "closed_loop_waves")
            self.assertTrue(
                all(
                    row["query"]["ledger"]["cost_is_estimate"]
                    for row in observations["requests"]
                )
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotEqual(manifest["software"]["transformers"], "unknown")
            self.assertTrue(manifest["trace_id"].startswith("test-trace-"))

    def test_sim_evidence_is_reproducible_and_releases_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            (reference / "manifest.json").write_text(
                json.dumps(
                    test_experiment_protocol.ExperimentProtocolTests.valid_manifest()
                )
            )
            for name in ("first", "second"):
                sim_run(root / name, reference)
            first = root / "first"
            self.assertEqual(
                (first / "requests.jsonl").read_text(),
                (root / "second/requests.jsonl").read_text(),
            )
            summary = json.loads((first / "summary.json").read_text())
            self.assertTrue(summary["simulation"]["is_simulated"])
            self.assertIsNotNone(summary["metrics"]["tpot_ms"]["p99"])
            snapshot = json.loads((first / "sim_snapshot.json").read_text())
            self.assertEqual(snapshot["stats"]["current_logical_blocks"], 0)
            manifest = json.loads((first / "manifest.json").read_text())
            record = json.loads((first / "requests.jsonl").read_text().splitlines()[0])
            record["estimated_cost"] = -1
            with self.assertRaises(ProtocolError):
                validate_request(record, 1, manifest)
            record["estimated_cost"] = None
            record["cost_is_estimate"] = False
            with self.assertRaises(ProtocolError):
                validate_request(record, 1, manifest)
