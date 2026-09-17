import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPOSITORY_ROOT / "benchmarks" / "analyze.py"
SCHEMA_PATH = REPOSITORY_ROOT / "config" / "experiment.schema.json"


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "cachepilot_experiment_analyzer", ANALYZER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentProtocolTests(unittest.TestCase):
    def test_machine_readable_schema_is_valid_json(self):
        payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            "https://json-schema.org/draft/2020-12/schema", payload["$schema"]
        )
        self.assertIn("Manifest", payload["$defs"])
        self.assertIn("Summary", payload["$defs"])

    def test_manifest_without_hardware_is_invalid(self):
        analyzer = load_analyzer()
        manifest = self.valid_manifest()
        del manifest["hardware"]
        with self.assertRaises(analyzer.ProtocolError):
            analyzer.validate_manifest(manifest)

    def test_analyzer_generates_summary_from_protocol_records(self):
        analyzer = load_analyzer()
        manifest = self.valid_manifest()
        request = {
            "record_type": "request",
            "schema_version": 1,
            "run_id": "run-1",
            "request_id": "request-1",
            "tenant_id": "tenant-a",
            "seed": 7,
            "arrival_ms": 0,
            "prompt_tokens": 8,
            "expected_output_tokens": 16,
            "completion_tokens": 16,
            "terminal_state": "FINISHED",
            "queue_ms": 1.0,
            "ttft_ms": 2.0,
            "tpot_ms": 0.5,
            "total_ms": 10.0,
            "worker_id": "sim-0",
            "logical_hit": False,
            "physical_hit": None,
            "reserved_blocks_peak": 1,
            "estimated_gpu_seconds": 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            trace_path = root / "trace.jsonl"
            requests_path = root / "requests.jsonl"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            trace_path.write_text(
                json.dumps(
                    {
                        "trace_version": 1,
                        "request_id": "request-1",
                        "tenant_id": "tenant-a",
                        "arrival_ms": 0,
                        "prompt_tokens": 8,
                        "expected_output_tokens": 16,
                        "seed": 7,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            requests_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            summary = analyzer.analyze(manifest_path, requests_path, trace_path)
        self.assertEqual(1, summary["request_count"])
        self.assertEqual(2.0, summary["metrics"]["ttft_ms"]["p50"])
        self.assertEqual("nearest_rank", summary["quantile_method"])

    @staticmethod
    def valid_manifest():
        sha = "a" * 40
        return {
            "artifact_type": "cachepilot_experiment_manifest",
            "schema_version": 1,
            "run_id": "run-1",
            "trace_id": "trace-1",
            "created_at_utc": "2026-09-18T00:00:00Z",
            "clock": {
                "event_clock": "monotonic_ns",
                "duration_unit": "ms",
                "arrival_origin": "trace_zero",
                "wall_clock_role": "metadata_only",
            },
            "seed": 7,
            "repetition_index": 0,
            "warmup": False,
            "hardware": {
                "host": "host",
                "platform": "macOS",
                "cpu": "cpu",
                "gpu": "none",
                "gpu_count": 0,
                "gpu_memory_bytes": 0,
                "driver": "not_installed",
                "cuda": "not_installed",
                "topology": "not_applicable",
            },
            "software": {
                "cachepilot": "0.1.0",
                "python": "3.13.15",
                "os": "macOS",
                "executor": "SimExecutor",
                "torch": "not_installed",
                "transformers": "not_installed",
                "vllm": "not_installed",
                "git_commit": sha,
            },
            "model": {
                "id": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": sha,
                "tokenizer_revision": sha,
                "dtype": "fp32",
                "quantization": "none",
                "context_limit": 8192,
            },
            "strategy": {
                "version": "1",
                "executor": "SimExecutor",
                "admission": "strict",
                "scheduler": "fcfs",
                "router": "single",
                "prefix_mode": "blind",
            },
        }
