#!/usr/bin/env python3
"""Record one formal GPU experiment run from the live CachePilot API."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from benchmarks.gpu_environment import collect_environment, validate_environment


def _get(url: str, tenant: str, timeout: float = 300.0) -> dict:
    request = Request(url, headers={"Accept": "application/json", "X-Tenant-ID": tenant})
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _post(url: str, payload: dict, headers: dict[str, str], timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.status, json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if hasattr(exc, "code"):
            return int(exc.code), json.loads(exc.read().decode("utf-8"))
        raise


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    environment = collect_environment(root)
    failures = validate_environment(environment, require_torch=True)
    if failures:
        raise RuntimeError("GPU environment rejected: " + "; ".join(failures))
    base = args.base_url.rstrip("/")
    _get(base + "/readyz", args.tenant)
    strategy_id = f"{args.admission}-{args.scheduler}-{args.prefix_mode}"
    run_id = args.run_id or f"{strategy_id}-{args.repetition_index}-{uuid.uuid4().hex[:8]}"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    traces = []
    for index, prompt_target in enumerate(args.contexts):
        request_id = f"{run_id}-{index:04d}"
        prompt = "benchmark " * prompt_target
        trace = {
            "trace_version": 1, "request_id": request_id, "tenant_id": args.tenant,
            "arrival_ms": index * args.arrival_spacing_ms,
            "prompt_tokens": prompt_target, "expected_output_tokens": args.max_tokens,
            "max_new_tokens": args.max_tokens, "priority": "interactive", "seed": args.seed + index,
        }
        traces.append(trace)
        status, body = _post(
            base + "/v1/chat/completions",
            {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "stream": False},
            {"X-Tenant-ID": args.tenant, "X-Request-ID": request_id, "X-Deadline-Ms": str(args.timeout_ms)},
            args.timeout_ms / 1000,
        )
        query = _get(base + f"/v1/requests/{request_id}", args.tenant)
        telemetry = query.get("telemetry") or {}
        ledger = query.get("ledger") or {}
        actual_prompt_tokens = telemetry.get("prompt_tokens")
        if actual_prompt_tokens is None:
            actual_prompt_tokens = (query.get("usage") or {}).get("prompt_tokens", prompt_target)
        trace["prompt_tokens"] = int(actual_prompt_tokens)
        terminal_state = telemetry.get("terminal_state") or query.get("state") or "FAILED"
        if terminal_state == "SUCCEEDED":
            terminal_state = "FINISHED"
        record = {
            "record_type": "request", "schema_version": 1, "run_id": run_id,
            "request_id": request_id, "tenant_id": args.tenant, "seed": args.seed + index,
            "arrival_ms": trace["arrival_ms"], "prompt_tokens": int(actual_prompt_tokens),
            "expected_output_tokens": args.max_tokens, "completion_tokens": telemetry.get("completion_tokens", 0),
            "terminal_state": terminal_state,
            "queue_ms": telemetry.get("queue_ms"), "ttft_ms": telemetry.get("ttft_ms"),
            "tpot_ms": telemetry.get("tpot_ms"), "total_ms": telemetry.get("total_ms"),
            "worker_id": telemetry.get("worker_id"), "logical_hit": bool(telemetry.get("logical_hit", False)),
            "physical_hit": telemetry.get("physical_hit"), "reserved_blocks_peak": telemetry.get("logical_kv_blocks_peak"),
            "estimated_gpu_seconds": ledger.get("estimated_gpu_seconds"), "estimated_cost": ledger.get("estimated_cost"),
            "admission_reason": telemetry.get("admission_reason"), "error_code": telemetry.get("error_code"),
            "error_stage": telemetry.get("error_stage"), "timeline": telemetry.get("timeline", []),
        }
        records.append(record)
        print(request_id, status, record["terminal_state"])
    gpu = environment["gpus"][0] if environment["gpus"] else {}
    manifest = {
        "artifact_type": "cachepilot_experiment_manifest", "schema_version": 1,
        "run_id": run_id, "trace_id": args.trace_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clock": {"event_clock": "monotonic_ns", "duration_unit": "ms", "arrival_origin": "trace_zero", "wall_clock_role": "metadata_only"},
        "seed": args.seed, "repetition_index": args.repetition_index, "warmup": args.warmup,
        "hardware": {"host": platform.node(), "platform": platform.platform(), "cpu": platform.processor(), "gpu": gpu.get("name", "unknown"), "gpu_count": environment["gpu_count"], "gpu_memory_bytes": gpu.get("memory_total_bytes", 0), "driver": gpu.get("driver", "unknown"), "cuda": environment.get("cuda") or "unknown", "topology": "single_gpu"},
        "software": {"cachepilot": "0.1.0", "python": environment["python"]["version"], "os": platform.platform(), "executor": args.executor, "torch": environment.get("pytorch") or "unknown", "transformers": "unknown", "vllm": "not_installed", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()},
        "model": {"id": environment["model"]["id"], "revision": environment["model"]["revision"], "tokenizer_revision": environment["model"]["tokenizer_revision"], "dtype": args.dtype, "quantization": "none", "context_limit": environment["model"]["context_limit"]},
        "strategy": {"version": "1", "executor": args.executor, "admission": args.admission, "scheduler": args.scheduler, "prefix_mode": args.prefix_mode},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (output / "trace.jsonl").open("w") as handle:
        for item in traces: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output / "requests.jsonl").open("w") as handle:
        for item in records: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, "benchmarks/analyze.py", "--manifest", str(output / "manifest.json"), "--trace", str(output / "trace.jsonl"), "--requests", str(output / "requests.jsonl"), "--output", str(output / "summary.json")], cwd=root, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tenant", default="team-a")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--executor", default="TorchExecutor")
    parser.add_argument("--admission", choices=("strict", "adaptive"), required=True)
    parser.add_argument("--scheduler", choices=("fcfs", "wfq"), required=True)
    parser.add_argument("--prefix-mode", choices=("blind", "aware"), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--trace-id", default="gpu-formal-trace-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetition-index", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32, 128, 512, 2048])
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=300000)
    parser.add_argument("--arrival-spacing-ms", type=int, default=0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
