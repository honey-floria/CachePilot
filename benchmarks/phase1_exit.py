#!/usr/bin/env python3
"""Validate the evidence required for the Phase 1 exit gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("ttft_ms", "tpot_ms", "total_ms")


class Phase1Error(ValueError):
    """Required Phase 1 evidence is missing or inconsistent."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase1Error(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase1Error(f"{path}: expected JSON object")
    return value


def validate(run_dirs: list[Path], chaos_report: Path) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise Phase1Error(
            "at least SimExecutor and one measured executor run are required"
        )
    summaries = []
    simulated = set()
    for run_dir in run_dirs:
        manifest = _read(run_dir / "manifest.json")
        summary = _read(run_dir / "summary.json")
        simulation = summary.get("simulation")
        if not isinstance(simulation, dict):
            raise Phase1Error(f"{run_dir}: summary.simulation is required")
        simulated.add(bool(simulation.get("is_simulated")))
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(field), dict)
            or metrics[field].get("p99") is None
            for field in REQUIRED_METRICS
        ):
            raise Phase1Error(f"{run_dir}: TTFT/TPOT/total P99 metrics are required")
        for field in (
            "throughput_completion_tokens_per_s", "fairness_jain", "rejection_rate",
            "cancellation_rate",
        ):
            if field not in summary:
                raise Phase1Error(f"{run_dir}: missing {field}")
        peaks = summary.get("resource_peaks")
        if not isinstance(peaks, dict) or "reserved_blocks_peak" not in peaks:
            raise Phase1Error(f"{run_dir}: reserved KV peak is required")
        requests_path = run_dir / "requests.jsonl"
        if not requests_path.is_file():
            raise Phase1Error(f"{run_dir}: requests.jsonl is required")
        records = [
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not any(
            "estimated_gpu_seconds" in record and "estimated_cost" in record
            for record in records
        ):
            raise Phase1Error(f"{run_dir}: estimated GPU seconds and cost are required")
        summaries.append({
            "run_id": summary.get("run_id"),
            "executor": manifest.get("strategy", {}).get("executor"),
            "is_simulated": bool(simulation.get("is_simulated")),
        })
    if simulated != {False, True}:
        raise Phase1Error("evidence must include both simulated and measured executors")
    chaos = _read(chaos_report)
    if chaos.get("status") != "PASS" or len(chaos.get("cases", [])) < 5:
        raise Phase1Error(
            "chaos report must pass cancellation, disconnect, timeout, "
            "exception and OOM"
        )
    return {
        "artifact_type": "cachepilot_phase1_exit_report",
        "schema_version": 1,
        "status": "PASS",
        "runs": summaries,
        "chaos_report": str(chaos_report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chaos-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if not args.skip_regression:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.integration.test_gateway_api",
             "tests.integration.test_vllm_gateway"],
            check=False,
        )
        if result.returncode:
            print(
                "PHASE1_EXIT=FAIL: API/SSE/cancel/timeout regression",
                file=sys.stderr,
            )
            return result.returncode
    try:
        report = validate(args.runs, args.chaos_report)
    except Phase1Error as exc:
        print(f"PHASE1_EXIT=FAIL: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PHASE1_EXIT=PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
