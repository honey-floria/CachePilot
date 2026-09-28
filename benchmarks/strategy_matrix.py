#!/usr/bin/env python3
"""Validate and compare the required strategy experiment matrix.

This tool does not start a model server.  It consumes the immutable run
artifacts produced on the GPU host and refuses to compare runs that do not
share the required controls or repetition protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from benchmarks.compare import ComparisonError, compare_summaries


REQUIRED_PAIRS = (
    ("admission", "strict", "adaptive"),
    ("scheduler", "fcfs", "wfq"),
    ("prefix_mode", "blind", "aware"),
)
CONTROL_FIELDS = (
    "trace_id", "seed", "model_id", "model_revision", "tokenizer_revision",
)
MANIFEST_FIELDS = ("model", "hardware", "software", "trace_id", "seed")


class MatrixError(ValueError):
    """The supplied runs cannot form a valid strategy comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"{path}: expected a JSON object")
    return value


def _run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(path / "manifest.json")
    summary = _read_json(path / "summary.json")
    strategy = manifest.get("strategy")
    if not isinstance(strategy, dict):
        raise MatrixError(f"{path}: manifest.strategy is required")
    for field in MANIFEST_FIELDS:
        if field not in manifest:
            raise MatrixError(f"{path}: manifest.{field} is required")
    if summary.get("run_id") != manifest.get("run_id"):
        raise MatrixError(f"{path}: summary.run_id does not match manifest")
    controls = summary.get("control_variables")
    if not isinstance(controls, dict):
        raise MatrixError(f"{path}: summary.control_variables is required")
    for field in CONTROL_FIELDS:
        if field not in controls:
            raise MatrixError(f"{path}: control_variables.{field} is required")
    return manifest, summary


def validate_matrix(run_dirs: Iterable[Path], *, min_repetitions: int = 3,
                    require_warmup: bool = True) -> dict[str, Any]:
    """Validate controls and return grouped summaries without aggregating them."""
    paths = tuple(Path(path) for path in run_dirs)
    if min_repetitions < 3:
        raise MatrixError("min_repetitions must be at least 3")
    if not paths:
        raise MatrixError("at least one run directory is required")
    loaded = [_run(path) for path in paths]
    baseline_manifest, baseline_summary = loaded[0]
    baseline_controls = tuple(
        baseline_summary["control_variables"][field] for field in CONTROL_FIELDS
    )
    baseline_identity = {
        "executor": baseline_manifest["software"]["executor"],
        "model": baseline_manifest["model"],
        "hardware": baseline_manifest["hardware"],
        "software": baseline_manifest["software"],
        "trace_id": baseline_manifest["trace_id"],
        "seed": baseline_manifest["seed"],
    }
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], bool]]] = {}
    for (manifest, summary), path in zip(loaded, paths):
        identity = {
            "executor": manifest["software"]["executor"],
            "model": manifest["model"], "hardware": manifest["hardware"],
            "software": manifest["software"], "trace_id": manifest["trace_id"],
            "seed": manifest["seed"],
        }
        if identity != baseline_identity:
            raise MatrixError(f"{path}: executor/model/hardware/trace controls differ")
        strategy = manifest["strategy"]
        strategy_key = (
            str(strategy.get("admission")), str(strategy.get("scheduler")),
            str(strategy.get("prefix_mode")),
        )
        groups.setdefault(strategy_key, []).append((summary, manifest.get("warmup") is True))
        controls = summary["control_variables"]
        if tuple(controls[field] for field in CONTROL_FIELDS) != baseline_controls:
            raise MatrixError(f"{path}: summary control variables differ")
    for field, left, right in REQUIRED_PAIRS:
        values = {str(manifest["strategy"].get(field)) for manifest, _ in loaded}
        if left not in values or right not in values:
            raise MatrixError(f"missing required {field} pair: {left} vs {right}")
    warmups = sum(1 for manifest, _ in loaded if manifest.get("warmup") is True)
    measured = len(loaded) - warmups
    if require_warmup and any(
        not any(is_warmup for _, is_warmup in strategy_runs)
        for strategy_runs in groups.values()
    ):
        raise MatrixError("each strategy requires at least one warm-up run")
    under_repeated = {
        key: sum(not is_warmup for _, is_warmup in strategy_runs)
        for key, strategy_runs in groups.items()
        if sum(not is_warmup for _, is_warmup in strategy_runs) < min_repetitions
    }
    if under_repeated:
        raise MatrixError(
            "each strategy needs at least {0} measured runs: {1}".format(
                min_repetitions, sorted(under_repeated)
            )
        )
    return {
        "artifact_type": "cachepilot_strategy_matrix",
        "schema_version": 1,
        "controls": baseline_identity,
        "run_ids": [summary["run_id"] for _, summary in loaded],
        "warmup_runs": warmups,
        "measured_runs": measured,
        "required_repetitions": min_repetitions,
        "strategies": [manifest["strategy"] for manifest, _ in loaded],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="ttft_ms")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        report = validate_matrix(args.runs)
        summaries = [_read_json(path / "summary.json") for path in args.runs]
        report["comparison"] = compare_summaries(args.metric, summaries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (MatrixError, ComparisonError) as exc:
        print(f"MATRIX_INVALID: {exc}", file=sys.stderr)
        return 2
    print(f"MATRIX_VALID: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
