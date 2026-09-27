#!/usr/bin/env python3
"""在能力语义一致时比较多个 CachePilot 实验汇总。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from cachepilot.executor_capabilities import (
    CapabilityError,
    capabilities_for_executor,
    require_comparable_metric,
)


class ComparisonError(ValueError):
    """汇总缺失字段或指标语义不可比较。"""


SUMMARY_METRICS = frozenset(
    {
        "queue_ms",
        "ttft_ms",
        "tpot_ms",
        "total_ms",
        "throughput_completion_tokens_per_s",
        "cancellation_rate",
        "reserved_blocks_peak",
    }
)


def compare_summaries(
    metric: str,
    summaries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """校验语义并返回不聚合原始运行的比较清单。"""

    if metric not in SUMMARY_METRICS:
        raise ComparisonError("unsupported summary metric: {0}".format(metric))
    summary_items = tuple(summaries)
    if len(summary_items) < 2:
        raise ComparisonError("comparison requires at least two summaries")
    executors = []
    runs = []
    controls = None
    for index, summary in enumerate(summary_items):
        run_id, executor, run_controls = _validate_summary(summary, index)
        if controls is None:
            controls = run_controls
        elif run_controls != controls:
            raise ComparisonError(
                "summary control variables differ for run {0}".format(run_id)
            )
        executors.append(executor)
        runs.append(
            {
                "run_id": run_id,
                "executor": executor,
                "value": _metric_value(summary, metric),
            }
        )
    try:
        signature = require_comparable_metric(metric, executors)
    except CapabilityError as exc:
        raise ComparisonError(str(exc)) from exc
    return {
        "artifact_type": "cachepilot_metric_comparison",
        "schema_version": 1,
        "metric": metric,
        "semantic_signature": signature,
        "runs": runs,
    }


def _validate_summary(
    summary: dict[str, Any],
    index: int,
) -> tuple[str, str, tuple[object, ...]]:
    where = "summary[{0}]".format(index)
    if not isinstance(summary, dict):
        raise ComparisonError("{0}: expected object".format(where))
    if summary.get("artifact_type") != "cachepilot_experiment_summary":
        raise ComparisonError("{0}: not a CachePilot experiment summary".format(where))
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ComparisonError("{0}.run_id: expected non-empty string".format(where))
    control = summary.get("control_variables")
    simulation = summary.get("simulation")
    if not isinstance(control, dict) or not isinstance(simulation, dict):
        raise ComparisonError(
            "{0}: control_variables and simulation are required".format(where)
        )
    executor = control.get("executor")
    if not isinstance(executor, str):
        raise ComparisonError("{0}: executor is required".format(where))
    try:
        capability = capabilities_for_executor(executor)
    except CapabilityError as exc:
        raise ComparisonError(str(exc)) from exc
    expected_simulated = capability.timing == "simulated_logical_clock"
    if simulation.get("executor") != executor or (
        simulation.get("is_simulated") is not expected_simulated
    ):
        raise ComparisonError(
            "{0}: simulation label conflicts with executor capabilities".format(where)
        )
    control_keys = (
        "trace_id",
        "seed",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    )
    try:
        controls = tuple(control[key] for key in control_keys)
    except KeyError as exc:
        raise ComparisonError(
            "{0}.control_variables: missing {1}".format(where, exc.args[0])
        ) from exc
    return run_id, executor, controls


def _metric_value(summary: dict[str, Any], metric: str) -> Any:
    if metric in {"queue_ms", "ttft_ms", "tpot_ms", "total_ms"}:
        container = summary.get("metrics")
    elif metric == "reserved_blocks_peak":
        container = summary.get("resource_peaks")
    else:
        container = summary
    if not isinstance(container, dict) or metric not in container:
        raise ComparisonError("summary is missing metric {0}".format(metric))
    return container[metric]


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError("cannot read summary {0}: {1}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ComparisonError("summary {0}: expected object".format(path))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=sorted(SUMMARY_METRICS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        comparison = compare_summaries(
            args.metric,
            (_read_summary(path) for path in args.summaries),
        )
        payload = json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.write_text(payload, encoding="utf-8")
    except ComparisonError as exc:
        print("COMPARISON_INVALID: {0}".format(exc), file=sys.stderr)
        return 2
    print("COMPARISON_VALID", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
