#!/usr/bin/env python3
"""导出单机指标快照、实验汇总和可读报告。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_analyzer = importlib.import_module("benchmarks.analyze")
ProtocolError = _analyzer.ProtocolError
analyze = _analyzer.analyze


class ExportError(ValueError):
    """现场快照或实验目录无法安全导出。"""


def fetch_metrics(
    base_url: str,
    *,
    timeout_seconds: float,
    opener: Callable[..., Any] = urlopen,
) -> tuple[str, str]:
    if not base_url.startswith(("http://", "https://")):
        raise ExportError("base URL must use http:// or https://")
    metrics_url = urljoin(base_url.rstrip("/") + "/", "metrics")
    request = Request(
        metrics_url,
        headers={"Accept": "text/plain; version=0.0.4"},
    )
    with opener(request, timeout=timeout_seconds) as response:
        try:
            payload = response.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExportError("/metrics response is not UTF-8") from exc
    if "cachepilot_gateway_up" not in payload:
        raise ExportError("/metrics response is not a CachePilot snapshot")
    return metrics_url, payload


def parse_metric_totals(payload: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] = totals.get(name, 0.0) + value
    return totals


def render_report(
    summary: dict[str, Any],
    metric_totals: dict[str, float],
    *,
    captured_at_utc: str,
    metrics_url: str,
) -> str:
    metrics = summary["metrics"]
    capacity = metric_totals.get("cachepilot_kv_capacity_blocks")
    reserved = metric_totals.get("cachepilot_reserved_kv_blocks")
    kv_utilization = (
        reserved / capacity if reserved is not None and capacity else None
    )
    lines = [
        "# CachePilot 单机实验报告",
        "",
        "## 来源",
        "",
        "- run_id: `{0}`".format(summary["run_id"]),
        "- trace_id: `{0}`".format(summary["trace_id"]),
        "- captured_at_utc: `{0}`".format(captured_at_utc),
        "- metrics_url: `{0}`".format(metrics_url),
        "- result_kind: `{0}`".format(summary["simulation"]["label"]),
        "",
        "## 在线快照",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        "| gateway up | {0} |".format(
            _format_number(metric_totals.get("cachepilot_gateway_up"))
        ),
        "| executor healthy | {0} |".format(
            _format_number(metric_totals.get("cachepilot_executor_healthy"))
        ),
        "| active sequences | {0} |".format(
            _format_number(metric_totals.get("cachepilot_active_sequences"))
        ),
        "| reserved KV blocks | {0} |".format(_format_number(reserved)),
        "| KV utilization | {0} |".format(_format_percent(kv_utilization)),
        "| estimated GPU seconds total | {0} |".format(
            _format_number(
                metric_totals.get("cachepilot_estimated_gpu_seconds_total")
            )
        ),
        "| estimated cost total | {0} |".format(
            _format_number(metric_totals.get("cachepilot_estimated_cost_total"))
        ),
        "",
        "## 实验延迟",
        "",
        "| 指标 | count | P50 (ms) | P95 (ms) | P99 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in (
        "queue_ms",
        "ttft_ms",
        "tpot_ms",
        "prefill_ms",
        "decode_ms",
        "total_ms",
    ):
        value = metrics[field]
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} |".format(
                field,
                value["count"],
                _format_number(value["p50"]),
                _format_number(value["p95"]),
                _format_number(value["p99"]),
            )
        )
    lines.extend(
        [
            "",
            "## 结果",
            "",
            "- measured requests: `{0}`".format(summary["measured_request_count"]),
            "- terminal counts: `{0}`".format(
                json.dumps(summary["terminal_counts"], ensure_ascii=False)
            ),
            "- throughput completion tokens/s: `{0}`".format(
                _format_number(summary["throughput_completion_tokens_per_s"])
            ),
            "- rejection rate: `{0}`".format(
                _format_percent(summary["rejection_rate"])
            ),
            "- cancellation rate: `{0}`".format(
                _format_percent(summary["cancellation_rate"])
            ),
            "- fairness (Jain): `{0}`".format(
                _format_number(summary["fairness_jain"])
            ),
            "- reserved KV blocks peak: `{0}`".format(
                _format_number(summary["resource_peaks"]["reserved_blocks_peak"])
            ),
            "",
            (
                "> 在线成本为 wall-time 估算值；"
                "模拟与实测结果不可混合比较。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def export_bundle(
    *,
    base_url: str,
    run_dir: Path,
    output_dir: Path,
    timeout_seconds: float = 5.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Path]:
    manifest_path = run_dir / "manifest.json"
    requests_path = run_dir / "requests.jsonl"
    trace_path = run_dir / "trace.jsonl"
    summary = analyze(manifest_path, requests_path, trace_path)
    metrics_url, metrics_payload = fetch_metrics(
        base_url,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    captured_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    metric_totals = parse_metric_totals(metrics_payload)
    report = render_report(
        summary,
        metric_totals,
        captured_at_utc=captured_at_utc,
        metrics_url=metrics_url,
    )
    snapshot = {
        "artifact_type": "cachepilot_operational_snapshot",
        "schema_version": 1,
        "captured_at_utc": captured_at_utc,
        "metrics_url": metrics_url,
        "metrics_sha256": hashlib.sha256(metrics_payload.encode("utf-8")).hexdigest(),
        "run_id": summary["run_id"],
        "files": {
            "metrics": "metrics.prom",
            "summary": "summary.json",
            "report": "report.md",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "metrics.prom",
        "snapshot": output_dir / "snapshot.json",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }
    _write_text(paths["metrics"], metrics_payload)
    _write_json(paths["snapshot"], snapshot)
    _write_json(paths["summary"], summary)
    _write_text(paths["report"], report)
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, payload: str) -> None:
    temporary = path.with_name(".{0}.tmp".format(path.name))
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "{0:.6g}".format(value)
    return str(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return "{0:.2%}".format(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        paths = export_bundle(
            base_url=args.base_url,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    except (ExportError, ProtocolError, HTTPError, URLError, OSError) as exc:
        print("DIAGNOSTICS_EXPORT_FAILED: {0}".format(exc), file=sys.stderr)
        return 2
    print("DIAGNOSTICS_EXPORTED: {0}".format(paths["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
