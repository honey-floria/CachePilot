#!/usr/bin/env python3
"""校验并汇总 CachePilot 实验产物。

分析器刻意不依赖第三方包，因此可在仅 CPU 的 Phase 0 环境中使用。
当运行清单缺少必需的来源信息，或 JSONL 记录不符合协议时，该运行将被判定为无效。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


class ProtocolError(ValueError):
    """实验产物违反 v1 协议时抛出的异常。"""


FLOAT_FIELDS = ("queue_ms", "ttft_ms", "tpot_ms", "total_ms")
REQUIRED_MANIFEST = (
    "artifact_type", "schema_version", "run_id", "trace_id", "created_at_utc",
    "clock", "seed", "repetition_index", "warmup", "hardware", "software",
    "model", "strategy",
)
REQUIRED_REQUEST = (
    "record_type", "schema_version", "run_id", "request_id", "tenant_id", "seed",
    "arrival_ms", "prompt_tokens", "expected_output_tokens", "completion_tokens",
    "terminal_state", *FLOAT_FIELDS, "worker_id", "logical_hit", "physical_hit",
    "reserved_blocks_peak", "estimated_gpu_seconds",
)
OPTIONAL_REQUEST = ("admission_reason", "timeline")
REQUIRED_TRACE = (
    "trace_version", "request_id", "tenant_id", "arrival_ms", "prompt_tokens",
    "expected_output_tokens", "seed",
)
TERMINAL_STATES = {"FINISHED", "CANCELLED", "TIMED_OUT", "REJECTED", "FAILED"}
FLOATING_VALUES = {"latest", "main", "master", "nightly", "dev"}
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _required(obj: dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ProtocolError(f"{where}: missing required field(s): {', '.join(missing)}")


def _nonempty_string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{where}: expected a non-empty string")


def _version_string(value: Any, where: str) -> None:
    _nonempty_string(value, where)
    if value.strip().lower() in FLOATING_VALUES:
        raise ProtocolError(
            f"{where}: floating version value is not allowed: {value!r}"
        )


def _integer(value: Any, where: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{where}: expected integer >= {minimum}")


def _number_or_null(value: Any, where: str) -> None:
    invalid = (
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            or not math.isfinite(value)
        )
    )
    if invalid:
        raise ProtocolError(f"{where}: expected a finite number >= 0 or null")


def _number(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{where}: expected a number")
    if value < 0 or not math.isfinite(value):
        raise ProtocolError(f"{where}: expected a finite number >= 0")


def _unexpected(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ProtocolError(f"{where}: unexpected field(s): {', '.join(extra)}")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ProtocolError("manifest: expected a JSON object")
    _unexpected(manifest, set(REQUIRED_MANIFEST), "manifest")
    _required(manifest, REQUIRED_MANIFEST, "manifest")
    if (
        manifest["artifact_type"] != "cachepilot_experiment_manifest"
        or manifest["schema_version"] != 1
    ):
        raise ProtocolError(
            "manifest: artifact_type/schema_version must identify protocol v1"
        )
    for field in ("run_id", "trace_id", "created_at_utc"):
        _nonempty_string(manifest[field], f"manifest.{field}")
    _integer(manifest["seed"], "manifest.seed")
    _integer(manifest["repetition_index"], "manifest.repetition_index")
    if not isinstance(manifest["warmup"], bool):
        raise ProtocolError("manifest.warmup: expected boolean")

    clock = manifest["clock"]
    if not isinstance(clock, dict):
        raise ProtocolError("manifest.clock: expected object")
    _unexpected(
        clock,
        {"event_clock", "duration_unit", "arrival_origin", "wall_clock_role"},
        "manifest.clock",
    )
    expected_clock = {
        "event_clock": "monotonic_ns",
        "duration_unit": "ms",
        "arrival_origin": "trace_zero",
        "wall_clock_role": "metadata_only",
    }
    for key, expected in expected_clock.items():
        if clock.get(key) != expected:
            raise ProtocolError(f"manifest.clock.{key}: expected {expected!r}")

    groups = {
        "hardware": (
            "host", "platform", "cpu", "gpu", "gpu_count", "gpu_memory_bytes",
            "driver", "cuda", "topology",
        ),
        "software": (
            "cachepilot", "python", "os", "executor", "torch", "transformers",
            "vllm", "git_commit",
        ),
        "model": (
            "id", "revision", "tokenizer_revision", "dtype", "quantization",
            "context_limit",
        ),
        "strategy": (
            "version", "executor", "admission", "scheduler", "router",
            "prefix_mode",
        ),
    }
    for group, fields in groups.items():
        value = manifest[group]
        if not isinstance(value, dict):
            raise ProtocolError(f"manifest.{group}: expected object")
        _unexpected(value, set(fields), f"manifest.{group}")
        _required(value, fields, f"manifest.{group}")
        for field in fields:
            if field in {"gpu_count", "gpu_memory_bytes", "context_limit"}:
                _integer(value[field], f"manifest.{group}.{field}")
            elif (
                group == "software"
                and field in {
                    "cachepilot", "python", "torch", "transformers", "vllm",
                    "git_commit",
                }
            ) or (group == "strategy" and field == "version") or (
                group == "hardware" and field in {"driver", "cuda"}
            ):
                _version_string(value[field], f"manifest.{group}.{field}")
            else:
                _nonempty_string(value[field], f"manifest.{group}.{field}")
    model = manifest["model"]
    for field in ("revision", "tokenizer_revision"):
        if not SHA40.fullmatch(model[field]):
            raise ProtocolError(
                f"manifest.model.{field}: expected a 40-character lowercase commit SHA"
            )
    return manifest


def validate_trace(record: Any, line: int) -> dict[str, Any]:
    where = f"trace line {line}"
    if not isinstance(record, dict):
        raise ProtocolError(f"{where}: expected a JSON object")
    _unexpected(record, set(REQUIRED_TRACE) | {
        "priority", "max_new_tokens", "prefix_group", "cancel_after_ms"
    }, where)
    _required(record, REQUIRED_TRACE, where)
    if record["trace_version"] != 1:
        raise ProtocolError(f"{where}.trace_version: expected 1")
    for field in ("request_id", "tenant_id"):
        _nonempty_string(record[field], f"{where}.{field}")
    for field in ("arrival_ms", "prompt_tokens", "expected_output_tokens", "seed"):
        _integer(record[field], f"{where}.{field}")
    if "priority" in record and record["priority"] not in {"interactive", "batch"}:
        raise ProtocolError(f"{where}.priority: expected interactive or batch")
    if "prefix_group" in record and record["prefix_group"] is not None:
        _nonempty_string(record["prefix_group"], f"{where}.prefix_group")
    for field in ("max_new_tokens", "cancel_after_ms"):
        if field in record and record[field] is not None:
            _integer(
                record[field], f"{where}.{field}",
                1 if field == "max_new_tokens" else 0,
            )
    return record


def validate_request(
    record: Any, line: int, manifest: dict[str, Any]
) -> dict[str, Any]:
    where = f"request line {line}"
    if not isinstance(record, dict):
        raise ProtocolError(f"{where}: expected a JSON object")
    _unexpected(record, set(REQUIRED_REQUEST) | set(OPTIONAL_REQUEST), where)
    _required(record, REQUIRED_REQUEST, where)
    if record["record_type"] != "request" or record["schema_version"] != 1:
        raise ProtocolError(
            f"{where}: record_type/schema_version must identify protocol v1"
        )
    for field in ("run_id", "request_id", "tenant_id"):
        _nonempty_string(record[field], f"{where}.{field}")
    if record["run_id"] != manifest["run_id"]:
        raise ProtocolError(f"{where}.run_id: does not match manifest.run_id")
    _integer(record["seed"], f"{where}.seed")
    for field in ("prompt_tokens", "expected_output_tokens"):
        _integer(record[field], f"{where}.{field}")
    _number(record["arrival_ms"], f"{where}.arrival_ms")
    if record["terminal_state"] not in TERMINAL_STATES:
        raise ProtocolError(f"{where}.terminal_state: unknown terminal state")
    if record["completion_tokens"] is not None:
        _integer(record["completion_tokens"], f"{where}.completion_tokens")
    for field in FLOAT_FIELDS + ("reserved_blocks_peak", "estimated_gpu_seconds"):
        _number_or_null(record[field], f"{where}.{field}")
    if not isinstance(record["logical_hit"], bool) or (
        record["physical_hit"] is not True
        and record["physical_hit"] is not False
        and record["physical_hit"] is not None
    ):
        raise ProtocolError(
            f"{where}: logical_hit/physical_hit must be boolean "
            "(physical_hit may be null)"
        )
    if record["worker_id"] is not None:
        _nonempty_string(record["worker_id"], f"{where}.worker_id")
    if "admission_reason" in record and record["admission_reason"] is not None:
        _nonempty_string(record["admission_reason"], f"{where}.admission_reason")
    if "timeline" in record:
        _validate_timeline(record["timeline"], where)
    return record


def _validate_timeline(value: Any, where: str) -> None:
    """校验可选逐阶段时间线，允许执行器写入额外阶段字段。"""

    if not isinstance(value, list):
        raise ProtocolError(f"{where}.timeline: expected an array")
    for index, event in enumerate(value):
        event_where = f"{where}.timeline[{index}]"
        if not isinstance(event, dict):
            raise ProtocolError(f"{event_where}: expected an object")
        _required(event, ("phase", "start_ms", "end_ms"), event_where)
        _nonempty_string(event["phase"], f"{event_where}.phase")
        _number(event["start_ms"], f"{event_where}.start_ms")
        _number(event["end_ms"], f"{event_where}.end_ms")
        if event["end_ms"] < event["start_ms"]:
            raise ProtocolError(f"{event_where}: end_ms precedes start_ms")


def read_jsonl(path: Path, validator) -> list[dict[str, Any]]:
    records = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ProtocolError(
                    f"{path}: blank line at {line_number} is not allowed"
                )
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    f"{path} line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            records.append(validator(value, line_number))
    if not records:
        raise ProtocolError(f"{path}: JSONL must contain at least one record")
    return records


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _metric(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(record[field]) for record in records if record[field] is not None]
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
    }


def summarize(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    source: dict[str, str],
) -> dict[str, Any]:
    measured = [record for record in records if not manifest["warmup"]]
    counts = Counter(record["terminal_state"] for record in measured)
    completion_values = [record["completion_tokens"] or 0 for record in measured]
    total_values = [
        record["total_ms"]
        for record in measured
        if record["total_ms"] is not None and record["total_ms"] > 0
    ]
    throughput = (
        sum(completion_values) / (max(total_values) / 1000.0)
        if total_values
        else None
    )
    tenant_tokens: defaultdict[str, int] = defaultdict(int)
    for record in measured:
        tenant_tokens[record["tenant_id"]] += record["completion_tokens"] or 0
    shares = list(tenant_tokens.values())
    fairness = (
        sum(shares) ** 2 / (len(shares) * sum(value * value for value in shares))
        if shares and sum(shares) > 0
        else None
    )
    denominator = len(measured) or 1
    timelines = []
    for record in measured:
        timelines.append(
            {
                "request_id": record["request_id"],
                "tenant_id": record["tenant_id"],
                "arrival_ms": record["arrival_ms"],
                "terminal_state": record["terminal_state"],
                "admission_reason": record.get("admission_reason"),
                "timeline": record.get("timeline", _derived_timeline(record)),
                "reserved_blocks_peak": record["reserved_blocks_peak"],
            }
        )
    admission_reasons = Counter(
        record.get("admission_reason", "unknown") for record in measured
    )
    peak_values = [
        record["reserved_blocks_peak"]
        for record in measured
        if record["reserved_blocks_peak"] is not None
    ]
    strategy = manifest["strategy"]
    executor = strategy["executor"].lower()
    simulated = "sim" in executor
    return {
        "artifact_type": "cachepilot_experiment_summary",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "trace_id": manifest["trace_id"],
        "seed": manifest["seed"],
        "repetition_index": manifest["repetition_index"],
        "warmup": manifest["warmup"],
        "request_count": len(records),
        "measured_request_count": len(measured),
        "terminal_counts": dict(sorted(counts.items())),
        "metrics": {field: _metric(measured, field) for field in FLOAT_FIELDS},
        "throughput_completion_tokens_per_s": throughput,
        "rejection_rate": (counts["REJECTED"] / denominator),
        "cancellation_rate": (
            (counts["CANCELLED"] + counts["TIMED_OUT"]) / denominator
        ),
        "fairness_jain": fairness,
        "request_timelines": timelines,
        "admission_reasons": dict(sorted(admission_reasons.items())),
        "resource_peaks": {
            "reserved_blocks_peak": max(peak_values) if peak_values else None,
            "estimated_gpu_seconds_peak": max(
                (record["estimated_gpu_seconds"] for record in measured
                 if record["estimated_gpu_seconds"] is not None),
                default=None,
            ),
        },
        "simulation": {
            "is_simulated": simulated,
            "executor": strategy["executor"],
            "label": "simulated" if simulated else "measured",
        },
        "control_variables": {
            "trace_id": manifest["trace_id"],
            "seed": manifest["seed"],
            "model_id": manifest["model"]["id"],
            "model_revision": manifest["model"]["revision"],
            "tokenizer_revision": manifest["model"]["tokenizer_revision"],
            "executor": strategy["executor"],
            "admission": strategy["admission"],
            "scheduler": strategy["scheduler"],
            "router": strategy["router"],
            "prefix_mode": strategy["prefix_mode"],
        },
        "quantile_method": "nearest_rank",
        "source": source,
    }


def _derived_timeline(record: dict[str, Any]) -> list[dict[str, Any]]:
    """从协议阶段耗时构造标准时间线；缺失阶段不会伪造时间。"""

    arrival = float(record["arrival_ms"])
    queue = record["queue_ms"]
    ttft = record["ttft_ms"]
    total = record["total_ms"]
    if queue is None or total is None:
        return []
    events = [{"phase": "queue", "start_ms": arrival, "end_ms": arrival + queue}]
    if ttft is not None:
        prefill_start = arrival + queue
        events.append({"phase": "prefill", "start_ms": prefill_start,
                       "end_ms": prefill_start + ttft})
        decode_start = prefill_start + ttft
        decode_end = arrival + total
        if decode_end >= decode_start:
            events.append({"phase": "decode", "start_ms": decode_start,
                           "end_ms": decode_end})
    events.append({"phase": "request", "start_ms": arrival,
                   "end_ms": arrival + total})
    return events


def analyze(
    manifest_path: Path, requests_path: Path, trace_path: Path
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid manifest {manifest_path}: {exc}") from exc
    validate_manifest(manifest)
    records = read_jsonl(
        requests_path,
        lambda value, line: validate_request(value, line, manifest),
    )
    request_ids = [record["request_id"] for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ProtocolError("requests: duplicate request_id")
    trace = read_jsonl(trace_path, validate_trace)
    trace_ids = [record["request_id"] for record in trace]
    if len(set(trace_ids)) != len(trace_ids):
        raise ProtocolError("trace: duplicate request_id")
    if set(trace_ids) != set(request_ids):
        raise ProtocolError("trace and requests: request_id sets differ")
    source = {"manifest": str(manifest_path), "requests": str(requests_path)}
    source["trace"] = str(trace_path)
    return summarize(
        manifest,
        records,
        source,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.manifest, args.requests, args.trace)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ProtocolError as exc:
        print(f"EXPERIMENT_INVALID: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"EXPERIMENT_WRITE_FAILED: {exc}", file=sys.stderr)
        return 3
    print(f"EXPERIMENT_VALID: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
