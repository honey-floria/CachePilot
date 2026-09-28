"""Run a guarded progressive load test against the real GPU service.

The runner deliberately refuses to run without the repository's single-GPU
preflight. A successful request is not a capacity result: every matrix point
is repeated and the report separates safe success, overload protection, OOM,
timeouts, and transport failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.gpu_environment import (  # noqa: E402
    collect_environment,
    validate_environment,
)


DEFAULT_CONTEXTS = (32, 128, 512, 2048, 4096, 8192)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8)
OOM_MARKERS = ("out of memory", "cuda out of memory", "cublas_status_alloc_failed")
TRACE_FIELDS = (
    "prompt_tokens", "completion_tokens", "terminal_state", "queue_ms",
    "ttft_ms", "total_ms", "error_code", "admission_reason", "reserved_blocks_peak",
)


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be positive comma-separated integers"
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "values must be positive comma-separated integers"
        )
    return values


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None,
              timeout_s: float = 300.0, extra_headers: dict[str, str] | None = None
              ) -> tuple[int, dict[str, Any], str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:  # nosec B310
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            return response.status, parsed if isinstance(parsed, dict) else {}, raw
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed if isinstance(parsed, dict) else {}, raw


def classify_result(
    status: int | None,
    body: str = "",
    exception: str | None = None,
) -> str:
    """Classify a request without retaining potentially sensitive response text."""
    lowered = body.lower()
    if any(marker in lowered for marker in OOM_MARKERS):
        return "oom"
    if exception:
        return "timeout" if "timed out" in exception.lower() else "transport_error"
    if status == 200:
        return "success"
    if status == 429:
        return "overload_protected"
    if status in (408, 504):
        return "timeout"
    if status is not None and status >= 500:
        return "failed"
    return "rejected"


def _prompt(target_tokens: int) -> str:
    return "benchmark " * target_tokens


def _one_request(base_url: str, context_tokens: int, index: int, max_tokens: int,
                 timeout_s: float) -> dict[str, Any]:
    request_id = f"progressive-{uuid.uuid4().hex}"
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": _prompt(context_tokens)}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        status, _, raw = _http_json(
            _url(base_url, "/v1/chat/completions"), method="POST", payload=payload,
            timeout_s=timeout_s,
            extra_headers={
                "X-Tenant-ID": "progressive-load",
                "X-Request-ID": request_id,
                "X-Deadline-Ms": str(int(timeout_s * 1000)),
            },
        )
        result: dict[str, Any] = {
            "request_id": request_id, "index": index,
            "requested_prompt_tokens": context_tokens, "status": status,
            "classification": classify_result(status, raw),
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if status is not None:
            trace_status, trace, _ = _http_json(
                _url(base_url, f"/v1/requests/{request_id}"),
                timeout_s=timeout_s,
                extra_headers={"X-Tenant-ID": "progressive-load"},
            )
            if trace_status == 200:
                telemetry = trace.get("telemetry", trace)
                if isinstance(telemetry, dict):
                    result["trace"] = {
                        field: telemetry.get(field)
                        for field in TRACE_FIELDS if field in telemetry
                    }
        return result
    except (TimeoutError, URLError, OSError) as exc:
        return {
            "request_id": request_id, "index": index,
            "requested_prompt_tokens": context_tokens, "status": None,
            "classification": classify_result(None, exception=str(exc)),
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def _step_summary(results: list[dict[str, Any]], context_tokens: int, concurrency: int,
                  repetition: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        key = str(result["classification"])
        counts[key] = counts.get(key, 0) + 1
    observed_prompt_tokens = [
        result["trace"]["prompt_tokens"]
        for result in results
        if isinstance(result.get("trace"), dict)
        and isinstance(result["trace"].get("prompt_tokens"), int)
    ]
    safe = len(results) == concurrency and counts == {"success": concurrency}
    return {
        "context_tokens_target": context_tokens,
        "observed_prompt_tokens": max(observed_prompt_tokens, default=None),
        "concurrency": concurrency,
        "repetition": repetition,
        "request_count": len(results),
        "counts": counts,
        "safe": safe,
        "results": results,
    }


def summarize_safety(
    steps: list[dict[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    """Aggregate matrix points; one successful repetition is never sufficient."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for step in steps:
        key = (int(step["context_tokens_target"]), int(step["concurrency"]))
        grouped.setdefault(key, []).append(step)
    points = []
    for (context_tokens, concurrency), runs in sorted(grouped.items()):
        counts: dict[str, int] = {}
        observed = [
            int(run["observed_prompt_tokens"])
            for run in runs
            if isinstance(run.get("observed_prompt_tokens"), int)
        ]
        for run in runs:
            for classification, count in run["counts"].items():
                counts[classification] = counts.get(classification, 0) + int(count)
        points.append({
            "context_tokens_target": context_tokens,
            "observed_prompt_tokens": max(observed, default=None),
            "concurrency": concurrency,
            "completed_repetitions": len(runs),
            "required_repetitions": repetitions,
            "counts": counts,
            "safe": len(runs) == repetitions and all(bool(run["safe"]) for run in runs),
        })
    return points


def _write_report(
    output_dir: Path,
    report: dict[str, Any],
    steps: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progressive_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    request_path = output_dir / "progressive_requests.jsonl"
    with request_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            for result in step["results"]:
                row = {key: value for key, value in step.items() if key != "results"}
                row.update(result)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    environment = collect_environment(root)
    failures = validate_environment(environment, require_torch=True)
    if failures:
        print("GPU_ENVIRONMENT=FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    status, ready, _ = _http_json(
        _url(args.base_url, "/readyz"), timeout_s=args.timeout
    )
    if status != 200 or ready.get("status") != "ready":
        print(f"service is not ready (status={status})", file=sys.stderr)
        return 2

    steps: list[dict[str, Any]] = []
    oom_seen = False
    for context_tokens in args.contexts:
        if oom_seen:
            break
        if args.warmup:
            _one_request(
                args.base_url, context_tokens, -1, args.max_tokens, args.timeout
            )
        for concurrency in args.concurrencies:
            if oom_seen:
                break
            for repetition in range(1, args.repetitions + 1):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as pool:
                    futures = [
                        pool.submit(
                            _one_request,
                            args.base_url,
                            context_tokens,
                            index,
                            args.max_tokens,
                            args.timeout,
                        )
                        for index in range(concurrency)
                    ]
                    results = [future.result() for future in futures]
                step = _step_summary(results, context_tokens, concurrency, repetition)
                steps.append(step)
                print(
                    f"context={context_tokens} concurrency={concurrency} "
                    f"repetition={repetition} safe={step['safe']} "
                    f"counts={step['counts']}"
                )
                if step["counts"].get("oom", 0):
                    oom_seen = True
                    break

    matrix_points = summarize_safety(steps, args.repetitions)
    safe_points = [point for point in matrix_points if point["safe"]]
    safe_context_limit = max(
        (
            point["observed_prompt_tokens"]
            for point in safe_points
            if point["observed_prompt_tokens"] is not None
        ),
        default=None,
    )
    safe_concurrency_at_limit = max(
        (
            point["concurrency"]
            for point in safe_points
            if point["observed_prompt_tokens"] == safe_context_limit
        ),
        default=None,
    )
    first_protection = next(
        (step for step in steps if step["counts"].get("overload_protected", 0)),
        None,
    )
    first_oom = next(
        (step for step in steps if step["counts"].get("oom", 0)), None
    )
    report = {
        "artifact_type": "cachepilot_progressive_load_report", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "service": {"base_url": args.base_url, "ready_status": status},
        "environment": environment,
        "configuration": {
            "contexts": list(args.contexts),
            "concurrencies": list(args.concurrencies),
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
        },
        "steps": len(steps),
        "matrix_points": matrix_points,
        "safety": {
            "definition": (
                "safe means every repeated request at a matrix point finished "
                "successfully with no protection, OOM, timeout, or failure"
            ),
            "safe_context_tokens_observed": safe_context_limit,
            "safe_concurrency_at_safe_context": safe_concurrency_at_limit,
            "first_overload_protection_point": first_protection,
            "first_oom_point": first_oom,
            "capacity_conclusion": (
                "bounded repeated matrix only; extrapolation beyond tested "
                "points is invalid"
            ),
        },
    }
    _write_report(args.output_dir, report, steps)
    print(f"PROGRESSIVE_LOAD_REPORT={args.output_dir / 'progressive_report.json'}")
    print(f"SAFE_CONTEXT_TARGET={safe_context_limit}")
    print(f"SAFE_CONCURRENCY_AT_SAFE_CONTEXT={safe_concurrency_at_limit}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/progressive-load")
    )
    parser.add_argument("--contexts", type=_parse_ints, default=DEFAULT_CONTEXTS)
    parser.add_argument(
        "--concurrencies", type=_parse_ints, default=DEFAULT_CONCURRENCIES
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("repetitions, max-tokens and timeout must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
