"""低基数 Prometheus 指标与逐请求结构化 trace。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional


_HISTOGRAM_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)


def _escape_label(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _labels(values: Mapping[str, object]) -> str:
    if not values:
        return ""
    return "{" + ",".join(
        '{0}="{1}"'.format(key, _escape_label(values[key]))
        for key in sorted(values)
    ) + "}"


def _metric_line(
    name: str,
    value: object,
    labels: Optional[Mapping[str, object]] = None,
) -> str:
    return "{0}{1} {2}\n".format(name, _labels(labels), value)


@dataclass
class _Histogram:
    count: int = 0
    total: float = 0.0
    buckets: dict[float, int] = field(
        default_factory=lambda: {bucket: 0 for bucket in _HISTOGRAM_BUCKETS}
    )

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for bucket in self.buckets:
            if value <= bucket:
                self.buckets[bucket] += 1
                break


@dataclass(frozen=True)
class RequestTraceSnapshot:
    """不包含 prompt 内容的逐请求观测快照。"""

    request_id: str
    tenant_id: str
    model: str
    prompt_tokens: Optional[int]
    admission_status: Optional[str]
    admission_reason: Optional[str]
    logical_kv_blocks: Optional[int]
    logical_kv_blocks_peak: Optional[int]
    completion_tokens: int
    terminal_state: Optional[str]
    error_code: Optional[str]
    error_stage: Optional[str]
    queue_ms: Optional[float]
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    prefill_ms: Optional[float]
    decode_ms: Optional[float]
    total_ms: Optional[float]
    timeline: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "admission_status": self.admission_status,
            "admission_reason": self.admission_reason,
            "logical_kv_blocks": self.logical_kv_blocks,
            "logical_kv_blocks_peak": self.logical_kv_blocks_peak,
            "completion_tokens": self.completion_tokens,
            "terminal_state": self.terminal_state,
            "error_code": self.error_code,
            "error_stage": self.error_stage,
            "queue_ms": self.queue_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "total_ms": self.total_ms,
            "timeline": list(self.timeline),
        }


@dataclass
class _RequestTrace:
    request_id: str
    tenant_id: str
    model: str
    received_ns: int
    prompt_tokens: Optional[int] = None
    admission_status: Optional[str] = None
    admission_reason: Optional[str] = None
    logical_kv_blocks: Optional[int] = None
    logical_kv_blocks_peak: Optional[int] = None
    completion_tokens: int = 0
    terminal_state: Optional[str] = None
    error_code: Optional[str] = None
    error_stage: Optional[str] = None
    stages: dict[str, int] = field(default_factory=dict)
    terminal_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    timeline: list[dict[str, object]] = field(default_factory=list)


class TelemetryCollector:
    """线程安全的低基数指标和请求 trace 收集器。"""

    def __init__(
        self,
        *,
        clock_ns: Optional[Callable[[], int]] = None,
    ) -> None:
        self._clock_ns = clock_ns or time.monotonic_ns
        self._lock = threading.Lock()
        self._traces: dict[str, _RequestTrace] = {}
        self._request_counts: dict[tuple[str, str, str], int] = {}
        self._admission_counts: dict[tuple[str, str], int] = {}
        self._error_counts: dict[tuple[str, str], int] = {}
        self._histograms: dict[tuple[str, str], _Histogram] = {}
        self._invalid_requests = 0

    def start_request(
        self,
        request_id: str,
        tenant_id: str,
        model: str,
        *,
        prompt_tokens: Optional[int] = None,
    ) -> None:
        with self._lock:
            if request_id in self._traces:
                return
            self._traces[request_id] = _RequestTrace(
                request_id=request_id,
                tenant_id=tenant_id,
                model=model,
                received_ns=self._clock_ns(),
                prompt_tokens=prompt_tokens,
            )

    def record_invalid_request(self, code: str, stage: str = "validation") -> None:
        with self._lock:
            self._invalid_requests += 1
            self._increment(self._error_counts, (code, stage))

    def set_prompt_tokens(self, request_id: str, prompt_tokens: int) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is not None:
                trace.prompt_tokens = prompt_tokens

    def mark_stage(self, request_id: str, stage: str) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is not None and stage not in trace.stages:
                trace.stages[stage] = self._clock_ns()

    def record_admission(
        self,
        request_id: str,
        status: str,
        reason: str,
        *,
        logical_kv_blocks: Optional[int] = None,
    ) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is not None:
                trace.admission_status = status
                trace.admission_reason = reason
                trace.logical_kv_blocks = logical_kv_blocks
                trace.logical_kv_blocks_peak = logical_kv_blocks
                if status == "ADMITTED":
                    trace.stages.setdefault("admitted", self._clock_ns())
            self._increment(self._admission_counts, (status, reason))

    def record_reservation(self, request_id: str, blocks: int) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is not None:
                trace.logical_kv_blocks = blocks
                trace.logical_kv_blocks_peak = max(trace.logical_kv_blocks_peak or 0, blocks)

    def record_token(self, request_id: str, token_count: int) -> None:
        if type(token_count) is not int or token_count < 1:
            raise ValueError("token_count must be a positive integer")
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is None:
                return
            now_ns = self._clock_ns()
            if trace.first_token_ns is None:
                trace.first_token_ns = now_ns
                trace.stages.setdefault("first_token", now_ns)
            trace.completion_tokens += token_count

    def record_error(
        self,
        request_id: Optional[str],
        code: str,
        stage: str,
    ) -> None:
        with self._lock:
            self._increment(self._error_counts, (code, stage))
            if request_id is not None and request_id in self._traces:
                self._traces[request_id].error_code = code
                self._traces[request_id].error_stage = stage

    def finish(self, request_id: str, terminal_state: str) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is None or trace.terminal_state is not None:
                return
            now_ns = self._clock_ns()
            trace.terminal_state = terminal_state
            trace.terminal_ns = now_ns
            trace.stages.setdefault("terminal", now_ns)
            self._increment(
                self._request_counts,
                (trace.tenant_id, trace.model, terminal_state),
            )
            self._observe_trace(trace)

    def snapshot(self, request_id: str) -> Optional[RequestTraceSnapshot]:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is None:
                return None
            return self._snapshot_trace(trace)

    def traces(self) -> tuple[RequestTraceSnapshot, ...]:
        with self._lock:
            return tuple(
                self._snapshot_trace(self._traces[request_id])
                for request_id in self._traces
            )

    def render_prometheus(
        self,
        *,
        active_sequences: int,
        reserved_kv_blocks: int,
        active_sequence_capacity: Optional[int] = None,
        kv_capacity_blocks: Optional[int] = None,
        executor_healthy: Optional[bool] = None,
    ) -> str:
        with self._lock:
            lines = [
                "# TYPE cachepilot_gateway_up gauge\n",
                _metric_line("cachepilot_gateway_up", 1),
                "# TYPE cachepilot_requests_started_total counter\n",
                _metric_line(
                    "cachepilot_requests_started_total",
                    len(self._traces),
                ),
                "# TYPE cachepilot_invalid_requests_total counter\n",
                _metric_line(
                    "cachepilot_invalid_requests_total",
                    self._invalid_requests,
                ),
                "# TYPE cachepilot_requests_total counter\n",
            ]
            for (tenant, model, state), value in sorted(self._request_counts.items()):
                lines.append(
                    _metric_line(
                        "cachepilot_requests_total",
                        value,
                        {"model": model, "state": state, "tenant": tenant},
                    )
                )
            lines.extend(
                [
                    "# TYPE cachepilot_admission_total counter\n",
                ]
            )
            for (status, reason), value in sorted(self._admission_counts.items()):
                lines.append(
                    _metric_line(
                        "cachepilot_admission_total",
                        value,
                        {"reason": reason, "status": status},
                    )
                )
            lines.append("# TYPE cachepilot_errors_total counter\n")
            for (code, stage), value in sorted(self._error_counts.items()):
                lines.append(
                    _metric_line(
                        "cachepilot_errors_total",
                        value,
                        {"code": code, "stage": stage},
                    )
                )
            for metric_name, help_text in (
                ("cachepilot_queue_wait_seconds", "Queue wait duration"),
                ("cachepilot_ttft_seconds", "Time to first token"),
                ("cachepilot_tpot_seconds", "Time per output token"),
                ("cachepilot_prefill_seconds", "Prefill duration"),
                ("cachepilot_decode_seconds", "Decode duration"),
            ):
                lines.append("# HELP {0} {1}\n".format(metric_name, help_text))
                lines.append("# TYPE {0} histogram\n".format(metric_name))
                for (model, _), histogram in sorted(
                    self._histograms.items()
                ):
                    if _ != metric_name:
                        continue
                    cumulative = 0
                    for bucket in _HISTOGRAM_BUCKETS:
                        cumulative += histogram.buckets[bucket]
                        lines.append(
                            _metric_line(
                                metric_name + "_bucket",
                                cumulative,
                                {"le": bucket, "model": model},
                            )
                        )
                    lines.append(
                        _metric_line(
                            metric_name + "_bucket",
                            histogram.count,
                            {"le": "+Inf", "model": model},
                        )
                    )
                    lines.append(
                        _metric_line(
                            metric_name + "_count",
                            histogram.count,
                            {"model": model},
                        )
                    )
                    lines.append(
                        _metric_line(
                            metric_name + "_sum",
                            histogram.total,
                            {"model": model},
                        )
                    )
            lines.extend(
                [
                    "# TYPE cachepilot_active_sequences gauge\n",
                    _metric_line(
                        "cachepilot_active_sequences", active_sequences
                    ),
                    "# TYPE cachepilot_reserved_kv_blocks gauge\n",
                    _metric_line(
                        "cachepilot_reserved_kv_blocks", reserved_kv_blocks
                    ),
                    "# TYPE cachepilot_logical_kv_blocks gauge\n",
                    _metric_line(
                        "cachepilot_logical_kv_blocks",
                        reserved_kv_blocks,
                        {"state": "reserved"},
                    ),
                ]
            )
            if active_sequence_capacity is not None:
                lines.extend(
                    [
                        "# TYPE cachepilot_active_sequence_capacity gauge\n",
                        _metric_line(
                            "cachepilot_active_sequence_capacity",
                            active_sequence_capacity,
                        ),
                    ]
                )
            if kv_capacity_blocks is not None:
                lines.extend(
                    [
                        "# TYPE cachepilot_kv_capacity_blocks gauge\n",
                        _metric_line(
                            "cachepilot_kv_capacity_blocks",
                            kv_capacity_blocks,
                        ),
                    ]
                )
            if executor_healthy is not None:
                lines.extend(
                    [
                        "# TYPE cachepilot_executor_healthy gauge\n",
                        _metric_line(
                            "cachepilot_executor_healthy",
                            1 if executor_healthy else 0,
                        ),
                    ]
                )
            return "".join(lines)

    @staticmethod
    def _increment(values: dict[tuple[str, ...], int], key: tuple[str, ...]) -> None:
        values[key] = values.get(key, 0) + 1

    def _observe_trace(self, trace: _RequestTrace) -> None:
        if trace.terminal_ns is None:
            return
        queue_seconds = self._duration_seconds(
            trace.stages.get("queued"),
            trace.stages.get("admitted", trace.terminal_ns),
        )
        prefill_seconds = self._duration_seconds(
            trace.stages.get("executing"), trace.first_token_ns
        )
        decode_seconds = self._duration_seconds(
            trace.first_token_ns, trace.terminal_ns
        )
        durations = {
            "cachepilot_queue_wait_seconds": queue_seconds,
            "cachepilot_prefill_seconds": prefill_seconds,
            "cachepilot_ttft_seconds": prefill_seconds,
            "cachepilot_decode_seconds": decode_seconds,
            "cachepilot_tpot_seconds": (
                decode_seconds / (trace.completion_tokens - 1)
                if decode_seconds is not None and trace.completion_tokens > 1
                else 0.0 if decode_seconds is not None else None
            ),
        }
        for metric_name, value in durations.items():
            if value is not None:
                histogram = self._histograms.setdefault(
                    (trace.model, metric_name), _Histogram()
                )
                histogram.observe(value)
        trace.timeline = self._timeline(trace)

    @staticmethod
    def _duration_seconds(
        start_ns: Optional[int], end_ns: Optional[int]
    ) -> Optional[float]:
        if start_ns is None or end_ns is None or end_ns < start_ns:
            return None
        return (end_ns - start_ns) / 1_000_000_000

    def _snapshot_trace(self, trace: _RequestTrace) -> RequestTraceSnapshot:
        queue = self._duration_seconds(
            trace.stages.get("queued"),
            trace.stages.get("admitted", trace.terminal_ns),
        )
        prefill = self._duration_seconds(
            trace.stages.get("executing"), trace.first_token_ns
        )
        decode = self._duration_seconds(trace.first_token_ns, trace.terminal_ns)
        total = self._duration_seconds(trace.received_ns, trace.terminal_ns)
        tpot = (
            decode / (trace.completion_tokens - 1)
            if decode is not None and trace.completion_tokens > 1
            else 0.0 if decode is not None else None
        )
        return RequestTraceSnapshot(
            request_id=trace.request_id,
            tenant_id=trace.tenant_id,
            model=trace.model,
            prompt_tokens=trace.prompt_tokens,
            admission_status=trace.admission_status,
            admission_reason=trace.admission_reason,
            logical_kv_blocks=trace.logical_kv_blocks,
            logical_kv_blocks_peak=trace.logical_kv_blocks_peak,
            completion_tokens=trace.completion_tokens,
            terminal_state=trace.terminal_state,
            error_code=trace.error_code,
            error_stage=trace.error_stage,
            queue_ms=self._to_ms(queue),
            ttft_ms=self._to_ms(prefill),
            tpot_ms=self._to_ms(tpot),
            prefill_ms=self._to_ms(prefill),
            decode_ms=self._to_ms(decode),
            total_ms=self._to_ms(total),
            timeline=tuple(trace.timeline),
        )

    @staticmethod
    def _to_ms(seconds: Optional[float]) -> Optional[float]:
        return None if seconds is None else seconds * 1000.0

    @staticmethod
    def _timeline(trace: _RequestTrace) -> list[dict[str, object]]:
        phases = (
            ("queue", "queued", "admitted"),
            ("prefill", "executing", "first_token"),
            ("decode", "first_token", "terminal"),
        )
        timeline = []
        for phase, start_name, end_name in phases:
            start_ns = trace.stages.get(start_name)
            end_ns = trace.stages.get(end_name)
            if start_ns is None or end_ns is None or end_ns < start_ns:
                continue
            timeline.append(
                {
                    "phase": phase,
                    "start_ms": (start_ns - trace.received_ns) / 1_000_000,
                    "end_ms": (end_ns - trace.received_ns) / 1_000_000,
                }
            )
        if trace.terminal_ns is not None:
            timeline.append(
                {
                    "phase": "request",
                    "start_ms": 0.0,
                    "end_ms": (trace.terminal_ns - trace.received_ns)
                    / 1_000_000,
                }
            )
        return timeline
