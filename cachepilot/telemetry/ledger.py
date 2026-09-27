"""逐请求账本与可重算的估算成本。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Mapping, Optional

from .metrics import RequestTraceSnapshot


class LedgerError(ValueError):
    """账本字段或成本重算结果无效。"""


def estimate_gpu_seconds(
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
) -> Optional[float]:
    """从原始阶段耗时估算执行器占用的 GPU 秒数。"""

    durations = [value for value in (prefill_ms, decode_ms) if value is not None]
    if not durations:
        return None
    if any(value < 0 for value in durations):
        raise LedgerError("phase durations must be non-negative")
    return sum(durations) / 1000.0


def estimate_cost(
    *,
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    gpu_hour_price: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """返回 ``(estimated_gpu_seconds, estimated_cost)``。

    成本口径固定为：``(prefill_ms + decode_ms) / 1000 * price / 3600``。
    ``gpu_hour_price`` 未配置时仍保留 GPU 秒估算，但成本值为 ``None``。
    """

    if gpu_hour_price is not None and gpu_hour_price < 0:
        raise LedgerError("gpu_hour_price must be non-negative")
    gpu_seconds = estimate_gpu_seconds(prefill_ms, decode_ms)
    if gpu_seconds is None or gpu_hour_price is None:
        return gpu_seconds, None
    return gpu_seconds, gpu_hour_price * gpu_seconds / 3600.0


@dataclass(frozen=True)
class RequestLedgerRecord:
    """终态请求的原始字段、命中来源、策略和估算成本。"""

    request_id: str
    tenant_id: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: int
    queue_ms: Optional[float]
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    prefill_ms: Optional[float]
    decode_ms: Optional[float]
    total_ms: Optional[float]
    timeline: tuple[dict[str, object], ...]
    strategy_version: str
    logical_kv_blocks: Optional[int]
    reservation_peak_blocks: Optional[int]
    logical_hit: bool
    logical_hit_source: str
    physical_hit: Optional[bool]
    physical_hit_source: str
    terminal_state: str
    error_code: Optional[str]
    error_stage: Optional[str]
    gpu_hour_price: Optional[float]
    estimated_gpu_seconds: Optional[float]
    estimated_cost: Optional[float]
    cost_currency: str
    cost_basis: str
    cost_is_estimate: bool

    @classmethod
    def from_trace(
        cls,
        trace: RequestTraceSnapshot,
        *,
        strategy_version: str,
        gpu_hour_price: Optional[float] = None,
        cost_currency: str = "USD",
        logical_hit: bool = False,
        logical_hit_source: str = "not_configured",
        physical_hit: Optional[bool] = None,
        physical_hit_source: Optional[str] = None,
    ) -> "RequestLedgerRecord":
        if trace.terminal_state is None:
            raise LedgerError("ledger records require a terminal request")
        if not strategy_version:
            raise LedgerError("strategy_version must be non-empty")
        if not logical_hit_source:
            raise LedgerError("logical_hit_source must be non-empty")
        if logical_hit and logical_hit_source == "not_configured":
            raise LedgerError(
                "logical_hit_source is required for a logical hit"
            )
        if not cost_currency:
            raise LedgerError("cost_currency must be non-empty")
        if physical_hit_source is None:
            if physical_hit is not None:
                raise LedgerError(
                    "physical_hit_source is required for an observed physical hit"
                )
            physical_hit_source = "unobservable"
        if not physical_hit_source:
            raise LedgerError("physical_hit_source must be non-empty")
        estimated_gpu_seconds, estimated_cost = estimate_cost(
            prefill_ms=trace.prefill_ms,
            decode_ms=trace.decode_ms,
            gpu_hour_price=gpu_hour_price,
        )
        return cls(
            request_id=trace.request_id,
            tenant_id=trace.tenant_id,
            model=trace.model,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            queue_ms=trace.queue_ms,
            ttft_ms=trace.ttft_ms,
            tpot_ms=trace.tpot_ms,
            prefill_ms=trace.prefill_ms,
            decode_ms=trace.decode_ms,
            total_ms=trace.total_ms,
            timeline=trace.timeline,
            strategy_version=strategy_version,
            logical_kv_blocks=trace.logical_kv_blocks,
            reservation_peak_blocks=trace.logical_kv_blocks_peak,
            logical_hit=logical_hit,
            logical_hit_source=logical_hit_source,
            physical_hit=physical_hit,
            physical_hit_source=physical_hit_source,
            terminal_state=trace.terminal_state,
            error_code=trace.error_code,
            error_stage=trace.error_stage,
            gpu_hour_price=gpu_hour_price,
            estimated_gpu_seconds=estimated_gpu_seconds,
            estimated_cost=estimated_cost,
            cost_currency=cost_currency,
            cost_basis="prefill_plus_decode_wall_time",
            cost_is_estimate=True,
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timeline"] = list(self.timeline)
        return payload


def recalculate_estimated_cost(record: Mapping[str, Any]) -> Optional[float]:
    """仅使用账本原始阶段耗时和价格重算估算成本。"""

    if record.get("cost_is_estimate") is not True:
        raise LedgerError("ledger cost must be explicitly marked as estimated")
    _, estimated_cost = estimate_cost(
        prefill_ms=record.get("prefill_ms"),
        decode_ms=record.get("decode_ms"),
        gpu_hour_price=record.get("gpu_hour_price"),
    )
    return estimated_cost


def validate_recalculated_cost(
    record: Mapping[str, Any], *, relative_tolerance: float = 1e-9
) -> None:
    """验证账本中的成本可以从原始字段重算。"""

    expected = recalculate_estimated_cost(record)
    actual = record.get("estimated_cost")
    if expected is None or actual is None:
        if expected != actual:
            raise LedgerError("estimated_cost is not reproducible from ledger fields")
        return
    if not isclose(actual, expected, rel_tol=relative_tolerance, abs_tol=1e-12):
        raise LedgerError(
            "estimated_cost is not reproducible from ledger fields: "
            "expected {0}, got {1}".format(expected, actual)
        )


class RequestLedger:
    """线程安全的内存请求账本，可导出 JSONL 原始记录。"""

    def __init__(
        self,
        *,
        strategy_version: str = "gateway-v1",
        gpu_hour_price: Optional[float] = None,
        cost_currency: str = "USD",
    ) -> None:
        if not strategy_version:
            raise LedgerError("strategy_version must be non-empty")
        if not cost_currency:
            raise LedgerError("cost_currency must be non-empty")
        if gpu_hour_price is not None and gpu_hour_price < 0:
            raise LedgerError("gpu_hour_price must be non-negative")
        self.strategy_version = strategy_version
        self.gpu_hour_price = gpu_hour_price
        self.cost_currency = cost_currency
        self._lock = threading.Lock()
        self._records: dict[str, RequestLedgerRecord] = {}

    def record_trace(
        self,
        trace: RequestTraceSnapshot,
        *,
        logical_hit: bool = False,
        logical_hit_source: str = "not_configured",
        physical_hit: Optional[bool] = None,
        physical_hit_source: Optional[str] = None,
    ) -> RequestLedgerRecord:
        record = RequestLedgerRecord.from_trace(
            trace,
            strategy_version=self.strategy_version,
            gpu_hour_price=self.gpu_hour_price,
            cost_currency=self.cost_currency,
            logical_hit=logical_hit,
            logical_hit_source=logical_hit_source,
            physical_hit=physical_hit,
            physical_hit_source=physical_hit_source,
        )
        with self._lock:
            return self._records.setdefault(record.request_id, record)

    def snapshot(self, request_id: str) -> Optional[RequestLedgerRecord]:
        with self._lock:
            return self._records.get(request_id)

    def records(self) -> tuple[RequestLedgerRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def export_jsonl(self) -> str:
        with self._lock:
            return "".join(
                json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in self._records.values()
            )

    def render_prometheus(self) -> str:
        """按 tenant/model 聚合可重算的估算 GPU 时间与成本。"""

        with self._lock:
            gpu_seconds: dict[tuple[str, str], float] = {}
            costs: dict[tuple[str, str, str], float] = {}
            for record in self._records.values():
                key = (record.tenant_id, record.model)
                if record.estimated_gpu_seconds is not None:
                    gpu_seconds[key] = (
                        gpu_seconds.get(key, 0.0) + record.estimated_gpu_seconds
                    )
                if record.estimated_cost is not None:
                    cost_key = (
                        record.cost_currency,
                        record.tenant_id,
                        record.model,
                    )
                    costs[cost_key] = costs.get(cost_key, 0.0) + record.estimated_cost

        lines = ["# TYPE cachepilot_estimated_gpu_seconds_total counter\n"]
        for (tenant, model), value in sorted(gpu_seconds.items()):
            lines.append(
                _ledger_metric_line(
                    "cachepilot_estimated_gpu_seconds_total",
                    value,
                    {"model": model, "tenant": tenant},
                )
            )
        lines.append("# TYPE cachepilot_estimated_cost_total counter\n")
        for (currency, tenant, model), value in sorted(costs.items()):
            lines.append(
                _ledger_metric_line(
                    "cachepilot_estimated_cost_total",
                    value,
                    {"currency": currency, "model": model, "tenant": tenant},
                )
            )
        return "".join(lines)


def _ledger_metric_line(
    name: str,
    value: float,
    labels: Mapping[str, str],
) -> str:
    rendered_labels = ",".join(
        '{0}="{1}"'.format(key, _escape_prometheus_label(labels[key]))
        for key in sorted(labels)
    )
    return "{0}{{{1}}} {2}\n".format(name, rendered_labels, value)


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
