"""执行器能力矩阵与指标语义比较门禁。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional


class CapabilityError(ValueError):
    """执行器能力或指标比较请求无效。"""


class IncomparableMetricError(CapabilityError):
    """待比较指标具有不同或不可观测的语义。"""


@dataclass(frozen=True)
class ExecutorCapabilities:
    """一个执行器对 batch、KV、prefix 和取消的稳定能力声明。"""

    executor: str
    timing: str
    batch: str
    batch_owner: str
    physical_kv: str
    physical_kv_observable: bool
    prefix: str
    physical_prefix_hit_observable: bool
    cancellation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SIM_EXECUTOR_CAPABILITIES = ExecutorCapabilities(
    executor="SimExecutor",
    timing="simulated_logical_clock",
    batch="simulated_continuous",
    batch_owner="cachepilot_simulator",
    physical_kv="none_logical_blocks_only",
    physical_kv_observable=False,
    prefix="cachepilot_logical_only",
    physical_prefix_hit_observable=False,
    cancellation="simulated_synchronous_terminal_transition",
)

TORCH_EXECUTOR_CAPABILITIES = ExecutorCapabilities(
    executor="TorchExecutor",
    timing="measured_monotonic_clock",
    batch="single_request",
    batch_owner="none",
    physical_kv="transformers_internal_unobservable",
    physical_kv_observable=False,
    prefix="cachepilot_logical_only_no_physical_reuse",
    physical_prefix_hit_observable=False,
    cancellation="cooperative_stopping_criteria",
)

VLLM_EXECUTOR_CAPABILITIES = ExecutorCapabilities(
    executor="VllmExecutor",
    timing="measured_monotonic_clock",
    batch="vllm_continuous",
    batch_owner="vllm",
    physical_kv="vllm_managed_unobservable",
    physical_kv_observable=False,
    prefix="cachepilot_logical_only_physical_unverified",
    physical_prefix_hit_observable=False,
    cancellation="vllm_abort_by_request_id",
)

EXECUTOR_CAPABILITY_MATRIX: Mapping[str, ExecutorCapabilities] = MappingProxyType(
    {
        capability.executor: capability
        for capability in (
            SIM_EXECUTOR_CAPABILITIES,
            TORCH_EXECUTOR_CAPABILITIES,
            VLLM_EXECUTOR_CAPABILITIES,
        )
    }
)

METRIC_FAMILY_BY_NAME: Mapping[str, str] = MappingProxyType(
    {
        "queue_ms": "latency",
        "ttft_ms": "latency",
        "tpot_ms": "latency",
        "total_ms": "latency",
        "throughput_completion_tokens_per_s": "throughput",
        "batch_utilization": "batch",
        "reserved_blocks_peak": "logical_kv",
        "logical_kv": "logical_kv",
        "physical_kv": "physical_kv",
        "logical_hit": "logical_prefix",
        "logical_prefix": "logical_prefix",
        "physical_hit": "physical_prefix",
        "physical_prefix": "physical_prefix",
        "cancellation_rate": "cancellation",
        "cancellation": "cancellation",
    }
)


def capabilities_for_executor(executor: str) -> ExecutorCapabilities:
    """返回已知执行器能力；未知名称不能进入实验比较。"""

    try:
        return EXECUTOR_CAPABILITY_MATRIX[executor]
    except KeyError as exc:
        raise CapabilityError(
            "unknown executor capability: {0}".format(executor)
        ) from exc


def metric_semantic_signature(
    executor: str,
    metric: str,
) -> Optional[str]:
    """返回指标语义签名；不可观测指标返回 ``None``。"""

    capability = capabilities_for_executor(executor)
    family = METRIC_FAMILY_BY_NAME.get(metric, metric)
    if family == "latency":
        return capability.timing
    if family == "throughput":
        return "{0}|{1}".format(capability.timing, capability.batch)
    if family == "batch":
        return capability.batch
    if family == "logical_kv":
        return "cachepilot_logical_reservation_blocks"
    if family == "physical_kv":
        return capability.physical_kv if capability.physical_kv_observable else None
    if family == "logical_prefix":
        return "cachepilot_tenant_scoped_logical_prefix"
    if family == "physical_prefix":
        return (
            capability.prefix
            if capability.physical_prefix_hit_observable
            else None
        )
    if family == "cancellation":
        return capability.cancellation
    raise CapabilityError("unknown metric family: {0}".format(metric))


def require_comparable_metric(metric: str, executors: Iterable[str]) -> str:
    """要求多个执行器对该指标具有同一可观测语义。"""

    executor_names = tuple(executors)
    if len(executor_names) < 2:
        raise CapabilityError("metric comparison requires at least two executors")
    signatures = {
        executor: metric_semantic_signature(executor, metric)
        for executor in executor_names
    }
    unavailable = [
        executor for executor, signature in signatures.items() if signature is None
    ]
    if unavailable:
        raise IncomparableMetricError(
            "metric {0} is unobservable for: {1}".format(
                metric,
                ", ".join(unavailable),
            )
        )
    distinct = set(signatures.values())
    if len(distinct) != 1:
        details = ", ".join(
            "{0}={1}".format(executor, signature)
            for executor, signature in signatures.items()
        )
        raise IncomparableMetricError(
            "metric {0} has incompatible semantics: {1}".format(metric, details)
        )
    return next(iter(distinct))
