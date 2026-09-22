"""Strict KV reservation 准入策略。

Strict 模式始终按 ``prompt_tokens + max_new_tokens`` 预留资源。它牺牲部分
利用率来换取可预测性，并在一把锁内同时检查全局容量、tenant 配额和
队列长度，保证并发提交不会突破硬上限。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

from cachepilot.runtime.kv_planner import (
    ContextLimitExceededError,
    KVPlanner,
    KVRequestPlan,
)
from cachepilot.runtime.resources import ResourceLeaseManager
from cachepilot.utils import CommonUtils


class AdmissionError(ValueError):
    """准入配置或请求操作无效时抛出。"""


class AdmissionStatus(str, Enum):
    """一次准入尝试的三种稳定结果。"""

    ADMITTED = "ADMITTED"
    QUEUED = "QUEUED"
    REJECTED = "REJECTED"


class AdmissionReason(str, Enum):
    """便于指标、日志和 API 映射的机器可读决策原因。"""

    ADMITTED = "admitted"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    REQUEST_EXCEEDS_KV_CAPACITY = "request_exceeds_kv_capacity"
    TENANT_NOT_CONFIGURED = "tenant_not_configured"
    TENANT_REQUEST_EXCEEDS_TOKEN_QUOTA = "tenant_request_exceeds_token_quota"
    MAX_ACTIVE_SEQUENCES = "max_active_sequences"
    KV_CAPACITY = "kv_capacity"
    TENANT_ACTIVE_TOKENS = "tenant_active_tokens"
    TENANT_CONCURRENCY = "tenant_concurrency"
    QUEUE_FULL = "queue_full"
    TENANT_QUEUE_FULL = "tenant_queue_full"


@dataclass(frozen=True)
class TenantAdmissionLimits:
    """单个 tenant 的资源与排队硬限制。

    Attributes:
        max_active_sequences: 同时处于活跃状态的最大请求数。
        max_active_tokens: 活跃请求 reservation token 的合计上限。
        max_queued_requests: 该 tenant 可等待的最大请求数；0 表示
            不允许排队。
    """

    max_active_sequences: int  # tenant 同时活跃的最大请求数。
    max_active_tokens: int  # tenant 活跃 reservation 的最大 token 总数。
    max_queued_requests: int  # tenant 允许排队的最大请求数。

    def __post_init__(self) -> None:
        """拒绝负数、零并发和零 token 等无意义配置。"""

        CommonUtils.require_positive_int(
            self.max_active_sequences, "max_active_sequences", AdmissionError
        )
        CommonUtils.require_positive_int(
            self.max_active_tokens, "max_active_tokens", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            self.max_queued_requests, "max_queued_requests", AdmissionError
        )


@dataclass(frozen=True)
class StrictAdmissionConfig:
    """Strict Admission 的全局容量配置。

    Attributes:
        total_blocks: 控制层掌握的逻辑 KV block 总量。
        safety_blocks: 永不接纳请求使用的安全余量。
        max_active_sequences: 跨 tenant 的最大活跃请求数。
        max_queued_requests: 跨 tenant 的最大排队请求数。
        tenant_limits: tenant ID 到独立限制的映射；未配置 tenant
            会被拒绝。
        retry_after_ms: 队列满拒绝时提供给客户端的建议重试间隔。
    """

    total_blocks: int  # 控制层可管理的逻辑 KV block 总数。
    safety_blocks: int  # 永不分配给请求的安全余量 block 数。
    max_active_sequences: int  # 全局同时活跃的最大请求数。
    max_queued_requests: int  # 全局允许排队的最大请求数。
    tenant_limits: Mapping[str, TenantAdmissionLimits]  # 各 tenant 的硬限制。
    retry_after_ms: int = 1000  # 队列满拒绝时建议客户端等待的毫秒数。

    def __post_init__(self) -> None:
        """校验全局限制，并复制 tenant 映射避免引用调用方容器。"""

        CommonUtils.require_positive_int(
            self.total_blocks, "total_blocks", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            self.safety_blocks, "safety_blocks", AdmissionError
        )
        CommonUtils.require_positive_int(
            self.max_active_sequences, "max_active_sequences", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            self.max_queued_requests, "max_queued_requests", AdmissionError
        )
        CommonUtils.require_positive_int(
            self.retry_after_ms, "retry_after_ms", AdmissionError
        )
        if self.safety_blocks >= self.total_blocks:
            raise AdmissionError("safety_blocks must be less than total_blocks")
        if not isinstance(self.tenant_limits, Mapping):
            raise AdmissionError("tenant_limits must be a mapping")

        copied_limits = {}
        for tenant_id, limits in self.tenant_limits.items():
            CommonUtils.require_identifier(
                tenant_id, "tenant_id", AdmissionError
            )
            if not isinstance(limits, TenantAdmissionLimits):
                raise AdmissionError(
                    "tenant limit for {0} must be TenantAdmissionLimits".format(
                        tenant_id
                    )
                )
            copied_limits[tenant_id] = limits
        object.__setattr__(self, "tenant_limits", copied_limits)

    @property
    def usable_blocks(self) -> int:
        """返回请求实际可预留的 block 数，即总量减安全余量。"""

        return self.total_blocks - self.safety_blocks


@dataclass(frozen=True)
class AdmissionDecision:
    """一次提交或重试的完整、可观测决策。

    ``plan`` 在请求尚未形成有效 KV 计划时为 ``None``；``retry_after_ms``
    仅用于队列已满等客户端稍后重试可能成功的拒绝。
    """

    request_id: str  # 本次决策对应的请求 ID。
    tenant_id: str  # 请求所属 tenant。
    status: AdmissionStatus  # 接纳、排队或拒绝状态。
    reason: AdmissionReason  # 产生该决策的机器可读原因。
    plan: Optional[KVRequestPlan]  # 请求的逻辑 KV 规划；无有效规划时为空。
    estimated_output_tokens: Optional[int] = None  # 本次预留采用的输出估计。
    fallback_to_strict: bool = False  # Adaptive 是否因样本不足回退 Strict。
    retry_after_ms: Optional[int] = None  # 拒绝后建议重试的等待毫秒数。


@dataclass(frozen=True)
class AdmissionSnapshot:
    """同一临界区内读取的全局与 tenant 准入账本快照。"""

    active_sequences: int  # 当前全局活跃请求数。
    queued_requests: int  # 当前全局排队请求数。
    reserved_blocks: int  # 当前已预留的逻辑 KV block 数。
    usable_blocks: int  # 扣除安全余量后的可用 block 总数。
    tenant_active_sequences: Tuple[Tuple[str, int], ...]  # tenant 活跃数。
    tenant_active_tokens: Tuple[Tuple[str, int], ...]  # tenant 活跃 token 数。
    tenant_queued_requests: Tuple[Tuple[str, int], ...]  # tenant 排队数。


@dataclass
class _AdmissionRequest:
    """Controller 内部请求记录；Adaptive 子类会更新当前 reservation。"""

    request_id: str  # 请求唯一标识。
    tenant_id: str  # 请求所属 tenant。
    prompt_tokens: int  # 输入 token 数。
    max_new_tokens: int  # 客户端声明的最大输出 token 数。
    plan: KVRequestPlan  # 当前 reservation 对应的 KV 规划。
    initial_estimated_output_tokens: int  # 首次准入使用的输出估计。
    estimated_output_tokens: int  # 当前 reservation 使用的输出估计。
    fallback_to_strict: bool  # 是否因样本不足使用 Strict 估计。


class StrictAdmissionController:
    """按请求声明的最大输出长度进行保守、原子准入。

    Args:
        planner: 把 token 需求转换为逻辑 block 的 KV Planner。
        config: 全局容量、队列和 tenant 限额配置。

    Note:
        Controller 只负责容量准入，不决定排队请求的公平顺序；后续
        Scheduler 选择某个 request ID 后再调用 ``retry_queued``。
    """

    def __init__(
        self,
        planner: KVPlanner,  # 把 token 需求换算为逻辑 KV block 的规划器。
        config: StrictAdmissionConfig,  # 全局容量、队列和 tenant 限制。
    ) -> None:
        if not isinstance(planner, KVPlanner):
            raise TypeError("planner must be a KVPlanner")
        if not isinstance(config, StrictAdmissionConfig):
            raise TypeError("config must be a StrictAdmissionConfig")
        self._planner = planner
        self._config = config
        self._resources = ResourceLeaseManager(config.usable_blocks)
        self._active: Dict[str, _AdmissionRequest] = {}
        self._queued: Dict[str, _AdmissionRequest] = {}
        self._completed = set()
        self._tenant_active_sequences: Dict[str, int] = {}
        self._tenant_active_tokens: Dict[str, int] = {}
        self._tenant_queued_requests: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> StrictAdmissionConfig:
        """返回 Controller 使用的准入配置。"""

        return self._config

    def submit(
        self,
        request_id: str,  # 尚未进入准入账本的唯一请求 ID。
        tenant_id: str,  # 资源配额所属 tenant ID。
        prompt_tokens: int,  # tokenizer 计算的输入 token 数。
        max_new_tokens: int,  # 客户端声明的最大输出 token 数。
    ) -> AdmissionDecision:
        """提交新请求；暂时容量不足时进入有界队列。

        Args:
            request_id: 尚未进入本 Controller 的唯一请求 ID。
            tenant_id: 资源配额所属 tenant；必须存在于配置中。
            prompt_tokens: tokenizer 计算出的输入 KV token 数。
            max_new_tokens: 客户端声明的最大输出 token 数。Strict 模式会
                完整预留该值，而不是使用历史平均值。

        Returns:
            ADMITTED、QUEUED 或 REJECTED 决策及稳定原因。

        Logic:
            永久不可能满足的请求直接拒绝；仅因当前活跃负载受阻的请求，
            在全局与 tenant 队列都有空间时进入队列。
        """

        CommonUtils.require_identifier(
            request_id, "request_id", AdmissionError
        )
        CommonUtils.require_identifier(
            tenant_id, "tenant_id", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            prompt_tokens, "prompt_tokens", AdmissionError
        )
        CommonUtils.require_positive_int(
            max_new_tokens, "max_new_tokens", AdmissionError
        )

        with self._lock:
            self._ensure_new_request_id(request_id)
            limits = self._config.tenant_limits.get(tenant_id)
            if limits is None:
                return AdmissionDecision(
                    request_id,
                    tenant_id,
                    AdmissionStatus.REJECTED,
                    AdmissionReason.TENANT_NOT_CONFIGURED,
                    None,
                )

            try:
                request = self._build_request(
                    request_id,
                    tenant_id,
                    prompt_tokens,
                    max_new_tokens,
                )
            except ContextLimitExceededError:
                return AdmissionDecision(
                    request_id,
                    tenant_id,
                    AdmissionStatus.REJECTED,
                    AdmissionReason.CONTEXT_LIMIT_EXCEEDED,
                    None,
                )

            # 单请求自身超限，等待其他请求结束也不会变得可接纳。
            permanent_reason = self._permanent_rejection_reason(request, limits)
            if permanent_reason is not None:
                return self._decision(
                    request, AdmissionStatus.REJECTED, permanent_reason
                )

            # 活跃占用导致的阻塞是暂时的，可进入有界队列。
            blocked_reason = self._temporary_block_reason(request, limits)
            if blocked_reason is None:
                return self._admit(request)
            return self._queue_or_reject(request, limits, blocked_reason)

    def retry_queued(
        self,
        request_id: str,  # Scheduler 选中并要求重新评估的排队请求 ID。
    ) -> AdmissionDecision:
        """重新评估 Scheduler 选中的一个排队请求。

        Args:
            request_id: 当前必须存在于本 Controller 队列中的请求 ID。

        Returns:
            资源仍不足时继续 QUEUED；满足全部硬约束时原子转为
            ADMITTED。
        """

        CommonUtils.require_identifier(
            request_id, "request_id", AdmissionError
        )
        with self._lock:
            request = self._queued.get(request_id)
            if request is None:
                raise AdmissionError("request is not queued: {0}".format(request_id))
            limits = self._config.tenant_limits[request.tenant_id]
            blocked_reason = self._temporary_block_reason(request, limits)
            if blocked_reason is not None:
                return self._decision(
                    request, AdmissionStatus.QUEUED, blocked_reason
                )

            del self._queued[request_id]
            self._decrement(self._tenant_queued_requests, request.tenant_id)
            return self._admit(request)

    def release(
        self,
        request_id: str,  # 要从活跃或等待账本释放的请求 ID。
    ) -> bool:
        """释放活跃 reservation 或从等待队列移除请求。

        Args:
            request_id: 已接纳、排队、完成或未知的请求 ID。

        Returns:
            本次确实移除记录时返回 ``True``；重复释放或未知请求返回
            ``False``，不会重复扣减任何计数。
        """

        CommonUtils.require_identifier(
            request_id, "request_id", AdmissionError
        )
        with self._lock:
            return self._release_locked(request_id)

    def snapshot(self) -> AdmissionSnapshot:
        """原子读取容量、队列以及各 tenant 占用的不可变快照。"""

        with self._lock:
            return AdmissionSnapshot(
                active_sequences=len(self._active),
                queued_requests=len(self._queued),
                reserved_blocks=self._resources.total_logical_blocks,
                usable_blocks=self._config.usable_blocks,
                tenant_active_sequences=tuple(
                    sorted(self._tenant_active_sequences.items())
                ),
                tenant_active_tokens=tuple(
                    sorted(self._tenant_active_tokens.items())
                ),
                tenant_queued_requests=tuple(
                    sorted(self._tenant_queued_requests.items())
                ),
            )

    def _permanent_rejection_reason(
        self,
        request: _AdmissionRequest,  # 要检查永久拒绝条件的内部请求。
        limits: TenantAdmissionLimits,  # 该请求所属 tenant 的硬限制。
    ) -> Optional[AdmissionReason]:
        """返回请求自身造成的永久拒绝原因，否则返回 ``None``。"""

        if request.plan.logical_blocks > self._config.usable_blocks:
            return AdmissionReason.REQUEST_EXCEEDS_KV_CAPACITY
        if request.plan.total_tokens > limits.max_active_tokens:
            return AdmissionReason.TENANT_REQUEST_EXCEEDS_TOKEN_QUOTA
        return None

    def _build_request(
        self,
        request_id: str,  # 请求唯一标识。
        tenant_id: str,  # 请求所属 tenant。
        prompt_tokens: int,  # 输入 token 数。
        max_new_tokens: int,  # 最大输出 token 数。
    ) -> _AdmissionRequest:
        """创建 Strict 请求记录；Adaptive 子类覆盖此估算步骤。"""

        plan = self._planner.plan_request(prompt_tokens, max_new_tokens)
        return _AdmissionRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            plan=plan,
            initial_estimated_output_tokens=max_new_tokens,
            estimated_output_tokens=max_new_tokens,
            fallback_to_strict=False,
        )

    def _temporary_block_reason(
        self,
        request: _AdmissionRequest,  # 要检查临时阻塞条件的内部请求。
        limits: TenantAdmissionLimits,  # 该请求所属 tenant 的硬限制。
    ) -> Optional[AdmissionReason]:
        """按稳定优先级检查当前负载造成的临时阻塞原因。"""

        if len(self._active) >= self._config.max_active_sequences:
            return AdmissionReason.MAX_ACTIVE_SEQUENCES
        if (
            self._resources.total_logical_blocks + request.plan.logical_blocks
            > self._config.usable_blocks
        ):
            return AdmissionReason.KV_CAPACITY
        if (
            self._tenant_active_tokens.get(request.tenant_id, 0)
            + request.plan.total_tokens
            > limits.max_active_tokens
        ):
            return AdmissionReason.TENANT_ACTIVE_TOKENS
        if (
            self._tenant_active_sequences.get(request.tenant_id, 0)
            >= limits.max_active_sequences
        ):
            return AdmissionReason.TENANT_CONCURRENCY
        return None

    def _admit(
        self,
        request: _AdmissionRequest,  # 已通过全部硬限制检查的内部请求。
    ) -> AdmissionDecision:
        """持锁且检查通过时，原子登记 reservation 与 tenant 计数。"""

        self._resources.reserve(request.request_id, request.plan.logical_blocks)
        self._active[request.request_id] = request
        self._increment(self._tenant_active_sequences, request.tenant_id)
        self._tenant_active_tokens[request.tenant_id] = (
            self._tenant_active_tokens.get(request.tenant_id, 0)
            + request.plan.total_tokens
        )
        return self._decision(
            request, AdmissionStatus.ADMITTED, AdmissionReason.ADMITTED
        )

    def _queue_or_reject(
        self,
        request: _AdmissionRequest,  # 暂时无法接纳的内部请求。
        limits: TenantAdmissionLimits,  # 该请求所属 tenant 的队列限制。
        blocked_reason: AdmissionReason,  # 导致请求暂时阻塞的原因。
    ) -> AdmissionDecision:
        """有空间则登记等待，否则返回带重试建议的拒绝。"""

        if len(self._queued) >= self._config.max_queued_requests:
            return self._decision(
                request,
                AdmissionStatus.REJECTED,
                AdmissionReason.QUEUE_FULL,
                retry_after_ms=self._config.retry_after_ms,
            )
        if (
            self._tenant_queued_requests.get(request.tenant_id, 0)
            >= limits.max_queued_requests
        ):
            return self._decision(
                request,
                AdmissionStatus.REJECTED,
                AdmissionReason.TENANT_QUEUE_FULL,
                retry_after_ms=self._config.retry_after_ms,
            )

        self._queued[request.request_id] = request
        self._increment(self._tenant_queued_requests, request.tenant_id)
        return self._decision(request, AdmissionStatus.QUEUED, blocked_reason)

    def _ensure_new_request_id(
        self,
        request_id: str,  # 要检查是否从未进入准入账本的请求 ID。
    ) -> None:
        """阻止活跃、排队或已完成 request ID 再次进入准入账本。"""

        if (
            request_id in self._active
            or request_id in self._queued
            or request_id in self._completed
        ):
            raise AdmissionError(
                "request_id has already entered admission: {0}".format(request_id)
            )

    def _release_locked(
        self,
        request_id: str,  # 持锁状态下要清理的请求 ID。
    ) -> bool:
        """持锁时清理活跃或排队记录，并保持所有计数同步。"""

        request = self._active.pop(request_id, None)
        if request is not None:
            self._resources.release(request_id)
            self._decrement(self._tenant_active_sequences, request.tenant_id)
            self._tenant_active_tokens[request.tenant_id] -= (
                request.plan.total_tokens
            )
            if self._tenant_active_tokens[request.tenant_id] == 0:
                del self._tenant_active_tokens[request.tenant_id]
            self._completed.add(request_id)
            return True

        request = self._queued.pop(request_id, None)
        if request is not None:
            self._decrement(self._tenant_queued_requests, request.tenant_id)
            self._completed.add(request_id)
            return True
        return False

    @staticmethod
    def _decision(
        request: _AdmissionRequest,  # 要转换成不可变决策的内部请求。
        status: AdmissionStatus,  # 决策状态。
        reason: AdmissionReason,  # 决策原因。
        retry_after_ms: Optional[int] = None,  # 可选的建议重试间隔。
    ) -> AdmissionDecision:
        """把内部请求记录转换成不会暴露可变状态的决策对象。"""

        return AdmissionDecision(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            status=status,
            reason=reason,
            plan=request.plan,
            estimated_output_tokens=request.estimated_output_tokens,
            fallback_to_strict=request.fallback_to_strict,
            retry_after_ms=retry_after_ms,
        )

    @staticmethod
    def _increment(
        values: Dict[str, int],  # 要修改的稀疏 tenant 计数器。
        key: str,  # 要增加计数的 tenant ID。
    ) -> None:
        """增加稀疏 tenant 计数器。"""

        values[key] = values.get(key, 0) + 1

    @staticmethod
    def _decrement(
        values: Dict[str, int],  # 要修改的稀疏 tenant 计数器。
        key: str,  # 要减少计数的 tenant ID。
    ) -> None:
        """减少稀疏 tenant 计数器，并删除归零项。"""

        values[key] -= 1
        if values[key] == 0:
            del values[key]
