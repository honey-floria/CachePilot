"""使用单调时钟管理排队和执行 deadline。

持续时间不能由会被 NTP 或人工修改的墙上时钟判断，因此本模块只接受
``monotonic_ns``。Deadline Manager 不创建后台线程；Runtime loop 应在每轮
开始调用 ``expire_due``，使模拟器和真实服务共享确定性超时语义。
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple


class DeadlineError(ValueError):
    """Deadline 配置或生命周期操作无效时抛出。"""


class DeadlinePhase(str, Enum):
    """当前使用排队还是执行阶段 deadline。"""

    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"


class TimeoutReason(str, Enum):
    """区分阶段预算耗尽与客户端总请求预算耗尽。"""

    QUEUE_DEADLINE = "queue_deadline"
    EXECUTION_DEADLINE = "execution_deadline"
    REQUEST_DEADLINE = "request_deadline"


@dataclass(frozen=True)
class DeadlinePolicy:
    """服务端为排队和执行阶段设置的最长持续时间，单位毫秒。"""

    queue_timeout_ms: int  # 请求在队列阶段允许停留的最大毫秒数。
    execution_timeout_ms: int  # 请求在执行阶段允许运行的最大毫秒数。

    def __post_init__(self) -> None:
        """两个阶段 timeout 都必须是正整数毫秒。"""

        _require_positive_int(self.queue_timeout_ms, "queue_timeout_ms")
        _require_positive_int(self.execution_timeout_ms, "execution_timeout_ms")


@dataclass(frozen=True)
class TimeoutEvent:
    """一次确定的超时结果。

    Attributes:
        request_id: 到期请求 ID。
        phase: 到期时所处阶段。
        reason: 最先生效的 deadline 类型。
        expired_at_ns: 扫描时的单调时钟值。
        resources_released: 资源回调本次是否实际移除了请求资源。
    """

    request_id: str  # 到期请求 ID。
    phase: DeadlinePhase  # 到期时请求所处阶段。
    reason: TimeoutReason  # 最先生效的 deadline 类型。
    expired_at_ns: int  # 扫描到超时时的单调时钟纳秒值。
    resources_released: bool  # 本次是否实际释放了请求资源。


@dataclass(frozen=True)
class DeadlineSnapshot:
    """当前被跟踪请求数量及最近有效 deadline。"""

    queued_requests: int  # 当前被跟踪的排队请求数。
    executing_requests: int  # 当前被跟踪的执行中请求数。
    next_deadline_ns: Optional[int]  # 最近有效 deadline 的单调纳秒值。


@dataclass
class _TrackedDeadline:
    """内部可变 deadline 记录；阶段切换时会重置 phase deadline。"""

    request_id: str  # 被跟踪的请求 ID。
    phase: DeadlinePhase  # 请求当前所处阶段。
    request_deadline_ns: int  # 端到端请求 deadline 的单调纳秒值。
    phase_deadline_ns: int  # 当前阶段 deadline 的单调纳秒值。


class RequestDeadlineManager:
    """在 runtime tick 中确定性过期请求并触发资源回收。

    Args:
        policy: 排队和执行阶段各自的超时配置。
        release_resources: 到期时调用的回收函数，参数为 request ID；
            返回值表示本次是否实际释放了活跃或排队资源。
        monotonic_ns: 可注入的单调纳秒时钟；生产默认
            ``time.monotonic_ns``，测试和模拟器可传入逻辑时钟。
    """

    def __init__(
        self,
        policy: DeadlinePolicy,  # 排队和执行阶段的超时策略。
        release_resources: Callable[[str], bool],  # 到期时调用的资源回收函数。
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,  # 单调逻辑时钟。
    ) -> None:
        if not isinstance(policy, DeadlinePolicy):
            raise TypeError("policy must be a DeadlinePolicy")
        if not callable(release_resources):
            raise TypeError("release_resources must be callable")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        self._policy = policy
        self._release_resources = release_resources
        self._monotonic_ns = monotonic_ns
        self._tracked: Dict[str, _TrackedDeadline] = {}
        self._completed = set()
        self._lock = threading.Lock()

    def track_queued(
        self,
        request_id: str,  # 首次进入 deadline 管理的唯一请求 ID。
        request_deadline_ms: int,  # 客户端端到端请求预算毫秒数。
    ) -> None:
        """从当前单调时刻开始跟踪排队与请求总 deadline。

        Args:
            request_id: 首次进入 deadline 管理的唯一请求 ID。
            request_deadline_ms: 客户端声明或契约默认的端到端预算。

        每个请求同时保存总 deadline 和当前阶段 deadline；实际到期时间
        取两者中更早者，因此服务端阶段预算不会延长客户端总预算。
        """

        _require_identifier(request_id)
        _require_positive_int(request_deadline_ms, "request_deadline_ms")
        now_ns = self._monotonic_ns()
        with self._lock:
            if request_id in self._tracked or request_id in self._completed:
                raise DeadlineError(
                    "request_id is already tracked: {0}".format(request_id)
                )
            self._tracked[request_id] = _TrackedDeadline(
                request_id=request_id,
                phase=DeadlinePhase.QUEUED,
                request_deadline_ns=now_ns + _milliseconds_to_ns(request_deadline_ms),
                phase_deadline_ns=now_ns
                + _milliseconds_to_ns(self._policy.queue_timeout_ms),
            )

    def mark_executing(
        self,
        request_id: str,  # 要从排队切换到执行阶段的请求 ID。
    ) -> Optional[TimeoutEvent]:
        """从排队切换到执行阶段，并开始计算执行 deadline。

        Args:
            request_id: 已被 ``track_queued`` 跟踪的请求 ID。

        Returns:
            正常切换返回 ``None``；如果调用时 deadline 已到，则不会复活
            请求，而是立即回收并返回 ``TimeoutEvent``。
        """

        _require_identifier(request_id)
        now_ns = self._monotonic_ns()
        with self._lock:
            tracked = self._tracked.get(request_id)
            if tracked is None:
                raise DeadlineError("request is not tracked: {0}".format(request_id))
            due_reason = self._due_reason(tracked, now_ns)
            if due_reason is not None:
                return self._expire_locked(tracked, due_reason, now_ns)
            if tracked.phase is DeadlinePhase.EXECUTING:
                return None
            tracked.phase = DeadlinePhase.EXECUTING
            tracked.phase_deadline_ns = now_ns + _milliseconds_to_ns(
                self._policy.execution_timeout_ms
            )
            return None

    def complete(
        self,
        request_id: str,  # 要停止 deadline 跟踪的请求 ID。
    ) -> bool:
        """停止跟踪正常结束、取消或外部已终止的请求。

        Returns:
            找到并移除跟踪记录时为 ``True``，重复调用为 ``False``。
        """

        _require_identifier(request_id)
        with self._lock:
            tracked = self._tracked.pop(request_id, None)
            if tracked is None:
                return False
            self._completed.add(request_id)
            return True

    def expire_due(self) -> Tuple[TimeoutEvent, ...]:
        """扫描并过期当前时刻所有到期请求。

        Returns:
            按 deadline、request ID 稳定排序的超时事件元组，使相同 trace
            和逻辑时钟得到可重复结果。

        Runtime loop 应在每轮完成/回收阶段调用本方法，确保超时请求在
        选择新 batch 前归还 reservation。
        """

        now_ns = self._monotonic_ns()
        with self._lock:
            due = []
            for tracked in self._tracked.values():
                reason = self._due_reason(tracked, now_ns)
                if reason is not None:
                    due.append((self._effective_deadline(tracked), tracked, reason))
            # deadline 相同时用 request ID 打破平局，保证回放确定性。
            due.sort(key=lambda item: (item[0], item[1].request_id))
            return tuple(
                self._expire_locked(tracked, reason, now_ns)
                for _, tracked, reason in due
            )

    def snapshot(self) -> DeadlineSnapshot:
        """返回排队/执行数量和下一次应唤醒扫描的单调时间。"""

        with self._lock:
            queued = sum(
                tracked.phase is DeadlinePhase.QUEUED
                for tracked in self._tracked.values()
            )
            executing = len(self._tracked) - queued
            next_deadline = min(
                (
                    self._effective_deadline(tracked)
                    for tracked in self._tracked.values()
                ),
                default=None,
            )
            return DeadlineSnapshot(queued, executing, next_deadline)

    def _due_reason(
        self,
        tracked: _TrackedDeadline,  # 要判断是否到期的内部记录。
        now_ns: int,  # 本轮扫描使用的单调时钟纳秒值。
    ) -> Optional[TimeoutReason]:
        """按总请求预算优先判断记录在 ``now_ns`` 是否到期。"""

        if now_ns >= tracked.request_deadline_ns:
            return TimeoutReason.REQUEST_DEADLINE
        if now_ns < tracked.phase_deadline_ns:
            return None
        if tracked.phase is DeadlinePhase.QUEUED:
            return TimeoutReason.QUEUE_DEADLINE
        return TimeoutReason.EXECUTION_DEADLINE

    @staticmethod
    def _effective_deadline(
        tracked: _TrackedDeadline,  # 要计算最早有效 deadline 的记录。
    ) -> int:
        """返回总 deadline 与当前阶段 deadline 中更早的一个。"""

        return min(tracked.request_deadline_ns, tracked.phase_deadline_ns)

    def _expire_locked(
        self,
        tracked: _TrackedDeadline,  # 已确认到期的内部记录。
        reason: TimeoutReason,  # 导致本次到期的 deadline 类型。
        now_ns: int,  # 执行到期处理时的单调时钟纳秒值。
    ) -> TimeoutEvent:
        """持锁时执行回收、移除记录并构造超时事件。"""

        resources_released = bool(self._release_resources(tracked.request_id))
        del self._tracked[tracked.request_id]
        self._completed.add(tracked.request_id)
        return TimeoutEvent(
            request_id=tracked.request_id,
            phase=tracked.phase,
            reason=reason,
            expired_at_ns=now_ns,
            resources_released=resources_released,
        )


def _milliseconds_to_ns(
    value: int,  # 要转换为纳秒的整数毫秒值。
) -> int:
    """使用整数运算把毫秒转换为纳秒，避免浮点误差。"""

    return value * 1_000_000


def _require_identifier(
    request_id: str,  # 要校验的请求 ID。
) -> None:
    """验证 request ID 为非空字符串。"""

    if type(request_id) is not str or not request_id:
        raise DeadlineError("request_id must be a non-empty string")


def _require_positive_int(
    value: int,  # 要校验的整数值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    """验证 timeout 等配置为正整数，并拒绝 bool。"""

    if type(value) is not int or value < 1:
        raise DeadlineError(
            "{0} must be a positive integer".format(field_name)
        )
