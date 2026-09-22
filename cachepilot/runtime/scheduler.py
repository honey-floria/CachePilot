"""确定性的 FCFS 与加权公平队列调度器。

调度器只决定已排队请求的选择顺序，不执行容量准入。两个策略都维护
``interactive``/``batch`` 两个调度类别，并在每个类别内为 tenant 保存
独立 FIFO 子队列。调用方应在选择后再让 Admission Controller 重试准入。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Callable, Deque, Dict, Mapping, Optional, Tuple


class SchedulerError(ValueError):
    """调度配置或队列操作无效。"""


class SchedulingPriority(str, Enum):
    """首版支持的两个请求调度类别。"""

    INTERACTIVE = "interactive"
    BATCH = "batch"


_PRIORITY_ORDER = (
    SchedulingPriority.INTERACTIVE,
    SchedulingPriority.BATCH,
)


@dataclass(frozen=True)
class SchedulingRequest:
    """进入调度队列所需的最小请求描述。

    ``service_cost`` 是策略可比较的正整数工作量，首版建议使用预计需要
    执行的 token 数。FCFS 不使用该字段；WFQ 用它计算虚拟完成标签。
    """

    request_id: str  # 请求的全局唯一标识，用于去重和返回调度结果。
    tenant_id: str  # 请求所属租户，决定进入哪个 tenant FIFO 子队列。
    priority: SchedulingPriority  # 调度类别，只能是 interactive 或 batch。
    service_cost: int = 1  # 预计工作量；WFQ 用它计算虚拟完成标签。

    def __post_init__(self) -> None:
        _require_identifier(self.request_id, "request_id")
        _require_identifier(self.tenant_id, "tenant_id")
        if isinstance(self.priority, str):
            try:
                normalized_priority = SchedulingPriority(self.priority)
            except ValueError as error:
                raise SchedulerError(
                    "priority must be interactive or batch"
                ) from error
            object.__setattr__(self, "priority", normalized_priority)
        elif not isinstance(self.priority, SchedulingPriority):
            raise SchedulerError("priority must be interactive or batch")
        _require_positive_int(self.service_cost, "service_cost")


@dataclass(frozen=True)
class ScheduleDecision:
    """一次成功出队的稳定结果。"""

    request: SchedulingRequest              # 本次被选中并移出队列的请求。
    enqueued_at_ns: int                     # 请求进入调度队列时的单调时钟纳秒值。
    selected_at_ns: int                     # 调度器选中请求时的单调时钟纳秒值。
    queue_wait_ns: int                      # 请求从入队到被选中的实际等待纳秒数。
    starvation_promoted: bool               # 是否因达到最大饥饿时间而被强制提升。
    virtual_start: Optional[Fraction]       # WFQ 虚拟开始标签；FCFS 中为空。
    virtual_finish: Optional[Fraction]      # WFQ 虚拟完成标签；FCFS 中为空。


@dataclass(frozen=True)
class SchedulerSnapshot:
    """同一临界区内读取的队列和 WFQ 虚拟时间。"""

    queued_requests: int                                # 当前所有 priority 和 tenant 的排队请求总数。
    priority_counts: Tuple[Tuple[str, int], ...]        # 各优先级的请求数量。
    tenant_counts: Tuple[Tuple[str, str, int], ...]     # 各租户子队列数量。
    virtual_times: Tuple[Tuple[str, Fraction], ...]     # 各优先级 WFQ 虚拟时间。


@dataclass
class _QueuedRequest:
    request: SchedulingRequest                  # 对外请求描述。
    enqueued_at_ns: int                         # 入队时的单调时钟纳秒值。
    sequence: int                               # 严格递增的入队序号，用于稳定打破平局。
    virtual_start: Optional[Fraction] = None    # WFQ 虚拟开始标签。
    virtual_finish: Optional[Fraction] = None   # WFQ 虚拟完成标签。


class _TenantQueueScheduler:
    """FCFS/WFQ 共用的 tenant 子队列和并发保护。"""

    def __init__(
        self,
        monotonic_ns: Callable[[], int],  # 返回非负纳秒值的单调逻辑时钟。
    ) -> None:
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        self._monotonic_ns = monotonic_ns
        self._queues: Dict[
            SchedulingPriority, Dict[str, Deque[_QueuedRequest]]
        ] = {priority: {} for priority in _PRIORITY_ORDER}
        self._request_ids = set()
        self._next_sequence = 0
        self._last_now_ns: Optional[int] = None
        self._lock = threading.Lock()

    def enqueue(
        self,
        request: SchedulingRequest,  # 要追加到 priority/tenant 子队列的请求。
    ) -> None:
        """把请求追加到对应 priority/tenant FIFO 子队列。"""

        if not isinstance(request, SchedulingRequest):
            raise TypeError("request must be a SchedulingRequest")
        now_ns = self._read_clock()
        with self._lock:
            if request.request_id in self._request_ids:
                raise SchedulerError(
                    "request is already queued: {0}".format(request.request_id)
                )
            entry = _QueuedRequest(
                request=request,
                enqueued_at_ns=now_ns,
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            self._prepare_entry(entry)
            tenant_queues = self._queues[request.priority]
            tenant_queues.setdefault(request.tenant_id, deque()).append(entry)
            self._request_ids.add(request.request_id)

    def select(self) -> Optional[ScheduleDecision]:
        """选择并移除下一个请求；队列为空时返回 ``None``。"""

        now_ns = self._read_clock()
        with self._lock:
            entry, starvation_promoted = self._select_entry(now_ns)
            if entry is None:
                return None
            self._remove_head(entry)
            self._after_select(entry)
            return ScheduleDecision(
                request=entry.request,
                enqueued_at_ns=entry.enqueued_at_ns,
                selected_at_ns=now_ns,
                queue_wait_ns=now_ns - entry.enqueued_at_ns,
                starvation_promoted=starvation_promoted,
                virtual_start=entry.virtual_start,
                virtual_finish=entry.virtual_finish,
            )

    def snapshot(self) -> SchedulerSnapshot:
        """返回按 priority、tenant 稳定排序的队列计数。"""

        with self._lock:
            priority_counts = []
            tenant_counts = []
            total = 0
            for priority in _PRIORITY_ORDER:
                priority_total = 0
                for tenant_id in sorted(self._queues[priority]):
                    count = len(self._queues[priority][tenant_id])
                    priority_total += count
                    tenant_counts.append((priority.value, tenant_id, count))
                priority_counts.append((priority.value, priority_total))
                total += priority_total
            return SchedulerSnapshot(
                queued_requests=total,
                priority_counts=tuple(priority_counts),
                tenant_counts=tuple(tenant_counts),
                virtual_times=self._virtual_time_snapshot(),
            )

    def _read_clock(self) -> int:
        now_ns = self._monotonic_ns()
        if type(now_ns) is not int or now_ns < 0:
            raise SchedulerError("monotonic_ns must return a non-negative integer")
        with self._lock:
            if self._last_now_ns is not None and now_ns < self._last_now_ns:
                raise SchedulerError("monotonic clock moved backwards")
            self._last_now_ns = now_ns
        return now_ns

    def _heads(
        self,
        priority: SchedulingPriority,  # 要读取队首请求的调度类别。
    ) -> Tuple[_QueuedRequest, ...]:
        return tuple(
            tenant_queue[0]
            for tenant_queue in self._queues[priority].values()
            if tenant_queue
        )

    def _remove_head(
        self,
        entry: _QueuedRequest,  # 已被策略选中的 tenant 队首记录。
    ) -> None:
        tenant_queues = self._queues[entry.request.priority]
        tenant_queue = tenant_queues[entry.request.tenant_id]
        removed = tenant_queue.popleft()
        if removed is not entry:
            raise RuntimeError("scheduler selected a non-head tenant request")
        if not tenant_queue:
            del tenant_queues[entry.request.tenant_id]
        self._request_ids.remove(entry.request.request_id)

    def _prepare_entry(
        self,
        entry: _QueuedRequest,  # 即将入队并由具体策略补充标签的记录。
    ) -> None:
        del entry

    def _after_select(
        self,
        entry: _QueuedRequest,  # 已成功出队并用于更新策略状态的记录。
    ) -> None:
        del entry

    def _virtual_time_snapshot(self) -> Tuple[Tuple[str, Fraction], ...]:
        return ()

    def _select_entry(
        self,
        now_ns: int,  # 本轮选择使用的单调时钟纳秒值。
    ) -> Tuple[Optional[_QueuedRequest], bool]:
        raise NotImplementedError


class FCFSScheduler(_TenantQueueScheduler):
    """严格优先 interactive、类别内全局先到先服务的基线调度器。"""

    def __init__(
        self,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,  # 单调逻辑时钟。
    ) -> None:
        super().__init__(monotonic_ns)

    def _select_entry(
        self,
        now_ns: int,  # 本轮选择时间；FCFS 不参与排序。
    ) -> Tuple[Optional[_QueuedRequest], bool]:
        del now_ns
        for priority in _PRIORITY_ORDER:
            heads = self._heads(priority)
            if heads:
                return min(heads, key=lambda entry: entry.sequence), False
        return None, False


class WFQScheduler(_TenantQueueScheduler):
    """带最大队首饥饿时间保护的确定性 WFQ 调度器。

    每个 priority 类别维护独立虚拟时间。请求入队时计算
    ``finish = max(class_virtual_time, tenant_last_finish) + cost / weight``，
    正常选择最小完成标签。interactive 对 batch 保持严格优先，但任意
    tenant 队首等待达到 ``max_starvation_ns`` 后，会按实际到达顺序跨类别
    提升，从而使低权重 tenant 和 batch 请求不会永久饥饿。
    """

    def __init__(
        self,
        tenant_weights: Mapping[str, int],  # tenant ID 到正整数权重的映射。
        max_starvation_ns: int,  # tenant 队首允许等待的最大纳秒数。
        monotonic_ns: Callable[[], int] = time.monotonic_ns,  # 单调逻辑时钟。
        default_weight: int = 1,  # 未显式配置 tenant 时使用的正整数权重。
    ) -> None:
        if not isinstance(tenant_weights, Mapping):
            raise SchedulerError("tenant_weights must be a mapping")
        copied_weights = {}
        for tenant_id, weight in tenant_weights.items():
            _require_identifier(tenant_id, "tenant_id")
            _require_positive_int(weight, "tenant weight")
            copied_weights[tenant_id] = weight
        _require_positive_int(default_weight, "default_weight")
        _require_positive_int(max_starvation_ns, "max_starvation_ns")
        self._tenant_weights = copied_weights
        self._default_weight = default_weight
        self._max_starvation_ns = max_starvation_ns
        self._virtual_time = {
            priority: Fraction(0) for priority in _PRIORITY_ORDER
        }
        self._tenant_last_finish: Dict[Tuple[SchedulingPriority, str], Fraction] = {}
        super().__init__(monotonic_ns)

    def _prepare_entry(
        self,
        entry: _QueuedRequest,  # 即将计算 WFQ 虚拟标签的入队记录。
    ) -> None:
        priority = entry.request.priority
        tenant_key = (priority, entry.request.tenant_id)
        virtual_start = max(
            self._virtual_time[priority],
            self._tenant_last_finish.get(tenant_key, Fraction(0)),
        )
        weight = self._tenant_weights.get(
            entry.request.tenant_id, self._default_weight
        )
        virtual_finish = virtual_start + Fraction(
            entry.request.service_cost, weight
        )
        entry.virtual_start = virtual_start
        entry.virtual_finish = virtual_finish
        self._tenant_last_finish[tenant_key] = virtual_finish

    def _select_entry(
        self,
        now_ns: int,  # 判断饥饿并记录本轮选择的单调时钟纳秒值。
    ) -> Tuple[Optional[_QueuedRequest], bool]:
        all_heads = tuple(
            entry
            for priority in _PRIORITY_ORDER
            for entry in self._heads(priority)
        )
        starved = tuple(
            entry
            for entry in all_heads
            if now_ns - entry.enqueued_at_ns >= self._max_starvation_ns
        )
        if starved:
            return min(starved, key=lambda entry: entry.sequence), True

        for priority in _PRIORITY_ORDER:
            heads = self._heads(priority)
            if heads:
                return min(
                    heads,
                    key=lambda entry: (
                        entry.virtual_finish,
                        entry.sequence,
                        entry.request.request_id,
                    ),
                ), False
        return None, False

    def _after_select(
        self,
        entry: _QueuedRequest,  # 用于推进所属优先级虚拟时间的出队记录。
    ) -> None:
        if entry.virtual_finish is None:
            raise RuntimeError("WFQ entry is missing its virtual finish tag")
        priority = entry.request.priority
        self._virtual_time[priority] = max(
            self._virtual_time[priority], entry.virtual_finish
        )

    def _virtual_time_snapshot(self) -> Tuple[Tuple[str, Fraction], ...]:
        return tuple(
            (priority.value, self._virtual_time[priority])
            for priority in _PRIORITY_ORDER
        )


def _require_identifier(
    value: str,  # 要校验的标识符值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    if type(value) is not str or not value:
        raise SchedulerError(
            "{0} must be a non-empty string".format(field_name)
        )


def _require_positive_int(
    value: int,  # 要校验的整数值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    if type(value) is not int or value < 1:
        raise SchedulerError(
            "{0} must be a positive integer".format(field_name)
        )
