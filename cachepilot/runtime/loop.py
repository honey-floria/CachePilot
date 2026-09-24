"""按三重硬预算推进 SimExecutor 的确定性调度循环。

用一个固定顺序不断推进系统，并且每一轮都同时检查请求数量、批次 Token 数和 KV 容量，绝不因为混合长短请求而超额运行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Tuple

from cachepilot.executors import (
    SimExecutor,
    SimExecutorSnapshot,
    SimRequest,
    SimRequestSnapshot,
    SimRequestState,
)
from cachepilot.runtime.scheduler import (
    ScheduleDecision,
    SchedulingPriority,
    SchedulingRequest,
)
from cachepilot.utils import CommonUtils


class RuntimeLoopError(ValueError):
    """调度循环配置、请求或账本不变量无效。"""


class RequestExceedsLoopCapacityError(RuntimeLoopError):
    """单个请求完整 KV reservation 超过循环硬容量。"""


class _SchedulingQueue(Protocol):
    """RuntimeLoop 使用的 FCFS/WFQ 最小公共接口。"""

    def enqueue(self, request: SchedulingRequest) -> None:
        """把请求加入策略队列。"""

    def select(self) -> Optional[ScheduleDecision]:
        """选择并移除下一个请求。"""


@dataclass(frozen=True)
class RuntimeLoopConfig:
    """一个 worker 调度循环的三重硬预算。"""

    max_active_sequences: int  # 同时占有执行 reservation 的最大请求数。
    max_batched_tokens: int  # 单个 tick 允许推进的 prefill/decode token 总数。
    max_kv_blocks: int  # 所有 active 请求允许占用的逻辑 KV block 总数。

    def __post_init__(self) -> None:
        """要求三项硬预算都是正整数。"""

        CommonUtils.require_positive_int(
            self.max_active_sequences,
            "max_active_sequences",
            RuntimeLoopError,
        )
        CommonUtils.require_positive_int(
            self.max_batched_tokens,
            "max_batched_tokens",
            RuntimeLoopError,
        )
        CommonUtils.require_positive_int(
            self.max_kv_blocks,
            "max_kv_blocks",
            RuntimeLoopError,
        )


@dataclass(frozen=True)
class RuntimeRequest:
    """进入 runtime 调度循环所需的请求描述。"""

    request_id: str  # 请求唯一标识。
    tenant_id: str  # Scheduler 公平队列使用的 tenant ID。
    priority: SchedulingPriority  # interactive 或 batch 调度类别。
    prompt_tokens: int  # prefill 阶段输入 token 数。
    output_tokens: int  # 模拟 decode 阶段目标输出 token 数。
    cache_hit_tokens: int = 0  # Prefix Index 返回的最长逻辑命中 token 数。
    seed: int = 0  # 请求级确定性 seed。
    cancel_after_ns: Optional[int] = None  # 相对提交时间的自动取消延迟。
    client_drain_tokens_per_tick: Optional[int] = None  # 客户端排空速率覆盖值。

    def __post_init__(self) -> None:
        """校验调度标识、token 数以及可选模拟参数。"""

        CommonUtils.require_identifier(
            self.request_id, "request_id", RuntimeLoopError
        )
        CommonUtils.require_identifier(
            self.tenant_id, "tenant_id", RuntimeLoopError
        )
        if isinstance(self.priority, str):
            try:
                priority = SchedulingPriority(self.priority)
            except ValueError as error:
                raise RuntimeLoopError(
                    "priority must be interactive or batch"
                ) from error
            object.__setattr__(self, "priority", priority)
        elif not isinstance(self.priority, SchedulingPriority):
            raise RuntimeLoopError("priority must be interactive or batch")
        CommonUtils.require_non_negative_int(
            self.prompt_tokens, "prompt_tokens", RuntimeLoopError
        )
        CommonUtils.require_non_negative_int(
            self.output_tokens, "output_tokens", RuntimeLoopError
        )
        CommonUtils.require_non_negative_int(
            self.cache_hit_tokens, "cache_hit_tokens", RuntimeLoopError
        )
        if self.cache_hit_tokens > self.prompt_tokens + self.output_tokens:
            raise RuntimeLoopError(
                "cache_hit_tokens cannot exceed total request tokens"
            )
        CommonUtils.require_non_negative_int(
            self.seed, "seed", RuntimeLoopError
        )
        if self.cancel_after_ns is not None:
            CommonUtils.require_non_negative_int(
                self.cancel_after_ns,
                "cancel_after_ns",
                RuntimeLoopError,
            )
        if self.client_drain_tokens_per_tick is not None:
            CommonUtils.require_non_negative_int(
                self.client_drain_tokens_per_tick,
                "client_drain_tokens_per_tick",
                RuntimeLoopError,
            )


@dataclass(frozen=True)
class RuntimeBudgetSnapshot:
    """某个 tick 边界上的三重预算使用量。"""

    active_sequences: int  # 当前持有执行 reservation 的请求数。
    batched_tokens: int  # 本 tick 实际推进的 token 数。
    kv_blocks: int  # 执行器当前实际占用的逻辑 KV block 数。
    reserved_kv_blocks: int  # active 请求完整生命周期 reservation 总数。


@dataclass(frozen=True)
class RuntimeTickResult:
    """单轮调度的顺序、选择和预算证据。"""

    tick: int  # 从 0 开始的调度轮次。
    reclaimed_request_ids: Tuple[str, ...]  # 本轮开始时完成并回收的请求。
    admitted_request_ids: Tuple[str, ...]  # 本轮新获得执行 reservation 的请求。
    work_allocations: Tuple[Tuple[str, int], ...]  # 本轮逐请求 token 配额。
    before: RuntimeBudgetSnapshot  # 推进执行器前的账本快照。
    after: RuntimeBudgetSnapshot  # 推进执行器后的实际账本快照。
    executor: SimExecutorSnapshot  # 本轮结束后的完整执行器快照。


@dataclass(frozen=True)
class RuntimeLoopSnapshot:
    """调度循环当前等待、活跃、完成和峰值状态。"""

    tick: int  # 下一轮将使用的 tick 序号。
    pending_request_ids: Tuple[str, ...]  # 尚未获得执行 reservation 的请求。
    active_request_ids: Tuple[str, ...]  # 当前持有执行 reservation 的请求。
    completed_request_ids: Tuple[str, ...]  # 已完成回收的请求顺序。
    current_kv_blocks: int  # 当前执行器实际 KV block 总数。
    peak_kv_blocks: int  # 调度循环观测到的实际 KV block 峰值。
    peak_batched_tokens: int  # 任一 tick 实际推进 token 数峰值。
    peak_active_sequences: int  # 执行 reservation 数峰值。


@dataclass
class _ActiveRequest:
    """循环内部持有的执行 reservation。"""

    request: RuntimeRequest  # 原始 runtime 请求。
    reserved_blocks: int  # 按完整 prompt + output 预留的 KV block 数。


class RuntimeLoop:
    """协调 Scheduler 与 SimExecutor 的三重预算 worker loop。"""

    def __init__(
        self,
        config: RuntimeLoopConfig,  # active、batch token 和 KV 硬预算。
        scheduler: _SchedulingQueue,  # FCFS 或 WFQ 调度队列。
        executor: SimExecutor,  # 接收逐请求 token 配额的模拟执行器。
    ) -> None:
        """创建空循环并验证执行器 batch 容量覆盖 active 硬上限。"""

        if not isinstance(config, RuntimeLoopConfig):
            raise TypeError("config must be a RuntimeLoopConfig")
        if not callable(getattr(scheduler, "enqueue", None)) or not callable(
            getattr(scheduler, "select", None)
        ):
            raise TypeError("scheduler must provide enqueue and select")
        if not isinstance(executor, SimExecutor):
            raise TypeError("executor must be a SimExecutor")
        if executor.config.max_batch_size < config.max_active_sequences:
            raise RuntimeLoopError(
                "executor max_batch_size cannot be less than max_active_sequences"
            )
        self._config = config
        self._scheduler = scheduler
        self._executor = executor
        self._pending: Dict[str, RuntimeRequest] = {}
        self._active: Dict[str, _ActiveRequest] = {}
        self._known_request_ids = set()
        self._completed_request_ids = []
        self._deferred_request_id: Optional[str] = None
        self._tick = 0
        self._rotation = 0
        self._current_kv_blocks = 0
        self._peak_kv_blocks = 0
        self._peak_batched_tokens = 0
        self._peak_active_sequences = 0

    @property
    def has_work(self) -> bool:
        """返回是否仍有等待、计算或等待客户端排空的请求。"""

        return bool(self._pending or self._active or self._executor.has_work)

    def submit(
        self,
        request: RuntimeRequest,  # 要加入公平调度队列的请求。
    ) -> None:
        """校验单请求 KV 上限并交给 FCFS/WFQ 等待队列。

        调用不会立即占用 active sequence 或 KV reservation；这些资源只在
        ``step`` 的完成回收阶段之后分配。
        """

        if not isinstance(request, RuntimeRequest):
            raise TypeError("request must be a RuntimeRequest")
        if request.request_id in self._known_request_ids:
            raise RuntimeLoopError(
                "request is already known: {0}".format(request.request_id)
            )
        required_blocks = self._required_blocks(request)
        if required_blocks > self._config.max_kv_blocks:
            raise RequestExceedsLoopCapacityError(
                "request requires {0} KV blocks but limit is {1}".format(
                    required_blocks,
                    self._config.max_kv_blocks,
                )
            )
        service_cost = max(1, request.prompt_tokens + request.output_tokens)
        self._scheduler.enqueue(
            SchedulingRequest(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                priority=request.priority,
                service_cost=service_cost,
                cache_hit_tokens=request.cache_hit_tokens,
            )
        )
        self._pending[request.request_id] = request
        self._known_request_ids.add(request.request_id)

    def step(self) -> RuntimeTickResult:
        """按“回收 → KV 对账 → 接纳 → 三重预算推进”执行一轮。

        上一轮已完成计算、取消或失败的请求首先释放 active/KV reservation；
        随后从执行器快照更新实际 KV 账本。只有完成这些动作后才接纳新请求，
        并为 active 请求分配不超过 batch token 与实际 KV 上限的工作额度。

        1. _reclaim_completed() 回收上一轮已经完成、取消或失败的请求
        2. _sync_kv_ledger()    根据 SimExecutor 快照重新统计实际 KV
        3. _admit_waiting()     按 FCFS/WFQ 顺序接纳新请求
        4. _allocate_work()     分配本轮每个请求可以执行多少 token
        5. executor.step(...)   执行 prefill/decode
        6. 更新 KV 与峰值，检查所有硬上限
        """

        opening_snapshot = self._executor.snapshot()
        reclaimed = self._reclaim_completed(opening_snapshot)
        self._sync_kv_ledger(opening_snapshot)
        admitted = self._admit_waiting()
        ready_snapshot = self._executor.snapshot()
        work_allocations = self._allocate_work(ready_snapshot)
        before = self._budget_snapshot(batched_tokens=0)

        progress_before = self._progress_by_request(ready_snapshot)
        executor_snapshot = self._executor.step(dict(work_allocations))
        actual_batched_tokens = self._actual_progress(
            progress_before,
            executor_snapshot,
        )
        self._sync_kv_ledger(executor_snapshot)
        self._update_peaks(actual_batched_tokens)
        after = self._budget_snapshot(batched_tokens=actual_batched_tokens)
        self._assert_hard_limits(after)

        result = RuntimeTickResult(
            tick=self._tick,
            reclaimed_request_ids=reclaimed,
            admitted_request_ids=admitted,
            work_allocations=work_allocations,
            before=before,
            after=after,
            executor=executor_snapshot,
        )
        self._tick += 1
        return result

    def run_until_idle(
        self,
        max_ticks: int = 100_000,  # 防止停滞配置造成无限循环。
    ) -> RuntimeLoopSnapshot:
        """持续执行调度轮次，直到等待、计算和 streaming 工作全部结束。"""

        CommonUtils.require_positive_int(
            max_ticks, "max_ticks", RuntimeLoopError
        )
        executed = 0
        while self.has_work and executed < max_ticks:
            self.step()
            executed += 1
        if self.has_work:
            raise RuntimeLoopError(
                "runtime loop still has work after {0} ticks".format(max_ticks)
            )
        return self.snapshot()

    def snapshot(self) -> RuntimeLoopSnapshot:
        """返回等待、活跃、完成顺序和历史峰值，不推进逻辑时钟。"""

        return RuntimeLoopSnapshot(
            tick=self._tick,
            pending_request_ids=tuple(self._pending),
            active_request_ids=tuple(self._active),
            completed_request_ids=tuple(self._completed_request_ids),
            current_kv_blocks=self._current_kv_blocks,
            peak_kv_blocks=self._peak_kv_blocks,
            peak_batched_tokens=self._peak_batched_tokens,
            peak_active_sequences=self._peak_active_sequences,
        )

    def _reclaim_completed(
        self,
        executor_snapshot: SimExecutorSnapshot,  # 本轮开始时的执行器状态。
    ) -> Tuple[str, ...]:
        """释放已结束计算请求的 active slot 和完整 KV reservation。

        ``STREAMING`` 已不再占用计算 slot 或 KV，因此与终态请求一样回收；
        客户端缓冲仍由 SimExecutor 独立排空。
        """

        states = {item.request_id: item.state for item in executor_snapshot.requests}
        reclaimed = []
        for request_id in tuple(self._active):
            state = states[request_id]
            if state in {
                SimRequestState.STREAMING,
                SimRequestState.FINISHED,
                SimRequestState.CANCELLED,
                SimRequestState.FAILED,
            }:
                del self._active[request_id]
                self._completed_request_ids.append(request_id)
                reclaimed.append(request_id)
        return tuple(reclaimed)

    def _sync_kv_ledger(
        self,
        executor_snapshot: SimExecutorSnapshot,  # 用作账本真值的执行器快照。
    ) -> None:
        """从执行器请求快照重建当前逻辑 KV block 总数。"""

        self._current_kv_blocks = sum(
            request.logical_blocks for request in executor_snapshot.requests
        )

    def _admit_waiting(self) -> Tuple[str, ...]:
        """在 active 和完整 KV reservation 均有空间时接纳等待请求。

        若当前策略选中的请求暂时放不下，保留为 deferred 队首并停止本轮
        接纳，避免绕过 FCFS/WFQ 已作出的公平顺序。
        """

        admitted = []
        while (
            self._pending
            and len(self._active) < self._config.max_active_sequences
        ):
            request_id = self._next_waiting_request_id()
            if request_id is None:
                break
            request = self._pending[request_id]
            required_blocks = self._required_blocks(request)
            if self._reserved_blocks() + required_blocks > self._config.max_kv_blocks:
                self._deferred_request_id = request_id
                break

            self._deferred_request_id = None
            del self._pending[request_id]
            self._active[request_id] = _ActiveRequest(request, required_blocks)
            self._executor.submit(
                SimRequest(
                    request_id=request.request_id,
                    prompt_tokens=request.prompt_tokens,
                    output_tokens=request.output_tokens,
                    seed=request.seed,
                    cancel_after_ns=request.cancel_after_ns,
                    client_drain_tokens_per_tick=(
                        request.client_drain_tokens_per_tick
                    ),
                )
            )
            admitted.append(request_id)
        return tuple(admitted)

    def _next_waiting_request_id(self) -> Optional[str]:
        """优先返回上轮容量阻塞请求，否则调用 Scheduler 选择新队首。"""

        if self._deferred_request_id is not None:
            return self._deferred_request_id
        decision = self._scheduler.select()
        if decision is None:
            return None
        request_id = decision.request.request_id
        if request_id not in self._pending:
            raise RuntimeLoopError(
                "scheduler selected unknown pending request: {0}".format(
                    request_id
                )
            )
        return request_id

    def _allocate_work(
        self,
        executor_snapshot: SimExecutorSnapshot,  # 推进前的请求状态。
    ) -> Tuple[Tuple[str, int], ...]:
        """在 active 请求之间分配本轮 token 预算。"""

        snapshots = {item.request_id: item for item in executor_snapshot.requests}
        active_ids = tuple(self._active)
        if not active_ids:
            return ()
        start = self._rotation % len(active_ids)
        ordered_ids = active_ids[start:] + active_ids[:start]
        self._rotation = (start + 1) % len(active_ids)

        remaining_tokens = self._config.max_batched_tokens
        projected_kv_blocks = self._current_kv_blocks
        allocations = []
        for request_id in ordered_ids:
            if remaining_tokens == 0:
                break
            request = self._active[request_id].request
            request_snapshot = snapshots[request_id]
            desired = min(
                self._desired_work(request, request_snapshot),
                remaining_tokens,
            )
            allocation, projected_request_blocks = self._fit_kv_budget(
                request_snapshot,
                desired,
                projected_kv_blocks,
            )
            if allocation == 0:
                continue
            allocations.append((request_id, allocation))
            remaining_tokens -= allocation
            projected_kv_blocks += (
                projected_request_blocks - request_snapshot.logical_blocks
            )
        return tuple(allocations)

    def _desired_work(
        self,
        request: RuntimeRequest,  # 请求固定 token 规模。
        snapshot: SimRequestSnapshot,  # 请求推进前状态。
    ) -> int:
        """计算请求本轮最多能处理多少 token。"""

        remaining_prompt = request.prompt_tokens - snapshot.prompt_tokens_processed
        if remaining_prompt > 0:
            return min(
                self._executor.config.prefill_tokens_per_tick,
                remaining_prompt,
            )
        remaining_output = request.output_tokens - snapshot.generated_tokens
        available_buffer = (
            self._executor.config.output_buffer_tokens
            - snapshot.buffered_tokens
        )
        return min(
            self._executor.config.decode_tokens_per_tick,
            remaining_output,
            available_buffer,
        )

    def _fit_kv_budget(
        self,
        snapshot: SimRequestSnapshot,  # 请求推进前状态。
        desired_tokens: int,  # batch token 预算允许的工作量。
        projected_kv_blocks: int,  # 已计入先前选择后的全局 KV block 数。
    ) -> Tuple[int, int]:
        """缩减工作量直到推进后的实际 KV block 不超过硬上限。"""

        if desired_tokens == 0:
            return 0, snapshot.logical_blocks
        current_tokens = (
            snapshot.prompt_tokens_processed + snapshot.generated_tokens
        )
        available_blocks = self._config.max_kv_blocks - projected_kv_blocks
        maximum_request_tokens = (
            snapshot.logical_blocks + available_blocks
        ) * self._executor.config.block_size
        allowed_tokens = max(0, maximum_request_tokens - current_tokens)
        allocation = min(desired_tokens, allowed_tokens)
        projected_request_blocks = CommonUtils.ceil_div(
            current_tokens + allocation,
            self._executor.config.block_size,
        )
        return allocation, projected_request_blocks

    @staticmethod
    def _progress_by_request(
        executor_snapshot: SimExecutorSnapshot,  # 推进前执行器快照。
    ) -> Mapping[str, int]:
        """记录每个请求已完成的 prefill + decode token 总量。"""

        return {
            item.request_id: (
                item.prompt_tokens_processed + item.generated_tokens
            )
            for item in executor_snapshot.requests
        }

    @staticmethod
    def _actual_progress(
        before: Mapping[str, int],  # 推进前各请求累计工作量。
        after: SimExecutorSnapshot,  # 推进后的执行器快照。
    ) -> int:
        """计算本 tick 实际执行的 prefill/decode token 总数。"""

        return sum(
            item.prompt_tokens_processed
            + item.generated_tokens
            - before.get(item.request_id, 0)
            for item in after.requests
        )

    def _budget_snapshot(
        self,
        batched_tokens: int,  # 本 tick 已实际推进的 token 数。
    ) -> RuntimeBudgetSnapshot:
        """生成当前 active、token、实际 KV 与 reservation 使用量快照。"""

        return RuntimeBudgetSnapshot(
            active_sequences=len(self._active),
            batched_tokens=batched_tokens,
            kv_blocks=self._current_kv_blocks,
            reserved_kv_blocks=self._reserved_blocks(),
        )

    def _reserved_blocks(self) -> int:
        """返回所有 active 请求完整 KV reservation 总数。"""

        return sum(item.reserved_blocks for item in self._active.values())

    def _required_blocks(
        self,
        request: RuntimeRequest,  # 要计算完整 KV reservation 的请求。
    ) -> int:
        """按 prompt + output 和执行器 block size 计算完整 reservation。"""

        return CommonUtils.ceil_div(
            request.prompt_tokens + request.output_tokens,
            self._executor.config.block_size,
        )

    def _update_peaks(self, batched_tokens: int) -> None:
        """更新实际 KV、batch token 和 active sequence 历史峰值。"""

        self._peak_kv_blocks = max(
            self._peak_kv_blocks,
            self._current_kv_blocks,
        )
        self._peak_batched_tokens = max(
            self._peak_batched_tokens,
            batched_tokens,
        )
        self._peak_active_sequences = max(
            self._peak_active_sequences,
            len(self._active),
        )

    def _assert_hard_limits(self, budget: RuntimeBudgetSnapshot) -> None:
        """在每轮末验证三重硬预算和 reservation 不变量。"""

        if budget.active_sequences > self._config.max_active_sequences:
            raise RuntimeLoopError("active sequence hard limit exceeded")
        if budget.batched_tokens > self._config.max_batched_tokens:
            raise RuntimeLoopError("batched token hard limit exceeded")
        if budget.kv_blocks > self._config.max_kv_blocks:
            raise RuntimeLoopError("KV block hard limit exceeded")
        if budget.reserved_kv_blocks > self._config.max_kv_blocks:
            raise RuntimeLoopError("KV reservation hard limit exceeded")
