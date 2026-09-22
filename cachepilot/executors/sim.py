"""使用可控逻辑时钟的确定性模拟执行器。

SimExecutor 不执行真实模型计算。它按固定 tick 推进 prefill 和 decode，维护
逻辑 KV block、连续 batch、客户端输出缓冲，并提供取消与 worker 故障注入。
相同配置、请求序列和逻辑时钟会产生逐项相同的事件与统计快照。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional, Tuple


class SimExecutorError(ValueError):
    """模拟执行器配置或操作无效。"""


class WorkerUnavailableError(SimExecutorError):
    """worker 故障后仍提交或推进工作时抛出。"""


class SimulationLimitError(SimExecutorError):
    """运行达到 tick 上限但仍有未完成工作时抛出。"""


class SimRequestState(str, Enum):
    """模拟请求在执行器内部的稳定阶段。"""

    QUEUED = "QUEUED"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    STREAMING = "STREAMING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class SimEventKind(str, Enum):
    """用于确定性回放与测试的模拟事件类型。"""

    SUBMITTED = "submitted"
    PREFILL_STARTED = "prefill_started"
    PREFILL_PROGRESS = "prefill_progress"
    PREFILL_COMPLETED = "prefill_completed"
    TOKEN_GENERATED = "token_generated"
    TOKENS_DELIVERED = "tokens_delivered"
    CLIENT_BACKPRESSURE = "client_backpressure"
    KV_GROWN = "kv_grown"
    KV_RELEASED = "kv_released"
    COMPUTE_FINISHED = "compute_finished"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    WORKER_FAILED = "worker_failed"
    REQUEST_FAILED = "request_failed"


TERMINAL_SIM_STATES = frozenset(
    {
        SimRequestState.FINISHED,
        SimRequestState.CANCELLED,
        SimRequestState.FAILED,
    }
)


class LogicalClock:
    """只由调用方或 SimExecutor 显式推进的非负纳秒时钟。"""

    def __init__(
        self,
        initial_ns: int = 0,  # 时钟初始纳秒值。
    ) -> None:
        _require_non_negative_int(initial_ns, "initial_ns")
        self._now_ns = initial_ns

    def __call__(self) -> int:
        """返回当前逻辑纳秒值，不自动推进。"""

        return self._now_ns

    def advance(
        self,
        delta_ns: int,  # 本次要向前推进的纳秒数。
    ) -> int:
        """推进时钟并返回新时间。"""

        _require_non_negative_int(delta_ns, "delta_ns")
        self._now_ns += delta_ns
        return self._now_ns


@dataclass(frozen=True)
class SimExecutorConfig:
    """控制模拟粒度、batch、KV 与客户端缓冲的固定配置。"""

    tick_ns: int  # 每次 ``step`` 后逻辑时钟推进的纳秒数。
    block_size: int  # 一个逻辑 KV block 可容纳的 token 数。
    max_batch_size: int  # 同时参与 prefill/decode 的最大请求数。
    prefill_tokens_per_tick: int  # 每个活跃请求每 tick 处理的 prompt token 数。
    decode_tokens_per_tick: int  # 每个解码请求每 tick 最多生成的 token 数。
    output_buffer_tokens: int  # 每个请求尚未交付客户端的 token 缓冲上限。
    client_drain_tokens_per_tick: int  # 默认每 tick 可交付客户端的 token 数。
    seed: int = 0  # 运行根 seed，写入快照以固定实验配置。
    worker_id: str = "sim-worker-0"  # 模拟 worker 的稳定标识。

    def __post_init__(self) -> None:
        for field_name in (
            "tick_ns",
            "block_size",
            "max_batch_size",
            "prefill_tokens_per_tick",
            "decode_tokens_per_tick",
            "output_buffer_tokens",
        ):
            _require_positive_int(getattr(self, field_name), field_name)
        _require_non_negative_int(
            self.client_drain_tokens_per_tick,
            "client_drain_tokens_per_tick",
        )
        _require_non_negative_int(self.seed, "seed")
        _require_identifier(self.worker_id, "worker_id")


@dataclass(frozen=True)
class SimRequest:
    """提交给 SimExecutor 的确定性请求描述。"""

    request_id: str  # 请求唯一标识。
    prompt_tokens: int  # prefill 阶段需要处理的输入 token 数。
    output_tokens: int  # decode 阶段计划生成的输出 token 数。
    seed: int = 0  # 请求级 seed，写入事件以支持 trace 核对。
    cancel_after_ns: Optional[int] = None  # 相对提交时间的自动取消延迟。
    client_drain_tokens_per_tick: Optional[int] = None  # 请求级客户端排空速率。

    def __post_init__(self) -> None:
        _require_identifier(self.request_id, "request_id")
        _require_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _require_non_negative_int(self.output_tokens, "output_tokens")
        _require_non_negative_int(self.seed, "seed")
        if self.cancel_after_ns is not None:
            _require_non_negative_int(self.cancel_after_ns, "cancel_after_ns")
        if self.client_drain_tokens_per_tick is not None:
            _require_non_negative_int(
                self.client_drain_tokens_per_tick,
                "client_drain_tokens_per_tick",
            )


@dataclass(frozen=True)
class SimEvent:
    """一条不可变且可稳定比较的模拟事件。"""

    sequence: int  # 全局严格递增事件序号。
    at_ns: int  # 事件发生时的逻辑时钟纳秒值。
    kind: SimEventKind  # 事件类型。
    request_id: Optional[str]  # 关联请求 ID；worker 级事件为空。
    details: Tuple[Tuple[str, object], ...]  # 按调用顺序保存的稳定详情键值。


@dataclass(frozen=True)
class SimRequestSnapshot:
    """单个模拟请求的不可变执行快照。"""

    request_id: str  # 请求唯一标识。
    state: SimRequestState  # 当前模拟阶段或终态。
    prompt_tokens_processed: int  # 已完成 prefill 的 token 数。
    generated_tokens: int  # 已生成的输出 token 数。
    delivered_tokens: int  # 已交付给客户端的输出 token 数。
    buffered_tokens: int  # 仍在客户端输出缓冲中的 token 数。
    logical_blocks: int  # 当前占用的逻辑 KV block 数。
    peak_logical_blocks: int  # 请求生命周期内的逻辑 KV block 峰值。
    submitted_at_ns: int  # 请求提交的逻辑时钟纳秒值。
    terminal_at_ns: Optional[int]  # 请求进入终态的逻辑时间。


@dataclass(frozen=True)
class SimExecutorStats:
    """一次模拟运行的累计统计。"""

    ticks: int  # 已执行的逻辑 tick 数。
    submitted_requests: int  # 已成功提交的请求总数。
    finished_requests: int  # 正常完成并交付全部输出的请求数。
    cancelled_requests: int  # 被取消的请求数。
    failed_requests: int  # 因 worker 故障失败的请求数。
    generated_tokens: int  # 所有请求累计生成的输出 token 数。
    delivered_tokens: int  # 所有请求累计交付客户端的 token 数。
    current_logical_blocks: int  # 当前所有请求占用的逻辑 KV block 数。
    peak_logical_blocks: int  # 运行期间逻辑 KV block 总占用峰值。
    peak_active_sequences: int  # 运行期间同时计算的请求数峰值。


@dataclass(frozen=True)
class SimExecutorSnapshot:
    """worker、请求、事件和统计的一致只读快照。"""

    worker_id: str  # 模拟 worker 标识。
    healthy: bool  # worker 当前是否可接收并推进工作。
    now_ns: int  # 快照时的逻辑时钟值。
    seed: int  # 配置中记录的运行根 seed。
    queued_request_ids: Tuple[str, ...]  # 尚未进入计算 batch 的请求顺序。
    active_request_ids: Tuple[str, ...]  # 当前参与 prefill/decode 的请求顺序。
    requests: Tuple[SimRequestSnapshot, ...]  # 按提交顺序排列的请求快照。
    events: Tuple[SimEvent, ...]  # 完整且按 sequence 排列的事件日志。
    stats: SimExecutorStats  # 累计模拟统计。


@dataclass
class _SimRequestRecord:
    """SimExecutor 内部可变请求记录。"""

    request: SimRequest  # 调用方提交的不可变请求描述。
    sequence: int  # 请求提交顺序。
    state: SimRequestState  # 当前模拟阶段或终态。
    submitted_at_ns: int  # 请求提交时间。
    cancel_at_ns: Optional[int]  # 自动取消的绝对逻辑时间。
    prompt_tokens_processed: int = 0  # 已完成 prefill 的 token 数。
    generated_tokens: int = 0  # 已生成的输出 token 数。
    delivered_tokens: int = 0  # 已交付客户端的输出 token 数。
    buffered_tokens: int = 0  # 尚未交付客户端的输出 token 数。
    logical_blocks: int = 0  # 当前逻辑 KV block 数。
    peak_logical_blocks: int = 0  # 历史逻辑 KV block 峰值。
    terminal_at_ns: Optional[int] = None  # 进入终态的逻辑时间。


class SimExecutor:
    """单线程、tick 驱动的确定性 continuous-batching 模拟器。"""

    def __init__(
        self,
        config: SimExecutorConfig,  # 固定 tick、batch、KV 和缓冲配置。
        clock: Optional[LogicalClock] = None,  # 可选外部可控逻辑时钟。
    ) -> None:
        if not isinstance(config, SimExecutorConfig):
            raise TypeError("config must be a SimExecutorConfig")
        if clock is not None and not isinstance(clock, LogicalClock):
            raise TypeError("clock must be a LogicalClock or None")
        self._config = config
        self._clock = clock or LogicalClock()
        self._healthy = True
        self._records: Dict[str, _SimRequestRecord] = {}
        self._submission_order = []
        self._queued: Deque[str] = deque()
        self._active: Dict[str, _SimRequestRecord] = {}
        self._events = []
        self._next_event_sequence = 0
        self._ticks = 0
        self._peak_logical_blocks = 0
        self._peak_active_sequences = 0

    @property
    def clock(self) -> LogicalClock:
        """返回执行器正在使用的可控逻辑时钟。"""

        return self._clock

    @property
    def has_work(self) -> bool:
        """返回是否仍有排队、计算或等待客户端排空的请求。"""

        return any(
            record.state not in TERMINAL_SIM_STATES
            for record in self._records.values()
        )

    def submit(
        self,
        request: SimRequest,  # 要进入模拟执行队列的请求。
    ) -> None:
        """按调用顺序提交请求；worker 故障后拒绝新工作。"""

        if not isinstance(request, SimRequest):
            raise TypeError("request must be a SimRequest")
        if not self._healthy:
            raise WorkerUnavailableError("worker is not healthy")
        if request.request_id in self._records:
            raise SimExecutorError(
                "request is already known: {0}".format(request.request_id)
            )
        now_ns = self._clock()
        cancel_at_ns = (
            None
            if request.cancel_after_ns is None
            else now_ns + request.cancel_after_ns
        )
        record = _SimRequestRecord(
            request=request,
            sequence=len(self._submission_order),
            state=SimRequestState.QUEUED,
            submitted_at_ns=now_ns,
            cancel_at_ns=cancel_at_ns,
        )
        self._records[request.request_id] = record
        self._submission_order.append(request.request_id)
        self._queued.append(request.request_id)
        self._emit(
            SimEventKind.SUBMITTED,
            request.request_id,
            prompt_tokens=request.prompt_tokens,
            output_tokens=request.output_tokens,
            seed=request.seed,
        )

    def step(self) -> SimExecutorSnapshot:
        """推进一个 tick，并返回推进后的完整快照。"""

        if not self._healthy:
            raise WorkerUnavailableError("worker is not healthy")
        now_ns = self._clock()
        self._cancel_due(now_ns)
        self._drain_clients()
        self._fill_batch()
        active_request_ids = tuple(self._active)
        for request_id in active_request_ids:
            record = self._active.get(request_id)
            if record is None:
                continue
            if record.state is SimRequestState.PREFILLING:
                self._advance_prefill(record)
            elif record.state is SimRequestState.DECODING:
                self._advance_decode(record)
        self._finish_drained_streams()
        self._update_peaks()
        self._ticks += 1
        self._clock.advance(self._config.tick_ns)
        return self.snapshot()

    def run_until_idle(
        self,
        max_ticks: int = 100_000,  # 防止零速客户端等配置导致无限运行。
    ) -> SimExecutorSnapshot:
        """持续推进直到所有请求终止或达到 tick 上限。"""

        _require_positive_int(max_ticks, "max_ticks")
        executed = 0
        while self.has_work and executed < max_ticks:
            self.step()
            executed += 1
        if self.has_work:
            raise SimulationLimitError(
                "simulation still has work after {0} ticks".format(max_ticks)
            )
        return self.snapshot()

    def cancel(
        self,
        request_id: str,  # 要取消的已知请求 ID。
    ) -> bool:
        """立即取消非终态请求并释放其逻辑 KV。"""

        _require_identifier(request_id, "request_id")
        record = self._records.get(request_id)
        if record is None:
            raise SimExecutorError("request is not known: {0}".format(request_id))
        if record.state in TERMINAL_SIM_STATES:
            return False
        self._terminate(record, SimRequestState.CANCELLED, SimEventKind.CANCELLED)
        return True

    def drain_client(
        self,
        request_id: str,  # 要手动排空输出缓冲的请求 ID。
        token_count: int,  # 本次最多交付客户端的 token 数。
    ) -> int:
        """手动交付缓冲 token，适合模拟客户端恢复读取。"""

        _require_identifier(request_id, "request_id")
        _require_non_negative_int(token_count, "token_count")
        record = self._records.get(request_id)
        if record is None:
            raise SimExecutorError("request is not known: {0}".format(request_id))
        delivered = self._deliver(record, token_count)
        self._finish_drained_stream(record)
        return delivered

    def fail_worker(
        self,
        reason: str = "injected_failure",  # 写入故障事件的稳定原因。
    ) -> bool:
        """注入 worker 故障，并使所有非终态请求失败和释放 KV。"""

        _require_identifier(reason, "reason")
        if not self._healthy:
            return False
        self._healthy = False
        self._emit(SimEventKind.WORKER_FAILED, None, reason=reason)
        for request_id in tuple(self._submission_order):
            record = self._records[request_id]
            if record.state not in TERMINAL_SIM_STATES:
                self._terminate(
                    record,
                    SimRequestState.FAILED,
                    SimEventKind.REQUEST_FAILED,
                    reason=reason,
                )
        return True

    def request_snapshot(
        self,
        request_id: str,  # 要读取执行状态的请求 ID。
    ) -> SimRequestSnapshot:
        """返回指定请求的不可变快照。"""

        _require_identifier(request_id, "request_id")
        record = self._records.get(request_id)
        if record is None:
            raise SimExecutorError("request is not known: {0}".format(request_id))
        return self._request_snapshot(record)

    def snapshot(self) -> SimExecutorSnapshot:
        """返回 worker、请求、事件和统计的一致快照。"""

        terminal_counts = {
            state: sum(record.state is state for record in self._records.values())
            for state in TERMINAL_SIM_STATES
        }
        stats = SimExecutorStats(
            ticks=self._ticks,
            submitted_requests=len(self._records),
            finished_requests=terminal_counts[SimRequestState.FINISHED],
            cancelled_requests=terminal_counts[SimRequestState.CANCELLED],
            failed_requests=terminal_counts[SimRequestState.FAILED],
            generated_tokens=sum(
                record.generated_tokens for record in self._records.values()
            ),
            delivered_tokens=sum(
                record.delivered_tokens for record in self._records.values()
            ),
            current_logical_blocks=self._current_logical_blocks(),
            peak_logical_blocks=self._peak_logical_blocks,
            peak_active_sequences=self._peak_active_sequences,
        )
        return SimExecutorSnapshot(
            worker_id=self._config.worker_id,
            healthy=self._healthy,
            now_ns=self._clock(),
            seed=self._config.seed,
            queued_request_ids=tuple(self._queued),
            active_request_ids=tuple(self._active),
            requests=tuple(
                self._request_snapshot(self._records[request_id])
                for request_id in self._submission_order
            ),
            events=tuple(self._events),
            stats=stats,
        )

    def _cancel_due(
        self,
        now_ns: int,  # 本轮 tick 开始时的逻辑时间。
    ) -> None:
        for request_id in tuple(self._submission_order):
            record = self._records[request_id]
            if (
                record.state not in TERMINAL_SIM_STATES
                and record.cancel_at_ns is not None
                and now_ns >= record.cancel_at_ns
            ):
                self._terminate(
                    record,
                    SimRequestState.CANCELLED,
                    SimEventKind.CANCELLED,
                    automatic=True,
                )

    def _drain_clients(self) -> None:
        for request_id in tuple(self._submission_order):
            record = self._records[request_id]
            if record.state in TERMINAL_SIM_STATES:
                continue
            drain_rate = record.request.client_drain_tokens_per_tick
            if drain_rate is None:
                drain_rate = self._config.client_drain_tokens_per_tick
            self._deliver(record, drain_rate)

    def _deliver(
        self,
        record: _SimRequestRecord,  # 要从输出缓冲交付 token 的请求记录。
        token_count: int,  # 本次最多交付的 token 数。
    ) -> int:
        delivered = min(record.buffered_tokens, token_count)
        if delivered == 0:
            return 0
        record.buffered_tokens -= delivered
        record.delivered_tokens += delivered
        self._emit(
            SimEventKind.TOKENS_DELIVERED,
            record.request.request_id,
            count=delivered,
            delivered_tokens=record.delivered_tokens,
            buffered_tokens=record.buffered_tokens,
        )
        return delivered

    def _fill_batch(self) -> None:
        while self._queued and len(self._active) < self._config.max_batch_size:
            request_id = self._queued.popleft()
            record = self._records[request_id]
            if record.state is not SimRequestState.QUEUED:
                continue
            self._active[request_id] = record
            record.state = SimRequestState.PREFILLING
            self._emit(SimEventKind.PREFILL_STARTED, request_id)
            if record.request.prompt_tokens == 0:
                self._complete_prefill(record)

    def _advance_prefill(
        self,
        record: _SimRequestRecord,  # 当前处于 PREFILLING 的请求记录。
    ) -> None:
        remaining = record.request.prompt_tokens - record.prompt_tokens_processed
        processed = min(self._config.prefill_tokens_per_tick, remaining)
        record.prompt_tokens_processed += processed
        self._grow_kv(record, record.prompt_tokens_processed)
        self._emit(
            SimEventKind.PREFILL_PROGRESS,
            record.request.request_id,
            processed_tokens=processed,
            total_processed=record.prompt_tokens_processed,
        )
        if record.prompt_tokens_processed == record.request.prompt_tokens:
            self._complete_prefill(record)

    def _complete_prefill(
        self,
        record: _SimRequestRecord,  # 已完成全部 prompt token 的请求记录。
    ) -> None:
        self._emit(
            SimEventKind.PREFILL_COMPLETED,
            record.request.request_id,
            prompt_tokens=record.prompt_tokens_processed,
        )
        if record.request.output_tokens == 0:
            self._finish_compute(record)
            return
        record.state = SimRequestState.DECODING

    def _advance_decode(
        self,
        record: _SimRequestRecord,  # 当前处于 DECODING 的请求记录。
    ) -> None:
        remaining = record.request.output_tokens - record.generated_tokens
        available_buffer = (
            self._config.output_buffer_tokens - record.buffered_tokens
        )
        generated = min(
            self._config.decode_tokens_per_tick,
            remaining,
            available_buffer,
        )
        for _ in range(generated):
            record.generated_tokens += 1
            record.buffered_tokens += 1
            total_kv_tokens = (
                record.prompt_tokens_processed + record.generated_tokens
            )
            self._grow_kv(record, total_kv_tokens)
            self._emit(
                SimEventKind.TOKEN_GENERATED,
                record.request.request_id,
                generated_tokens=record.generated_tokens,
                buffered_tokens=record.buffered_tokens,
            )
        if generated < min(self._config.decode_tokens_per_tick, remaining):
            self._emit(
                SimEventKind.CLIENT_BACKPRESSURE,
                record.request.request_id,
                buffered_tokens=record.buffered_tokens,
                buffer_limit=self._config.output_buffer_tokens,
            )
        if record.generated_tokens == record.request.output_tokens:
            self._finish_compute(record)

    def _finish_compute(
        self,
        record: _SimRequestRecord,  # 已完成 prefill/decode 计算的请求记录。
    ) -> None:
        self._active.pop(record.request.request_id, None)
        self._release_kv(record)
        self._emit(
            SimEventKind.COMPUTE_FINISHED,
            record.request.request_id,
            generated_tokens=record.generated_tokens,
        )
        if record.buffered_tokens:
            record.state = SimRequestState.STREAMING
        else:
            self._finish_request(record)

    def _finish_drained_streams(self) -> None:
        for request_id in tuple(self._submission_order):
            self._finish_drained_stream(self._records[request_id])

    def _finish_drained_stream(
        self,
        record: _SimRequestRecord,  # 可能已交付全部缓冲的请求记录。
    ) -> None:
        if (
            record.state is SimRequestState.STREAMING
            and record.buffered_tokens == 0
        ):
            self._finish_request(record)

    def _finish_request(
        self,
        record: _SimRequestRecord,  # 已完成计算且缓冲已排空的请求记录。
    ) -> None:
        record.state = SimRequestState.FINISHED
        record.terminal_at_ns = self._clock()
        self._emit(
            SimEventKind.FINISHED,
            record.request.request_id,
            delivered_tokens=record.delivered_tokens,
        )

    def _grow_kv(
        self,
        record: _SimRequestRecord,  # 要根据 token 数更新 KV 的请求记录。
        kv_tokens: int,  # 当前需要 KV 保存的 token 总数。
    ) -> None:
        required_blocks = _ceil_div(kv_tokens, self._config.block_size)
        if required_blocks <= record.logical_blocks:
            return
        previous_blocks = record.logical_blocks
        record.logical_blocks = required_blocks
        record.peak_logical_blocks = max(
            record.peak_logical_blocks,
            required_blocks,
        )
        self._emit(
            SimEventKind.KV_GROWN,
            record.request.request_id,
            previous_blocks=previous_blocks,
            logical_blocks=required_blocks,
            kv_tokens=kv_tokens,
        )
        self._update_peaks()

    def _release_kv(
        self,
        record: _SimRequestRecord,  # 要释放逻辑 KV 的请求记录。
    ) -> None:
        if record.logical_blocks == 0:
            return
        released_blocks = record.logical_blocks
        record.logical_blocks = 0
        self._emit(
            SimEventKind.KV_RELEASED,
            record.request.request_id,
            released_blocks=released_blocks,
        )

    def _terminate(
        self,
        record: _SimRequestRecord,  # 要进入异常终态的请求记录。
        state: SimRequestState,  # CANCELLED 或 FAILED 终态。
        event_kind: SimEventKind,  # 与终态对应的事件类型。
        **details: object,  # 写入终态事件的附加稳定详情。
    ) -> None:
        self._remove_from_queue(record.request.request_id)
        self._active.pop(record.request.request_id, None)
        self._release_kv(record)
        record.buffered_tokens = 0
        record.state = state
        record.terminal_at_ns = self._clock()
        self._emit(event_kind, record.request.request_id, **details)

    def _remove_from_queue(
        self,
        request_id: str,  # 要从等待队列移除的请求 ID。
    ) -> None:
        if request_id not in self._queued:
            return
        self._queued = deque(
            queued_id for queued_id in self._queued if queued_id != request_id
        )

    def _emit(
        self,
        kind: SimEventKind,  # 要追加的事件类型。
        request_id: Optional[str],  # 关联请求 ID；worker 事件为空。
        **details: object,  # 按调用顺序记录的事件详情。
    ) -> None:
        self._events.append(
            SimEvent(
                sequence=self._next_event_sequence,
                at_ns=self._clock(),
                kind=kind,
                request_id=request_id,
                details=tuple(details.items()),
            )
        )
        self._next_event_sequence += 1

    def _current_logical_blocks(self) -> int:
        return sum(record.logical_blocks for record in self._records.values())

    def _update_peaks(self) -> None:
        self._peak_logical_blocks = max(
            self._peak_logical_blocks,
            self._current_logical_blocks(),
        )
        self._peak_active_sequences = max(
            self._peak_active_sequences,
            len(self._active),
        )

    @staticmethod
    def _request_snapshot(
        record: _SimRequestRecord,  # 要转换为不可变快照的内部请求记录。
    ) -> SimRequestSnapshot:
        return SimRequestSnapshot(
            request_id=record.request.request_id,
            state=record.state,
            prompt_tokens_processed=record.prompt_tokens_processed,
            generated_tokens=record.generated_tokens,
            delivered_tokens=record.delivered_tokens,
            buffered_tokens=record.buffered_tokens,
            logical_blocks=record.logical_blocks,
            peak_logical_blocks=record.peak_logical_blocks,
            submitted_at_ns=record.submitted_at_ns,
            terminal_at_ns=record.terminal_at_ns,
        )


def _ceil_div(
    dividend: int,  # 被除数。
    divisor: int,  # 正整数除数。
) -> int:
    if dividend == 0:
        return 0
    return (dividend + divisor - 1) // divisor


def _require_identifier(
    value: str,  # 要校验的标识符值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    if type(value) is not str or not value:
        raise SimExecutorError(
            "{0} must be a non-empty string".format(field_name)
        )


def _require_positive_int(
    value: int,  # 要校验的正整数值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    if type(value) is not int or value < 1:
        raise SimExecutorError(
            "{0} must be a positive integer".format(field_name)
        )


def _require_non_negative_int(
    value: int,  # 要校验的非负整数值。
    field_name: str,  # 错误消息中使用的字段名称。
) -> None:
    if type(value) is not int or value < 0:
        raise SimExecutorError(
            "{0} must be a non-negative integer".format(field_name)
        )
