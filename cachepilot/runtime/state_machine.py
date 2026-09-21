"""请求生命周期状态机。

状态机保证主路径顺序、终态不可逆、事件幂等，以及终态后禁止输出 token。
它不负责调度或资源分配，这些副作用由 Registry 和 Runtime 协调。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class RequestState(str, Enum):
    """请求从接收到结束的稳定状态集合。"""

    RECEIVED = "RECEIVED"
    TOKENIZED = "TOKENIZED"
    QUEUED = "QUEUED"
    ADMITTED = "ADMITTED"
    ROUTED = "ROUTED"
    EXECUTING = "EXECUTING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATES = frozenset(
    {
        RequestState.FINISHED,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.REJECTED,
        RequestState.FAILED,
    }
)

_MAIN_PATH = {
    RequestState.RECEIVED: RequestState.TOKENIZED,
    RequestState.TOKENIZED: RequestState.QUEUED,
    RequestState.QUEUED: RequestState.ADMITTED,
    RequestState.ADMITTED: RequestState.ROUTED,
    RequestState.ROUTED: RequestState.EXECUTING,
    RequestState.EXECUTING: RequestState.FINISHED,
}


class StateMachineError(ValueError):
    """请求生命周期事件无效时抛出的基类。"""


class InvalidTransitionError(StateMachineError):
    """目标状态不是当前状态的合法后继时抛出。"""


class EventConflictError(StateMachineError):
    """同一事件 ID 被复用于不同操作时抛出。"""


class TokenEmissionError(StateMachineError):
    """当前状态不允许输出 token 时抛出。"""


@dataclass(frozen=True)
class StateTransition:
    """一条已提交的状态事件。

    ``previous_state`` 仅在初始 RECEIVED 事件中为 ``None``。
    """

    event_id: str
    previous_state: Optional[RequestState]
    state: RequestState


@dataclass(frozen=True)
class TransitionResult:
    """状态提交结果；``applied=False`` 表示同一事件的幂等重放。"""

    transition: StateTransition
    applied: bool
    current_state: RequestState


@dataclass(frozen=True)
class StateMachineSnapshot:
    """同一时刻读取的状态、终态标记、token 计数和事件日志。"""

    state: RequestState
    terminal: bool
    emitted_token_count: int
    transitions: Tuple[StateTransition, ...]


class RequestStateMachine:
    """管理单个请求的原子、可幂等重放生命周期。

    Args:
        request_id: 状态机所属的非空请求 ID。
        received_event_id: 创建请求时 RECEIVED 事件的唯一 ID。
    """

    def __init__(self, request_id: str, received_event_id: str) -> None:
        self._validate_identifier(request_id, "request_id")
        self._validate_identifier(received_event_id, "received_event_id")
        self._request_id = request_id
        self._state = RequestState.RECEIVED
        self._transitions = [
            StateTransition(
                event_id=received_event_id,
                previous_state=None,
                state=RequestState.RECEIVED,
            )
        ]
        self._events: Dict[str, Tuple[str, object]] = {
            received_event_id: ("transition", RequestState.RECEIVED)
        }
        self._emitted_token_count = 0
        self._lock = threading.Lock()

    @property
    def request_id(self) -> str:
        """返回状态机所属请求 ID；该值创建后不变。"""

        return self._request_id

    @property
    def state(self) -> RequestState:
        """原子读取当前请求状态。"""

        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        """返回请求是否已经进入任一不可逆终态。"""

        with self._lock:
            return self._state in TERMINAL_STATES

    @property
    def emitted_token_count(self) -> int:
        """返回已成功登记的输出 token 事件数量。"""

        with self._lock:
            return self._emitted_token_count

    @property
    def transitions(self) -> Tuple[StateTransition, ...]:
        """返回不可变的状态事件序列副本。"""

        with self._lock:
            return tuple(self._transitions)

    def snapshot(self) -> StateMachineSnapshot:
        """在同一临界区读取相互一致的状态、计数和事件日志。"""

        with self._lock:
            return StateMachineSnapshot(
                state=self._state,
                terminal=self._state in TERMINAL_STATES,
                emitted_token_count=self._emitted_token_count,
                transitions=tuple(self._transitions),
            )

    def transition(
        self, target: RequestState, event_id: str
    ) -> TransitionResult:
        """原子应用状态事件。

        Args:
            target: 希望进入的目标状态。
            event_id: 本次事件的全局唯一 ID，用于幂等重放。

        Returns:
            状态转换结果。重复提交相同 ``event_id + target`` 时
            ``applied`` 为 ``False``，但仍返回最初那条转换。

        Raises:
            EventConflictError: event ID 已被不同事件占用。
            InvalidTransitionError: 目标不是合法后继，或请求已在终态。
        """

        if not isinstance(target, RequestState):
            raise TypeError("target must be a RequestState")
        self._validate_identifier(event_id, "event_id")

        with self._lock:
            # 所有事件共用 ID 命名空间，避免状态与 token 事件冲突。
            existing = self._events.get(event_id)
            if existing is not None:
                if existing != ("transition", target):
                    raise EventConflictError(
                        "event_id has already been used for a different event"
                    )
                transition = next(
                    item for item in self._transitions if item.event_id == event_id
                )
                return TransitionResult(
                    transition=transition,
                    applied=False,
                    current_state=self._state,
                )

            # 先验证再修改，非法事件不会污染状态或事件日志。
            if not self._can_transition(self._state, target):
                raise InvalidTransitionError(
                    "cannot transition request {0} from {1} to {2}".format(
                        self._request_id,
                        self._state.value,
                        target.value,
                    )
                )

            transition = StateTransition(
                event_id=event_id,
                previous_state=self._state,
                state=target,
            )
            self._state = target
            self._transitions.append(transition)
            self._events[event_id] = ("transition", target)
            return TransitionResult(
                transition=transition,
                applied=True,
                current_state=self._state,
            )

    def record_token_emission(self, event_id: str) -> bool:
        """登记一次 token 输出，并告诉调用方是否应真正发送。

        Args:
            event_id: token 输出事件的唯一 ID。

        Returns:
            首次登记返回 ``True``；同一 token 事件重放返回 ``False``，
            调用方应避免重复向客户端发送 token。

        Raises:
            TokenEmissionError: 请求不处于 EXECUTING 状态。
            EventConflictError: event ID 已用于其他事件。
        """

        self._validate_identifier(event_id, "event_id")
        with self._lock:
            existing = self._events.get(event_id)
            if existing is not None:
                if existing != ("token", 1):
                    raise EventConflictError(
                        "event_id has already been used for a different event"
                    )
                return False

            if self._state is not RequestState.EXECUTING:
                raise TokenEmissionError(
                    "request {0} cannot emit tokens while in {1}".format(
                        self._request_id,
                        self._state.value,
                    )
                )

            self._events[event_id] = ("token", 1)
            self._emitted_token_count += 1
            return True

    @staticmethod
    def _can_transition(current: RequestState, target: RequestState) -> bool:
        """判断转换是否合法；非终态均可直接进入错误终态。"""

        if current in TERMINAL_STATES:
            return False
        if target in TERMINAL_STATES:
            return True
        return _MAIN_PATH.get(current) is target

    @staticmethod
    def _validate_identifier(value: str, field: str) -> None:
        """验证请求 ID 和事件 ID 均为非空字符串。"""

        if type(value) is not str or not value:
            raise ValueError("{0} must be a non-empty string".format(field))
