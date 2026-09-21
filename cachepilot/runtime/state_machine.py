"""请求生命周期状态机。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class RequestState(str, Enum):
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
    event_id: str
    previous_state: Optional[RequestState]
    state: RequestState


@dataclass(frozen=True)
class TransitionResult:
    transition: StateTransition
    applied: bool
    current_state: RequestState


@dataclass(frozen=True)
class StateMachineSnapshot:
    state: RequestState
    terminal: bool
    emitted_token_count: int
    transitions: Tuple[StateTransition, ...]


class RequestStateMachine:
    """管理单个请求的原子、可幂等重放生命周期。"""

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
        return self._request_id

    @property
    def state(self) -> RequestState:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in TERMINAL_STATES

    @property
    def emitted_token_count(self) -> int:
        with self._lock:
            return self._emitted_token_count

    @property
    def transitions(self) -> Tuple[StateTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def snapshot(self) -> StateMachineSnapshot:
        """在同一临界区读取状态、计数和事件日志。"""

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

        重复的相同事件返回原结果而不重复转换。
        """

        if not isinstance(target, RequestState):
            raise TypeError("target must be a RequestState")
        self._validate_identifier(event_id, "event_id")

        with self._lock:
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
        """登记 token 输出事件，并返回是否应执行本次实际输出。"""

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
        if current in TERMINAL_STATES:
            return False
        if target in TERMINAL_STATES:
            return True
        return _MAIN_PATH.get(current) is target

    @staticmethod
    def _validate_identifier(value: str, field: str) -> None:
        if type(value) is not str or not value:
            raise ValueError("{0} must be a non-empty string".format(field))
