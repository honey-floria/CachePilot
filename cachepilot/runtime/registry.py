"""线程安全的内存请求 Registry。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from cachepilot.gateway.contracts import (
    CHAT_COMPLETIONS_PATH,
    ClaimResult,
    ClaimStatus,
    ValidatedChatRequest,
)
from cachepilot.runtime.state_machine import (
    RequestState,
    RequestStateMachine,
    StateTransition,
    TransitionResult,
)
from cachepilot.runtime.resources import (
    ResourceLeaseError,
    ResourceLeaseManager,
    ResourceLeaseSnapshot,
)


class RegistryError(ValueError):
    """Registry 操作无效时抛出的基类。"""


class RequestNotFoundError(RegistryError):
    """请求 ID 不存在时抛出。"""


@dataclass(frozen=True)
class RequestSnapshot:
    request: ValidatedChatRequest
    state: RequestState
    terminal: bool
    emitted_token_count: int
    events: Tuple[StateTransition, ...]

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def tenant_id(self) -> str:
        return self.request.tenant_id


@dataclass(frozen=True)
class _RegistryEntry:
    request: ValidatedChatRequest
    fingerprint: str
    machine: RequestStateMachine


class RequestRegistry:
    """按 request ID 和租户幂等键管理请求生命周期。"""

    def __init__(self, resource_capacity_blocks: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[str, _RegistryEntry] = {}
        self._idempotency_keys: Dict[Tuple[str, str, str], str] = {}
        self._resources = ResourceLeaseManager(resource_capacity_blocks)

    def register(
        self,
        request: ValidatedChatRequest,
        received_event_id: Optional[str] = None,
    ) -> ClaimResult:
        """原子注册请求，或返回与现有请求一致的去重结果。"""

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        if received_event_id is None:
            received_event_id = "received:{0}".format(request.request_id)

        fingerprint = request.fingerprint()
        with self._lock:
            idempotency_scope = self._idempotency_scope(request)
            if idempotency_scope is not None:
                existing_request_id = self._idempotency_keys.get(
                    idempotency_scope
                )
                if existing_request_id is not None:
                    entry = self._requests[existing_request_id]
                    if entry.fingerprint != fingerprint:
                        return ClaimResult(
                            ClaimStatus.IDEMPOTENCY_KEY_CONFLICT,
                            existing_request_id,
                        )
                    status = (
                        ClaimStatus.IDEMPOTENCY_REPLAY_UNAVAILABLE
                        if entry.machine.is_terminal
                        else ClaimStatus.IDEMPOTENCY_IN_PROGRESS
                    )
                    return ClaimResult(status, existing_request_id)

            if request.request_id in self._requests:
                return ClaimResult(
                    ClaimStatus.REQUEST_ID_CONFLICT,
                    request.request_id,
                )

            machine = RequestStateMachine(
                request.request_id,
                received_event_id,
            )
            self._requests[request.request_id] = _RegistryEntry(
                request=request,
                fingerprint=fingerprint,
                machine=machine,
            )
            if idempotency_scope is not None:
                self._idempotency_keys[idempotency_scope] = request.request_id
            return ClaimResult(ClaimStatus.ACCEPTED, request.request_id)

    def get(self, request_id: str) -> RequestSnapshot:
        """按 request ID 返回当前状态与不可变事件日志快照。"""

        return self._snapshot(self._entry(request_id))

    def get_for_tenant(
        self, request_id: str, tenant_id: str
    ) -> Optional[RequestSnapshot]:
        """按租户安全查询；不存在和跨租户请求均返回 None。"""

        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None or entry.request.tenant_id != tenant_id:
                return None
        return self._snapshot(entry)

    def get_by_idempotency_key(
        self, tenant_id: str, idempotency_key: str
    ) -> Optional[RequestSnapshot]:
        """按租户作用域的幂等键查询原请求。"""

        scope = (tenant_id, CHAT_COMPLETIONS_PATH, idempotency_key)
        with self._lock:
            request_id = self._idempotency_keys.get(scope)
            entry = self._requests.get(request_id) if request_id else None
        return self._snapshot(entry) if entry is not None else None

    def transition(
        self,
        request_id: str,
        target: RequestState,
        event_id: str,
    ) -> TransitionResult:
        """将状态事件提交给指定请求的原子状态机。"""

        result = self._entry(request_id).machine.transition(target, event_id)
        if result.applied and result.current_state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.TIMED_OUT,
            RequestState.REJECTED,
            RequestState.FAILED,
        }:
            self._release_if_present(request_id)
        return result

    def record_token_emission(self, request_id: str, event_id: str) -> bool:
        """通过指定请求的状态机登记 token 输出。"""

        return self._entry(request_id).machine.record_token_emission(event_id)

    def reserve(self, request_id: str, logical_blocks: int) -> ResourceLeaseSnapshot:
        """为已处于 ADMITTED 状态的请求申请逻辑 KV reservation。"""

        entry = self._entry(request_id)
        if entry.machine.state is not RequestState.ADMITTED:
            raise ValueError("only ADMITTED requests may reserve logical KV blocks")
        return self._resources.reserve(request_id, logical_blocks)

    def grow_reservation(
        self, request_id: str, additional_blocks: int
    ) -> ResourceLeaseSnapshot:
        """增长请求的逻辑 KV reservation。"""

        return self._resources.grow(request_id, additional_blocks)

    def attach_physical_handle(
        self, request_id: str, handle: object
    ) -> ResourceLeaseSnapshot:
        """绑定执行器物理 handle，和逻辑 block 分开记账。"""

        return self._resources.attach_physical_handle(request_id, handle)

    def release_resources(self, request_id: str) -> ResourceLeaseSnapshot:
        """显式释放请求资源；重复调用安全且不会重复扣减。"""

        return self._resources.release(request_id)

    def resource_snapshot(self, request_id: str) -> ResourceLeaseSnapshot:
        """读取请求的资源租约快照。"""

        return self._resources.snapshot(request_id)

    def _release_if_present(self, request_id: str) -> None:
        try:
            self._resources.release(request_id)
        except ResourceLeaseError:
            return

    def _entry(self, request_id: str) -> _RegistryEntry:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be a non-empty string")
        with self._lock:
            entry = self._requests.get(request_id)
        if entry is None:
            raise RequestNotFoundError(
                "request not found: {0}".format(request_id)
            )
        return entry

    @staticmethod
    def _snapshot(entry: _RegistryEntry) -> RequestSnapshot:
        machine = entry.machine.snapshot()
        return RequestSnapshot(
            request=entry.request,
            state=machine.state,
            terminal=machine.terminal,
            emitted_token_count=machine.emitted_token_count,
            events=machine.transitions,
        )

    @staticmethod
    def _idempotency_scope(
        request: ValidatedChatRequest,
    ) -> Optional[Tuple[str, str, str]]:
        if request.idempotency_key is None:
            return None
        return (
            request.tenant_id,
            CHAT_COMPLETIONS_PATH,
            request.idempotency_key,
        )
