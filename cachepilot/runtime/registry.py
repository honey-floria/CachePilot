"""线程安全的内存请求 Registry。

Registry 是请求契约、生命周期状态机和资源租约之间的协调层。它负责
request ID 唯一性、租户作用域幂等键、事件查询，以及终态后的资源释放。
"""

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
    """供 API、调度器和观测组件读取的请求只读视图。

    Attributes:
        request: 已通过 Gateway 严格校验的规范化请求。
        state: 当前生命周期状态。
        terminal: 当前状态是否为不可逆终态。
        emitted_token_count: 已登记的输出 token 数。
        events: 从 RECEIVED 开始的不可变状态事件序列。
    """

    request: ValidatedChatRequest
    state: RequestState
    terminal: bool
    emitted_token_count: int
    events: Tuple[StateTransition, ...]

    @property
    def request_id(self) -> str:
        """便捷返回规范化请求中的 request ID。"""

        return self.request.request_id

    @property
    def tenant_id(self) -> str:
        """便捷返回拥有该请求的 tenant ID。"""

        return self.request.tenant_id


@dataclass(frozen=True)
class _RegistryEntry:
    """Registry 内部条目；绑定请求、幂等指纹和独立状态机。"""

    request: ValidatedChatRequest
    fingerprint: str
    machine: RequestStateMachine


class RequestRegistry:
    """按 request ID 和租户幂等键管理请求生命周期。

    Args:
        resource_capacity_blocks: 传给资源账本的可选逻辑 KV block 上限。
            ``None`` 表示 Registry 只记账，不在此层设置容量上限。
    """

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
        """原子注册请求，或返回与现有请求一致的去重结果。

        Args:
            request: Gateway 已验证并规范化的聊天请求。
            received_event_id: 可选的 RECEIVED 事件 ID；缺省时由 request ID
                确定性生成，便于重放。

        Returns:
            ``ClaimResult``，区分首次接受、request ID 冲突、幂等请求
            处理中、终态不可重放，以及相同幂等键承载不同请求等情况。

        Note:
            幂等键作用域包含 tenant 和 API path，因此不同 tenant 使用
            相同 key 不会互相观察或去重。
        """

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        if received_event_id is None:
            received_event_id = "received:{0}".format(request.request_id)

        fingerprint = request.fingerprint()
        with self._lock:
            # 先处理语义更强的幂等键，再检查裸 request ID 冲突。
            idempotency_scope = self._idempotency_scope(request)
            if idempotency_scope is not None:
                existing_request_id = self._idempotency_keys.get(
                    idempotency_scope
                )
                if existing_request_id is not None:
                    entry = self._requests[existing_request_id]
                    # 相同 key 只能重试完全一致的规范化请求。
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
        """按 request ID 返回当前状态与不可变事件日志快照。

        Raises:
            RequestNotFoundError: request ID 未注册。
        """

        return self._snapshot(self._entry(request_id))

    def get_for_tenant(
        self, request_id: str, tenant_id: str
    ) -> Optional[RequestSnapshot]:
        """按租户安全查询请求。

        不存在和跨租户访问统一返回 ``None``，防止调用方利用响应差异
        探测其他 tenant 的 request ID。
        """

        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None or entry.request.tenant_id != tenant_id:
                return None
        return self._snapshot(entry)

    def get_by_idempotency_key(
        self, tenant_id: str, idempotency_key: str
    ) -> Optional[RequestSnapshot]:
        """按 tenant 与聊天接口作用域的幂等键查询原始请求。"""

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
        """将状态事件提交给指定请求的原子状态机。

        Args:
            request_id: 目标请求 ID。
            target: 希望进入的生命周期状态。
            event_id: 支持幂等重放的唯一事件 ID。

        Returns:
            底层状态机的 ``TransitionResult``。

        Logic:
            只有首次成功进入终态时才触发资源释放。状态事件重放不会重复
            释放，而资源账本本身也提供第二层幂等保护。
        """

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
        """登记 token 输出事件，并决定是否真正发送 token。

        Args:
            request_id: 正在执行的请求 ID。
            event_id: 本次 token 输出的唯一事件 ID。
        """

        return self._entry(request_id).machine.record_token_emission(event_id)

    def reserve(self, request_id: str, logical_blocks: int) -> ResourceLeaseSnapshot:
        """为处于 ADMITTED 状态的请求申请逻辑 KV reservation。

        Args:
            request_id: 已被准入的请求 ID。
            logical_blocks: KV Planner 给出的正整数 block 数。

        Raises:
            ValueError: 请求尚未进入 ADMITTED 状态。
            ResourceLeaseError: 重复预留或容量不足。
        """

        entry = self._entry(request_id)
        if entry.machine.state is not RequestState.ADMITTED:
            raise ValueError("only ADMITTED requests may reserve logical KV blocks")
        return self._resources.reserve(request_id, logical_blocks)

    def grow_reservation(
        self, request_id: str, additional_blocks: int
    ) -> ResourceLeaseSnapshot:
        """为长尾生成增长逻辑 KV reservation。

        ``additional_blocks`` 是增量而非新总量，容量检查由账本原子执行。
        """

        return self._resources.grow(request_id, additional_blocks)

    def attach_physical_handle(
        self, request_id: str, handle: object
    ) -> ResourceLeaseSnapshot:
        """绑定执行器物理 handle，并与逻辑 block 分开记账。

        Args:
            request_id: 已持有资源租约的请求 ID。
            handle: 执行器返回的 sequence/allocation 等可哈希标识。
        """

        return self._resources.attach_physical_handle(request_id, handle)

    def release_resources(self, request_id: str) -> ResourceLeaseSnapshot:
        """显式释放资源；重复调用安全且不会重复扣减总量。"""

        return self._resources.release(request_id)

    def resource_snapshot(self, request_id: str) -> ResourceLeaseSnapshot:
        """读取请求的资源租约快照，不改变状态或所有权。"""

        return self._resources.snapshot(request_id)

    def _release_if_present(self, request_id: str) -> None:
        """终态清理钩子；没有租约或已释放时静默结束。"""

        try:
            self._resources.release(request_id)
        except ResourceLeaseError:
            return

    def _entry(self, request_id: str) -> _RegistryEntry:
        """校验 ID 并取得内部条目，不把可变容器暴露给调用方。"""

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
        """把状态机快照与规范化请求组合为只读视图。"""

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
        """构造 tenant + API path + key 的隔离作用域。"""

        if request.idempotency_key is None:
            return None
        return (
            request.tenant_id,
            CHAT_COMPLETIONS_PATH,
            request.idempotency_key,
        )
