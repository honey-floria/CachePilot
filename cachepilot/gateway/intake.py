"""聊天补全提交的校验优先边界。

HTTP 适配器必须通过此边界提交已解析的请求。将校验步骤放在生命周期接收器
之前，可防止被拒绝的输入创建注册表（Registry）条目或 KV 预留。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from cachepilot.gateway.contracts import (
    MAX_MAX_TOKENS,
    ValidatedChatRequest,
    validate_chat_completion_request,
)


class AcceptedRequestSink(Protocol):
    """已通过校验的请求进入下游生命周期的入口。"""

    def accept(self, request: ValidatedChatRequest) -> None:
        """登记已接受的请求，以便后续准入和执行。"""


class ChatRequestIntake:
    """在请求进入运行时状态之前完成全部校验。"""

    def __init__(
        self,
        *,
        configured_model: str,
        sink: AcceptedRequestSink,
        max_tokens_limit: int = MAX_MAX_TOKENS,
    ) -> None:
        self._configured_model = configured_model
        self._sink = sink
        self._max_tokens_limit = max_tokens_limit

    def submit(
        self,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> ValidatedChatRequest:
        """校验请求，然后将规范化结果交给生命周期接收器。"""

        request = validate_chat_completion_request(
            body,
            headers,
            configured_model=self._configured_model,
            max_tokens_limit=self._max_tokens_limit,
        )
        self._sink.accept(request)
        return request
