"""Gateway 可替换的聊天生成后端边界。"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from cachepilot.gateway.contracts import ValidatedChatRequest


@dataclass(frozen=True)
class GeneratedText:
    """一次生成增量及其可靠 token 数。"""

    text: str
    token_count: int = 1


class ChatBackend(Protocol):
    """Torch、vLLM 和测试后端共同实现的最小生成接口。"""

    def generate(
        self,
        request: ValidatedChatRequest,
    ) -> AsyncIterator[GeneratedText]:
        """按顺序生成文本增量。"""

    async def cancel(self, request_id: str, reason: str = "explicit") -> None:
        """把带原因的取消信号传播给执行器。"""

    async def is_ready(self) -> bool:
        """返回后端当前是否可以接收请求。"""


class PromptTokenCounter(Protocol):
    """使用已配置 tokenizer 计算正式 prompt token 数的边界。"""

    def count_prompt_tokens(self, request: ValidatedChatRequest) -> int:
        """返回应用 chat template 后的 prompt token 数。"""


class ConservativePromptTokenCounter:
    """不加载模型时使用的保守开发计数器。

    该计数只用于 Gateway 的 CPU 开发服务容量保护。真实执行器必须注入与固定
    tokenizer/chat template 一致的实现，不能把这里的字节上界用于性能报告。
    """

    def count_prompt_tokens(self, request: ValidatedChatRequest) -> int:
        content_bytes = sum(
            len(message.content.encode("utf-8")) for message in request.messages
        )
        template_overhead = len(request.messages) * 4 + 2
        return content_bytes + template_overhead


class DeterministicChatBackend:
    """用于 API 验收的确定性 CPU 开发后端，不宣称模型推理能力。"""

    def __init__(self, *, token_delay_seconds: float = 0.0) -> None:
        if token_delay_seconds < 0:
            raise ValueError("token_delay_seconds must be non-negative")
        self._token_delay_seconds = token_delay_seconds
        self._cancelled = set()
        self._cancel_reasons = {}

    async def is_ready(self) -> bool:
        return True

    async def cancel(self, request_id: str, reason: str = "explicit") -> None:
        self._cancelled.add(request_id)
        self._cancel_reasons[request_id] = reason

    @property
    def cancel_reasons(self):
        """返回只读风格的取消原因快照，供验收和观测使用。"""

        return dict(self._cancel_reasons)

    async def generate(
        self,
        request: ValidatedChatRequest,
    ) -> AsyncIterator[GeneratedText]:
        last_user_message = next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        )
        response = "Echo: {0}".format(last_user_message)
        pieces = re.findall(r"\s+|\S+", response)
        for piece in pieces[: request.max_tokens]:
            if request.request_id in self._cancelled:
                break
            if self._token_delay_seconds:
                await asyncio.sleep(self._token_delay_seconds)
            yield GeneratedText(piece)
