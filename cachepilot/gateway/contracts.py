"""CachePilot 首个 API 版本的严格请求契约校验。

本模块刻意不依赖任何 Web 框架。后续可由 FastAPI 封装这些类型，同时确保
该契约仍能在仅有 CPU 的基础环境中执行。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_MAX_TOKENS = 256
MAX_MAX_TOKENS = 4096
DEFAULT_DEADLINE_MS = {
    "interactive": 30_000,
    "batch": 300_000,
}
MAX_DEADLINE_MS = 3_600_000

_BODY_FIELDS = frozenset({"model", "messages", "stream", "max_tokens"})
_MESSAGE_FIELDS = frozenset({"role", "content"})
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
_PRIORITIES = frozenset({"interactive", "batch"})
_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ContractViolation(ValueError):
    """稳定且对客户端可见的请求契约违规。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        param: Optional[str] = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param
        self.status_code = status_code

    def as_error_response(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "error": {
                "type": "invalid_request_error",
                "code": self.code,
                "message": self.message,
                "param": self.param,
                "request_id": request_id,
            }
        }


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ValidatedChatRequest:
    request_id: str
    tenant_id: str
    priority: str
    deadline_ms: int
    idempotency_key: Optional[str]
    model: str
    messages: Tuple[ChatMessage, ...]
    stream: bool
    max_tokens: int

    def fingerprint(self) -> str:
        """返回幂等性防护所使用的规范指纹。"""

        canonical = {
            "path": CHAT_COMPLETIONS_PATH,
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "deadline_ms": self.deadline_ms,
            "body": {
                "model": self.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in self.messages
                ],
                "stream": self.stream,
                "max_tokens": self.max_tokens,
            },
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def validate_chat_completion_request(
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    configured_model: str,
    max_tokens_limit: int = MAX_MAX_TOKENS,
) -> ValidatedChatRequest:
    """校验并规范化 v1 流式聊天请求。

    请求头名称不区分大小写。除了解析十进制的截止时间请求头外，不会刻意
    对值进行类型强制转换。
    """

    if not isinstance(body, Mapping):
        raise ContractViolation(
            "invalid_body",
            "Request body must be a JSON object.",
            param="body",
        )

    unknown_fields = sorted(set(body) - _BODY_FIELDS)
    if unknown_fields:
        field = unknown_fields[0]
        raise ContractViolation(
            "unknown_field",
            "Unsupported request field: {0}.".format(field),
            param=field,
        )

    missing_fields = sorted({"model", "messages", "stream"} - set(body))
    if missing_fields:
        field = missing_fields[0]
        raise ContractViolation(
            "missing_field",
            "Required request field is missing: {0}.".format(field),
            param=field,
        )

    model = body["model"]
    if type(model) is not str or not model:
        raise ContractViolation(
            "invalid_type",
            "model must be a non-empty string.",
            param="model",
        )
    if model != configured_model:
        raise ContractViolation(
            "model_not_found",
            "The requested model is not served by this deployment.",
            param="model",
            status_code=404,
        )

    stream = body["stream"]
    if type(stream) is not bool or stream is not True:
        raise ContractViolation(
            "streaming_required",
            "The first API version requires stream=true.",
            param="stream",
        )

    messages_value = body["messages"]
    if type(messages_value) is not list or not messages_value:
        raise ContractViolation(
            "invalid_messages",
            "messages must be a non-empty JSON array.",
            param="messages",
        )
    messages = _validate_messages(messages_value)

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    if type(max_tokens) is not int:
        raise ContractViolation(
            "invalid_type",
            "max_tokens must be an integer.",
            param="max_tokens",
        )
    if max_tokens < 1 or max_tokens > max_tokens_limit:
        raise ContractViolation(
            "value_out_of_range",
            "max_tokens must be between 1 and {0}.".format(max_tokens_limit),
            param="max_tokens",
        )

    normalized_headers = {key.lower(): value for key, value in headers.items()}
    tenant_id = _required_header(normalized_headers, "x-tenant-id")
    if not _TENANT_PATTERN.fullmatch(tenant_id):
        raise ContractViolation(
            "invalid_tenant",
            "X-Tenant-ID has an invalid format.",
            param="X-Tenant-ID",
        )

    priority = normalized_headers.get("x-priority", "interactive")
    if priority not in _PRIORITIES:
        raise ContractViolation(
            "invalid_priority",
            "X-Priority must be interactive or batch.",
            param="X-Priority",
        )

    deadline_ms = _parse_deadline_ms(normalized_headers, priority)

    request_id = normalized_headers.get("x-request-id")
    if request_id is None:
        request_id = "req_{0}".format(uuid.uuid4().hex)
    elif not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ContractViolation(
            "invalid_request_id",
            "X-Request-ID has an invalid format.",
            param="X-Request-ID",
        )

    idempotency_key = normalized_headers.get("idempotency-key")
    if idempotency_key is not None and not _IDEMPOTENCY_KEY_PATTERN.fullmatch(
        idempotency_key
    ):
        raise ContractViolation(
            "invalid_idempotency_key",
            "Idempotency-Key has an invalid format.",
            param="Idempotency-Key",
        )

    return ValidatedChatRequest(
        request_id=request_id,
        tenant_id=tenant_id,
        priority=priority,
        deadline_ms=deadline_ms,
        idempotency_key=idempotency_key,
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
    )


def _validate_messages(values: Sequence[Any]) -> Tuple[ChatMessage, ...]:
    messages = []
    for index, value in enumerate(values):
        param = "messages[{0}]".format(index)
        if not isinstance(value, Mapping):
            raise ContractViolation(
                "invalid_message",
                "Each message must be a JSON object.",
                param=param,
            )

        unknown_fields = sorted(set(value) - _MESSAGE_FIELDS)
        if unknown_fields:
            field = unknown_fields[0]
            raise ContractViolation(
                "unknown_field",
                "Unsupported message field: {0}.".format(field),
                param="{0}.{1}".format(param, field),
            )

        missing_fields = sorted(_MESSAGE_FIELDS - set(value))
        if missing_fields:
            field = missing_fields[0]
            raise ContractViolation(
                "missing_field",
                "Required message field is missing: {0}.".format(field),
                param="{0}.{1}".format(param, field),
            )

        role = value["role"]
        if type(role) is not str or role not in _MESSAGE_ROLES:
            raise ContractViolation(
                "invalid_role",
                "Message role must be system, user, or assistant.",
                param="{0}.role".format(param),
            )

        content = value["content"]
        if type(content) is not str:
            raise ContractViolation(
                "text_content_required",
                "Message content must be a string; content parts are unsupported.",
                param="{0}.content".format(param),
            )

        messages.append(ChatMessage(role=role, content=content))

    return tuple(messages)


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name)
    if value is None or value == "":
        canonical_name = "-".join(part.capitalize() for part in name.split("-"))
        raise ContractViolation(
            "tenant_required",
            "{0} is required.".format(canonical_name),
            param=canonical_name,
        )
    return value


def _parse_deadline_ms(headers: Mapping[str, str], priority: str) -> int:
    raw_value = headers.get("x-deadline-ms")
    if raw_value is None:
        return DEFAULT_DEADLINE_MS[priority]
    if not raw_value.isascii() or not raw_value.isdecimal():
        raise ContractViolation(
            "invalid_deadline",
            "X-Deadline-Ms must be a decimal integer.",
            param="X-Deadline-Ms",
        )
    value = int(raw_value)
    if value < 1 or value > MAX_DEADLINE_MS:
        raise ContractViolation(
            "invalid_deadline",
            "X-Deadline-Ms must be between 1 and {0}.".format(MAX_DEADLINE_MS),
            param="X-Deadline-Ms",
        )
    return value


class ClaimStatus(str, Enum):
    ACCEPTED = "accepted"
    REQUEST_ID_CONFLICT = "request_id_conflict"
    IDEMPOTENCY_IN_PROGRESS = "idempotency_in_progress"
    IDEMPOTENCY_REPLAY_UNAVAILABLE = "idempotency_replay_unavailable"
    IDEMPOTENCY_KEY_CONFLICT = "idempotency_key_conflict"


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    request_id: str


@dataclass
class _IdempotencyRecord:
    request_id: str
    fingerprint: str
    terminal: bool = False


class IdempotencyGuard:
    """最小化的原子重复提交防护。

    这不是生命周期注册表（Registry）。它先固定重复提交的语义，后续可以在
    不改变外部契约的情况下改由注册表提供支持。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_ids = set()
        self._records: Dict[Tuple[str, str, str], _IdempotencyRecord] = {}
        self._record_keys_by_request_id: Dict[str, Tuple[str, str, str]] = {}

    def claim(self, request: ValidatedChatRequest) -> ClaimResult:
        with self._lock:
            if request.idempotency_key is not None:
                key = (
                    request.tenant_id,
                    CHAT_COMPLETIONS_PATH,
                    request.idempotency_key,
                )
                record = self._records.get(key)
                if record is not None:
                    if record.fingerprint != request.fingerprint():
                        return ClaimResult(
                            ClaimStatus.IDEMPOTENCY_KEY_CONFLICT,
                            record.request_id,
                        )
                    if record.terminal:
                        return ClaimResult(
                            ClaimStatus.IDEMPOTENCY_REPLAY_UNAVAILABLE,
                            record.request_id,
                        )
                    return ClaimResult(
                        ClaimStatus.IDEMPOTENCY_IN_PROGRESS,
                        record.request_id,
                    )

            if request.request_id in self._request_ids:
                return ClaimResult(
                    ClaimStatus.REQUEST_ID_CONFLICT,
                    request.request_id,
                )

            self._request_ids.add(request.request_id)
            if request.idempotency_key is not None:
                key = (
                    request.tenant_id,
                    CHAT_COMPLETIONS_PATH,
                    request.idempotency_key,
                )
                self._records[key] = _IdempotencyRecord(
                    request_id=request.request_id,
                    fingerprint=request.fingerprint(),
                )
                self._record_keys_by_request_id[request.request_id] = key

            return ClaimResult(ClaimStatus.ACCEPTED, request.request_id)

    def mark_terminal(self, request_id: str) -> None:
        with self._lock:
            key = self._record_keys_by_request_id.get(request_id)
            if key is not None:
                self._records[key].terminal = True
