"""CachePilot Phase 1 FastAPI Gateway。"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Mapping, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from cachepilot.config.baseline import load_model_baseline
from cachepilot.gateway.backends import (
    ChatBackend,
    ConservativePromptTokenCounter,
    DeterministicChatBackend,
    GeneratedText,
    PromptTokenCounter,
)
from cachepilot.gateway.contracts import (
    ClaimStatus,
    ContractViolation,
    ValidatedChatRequest,
    validate_chat_completion_request,
)
from cachepilot.runtime.admission import (
    AdmissionReason,
    AdmissionStatus,
    StrictAdmissionConfig,
    StrictAdmissionController,
    TenantAdmissionLimits,
)
from cachepilot.runtime.kv_planner import KVModelSpec, KVPlanner
from cachepilot.runtime.registry import RequestRegistry, RequestSnapshot
from cachepilot.runtime.state_machine import InvalidTransitionError, RequestState


_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class GatewaySettings:
    """Gateway、KV 准入和 tenant 配额的启动配置。"""

    model_id: str
    context_limit: int = 8192
    total_kv_blocks: int = 512
    safety_kv_blocks: int = 32
    max_active_sequences: int = 32
    max_queued_requests: int = 128
    block_size: int = 16
    stream_buffer_tokens: int = 8
    stream_poll_interval_seconds: float = 0.05
    tenant_limits: Mapping[str, TenantAdmissionLimits] = field(
        default_factory=lambda: {
            "team-a": TenantAdmissionLimits(8, 8192, 32),
            "team-b": TenantAdmissionLimits(8, 8192, 32),
        }
    )


@dataclass(frozen=True)
class PreparedChatRequest:
    """已完成身份、去重、tokenization 和准入的请求。"""

    request: ValidatedChatRequest
    prompt_tokens: int
    deadline_at_monotonic: float


@dataclass(frozen=True)
class _StreamQueueItem:
    """有界 SSE 队列中的一个生成结果。"""

    generated: Optional[GeneratedText] = None
    done: bool = False


class GatewayError(Exception):
    """可稳定映射为 HTTP JSON 或 SSE error event 的错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        request_id: Optional[str] = None,
        param: Optional[str] = None,
        error_type: str = "request_error",
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.param = param
        self.error_type = error_type
        self.retry_after_seconds = retry_after_seconds

    def payload(self) -> Dict[str, Any]:
        return {
            "error": {
                "type": self.error_type,
                "code": self.code,
                "message": self.message,
                "param": self.param,
                "request_id": self.request_id,
            }
        }


class GatewayRuntime:
    """协调契约、Registry、Admission 与生成后端。"""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        backend: Optional[ChatBackend] = None,
        token_counter: Optional[PromptTokenCounter] = None,
        registry: Optional[RequestRegistry] = None,
        admission: Optional[StrictAdmissionController] = None,
    ) -> None:
        if settings.stream_buffer_tokens < 1:
            raise ValueError("stream_buffer_tokens must be positive")
        if settings.stream_poll_interval_seconds <= 0:
            raise ValueError("stream_poll_interval_seconds must be positive")
        self.settings = settings
        self.backend = backend or DeterministicChatBackend()
        self.token_counter = token_counter or ConservativePromptTokenCounter()
        self.registry = registry or RequestRegistry(
            settings.total_kv_blocks - settings.safety_kv_blocks
        )
        self.admission = admission or _build_admission(settings)
        self._usage_lock = threading.Lock()
        self._completion_tokens: Dict[str, int] = {}
        self._requests_total = 0
        self._invalid_requests_total = 0

    def prepare(
        self,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> PreparedChatRequest:
        """完整验证通过后才创建 Registry 条目和 reservation。"""

        try:
            request = validate_chat_completion_request(
                body,
                headers,
                configured_model=self.settings.model_id,
                max_tokens_limit=min(4096, self.settings.context_limit),
            )
        except ContractViolation as exc:
            self._increment_invalid_requests()
            raise GatewayError(
                exc.code,
                exc.message,
                status_code=exc.status_code,
                request_id=_safe_request_id(headers),
                param=exc.param,
                error_type="invalid_request_error",
            ) from exc

        if request.tenant_id not in self.settings.tenant_limits:
            self._increment_invalid_requests()
            raise GatewayError(
                "tenant_not_authorized",
                "The tenant is not authorized for this deployment.",
                status_code=403,
                request_id=request.request_id,
                param="X-Tenant-ID",
                error_type="authentication_error",
            )

        claim = self.registry.register(request)
        if claim.status is not ClaimStatus.ACCEPTED:
            raise _claim_error(claim.status, claim.request_id)

        try:
            prompt_tokens = self.token_counter.count_prompt_tokens(request)
            self._advance_to_queued(request.request_id)
            decision = self.admission.submit(
                request.request_id,
                request.tenant_id,
                prompt_tokens,
                request.max_tokens,
            )
            if decision.status is not AdmissionStatus.ADMITTED:
                self.admission.release(request.request_id)
                self.registry.transition(
                    request.request_id,
                    RequestState.REJECTED,
                    "gateway:rejected:{0}".format(request.request_id),
                )
                raise _admission_error(decision.reason, request.request_id)

            self.registry.transition(
                request.request_id,
                RequestState.ADMITTED,
                "gateway:admitted:{0}".format(request.request_id),
            )
            if decision.plan is None:
                raise RuntimeError("admitted request must include a KV plan")
            self.registry.reserve(request.request_id, decision.plan.logical_blocks)
            self.registry.transition(
                request.request_id,
                RequestState.ROUTED,
                "gateway:routed:{0}".format(request.request_id),
            )
            self.registry.transition(
                request.request_id,
                RequestState.EXECUTING,
                "gateway:executing:{0}".format(request.request_id),
            )
        except GatewayError:
            raise
        except Exception as exc:
            self.admission.release(request.request_id)
            self._transition_terminal(request.request_id, RequestState.FAILED)
            raise GatewayError(
                "internal_error",
                "The request could not enter execution.",
                status_code=500,
                request_id=request.request_id,
            ) from exc

        with self._usage_lock:
            self._completion_tokens[request.request_id] = 0
            self._requests_total += 1
        return PreparedChatRequest(
            request=request,
            prompt_tokens=prompt_tokens,
            deadline_at_monotonic=time.monotonic() + request.deadline_ms / 1000,
        )

    async def complete(self, prepared: PreparedChatRequest) -> Dict[str, Any]:
        """收集生成增量并返回普通 Chat Completions 响应。"""

        pieces = []
        try:
            iterator = self.backend.generate(prepared.request).__aiter__()
            while True:
                try:
                    generated = await self._next_generated(iterator, prepared)
                except StopAsyncIteration:
                    break
                if self.registry.get(prepared.request.request_id).terminal:
                    break
                self._record_generated(
                    prepared.request.request_id,
                    generated.token_count,
                )
                pieces.append(generated.text)
            await self._raise_if_deadline_exceeded(prepared)
            snapshot = self.registry.get(prepared.request.request_id)
            if snapshot.state is RequestState.CANCELLED:
                raise GatewayError(
                    "request_cancelled",
                    "The request was cancelled.",
                    status_code=409,
                    request_id=prepared.request.request_id,
                )
            self._transition_terminal(
                prepared.request.request_id,
                RequestState.FINISHED,
            )
            return _completion_response(
                prepared,
                "".join(pieces),
                self.completion_tokens(prepared.request.request_id),
                self._finish_reason(prepared.request),
            )
        except GatewayError:
            raise
        except asyncio.CancelledError:
            await self.cancel(
                prepared.request.request_id,
                prepared.request.tenant_id,
                reason="disconnect",
            )
            raise
        except Exception as exc:
            self._transition_terminal(
                prepared.request.request_id,
                RequestState.FAILED,
            )
            raise GatewayError(
                "executor_failed",
                "The generation backend failed.",
                status_code=500,
                request_id=prepared.request.request_id,
            ) from exc
        finally:
            self.admission.release(prepared.request.request_id)

    async def stream(
        self,
        prepared: PreparedChatRequest,
        http_request: Request,
    ) -> AsyncIterator[str]:
        """输出标准 SSE data 事件，并在结束或错误后发送 `[DONE]`。"""

        request = prepared.request
        yield _sse_data(_role_chunk(request))
        queue: asyncio.Queue[_StreamQueueItem] = asyncio.Queue(
            maxsize=self.settings.stream_buffer_tokens
        )
        producer = asyncio.create_task(
            self._produce_stream(prepared, queue),
            name="cachepilot-stream-{0}".format(request.request_id),
        )
        try:
            while True:
                if await http_request.is_disconnected():
                    await self.cancel(
                        request.request_id,
                        request.tenant_id,
                        reason="disconnect",
                    )
                    return
                snapshot = self.registry.get(request.request_id)
                if snapshot.terminal and snapshot.state is not RequestState.FINISHED:
                    break
                if producer.done() and queue.empty():
                    await producer
                    break
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=min(
                            self._remaining_deadline(prepared),
                            self.settings.stream_poll_interval_seconds,
                        ),
                    )
                except asyncio.TimeoutError:
                    await self._raise_if_deadline_exceeded(prepared)
                    continue
                if item.done:
                    await producer
                    break
                generated = item.generated
                if generated is None:
                    raise RuntimeError("stream queue item has no generated text")
                if self.registry.get(request.request_id).terminal:
                    break
                self._record_generated(request.request_id, generated.token_count)
                yield _sse_data(_content_chunk(request, generated.text))

            await self._raise_if_deadline_exceeded(prepared)
            snapshot = self.registry.get(request.request_id)
            if snapshot.state is RequestState.CANCELLED:
                yield _sse_error(
                    GatewayError(
                        "request_cancelled",
                        "The request was cancelled.",
                        status_code=409,
                        request_id=request.request_id,
                    )
                )
            elif snapshot.state is RequestState.TIMED_OUT:
                yield _sse_error(
                    GatewayError(
                        "deadline_exceeded",
                        "The request deadline was exceeded.",
                        status_code=504,
                        request_id=request.request_id,
                    )
                )
            elif snapshot.terminal and snapshot.state is not RequestState.FINISHED:
                yield _sse_error(
                    GatewayError(
                        "executor_failed",
                        "The request terminated before completion.",
                        status_code=500,
                        request_id=request.request_id,
                    )
                )
            else:
                self._transition_terminal(request.request_id, RequestState.FINISHED)
                yield _sse_data(
                    _terminal_chunk(
                        prepared,
                        self.completion_tokens(request.request_id),
                        self._finish_reason(request),
                    )
                )
            yield "data: [DONE]\n\n"
        except GatewayError as exc:
            yield _sse_error(exc)
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            await self.cancel(
                request.request_id,
                request.tenant_id,
                reason="disconnect",
            )
            raise
        except Exception:
            self._transition_terminal(request.request_id, RequestState.FAILED)
            yield _sse_error(
                GatewayError(
                    "executor_failed",
                    "The generation backend failed.",
                    status_code=500,
                    request_id=request.request_id,
                )
            )
            yield "data: [DONE]\n\n"
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            self.admission.release(request.request_id)

    async def _produce_stream(
        self,
        prepared: PreparedChatRequest,
        queue: asyncio.Queue[_StreamQueueItem],
    ) -> None:
        """在固定容量队列中生成 token；队列满时自然施加背压。"""

        iterator = self.backend.generate(prepared.request).__aiter__()
        while True:
            try:
                generated = await self._next_generated(iterator, prepared)
            except StopAsyncIteration:
                await queue.put(_StreamQueueItem(done=True))
                return
            try:
                await asyncio.wait_for(
                    queue.put(_StreamQueueItem(generated=generated)),
                    timeout=self._remaining_deadline(prepared),
                )
            except asyncio.TimeoutError as exc:
                await self._timeout_request(prepared)
                raise GatewayError(
                    "deadline_exceeded",
                    "The request deadline was exceeded.",
                    status_code=504,
                    request_id=prepared.request.request_id,
                ) from exc

    async def _next_generated(self, iterator: Any, prepared: PreparedChatRequest):
        """以请求 deadline 限制每次执行器拉取，避免 backend 无限阻塞。"""

        await self._raise_if_deadline_exceeded(prepared)
        try:
            return await asyncio.wait_for(
                iterator.__anext__(),
                timeout=self._remaining_deadline(prepared),
            )
        except asyncio.TimeoutError as exc:
            await self._timeout_request(prepared)
            raise GatewayError(
                "deadline_exceeded",
                "The request deadline was exceeded.",
                status_code=504,
                request_id=prepared.request.request_id,
            ) from exc

    def _remaining_deadline(self, prepared: PreparedChatRequest) -> float:
        return max(0.001, prepared.deadline_at_monotonic - time.monotonic())

    def query(self, request_id: str, tenant_id: str) -> Dict[str, Any]:
        snapshot = self._tenant_snapshot(request_id, tenant_id)
        prompt_tokens = self.token_counter.count_prompt_tokens(snapshot.request)
        return {
            "request_id": request_id,
            "state": snapshot.state.value,
            "terminal": snapshot.terminal,
            "usage": _usage(prompt_tokens, self.completion_tokens(request_id)),
        }

    async def cancel(
        self,
        request_id: str,
        tenant_id: str,
        *,
        reason: str = "explicit",
    ) -> bool:
        snapshot = self._tenant_snapshot(request_id, tenant_id)
        if snapshot.state is RequestState.CANCELLED:
            return False
        if snapshot.terminal:
            raise GatewayError(
                "request_terminal",
                "The request is already terminal and cannot be cancelled.",
                status_code=409,
                request_id=request_id,
            )
        try:
            result = self.registry.transition(
                request_id,
                RequestState.CANCELLED,
                "gateway:cancelled:{0}".format(request_id),
            )
        except InvalidTransitionError:
            latest = self.registry.get(request_id)
            if latest.state is RequestState.CANCELLED:
                return False
            raise GatewayError(
                "request_terminal",
                "The request is already terminal and cannot be cancelled.",
                status_code=409,
                request_id=request_id,
            )
        await self.backend.cancel(request_id, reason=reason)
        self.admission.release(request_id)
        return result.applied

    async def ready(self) -> bool:
        return await self.backend.is_ready()

    def completion_tokens(self, request_id: str) -> int:
        with self._usage_lock:
            return self._completion_tokens.get(request_id, 0)

    def metrics(self) -> str:
        snapshot = self.admission.snapshot()
        with self._usage_lock:
            requests_total = self._requests_total
            invalid_total = self._invalid_requests_total
        return (
            "# TYPE cachepilot_gateway_up gauge\n"
            "cachepilot_gateway_up 1\n"
            "# TYPE cachepilot_requests_total counter\n"
            "cachepilot_requests_total {0}\n"
            "# TYPE cachepilot_invalid_requests_total counter\n"
            "cachepilot_invalid_requests_total {1}\n"
            "# TYPE cachepilot_active_sequences gauge\n"
            "cachepilot_active_sequences {2}\n"
            "# TYPE cachepilot_reserved_kv_blocks gauge\n"
            "cachepilot_reserved_kv_blocks {3}\n"
        ).format(
            requests_total,
            invalid_total,
            snapshot.active_sequences,
            snapshot.reserved_blocks,
        )

    def _advance_to_queued(self, request_id: str) -> None:
        self.registry.transition(
            request_id,
            RequestState.TOKENIZED,
            "gateway:tokenized:{0}".format(request_id),
        )
        self.registry.transition(
            request_id,
            RequestState.QUEUED,
            "gateway:queued:{0}".format(request_id),
        )

    def _record_generated(self, request_id: str, token_count: int) -> None:
        if type(token_count) is not int or token_count < 1:
            raise ValueError("backend token_count must be a positive integer")
        with self._usage_lock:
            offset = self._completion_tokens[request_id]
        for index in range(token_count):
            self.registry.record_token_emission(
                request_id,
                "gateway:token:{0}:{1}".format(request_id, offset + index),
            )
        with self._usage_lock:
            self._completion_tokens[request_id] += token_count

    def _finish_reason(self, request: ValidatedChatRequest) -> str:
        if self.completion_tokens(request.request_id) >= request.max_tokens:
            return "length"
        return "stop"

    async def _raise_if_deadline_exceeded(
        self,
        prepared: PreparedChatRequest,
    ) -> None:
        snapshot = self.registry.get(prepared.request.request_id)
        if snapshot.state is RequestState.TIMED_OUT:
            raise GatewayError(
                "deadline_exceeded",
                "The request deadline was exceeded.",
                status_code=504,
                request_id=prepared.request.request_id,
            )
        if time.monotonic() < prepared.deadline_at_monotonic:
            return
        if snapshot.state is RequestState.CANCELLED:
            raise GatewayError(
                "request_cancelled",
                "The request was cancelled.",
                status_code=409,
                request_id=prepared.request.request_id,
            )
        if snapshot.terminal and snapshot.state is not RequestState.TIMED_OUT:
            return
        await self._timeout_request(prepared)
        raise GatewayError(
            "deadline_exceeded",
            "The request deadline was exceeded.",
            status_code=504,
            request_id=prepared.request.request_id,
        )

    async def _timeout_request(self, prepared: PreparedChatRequest) -> bool:
        """Atomically claim timeout, notify backend, and release admission."""

        request_id = prepared.request.request_id
        applied = self._transition_terminal(request_id, RequestState.TIMED_OUT)
        if not applied:
            return False
        await self.backend.cancel(request_id, reason="timeout")
        self.admission.release(request_id)
        return True

    def _tenant_snapshot(self, request_id: str, tenant_id: str) -> RequestSnapshot:
        _validate_lookup_identity(request_id, tenant_id)
        if tenant_id not in self.settings.tenant_limits:
            raise _not_found(request_id)
        snapshot = self.registry.get_for_tenant(request_id, tenant_id)
        if snapshot is None:
            raise _not_found(request_id)
        return snapshot

    def _transition_terminal(self, request_id: str, state: RequestState) -> bool:
        try:
            result = self.registry.transition(
                request_id,
                state,
                "gateway:{0}:{1}".format(state.value.lower(), request_id),
            )
            return result.applied
        except InvalidTransitionError:
            return False

    def _increment_invalid_requests(self) -> None:
        with self._usage_lock:
            self._invalid_requests_total += 1


def create_app(
    settings: Optional[GatewaySettings] = None,
    *,
    backend: Optional[ChatBackend] = None,
    token_counter: Optional[PromptTokenCounter] = None,
) -> FastAPI:
    """创建可测试、可注入执行器的 FastAPI 应用。"""

    runtime = GatewayRuntime(
        settings or default_settings(),
        backend=backend,
        token_counter=token_counter,
    )
    app = FastAPI(title="CachePilot API", version="0.2.0")
    app.state.gateway_runtime = runtime

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        request: Request,
        exc: GatewayError,
    ) -> JSONResponse:
        del request
        return _error_response(exc)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            runtime._increment_invalid_requests()
            raise GatewayError(
                "invalid_body",
                "Request body must be valid JSON.",
                status_code=400,
                request_id=_safe_request_id(request.headers),
                param="body",
                error_type="invalid_request_error",
            ) from exc
        prepared = runtime.prepare(body, request.headers)
        headers = {"X-Request-ID": prepared.request.request_id}
        if prepared.request.stream:
            headers.update(
                {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
            return StreamingResponse(
                runtime.stream(prepared, request),
                media_type="text/event-stream",
                headers=headers,
            )
        return JSONResponse(await runtime.complete(prepared), headers=headers)

    @app.get("/v1/requests/{request_id}")
    async def get_request(request_id: str, request: Request):
        return runtime.query(request_id, _tenant_header(request.headers))

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_request(request_id: str, request: Request):
        applied = await runtime.cancel(
            request_id,
            _tenant_header(request.headers),
        )
        return JSONResponse(
            {"request_id": request_id, "state": "CANCELLED"},
            status_code=202 if applied else 200,
            headers={"X-Request-ID": request_id},
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "cachepilot-gateway"}

    @app.get("/readyz")
    async def readyz():
        if await runtime.ready():
            return {"status": "ready", "service": "cachepilot-gateway"}
        return JSONResponse(
            {"status": "not_ready", "service": "cachepilot-gateway"},
            status_code=503,
        )

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            runtime.metrics(),
            media_type="text/plain; version=0.0.4",
        )

    return app


def default_settings() -> GatewaySettings:
    repository_root = Path(__file__).resolve().parents[2]
    baseline = load_model_baseline(repository_root / "config" / "model.json")
    return GatewaySettings(
        model_id=baseline.model_id,
        context_limit=baseline.service_context_limit,
    )


def _build_admission(settings: GatewaySettings) -> StrictAdmissionController:
    baseline_path = Path(__file__).resolve().parents[2] / "config" / "model.json"
    baseline = load_model_baseline(baseline_path)
    planner = KVPlanner(
        KVModelSpec.from_model_baseline(
            baseline,
            dtype="bfloat16",
            block_size=settings.block_size,
            context_limit=settings.context_limit,
        )
    )
    return StrictAdmissionController(
        planner,
        StrictAdmissionConfig(
            total_blocks=settings.total_kv_blocks,
            safety_blocks=settings.safety_kv_blocks,
            max_active_sequences=settings.max_active_sequences,
            max_queued_requests=settings.max_queued_requests,
            tenant_limits=settings.tenant_limits,
        ),
    )


def _claim_error(status: ClaimStatus, request_id: str) -> GatewayError:
    messages = {
        ClaimStatus.REQUEST_ID_CONFLICT: "The request ID is already in use.",
        ClaimStatus.IDEMPOTENCY_IN_PROGRESS: "The idempotent request is in progress.",
        ClaimStatus.IDEMPOTENCY_REPLAY_UNAVAILABLE: (
            "The completed idempotent response is not available for replay."
        ),
        ClaimStatus.IDEMPOTENCY_KEY_CONFLICT: (
            "The idempotency key was used with different request parameters."
        ),
    }
    return GatewayError(
        status.value,
        messages[status],
        status_code=409,
        request_id=request_id,
    )


def _admission_error(reason: AdmissionReason, request_id: str) -> GatewayError:
    quota_reasons = {
        AdmissionReason.TENANT_NOT_CONFIGURED,
        AdmissionReason.TENANT_REQUEST_EXCEEDS_TOKEN_QUOTA,
        AdmissionReason.TENANT_ACTIVE_TOKENS,
        AdmissionReason.TENANT_CONCURRENCY,
        AdmissionReason.TENANT_QUEUE_FULL,
    }
    queue_reasons = {
        AdmissionReason.QUEUE_FULL,
        AdmissionReason.MAX_ACTIVE_SEQUENCES,
        AdmissionReason.KV_CAPACITY,
    }
    if reason in quota_reasons:
        code = "tenant_quota_exceeded"
    elif reason in queue_reasons:
        code = "queue_full"
    else:
        code = "admission_rejected"
    return GatewayError(
        code,
        "The request was rejected by admission control: {0}.".format(reason.value),
        status_code=429,
        request_id=request_id,
        retry_after_seconds=1,
    )


def _error_response(error: GatewayError) -> JSONResponse:
    headers = {}
    if error.request_id is not None:
        headers["X-Request-ID"] = error.request_id
    if error.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_after_seconds)
    return JSONResponse(
        error.payload(),
        status_code=error.status_code,
        headers=headers,
    )


def _validate_lookup_identity(request_id: str, tenant_id: str) -> None:
    if not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise GatewayError(
            "invalid_request_id",
            "request_id has an invalid format.",
            status_code=400,
            param="request_id",
            error_type="invalid_request_error",
        )
    if not _TENANT_PATTERN.fullmatch(tenant_id):
        raise GatewayError(
            "invalid_tenant",
            "X-Tenant-ID has an invalid format.",
            status_code=400,
            request_id=request_id,
            param="X-Tenant-ID",
            error_type="invalid_request_error",
        )


def _tenant_header(headers: Mapping[str, str]) -> str:
    tenant_id = headers.get("x-tenant-id") or headers.get("X-Tenant-ID")
    if tenant_id is None or tenant_id == "":
        raise GatewayError(
            "tenant_required",
            "X-Tenant-ID is required.",
            status_code=400,
            param="X-Tenant-ID",
            error_type="invalid_request_error",
        )
    return tenant_id


def _safe_request_id(headers: Mapping[str, str]) -> Optional[str]:
    value = headers.get("x-request-id") or headers.get("X-Request-ID")
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return None


def _not_found(request_id: str) -> GatewayError:
    return GatewayError(
        "request_not_found",
        "The request was not found.",
        status_code=404,
        request_id=request_id,
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _base_chunk(request: ValidatedChatRequest) -> Dict[str, Any]:
    return {
        "id": request.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
    }


def _role_chunk(request: ValidatedChatRequest) -> Dict[str, Any]:
    chunk = _base_chunk(request)
    chunk["choices"] = [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
    ]
    return chunk


def _content_chunk(
    request: ValidatedChatRequest,
    content: str,
) -> Dict[str, Any]:
    chunk = _base_chunk(request)
    chunk["choices"] = [
        {"index": 0, "delta": {"content": content}, "finish_reason": None}
    ]
    return chunk


def _terminal_chunk(
    prepared: PreparedChatRequest,
    completion_tokens: int,
    finish_reason: str,
) -> Dict[str, Any]:
    chunk = _base_chunk(prepared.request)
    chunk["choices"] = [
        {"index": 0, "delta": {}, "finish_reason": finish_reason}
    ]
    chunk["usage"] = _usage(prepared.prompt_tokens, completion_tokens)
    return chunk


def _completion_response(
    prepared: PreparedChatRequest,
    content: str,
    completion_tokens: int,
    finish_reason: str,
) -> Dict[str, Any]:
    return {
        "id": prepared.request.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": prepared.request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(prepared.prompt_tokens, completion_tokens),
    }


def _sse_data(payload: Mapping[str, Any]) -> str:
    return "data: {0}\n\n".format(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _sse_error(error: GatewayError) -> str:
    return "event: error\ndata: {0}\n\n".format(
        json.dumps(error.payload(), ensure_ascii=False, separators=(",", ":"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


app = create_app()


if __name__ == "__main__":
    main()
