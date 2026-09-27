"""vLLM 异步引擎适配器。

本模块只转发 prompt、SamplingParams、流式输出、abort 和 usage。continuous
batching、物理 KV、调度与 kernel 均由 vLLM 拥有，CachePilot 不建立第二层 batch。
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from importlib import metadata
from typing import Any, AsyncIterator, Callable, Optional

from cachepilot.gateway.backends import GeneratedText
from cachepilot.gateway.contracts import ValidatedChatRequest


class VllmExecutorError(RuntimeError):
    """vLLM 配置、协议或推理错误。"""


class VllmExecutorUnavailableError(VllmExecutorError):
    """当前环境没有锁定版本的 vLLM 运行时。"""


@dataclass(frozen=True)
class VllmExecutorConfig:
    """单 GPU vLLM 引擎配置。"""

    model_id: str
    tokenizer_id: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: str = "auto"
    context_limit: int = 8192
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    expected_vllm_version: str = "0.24.0"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.tokenizer_id is not None and not self.tokenizer_id:
            raise ValueError("tokenizer_id must be non-empty when provided")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        if type(self.context_limit) is not int or self.context_limit < 1:
            raise ValueError("context_limit must be a positive integer")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not self.expected_vllm_version:
            raise ValueError("expected_vllm_version must be non-empty")


@dataclass(frozen=True)
class VllmUsage:
    """最近一次 vLLM 输出对应的可核对 token usage。"""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Optional[str]
    finished: bool


class VllmExecutor:
    """把 Gateway ``ChatBackend`` 边界适配到 vLLM AsyncLLMEngine。"""

    supports_batching = True
    batching_owner = "vllm"
    implements_batching = False

    def __init__(
        self,
        config: VllmExecutorConfig,
        *,
        engine: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        sampling_params_factory: Optional[Callable[[int], Any]] = None,
    ) -> None:
        if not isinstance(config, VllmExecutorConfig):
            raise TypeError("config must be a VllmExecutorConfig")
        if (engine is None) != (tokenizer is None):
            raise ValueError("engine and tokenizer must be provided together")
        self.config = config
        self._lock = threading.Lock()
        self._active_request_ids: set[str] = set()
        self._cancel_reasons: dict[str, str] = {}
        self._usage: dict[str, VllmUsage] = {}

        if engine is None:
            engine, tokenizer, default_factory = self._load_components(config)
            if sampling_params_factory is None:
                sampling_params_factory = default_factory
        if sampling_params_factory is None:
            raise ValueError(
                "sampling_params_factory is required with an injected engine"
            )
        self.engine = engine
        self.tokenizer = tokenizer
        self._sampling_params_factory = sampling_params_factory

    async def is_ready(self) -> bool:
        """调用 vLLM health check；旧引擎无该入口时以已构造为 ready。"""

        check_health = getattr(self.engine, "check_health", None)
        if not callable(check_health):
            return True
        try:
            result = check_health()
            if inspect.isawaitable(result):
                await result
        except Exception:
            return False
        return True

    @property
    def cancel_reasons(self) -> dict[str, str]:
        with self._lock:
            return dict(self._cancel_reasons)

    def usage(self, request_id: str) -> Optional[VllmUsage]:
        """返回请求最近 usage 快照；尚无引擎输出时返回 ``None``。"""

        with self._lock:
            return self._usage.get(request_id)

    def count_prompt_tokens(self, request: ValidatedChatRequest) -> int:
        """使用同一 tokenizer/chat template 计算 Gateway 准入 token 数。"""

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        messages = self._messages(request)
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    async def cancel(self, request_id: str, reason: str = "explicit") -> None:
        """把活跃请求取消原样转发为 vLLM ``abort(request_id)``。"""

        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not reason:
            raise ValueError("reason must be non-empty")
        with self._lock:
            if request_id in self._cancel_reasons:
                return
            self._cancel_reasons[request_id] = reason
            active = request_id in self._active_request_ids
        if not active:
            return
        try:
            result = self.engine.abort(request_id)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            raise VllmExecutorError(
                "vLLM abort failed for request {0}".format(request_id)
            ) from exc

    async def generate(
        self,
        request: ValidatedChatRequest,
    ) -> AsyncIterator[GeneratedText]:
        """消费 vLLM delta 流并逐块返回可靠 token 数。"""

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        request_id = request.request_id
        with self._lock:
            if request_id in self._active_request_ids:
                raise VllmExecutorError(
                    "request is already active: {0}".format(request_id)
                )
            if request_id in self._cancel_reasons:
                return
            self._active_request_ids.add(request_id)

        prompt = self._chat_prompt(request)
        sampling_params = self._sampling_params_factory(request.max_tokens)
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = None
        finished = False
        try:
            stream = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            async for output in stream:
                if self._is_cancelled(request_id):
                    return
                self._validate_output_request_id(output, request_id)
                prompt_token_ids = getattr(output, "prompt_token_ids", None)
                if prompt_token_ids is not None:
                    prompt_tokens = len(prompt_token_ids)
                candidates = getattr(output, "outputs", None)
                if not candidates:
                    raise VllmExecutorError(
                        "vLLM output has no candidates for request {0}".format(
                            request_id
                        )
                    )
                candidate = candidates[0]
                token_ids = tuple(getattr(candidate, "token_ids", ()) or ())
                completion_tokens += len(token_ids)
                candidate_finish_reason = getattr(candidate, "finish_reason", None)
                if candidate_finish_reason is not None:
                    finish_reason = str(candidate_finish_reason)
                finished = bool(getattr(output, "finished", False))
                self._store_usage(
                    request_id,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    finished,
                )
                if self._is_cancelled(request_id):
                    return
                text = str(getattr(candidate, "text", ""))
                if token_ids:
                    yield GeneratedText(text=text, token_count=len(token_ids))
            finished = True
            self._store_usage(
                request_id,
                prompt_tokens,
                completion_tokens,
                finish_reason,
                finished,
            )
        except VllmExecutorError:
            raise
        except Exception as exc:
            raise VllmExecutorError(
                "vLLM generation failed for request {0}".format(request_id)
            ) from exc
        finally:
            with self._lock:
                self._active_request_ids.discard(request_id)

    def _chat_prompt(self, request: ValidatedChatRequest) -> str:
        messages = self._messages(request)
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            raise VllmExecutorError("tokenizer does not provide apply_chat_template")
        prompt = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str) or not prompt:
            raise VllmExecutorError("chat template returned an empty prompt")
        return prompt

    @staticmethod
    def _messages(request: ValidatedChatRequest) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

    def _store_usage(
        self,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: Optional[str],
        finished: bool,
    ) -> None:
        usage = VllmUsage(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            finished=finished,
        )
        with self._lock:
            self._usage[request_id] = usage

    def _is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancel_reasons

    @staticmethod
    def _validate_output_request_id(output: Any, expected: str) -> None:
        actual = getattr(output, "request_id", None)
        if actual != expected:
            raise VllmExecutorError(
                "vLLM request ID mismatch: expected {0}, got {1}".format(
                    expected,
                    actual,
                )
            )

    @classmethod
    def _load_components(
        cls,
        config: VllmExecutorConfig,
    ) -> tuple[Any, Any, Callable[[int], Any]]:
        cls._require_locked_version(config.expected_vllm_version)
        try:
            from transformers import AutoTokenizer
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
            from vllm.sampling_params import RequestOutputKind
        except ImportError as exc:
            raise VllmExecutorUnavailableError(
                "Could not import locked vLLM runtime: {0}".format(exc)
            ) from exc

        engine_args = AsyncEngineArgs(
            model=config.model_id,
            tokenizer=config.tokenizer_id or config.model_id,
            revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            dtype=config.dtype,
            max_model_len=config.context_limit,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id or config.model_id,
            revision=config.tokenizer_revision,
            trust_remote_code=config.trust_remote_code,
        )

        def sampling_params_factory(max_tokens: int) -> Any:
            return SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )

        return engine, tokenizer, sampling_params_factory

    @staticmethod
    def _require_locked_version(expected: str) -> None:
        try:
            installed = metadata.version("vllm")
        except metadata.PackageNotFoundError as exc:
            raise VllmExecutorUnavailableError(
                "vllm is not installed; expected version {0}".format(expected)
            ) from exc
        if installed != expected:
            raise VllmExecutorUnavailableError(
                "vllm version mismatch: expected {0}, found {1}".format(
                    expected,
                    installed,
                )
            )
