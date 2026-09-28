"""单请求 PyTorch/Transformers 生成执行器。

本执行器只实现一个请求一次生成的正确性路径。它把 chat template、padding、
attention mask、position IDs、EOS/停止 token 和输入宽度输出切片集中在一个边界
内；教学型 batching 必须在这条路径拥有独立测试后再增加。
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping, Optional, Sequence

from cachepilot.executor_capabilities import TORCH_EXECUTOR_CAPABILITIES
from cachepilot.cache.prefix_index import PrefixScopeKey
from cachepilot.gateway.backends import GeneratedText
from cachepilot.gateway.contracts import ValidatedChatRequest


class TorchExecutorError(RuntimeError):
    """TorchExecutor 配置、依赖或推理失败。"""


class TorchExecutorUnavailableError(TorchExecutorError):
    """当前环境没有可用的 PyTorch/Transformers 运行时。"""


class _CancellationStoppingCriteria:
    """让 Transformers 在请求取消后尽快停止生成。"""

    def __init__(self, is_cancelled):
        self._is_cancelled = is_cancelled

    def __call__(self, input_ids, scores, **kwargs):
        del input_ids, scores, kwargs
        return self._is_cancelled()


class _StoppingCriteriaList(list):
    """兼容 Transformers 期望的可调用 stopping criteria 容器。"""

    def __call__(self, input_ids, scores, **kwargs):
        return any(
            criterion(input_ids, scores, **kwargs) for criterion in self
        )


@dataclass(frozen=True)
class TorchExecutorConfig:
    """单请求 Torch 执行器配置。"""

    model_id: str
    tokenizer_id: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    context_limit: int = 8192
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.tokenizer_id is not None and not self.tokenizer_id:
            raise ValueError("tokenizer_id must be non-empty when provided")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if type(self.context_limit) is not int or self.context_limit < 1:
            raise ValueError("context_limit must be a positive integer")


class TorchExecutor:
    """使用 Transformers ``generate`` 的单请求聊天后端。

    ``model`` 和 ``tokenizer`` 可注入轻量 fake 对象，便于在没有 GPU 或模型
    下载权限的环境测试输入张量和停止条件。未注入时才从 Transformers 加载模型。
    """

    capabilities = TORCH_EXECUTOR_CAPABILITIES
    supports_batching = False
    batching_owner = "none"
    implements_batching = False

    def __init__(
        self,
        config: TorchExecutorConfig,
        *,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        if not isinstance(config, TorchExecutorConfig):
            raise TypeError("config must be a TorchExecutorConfig")
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be provided together")
        self.config = config
        self._cancelled: dict[str, str] = {}
        self._cancel_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._ready = False

        if model is None:
            model, tokenizer = self._load_components(config)
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._resolve_device(config.device)
        self._move_model_to_device()
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._ready = True

    async def is_ready(self) -> bool:
        """返回模型是否已加载且可接收请求。"""

        return self._ready

    def count_prompt_tokens(self, request: ValidatedChatRequest) -> int:
        """使用同一 tokenizer/chat template 计算 Gateway 准入 token 数。"""

        return len(self.prompt_token_ids(request))

    def prefix_key(self, request: ValidatedChatRequest) -> Optional[PrefixScopeKey]:
        """返回带固定版本作用域的逻辑 token key，不声明物理 KV 命中。"""

        if not self.config.model_revision or not self.config.tokenizer_revision:
            return None
        return PrefixScopeKey.create(
            tenant_id=request.tenant_id,
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            tokenizer_revision=self.config.tokenizer_revision,
            quantization_config="none",
            tokenized_prefix=self.prompt_token_ids(request),
        )

    def prompt_token_ids(self, request: ValidatedChatRequest) -> tuple[int, ...]:
        """在同一 tokenizer 锁内取得准入和逻辑 prefix 共用的 token 序列。"""

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        with self._tokenizer_lock:
            messages = [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ]
            apply_template = getattr(self.tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                token_ids = apply_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if not isinstance(token_ids, str):
                    return self._flat_token_ids(token_ids)

            prompt = self._chat_prompt(request)
            encoded = self.tokenizer(prompt, add_special_tokens=False)
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise TorchExecutorError(
                    "tokenizer output is missing input_ids for prompt counting"
                )
            return self._flat_token_ids(encoded)

    @staticmethod
    def _flat_token_ids(token_ids: Any) -> tuple[int, ...]:
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        shape = getattr(token_ids, "shape", ())
        if len(shape) >= 2 or (
            isinstance(token_ids, (list, tuple)) and token_ids
            and isinstance(token_ids[0], (list, tuple))
        ):
            token_ids = token_ids[0]
        return tuple(int(token) for token in token_ids)

    @property
    def cancel_reasons(self) -> dict[str, str]:
        """返回取消原因快照，供 Gateway 观测和验收使用。"""

        with self._cancel_lock:
            return dict(self._cancelled)

    async def cancel(self, request_id: str, reason: str = "explicit") -> None:
        """标记请求取消；生成线程不会再向客户端交付 token。"""

        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not reason:
            raise ValueError("reason must be non-empty")
        with self._cancel_lock:
            self._cancelled[request_id] = reason

    async def generate(
        self,
        request: ValidatedChatRequest,
    ) -> AsyncIterator[GeneratedText]:
        """异步生成单请求文本增量。"""

        if not isinstance(request, ValidatedChatRequest):
            raise TypeError("request must be a ValidatedChatRequest")
        if self._is_cancelled(request.request_id):
            return
        try:
            token_ids = await asyncio.to_thread(self._generate_token_ids, request)
        except TorchExecutorError:
            raise
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise TorchExecutorError("Torch generation ran out of memory") from exc
            raise TorchExecutorError("Torch generation failed") from exc

        eos_ids = self._eos_token_ids()
        for token_id in token_ids:
            if self._is_cancelled(request.request_id):
                return
            token_id = self._scalar_int(token_id)
            if token_id in eos_ids:
                return
            text = self._decode_token(token_id)
            if text:
                yield GeneratedText(text, token_count=1)

    def _generate_token_ids(
        self,
        request: ValidatedChatRequest,
    ) -> Sequence[Any]:
        with self._generation_lock:
            if self._is_cancelled(request.request_id):
                return []
            return self._generate_token_ids_locked(request)

    def _generate_token_ids_locked(
        self,
        request: ValidatedChatRequest,
    ) -> Sequence[Any]:
        encoded = self._encode(request)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        position_ids = self._position_ids(attention_mask)
        model_inputs = {
            "input_ids": self._to_device(input_ids),
            "attention_mask": self._to_device(attention_mask),
            "position_ids": self._to_device(position_ids),
            "max_new_tokens": request.max_tokens,
            "do_sample": False,
            "return_dict_in_generate": False,
            "use_cache": True,
        }
        pad_token_id = self._pad_token_id()
        eos_ids = self._eos_token_ids()
        if pad_token_id is not None:
            model_inputs["pad_token_id"] = pad_token_id
        if eos_ids:
            model_inputs["eos_token_id"] = (
                eos_ids[0] if len(eos_ids) == 1 else list(eos_ids)
            )
        model_inputs["stopping_criteria"] = _StoppingCriteriaList([
            _CancellationStoppingCriteria(
                lambda: self._is_cancelled(request.request_id)
            )
        ])
        with self._inference_context():
            generated = self.model.generate(**model_inputs)
        row = generated[0]
        input_width = self._shape_width(input_ids)
        return row[input_width:]

    def _encode(self, request: ValidatedChatRequest) -> Mapping[str, Any]:
        with self._tokenizer_lock:
            prompt = self._chat_prompt(request)
            try:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max(1, self.config.context_limit - request.max_tokens),
                    return_attention_mask=True,
                )
            except TypeError:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                )
        if not isinstance(encoded, Mapping):
            raise TorchExecutorError("tokenizer must return a mapping")
        if "input_ids" not in encoded:
            raise TorchExecutorError("tokenizer output is missing input_ids")
        if "attention_mask" not in encoded:
            encoded = dict(encoded)
            encoded["attention_mask"] = self._ones_like(encoded["input_ids"])
        return encoded

    def _chat_prompt(self, request: ValidatedChatRequest) -> Any:
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(
            "{0}: {1}".format(message["role"], message["content"])
            for message in messages
        ) + "\nassistant:"

    @staticmethod
    def _position_ids(attention_mask: Any) -> Any:
        """按非 padding token 累计位置，兼容左/右 padding。"""

        position_ids = attention_mask.long().cumsum(-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _inference_context(self):
        try:
            torch = self._torch_module()
        except TorchExecutorUnavailableError:
            return nullcontext()
        return torch.inference_mode()

    def _move_model_to_device(self) -> None:
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def _to_device(self, value: Any) -> Any:
        return value.to(self.device) if hasattr(value, "to") else value

    def _resolve_device(self, configured: str) -> str:
        if configured != "auto":
            return configured
        try:
            torch = self._torch_module()
        except TorchExecutorUnavailableError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _pad_token_id(self) -> Optional[int]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        return None

    def _eos_token_ids(self) -> tuple[int, ...]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            value = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if value is None:
            return ()
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(int(item) for item in value)
        return (int(value),)

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _scalar_int(value: Any) -> int:
        item = value.item() if hasattr(value, "item") else value
        return int(item)

    @staticmethod
    def _shape_width(value: Any) -> int:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 2:
            return len(value[0])
        return int(shape[-1])

    @staticmethod
    def _ones_like(value: Any) -> Any:
        try:
            torch = TorchExecutor._torch_module()
        except TorchExecutorUnavailableError:
            if isinstance(value, Sequence):
                return [[1 for _ in row] for row in value]
            raise
        return torch.ones_like(value)

    def _is_cancelled(self, request_id: str) -> bool:
        with self._cancel_lock:
            return request_id in self._cancelled

    @staticmethod
    def _torch_module() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise TorchExecutorUnavailableError(
                "TorchExecutor requires the torch package"
            ) from exc
        return torch

    @classmethod
    def _load_components(cls, config: TorchExecutorConfig) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise TorchExecutorUnavailableError(
                "TorchExecutor requires transformers and torch"
            ) from exc
        torch = cls._torch_module()
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = cls._resolve_dtype(torch, config.dtype, device)
        tokenizer_kwargs = {"trust_remote_code": config.trust_remote_code}
        model_kwargs = {"trust_remote_code": config.trust_remote_code}
        if config.tokenizer_revision is not None:
            tokenizer_kwargs["revision"] = config.tokenizer_revision
        if config.model_revision is not None:
            model_kwargs["revision"] = config.model_revision
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id or config.model_id,
            **tokenizer_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **model_kwargs,
        )
        return model, tokenizer

    @staticmethod
    def _resolve_dtype(torch: Any, dtype: str, device: str) -> Optional[Any]:
        if dtype == "auto":
            return None if device == "cpu" else torch.bfloat16
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
