"""不依赖执行器的逻辑 KV block 与理论字节规划。

Planner 使用模型架构推导理论 KV 数据量，并把 token 需求向上取整为完整
block。结果用于准入预算，不表示 vLLM 等执行器已经分配了对应显存，
也不包含执行器元数据、对齐、预分配或显存碎片开销。
"""

from __future__ import annotations

from dataclasses import dataclass

from cachepilot.config.baseline import ModelBaseline
from cachepilot.utils import CommonUtils


class KVPlannerError(ValueError):
    """KV 规划输入无效时抛出。"""


class ContextLimitExceededError(KVPlannerError):
    """请求的 KV token 数超过服务上下文限制时抛出。"""


class UsableKVCapacityRequiredError(KVPlannerError):
    """没有明确 KV 字节预算却尝试推导容量时抛出。"""


_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
}


@dataclass(frozen=True)
class KVModelSpec:
    """进行 KV 理论计算所需的最小模型规格。

    Attributes:
        num_hidden_layers: Transformer 隐藏层数量。
        num_key_value_heads: 每层参与 KV Cache 的 KV head 数；GQA 模型不能
            错用 attention head 数。
        head_dim: 每个 KV head 的维度。
        dtype: KV 元素类型，初始化时会规范为完整名称。
        block_size: 一个逻辑 KV block 可容纳的 token 数。
        context_limit: 服务实际允许的最大上下文 token 数。
    """

    num_hidden_layers: int  # Transformer 隐藏层数量。
    num_key_value_heads: int  # 每层参与 KV Cache 的 KV head 数。
    head_dim: int  # 每个 KV head 的维度。
    dtype: str  # KV Cache 标量数据类型。
    block_size: int  # 一个逻辑 KV block 容纳的 token 数。
    context_limit: int  # 服务允许的最大上下文 token 数。

    def __post_init__(self) -> None:
        """校验正整数结构字段并规范化 dtype 别名。"""

        for field_name in (
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "block_size",
            "context_limit",
        ):
            value = getattr(self, field_name)
            CommonUtils.require_positive_int(
                value, field_name, KVPlannerError
            )

        if type(self.dtype) is not str or not self.dtype.strip():
            raise KVPlannerError("dtype must be a non-empty string")
        normalized_dtype = _normalize_dtype(self.dtype)
        object.__setattr__(self, "dtype", normalized_dtype)

    @classmethod
    def from_model_baseline(
        cls,
        baseline: ModelBaseline,  # 已校验并固定版本的模型基线。
        *,
        dtype: str,  # 实际用于 KV Cache 的数据类型。
        block_size: int,  # 执行器或模拟器采用的 token block 大小。
        context_limit: int | None = None,  # 可选的服务上下文上限。
    ) -> KVModelSpec:
        """从已校验模型基线创建规划配置。

        Args:
            baseline: ``config/model.json`` 加载得到的固定模型基线。
            dtype: 实际用于 KV Cache 的数据类型，例如 ``bfloat16``。
            block_size: 执行/模拟配置采用的 token block 大小。
            context_limit: 可选服务上下文上限；缺省使用基线中的
                保守服务值。

        Returns:
            可直接传给 ``KVPlanner`` 的不可变规格。

        Raises:
            KVPlannerError: 上下文上限无效或超过模型理论最大上下文。
        """

        if not isinstance(baseline, ModelBaseline):
            raise TypeError("baseline must be a ModelBaseline")
        selected_context_limit = (
            baseline.service_context_limit
            if context_limit is None
            else context_limit
        )
        CommonUtils.require_positive_int(
            selected_context_limit,
            "context_limit",
            KVPlannerError,
        )
        if selected_context_limit > baseline.model_max_context_tokens:
            raise KVPlannerError(
                "context_limit cannot exceed the model maximum context"
            )
        return cls(
            num_hidden_layers=baseline.num_hidden_layers,
            num_key_value_heads=baseline.num_key_value_heads,
            head_dim=baseline.head_dim,
            dtype=dtype,
            block_size=block_size,
            context_limit=selected_context_limit,
        )


@dataclass(frozen=True)
class KVRequestPlan:
    """单个请求的逻辑 reservation 计划。

    ``theoretical_bytes`` 按完整 block 计算，所以会包含最后一个 block 中
    ``padding_tokens`` 对应的空间。
    """

    prompt_tokens: int  # 输入中需要保留 KV 的 token 数。
    expected_output_tokens: int  # 策略预计生成并保留的输出 token 数。
    total_tokens: int  # 输入与预计输出 token 总数。
    logical_blocks: int  # 按完整 block 向上取整后的逻辑块数。
    allocated_tokens: int  # 所有逻辑块合计可容纳的 token 数。
    padding_tokens: int  # 最后一个 block 中未使用的 token 槽位数。
    theoretical_bytes: int  # 这些完整 block 的理论 KV 字节数。


@dataclass(frozen=True)
class KVCapacityPlan:
    """把明确的 KV 专用字节预算换算成完整 block 后的容量结果。"""

    usable_kv_bytes: int  # 明确可供 KV 使用的字节预算。
    logical_blocks: int  # 预算可容纳的完整逻辑 block 数。
    token_capacity: int  # 完整逻辑 block 对应的 token 容量。
    allocated_bytes: int  # 完整逻辑 block 实际计入的字节数。
    unused_bytes: int  # 无法组成完整 block 的尾部字节数。


class KVPlanner:
    """计算逻辑 block 和理论 KV 数据量，不声明物理分配结果。

    Args:
        spec: 已校验的模型 KV 规格。
    """

    def __init__(
        self,
        spec: KVModelSpec,  # 已校验的模型 KV 规格。
    ) -> None:
        if not isinstance(spec, KVModelSpec):
            raise TypeError("spec must be a KVModelSpec")
        self._spec = spec

    @property
    def spec(self) -> KVModelSpec:
        """返回 Planner 使用的不可变模型规格。"""

        return self._spec

    @property
    def dtype_bytes(self) -> int:
        """返回单个 KV 标量占用的理论字节数。"""

        return _DTYPE_BYTES[self._spec.dtype]

    @property
    def bytes_per_token(self) -> int:
        """计算一个 token 在所有层的 Key 与 Value 理论字节数。

        公式为 ``layers × 2(K+V) × kv_heads × head_dim × dtype_bytes``。
        """

        return (
            self._spec.num_hidden_layers
            * 2
            * self._spec.num_key_value_heads
            * self._spec.head_dim
            * self.dtype_bytes
        )

    @property
    def bytes_per_block(self) -> int:
        """返回一个完整逻辑 block 的理论 KV 字节数。"""

        return self.bytes_per_token * self._spec.block_size

    @property
    def context_blocks(self) -> int:
        """返回覆盖服务上下文上限所需的完整 block 数。"""

        return CommonUtils.ceil_div(
            self._spec.context_limit, self._spec.block_size
        )

    @property
    def theoretical_context_bytes(self) -> int:
        """返回完整上下文按 block 对齐后的理论 KV 字节数。"""

        return self.context_blocks * self.bytes_per_block

    def plan_request(
        self,
        prompt_tokens: int,  # tokenization 后需要保留 KV 的输入 token 数。
        expected_output_tokens: int,  # 策略预计生成的输出 token 数。
    ) -> KVRequestPlan:
        """按完整 block 为一个请求规划逻辑 reservation。

        Args:
            prompt_tokens: tokenization 后需要保留 KV 的输入 token 数，
                可为 0。
            expected_output_tokens: 策略预计生成并保留的输出 token 数。

        Returns:
            包含总 token、block、padding 和理论字节数的计划。

        Raises:
            ContextLimitExceededError: 输入与预计输出之和超过服务上限。
            KVPlannerError: token 数为负，或总 KV token 数为 0。
        """

        CommonUtils.require_non_negative_int(
            prompt_tokens, "prompt_tokens", KVPlannerError
        )
        CommonUtils.require_non_negative_int(
            expected_output_tokens,
            "expected_output_tokens",
            KVPlannerError,
        )
        total_tokens = prompt_tokens + expected_output_tokens
        if total_tokens < 1:
            raise KVPlannerError("request must contain at least one KV token")
        if total_tokens > self._spec.context_limit:
            raise ContextLimitExceededError(
                "request requires {0} tokens but context_limit is {1}".format(
                    total_tokens,
                    self._spec.context_limit,
                )
            )

        # 物理 allocator 以完整 block 分配，尾部不足一块也占一块。
        logical_blocks = CommonUtils.ceil_div(
            total_tokens, self._spec.block_size
        )
        allocated_tokens = logical_blocks * self._spec.block_size
        return KVRequestPlan(
            prompt_tokens=prompt_tokens,
            expected_output_tokens=expected_output_tokens,
            total_tokens=total_tokens,
            logical_blocks=logical_blocks,
            allocated_tokens=allocated_tokens,
            padding_tokens=allocated_tokens - total_tokens,
            theoretical_bytes=logical_blocks * self.bytes_per_block,
        )

    def plan_capacity(
        self,
        usable_kv_bytes: int | None,  # 明确可供 KV 使用的字节预算。
    ) -> KVCapacityPlan:
        """把明确的 KV 专用字节预算换算为逻辑容量。

        Args:
            usable_kv_bytes: 扣除模型权重、CUDA workspace、运行时 buffer
                和安全余量后，明确可供 KV 使用的字节数。

        Returns:
            仅包含完整 block 的容量及除不尽的尾部字节。

        Raises:
            UsableKVCapacityRequiredError: 未提供明确 KV 专用预算。
            KVPlannerError: 预算不是正整数。
        """

        if usable_kv_bytes is None:
            raise UsableKVCapacityRequiredError(
                "usable_kv_bytes is required; total GPU memory is not enough "
                "to infer real usable KV capacity"
            )
        CommonUtils.require_positive_int(
            usable_kv_bytes,
            "usable_kv_bytes",
            KVPlannerError,
        )

        # 向下取整，不把放不下完整 block 的尾部空间计入容量。
        logical_blocks = usable_kv_bytes // self.bytes_per_block
        allocated_bytes = logical_blocks * self.bytes_per_block
        return KVCapacityPlan(
            usable_kv_bytes=usable_kv_bytes,
            logical_blocks=logical_blocks,
            token_capacity=logical_blocks * self._spec.block_size,
            allocated_bytes=allocated_bytes,
            unused_bytes=usable_kv_bytes - allocated_bytes,
        )


def _normalize_dtype(
    dtype: str,  # 要规范化并验证的 KV 数据类型名称。
) -> str:
    """把常用 dtype 别名转换为受支持的规范名称。"""

    normalized = dtype.strip().lower()
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    if normalized not in _DTYPE_BYTES:
        raise KVPlannerError(
            "unsupported KV dtype: {0}; supported dtypes are {1}".format(
                dtype,
                ", ".join(sorted(_DTYPE_BYTES)),
            )
        )
    return normalized
