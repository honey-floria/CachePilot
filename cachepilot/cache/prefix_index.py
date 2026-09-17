"""最小化的租户作用域逻辑前缀索引。

该精确匹配索引仅用于建立初始 API 范围要求的隔离契约。
最长前缀查找、淘汰以及执行器上报的物理命中将由后续的前缀索引实现负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set, Tuple


@dataclass(frozen=True)
class PrefixScopeKey:
    """允许参与逻辑 KV 复用的全部维度。"""

    tenant_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    quantization_config: str
    tokenized_prefix: Tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        quantization_config: str,
        tokenized_prefix: Iterable[int],
    ) -> "PrefixScopeKey":
        """构建可哈希键，同时精确保留词元 ID 及其顺序。"""

        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        return cls(
            tenant_id=tenant_id,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            quantization_config=quantization_config,
            tokenized_prefix=tuple(tokenized_prefix),
        )


class TenantPrefixIndex:
    """强制限定租户作用域的精确逻辑前缀成员关系。"""

    def __init__(self) -> None:
        self._entries: Set[PrefixScopeKey] = set()

    def record(self, key: PrefixScopeKey) -> None:
        self._entries.add(key)

    def contains(self, key: PrefixScopeKey) -> bool:
        return key in self._entries
