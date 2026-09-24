"""tenant 与模型版本隔离的逻辑 token 前缀索引。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

from cachepilot.utils import CommonUtils


class PrefixIndexError(ValueError):
    """前缀键、索引操作或 token 序列无效。"""


@dataclass(frozen=True)
class PrefixScope:
    """决定逻辑前缀是否允许互相复用的隔离维度。"""

    tenant_id: str  # prefix 所属 tenant，默认禁止跨 tenant 共享。
    model_id: str  # 模型配置 ID。
    model_revision: str  # 固定模型 revision。
    tokenizer_revision: str  # 固定 tokenizer revision。
    quantization_config: str  # 量化配置或明确的 none 标识。

    def __post_init__(self) -> None:
        """要求所有隔离维度都是非空字符串。"""

        for field_name in (
            "tenant_id",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "quantization_config",
        ):
            CommonUtils.require_identifier(
                getattr(self, field_name), field_name, PrefixIndexError
            )


@dataclass(frozen=True)
class PrefixScopeKey:
    """隔离作用域与按顺序 tokenized prefix 组成的逻辑缓存键。"""

    tenant_id: str  # prefix 所属 tenant。
    model_id: str  # 模型配置 ID。
    model_revision: str  # 固定模型 revision。
    tokenizer_revision: str  # 固定 tokenizer revision。
    quantization_config: str  # 量化配置。
    tokenized_prefix: Tuple[int, ...]  # 精确保留顺序的 token ID 元组。

    def __post_init__(self) -> None:
        """校验作用域和非空 token 前缀。"""

        self.scope
        if not self.tokenized_prefix:
            raise PrefixIndexError("tokenized_prefix must not be empty")
        for token_id in self.tokenized_prefix:
            CommonUtils.require_non_negative_int(
                token_id, "tokenized_prefix item", PrefixIndexError
            )

    @property
    def scope(self) -> PrefixScope:
        """返回不包含 token 序列的隔离作用域。"""

        return PrefixScope(
            tenant_id=self.tenant_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            quantization_config=self.quantization_config,
        )

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,  # prefix 所属 tenant。
        model_id: str,  # 模型配置 ID。
        model_revision: str,  # 固定模型 revision。
        tokenizer_revision: str,  # 固定 tokenizer revision。
        quantization_config: str,  # 量化配置。
        tokenized_prefix: Iterable[int],  # 保留顺序的 token ID 序列。
    ) -> PrefixScopeKey:
        """复制 token 序列并构建可哈希、不可变的逻辑键。"""

        try:
            tokens = tuple(tokenized_prefix)
        except TypeError as error:
            raise PrefixIndexError(
                "tokenized_prefix must be an iterable of integers"
            ) from error
        return cls(
            tenant_id=tenant_id,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            quantization_config=quantization_config,
            tokenized_prefix=tokens,
        )


@dataclass(frozen=True)
class PrefixLookupResult:
    """一次最长逻辑前缀查询结果。"""

    logical_hit: bool  # 是否在相同隔离作用域找到 token 前缀。
    matched_tokens: int  # 最长逻辑命中的 token 数；未命中为 0。
    matched_key: Optional[PrefixScopeKey]  # 最长命中的完整逻辑键。
    physical_hit: Optional[bool] = None  # 执行器未核验，因此始终不可观测。


@dataclass(frozen=True)
class PrefixIndexSnapshot:
    """逻辑索引的稳定统计快照。"""

    scopes: int  # 当前非空隔离作用域数量。
    entries: int  # 当前逻辑前缀总数。
    logical_hits: int  # 累计逻辑命中查询数。
    logical_misses: int  # 累计逻辑未命中查询数。


class TenantPrefixIndex:
    """线程安全的 tenant/version 隔离最长逻辑 token 前缀索引。

    本索引只声明控制层发现了可复用前缀，不保存执行器物理 KV handle，
    也不把逻辑命中转换为物理命中。
    """

    def __init__(self) -> None:
        """创建空索引和零命中统计。"""

        self._entries: Dict[PrefixScope, Set[Tuple[int, ...]]] = {}
        self._logical_hits = 0
        self._logical_misses = 0
        self._lock = threading.Lock()

    def record(self, key: PrefixScopeKey) -> bool:
        """幂等记录逻辑前缀；首次插入返回 ``True``。"""

        if not isinstance(key, PrefixScopeKey):
            raise TypeError("key must be a PrefixScopeKey")
        with self._lock:
            prefixes = self._entries.setdefault(key.scope, set())
            before = len(prefixes)
            prefixes.add(key.tokenized_prefix)
            return len(prefixes) != before

    def contains(self, key: PrefixScopeKey) -> bool:
        """返回完整作用域和 token 序列是否精确存在，不修改命中统计。"""

        if not isinstance(key, PrefixScopeKey):
            raise TypeError("key must be a PrefixScopeKey")
        with self._lock:
            return key.tokenized_prefix in self._entries.get(key.scope, set())

    def lookup(self, key: PrefixScopeKey) -> PrefixLookupResult:
        """在完全相同作用域内查找最长已记录 token 前缀。

        返回值的 ``physical_hit`` 固定为 ``None``，因为只有执行器才能证明
        物理 KV 是否真正被复用。逻辑命中不能据此上报物理命中。
        """

        if not isinstance(key, PrefixScopeKey):
            raise TypeError("key must be a PrefixScopeKey")
        with self._lock:
            prefixes = self._entries.get(key.scope, set())
            for length in range(len(key.tokenized_prefix), 0, -1):
                candidate = key.tokenized_prefix[:length]
                if candidate in prefixes:
                    self._logical_hits += 1
                    matched_key = PrefixScopeKey(
                        tenant_id=key.tenant_id,
                        model_id=key.model_id,
                        model_revision=key.model_revision,
                        tokenizer_revision=key.tokenizer_revision,
                        quantization_config=key.quantization_config,
                        tokenized_prefix=candidate,
                    )
                    return PrefixLookupResult(True, length, matched_key)
            self._logical_misses += 1
            return PrefixLookupResult(False, 0, None)

    def discard(self, key: PrefixScopeKey) -> bool:
        """删除精确逻辑前缀并清理空作用域；不存在时返回 ``False``。"""

        if not isinstance(key, PrefixScopeKey):
            raise TypeError("key must be a PrefixScopeKey")
        with self._lock:
            prefixes = self._entries.get(key.scope)
            if prefixes is None or key.tokenized_prefix not in prefixes:
                return False
            prefixes.remove(key.tokenized_prefix)
            if not prefixes:
                del self._entries[key.scope]
            return True

    def invalidate_scope(self, scope: PrefixScope) -> int:
        """删除作用域内全部逻辑前缀，并返回删除数量。"""

        if not isinstance(scope, PrefixScope):
            raise TypeError("scope must be a PrefixScope")
        with self._lock:
            return len(self._entries.pop(scope, set()))

    def snapshot(self) -> PrefixIndexSnapshot:
        """返回作用域、条目与逻辑查询统计，不暴露高基数 token 内容。"""

        with self._lock:
            return PrefixIndexSnapshot(
                scopes=len(self._entries),
                entries=sum(len(prefixes) for prefixes in self._entries.values()),
                logical_hits=self._logical_hits,
                logical_misses=self._logical_misses,
            )
