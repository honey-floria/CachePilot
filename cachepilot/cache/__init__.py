"""租户作用域的缓存元数据公共接口。"""

from .prefix_index import (
    PrefixIndexError,
    PrefixIndexSnapshot,
    PrefixLookupResult,
    PrefixScope,
    PrefixScopeKey,
    TenantPrefixIndex,
)

__all__ = [
    "PrefixIndexError",
    "PrefixIndexSnapshot",
    "PrefixLookupResult",
    "PrefixScope",
    "PrefixScopeKey",
    "TenantPrefixIndex",
]
