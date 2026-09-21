"""请求逻辑 KV reservation 与物理执行器 handle 的资源账本。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple


class ResourceLeaseError(ValueError):
    """资源租约操作无效时抛出的基类。"""


class LeaseNotFoundError(ResourceLeaseError):
    """请求没有资源租约时抛出。"""


class CapacityExceededError(ResourceLeaseError):
    """逻辑 KV 容量不足时抛出。"""


class LeaseReleasedError(ResourceLeaseError):
    """已释放租约仍尝试增长或绑定 handle 时抛出。"""


@dataclass(frozen=True)
class ResourceLeaseSnapshot:
    request_id: str
    logical_blocks: int
    peak_logical_blocks: int
    physical_handles: Tuple[Hashable, ...]
    released: bool


@dataclass
class _ResourceLease:
    request_id: str
    logical_blocks: int
    peak_logical_blocks: int
    physical_handles: list[Hashable]
    released: bool = False


class ResourceLeaseManager:
    """线程安全管理逻辑 KV block 与物理 handle 的独立所有权。"""

    def __init__(self, capacity_blocks: Optional[int] = None) -> None:
        if capacity_blocks is not None and (
            type(capacity_blocks) is not int or capacity_blocks < 1
        ):
            raise ValueError("capacity_blocks must be a positive integer or None")
        self._capacity_blocks = capacity_blocks
        self._leases: Dict[str, _ResourceLease] = {}
        self._total_logical_blocks = 0
        self._lock = threading.Lock()

    @property
    def capacity_blocks(self) -> Optional[int]:
        return self._capacity_blocks

    @property
    def total_logical_blocks(self) -> int:
        with self._lock:
            return self._total_logical_blocks

    def reserve(self, request_id: str, logical_blocks: int) -> ResourceLeaseSnapshot:
        """申请初始逻辑 reservation；同一请求只能申请一次。"""

        self._validate_request_id(request_id)
        self._validate_blocks(logical_blocks)
        with self._lock:
            if request_id in self._leases:
                raise ResourceLeaseError(
                    "request already has a resource lease: {0}".format(request_id)
                )
            self._ensure_capacity(logical_blocks)
            lease = _ResourceLease(
                request_id=request_id,
                logical_blocks=logical_blocks,
                peak_logical_blocks=logical_blocks,
                physical_handles=[],
            )
            self._leases[request_id] = lease
            self._total_logical_blocks += logical_blocks
            return self._snapshot(lease)

    def grow(self, request_id: str, additional_blocks: int) -> ResourceLeaseSnapshot:
        """增长现有逻辑 reservation，并更新峰值。"""

        self._validate_request_id(request_id)
        self._validate_blocks(additional_blocks)
        with self._lock:
            lease = self._get_active_lease(request_id)
            self._ensure_capacity(additional_blocks)
            lease.logical_blocks += additional_blocks
            lease.peak_logical_blocks = max(
                lease.peak_logical_blocks,
                lease.logical_blocks,
            )
            self._total_logical_blocks += additional_blocks
            return self._snapshot(lease)

    def attach_physical_handle(
        self, request_id: str, handle: Hashable
    ) -> ResourceLeaseSnapshot:
        """记录执行器分配的物理 handle，不混入逻辑 block 容量。"""

        self._validate_request_id(request_id)
        try:
            hash(handle)
        except TypeError as exc:
            raise TypeError("physical handle must be hashable") from exc
        with self._lock:
            lease = self._get_active_lease(request_id)
            if handle not in lease.physical_handles:
                lease.physical_handles.append(handle)
            return self._snapshot(lease)

    def release(self, request_id: str) -> ResourceLeaseSnapshot:
        """释放逻辑 reservation 和物理 handles；重复调用保持幂等。"""

        self._validate_request_id(request_id)
        with self._lock:
            lease = self._leases.get(request_id)
            if lease is None:
                raise LeaseNotFoundError(
                    "resource lease not found: {0}".format(request_id)
                )
            if not lease.released:
                self._total_logical_blocks -= lease.logical_blocks
                lease.logical_blocks = 0
                lease.physical_handles.clear()
                lease.released = True
            return self._snapshot(lease)

    def snapshot(self, request_id: str) -> ResourceLeaseSnapshot:
        """读取租约快照，不改变所有权。"""

        self._validate_request_id(request_id)
        with self._lock:
            return self._snapshot(self._leases.get(request_id))

    def _get_active_lease(self, request_id: str) -> _ResourceLease:
        lease = self._leases.get(request_id)
        if lease is None:
            raise LeaseNotFoundError(
                "resource lease not found: {0}".format(request_id)
            )
        if lease.released:
            raise LeaseReleasedError(
                "resource lease has been released: {0}".format(request_id)
            )
        return lease

    def _ensure_capacity(self, additional_blocks: int) -> None:
        if (
            self._capacity_blocks is not None
            and self._total_logical_blocks + additional_blocks
            > self._capacity_blocks
        ):
            raise CapacityExceededError(
                "logical KV capacity exceeded: requested {0}, available {1}".format(
                    additional_blocks,
                    self._capacity_blocks - self._total_logical_blocks,
                )
            )

    @staticmethod
    def _snapshot(lease: Optional[_ResourceLease]) -> ResourceLeaseSnapshot:
        if lease is None:
            raise LeaseNotFoundError("resource lease not found")
        return ResourceLeaseSnapshot(
            request_id=lease.request_id,
            logical_blocks=lease.logical_blocks,
            peak_logical_blocks=lease.peak_logical_blocks,
            physical_handles=tuple(lease.physical_handles),
            released=lease.released,
        )

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be a non-empty string")

    @staticmethod
    def _validate_blocks(blocks: int) -> None:
        if type(blocks) is not int or blocks < 1:
            raise ValueError("logical blocks must be a positive integer")
