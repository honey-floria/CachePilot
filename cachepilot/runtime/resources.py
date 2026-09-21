"""请求逻辑 KV reservation 与物理执行器 handle 的资源账本。

本模块只维护控制面的资源所有权，不直接申请或释放 GPU 显存：

* ``logical_blocks`` 表示准入控制预留的逻辑 KV block 数；
* ``physical_handles`` 表示执行器返回的物理资源标识；
* 二者分别记账，避免把理论预算误报为执行器真实显存占用。

所有修改都在同一把互斥锁内完成，因此容量检查与更新是原子的。
"""

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
    """对外暴露的不可变资源租约快照。

    Attributes:
        request_id: 拥有该租约的请求 ID。
        logical_blocks: 当前仍被请求占用的逻辑 KV block 数。
        peak_logical_blocks: 请求生命周期内的逻辑 block 峰值。
        physical_handles: 执行器物理资源标识的只读副本。
        released: 租约是否已经执行过释放。
    """

    request_id: str
    logical_blocks: int
    peak_logical_blocks: int
    physical_handles: Tuple[Hashable, ...]
    released: bool


@dataclass
class _ResourceLease:
    """账本内部使用的可变租约；只能在 manager 锁内修改。"""

    request_id: str
    logical_blocks: int
    peak_logical_blocks: int
    physical_handles: list[Hashable]
    released: bool = False


class ResourceLeaseManager:
    """线程安全管理逻辑 KV block 与物理 handle 的独立所有权。

    Args:
        capacity_blocks: 可选逻辑 KV 总容量。``None`` 表示只记账而不设置
            上限；正整数表示所有活跃租约合计不能超过该值。
    """

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
        """返回配置的逻辑容量上限；``None`` 表示未设置。"""

        return self._capacity_blocks

    @property
    def total_logical_blocks(self) -> int:
        """原子读取所有未释放租约当前占用的 block 总数。"""

        with self._lock:
            return self._total_logical_blocks

    def reserve(self, request_id: str, logical_blocks: int) -> ResourceLeaseSnapshot:
        """为请求申请初始逻辑 reservation。

        Args:
            request_id: 全局唯一请求 ID。
            logical_blocks: 需要预留的正整数 block 数。

        Returns:
            创建完成后的不可变租约快照。

        Raises:
            ResourceLeaseError: 同一请求已经创建过租约。
            CapacityExceededError: 新增 block 会突破配置容量。
            ValueError: 请求 ID 或 block 数格式无效。
        """

        self._validate_request_id(request_id)
        self._validate_blocks(logical_blocks)
        with self._lock:
            if request_id in self._leases:
                raise ResourceLeaseError(
                    "request already has a resource lease: {0}".format(request_id)
                )
            # 容量检查与总量增加处于同一临界区，防止并发超额。
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
        """增长现有逻辑 reservation，并更新历史峰值。

        Args:
            request_id: 已持有活跃租约的请求 ID。
            additional_blocks: 本次额外申请的正整数 block 数。

        Returns:
            增长后的租约快照。

        Raises:
            LeaseNotFoundError: 请求尚未创建租约。
            LeaseReleasedError: 租约已经释放，不允许重新增长。
            CapacityExceededError: 增长会突破逻辑容量上限。
        """

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
        """记录执行器分配的物理 handle，不混入逻辑容量。

        Args:
            request_id: 已持有活跃租约的请求 ID。
            handle: 执行器用于定位、取消或释放物理资源的可哈希标识。

        Returns:
            绑定后的租约快照；重复 handle 不会产生重复记录。

        Raises:
            TypeError: ``handle`` 不可哈希。
            LeaseNotFoundError: 请求没有租约。
            LeaseReleasedError: 请求租约已经释放。
        """

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
        """释放逻辑 reservation 并清除物理 handle 记录。

        Args:
            request_id: 要释放资源的请求 ID。

        Returns:
            释放后的快照；重复调用不会再次扣减总量。

        Note:
            清空 handle 只表示控制层不再持有该标识。真实执行器接入后，
            调用方必须先执行 abort/free，再清理本账本。
        """

        self._validate_request_id(request_id)
        with self._lock:
            lease = self._leases.get(request_id)
            if lease is None:
                raise LeaseNotFoundError(
                    "resource lease not found: {0}".format(request_id)
                )
            # released 门闩保证重复终止不会把总量扣成负数。
            if not lease.released:
                self._total_logical_blocks -= lease.logical_blocks
                lease.logical_blocks = 0
                lease.physical_handles.clear()
                lease.released = True
            return self._snapshot(lease)

    def snapshot(self, request_id: str) -> ResourceLeaseSnapshot:
        """读取指定请求的不可变租约快照，不改变资源所有权。"""

        self._validate_request_id(request_id)
        with self._lock:
            return self._snapshot(self._leases.get(request_id))

    def _get_active_lease(self, request_id: str) -> _ResourceLease:
        """持锁时取得未释放租约，否则抛出稳定异常。"""

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
        """在调用方已持锁时检查新增 block 是否突破总容量。"""

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
        """把内部可变租约转换为不会泄露可变列表的只读快照。"""

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
