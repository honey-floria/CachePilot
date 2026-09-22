"""基于历史输出长度 P95 的 Adaptive Admission。

来自相同 tenant、相同 prompt长度桶的历史实际输出长度

Adaptive 模式仍复用 Strict 的所有硬上限，但初始 reservation 改为 tenant
与 prompt 长度分桶后的历史输出 P95 加安全余量。生成超过估算时必须先增长租约；
如果增长会突破 KV 或 tenant 容量，调用方必须停止生成。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from cachepilot.runtime.admission import (
    AdmissionError,
    AdmissionReason,
    StrictAdmissionConfig,
    StrictAdmissionController,
    _AdmissionRequest,
)
from cachepilot.runtime.kv_planner import KVPlanner
from cachepilot.utils import CommonUtils


class GrowthStatus(str, Enum):
    """生成过程是否可以在完成本次容量检查后继续。"""

    CONTINUE = "CONTINUE"
    STOP_REQUIRED = "STOP_REQUIRED"


class GrowthReason(str, Enum):
    """生成增长检查的机器可读原因。"""

    WITHIN_RESERVATION = "within_reservation"
    RESERVATION_GROWN = "reservation_grown"
    MAX_NEW_TOKENS = "max_new_tokens"
    KV_CAPACITY = "kv_capacity"
    TENANT_ACTIVE_TOKENS = "tenant_active_tokens"


@dataclass(frozen=True)
class AdaptiveAdmissionConfig:
    """Adaptive 策略特有配置。

    Attributes:
        strict: 复用的全局、tenant 和队列硬限制。
        prompt_bucket_boundaries: 递增的 prompt token 上界，例如
            ``(128, 512)`` 会形成 ``<=128``、``129..512`` 和 ``>512`` 三桶。
        min_samples_per_bucket: 使用 P95 前要求同 tenant 同桶具备的最少样本。
        safety_margin_tokens: 加在 P95 上的固定输出 token 安全余量。
        max_samples_per_bucket: 每桶保留的最近样本上限，防止内存
            无界增长。
    """

    strict: StrictAdmissionConfig  # 复用的全局、tenant 和队列硬限制。
    prompt_bucket_boundaries: Tuple[int, ...]  # 严格递增的 prompt token 桶上界。
    min_samples_per_bucket: int  # 启用 P95 估算前每桶所需的最少样本数。
    safety_margin_tokens: int  # 添加到 P95 上的固定输出 token 安全余量。
    max_samples_per_bucket: int = 1000  # 每个历史桶保留的最近样本上限。

    def __post_init__(self) -> None:
        """校验样本窗口与严格递增的 prompt 分桶边界。"""

        if not isinstance(self.strict, StrictAdmissionConfig):
            raise TypeError("strict must be a StrictAdmissionConfig")
        CommonUtils.require_positive_int(
            self.min_samples_per_bucket,
            "min_samples_per_bucket",
            AdmissionError,
        )
        CommonUtils.require_non_negative_int(
            self.safety_margin_tokens,
            "safety_margin_tokens",
            AdmissionError,
        )
        CommonUtils.require_positive_int(
            self.max_samples_per_bucket,
            "max_samples_per_bucket",
            AdmissionError,
        )
        if self.max_samples_per_bucket < self.min_samples_per_bucket:
            raise AdmissionError(
                "max_samples_per_bucket cannot be less than min_samples_per_bucket"
            )

        previous = 0
        for boundary in self.prompt_bucket_boundaries:
            CommonUtils.require_positive_int(
                boundary,
                "prompt_bucket_boundaries item",
                AdmissionError,
            )
            if boundary <= previous:
                raise AdmissionError(
                    "prompt_bucket_boundaries must be strictly increasing"
                )
            previous = boundary


@dataclass(frozen=True)
class GrowthDecision:
    """一次运行中 reservation 增长检查的结果快照。"""

    request_id: str  # 本次增长检查对应的请求 ID。
    status: GrowthStatus  # 继续生成或必须停止的状态。
    reason: GrowthReason  # 产生增长决策的机器可读原因。
    generated_output_tokens: int  # 当前已生成的输出 token 总数。
    reserved_output_tokens: int  # 当前 reservation 覆盖的输出 token 数。
    reserved_blocks: int  # 当前 reservation 占用的逻辑 KV block 数。


@dataclass(frozen=True)
class AdaptiveAdmissionSnapshot:
    """Adaptive 估算质量与各历史桶样本量的观测快照。"""

    fallback_count: int  # 因样本不足回退 Strict 的累计次数。
    estimation_count: int  # 已记录最终估算误差的请求数。
    underestimation_count: int  # 首次估计低于实际输出的请求数。
    signed_error_tokens: int  # 实际值减首次估计值的累计有符号误差。
    absolute_error_tokens: int  # 首次估计绝对误差的累计 token 数。
    bucket_sample_counts: Tuple[Tuple[str, int, int], ...]  # 各 tenant/桶样本数。


class AdaptiveAdmissionController(StrictAdmissionController):
    """使用 tenant/prompt 分桶 P95，样本不足时回退 Strict。

    Args:
        planner: 逻辑 KV block 规划器。
        config: 包含 Strict 硬限制和 Adaptive 历史参数的配置。
    """

    def __init__(
        self,
        planner: KVPlanner,  # 逻辑 KV block 规划器。
        config: AdaptiveAdmissionConfig,  # Strict 硬限制和历史估算配置。
    ) -> None:
        if not isinstance(config, AdaptiveAdmissionConfig):
            raise TypeError("config must be an AdaptiveAdmissionConfig")
        super().__init__(planner, config.strict)
        self._adaptive_config = config
        self._history: Dict[Tuple[str, int], List[int]] = {}
        self._fallback_count = 0
        self._estimation_count = 0
        self._underestimation_count = 0
        self._signed_error_tokens = 0
        self._absolute_error_tokens = 0

    @property
    def adaptive_config(self) -> AdaptiveAdmissionConfig:
        """返回 Adaptive 策略配置。"""

        return self._adaptive_config

    def observe_output(
        self,
        tenant_id: str,  # 样本所属 tenant ID。
        prompt_tokens: int,  # 用于选择长度桶的输入 token 数。
        output_tokens: int,  # 样本实际生成的输出 token 数。
    ) -> None:
        """记录一个已知完成样本，供后续请求估算使用。

        Args:
            tenant_id: 样本所属 tenant，历史默认不跨 tenant 共享。
            prompt_tokens: 用于选择 prompt 长度桶的输入 token 数。
            output_tokens: 请求实际生成的输出 token 数。

        该入口主要用于恢复历史、离线预热和测试；正常在线请求应通过
        ``complete`` 同时记录误差、样本并释放资源。
        """

        CommonUtils.require_identifier(
            tenant_id, "tenant_id", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            prompt_tokens, "prompt_tokens", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            output_tokens, "output_tokens", AdmissionError
        )
        with self._lock:
            self._record_history_locked(tenant_id, prompt_tokens, output_tokens)

    def reserve_generated_tokens(
        self,
        request_id: str,  # 已接纳且活跃的请求 ID。
        generated_output_tokens: int,  # 截至当前已生成的输出 token 总数。
    ) -> GrowthDecision:
        """生成增长时扩展 reservation，硬容量不足则要求停止。

        Args:
            request_id: 已被 Adaptive Controller 接纳的活跃请求 ID。
            generated_output_tokens: 截至当前已经生成的输出 token 总数，
                不是本轮增量。

        Returns:
            ``CONTINUE`` 表示当前 token 数在 reservation 内或增长成功；
            ``STOP_REQUIRED`` 表示继续会突破 max_new_tokens、KV 容量或 tenant
            token 硬上限。

        Logic:
            先计算新计划，再同时检查全局 block 和 tenant token 增量；
            全部通过后才修改资源账本，因此失败不会留下半更新状态。
        """

        CommonUtils.require_identifier(
            request_id, "request_id", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            generated_output_tokens,
            "generated_output_tokens",
            AdmissionError,
        )
        with self._lock:
            request = self._active.get(request_id)
            if request is None:
                raise AdmissionError(
                    "request is not active: {0}".format(request_id)
                )
            # 客户端声明的 max_new_tokens 始终是最终硬上限。
            if generated_output_tokens > request.max_new_tokens:
                return self._growth_decision(
                    request,
                    generated_output_tokens,
                    GrowthStatus.STOP_REQUIRED,
                    GrowthReason.MAX_NEW_TOKENS,
                )
            # 仍位于当前 reservation 内时无需触碰共享容量账本。
            if generated_output_tokens <= request.estimated_output_tokens:
                return self._growth_decision(
                    request,
                    generated_output_tokens,
                    GrowthStatus.CONTINUE,
                    GrowthReason.WITHIN_RESERVATION,
                )

            new_plan = self._planner.plan_request(
                request.prompt_tokens,
                generated_output_tokens,
            )
            additional_blocks = (
                new_plan.logical_blocks - request.plan.logical_blocks
            )
            additional_tokens = new_plan.total_tokens - request.plan.total_tokens
            # 先完成全部硬限制检查，再更新 grow 或 tenant 计数。
            if (
                self._resources.total_logical_blocks + additional_blocks
                > self._config.usable_blocks
            ):
                return self._growth_decision(
                    request,
                    generated_output_tokens,
                    GrowthStatus.STOP_REQUIRED,
                    GrowthReason.KV_CAPACITY,
                )

            limits = self._config.tenant_limits[request.tenant_id]
            if (
                self._tenant_active_tokens.get(request.tenant_id, 0)
                + additional_tokens
                > limits.max_active_tokens
            ):
                return self._growth_decision(
                    request,
                    generated_output_tokens,
                    GrowthStatus.STOP_REQUIRED,
                    GrowthReason.TENANT_ACTIVE_TOKENS,
                )

            if additional_blocks > 0:
                self._resources.grow(request_id, additional_blocks)
            self._tenant_active_tokens[request.tenant_id] += additional_tokens
            request.plan = new_plan
            request.estimated_output_tokens = generated_output_tokens
            return self._growth_decision(
                request,
                generated_output_tokens,
                GrowthStatus.CONTINUE,
                GrowthReason.RESERVATION_GROWN,
            )

    def complete(
        self,
        request_id: str,  # 已接纳且仍活跃的请求 ID。
        actual_output_tokens: int,  # 请求最终实际生成的输出 token 数。
    ) -> bool:
        """记录估算误差和历史样本，然后释放请求 reservation。

        Args:
            request_id: 已接纳且仍活跃的请求 ID。
            actual_output_tokens: 请求实际生成的最终输出 token 数。

        Returns:
            找到并完成活跃请求时返回 ``True``；请求已不活跃时返回
            ``False``。

        Raises:
            AdmissionError: 实际输出超过 max_new_tokens，或调用方没有先
                通过 ``reserve_generated_tokens`` 为超出初始估算的输出扩容。

        估算误差始终相对“首次接纳时的估算”计算，不会因中途增长而
        被抹平。
        """

        CommonUtils.require_identifier(
            request_id, "request_id", AdmissionError
        )
        CommonUtils.require_non_negative_int(
            actual_output_tokens, "actual_output_tokens", AdmissionError
        )
        with self._lock:
            request = self._active.get(request_id)
            if request is None:
                return False
            if actual_output_tokens > request.max_new_tokens:
                raise AdmissionError(
                    "actual_output_tokens cannot exceed max_new_tokens"
                )
            if actual_output_tokens > request.estimated_output_tokens:
                raise AdmissionError(
                    "output growth must be reserved before completion"
                )

            error = actual_output_tokens - request.initial_estimated_output_tokens
            self._estimation_count += 1
            self._signed_error_tokens += error
            self._absolute_error_tokens += abs(error)
            if error > 0:
                self._underestimation_count += 1
            self._record_history_locked(
                request.tenant_id,
                request.prompt_tokens,
                actual_output_tokens,
            )
            return self._release_locked(request_id)

    def adaptive_snapshot(self) -> AdaptiveAdmissionSnapshot:
        """原子读取回退次数、估算误差和各桶样本数量。"""

        with self._lock:
            return AdaptiveAdmissionSnapshot(
                fallback_count=self._fallback_count,
                estimation_count=self._estimation_count,
                underestimation_count=self._underestimation_count,
                signed_error_tokens=self._signed_error_tokens,
                absolute_error_tokens=self._absolute_error_tokens,
                bucket_sample_counts=tuple(
                    sorted(
                        (tenant_id, bucket, len(samples))
                        for (tenant_id, bucket), samples in self._history.items()
                    )
                ),
            )

    def _build_request(
        self,
        request_id: str,  # 请求唯一标识。
        tenant_id: str,  # 请求所属 tenant。
        prompt_tokens: int,  # 输入 token 数。
        max_new_tokens: int,  # 客户端声明的最大输出 token 数。
    ) -> _AdmissionRequest:
        """使用 P95 或 Strict 回退值创建请求的初始 reservation 计划。"""

        # 即使初始估算较小，也必须先验证最坏上下文是否合法。
        self._planner.plan_request(prompt_tokens, max_new_tokens)
        bucket = self._bucket_for(prompt_tokens)
        samples = self._history.get((tenant_id, bucket), ())
        fallback = len(samples) < self._adaptive_config.min_samples_per_bucket
        if fallback:
            # 冷启动阶段宁可多预留，也不基于稀疏样本冒险低估。
            estimated_output_tokens = max_new_tokens
            self._fallback_count += 1
        else:
            percentile = _nearest_rank_percentile(samples, 0.95)
            estimated_output_tokens = min(
                max_new_tokens,
                max(1, percentile + self._adaptive_config.safety_margin_tokens),
            )

        plan = self._planner.plan_request(prompt_tokens, estimated_output_tokens)
        return _AdmissionRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            plan=plan,
            initial_estimated_output_tokens=estimated_output_tokens,
            estimated_output_tokens=estimated_output_tokens,
            fallback_to_strict=fallback,
        )

    def _record_history_locked(
        self,
        tenant_id: str,  # 样本所属 tenant。
        prompt_tokens: int,  # 样本输入 token 数。
        output_tokens: int,  # 样本实际输出 token 数。
    ) -> None:
        """调用方持锁时追加样本，并按 FIFO 裁剪为固定窗口。"""

        key = (tenant_id, self._bucket_for(prompt_tokens))
        samples = self._history.setdefault(key, [])
        samples.append(output_tokens)
        excess = len(samples) - self._adaptive_config.max_samples_per_bucket
        if excess > 0:
            del samples[:excess]

    def _bucket_for(
        self,
        prompt_tokens: int,  # 要映射到长度桶的输入 token 数。
    ) -> int:
        """返回 prompt 所属桶索引；最后一桶表示溢出区间。"""

        for index, boundary in enumerate(
            self._adaptive_config.prompt_bucket_boundaries
        ):
            if prompt_tokens <= boundary:
                return index
        return len(self._adaptive_config.prompt_bucket_boundaries)

    @staticmethod
    def _growth_decision(
        request: _AdmissionRequest,  # 当前 reservation 的内部请求记录。
        generated_output_tokens: int,  # 当前已生成的输出 token 总数。
        status: GrowthStatus,  # 增长检查状态。
        reason: GrowthReason,  # 增长检查原因。
    ) -> GrowthDecision:
        """用请求当前 reservation 构造不可变增长决策。"""

        return GrowthDecision(
            request_id=request.request_id,
            status=status,
            reason=reason,
            generated_output_tokens=generated_output_tokens,
            reserved_output_tokens=request.estimated_output_tokens,
            reserved_blocks=request.plan.logical_blocks,
        )


def _nearest_rank_percentile(
    samples: Tuple[int, ...] | List[int],  # 用于计算分位数的整数样本。
    value: float,  # 取值在 0 到 1 之间的目标分位比例。
) -> int:
    """按实验协议的 nearest-rank 定义计算分位数，不做插值。"""

    ordered = sorted(samples)
    rank = max(1, math.ceil(value * len(ordered)))
    return ordered[rank - 1]
