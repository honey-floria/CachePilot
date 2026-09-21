"""CachePilot 请求生命周期与服务运行时公共接口。

调用方优先从本模块导入稳定类型；以下子模块分别负责准入、deadline、
KV 规划、Registry、资源账本和状态机。执行器与调度循环尚未在本阶段导出。
"""

from .admission import (
    AdmissionDecision,
    AdmissionReason,
    AdmissionSnapshot,
    AdmissionStatus,
    StrictAdmissionConfig,
    StrictAdmissionController,
    TenantAdmissionLimits,
)
from .adaptive_admission import (
    AdaptiveAdmissionConfig,
    AdaptiveAdmissionController,
    AdaptiveAdmissionSnapshot,
    GrowthDecision,
    GrowthReason,
    GrowthStatus,
)
from .deadlines import (
    DeadlinePhase,
    DeadlinePolicy,
    DeadlineSnapshot,
    RequestDeadlineManager,
    TimeoutEvent,
    TimeoutReason,
)
from .empty_service import create_server
from .kv_planner import (
    ContextLimitExceededError,
    KVCapacityPlan,
    KVModelSpec,
    KVPlanner,
    KVPlannerError,
    KVRequestPlan,
    UsableKVCapacityRequiredError,
)
from .registry import RequestRegistry, RequestSnapshot
from .resources import ResourceLeaseManager, ResourceLeaseSnapshot
from .state_machine import RequestState, RequestStateMachine

__all__ = [
    # Strict/Adaptive 准入决策与配置。
    "AdmissionDecision",
    "AdmissionReason",
    "AdmissionSnapshot",
    "AdmissionStatus",
    "AdaptiveAdmissionConfig",
    "AdaptiveAdmissionController",
    "AdaptiveAdmissionSnapshot",
    "GrowthDecision",
    "GrowthReason",
    "GrowthStatus",
    "StrictAdmissionConfig",
    "StrictAdmissionController",
    "TenantAdmissionLimits",
    "DeadlinePhase",
    "DeadlinePolicy",
    "DeadlineSnapshot",
    "RequestDeadlineManager",
    "TimeoutEvent",
    "TimeoutReason",
    # Registry 与 KV 理论容量规划。
    "RequestRegistry",
    "RequestSnapshot",
    "KVModelSpec",
    "KVPlanner",
    "KVPlannerError",
    "KVRequestPlan",
    "KVCapacityPlan",
    "ContextLimitExceededError",
    "UsableKVCapacityRequiredError",
    # 资源所有权、请求状态机和骨架探活服务。
    "ResourceLeaseManager",
    "ResourceLeaseSnapshot",
    "RequestState",
    "RequestStateMachine",
    "create_server",
]
