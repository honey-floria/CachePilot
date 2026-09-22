# ADR-0006：FCFS 与 WFQ 调度语义

## 状态

已接受，2026-09-22。

## 决策

Phase 0 调度器只负责从已排队请求中决定下一次准入重试顺序，不替代
Admission Controller 的 KV、并发与 tenant 配额检查。

两个策略都维护 `interactive`、`batch` 两个调度类别，每个类别按 tenant
拆成 FIFO 子队列。tenant 队首是唯一可选请求，因此同一 tenant 内不会乱序。

- FCFS：`interactive` 严格优先于 `batch`，类别内按全局入队序号选择最早请求。
- WFQ：每个类别维护独立虚拟时间。请求的虚拟开始标签为类别虚拟时间和
  tenant 上一完成标签的较大值；虚拟完成标签为
  `virtual_start + service_cost / tenant_weight`。类别内选择完成标签最小的
  tenant 队首，标签相同按入队序号和 request ID 稳定打破平局。
- 工作量：`service_cost` 是正整数，Phase 0 使用预计执行 token 数。权重是
  正整数；未显式配置的 tenant 使用 `default_weight=1`。
- 优先级：未触发饥饿保护时，WFQ 与 FCFS 一样优先选择 interactive。
- 饥饿保护：WFQ 的 tenant 队首实际等待达到 `max_starvation_ns` 后，跨类别
  按最早入队顺序强制选择。该上限约束“已成为 tenant 队首后的等待”，不承诺
  在容量不足时为任意深度的积压请求提供端到端延迟上限。

虚拟标签使用整数分数计算，不依赖浮点舍入；时间由可注入的单调逻辑时钟
提供。相同入队调用、时钟、权重和成本会产生相同顺序。

## 后果

FCFS 是容易解释的优先级基线，但持续 interactive 流量可能使 batch 饥饿。
WFQ 以精确标签提供 tenant 加权共享，并通过最大等待提升保护低权重 tenant
和 batch。调度器不会跳过 tenant 队首，也不会自行判断请求是否满足资源预算；
后续调度循环必须将选择结果交给准入控制器。
