# ADR-0008：Runtime Loop 三重预算与回收顺序

## 状态

已接受，2026-09-22。

## 决策

Phase 0 的单 worker `RuntimeLoop` 协调 FCFS/WFQ 与 `SimExecutor`。每轮严格
按照以下顺序执行：

1. 从上一轮执行器快照识别已经结束计算、取消或失败的请求；
2. 释放这些请求的 active slot 和完整 KV reservation；
3. 从执行器请求快照重建当前实际逻辑 KV block 账本；
4. 在 active sequence 和完整 KV reservation 均有空间时接纳等待请求；
5. 按轮转顺序为 active 请求分配本轮 token 工作额度；
6. 用逐请求额度推进 `SimExecutor`，更新实际 KV 与峰值并验证硬上限。

循环同时维护两种 KV 数值：

- `reserved_kv_blocks`：按 `prompt_tokens + output_tokens` 完整预留，用于决定
  新请求是否可以获得 active slot。它保证已接纳请求后续增长始终有空间，
  避免多个长请求占满当前 KV 后都无法继续增长的死锁。
- `kv_blocks`：从执行器快照汇总的当前实际占用，用于逐轮观测、峰值和硬
  上限断言。

每轮实际 prefill/decode token 总数不得超过 `max_batched_tokens`。若一个请求
的阶段额度大于剩余 token 或 KV 空间，循环会缩减该请求本轮额度，而不是突破
硬限制。active 请求采用轮转起点，避免列表首请求持续独占较小的 token 预算。

Scheduler 只决定等待请求取得 reservation 的顺序。策略选中的请求若暂时因
KV reservation 放不下，会保留为 deferred 队首，下一轮先重试；循环不会绕过
该决定接纳后续请求。

`STREAMING` 请求已完成计算且 SimExecutor 已释放 KV，因此 runtime loop 会回收
其 active slot 和 reservation；尚未交付的客户端缓冲仍由 SimExecutor 排空。

## 不变量

每轮结束必须同时满足：

```text
active_sequences <= max_active_sequences
batched_tokens <= max_batched_tokens
kv_blocks <= max_kv_blocks
reserved_kv_blocks <= max_kv_blocks
```

相同 Scheduler、请求序列、配置和逻辑时钟产生相同逐 tick 结果与峰值统计。
