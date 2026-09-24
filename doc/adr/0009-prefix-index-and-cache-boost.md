# ADR-0009：逻辑 Prefix Index 与公平 Cache Boost

## 状态

已接受，2026-09-24。

## 决策

逻辑 prefix key 必须同时包含以下隔离维度：

```text
tenant_id
model_id
model_revision
tokenizer_revision
quantization_config
tokenized_prefix
```

任一作用域字段不同都视为不同缓存空间。查询只在完全相同的作用域中，
从完整请求 token 序列查找最长已记录前缀。索引不保存 prompt 文本、物理
KV handle 或执行器地址。

Prefix Index 只能返回逻辑命中。`PrefixLookupResult.physical_hit` 固定为
`None`，表示执行器尚未提供可验证信号；逻辑匹配不能上报为物理命中。

`PrefixAwareWFQScheduler` 在同一 priority 类别的 tenant 队首之间计算：

```text
cache_score = virtual_finish - cache_hit_tokens / tenant_weight
```

该分数只决定本次选择，不修改原始 WFQ 虚拟完成标签。Cache candidate 只有
在确实越过基线 WFQ 请求时才计作一次 boost，并同时受三项边界保护：

1. 连续 boost 不超过 `max_consecutive_boosts`；
2. 基线请求等待达到 `max_bypass_wait_ns` 后不得再被 cache 绕过；
3. 基线 tenant 的历史服务份额低于 `min_tenant_share` 时不得被绕过。

WFQ 原有的 `max_starvation_ns` 保护优先级更高，并且可以跨 interactive/batch
类别提升；cache boost 不跨 priority 类别。RuntimeLoop 只把逻辑命中 token
数量传给 Scheduler，不据此减少 KV reservation，也不产生物理命中指标。

## 后果

相同 token 前缀不会跨 tenant、模型、revision、tokenizer 或量化配置命中。
Cache affinity 可以减少潜在重复 prefill 的排序成本，但不能永久压制未命中
tenant。物理 KV 复用仍需未来执行器显式核验。
