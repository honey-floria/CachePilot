# ADR-0011：执行器能力矩阵与指标语义门禁

## 状态

Accepted

## 决策

Sim、Torch 和 vLLM 的执行语义不同，不能只因为字段名相同就把指标放进同一组
比较。机器可读矩阵位于 `cachepilot/executor_capabilities.py`，各执行器通过
`capabilities` 类属性暴露同一份声明。

| 执行器 | Batch | 物理 KV | Prefix | 取消 |
|---|---|---|---|---|
| SimExecutor | CachePilot 模拟器拥有的逻辑 continuous batch | 不存在真实物理 KV；只维护逻辑 blocks | 只接受 CachePilot tenant 隔离的逻辑命中，不产生物理命中 | 模拟时钟内同步进入取消终态并释放逻辑资源 |
| TorchExecutor | 单请求；没有 batch | Transformers `generate` 内部 cache，不暴露物理 handle 或可靠占用 | 没有物理 prefix 复用；只允许报告控制层逻辑命中 | stopping criteria 协作停止，并禁止取消后继续交付 token |
| VllmExecutor | vLLM 拥有真实 continuous batching，CachePilot 不重做 | vLLM/PagedAttention 拥有，当前稳定接口不提供可核对的物理 KV 指标 | vLLM 可能具有内部机制，但当前适配器没有可靠命中信号，因此只报告逻辑命中 | 按原 request ID 转发 `abort`，并禁止 abort 后继续交付 token |

“拥有”与“可观测”分开记录。vLLM 拥有物理 KV 不代表 CachePilot 可以从私有对象
推导占用或命中；Torch 内部使用 generation cache 也不等于提供了物理 KV 遥测。

## 指标比较规则

- Sim 的 TTFT、TPOT、queue 和 total 来自逻辑时钟，不能与 Torch/vLLM 的单调
  墙钟测量混合；Torch 与 vLLM 的延迟单位和时钟语义一致时可以保留为并列原始值。
- 吞吐比较还必须具有相同 batch 语义；单请求 Torch、模拟 continuous batch 和
  vLLM continuous batch 不能被聚合成一个总体数字。
- `reserved_blocks_peak` 永远是 CachePilot 逻辑 reservation，不得改名或解释为
  GPU 物理 KV。
- `logical_hit` 是 tenant/version 隔离的控制层信号；`physical_hit` 只有执行器提供
  能力矩阵声明的可验证信号时才能为布尔值，并且原始记录必须同时写入完全匹配的
  `physical_hit_signal`。当前三个执行器都没有该信号，因此必须写 `null`。
- 取消指标只有取消机制语义相同时才能聚合；模拟终态、协作 stopping criteria 和
  vLLM abort 分属三种语义。

`benchmarks/analyze.py` 校验 manifest 中 software/strategy 的执行器名称一致，并在
当前能力不可观测时拒绝非空 `physical_hit`，信号与命中值不成对或来源不匹配时也
拒绝记录。汇总的 `prefix_observation` 会把物理命中计数保留为 `null`，并明确显示
`不可观测`，而不是把缺失信号解释成 miss。跨运行比较必须经过：

```bash
python -m benchmarks.compare \
  --metric ttft_ms \
  runs/run-a/summary.json \
  runs/run-b/summary.json
```

比较工具同时要求 trace、seed、模型和 tokenizer 版本一致；语义不一致时输出
`COMPARISON_INVALID`，不会计算或输出混合聚合值。

## 后果

- 能力声明成为代码、测试和实验比较共同使用的单一来源；
- 新执行器必须先声明能力，未知执行器不能进入 v1 分析或比较；
- 未来获得可靠物理 KV/prefix 信号时，必须更新能力声明和测试，不能仅修改报告。
