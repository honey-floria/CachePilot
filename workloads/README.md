# 工作负载

固定工作负载和 trace（轨迹）由 `generator.py` 生成。协议与字段定义见
[ADR-0005](../doc/adr/0005-experiment-protocol.md) 及
`config/experiment.schema.json`。当前为七类 workload 各提交一个固定 seed 的
小型 JSONL 样例，供 CPU 回放和 Colab 冒烟测试验证路径和格式，不代表性能数据。

生成示例：

```bash
python workloads/generator.py mixed-length --seed 7 --count 12 \
  --output /tmp/mixed-length.jsonl
```

可用 workload：`uniform`、`mixed-length`、`burst`、`noisy-neighbor`、
`shared-prefix`、`cancellation-heavy` 和 `long-context`。生成器会检查
`trace_version`、必填字段、唯一 request ID、非递减 `arrival_ms`、seed 和
可选取消/优先级字段。

每行至少包含 `trace_version`、`request_id`、`tenant_id`、`arrival_ms`、
`prompt_tokens`、`expected_output_tokens` 和 `seed`。同一对照实验复用完全相同的
JSONL 和行级 seed；不得用当前时间或未记录的随机状态生成差异。
