# 工作负载

固定工作负载和 trace（轨迹）将在 Phase 0 实现。协议与字段定义见
[ADR-0005](../doc/adr/0005-experiment-protocol.md) 及
`config/experiment.schema.json`。当前提交一个最小 JSONL 样例，供骨架与 Colab
冒烟测试验证路径和格式，不代表性能数据。

每行至少包含 `trace_version`、`request_id`、`tenant_id`、`arrival_ms`、
`prompt_tokens`、`expected_output_tokens` 和 `seed`。同一对照实验复用完全相同的
JSONL 和行级 seed；不得用当前时间或未记录的随机状态生成差异。
