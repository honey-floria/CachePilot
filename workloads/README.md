# 工作负载

固定工作负载和 trace（轨迹）将在 Phase 0 实现。当前提交一个最小 JSONL 样例，供
骨架与 Colab 冒烟测试验证路径和格式，不代表性能数据。

每行至少包含 `trace_version`、`request_id`、`tenant_id`、`arrival_ms`、
`prompt_tokens`、`expected_output_tokens` 和 `seed`。
