# 基准测试

实验协议已由 [ADR-0005](../doc/adr/0005-experiment-protocol.md) 固定，机器可读
schema 位于 `config/experiment.schema.json`。一次实验必须保留
`manifest.json`、`trace.jsonl`、`requests.jsonl` 和分析器生成的 `summary.json`。

分析原始记录：

```bash
python benchmarks/analyze.py \
  --manifest runs/<run_id>/manifest.json \
  --trace runs/<run_id>/trace.jsonl \
  --requests runs/<run_id>/requests.jsonl \
  --output runs/<run_id>/summary.json
```

缺少硬件、软件、模型或策略版本等关键元数据时，分析器会以非零状态退出；通过校验
不代表结果已经达到任何性能目标。`colab_acceptance.py` 仅验证 CPU 安装、测试和空服务
探活，不产生或宣称真实性能结果。

生成的 `summary.json` 同时包含：

- `request_timelines`：逐请求到达时间、终态、准入原因、阶段时间线和 reservation 峰值；
- `metrics`：queue、TTFT、TPOT、total 的 P50/P95/P99（`nearest_rank`）；
- 请求 trace 同时保留 `prefill_ms`、`decode_ms`、准入原因、逻辑 KV block 和错误代码；
- `resource_peaks`：逻辑 KV block 和估算 GPU 秒峰值；
- `prefix_observation`：逻辑命中计数，以及物理命中的可观测状态；执行器没有
  可验证信号时，计数为 `null` 并明确显示 `不可观测`；
- `throughput_completion_tokens_per_s`、`fairness_jain`、`rejection_rate` 和 `cancellation_rate`；
- `control_variables`：trace、seed、模型、executor、admission、scheduler 和 prefix 策略；
- `simulation`：明确标记 `simulated` 或 `measured`，不会把模拟结果误报为真实硬件结果。

对照 Strict/Adaptive 或 FCFS/WFQ 时，必须复用相同 `trace_id`、seed、模型版本和
executor；只改变 `control_variables.admission` 或 `control_variables.scheduler`，再比较
同一组指标。

跨运行比较先经过能力语义门禁：

```bash
python -m benchmarks.compare \
  --metric ttft_ms \
  runs/run-a/summary.json \
  runs/run-b/summary.json
```

Sim 逻辑时钟延迟与真实执行器延迟、不同 batch 语义的吞吐、不可观测的物理 KV/
prefix，以及不同取消机制的指标会返回 `COMPARISON_INVALID`。比较工具只保留各运行
原始汇总值和共同语义签名，不产生跨语义聚合值。

实验 manifest 的 `gpu_count` 只能为 `0`（CPU 模拟）或 `1`（单卡实测）；分析器会
拒绝任何多 GPU 记录。
