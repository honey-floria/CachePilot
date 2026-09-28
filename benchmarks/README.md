# 基准测试

## Colab Phase 1 完整流程

打开 [`colab_phase1_matrix.ipynb`](../notebooks/colab_phase1_matrix.ipynb)，选择单 GPU，
将包含本次代码与 `.git` 的仓库放到 Drive，修改首个单元的 `SOURCE` 后顺序执行。
此 notebook 覆盖 TODO 2.3 的四项验证；`colab_acceptance.ipynb` 保留基础接入用途。

流程为：独立 Python 3.13.15 环境 → Phase 0/API 回归 → 四策略渐进扫描 →
每策略三轮（重启、warm-up、正式测量）→ 矩阵校验 → 五项真实 GPU 故障 →
Sim 原始证据 → Phase 1 子门禁 → 四项结论与 ZIP 归档。
锁定 Python/依赖不可下载时直接判环境阻塞，不能绕过版本约束。

- `progressive/*`：每点三次的安全点、保护/OOM 观测与原始请求；不外推容量。
- `matrix/*`：12 个 warm-up + 12 个 measured run，各含协议四件套与
  `observations.json`（实际发送/结束时间、窗口吞吐、完整 query/ledger）。
- `chaos-control.json`：原有确定性故障检查；`chaos-gpu.json`：真实 HTTP、模型与
  CUDA allocator 的受控故障、唯一终态、资源回收和恢复请求。
- `sim/*`：实际 SimExecutor 事件及协议数据，明确标注 simulated。
- `comparison-rows.json`、`conclusions.json`、`report.md`：逐次结果和最终结论。
- `source.json`、`source-snapshot/`、依赖/环境日志与 `checksums.json`：复现依据。

CUDA OOM 通过 model.forward 内超容量分配触发，不伪装成自然负载容量边界。
Torch 先完成整段生成再交付 token，TTFT/TPOT 仅能解释为 Gateway 交付指标；
物理 prefix 命中仍不可观测。prefix-aware 没有逻辑命中证据时标记
`INCONCLUSIVE`，Phase 1 不通过；static/continuous 对照为 N/A。
未配置 GPU 小时价格时成本为 `null`，并标记为估算。

`benchmarks.colab_phase1` 提供 Sim 和真实 GPU 故障辅助命令，后者必须独占实验服务：

```bash
python -m benchmarks.colab_phase1 sim --reference runs/<measured-run> --output runs/<sim-run>
python -m benchmarks.colab_phase1 gpu-chaos --dtype float16 --output runs/<new-chaos-report>.json
```

故障辅助命令检查 Torch 生成线程锁以确认后台生成结束，因此与本仓库
TorchExecutor 实现绑定。显存检查使用 allocated bytes 和 16 MiB 容差，不要求
NVIDIA 驱动显示的进程占用归零；模型权重应继续驻留以完成恢复请求。

## 实验协议

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

## 必要策略矩阵

GPU 服务器上完成一组矩阵后，用 `strategy_matrix.py` 做最终验收。每个策略至少保存
三次非 warm-up run，并额外保存至少一次 `warmup=true` run；所有 run 必须使用相同的
单卡、模型/tokenizer revision、Torch/Transformers、服务代码 commit 和 trace。脚本会
拒绝混入不同硬件或执行器的结果，也会检查 Strict/Adaptive、FCFS/WFQ 和
prefix-blind/prefix-aware 三组差异：

```bash
python -m benchmarks.strategy_matrix --metric ttft_ms \
  --output runs/required-matrix/matrix.json \
  runs/required-matrix/strict-fcfs-0 \
  runs/required-matrix/strict-fcfs-1 \
  runs/required-matrix/strict-fcfs-2 \
  runs/required-matrix/adaptive-fcfs-0 \
  runs/required-matrix/adaptive-fcfs-1 \
  runs/required-matrix/adaptive-fcfs-2 \
  runs/required-matrix/strict-wfq-0 \
  runs/required-matrix/strict-wfq-1 \
  runs/required-matrix/strict-wfq-2 \
  runs/required-matrix/prefix-blind-0 \
  runs/required-matrix/prefix-blind-1 \
  runs/required-matrix/prefix-blind-2 \
  runs/required-matrix/prefix-aware-0 \
  runs/required-matrix/prefix-aware-1 \
  runs/required-matrix/prefix-aware-2
```

当前 `TorchExecutor` 的能力声明是单请求 batch，因此不生成 static-vs-continuous
结论；只有执行器能力矩阵明确声明同时支持两种 batch 语义时，才应新增该 pair。

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

单机运行中可把在线指标快照和已完成 run 一并导出为诊断包：

```bash
make ops-export \
  BASE_URL=http://127.0.0.1:8000 \
  RUN_DIR=runs/<run_id> \
  OUTPUT_DIR=artifacts/<run_id>-incident
```

输出包含原始 `metrics.prom`、带 SHA-256 的 `snapshot.json`、重新校验生成的
`summary.json` 和可读 `report.md`。处置步骤及指标解释见[单机运行手册](../doc/RUNBOOK.md)。
