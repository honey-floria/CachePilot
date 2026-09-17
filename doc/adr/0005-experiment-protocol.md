# ADR-0005：可复现实验协议

- 状态：已接受
- 日期：2026-09-18
- 依赖：ADR-0002、ADR-0003、ADR-0004

## 背景

如果只保存一张性能表，无法判断结果是否来自不同的模型快照、GPU、驱动、执行器策略或时钟口径。CachePilot 的实验必须既能回放，也能在分析阶段拒绝缺少关键上下文的结果。本 ADR 固定 Phase 0–2 共用的 trace、原始记录和汇总协议；机器可读定义位于 [`config/experiment.schema.json`](../../config/experiment.schema.json)。

## 实验运行单位

一次运行（run）由四个文件组成：

```text
<run_id>/manifest.json       # 一次运行的不可变元数据
<run_id>/trace.jsonl         # 输入请求轨迹
<run_id>/requests.jsonl      # 逐请求实际结果
<run_id>/summary.json        # 由分析器生成的汇总
```

`manifest.json`、`trace.jsonl` 和 `requests.jsonl` 是原始证据，不能用手工修改后的表格替代。`summary.json` 必须由 `python benchmarks/analyze.py` 从原始文件生成；报告应同时提交原始文件、命令行和生成的 summary。

## Trace schema

每行是一个 UTF-8、独立 JSON 对象，不能有空行或跨行 JSON。协议版本为 `trace_version: 1`。必填字段为：

| 字段 | 类型 | 语义 |
|---|---|---|
| `request_id` | string | 轨迹内唯一且不透明的请求 ID |
| `tenant_id` | string | 租户标识；用于公平性分组 |
| `arrival_ms` | non-negative integer | 从 trace 零点开始的相对到达时间 |
| `prompt_tokens` | non-negative integer | 固定 tokenizer/chat template 得到的输入 token 数 |
| `expected_output_tokens` | non-negative integer | 模拟器或压测器的期望输出长度 |
| `seed` | non-negative integer | 该请求的确定性随机种子 |

可选字段 `priority`（`interactive`/`batch`）、`max_new_tokens`、`prefix_group` 和 `cancel_after_ms` 也受 schema 约束。缺省 priority、max_new_tokens 的服务语义沿用 ADR-0002；workload 生成器必须在提交实验前把隐含默认值显式化。

## 时钟与时间口径

- `arrival_ms` 永远相对于 trace 零点，不是墙上时间。
- 服务事件使用单调时钟 `monotonic_ns`；所有持续时间（`queue_ms`、`ttft_ms`、`tpot_ms`、`total_ms`）统一换算为毫秒，保留至少三位小数。
- UTC 时间只用于 `created_at_utc` 等关联和审计，不能用于判断 deadline、排序事件或计算耗时。
- manifest 的 `clock` 必须精确为：`event_clock=monotonic_ns`、`duration_unit=ms`、`arrival_origin=trace_zero`、`wall_clock_role=metadata_only`。

## Seed 与重复规则

manifest 的 `seed` 是运行根 seed；trace 每行的 `seed` 是请求 seed。任何随机行为只能由这些 seed 和稳定的组件标签派生，禁止使用当前时间、进程 ID 或未记录的全局随机状态。

同一对照实验的所有策略必须使用同一份 trace、同一行级 seed 和相同的 `trace_id`。每个配置至少执行三次测量，并可先执行若干 warm-up；`warmup=true` 的运行只能用于预热，分析器不会把它计入 `measured_request_count`。`repetition_index` 从 0 开始且必须记录。

## Manifest 元数据

分析器要求以下元数据完整且非空：

- hardware：主机、平台、CPU、GPU、GPU 数量与显存、NVIDIA driver、CUDA 和 `nvidia-smi topo -m` 拓扑输出；
- software：CachePilot、Python、OS、executor、PyTorch、Transformers、vLLM 和 git commit；CPU profile 中未安装的 GPU 包必须明确写 `not_installed`，不能省略；
- model：模型 ID、40 位 revision、tokenizer revision、dtype、量化方式和 context limit；
- strategy：策略版本、executor、admission、scheduler、router 和 prefix mode。

版本字段不能使用 `latest`、`main`、`master`、`nightly` 或其他浮动别名。模型 revision 与 tokenizer revision 必须是 ADR-0003 所要求的固定 commit SHA。

## 逐请求记录与汇总

`requests.jsonl` 每行对应一个输入 request，记录终态（`FINISHED`、`CANCELLED`、`TIMED_OUT`、`REJECTED` 或 `FAILED`）、completion token、阶段耗时、worker、逻辑/物理 prefix 命中、reservation 峰值和估算 GPU 秒。无法观测的值写 `null`，不得用字符数、墙上时间或逻辑命中冒充物理数据。

`summary.json` 固定输出请求数、终态计数、TTFT/TPOT/queue/total 的 P50/P95/P99、完成 token 吞吐、拒绝率、取消/超时率和 Jain 公平性。分位数使用 `nearest_rank`，避免不同工具默认插值造成差异；空样本的指标为 `null`。汇总 schema 同样位于 `config/experiment.schema.json`。

其中 `request_count` 是原始逐请求记录总数，`measured_request_count`、终态计数、分位数和比率只包含非 warm-up 记录；这样预热数据仍可审计，但不会进入策略比较。

## 校验入口与验收

```bash
python benchmarks/analyze.py \
  --manifest runs/<run_id>/manifest.json \
  --trace runs/<run_id>/trace.jsonl \
  --requests runs/<run_id>/requests.jsonl \
  --output runs/<run_id>/summary.json
```

成功时输出 `EXPERIMENT_VALID` 并生成确定性 JSON；缺少任一关键 manifest 字段、浮动版本、非法时钟、重复 request ID、trace/结果集合不一致或 JSONL 记录不符合 schema 时，以非零状态退出并输出 `EXPERIMENT_INVALID`。因此未带硬件、软件、模型或策略版本的实验不能进入报告。

对照报告还必须注明：执行器能力（Sim/Torch/vLLM）、GPU 数量及拓扑、warm-up 与测量次数、是否包含拒绝/取消请求，以及不能观测物理 KV 时的限制。不得把模拟、单卡、PCIe 双卡和 NVLink 双卡结果合并成一个结论。

## 后果

- 原始逐请求数据和元数据成为实验的最小证据，结果可以在 CPU 环境先进行协议校验；
- 分析器不依赖第三方包，适合 Phase 0 安装基线；
- 任何新增字段或改变时钟、分位数、seed 语义的行为都必须提升 schema/ADR 版本，而不能悄悄改变旧结果含义。
