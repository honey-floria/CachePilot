# CachePilot

CachePilot 是一个面向**单 GPU、多租户混合负载**的 LLM 推理服务控制层。它建立在
PyTorch、vLLM 等执行器之上，负责请求生命周期、KV 容量准入、租户公平调度、
流式输出、可观测性、成本归因和可复现实验。

项目只支持一个服务实例绑定一张 GPU。多 GPU 副本、跨卡路由、张量并行、
Prefill/Decode 拆分和 KV 跨卡传输均不在当前范围内。

> **当前状态：Phase 0 已完成，Phase 1 正在实施。** 仓库已有确定性模拟运行时、
> KV 预算、Strict/Adaptive Admission、FCFS/WFQ、Prefix Index、FastAPI Gateway
> 和实验分析器；真实 Torch/vLLM 单卡执行器、完整流控与生产观测仍待完成。

## 项目目标

```text
CPU 确定性模拟与契约验证
              ↓
单 GPU 真实推理服务与对照实验
```

| 阶段 | 必须交付 | 验证环境 |
|---|---|---|
| Phase 0：模拟核心 | 状态机、KV 逻辑预算、FCFS/WFQ、确定性回放 | 本地 CPU |
| Phase 1：单卡服务 | OpenAI 风格 API、真实生成、取消、观测、成本和对照实验 | Colab 或单 GPU 主机 |

## 核心架构

```text
Client
  │
  ▼
Gateway ──→ Registry ──→ Admission / Scheduler ──→ Single Executor ──→ SSE
                              │                         │
                              ├─ KV Planner             └─ Torch / vLLM
                              └─ Prefix Index
                                        │
                                        ▼
                              Metrics / Trace / Ledger
```

- `Gateway` 负责契约校验、tenant 身份、普通/流式响应、查询与取消；
- `Registry` 负责唯一终态、幂等事件和一次性资源释放；
- `Admission/Scheduler` 负责单卡容量保护、FCFS/WFQ 和有限 prefix boost；
- `SimExecutor` 负责 CPU 确定性验证，`TorchExecutor` 负责 Transformers 单请求真实生成，`VllmExecutor` 只适配 vLLM 的流式生成、取消和 usage；
- 推理引擎拥有物理 KV、GPU kernel 和内部 continuous batching，CachePilot 不重复实现。

## 完成标准

- 普通与流式 API 可用，取消、超时、断连和异常均产生唯一终态；
- 所有终态路径释放逻辑 KV reservation，不出现负数、重复释放或悬挂请求；
- 单 GPU 容量由 active sequences、batch tokens 和 KV blocks 三重预算保护；
- 固定 workload 与 seed 可重放，并保存逐请求原始记录；
- 完成 Strict/Adaptive、FCFS/WFQ、prefix-blind/prefix-aware 对照；
- 报告 TTFT、TPOT、吞吐、P99、公平性、拒绝率、KV 峰值和估算成本；
- 性能结论固定模型、执行器、GPU、驱动、CUDA、配置和重复次数。

## 快速开始

项目固定使用 Python `3.13.15`。在 CPU 环境中执行：

```bash
python3 -m pip install -r requirements/requirements-cpu.txt
make check
make phase0-exit
```

启动当前 FastAPI Gateway：

```bash
make serve
```

当前默认后端仍是确定性 CPU 开发后端，只用于 API、SSE 和生命周期验收，不代表真实
GPU 推理；TorchExecutor 或 VllmExecutor 需要安装各自锁定依赖并显式注入 Gateway。Google Colab 统一运行
[`notebooks/colab_acceptance.ipynb`](notebooks/colab_acceptance.ipynb)，该 notebook
同时覆盖仓库测试、Gateway/SSE、tenant 隔离、取消、真实 TCP HTTP 探活、
TorchExecutor 单请求以及可选的真实 vLLM stream/abort/usage 验收。

## 项目结构

```text
cachepilot/
  gateway/       API、请求契约与生成后端边界
  runtime/       状态机、Registry、准入、调度和运行循环
  cache/         KV 规划与租户隔离的 Prefix Index
  executors/     确定性模拟器及后续单卡执行器
  telemetry/     指标、Trace 与成本账本
workloads/       固定工作负载和生成器
benchmarks/      实验校验、分析和阶段验收
tests/           unit、property、contract、integration、load、chaos
doc/             设计、TODO、ADR 和验收记录
```

## 文档

- [工程设计](doc/DESIGN.md)：单卡边界、架构、不变量、指标与验收。
- [实施 TODO](doc/TODO.md)：Phase 0–1 的任务、资源和风险。
- [项目结构](doc/PROJECT_STRUCTURE.md)：目录与文件职责。
- [实验协议](doc/adr/0005-experiment-protocol.md)：trace、manifest、原始记录和汇总规则。
- [单机运行手册](doc/RUNBOOK.md)：KV/延迟/worker/成本故障处置与现场导出。
- [Phase 0 验收](doc/acceptance/0005-phase0-exit.md)：CPU 内核的可执行验收证据。

接口包括 `/v1/chat/completions`、`/v1/requests/{id}`、
`/v1/requests/{id}/cancel`、`/healthz`、`/readyz` 与 `/metrics`。正式支持范围以
OpenAPI 和契约测试为准。
