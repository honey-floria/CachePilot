# CachePilot

CachePilot 是一个面向多租户混合负载的 LLM 推理服务控制层。它建立在 vLLM 等现有推理引擎之上，重点解决请求生命周期、KV 容量准入、tenant 公平调度、单机多 GPU 路由、故障恢复、成本归因与可复现实验问题。

CachePilot 不重新实现 CUDA kernel、attention、模型并行或执行器内部的 continuous batching。模型执行和物理 KV 分配由推理引擎负责；CachePilot 管理进入执行器之前的策略和跨 worker 的控制逻辑。

> **当前状态：设计阶段。** 仓库目前只有设计文档和实施计划，没有可运行服务、安装包或已验证的性能数据。文档中的接口、指标和实验均为待实现目标。

## 项目目标

项目计划完成 Phase 0–2，形成一条可验证的工程主线：

```text
CPU 确定性模拟
        ↓
单 GPU 真实推理服务
        ↓
同机双 GPU replica routing 与故障恢复
```

| 阶段 | 必须交付 | 最低验证环境 |
|---|---|---|
| Phase 0：模拟核心 | 状态机、KV 逻辑预算、FCFS/WFQ、确定性回放 | 本地 CPU |
| Phase 1：单卡服务 | OpenAI 风格 API、真实生成、取消、观测与对照实验 | Colab 或单 GPU 主机 |
| Phase 2：双卡路由 | Worker Directory、replica pool、least-loaded/cache-aware 路由、故障恢复 | 同机两张兼容 GPU |

真实 Prefill/Decode（P/D）KV handoff 是 Phase 2 的可选加分项，不是项目完成条件。只有执行器提供经过验证的 KV transfer 接口，并且具备合适的双 GPU 环境时才实施；否则只允许提交协议模拟，并明确标注未完成真实传输。

分布式 KV cache、跨节点传输和 Kubernetes 控制器不在当前范围内。

## 工程问题

客户端通过 `POST /v1/chat/completions` 提交请求。Gateway 校验 tenant 和参数并处理流式输出；Registry 管理状态和取消；Admission/Scheduler 根据公平性、SLO 和逻辑 KV 预算决定何时接纳；Router 在 Phase 2 选择 worker；Executor 执行模型；Telemetry 记录等待、生成、资源与估算成本。

```text
请求 → Gateway → Registry → Admission/Scheduler → Router → Executor → SSE
                     ↘ KV Planner / Prefix Index ↗       ↓
                                      Metrics / Trace / Ledger
```

执行器边界必须清晰：

- `SimExecutor` 在 CPU 上验证调度、不变量和故障路径；
- `TorchExecutor` 用小模型验证真实 token 生成，可选实现由 CachePilot 管理的教学型 batching；
- `VllmExecutor` 用于真实性能实验，CachePilot 只管理外层队列、准入、路由和观测，不把 vLLM 内部 batching 或物理 KV 状态算作自己的实现；
- 逻辑 prefix 匹配与执行器实际复用 KV 必须分别计数。

## 完成标准

项目完成不是指“代码写完”，而是同时具备正确性证据和真实性能数据：

- API 能普通/流式返回，取消、超时、断连和异常均产生唯一终态；
- 所有终态路径最终释放逻辑 reservation，无负数、重复释放或悬挂请求；
- 固定 workload 和 seed 可重放，实验保存原始逐请求记录；
- 单卡完成 Strict/Adaptive、FCFS/WFQ 和 prefix-blind/aware 对照；
- 双卡完成 least-loaded 与 cache-aware routing 对照，以及 worker 故障、恢复和 drain 测试；
- 报告同时给出 TTFT、TPOT、吞吐、P99、公平性、拒绝率、KV 压力和估算成本；
- 所有性能结论标明模型 revision、执行器版本、GPU、驱动、CUDA、配置和限制。

## 开发顺序

1. 先完成 Phase 0 的 CPU 测试、状态机、资源不变量与回放器。
2. 再完成 Phase 1 的 API、至少一个真实执行器和单卡实验。Colab 只用于这一阶段。
3. Phase 1 通过出口验收后，再申请同机双 GPU 完成 Phase 2 replica routing。
4. 只有 replica routing 已稳定且 KV transfer 能力已验证，才尝试真实 P/D handoff。

## 文档

- [工程设计](doc/DESIGN.md)：系统边界、架构、协议、指标与阶段验收。
- [实施 TODO 与资源规划](doc/IMPLEMENTATION_TODO.md)：按依赖排序的任务、验收证据、硬件门槛和缩减路径。

## 预期仓库结构

```text
cachepilot/           gateway/ runtime/ routing/ cache/ executors/ telemetry/
workloads/            生成器与固定 trace
benchmarks/           回放、分析与实验报告
tests/                unit/ integration/ load/ chaos/
deploy/               单机 Compose 与监控配置
doc/                  设计、TODO、ADR、实验记录与运行手册
```

接口草案包括 `/v1/chat/completions`、`/v1/requests/{id}`、`/v1/requests/{id}/cancel`、`/healthz`、`/readyz` 与 `/metrics`。正式支持范围由 OpenAPI 和契约测试确定；未实现的字段、端点和执行器能力不得宣称兼容。
