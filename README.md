# CachePilot

CachePilot 是一个面向多租户混合负载的 LLM 推理服务控制层。它建立在 vLLM 等现有推理引擎之上，重点解决请求生命周期、KV 容量准入、租户公平调度、单机多 GPU 路由、故障恢复、成本归因与可复现实验问题。

CachePilot 不重新实现 CUDA kernel、attention、模型并行或执行器内部的连续批处理。模型执行和物理 KV 分配由推理引擎负责；CachePilot 管理进入执行器之前的策略和跨 Worker 的控制逻辑。

> **当前状态：仓库骨架已建立并通过 Colab 验收。** 当前提交提供可安装的 CPU 优先包、严格契约测试、标准目录和仅用于探活的空服务；尚未实现完整运行时，也没有已验证的性能数据。文档中的接口、指标和实验均为待实现目标。

## 项目目标

项目计划完成 Phase 0–2，形成一条可验证的工程主线：

```text
CPU 确定性模拟
        ↓
单 GPU 真实推理服务
        ↓
同机双 GPU 副本路由与故障恢复
```

| 阶段 | 必须交付 | 最低验证环境 |
|---|---|---|
| Phase 0：模拟核心 | 状态机、KV 逻辑预算、FCFS/WFQ、确定性回放 | 本地 CPU |
| Phase 1：单卡服务 | OpenAI 风格 API、真实生成、取消、观测与对照实验 | Colab 或单 GPU 主机 |
| Phase 2：双卡路由 | Worker Directory、副本池、最少负载/缓存感知路由、故障恢复 | 同机两张兼容 GPU |

真实 Prefill/Decode（P/D）KV 交接是 Phase 2 的可选加分项，不是项目完成条件。只有执行器提供经过验证的 KV 传输接口，并且具备合适的双 GPU 环境时才实施；否则只允许提交协议模拟，并明确标注未完成真实传输。

分布式 KV 缓存、跨节点传输和 Kubernetes 控制器不在当前范围内。

## 工程问题

客户端通过 `POST /v1/chat/completions` 提交请求。Gateway 校验租户和参数并处理流式输出；Registry 管理状态和取消；Admission/Scheduler 根据公平性、SLO 和逻辑 KV 预算决定何时接纳；Router 在 Phase 2 选择 Worker；Executor 执行模型；Telemetry 记录等待、生成、资源与估算成本。

```text
请求 → Gateway → Registry → Admission/Scheduler → Router → Executor → SSE
                     ↘ KV Planner / Prefix Index ↗       ↓
                                      Metrics / Trace / Ledger
```

执行器边界必须清晰：

- `SimExecutor` 在 CPU 上验证调度、不变量和故障路径；
- `TorchExecutor` 用小模型验证真实 token 生成，可选实现由 CachePilot 管理的教学型批处理；
- `VllmExecutor` 用于真实性能实验，CachePilot 只管理外层队列、准入、路由和观测，不把 vLLM 内部批处理或物理 KV 状态算作自己的实现；
- 逻辑 prefix 匹配与执行器实际复用 KV 必须分别计数。

## 完成标准

项目完成不是指“代码写完”，而是同时具备正确性证据和真实性能数据：

- API 能普通/流式返回，取消、超时、断连和异常均产生唯一终态；
- 所有终态路径最终释放逻辑 reservation，无负数、重复释放或悬挂请求；
- 固定工作负载和 seed 可重放，实验保存原始逐请求记录；
- 单卡完成 Strict/Adaptive、FCFS/WFQ 和前缀无感知/前缀感知对照；
- 双卡完成最少负载与缓存感知路由对照，以及 Worker 故障、恢复和排空测试；
- 报告同时给出 TTFT、TPOT、吞吐、P99、公平性、拒绝率、KV 压力和估算成本；
- 所有性能结论标明模型修订版本、执行器版本、GPU、驱动、CUDA、配置和限制。

## 开发顺序

1. 先完成 Phase 0 的 CPU 测试、状态机、资源不变量与回放器。
2. 再完成 Phase 1 的 API、至少一个真实执行器和单卡实验。Colab 只用于这一阶段。
3. Phase 1 通过出口验收后，再申请同机双 GPU 完成 Phase 2 副本路由。
4. 只有副本路由已稳定且 KV 传输能力已验证，才尝试真实 P/D 交接。

## 文档

- [工程设计](doc/DESIGN.md)：系统边界、架构、协议、指标与阶段验收。
- [实施 TODO 与资源规划](doc/TODO.md)：按依赖排序的任务、验收证据、硬件门槛和缩减路径。
- [实验协议 ADR](doc/adr/0005-experiment-protocol.md)：trace、时钟、seed、原始记录和汇总 schema，以及分析器校验规则。
- [仓库骨架验收记录](doc/acceptance/0004-repository-skeleton.md)：CPU/Colab 安装、测试和空服务冒烟测试。

## 骨架安装与验收

项目固定使用 Python `3.13.15`。在全新 CPU 环境中执行：

```bash
python3 -m pip install -r requirements/requirements-cpu.txt
make check
make serve  # 另一个终端访问 /healthz、/readyz、/metrics
```

`make lint` 使用固定版本 Ruff，`make test` 使用固定版本 pytest。空服务只返回
健康/就绪/metrics 探活结果，不执行模型推理，也不代表 Phase 0–2 已完成。

Google Colab 验收可直接打开 [`notebooks/colab_acceptance.ipynb`](notebooks/colab_acceptance.ipynb) 笔记本，
依次运行固定 CPU 安装和 `python benchmarks/colab_acceptance.py`。脚本会检查
Python 基线、运行测试并探测空服务三个端点；该骨架验收已在 Colab 实际通过，成功输出
`COLAB_ACCEPTANCE=PASS`，且 `/healthz`、`/readyz`、`/metrics` 均返回 `200`。

## 预期仓库结构

```text
cachepilot/           gateway/ runtime/ routing/ cache/ executors/ telemetry/
workloads/            生成器与固定 trace
benchmarks/           回放、分析与实验报告
tests/                unit/ integration/ load/ chaos/
deploy/               单机 Compose 与监控配置
doc/                  设计、TODO、ADR、实验记录与运行手册
```

当前骨架已创建 `cachepilot/{runtime,routing,executors,telemetry}`、
`workloads/`、`benchmarks/`、`deploy/`、`tests/{unit,integration,load,chaos}/`
和 Colab 验收笔记本；具体运行时能力仍按 TODO 中的阶段顺序实现。

接口草案包括 `/v1/chat/completions`、`/v1/requests/{id}`、`/v1/requests/{id}/cancel`、`/healthz`、`/readyz` 与 `/metrics`。正式支持范围由 OpenAPI 和契约测试确定；未实现的字段、端点和执行器能力不得宣称兼容。
