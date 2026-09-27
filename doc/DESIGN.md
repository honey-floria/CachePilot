# CachePilot 单 GPU 工程设计

**状态：** Phase 0 已完成，Phase 1 实施中
**目标：** 从 CPU 可验证内核演进为单 GPU、多租户、可观测且可复现的 LLM 推理服务。

---

## 1. 定位与边界

CachePilot 是建立在 PyTorch、vLLM 等执行器之上的服务控制层。它回答的是：

- 请求是否有效、属于哪个 tenant、何时进入执行；
- 单张 GPU 的 active sequences、batch tokens 和 KV blocks 是否安全；
- 混合长度与不同优先级请求如何公平共享单卡；
- 取消、超时、断连和执行器异常后如何得到唯一终态并回收资源；
- 延迟、吞吐、公平性、缓存收益和估算成本如何被复现和解释。

CachePilot 不实现 attention、CUDA kernel、模型并行、物理 KV allocator，也不在
vLLM 外重复实现同一层 continuous batching。

### 1.1 单卡约束

一个服务实例只允许绑定一张 GPU，并只维护一个真实执行器。CPU 的 `SimExecutor`
用于开发和确定性验证，不改变这一部署模型。

以下能力明确排除：

- 多 GPU 副本与跨卡请求路由；
- Prefill/Decode 分离与 KV handoff；
- 张量并行、流水线并行和跨节点服务；
- Worker Directory、跨卡健康探测与故障迁移；
- 多模型路由与 Kubernetes 控制器。

## 2. 目标函数与不变量

### 2.1 目标

| 维度 | 指标 |
|---|---|
| 延迟 | TTFT、TPOT、queue/prefill/decode、P50/P95/P99 |
| 吞吐 | requests/s、prompt tokens/s、completion tokens/s |
| 公平 | tenant 吞吐份额、Jain 指数、最大等待时间 |
| 容量 | active sequences、reserved/used/free KV blocks、峰值显存 |
| 可靠 | 拒绝率、取消率、超时率、失败率、资源回收时间 |
| 成本 | 估算 GPU 秒、每请求成本、每 1K token 成本 |

### 2.2 硬不变量

```text
active_sequences <= max_active_sequences
batched_tokens <= max_batched_tokens
reserved_kv_blocks <= usable_kv_blocks - safety_blocks
tenant_usage <= tenant_quota
gpu_count in {0, 1}
```

- 终态不可逆，每个请求只产生一个终态；
- 每个 reservation 和物理 handle 最多释放一次；
- 无效请求不进入 Registry，不占 KV；
- 取消、超时、断连、完成和执行器失败竞争时，以第一次成功的原子终态迁移为准；
- 逻辑 prefix 命中不能冒充执行器物理 KV 命中；
- 模拟结果不能冒充真实 GPU 性能数据。

## 3. 阶段范围

```text
Phase 0  CPU：契约、状态机、KV 预算、调度、模拟器、回放器
   ↓
Phase 1  单 GPU：真实执行器、流控、观测、成本、对照实验
```

| 阶段 | 出口条件 | 环境 |
|---|---|---|
| 0 | CPU 测试、资源不变量和固定 trace 重放通过 | 普通 CPU |
| 1 | API/SSE/取消/超时通过，至少一个真实执行器完成单卡实验 | 一张 GPU |

## 4. 总体架构

```text
┌──────────────┐
│ Client / SDK │
└──────┬───────┘
       ▼
┌────────────────────────────────────────────────────────────┐
│ Gateway                                                    │
│ request validation · tenant · idempotency · SSE · cancel  │
└──────┬─────────────────────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────────────────────┐
│ Single-GPU Control Plane                                   │
│ Registry · Deadline · Admission · FCFS/WFQ · Prefix boost │
└──────┬───────────────────────┬─────────────────────────────┘
       │                       │
       │                ┌──────▼──────────────┐
       │                │ KV Planner / Index │
       │                └──────┬──────────────┘
       ▼                       │
┌──────────────────────────────▼─────────────────────────────┐
│ Single Executor                                            │
│ SimExecutor (CPU) or TorchExecutor / VllmExecutor (1 GPU) │
└──────┬─────────────────────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────────────────────┐
│ Metrics · Trace · Request Ledger · Benchmark Artifacts    │
└────────────────────────────────────────────────────────────┘
```

| 平面 | 组件 | 职责 |
|---|---|---|
| 接入 | Gateway | 校验、身份、幂等、普通/SSE 响应、查询和取消 |
| 控制 | Registry、Admission、Scheduler | 生命周期、容量、公平性和执行时机 |
| 元数据 | KV Planner、Prefix Index | 逻辑容量、隔离作用域和命中候选 |
| 执行 | 单个 Executor | token 生成、内部 batching 和物理 KV |
| 观测 | Metrics、Trace、Ledger | 性能、错误、资源和成本证据 |

## 5. API 与生命周期

### 5.1 接口

```text
POST /v1/chat/completions
POST /v1/requests/{id}/cancel
GET  /v1/requests/{id}
GET  /healthz
GET  /readyz
GET  /metrics
```

`POST /v1/chat/completions` 支持普通响应、`stream=true`、tenant 身份、优先级、
deadline、idempotency key 和 `max_tokens`。正式字段以 OpenAPI 和契约测试为准。

### 5.2 状态机

```text
RECEIVED → TOKENIZED → QUEUED → ADMITTED → ROUTED → EXECUTING → FINISHED
任一非终态 ───────────────────────→ CANCELLED / TIMED_OUT / REJECTED / FAILED
```

`ROUTED` 在单卡设计中表示请求已绑定到本实例唯一执行器，不代表跨 Worker 选路。
保留该状态是为了明确 admission 与 executor ownership 的交接点。

### 5.3 并发与资源所有权

- `RequestRegistry` 的锁保护请求表和幂等索引；
- 每个状态机的锁保护状态、事件 ID 与 token 计数；
- `ResourceLeaseManager` 的锁保护 KV reservation 与物理 handle；
- 终态迁移成功的一方负责触发幂等资源释放；
- Executor 异常时，所有未完成请求进入明确终态，禁止继续输出 token。

## 6. 单 GPU Runtime

### 6.1 KV Planner

```text
estimated_blocks = ceil((prompt_tokens + expected_output_tokens) / block_size)
KV bytes/token = layers × 2 × kv_heads × head_dim × dtype_bytes
```

CachePilot 只管理逻辑 block。只有显式提供可用 KV 字节预算时，才能推导容量；
不得直接用 GPU 总显存声称真实 KV 容量。

| 策略 | 规则 | 取舍 |
|---|---|---|
| Strict | 按 `prompt + max_new_tokens` 预留 | 安全但可能浪费 |
| Adaptive | 历史输出 P95 + 安全余量 | 利用率更高，需要硬增长上限 |

Adaptive 在样本不足时回退 Strict；实际生成超过预测时必须重新申请，申请失败则停止
增长并产生可解释终态，不能越过硬容量。

### 6.2 调度

队列按 `interactive`、`batch` 和 tenant 子队列组织：

| 策略 | 用途 |
|---|---|
| FCFS | 稳定、可解释的到达顺序基线 |
| WFQ | 按 tenant 权重公平共享单卡 |
| Prefix-aware WFQ | 在 WFQ 上增加有限 cache boost |

Prefix boost 受最大连续提升、最大等待时间和 tenant 最小份额保护，不能为了提高命中率
让其他 tenant 饥饿。

### 6.3 Runtime Loop

```python
while executor.is_healthy():
    complete_finished_requests()
    reclaim_cancelled_and_timed_out_requests()
    update_kv_accounting()
    batch = scheduler.select_admissible_requests(
        active_sequence_budget=AVAILABLE_SEQUENCE_SLOTS,
        token_budget=AVAILABLE_BATCH_TOKENS,
        kv_budget=AVAILABLE_KV_BLOCKS,
    )
    executor.advance(batch)
    gateway.publish_tokens(batch)
```

每轮必须先完成和回收，再接纳新请求。仅限制并发数无法覆盖长上下文和大输出导致的
KV/计算差异，因此三重预算缺一不可。

### 6.4 执行器边界

| 执行器 | 用途 | Batch 所有者 | 性能结论 |
|---|---|---|---|
| SimExecutor | CPU 回归、故障注入、确定性重放 | CachePilot | 仅模拟 |
| TorchExecutor | 小模型功能正确性、可选教学型 batching | CachePilot 或无 batching | 仅在明确限制下 |
| VllmExecutor | 单卡真实性能实验 | vLLM | 可报告实测 |

`VllmExecutor` 只适配 prompt、stream、abort、usage 和指标，不在外层重做 continuous
batching。否则会形成双重队列并破坏延迟归因。

## 7. Prefix Cache 与成本

### 7.1 隔离键

```text
prefix_key = hash(
  tenant_id + model_revision + tokenizer_revision + quantization_config
  + tokenized_cacheable_prefix
)
```

- **逻辑命中**：Prefix Index 找到兼容候选；
- **物理命中**：执行器确认实际复用了 KV blocks；
- 物理信号不可用时记录 `null`，不得从逻辑命中推断。

### 7.2 成本

```text
request_cost = gpu_hour_price × attributed_gpu_seconds / 3600
```

成本按 prefill/decode token 与执行时间估算。报告必须同时给出 P99、公平性、拒绝率和
配置，避免只优化账单而牺牲服务质量。

## 8. 可观测性

### 8.1 指标

```text
cachepilot_requests_total{tenant, model, state}
cachepilot_requests_started_total
cachepilot_ttft_seconds
cachepilot_tpot_seconds
cachepilot_queue_wait_seconds
cachepilot_prefill_seconds
cachepilot_decode_seconds
cachepilot_active_sequences
cachepilot_logical_kv_blocks{state="reserved"}
cachepilot_admission_total{status, reason}
cachepilot_errors_total{code, stage}
cachepilot_prefix_events_total{kind="logical|physical", outcome}
cachepilot_executor_healthy
cachepilot_estimated_cost_total{tenant, model, stage}
```

Prometheus label 不包含 request ID、prompt 或 prefix key。逐请求信息进入结构化 trace
和 ledger，而不是高基数 metrics。

Gateway 的逐请求 trace 记录 request/tenant/model、准入状态与原因、prompt token 数（不含
prompt 内容）、queue/prefill/decode/TTFT/TPOT/total 阶段耗时、逻辑 KV block、终态和
错误代码；Prometheus 只保留 model、tenant、state、reason、stage 等受控低基数 label。

### 8.2 Trace

```text
gateway.receive
  → tokenize
  → prefix.lookup
  → kv.plan
  → scheduler.wait
  → admission
  → executor.submit
  → prefill
  → decode
  → stream.finish
  → ledger.record
```

请求账本至少记录 request/tenant/model、token 数、queue/prefill/decode/total、逻辑与物理
命中、reserved/peak blocks、估算 GPU 秒、成本、策略版本和终态。

请求账本使用 `RequestLedgerRecord` 保存终态请求的原始阶段耗时和派生字段。成本口径固定为：

```text
estimated_gpu_seconds = (prefill_ms + decode_ms) / 1000
estimated_cost = gpu_hour_price × estimated_gpu_seconds / 3600
```

`cost_is_estimate` 必须为 `true`，`cost_basis` 必须说明上述 wall-time 估算口径；GPU 小时价格
未配置时仍记录 `estimated_gpu_seconds`，但 `estimated_cost` 为 `null`。账本的
`recalculate_estimated_cost` 从原始字段重算成本，不能使用 Prometheus 聚合值或逻辑命中
推断物理命中。当前 Gateway 未接入 Prefix Index 时，`logical_hit_source` 为
`not_configured`，物理命中为 `null` 且来源为 `unobservable`。

## 9. Benchmark

### 9.1 工作负载

| Workload | 目的 |
|---|---|
| Uniform | 基准吞吐 |
| Mixed-length | 长短混合下的 P99 与 batching |
| Burst | 准入和过载保护 |
| Noisy-neighbor | tenant 公平性 |
| Shared-prefix | prefix-aware 调度收益 |
| Cancellation-heavy | 取消和回收正确性 |
| Long-context | 单卡 KV 压力边界 |

### 9.2 必做对照

1. Strict vs Adaptive reservation；
2. FCFS vs WFQ；
3. Prefix-blind vs prefix-aware；
4. TorchExecutor 确实实现两种 batching 后，才比较 static vs continuous。

所有对照固定 trace、seed、模型 revision、执行器版本和唯一 GPU；warm-up 后至少重复
三次。CPU 模拟与单 GPU 实测必须分开报告。

### 9.3 单卡协议约束

- `gpu_count` 只能是 `0`（模拟）或 `1`（实测）；
- `gpu_count=1` 时必须记录 GPU、显存、driver、CUDA 和单卡拓扑信息；
- 检测到两张及以上可见 GPU 时，实验入口应提前失败；
- 每个 run 保存 manifest、trace、requests 和分析器生成的 summary。

## 10. 可靠性与安全

- Gateway、Registry 和状态机操作保持幂等；
- 单执行器不健康时 readiness 失败，并停止接纳新请求；
- 取消、超时、断连、OOM 和执行器异常都必须回收 reservation；
- API key 不写日志，prompt 默认不进入 metrics；
- Prefix/KV 元数据默认 tenant-scoped；
- 每个 tenant 设置并发、token、队列长度和速率限制；
- 每次资源释放写入可审计事件。

运行手册至少覆盖 KV pressure、TTFT/TPOT 回退、执行器 unhealthy、OOM 和 cost spike。

## 11. 技术栈与结构

| 领域 | 选择 |
|---|---|
| 语言 | Python 3.13.15 |
| API | FastAPI、SSE |
| 执行器 | SimExecutor、PyTorch、vLLM |
| 元数据 | 内存；需要持久化时使用 SQLite |
| 观测 | Prometheus、结构化日志、逐请求 trace |
| 部署 | 本地进程或单 GPU Docker Compose |
| 压测 | asyncio trace replay |

```text
cachepilot/
  gateway/       api.py, contracts.py, backends.py
  runtime/       state_machine.py, registry.py, admission.py,
                 scheduler.py, deadlines.py, loop.py
  cache/         kv_planner.py, prefix_index.py
  executors/     sim.py, torch_executor.py, vllm_executor.py
  telemetry/     metrics.py, tracing.py, cost_ledger.py
```

## 12. 工时与验收

| 阶段 | 预计小时 | 可交付结果 |
|---|---:|---|
| Phase 0：模拟内核 | 70–100 | 状态机、模拟器、KV 预算、调度和回放 |
| Phase 1：单 GPU 服务 | 120–180 | API、真实执行器、观测、成本和单卡实验 |

总投入预计 **190–280 小时**。

### 最终验收清单

- [ ] OpenAI-compatible 普通与流式 API；
- [ ] 取消、超时、断连、失败与一次性资源回收；
- [ ] SimExecutor 与至少一个真实单卡执行器；
- [ ] Strict/Adaptive KV reservation；
- [ ] FCFS、WFQ 与 Prefix-aware WFQ；
- [ ] Prometheus、trace 和成本账本；
- [ ] 三组必要对照的原始数据与报告；
- [ ] 单 GPU 启动、压测、故障处理和运行手册；
- [ ] 实验协议拒绝 `gpu_count > 1`；
- [ ] README、ADR 和已知限制与实际能力一致。

## 13. 项目描述

> CachePilot is a single-GPU, multi-tenant control layer for LLM serving. It combines
> KV-budgeted admission, fair scheduling, OpenAI-compatible streaming, request-level
> observability, failure-safe resource reclamation, and reproducible workload replay.
