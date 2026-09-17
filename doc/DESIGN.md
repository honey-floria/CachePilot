# CachePilot：多租户 LLM Serving 控制层工程设计

**版本：** v1.1  
**状态：** 项目设计基线  
**目标：** 从 CPU 可验证内核演进到单 GPU 服务和同机双 GPU replica runtime，完成可测试、可观测、可复现的多租户推理控制层。

---

## 1. 项目摘要

CachePilot 是一个面向多租户、混合请求负载的 LLM Serving 控制层。它对外提供 OpenAI-compatible API，对内管理请求生命周期、队列、公平调度、KV-cache 逻辑容量、prefix 感知、成本账本与可观测性，并在同机双 GPU 上实现多 worker 路由、容量隔离和故障恢复。

项目的工程边界很重要：CachePilot **不是**从零实现 attention kernel，也不是替代 vLLM/SGLang。推理引擎负责模型执行、CUDA kernel、连续 batching 与物理 KV allocator；CachePilot 负责控制层与运行时策略：

- 当前请求应接纳、排队、拒绝还是降级？
- 请求应路由到哪个 worker，才能兼顾缓存命中、负载和 SLO？
- 不同 tenant 如何公平共享昂贵 GPU？
- 多个同模型 worker 之间如何兼顾缓存亲和性、负载和故障隔离？
- 每个请求的延迟、显存、缓存收益和成本从何而来？

真实 Prefill/Decode（P/D）KV handoff 是条件允许时的扩展实验，不是项目完成条件。分布式 KV cache、跨节点服务和 Kubernetes 控制器不在当前项目范围内。

## 2. 核心问题与目标函数

LLM 在线推理的请求具有异构性：prompt 长度、输出长度、到达时间、优先级、共享上下文和取消概率不同；而 KV cache 会随生成增长并长期占用 GPU 显存。

CachePilot 的优化目标不是单个数字，而是带约束的优化：

```text
maximize: useful_tokens_per_gpu_second
subject to:
  interactive P99 TTFT <= SLO
  interactive P99 TPOT <= SLO
  tenant fairness >= configured threshold
  KV/GPU memory usage < safe capacity
  request cancellation and failures reclaim resources correctly
```

需要持续报告的主要指标：

| 类别 | 指标 |
|---|---|
| 用户体验 | TTFT、TPOT、端到端延迟、P50/P95/P99 |
| 性能 | tokens/s、prefill tokens/s、decode tokens/s、GPU 利用率 |
| 容量 | active sequences、KV reserved/used/free、OOM/拒绝率 |
| 公平性 | 各 tenant 延迟、服务份额、饥饿时间 |
| 成本 | GPU 秒、成本/请求、成本/输出 token、缓存节省 |
| 可靠性 | 取消回收时间、失败率、重试率、worker 健康度 |

## 3. 项目阶段与范围

```text
Phase 0  Simulator + Runtime Core
Phase 1  单 GPU 真实模型与完整可观测性
Phase 2  同机双 GPU：replica pool、cache-aware router、故障恢复
Optional 同机 P/D 协议与真实 KV handoff 实验
```

| 阶段 | 核心产出 | 可验证环境 |
|---|---|---|
| 0 | 状态机、模拟器、调度、KV 预算 | 无 GPU |
| 1 | 单卡服务、真实 benchmark、成本账本 | Colab/单 GPU VM |
| 2 | 双 worker replica、路由、容量隔离、故障恢复 | 同机 2×GPU |
| 可选 | P/D 协议模拟；能力验证通过后接入真实 KV handoff | 同机 2×GPU、兼容执行器 |

Phase 0–2 是当前项目边界。单卡是必须完成层；Phase 2 必须先完成 replica routing，不能为了追求 P/D 跳过单卡正确性或双 worker 基线。

### 3.1 仓库骨架与环境入口

仓库骨架阶段固定 Python `3.13.15`，CPU 安装入口为
`requirements/requirements-cpu.txt`，静态检查和测试入口分别为
`make lint` 与 `make test`，组合验收入口为 `make check`。当前
`cachepilot.runtime.empty_service` 只提供 `/healthz`、`/readyz` 和 `/metrics`
探活端点，作为全新 CPU 环境与 Google Colab 的安装/启动 smoke test；它不执行
模型推理，不计入 Phase 0–1 的功能或性能完成度。

可复现验收脚本位于 `benchmarks/colab_acceptance.py`， notebook 位于
`notebooks/colab_acceptance.ipynb`。脚本要求解释器与 ADR-0003 的精确 Python
基线一致，运行契约测试、启动空服务并检查三个端点，输出
`COLAB_ACCEPTANCE=PASS` 才算骨架验收通过。

---

## 4. 总体架构

```text
                       ┌────────────────────────────────┐
Clients ──────────────►│ API Gateway                    │
HTTP / OpenAI / SSE    │ Auth · Quota · Streaming       │
                       └──────────────┬─────────────────┘
                                      │
                       ┌──────────────▼─────────────────┐
                       │ Runtime Control Plane           │
                       │ Request Registry · Scheduler    │
                       │ Admission · SLO · Cost Policy   │
                       └───────┬───────────┬─────────────┘
                               │           │
                 ┌─────────────▼──┐  ┌─────▼──────────────┐
                 │ Prefix/KV Meta │  │ Worker Directory   │
                 │ Cache Index    │  │ health · capacity  │
                 └─────────────┬──┘  └─────┬──────────────┘
                               │           │
             ┌─────────────────▼───────────▼─────────────────┐
             │ Execution Plane                                │
             │ Single GPU worker / Replica workers             │
             │ Executor adapters / Optional P/D workers        │
             └─────────────────┬──────────────────────────────┘
                               │
             ┌─────────────────▼──────────────────────────────┐
             │ Observability                                   │
             │ Metrics · Traces · Cost Ledger · Benchmark DB   │
             └────────────────────────────────────────────────┘
```

### 4.1 控制面与数据面

| 平面 | 组件 | 作用 |
|---|---|---|
| 控制面 | Gateway、Registry、Scheduler、Router、Policy | 决定请求去哪、何时执行、是否拒绝 |
| 数据面 | 单卡/replica worker、vLLM executor | 实际执行模型与生成 token |
| 元数据面 | Prefix index、worker directory、KV metadata | 保存可复用上下文、容量和拓扑信息 |
| 观测面 | Prometheus、trace、成本账本 | 回答性能、故障和成本原因 |

---

## 5. API 与请求生命周期

### 5.1 对外接口

```text
POST /v1/chat/completions
POST /v1/requests/{id}/cancel
GET  /v1/requests/{id}
GET  /healthz
GET  /readyz
GET  /metrics
```

`POST /v1/chat/completions` 兼容 OpenAI 消息格式，支持 `stream=true`、tenant 身份、优先级、deadline、idempotency key 和可选的 `max_tokens`。

### 5.2 请求状态机

```text
RECEIVED → TOKENIZED → QUEUED → ADMITTED → ROUTED
                                            │
              ├──────────────────────────────→ EXECUTING → FINISHED
              │                                 （主路径）
              └→ PREFILLING → WAITING_FOR_DECODE → DECODING → FINISHED
                                （可选 P/D 路径）

任一非终态 → CANCELLED / TIMED_OUT / REJECTED / FAILED
```

### 5.3 状态机不变量

- 终态不可逆；
- 每个状态事件包含唯一 event id，支持幂等重放；
- 只有 `ADMITTED` 请求能占用逻辑 KV reservation；
- 每个请求只释放一次 reservation 和物理 KV handle；
- 客户端断连、worker 故障和正常结束竞争时，以第一次成功的原子终态转换为准；
- 终态请求不能再次输出 token。

---

## 6. 单 GPU Runtime（Phase 0–1）

### 6.1 单卡架构

```text
Client → Gateway → Registry → Admission/Scheduler → Single Worker → SSE
                              ↘ KV Planner / Prefix Index ↗
```

单卡版本要真实实现：

- OpenAI-compatible streaming、取消、超时与慢客户端保护；
- `SimExecutor` 下的 continuous batching；
- `TorchExecutor` 下的小模型真实 token 生成，batching 作为可选深化；
- `VllmExecutor` 下的外层准入、成本和观测；
- KV reservation、Strict/Adaptive 准入、FCFS/WFQ；
- prefix-aware 请求优先级与逻辑缓存统计；
- workload replay、dashboard 和可复现实验。

### 6.2 三类执行器

| 执行器 | 用途 | 谁拥有 batch 调度 |
|---|---|---|
| SimExecutor | 无 GPU 开发、确定性回归、故障注入 | CachePilot |
| TorchExecutor | 小模型功能正确性；可选教学型 batching loop | CachePilot 或无 batching |
| VllmExecutor | 单卡性能 benchmark | vLLM 内部；CachePilot 仅管外层策略 |

禁止在 `VllmExecutor` 外重复实现同一层 continuous batching；否则会有双重排队、错误的性能归因和不可控延迟。

### 6.3 KV Planner

CachePilot 不重写底层物理 allocator，而以 token block 进行逻辑容量规划：

```text
estimated_blocks = ceil((prompt_tokens + expected_output_tokens) / block_size)

KV bytes/token = layers × 2 × kv_heads × head_dim × dtype_bytes
```

接纳约束：

```text
active_sequences < max_active_sequences
estimated_blocks + reserved_blocks <= total_blocks - safety_blocks
tenant_active_tokens < tenant_quota
```

MVP 实现：

| 策略 | 规则 | 主要代价 |
|---|---|---|
| Strict reservation | 按 `max_new_tokens` 预留 | 容量浪费，但安全 |
| Adaptive reservation | 历史 P95 输出 + 安全余量 | 利用率更高，但受长尾影响 |

### 6.4 单卡调度

队列：`interactive_queue`、`batch_queue`、`overflow_queue`；每个 tenant 保留子队列。

| 策略 | 目的 |
|---|---|
| FCFS | 基线 |
| WFQ | tenant 按权重公平共享 GPU |

prefix-aware 只作为 WFQ 内有限加分：

```text
score = priority + cache_benefit - estimated_service_cost
        - virtual_finish_penalty - deadline_risk
```

保护机制：最大连续 cache boost、最大等待时间、最小 tenant 份额；P99 超过阈值时自动下调缓存优先级。

### 6.5 单卡 worker loop

```python
while worker.is_healthy():
    complete_finished_sequences()
    reclaim_cancelled_sequences()
    update_kv_accounting()
    batch = scheduler.select_admissible_requests(
        token_budget=MAX_BATCHED_TOKENS,
        kv_budget=AVAILABLE_KV_BLOCKS,
    )
    executor.prefill_or_decode(batch)
    gateway.stream_new_tokens(batch)
```

同时限制 `max_active_sequences`、`max_batched_tokens`、`max_kv_blocks`，不能仅用并发请求数代表资源消耗。

---

## 7. 多 GPU Runtime（Phase 2）

### 7.1 阶段目标

Phase 2 假设同一物理主机上的两张兼容 GPU。正式交付是 replica runtime，不追求超大模型并行：

1. 两个同模型 worker 如何注册、上报容量、drain 和恢复；
2. least-loaded 与 cache-aware routing 如何权衡命中、负载和公平性；
3. 取消、超时和 worker 失败如何正确传播并保持容量隔离。

只有 replica runtime 的正确性与实验完成后，才评估执行器是否支持真实 P/D KV handoff。

### 7.2 必做拓扑：Replica Pool

```text
                    ┌── GPU 0: colocated worker ──┐
Gateway → Router ───┤                               ├→ SSE
                    └── GPU 1: colocated worker ──┘
```

两个 worker 独立持有模型和 KV，各请求从开始到结束只归属一个 worker。重点验证负载均衡、prefix/cache affinity、tenant 隔离、worker draining 与故障恢复。

可选拓扑是 P/D Disaggregation：

```text
Gateway → Global Scheduler → GPU 0 Prefill Worker
                                   │ KV handoff
                                   ▼
                              GPU 1 Decode Worker → SSE
```

该拓扑只在执行器暴露稳定、可验证的 KV transfer 能力时实施。重新在 decode worker 上执行 prefill 不属于 KV handoff。

### 7.3 多卡 Router

每个 worker 定期报告：

```json
{
  "worker_id": "replica-gpu-1",
  "role": "replica",
  "model_revision": "...",
  "active_sequences": 28,
  "queue_depth": 7,
  "kv_used_blocks": 12500,
  "kv_free_blocks": 2500,
  "gpu_memory_used_mb": 39200,
  "prompt_tokens_per_second": 1250,
  "generation_tokens_per_second": 680,
  "healthy": true
}
```

路由约束：

- 目标 worker 必须模型、tokenizer、量化、KV layout 兼容；
- 优先选择拥有最长可用 prefix 的 worker；
- 若缓存亲和性导致严重负载不均衡，则向低负载 worker 回退；
- 不将接近 KV 极限的 worker 作为目标；
- worker draining 时只接收完成中请求，不接新请求。

可用评分：

```text
worker_score =
  α × prefix_hit_tokens
  - β × queue_delay_estimate
  - γ × kv_pressure
  - δ × error_penalty
```

### 7.4 可选 P/D handoff 协议

P/D 协议可以先由 SimExecutor 验证，但必须标记为协议模拟。真实接入时必须把“prefill 已完成”与“decode 已安全接管”分开：

```text
1. Scheduler 选择 prefill worker 与 decode worker
2. Prefill worker 创建 transfer_id 并计算 prompt KV
3. Prefill worker 写入 TransferReady(metadata, checksum, expiry)
4. Decode worker 拉取或接收 KV，校验兼容性与完整性
5. Decode worker 写入 TransferAccepted
6. Prefill worker 在收到确认前保留源 KV；确认后按策略释放
7. Decode worker 进入 DECODE_READY，开始流式生成
```

失败规则：

| 失败点 | 处理 |
|---|---|
| Prefill 失败 | 请求 `FAILED`，释放 prefill reservation |
| Transfer 超时 | 重试有限次数；之后 fallback 到 colocated worker 或失败 |
| Decode 拒绝 | prefill 保留源 KV 至 TTL；重新选择 decode worker |
| 客户端取消 | 取消 transfer、释放源/目的两端 reservation |
| Decode worker 崩溃 | 标记请求失败或从仍存在的源 KV 重新调度 |

必须记录：`prefill_ms`、`transfer_ms`、`decode_queue_ms`、`source_release_ms`、`transfer_bytes`。

### 7.5 Phase 2 完成边界

Phase 2 完成必须具备：真实双 GPU replica、worker 健康与 drain、容量隔离、路由对照和故障恢复证据。P/D 结果分为三级：

| 等级 | 可以声明的能力 |
|---|---|
| 未实现 | 不支持 P/D |
| SimExecutor 验证 | P/D 协议模拟 |
| 真实 KV transfer 及正确性验证 | 实验性 P/D handoff |

没有第三等级证据时，不得在 README、简历或报告中宣称实现了真实 P/D disaggregation。

---

## 8. 明确不在当前范围

以下能力不属于 Phase 0–2，也不作为当前项目完成或实习展示的前置条件：

- 多节点 KV cache 和 GPU → CPU → NVMe/远端分层缓存；
- 跨节点 KV transfer、网络存储与缓存一致性；
- Kubernetes CRD、控制器、自动扩缩容和 Helm 集群交付；
- 多模型路由、张量并行和超大模型部署；
- 自研 CUDA kernel、attention 和物理 KV allocator。

如果 Phase 0–2 已稳定、存在真实用户需求且具备长期硬件资源，可以另建设计文档评估这些方向；当前不预留未经验证的接口，也不把它们写入验收清单。

---

## 9. Prefix Cache 与成本优化

### 9.1 Prefix key

```text
prefix_key = hash(
  model_revision + tokenizer_revision + quantization_config
  + normalized_system_prompt + cacheable_prompt_prefix
)
```

### 9.2 逻辑命中与物理命中

- **逻辑命中**：Router/Scheduler 发现请求有可复用 prefix；
- **物理命中**：执行器实际复用了对应 KV blocks；
- 两者必须分开计数，不能以逻辑匹配虚报性能收益。

### 9.3 成本模型

```text
request_cost = gpu_hour_price × attributed_gpu_seconds / 3600
```

初版按 prefill/decode token 与 batch 执行时间加权归因。多卡阶段按 worker 分别统计 GPU 秒；可选 P/D 实验额外统计 transfer 时间和字节。所有结果标为估算成本，除非接入真实云账单。

### 9.4 成本策略

- 立即回收已取消请求的 KV reservation；
- prefix 命中时优先减少重复 prefill；
- 对低优先级 batch 使用较严格队列上限；
- cache-aware 路由不能为了命中率持续制造 worker 热点；
- 任何节省策略均需同时报告 P99、公平性和拒绝率，避免只优化账单。

---

## 10. 可观测性与数据模型

### 10.1 关键 Prometheus 指标

```text
cachepilot_requests_total{tenant, model, state}
cachepilot_ttft_seconds
cachepilot_tpot_seconds
cachepilot_queue_wait_seconds
cachepilot_active_sequences{worker}
cachepilot_kv_blocks{worker, state=reserved|used|free}
cachepilot_admission_total{decision, reason}
cachepilot_prefix_events_total{logical|physical, outcome}
cachepilot_worker_health{worker}
cachepilot_estimated_cost_total{tenant, model, stage}
```

可选 P/D 实验再增加 `cachepilot_transfer_seconds` 和 `cachepilot_transfer_bytes_total`，避免主线指标依赖未实现能力。

### 10.2 Trace 路径

```text
gateway.receive
  → tokenize
  → prefix.lookup
  → kv.plan
  → scheduler.wait
  → admission
  → router.select_worker
  → prefill
  → kv.transfer（仅可选 P/D）
  → decode
  → stream.finish
  → ledger.record
```

### 10.3 请求账本字段

```text
request_id, tenant_id, model_id, model_revision,
prompt_tokens, completion_tokens, received_at,
queue_ms, prefill_ms, decode_ms, total_ms,
selected_worker, prefix_key, logical_hit, physical_hit,
reserved_blocks, peak_blocks, estimated_gpu_seconds,
estimated_cost, policy_versions, terminal_state
```

可选 P/D 实验另加 `transfer_id`、`transfer_ms`、`transfer_bytes` 和 source/target worker 字段。

---

## 11. Benchmark 与评估方法

### 11.1 Workload 格式

```json
{
  "arrival_ms": 1200,
  "tenant_id": "team-a",
  "priority": "interactive",
  "prompt_tokens": 1024,
  "max_new_tokens": 256,
  "prefix_group": "shared-policy-v3",
  "cancel_after_ms": null
}
```

### 11.2 必备工作负载

| Workload | 目的 |
|---|---|
| Uniform | 基准吞吐 |
| Mixed-length | 长短请求混合下的 P99 和 batching |
| Burst | admission 与过载保护 |
| Noisy neighbor | tenant 公平性 |
| Shared-prefix | prefix/cache-aware routing |
| Cancellation-heavy | 回收正确性 |
| Long-context | KV 压力、准入和多 worker 负载偏斜 |

### 11.3 必做对照实验

1. Strict vs Adaptive reservation；
2. FCFS vs WFQ；
3. Prefix-blind vs prefix-aware；
4. Round-robin vs least-loaded router；
5. Cache-aware router vs least-loaded router；
6. 单卡基线 vs 双卡 replica，在报告中按 GPU 数归一化解释吞吐与成本。

只有 TorchExecutor 确实实现两种 batching 时，才增加 static vs continuous batching。只有真实 KV handoff 已通过正确性验证时，才增加 colocated/replica/P/D 对照。

### 11.4 实验纪律

- 固定模型 revision、量化、GPU、驱动、CUDA、executor 版本与随机种子；
- 每组至少重复三次，保留 warm-up；
- 记录硬件拓扑：`nvidia-smi topo -m`；
- 不将 SimExecutor、单卡、PCIe 双卡、NVLink 双卡混进同一结论；
- 报告收益，也报告因公平性、传输和尾延迟带来的负面结果。

具体的可执行实验协议（trace/JSONL/summary schema、单调时钟、seed 派生规则和
元数据完整性校验）见 [ADR-0005](adr/0005-experiment-protocol.md)。分析器拒绝缺少
硬件、软件、模型或策略版本的运行，避免把不可比较的结果写入同一报告。

---

## 12. 技术栈与仓库结构

| 领域 | Phase 0–1 | Phase 2 |
|---|---|---|
| 语言 | Python 3.13.15 | Python 3.13.15 |
| API | FastAPI、SSE | 内部 HTTP 或 gRPC worker transport |
| 执行器 | SimExecutor、PyTorch、vLLM | 两个独立 vLLM worker；P/D adapter 可选 |
| 元数据 | 内存/SQLite | Worker Directory 内存状态与持久化实验记录 |
| 观测 | Prometheus、结构化日志 | worker 级指标与故障事件 |
| 部署 | 本地进程、Docker Compose | 同机双 GPU Compose/启动脚本 |
| 压测 | asyncio trace replay | 同一回放器执行双卡对照 |

```text
cachepilot/
  gateway/          api.py, streaming.py, auth.py
  runtime/          request.py, state_machine.py, registry.py,
                    admission.py, scheduler.py, tenant_quota.py
  routing/          worker_directory.py, load_router.py, cache_router.py
  cache/            kv_planner.py, prefix_index.py
  executors/        base.py, simulator.py, torch_executor.py, vllm_executor.py
  telemetry/        metrics.py, tracing.py, cost_ledger.py
  workloads/        generator.py, traces/
  benchmarks/       replay.py, analyze.py, reports/
  deploy/           docker-compose.yml, grafana/
  tests/            unit/, integration/, load/, chaos/
  doc/              adr/, experiment-notes/, runbooks/
```

若进入可选 P/D，再增加 `pd_router.py` 与 `transfer_protocol.py`，不提前让主线依赖这些模块。

---

## 13. 可靠性、安全与运行手册

### 13.1 必做可靠性设计

- Gateway、Registry、Scheduler 的状态转换均幂等；
- worker 心跳失效后停止新请求路由；
- worker drain 时仅等待活跃请求完成或到 deadline，不接新请求；
- 每个请求始终具有唯一 worker 归属，失败迁移必须产生新的可追踪 attempt；
- 每次资源释放写入审计日志；
- 对每个 tenant 设置并发、token、队列长度和速率限制。

可选 P/D transfer 额外要求 TTL、checksum 和明确的 source/target ownership。

### 13.2 基本安全规则

- API key 不写入日志或 trace；
- prompt 内容默认不进入 metrics，日志只保存 hash/长度，除非显式开启脱敏采样；
- prefix/KV cache 默认 tenant-scoped；
- 运维端点与公开推理端点分离；
- worker 控制接口需鉴权，并校验模型与 tokenizer 版本兼容性。

### 13.3 Runbooks

应提供以下运行手册：

1. `KV pressure high`：限制 admission、降低 batch 接纳、排查长请求与泄漏；
2. `TTFT regression`：检查 prefill 队列、cache miss、路由和 GPU 饱和度；
3. `TPOT regression`：检查 decode queue、batch shape、显存压力和 worker 负载偏斜；
4. `worker unhealthy`：drain、请求失败/重路由、保存诊断信息；
5. `cost spike`：按 tenant、prompt/output、cache miss、取消滞后定位。

---

## 14. 开发路线与工时

| 阶段 | 预计小时 | 可交付结果 |
|---|---:|---|
| Phase 0：模拟器与核心 | 70–100 | 状态机、SimExecutor、KV 预算、FCFS/WFQ、回放器 |
| Phase 1：单 GPU 服务 | 120–180 | API、真实 executor、Prefix/Cost、观测、单卡实验 |
| Phase 2：双 GPU Replica | 60–100 | worker directory、router、故障恢复、双卡实验 |
| 可选 P/D | 额外 40–80 | 协议模拟；条件允许时真实 handoff 与对照 |

Phase 0–2 必做主线预计 **250–380 小时**。真实 P/D 可选，总投入预计 **290–460 小时**。时间不足时优先保证 Phase 0–1 完整度，不以未测试的功能数量代替阶段验收。

---

## 15. 验收清单

### 单卡必须完成

- [ ] OpenAI-compatible streaming API；
- [ ] request state machine、取消、超时、一次性资源回收；
- [ ] SimExecutor 与确定性 workload；
- [ ] Strict/Adaptive KV reservation；
- [ ] FCFS、WFQ、prefix-aware 策略；
- [ ] Prometheus、trace、成本账本；
- [ ] Strict/Adaptive、FCFS/WFQ、prefix-blind/aware 可复现实验；
- [ ] Docker Compose、README、Demo、设计 ADR。

### Phase 2 必须完成

- [ ] worker 注册、心跳、capacity report、draining；
- [ ] 两个独立 GPU worker 的容量与故障隔离；
- [ ] round-robin、least-loaded、cache-aware routing 对照；
- [ ] worker 失联、恢复、drain 和请求唯一归属测试；
- [ ] 双卡拓扑、逐 worker 指标、原始 trace 与实验报告。

### 可选 P/D 完成条件

- [ ] 执行器 KV transfer 能力和 KV layout 已锁定并验证；
- [ ] transfer 状态机通过取消、超时和故障回收测试；
- [ ] decode 侧确实复用 KV，未通过重新 prefill 冒充 transfer；
- [ ] colocated、replica、P/D 在相同双卡成本下完成正式对照。

---

## 16. 最终作品集描述

> CachePilot is a multi-tenant control layer for LLM serving. It combines KV-budgeted admission, fair scheduling, OpenAI-compatible streaming, request-level observability, and reproducible workload replay, then extends the same control model to a dual-GPU replica pool with load-aware/cache-aware routing and worker failure recovery.
