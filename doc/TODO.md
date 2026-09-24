# CachePilot 实施 TODO 与资源规划

本文把[工程设计](DESIGN.md)转成按依赖排序的 Phase 0–2 实施任务。勾选框代表待完成，不代表当前进度；当前仓库已具备 CPU-first 骨架和空服务，但尚无完整运行时代码。每项都要求“实现内容 + 验收证据”，不能用演示视频替代测试、原始实验数据或环境记录。

## 0. 契约与实验基线

- [x] **确定首版范围**：单模型、每请求一个 tenant、聊天生成、SSE；列出暂不支持的 OpenAI 字段、工具调用、图像输入和跨 tenant KV 共享。验收：`doc/adr/` 中存在范围 ADR，API 不默默接受无效字段。
- [x] **固定请求契约**：定义 request ID、tenant 来源、优先级、deadline、幂等键、token 口径、SSE 结束事件、错误码、取消与重试语义。验收：OpenAPI/schema 和契约测试覆盖有效、无效及重复提交。
- [x] **固定模型与依赖**：记录模型 ID/revision、tokenizer revision、许可证、最大上下文、Python/PyTorch/vLLM 兼容矩阵。验收：配置不依赖浮动的 `latest`。
- [x] **建立仓库骨架**：创建包、测试、workload、benchmark、配置和文档目录，固定 CPU 安装、静态检查和测试命令。验收：`requirements/requirements-cpu.txt`、`Makefile` 和 `pyproject.toml` 固定安装/检查入口；全新 Python 3.13.15 CPU 环境可以安装、运行测试和启动 `cachepilot.runtime.empty_service` 空服务；`notebooks/colab_acceptance.ipynb` 与 `benchmarks/colab_acceptance.py` 提供 Colab 验收路径。证据见[仓库骨架验收记录](acceptance/0004-repository-skeleton.md)。空服务仅用于探活，不代表推理运行时已实现。
- [x] **定义实验协议**：固定 trace schema、时钟口径、seed、JSONL/汇总 schema，以及硬件、软件、模型和策略版本字段。验收：`config/experiment.schema.json` 与 [ADR-0005](adr/0005-experiment-protocol.md) 固定协议；`python benchmarks/analyze.py` 在缺少关键元数据、浮动版本或记录不一致时以非零状态判为无效。

## 1. Phase 0：无 GPU 运行时内核（70–100 小时）

### 1.1 请求生命周期与资源所有权

- [x] **状态机**：实现 `RECEIVED → TOKENIZED → QUEUED → ADMITTED → ROUTED → EXECUTING → FINISHED`；终态为 `CANCELLED/TIMED_OUT/REJECTED/FAILED`。验收：非法转换被拒绝，重复事件幂等，终态不可逆且不再输出 token。实现见 `cachepilot/runtime/state_machine.py`，验证见 `tests/unit/test_state_machine.py`。
- [x] **Registry 与幂等性**：按 request ID 和幂等键查询、去重，保存当前状态与事件日志。验收：并发取消、完成和失败竞争时只产生一个终态。实现见 `cachepilot/runtime/registry.py`，验证见 `tests/unit/test_registry.py`。
- [x] **资源租约**：实现 reservation 申请、增长和一次性释放，区分逻辑 KV block 与执行器物理 handle。验收：正常结束、取消、超时、断连和异常后逻辑占用均回到基线。实现见 `cachepilot/runtime/resources.py` 和 `cachepilot/runtime/registry.py`，验证见 `tests/unit/test_resources.py`。

### 1.2 KV 容量与准入

- [x] **KV Planner**：根据模型层数、KV heads、head dim、dtype、block size 和上下文上限计算理论 block/bytes。验收：手算样例一致；配置不足时拒绝推导“真实可用显存”。实现见 `cachepilot/runtime/kv_planner.py`，验证见 `tests/unit/test_kv_planner.py`。
- [x] **Strict Admission**：按 prompt + `max_new_tokens` 预留，并限制 active sequences、总 blocks、tenant token/并发/队列。验收：不能超额接纳，容量不足时进入有界队列或明确拒绝。实现见 `cachepilot/runtime/admission.py`，验证见 `tests/unit/test_admission.py`。
- [x] **Adaptive Admission**：使用分桶的历史输出长度 P95 和安全余量；样本不足回退 Strict；生成增长时重新评估硬上限。验收：长尾请求不能越过硬容量，并记录估计误差和回退次数。实现见 `cachepilot/runtime/adaptive_admission.py`，验证见 `tests/unit/test_adaptive_admission.py`。
- [x] **过载与超时**：定义排队 deadline、执行 deadline、队列上限和重试建议。验收：burst 下行为可预测，所有超时请求最终回收资源。实现见 `cachepilot/runtime/admission.py` 和 `cachepilot/runtime/deadlines.py`，验证见 `tests/unit/test_admission.py` 与 `tests/unit/test_deadlines.py`。

### 1.3 调度与模拟执行器

- [x] **FCFS 与 WFQ**：按 interactive/batch 和 tenant 子队列实现基线调度，明确 WFQ 虚拟时间、权重和最大饥饿时间。验收：固定 trace 下顺序可重放，低权重 tenant 不永久饥饿。实现见 `cachepilot/runtime/scheduler.py`，语义见 [ADR-0006](adr/0006-fcfs-and-wfq-scheduling.md)，验证见 `tests/unit/test_scheduler.py`。
- [x] **SimExecutor**：使用可控逻辑时钟模拟 prefill/decode、KV 增长、continuous batching、慢客户端、取消和 worker 故障。验收：相同配置、trace 和 seed 产生一致事件与统计。实现见 `cachepilot/executors/sim.py`，语义见 [ADR-0007](adr/0007-sim-executor.md)，验证见 `tests/unit/test_sim_executor.py`。
- [x] **调度循环**：每轮先完成与回收，再更新 KV 账本，并按 active sequences、batch tokens 和 KV blocks 三重预算推进请求。验收：混合长短请求不能突破硬上限。实现见 `cachepilot/runtime/loop.py`，语义见 [ADR-0008](adr/0008-runtime-loop-budgets.md)，验证见 `tests/unit/test_runtime_loop.py`。
- [x] **Prefix Index**：按 tenant、模型/tokenizer/量化版本和 tokenized prefix 建立逻辑索引；cache boost 受公平边界约束。验收：跨 tenant/版本不互相命中，逻辑命中不计作物理命中。实现见 `cachepilot/cache/prefix_index.py` 与 `cachepilot/runtime/scheduler.py`，语义见 [ADR-0009](adr/0009-prefix-index-and-cache-boost.md)，验证见 `tests/unit/test_prefix_index.py` 与 `tests/unit/test_scheduler.py`。这是简单可靠的 MVP 实现，未来可以换成 Trie 提高效率.
- [x] **属性与竞争测试**：覆盖取消/完成竞争、deadline 边界、tenant 限额、重复事件、KV 释放和缓存失效。验收证据见 `tests/property/test_runtime_invariants.py` 与 `tests/regression/traces/cancel_finish_race.json`；固定 seed 的取消/完成竞争 trace 可重复验证终态唯一且 KV 回到基线。

### 1.4 回放与出口

- [x] **固定 workloads**：实现 Uniform、Mixed-length、Burst、Noisy-neighbor、Shared-prefix、Cancellation-heavy 和 Long-context。验收证据见 `workloads/generator.py`、`workloads/*.jsonl` 与 `tests/unit/test_workload_generator.py`；生成器固定 seed、校验 trace v1 schema 约束，并提交七类小样例。
- [ ] **分析器**：输出逐请求时间线、准入原因、资源峰值、P50/P95/P99、吞吐、公平性、拒绝率和取消率。验收：Strict/Adaptive 与 FCFS/WFQ 可以控制变量比较，并明确标注模拟结果。
- [ ] **Phase 0 出口**：CPU 测试全部通过；资源不变量成立；固定 trace 可复现；逻辑 KV 与物理 KV 的边界已有 ADR。未满足不得进入 GPU 集成。

## 2. Phase 1：单 GPU 真实服务（120–180 小时）

### 2.1 端到端服务

- [ ] **Gateway/API**：FastAPI 实现参数验证、tenant 身份、配额、普通/流式聊天、请求查询/取消和健康检查。验收：标准客户端可解析 SSE；非法请求不占 reservation。
- [ ] **流控与取消**：断连、显式取消和执行超时传播到 Scheduler 与 Executor；SSE 使用有界缓冲和慢客户端超时。验收：慢读、断连和重复取消不会无限缓冲或泄漏请求。
- [ ] **TorchExecutor**：先实现小模型单请求生成；仅在正确处理 padding、position、mask、停止条件和不等长序列后实现教学型 batching。验收：功能和限制有测试；未实现时不宣称 continuous batching。
- [ ] **VllmExecutor**：通过锁定版本的稳定接口转发 prompt、stream、abort 和 usage。验收：请求 ID、错误、取消与 token 数可核对；不重复实现 vLLM 内部 batching。
- [ ] **能力矩阵**：列出 Sim/Torch/vLLM 各自拥有的 batch、物理 KV、prefix 和取消能力。验收：不同语义的指标不会混合比较。

### 2.2 Prefix、观测与成本

- [ ] **Prefix 核验**：逻辑 key 默认 tenant 隔离；仅在执行器提供可验证信号时记录物理命中。验收：缺少物理信号时明确显示“不可观测”。
- [ ] **指标与 Trace**：记录请求数、准入原因、TTFT、TPOT、queue/prefill/decode 时间、逻辑 KV 和错误。验收：Prometheus label 不含 request ID、prompt 或高基数 prefix key。
- [ ] **请求账本**：记录 token、阶段耗时、策略版本、reservation 峰值、命中来源、终态和估算成本。验收：成本可由原始数据重算，并明确标为估算。
- [ ] **运行手册**：覆盖 KV 压力、TTFT/TPOT 回退、worker 故障和成本突增。验收：单机可导出指标快照和实验报告。

### 2.3 单卡实验与出口

- [ ] **环境脚本/notebook**：输出 GPU、显存、驱动、CUDA、Python、PyTorch、磁盘和模型版本。验收：环境不满足条件时提前失败或选择更小模型，凭据不进入日志。
- [ ] **渐进压测**：从短 prompt、单并发逐级增加上下文和并发，记录 OOM 与过载保护边界。验收：报告安全上限，不把单次成功当成容量结论。
- [ ] **必要对照**：Strict vs Adaptive、FCFS vs WFQ、prefix-blind vs prefix-aware；Torch 若真正支持两种 batch，再比较 static vs continuous。验收：同一执行器、模型和硬件内比较，warm-up 后至少重复三次。
- [ ] **Phase 1 出口**：API/SSE/取消/超时回归通过；Sim 与至少一个真实执行器端到端运行；原始记录和报告包含 TTFT、TPOT、吞吐、P99、公平性、拒绝率、KV 峰值和估算成本。未满足不得进入真实双卡实验。

## 3. Phase 2：同机双 GPU Replica Runtime（60–100 小时）

Phase 2 的正式目标是两个独立 colocated worker 的路由、容量隔离和故障恢复，不以真实 P/D KV handoff 作为完成条件。

### 3.1 Worker 管理

- [ ] **Worker Directory**：实现注册、心跳、模型/版本/容量上报、健康状态和 draining。验收：失联或 draining worker 不接收新请求，已有请求得到明确终态。
- [ ] **双 worker 基线**：在两张 GPU 上部署相同模型的独立 worker，分别记录队列、逻辑 KV、GPU 显存、错误和取消。验收：每个请求有唯一 worker 归属，一个 worker 失败不会污染另一 worker 的容量账本。
- [ ] **故障与恢复**：注入进程退出、心跳超时、拒绝请求和 drain。验收：新请求停止路由到故障节点；失败请求可追踪；恢复后经过 readiness 才重新接流量。

### 3.2 Replica 路由

- [ ] **Least-loaded 基线**：按活跃序列、队列延迟和 KV 压力选择 worker。验收：固定负载下路由决定可解释，并能在 worker 不健康时回退。
- [ ] **Cache-aware 路由**：在模型兼容和 tenant 隔离约束下加入 prefix affinity；当队列代价超过命中收益时回退到低负载 worker。验收：不能为提高命中率造成明显饥饿或持续热点。
- [ ] **双卡对照实验**：比较 round-robin、least-loaded 与 cache-aware。验收：相同 trace 下报告 P99 TTFT/TPOT、吞吐、负载偏斜、逻辑/物理命中、公平性、拒绝率与故障恢复时间。
- [ ] **Phase 2 出口**：真实双 GPU replica 测试通过；worker 注册、drain、失联和恢复可复现；路由实验有原始逐请求数据；README 提供可运行演示和限制说明。

### 3.3 可选：P/D 协议与真实 KV Handoff（额外 40–80 小时）

- [ ] **能力验证**：锁定执行器版本、KV layout、transfer API 和 GPU 拓扑。验收：证明 decode 侧确实复用传入 KV；重新 prefill 不算 KV handoff。
- [ ] **协议模拟**：实现 `TransferReady → TransferAccepted → DecodeReady`、TTL、checksum、有限重试和两端资源所有权。验收：取消、超时、拒绝和 worker 崩溃后 reservation 最终释放。
- [ ] **真实传输**：仅在能力验证通过后接入执行器，记录 bytes、transfer_ms、复制次数和两端显存。验收：续写正确性与 colocated 路径一致。
- [ ] **P/D 对照**：在相同双卡成本下比较 replica、colocated 与 P/D。验收：报告 TTFT、TPOT、吞吐和传输开销；负面结果同样保留。

未完成真实传输时，只能发布“P/D 协议模拟”，不能宣称支持真实 P/D disaggregation。

## 4. 明确不在当前范围

- 多节点 KV cache、CPU/NVMe/远端缓存层；
- 跨节点 KV transfer；
- Kubernetes CRD、控制器、自动扩缩容与 Helm 集群交付；
- 多模型路由、张量并行和超大模型部署；
- 自研 CUDA kernel、attention 或替代 vLLM 的物理 KV allocator。

这些方向只能在 Phase 0–2 完成且出现真实使用需求后另立项目，不预先列为待办。

## 5. 资源与工时

| 资源 | Phase 0 | Phase 1 | Phase 2 必做 | P/D 可选 |
|---|---|---|---|---|
| 算力 | 普通 CPU | 单张可用 GPU | 同机两张兼容 GPU | 同机双 GPU 且执行器支持 KV transfer |
| 软件 | Python、pytest | FastAPI、PyTorch、vLLM、Prometheus | worker RPC/HTTP、双进程服务 | 锁定版本的 P/D 接口 |
| 数据 | 固定模拟 trace | 原始请求记录和单卡报告 | 双卡路由与故障 trace | transfer 元数据与拓扑记录 |
| 人力 | 70–100 小时 | 120–180 小时 | 60–100 小时 | 额外 40–80 小时 |

必做主线预计 **250–380 小时**。真实 P/D 可选，总投入预计 **290–460 小时**。这些数字不包含 GPU 排队、模型下载和环境兼容失败时间。

## 6. 风险与缩减顺序

1. vLLM 与当前 CUDA/PyTorch 不兼容时，先保留 Torch 功能结果，不伪造 vLLM 数据。
2. Torch batching 正确性不足时，只保留单请求生成，性能实验交给 vLLM。
3. 没有稳定双 GPU 时，Phase 2 控制面可用 SimExecutor 验证，但不能宣称真实双卡结果。
4. 执行器不暴露可靠 KV transfer 时，停止在 replica routing，把 P/D 标记为协议模拟。
5. 时间不足时，优先保证 Phase 0–1 完整度，再做 Phase 2；不得用未测试功能换取更大的功能列表。

最终交付应包含可运行代码、自动测试、固定 workload、原始实验数据、分析报告、环境元数据、架构说明和已知限制。只有架构图或演示请求不算阶段完成。
