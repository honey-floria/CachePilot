# CachePilot 实施 TODO 与资源规划

本文把[工程设计](DESIGN.md)转换为只面向 CPU 模拟与单 GPU 服务的实施任务。
每项任务都要求“实现内容 + 验收证据”；多 GPU 路由、跨卡 KV 和 P/D 拆分不进入
当前路线图。

## 0. 契约与实验基线

- [x] **首版范围**：单模型、单 GPU、每请求一个 tenant、聊天生成和 SSE；API 不默默接受无效字段。
- [x] **请求契约**：固定 request ID、tenant、优先级、deadline、幂等键、token 口径、错误、取消与重试语义。
- [x] **模型与依赖**：固定模型/tokenizer revision、许可证、上下文和 Python/PyTorch/vLLM 兼容矩阵。
- [x] **仓库骨架**：固定 CPU 安装、检查、测试、服务和 Colab 验收入口。
- [x] **实验协议**：固定 trace、时钟、seed、manifest、逐请求记录和 summary schema；实验仅允许 CPU 或单 GPU。

## 1. Phase 0：无 GPU 运行时内核

### 1.1 生命周期与资源

- [x] **状态机**：实现主路径与异常终态，拒绝非法迁移并保证终态不可逆。
- [x] **Registry 与幂等性**：按 request ID 和 tenant 作用域幂等键去重；并发竞争只产生一个终态。
- [x] **资源租约**：实现逻辑 KV reservation 的申请、增长和一次性释放，并与物理 handle 分离。
- [x] **Deadline**：实现排队、执行和总 deadline，超时后确定性回收资源。

### 1.2 单卡容量与调度

- [x] **KV Planner**：根据模型架构与显式 KV 预算计算 block 和理论字节。
- [x] **Strict Admission**：按最大输出预留，限制 active sequences、KV、tenant token/并发和队列。
- [x] **Adaptive Admission**：使用历史 P95 与安全余量，样本不足回退 Strict，增长不得越过硬上限。
- [x] **FCFS/WFQ**：实现 tenant 子队列、权重和最大饥饿保护。
- [x] **Prefix Index**：实现 tenant/模型/tokenizer/量化隔离和受公平边界约束的 cache boost。
- [x] **Runtime Loop**：按 active sequences、batch tokens、KV blocks 三重预算推进。

### 1.3 模拟、回放与出口

- [x] **SimExecutor**：模拟 prefill/decode、continuous batching、背压、取消和单 Worker 故障。
- [x] **固定 Workloads**：覆盖 Uniform、Mixed-length、Burst、Noisy-neighbor、Shared-prefix、Cancellation-heavy 和 Long-context。
- [x] **分析器**：输出时间线、准入原因、资源峰值、分位数、吞吐、公平性、拒绝率和取消率。
- [x] **属性与竞争测试**：验证唯一终态、KV 回收、deadline、tenant 限额与确定性重放。
- [x] **Phase 0 出口**：`python benchmarks/phase0_exit.py` 通过，证据见[验收记录](acceptance/0005-phase0-exit.md)。

## 2. Phase 1：单 GPU 真实服务

### 2.1 端到端服务

- [x] **Gateway/API**：实现严格参数校验、tenant 身份、配额、普通/流式聊天、查询、取消和探活。
- [ ] **流控与取消**：把断连、显式取消和执行超时传播到 Scheduler 与 Executor；SSE 使用有界缓冲。
- [ ] **TorchExecutor**：先完成小模型单请求生成；只有正确处理 padding、position、mask 和停止条件后才增加教学型 batching。
- [ ] **VllmExecutor**：通过锁定版本的稳定接口转发 prompt、stream、abort 和 usage，不重复实现内部 batching。
- [ ] **能力矩阵**：明确 Sim/Torch/vLLM 的 batch、物理 KV、prefix、取消和指标能力。

### 2.2 观测、成本与运行保障

- [ ] **Prefix 核验**：逻辑命中与执行器确认的物理命中分别计数；物理信号缺失时记录为不可观测。
- [ ] **指标与 Trace**：记录 TTFT、TPOT、queue/prefill/decode、准入原因、逻辑 KV 和错误；禁止高基数 label。
- [ ] **请求账本**：保存 token、耗时、策略版本、reservation 峰值、命中来源、终态和估算 GPU 秒/成本。
- [ ] **单 Worker 健康**：执行器异常后停止接纳新请求，已有请求明确失败并回收 reservation；恢复后通过 readiness 再接流量。
- [ ] **运行手册**：覆盖 KV 压力、TTFT/TPOT 回退、执行器故障、OOM 和成本突增。

### 2.3 单卡实验与出口

- [ ] **环境检查**：输出唯一 GPU、显存、驱动、CUDA、Python、PyTorch、磁盘和模型版本；检测到多张可见 GPU 时拒绝启动实验。
- [ ] **渐进压测**：从短 prompt、单并发逐步增加上下文和并发，记录 OOM 与过载保护边界。
- [ ] **必要对照**：Strict/Adaptive、FCFS/WFQ、prefix-blind/prefix-aware；同一执行器、模型和 GPU 上至少重复三次。
- [ ] **故障验证**：注入取消、断连、执行超时、执行器异常和 OOM，验证唯一终态与资源回收。
- [ ] **Phase 1 出口**：至少一个真实执行器端到端运行；原始记录与报告覆盖 TTFT、TPOT、吞吐、P99、公平性、拒绝率、KV 峰值和估算成本。

## 3. 明确不在当前范围

- 多 GPU 副本、Worker Directory、round-robin/least-loaded/cache-aware 跨卡路由；
- Prefill/Decode 分离、KV handoff、跨卡或跨节点传输；
- 张量并行、流水线并行、多模型路由和超大模型部署；
- CPU/NVMe/远端 KV 分层缓存；
- Kubernetes 控制器、自动扩缩容和 Helm 集群交付；
- 自研 CUDA kernel、attention 或物理 KV allocator。

上述能力如果未来出现真实需求，应建立独立设计和独立里程碑，不能提前污染单卡主线。

## 4. 资源与工时

| 资源 | Phase 0 | Phase 1 |
|---|---|---|
| 算力 | 普通 CPU | 一张可用 GPU |
| 软件 | Python、pytest | FastAPI、PyTorch、vLLM、Prometheus |
| 数据 | 固定模拟 trace | 原始请求记录和单卡报告 |
| 人力 | 70–100 小时 | 120–180 小时 |

必做主线预计 **190–280 小时**，不包含 GPU 排队、模型下载和环境兼容失败时间。

## 5. 风险与缩减顺序

1. vLLM 与 CUDA/PyTorch 不兼容时，保留 Torch 功能结果，不伪造 vLLM 数据。
2. Torch batching 正确性不足时，只保留单请求生成，把性能实验交给 vLLM。
3. GPU 显存不足时，降低模型或上下文规模，并把限制写入报告，不降低安全余量冒险运行。
4. 物理 prefix 命中不可观测时，只报告逻辑命中，不推断真实性能收益。
5. 时间不足时，优先完成取消、超时、资源回收和原始实验数据，再增加策略数量。

最终交付包括可运行代码、自动测试、固定 workload、原始实验数据、分析报告、
环境元数据、架构说明和已知限制。只有架构图或演示请求不算完成。
