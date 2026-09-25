## CachePilot 问题整理

### 问题 1：Registry 和幂等性是怎么保证线程安全的？

主要通过三层锁和幂等标识保证：

1. **Registry 锁**

   `RequestRegistry._lock` 同时保护请求表和幂等键索引。检查幂等键、检查
   request ID、创建请求和写入索引都在同一个临界区内完成，避免两个线程
   同时创建重复请求。

2. **幂等键与 fingerprint**

   幂等键作用域是：

   ```text
   tenant_id + API path + idempotency_key
   ```

   相同 key 且 fingerprint 相同，返回原请求；相同 key 但请求内容不同，返回
   `IDEMPOTENCY_KEY_CONFLICT`。不同 tenant 不会互相去重。

3. **每请求状态机锁**

   每个 `RequestStateMachine` 使用独立锁保护状态和事件 ID。同一事件重复提交
   不会重复执行；取消、完成、超时同时发生时，只有第一个线程能写入终态。

4. **资源锁与幂等释放**

   `ResourceLeaseManager` 使用独立锁保护 KV 容量。租约还有 `released` 标记，
   因此资源最多释放一次，不会重复扣减成负数。

例如两个线程同时提交相同幂等键：

```text
线程 A 获得锁 → 创建请求并绑定幂等键
线程 B 随后获得锁 → 查到原请求并直接返回
```

总结：

```text
Registry 锁       → 防止重复注册
状态机锁          → 保证状态和事件幂等
资源锁 + released → 保证容量安全和一次性释放
```

当前实现保证单进程多线程安全；多进程或多节点场景仍需要数据库或 Redis 的
唯一约束和原子操作。

### 问题 2：为什么要使用三个独立锁？

因为三类数据由不同组件拥有，并发冲突也不同：

```text
RequestRegistry._lock       → 请求表和幂等索引
RequestStateMachine._lock   → 单个请求的状态、事件和 token 计数
ResourceLeaseManager._lock  → KV reservation、容量和物理 handle
```

如果所有操作共用一把大锁，一个请求改状态时会阻塞其他请求查询、注册或修改；
使用独立锁后，不同请求的状态可以并行处理。

- Registry 锁防止两个线程重复注册同一个 request ID 或幂等键；
- 状态机锁保证取消、完成、超时竞争时只有一个终态成功；
- 资源锁保证 KV 容量检查和预留是原子的，不会超额分配。

因此三个锁不是重复保护同一份数据，而是分别保护三个资源域，在保证线程安全的
同时减少不必要的阻塞。

### 问题3. CachePilot 主要解决什么问题？它与 vLLM、SGLang 等推理引擎的职责边界是什么？

   CachePilot 主要解决的不是“如何把一个模型算出来”，而是“多个不同租户、不同优先级、不同长度的请求，如何安全、公平、可观测地共享推理资源”。它是建立在 vLLM、SGLang 等推理引擎之上的 LLM Serving 控制层，重点负责：

   - 请求生命周期：接收、排队、准入、执行、取消、超时、失败和资源回收；
   - 容量控制：active sequences、batch tokens、逻辑 KV blocks 和 tenant 配额；
   - 调度策略：FCFS、WFQ、SLO、等待时间以及有限的 prefix/cache 优先级；
   - 单卡容量治理：在一张 GPU 上平衡吞吐、尾延迟、KV 压力和租户公平性；
   - 可观测性和成本：TTFT、TPOT、P99、拒绝率、KV 压力、GPU 秒和每请求成本；
   - 可复现实验：固定 workload、seed、模拟器和故障路径。

   两者的职责边界如下：

   | 层次 | CachePilot | vLLM / SGLang |
   |---|---|---|
   | API 与请求治理 | Gateway、租户、队列、取消、超时、SLO | 接收已送入执行器的请求 |
   | 准入决策 | 决定请求接纳、排队还是拒绝 | 负责接纳后的执行 |
   | 多租户公平性 | tenant 配额、WFQ、饥饿保护 | 通常不负责跨租户服务策略 |
   | 单卡执行边界 | 决定何时把请求交给唯一执行器 | 负责接纳后的真实生成 |
   | Batch 调度 | 仅在模拟器或教学型执行器中实现 | 负责真实 continuous batching |
   | 模型计算 | 不实现 attention、CUDA kernel 或模型并行 | 负责真实模型执行和 GPU kernel |
   | KV 管理 | 逻辑 KV 预算、reservation 和外层账本 | 物理 KV allocator、block layout 和实际复用 |

   CachePilot 不应在 vLLM 或 SGLang 外再实现一套相同的 continuous batching，否则会产生两层排队、延迟难以归因，以及两个 batch 调度器互相干扰。使用 `VllmExecutor` 时，batch 调度由 vLLM 内部负责，CachePilot 只管理外层队列、准入和观测。

   KV 也要区分逻辑层和物理层：CachePilot 根据请求预计需要的 token 数计算逻辑 KV blocks，用于准入和容量账本；vLLM/SGLang 决定 GPU 上物理 KV 的实际分配、排列、复用和释放。CachePilot 发现 prefix 匹配只能算逻辑命中，只有执行器确认实际复用了 KV，才能算物理命中。
  2. 一个请求从进入 Gateway 到最终完成，完整的生命周期和状态转换路径是怎样的？哪些状态转换是非法的？
  3. 项目为什么要区分逻辑 KV reservation 和执行器实际持有的物理 KV？KV block 的容量是如何估算的？
  4. Strict Admission 和 Adaptive Admission 的核心区别是什么？Adaptive 策略在历史样本不足或输出长度出现长尾时如何处理？
  5. 如果请求取消、超时、正常完成和执行器故障同时发生，项目如何保证只产生一个终态，并且资源只释放一次？
  6. RuntimeLoop 为什么同时限制 active sequences、batched tokens 和 KV blocks？如果只限制并发请求数，可能出现什么问题？
  7. FCFS 和 WFQ 调度器分别适用于什么场景？WFQ 中如何体现 tenant 权重，并避免低权重租户长期饥饿？
  8. Prefix-aware 调度中的 cache boost 如何实现？为什么 prefix 命中必须按 tenant、模型、tokenizer 和量化版本隔离？
  9. SimExecutor 如何保证相同 workload、seed 和配置能够得到可重复的结果？这种模拟器与真实 GPU 推理之间有哪些差异？
  10. 目前项目有哪些测试来验证状态机、资源不变量、并发竞争和实验可复现性？完成 Phase 1 还需要补充哪些测试和单卡性能证据？
