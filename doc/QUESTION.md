# CachePilot 问题整理

## 问题 1：Registry 和幂等性是怎么保证线程安全的？

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

## 问题 2：为什么要使用三个独立锁？

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
