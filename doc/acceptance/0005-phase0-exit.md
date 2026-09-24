# 验收记录 0005：Phase 0 出口

## 出口条件

Phase 0 只有同时满足以下条件，才允许进入 GPU 集成：

1. CPU 测试全部通过；当前仓库使用标准库 `unittest` 运行 138 个测试，沙箱中
   仅因禁止本地 socket 监听而跳过空服务测试。
2. 资源不变量成立：取消、完成、超时、失败和重复终止后，逻辑 KV reservation
   回到基线；tenant 限额、三重调度预算和并发竞争测试不突破硬上限。
3. 固定 workload trace 可重现：`benchmarks/phase0_exit.py` 用 seed `7` 对比
   `workloads/*.jsonl` 的七类小样例；SimExecutor、RuntimeLoop 和 Scheduler
   另有固定 trace 重放测试。
4. 逻辑 KV 与物理 KV 边界已有 ADR：逻辑 reservation/Prefix Index 不能冒充
   物理显存或物理 KV 命中；执行器没有可靠信号时物理命中保持不可观测。

## 验收命令

```bash
python benchmarks/phase0_exit.py
```

成功输出 `PHASE0_EXIT=PASS`。失败时必须先修复失败项，不能以失败结果进入 GPU
集成。该脚本只验证 CPU 控制面和模拟语义，不宣称真实 GPU 性能或物理 KV 复用。
