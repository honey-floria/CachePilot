# ADR-0007：SimExecutor 的逻辑时钟与执行语义

## 状态

已接受，2026-09-22。

## 决策

Phase 0 使用单线程、固定 tick 的 `SimExecutor` 验证执行顺序与故障路径，
不把模拟结果描述为真实模型或 GPU 性能。

每个 tick 按以下稳定顺序执行：

1. 处理已到期的自动取消；
2. 按请求配置的速率排空客户端输出缓冲；
3. 按提交顺序把等待请求补入未满的 continuous batch；
4. 对 tick 开始时的 batch 按顺序推进 prefill 或 decode；
5. 结束已经排空输出缓冲的 streaming 请求，更新峰值统计；
6. tick 计数加一，并把逻辑时钟推进固定 `tick_ns`。

Prefill 每 tick、每请求最多处理 `prefill_tokens_per_tick`；decode 每 tick、
每请求最多生成 `decode_tokens_per_tick`。尚未交付客户端的 token 占用有界
输出缓冲；缓冲不足时只生成可容纳的 token 并记录 backpressure 事件。

逻辑 KV block 按 `ceil((processed_prompt + generated_output) / block_size)`
增长。计算完成时立即释放 KV，尚未交付的输出可以继续处于 `STREAMING`，
因此慢客户端不会继续占用模拟 KV。取消和 worker 故障会清除输出缓冲、
释放 KV 并进入不可逆终态。

事件使用全局递增 sequence 和逻辑 `at_ns`。请求、事件、快照和统计按提交
顺序输出；实现不读取墙上时间、不使用进程随机状态。运行根 seed 与请求 seed
作为实验输入记录，即使当前固定成本模型不需要随机采样。相同配置、提交序列、
seed 和逻辑时钟操作必须得到相等的最终快照。

## 后果

该模型可以确定性覆盖 prefill/decode、KV 增长、batch 补位、客户端背压、
取消和 worker 故障。它不负责 Scheduler/Admission 的三重预算；下一阶段的
runtime loop 必须在调用执行器前独立约束 active sequences、batch tokens 和
KV blocks。
