# 回归 Trace

`traces/cancel_finish_race.json` 保存一个固定 seed 的取消/完成竞争故障回归
场景。`tests/property/test_runtime_invariants.py` 会读取该文件，重复执行指定
轮数，并验证每轮只有一个终态获胜、逻辑 KV block 恰好释放一次。

该 trace 是测试输入，不是性能实验记录；实验协议中的 workload trace 仍使用
`workloads/*.jsonl` 和 `config/experiment.schema.json`。
