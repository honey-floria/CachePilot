# 故障注入测试

故障验收入口：

```bash
python -m benchmarks.chaos_validation \
  --output runs/phase1-chaos/chaos_report.json
```

该命令使用真实 Gateway、Registry 和 Admission，注入取消、断连、执行超时、执行器
异常和 OOM，并检查每个请求只有一个终态转换且 reservation/active sequence 回到零。
OOM 是稳定的注入异常；GPU 服务器上仍需用真实上下文/并发重复一次，保留服务日志和
原始 run，才能把它作为硬件 OOM 证据。

Phase 1 出口检查：

```bash
python -m benchmarks.phase1_exit \
  --chaos-report runs/phase1-chaos/chaos_report.json \
  --output runs/phase1-exit/report.json \
  runs/<sim-run> runs/<torch-or-vllm-run>
```

它要求同时存在 Sim 和至少一个实测执行器，并检查 TTFT/TPOT/total P99、吞吐、公平性、
拒绝率、取消率、KV 峰值、GPU 秒和估算成本字段。
