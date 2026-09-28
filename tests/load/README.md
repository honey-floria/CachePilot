# 负载测试

用于 Phase 0/1 的固定 trace（轨迹）回放和受控压力测试。

真实 GPU 渐进压测必须在 Linux x86_64、单张 NVIDIA GPU、固定模型和锁定依赖上运行：

```bash
python benchmarks/progressive_load.py \
  --base-url http://127.0.0.1:8000 \
  --output-dir runs/progressive-<date> \
  --contexts 32,128,512,2048,4096,8192 \
  --concurrencies 1,2,4,8 \
  --repetitions 3 --warmup
```

先启动真实 GPU 服务（`python main.py`）。runner 会在请求前执行单 GPU/PyTorch
门禁；当前 TorchExecutor 的并发保护可能返回 429，这属于过载保护边界，不是 OOM。
每个矩阵点重复三次，结果写入 `progressive_report.json` 和
`progressive_requests.jsonl`。`safe_context_tokens_observed` 只表示被重复验证的
最大安全测试点，不代表更大上下文或并发必然安全；必须结合 `first_oom_point`、
`first_overload_protection_point` 和硬件元数据解读。
