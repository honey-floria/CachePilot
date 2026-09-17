# 基准测试

实验协议已由 [ADR-0005](../doc/adr/0005-experiment-protocol.md) 固定，机器可读
schema 位于 `config/experiment.schema.json`。一次实验必须保留
`manifest.json`、`trace.jsonl`、`requests.jsonl` 和分析器生成的 `summary.json`。

分析原始记录：

```bash
python benchmarks/analyze.py \
  --manifest runs/<run_id>/manifest.json \
  --trace runs/<run_id>/trace.jsonl \
  --requests runs/<run_id>/requests.jsonl \
  --output runs/<run_id>/summary.json
```

缺少硬件、软件、模型或策略版本等关键元数据时，分析器会以非零状态退出；通过校验
不代表结果已经达到任何性能目标。`colab_acceptance.py` 仅验证 CPU 安装、测试和空服务
探活，不产生或宣称真实性能结果。
