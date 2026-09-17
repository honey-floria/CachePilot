# ADR-0004：Python 3.13.15 实验基线

- 状态：已接受
- 日期：2026-09-18
- 依赖：ADR-0003

## 背景

CachePilot 的 Google Colab 标准运行时为 Python `3.13.15`。继续固定 Python
`3.11.13` 会让仓库验收环境与 Colab 默认环境不一致，增加额外的运行时切换成本。

## 决策

将 CachePilot 的精确 Python 基线从 `3.11.13` 迁移到 `3.13.15`，并同步更新：

- `.python-version`、`pyproject.toml` 和 `config/dependencies.json`；
- CPU/Colab 验收脚本、notebook、契约测试和项目文档；
- Ruff 的目标 Python 版本为 `py313`。

PyTorch、Transformers、vLLM、tokenizers、safetensors 和模型 revision 暂不变更。
现有依赖元数据声明 vLLM `0.24.0` 支持 Python `>=3.10,<3.15`，因此 Python
`3.13.15` 位于已记录的兼容范围内；这不等同于 GPU、CUDA 和 vLLM 已完成实测。

## 验收要求

迁移后的 CPU/Colab 验收必须使用精确的 Python `3.13.15`，并通过：

```text
make check
python benchmarks/colab_acceptance.py
```

Phase 1 GPU 验证还必须重新记录 Python、PyTorch、Transformers、vLLM、CUDA、驱动、
GPU 型号和 wheel 来源。未完成这些实测前，GPU profile 只能标记为“元数据兼容；GPU
实测待完成”。

## 后果

- Colab 默认运行时可以直接作为 CPU 骨架验收环境使用；
- 不同 Python minor/patch 版本的实验结果不得混合比较；
- 如果未来升级 Python 或 GPU 依赖，必须新增 ADR 或修订本 ADR，并重新生成环境证据。
