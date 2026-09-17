# 验收记录 0004：仓库骨架

## 交付内容

- `pyproject.toml`：固定项目元数据、Python `3.13.15`、pytest/Ruff 入口与 CLI。
- `requirements/requirements-cpu.txt` 与 `requirements/constraints-cpu.txt`：CPU-only 可编辑安装和精确开发工具版本。
- `Makefile`：`install-cpu`、`lint`、`test`、`check`、`serve` 和 `colab-acceptance`。
- `cachepilot/runtime/empty_service.py`：标准库空服务，提供 `/healthz`、`/readyz`、`/metrics`。
- `cachepilot/{runtime,routing,executors,telemetry}` 与 `tests/{unit,integration,load,chaos}`。
- `workloads/`、`benchmarks/`、`deploy/` 和 `notebooks/colab_acceptance.ipynb`。

## CPU/Colab 验收命令

在全新 Python `3.13.15` CPU 环境执行：

```bash
python3 -m pip install -r requirements/requirements-cpu.txt
make check
python benchmarks/colab_acceptance.py
```

通过标志为：

```text
COLAB_ACCEPTANCE=PASS
- pytest: PASS
- /healthz: 200
- /readyz: 200
- /metrics: 200
```

Notebook 已固定仓库 clone、CPU 安装和上述脚本调用，可在 Google Colab 逐格执行。
脚本会拒绝非 `3.13.15` 解释器，防止将其他运行时的结果混入基线。

## 本地证据与限制

当前开发机不是 Python `3.13.15`，因此不能伪造 Python 基线通过；
`python3 benchmarks/colab_acceptance.py` 会明确报告版本不匹配。契约、包导入和
其余 CPU 测试以 `python3 -m unittest discover -s tests -v` 验证通过（40 项中
39 项通过，1 项因当前沙箱禁止监听本地端口而跳过）；在允许本地 socket 的环境中，
完整 40 项测试和空服务探活均验证通过。

空服务只用于安装/启动 smoke test，不执行模型推理，不代表 Phase 0–2 已完成。
