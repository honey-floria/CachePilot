# 诊断导出测试数据

这三个文件是固定的模拟实验协议样例，供 `test_export_diagnostics.py` 使用。
样例来自 `runs/demo-sim-mixed-001`，不代表真实 GPU 验收结果。

测试输入随 `tests/` 一起复制，不依赖运行时产物目录 `runs/`。Colab notebook
刻意排除旧的 `runs/`，避免复制大量历史实验，因此测试不能从该目录读取输入。
`summary.json` 由测试调用分析器重新生成，不作为固定输入保存。
