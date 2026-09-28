# NVIDIA GPU 服务器运行与 VS Code 调试

## 一次性准备

在 VS Code Remote - SSH 中打开服务器上的项目目录，以下命令均在该窗口的终端执行。
服务器使用 Linux x86_64、NVIDIA 显卡及兼容 PyTorch CUDA 运行时的驱动。
RTX 30 系列和 RTX 40 系列（包括 4090）是目标硬件；实际兼容性以启动检查和真实请求为准。
当前开发环境没有 NVIDIA GPU，尚未完成这些显卡上的实机验收。

项目固定 Python **3.13.15**；先确保 `python3.13 --version` 对应此版本。

```bash
nvidia-smi
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-gpu.txt
python main.py --check
```

依赖沿用 `config/dependencies.json` 和 `requirements/constraints-gpu.txt`
的 Torch/Transformers 基线，不安装 vLLM。Linux 的 PyTorch wheel 自带所需 CUDA
运行库，通常无需另外安装完整 CUDA Toolkit，但主机驱动必须兼容该 wheel。
`--check` 会检查 CUDA、GPU 索引、精度支持并执行小型 GPU 运算，不下载模型。
如果失败，按输出检查驱动、CUDA 版 PyTorch 和 `CUDA_VISIBLE_DEVICES`。

## 启动

```bash
python main.py
```

首次运行会从 Hugging Face 下载仓库中固定 revision 的 Qwen2.5-0.5B-Instruct
模型及 tokenizer，需要网络和缓存磁盘空间。默认缓存位置遵循 Hugging Face 设置，
可以在启动前设置 `HF_HOME` 指向服务器的数据盘。

服务默认监听 `127.0.0.1:8000`，模型使用一张 GPU，默认 BF16。
可以使用 `python main.py --device 1 --dtype float16 --port 8001` 修改参数。
`--device` 是 `CUDA_VISIBLE_DEVICES` 过滤后的索引。
模型及 KV 规划共用 `config/model.json`，因此入口不提供随意更换模型的参数。

需要局域网访问时可加 `--host 0.0.0.0`；当前 tenant header 并非登录认证，
个人远程调试建议用 VS Code 的 Ports 面板转发 8000 端口。

在另一个服务器终端执行真实推理请求：

```bash
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: team-a' \
  -H 'X-Deadline-Ms: 300000' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Introduce yourself briefly."}],"max_tokens":128,"stream":false}'
```

当前 Torch 后端每次执行一个请求；同时到达的请求可能收到 429。可用启动参数为：

```bash
python main.py --admission strict --scheduler fcfs --prefix-mode blind
python main.py --admission adaptive --scheduler fcfs --prefix-mode blind
```

WFQ 与 Prefix-aware 现在已经接入 Gateway 调度队列；Prefix-aware 必须和 WFQ 一起使用：

```bash
python main.py --admission strict --scheduler wfq --prefix-mode blind
python main.py --admission strict --scheduler wfq --prefix-mode aware
```

当前真实 API 请求契约尚未携带 tokenized prefix，因此 aware 模式只有在上层注入逻辑
cache-hit token 后才会产生 cache boost；没有命中时仍按 WFQ 基线调度。
它完成整段 `generate` 后才交付文本增量，因此 SSE 不是逐 token 实时 GPU 流式生成，
不应用此路径评估真实流式 TTFT/TPOT。高吞吐服务后续可接入 vLLM。

## 必要策略对照

策略对照的原始证据必须在同一台单 GPU、同一模型/tokenizer revision、同一服务
commit 中分别保存。每个策略先做至少一次 warm-up，再做至少三次测量；每次测量
都保留 `manifest.json`、`trace.jsonl`、`requests.jsonl` 和分析器生成的
`summary.json`。完成 Strict/Adaptive、FCFS/WFQ、prefix-blind/prefix-aware
运行后，在仓库根目录执行：

```bash
python -m benchmarks.strategy_matrix \
  --metric ttft_ms \
  --output runs/required-matrix/matrix.json \
  runs/required-matrix/<all-run-directories>
```

该命令会拒绝不同执行器、模型、硬件、代码版本或 trace 的混合结果，并拒绝少于
三次非 warm-up 测量。当前 `TorchExecutor` 的能力矩阵声明为单请求 batch，因而
不做 static-vs-continuous；只有执行器实际声明并实现两种 batch 语义时才加入该
对照，不能用并发请求数替代 batch 支持。

## VS Code F5

1. 在 SSH 窗口的服务器端安装 Python 和 Python Debugger 扩展。
2. 执行 `Python: Select Interpreter`，选择项目的 `.venv/bin/python`。
3. 在运行与调试面板选择 **CachePilot: NVIDIA GPU**，按 F5。
4. 在 `cachepilot/gateway/api.py` 或
   `cachepilot/executors/torch_executor.py` 中打断点，再发送上面的请求。

调试配置使用同一个 `main.py` 入口、单进程、无自动 reload，便于跟踪 Python 调用。
断点暂停时请求 deadline 仍会计时，示例已设置 300 秒；超时后重新发送请求。
这可以调试调用 GPU 的 Python 代码，CUDA 内核内部调试需要专门工具。

本地无 GPU 时，继续使用 `make serve-sim` 运行 CPU 模拟服务。
