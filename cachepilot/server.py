"""NVIDIA 单卡真实推理服务入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import uvicorn

from cachepilot.config.baseline import load_model_baseline
from cachepilot.executors.torch_executor import TorchExecutor, TorchExecutorConfig
from cachepilot.gateway.api import GatewaySettings, create_app


def check_cuda(device_index: int, dtype: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "请先安装 GPU 依赖：python -m pip install -r requirements/requirements-gpu.txt"
        ) from exc
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("NVIDIA CUDA 不可用，请检查 nvidia-smi 和 PyTorch CUDA 安装。")
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(f"GPU 索引 {device_index} 不存在，请检查 CUDA_VISIBLE_DEVICES。")
    torch.cuda.set_device(device_index)
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("此 GPU 不支持 bfloat16，请使用 --dtype float16。")
    device = f"cuda:{device_index}"
    try:
        probe = torch.ones(1, device=device, dtype=getattr(torch, dtype))
        (probe + probe).sum().item()
        torch.cuda.synchronize(device_index)
    except RuntimeError as exc:
        raise RuntimeError(f"CUDA 运算检查失败：{exc}") from exc
    properties = torch.cuda.get_device_properties(device_index)
    logging.getLogger(__name__).info(
        "GPU: %s | VRAM: %.1f GiB | PyTorch: %s | CUDA: %s | dtype: %s",
        properties.name,
        properties.total_memory / 1024**3,
        torch.__version__,
        torch.version.cuda,
        dtype,
    )
    return device


def create_gpu_app(
    *,
    device_index: int = 0,
    dtype: str = "bfloat16",
    admission: str = "strict",
    scheduler: str = "fcfs",
    prefix_mode: str = "blind",
):
    device = check_cuda(device_index, dtype)
    baseline = load_model_baseline(
        Path(__file__).resolve().parents[1] / "config" / "model.json"
    )
    logging.getLogger(__name__).info("加载模型 %s", baseline.model_id)
    executor = TorchExecutor(
        TorchExecutorConfig(
            model_id=baseline.model_id,
            tokenizer_id=baseline.tokenizer_id,
            model_revision=baseline.model_revision,
            tokenizer_revision=baseline.tokenizer_revision,
            device=device,
            dtype=dtype,
            context_limit=baseline.service_context_limit,
            trust_remote_code=baseline.trust_remote_code,
        )
    )
    return create_app(
        GatewaySettings(
            model_id=baseline.model_id,
            context_limit=baseline.service_context_limit,
            max_active_sequences=8,
            admission_strategy=admission,
            scheduler_strategy=scheduler,
            prefix_mode=prefix_mode,
        ),
        backend=executor,
        token_counter=executor,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=int, default=0, help="可见 GPU 的索引，默认 0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--admission", choices=("strict", "adaptive"), default="strict")
    parser.add_argument("--scheduler", choices=("fcfs", "wfq"), default="fcfs")
    parser.add_argument("--prefix-mode", choices=("blind", "aware"), default="blind")
    parser.add_argument("--check", action="store_true", help="仅检查 CUDA，不下载模型")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.check:
            check_cuda(args.device, args.dtype)
            return
        app = create_gpu_app(
            device_index=args.device,
            dtype=args.dtype,
            admission=args.admission,
            scheduler=args.scheduler,
            prefix_mode=args.prefix_mode,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"GPU 服务启动失败：{exc}\n")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
