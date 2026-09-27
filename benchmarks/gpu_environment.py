"""Collect and validate the single-GPU experiment environment.

The probe deliberately prints a small allow-list of environment metadata. It
does not print environment variables, command lines, or exception payloads
that could contain a Hugging Face or cloud credential.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = (3, 13, 15)
DEFAULT_MIN_GPU_MEMORY_GIB = 8
DEFAULT_MIN_DISK_GIB = 10


def _nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            total_mib = int(fields[2])
            free_mib = int(fields[3])
        except ValueError:
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_bytes": total_mib * 1024 * 1024,
                "memory_free_bytes": free_mib * 1024 * 1024,
                "driver": fields[4],
            }
        )
    return devices


def _torch_metadata() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except (ImportError, OSError, RuntimeError):
        return {"pytorch": None, "cuda": None, "cuda_available": False}
    return {
        "pytorch": getattr(torch, "__version__", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def collect_environment(root: Path) -> dict[str, Any]:
    model_path = root / "config" / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    devices = _nvidia_smi()
    torch_metadata = _torch_metadata()
    disk = shutil.disk_usage(root)
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "python": {
            "version": ".".join(str(part) for part in sys.version_info[:3]),
            "implementation": platform.python_implementation(),
        },
        "gpu_count": len(devices),
        "gpus": devices,
        "cuda": torch_metadata["cuda"],
        "cuda_available": torch_metadata["cuda_available"],
        "pytorch": torch_metadata["pytorch"],
        "disk": {
            "path": str(root),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
        "model": {
            "id": model["model_id"],
            "revision": model["model_revision"],
            "tokenizer_id": model["tokenizer_id"],
            "tokenizer_revision": model["tokenizer_revision"],
            "context_limit": model["initial_service_context_limit"],
        },
    }


def validate_environment(
    environment: dict[str, Any],
    *,
    min_gpu_memory_gib: int = DEFAULT_MIN_GPU_MEMORY_GIB,
    min_disk_gib: int = DEFAULT_MIN_DISK_GIB,
    require_torch: bool = False,
) -> list[str]:
    failures: list[str] = []
    python_version = tuple(
        int(part) for part in environment["python"]["version"].split(".")
    )
    if python_version != EXPECTED_PYTHON:
        failures.append("python version does not match 3.13.15")
    if (
        environment["platform"]["system"] != "Linux"
        or environment["platform"]["machine"] != "x86_64"
    ):
        failures.append("single-GPU validation requires Linux x86_64")
    if environment["gpu_count"] != 1:
        failures.append(
            f"expected exactly one NVIDIA GPU, found {environment['gpu_count']}"
        )
    elif environment["gpus"][0]["memory_total_bytes"] < min_gpu_memory_gib * 1024**3:
        failures.append(f"GPU memory is below {min_gpu_memory_gib} GiB")
    if environment["disk"]["free_bytes"] < min_disk_gib * 1024**3:
        failures.append(f"free disk space is below {min_disk_gib} GiB")
    if require_torch and (not environment["cuda_available"] or not environment["pytorch"]):
        failures.append("PyTorch CUDA runtime is unavailable")
    if not environment["model"]["revision"] or not environment["model"]["tokenizer_revision"]:
        failures.append("model and tokenizer revisions must be pinned")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-gpu-memory-gib", type=int, default=DEFAULT_MIN_GPU_MEMORY_GIB)
    parser.add_argument("--min-disk-gib", type=int, default=DEFAULT_MIN_DISK_GIB)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()

    environment = collect_environment(args.root.resolve())
    failures = validate_environment(
        environment,
        min_gpu_memory_gib=args.min_gpu_memory_gib,
        min_disk_gib=args.min_disk_gib,
        require_torch=args.require_torch,
    )
    print(json.dumps(environment, ensure_ascii=False, sort_keys=True, indent=2))
    if failures:
        print("GPU_ENVIRONMENT=FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    print("GPU_ENVIRONMENT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
