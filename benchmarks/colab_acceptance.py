"""Colab/CPU 骨架验收入口。

运行方式：`python benchmarks/colab_acceptance.py`。该脚本只依赖标准库，
可在 notebook 中直接执行；它会运行测试、启动空服务并探测三个端点。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def _probe(url: str) -> tuple[int, str]:
    with urlopen(url, timeout=5) as response:  # nosec B310 - local smoke test URL
        return response.status, response.read().decode("utf-8")


def main() -> int:
    expected = (3, 11, 13)
    actual = sys.version_info[:3]
    if actual != expected:
        print(
            f"Python baseline mismatch: expected {'.'.join(map(str, expected))}, "
            f"got {'.'.join(map(str, actual))}",
            file=sys.stderr,
        )
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if test.returncode:
        return test.returncode

    port = "8765"
    service = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cachepilot.runtime.empty_service",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if service.poll() is not None:
                return service.returncode or 1
            try:
                status, health = _probe(f"http://127.0.0.1:{port}/healthz")
                if status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            return 1

        ready_status, ready = _probe(f"http://127.0.0.1:{port}/readyz")
        metrics_status, metrics = _probe(f"http://127.0.0.1:{port}/metrics")
        assert json.loads(health)["status"] == "ok"
        assert ready_status == 200 and json.loads(ready)["status"] == "ready"
        assert metrics_status == 200 and "cachepilot_empty_service_up 1" in metrics
        print("COLAB_ACCEPTANCE=PASS")
        print("- pytest: PASS")
        print("- /healthz: 200")
        print("- /readyz: 200")
        print("- /metrics: 200")
        return 0
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()


if __name__ == "__main__":
    raise SystemExit(main())
