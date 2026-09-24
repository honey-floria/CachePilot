#!/usr/bin/env python3
"""执行 Phase 0 出口的 CPU、trace 和 KV 语义检查。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    "uniform", "mixed-length", "burst", "noisy-neighbor",
    "shared-prefix", "cancellation-heavy", "long-context",
)


def _check_workloads() -> None:
    """确认七类提交样例与固定 seed 生成结果逐字一致。"""

    sys.path.insert(0, str(ROOT))
    from workloads.generator import generate_workload

    for name in WORKLOADS:
        expected = generate_workload(name, seed=7, count=4)
        actual = tuple(
            json.loads(line)
            for line in (ROOT / "workloads" / f"{name}.jsonl").read_text().splitlines()
        )
        if actual != expected:
            raise RuntimeError(f"workload is not reproducible: {name}")


def _check_kv_boundary() -> None:
    """确认 ADR 明确区分逻辑 KV 账本和物理 KV handle。"""

    text = "\n".join(
        (ROOT / "doc" / "adr" / name).read_text(encoding="utf-8")
        for name in ("0001-initial-api-scope.md", "0007-sim-executor.md",
                     "0009-prefix-index-and-cache-boost.md")
    )
    required_groups = (("logical KV", "逻辑 KV"), ("physical", "物理"), ("不可观测",))
    missing = [
        "/".join(group) for group in required_groups
        if not any(phrase in text for phrase in group)
    ]
    if missing:
        raise RuntimeError(f"KV boundary ADR evidence missing: {missing}")


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        print("PHASE0_EXIT=FAIL: CPU tests", file=sys.stderr)
        return result.returncode
    try:
        _check_workloads()
        _check_kv_boundary()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"PHASE0_EXIT=FAIL: {exc}", file=sys.stderr)
        return 2
    print("PHASE0_EXIT=PASS")
    print("- CPU tests: PASS")
    print("- resource invariants: covered by registry/admission/executor tests")
    print("- fixed workload replay: PASS")
    print("- logical/physical KV boundary ADR: PASS")
    print("- GPU integration gate: CLOSED until this check passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
