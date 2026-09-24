"""生成可重复的 Phase 0 workload trace。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Iterable

WORKLOAD_NAMES = (
    "uniform", "mixed-length", "burst", "noisy-neighbor",
    "shared-prefix", "cancellation-heavy", "long-context",
)


def _record(workload, index, tenant, arrival, prompt, output, seed, *,
            priority="interactive", prefix_group=None, cancel_after_ms=None):
    """构造一条显式填充默认字段的 trace v1 记录。"""
    record = {
        "trace_version": 1, "request_id": f"{workload}-{index:04d}",
        "tenant_id": tenant, "arrival_ms": arrival,
        "prompt_tokens": prompt, "expected_output_tokens": output,
        "max_new_tokens": max(1, output), "priority": priority, "seed": seed,
    }
    if prefix_group is not None:
        record["prefix_group"] = prefix_group
    if cancel_after_ms is not None:
        record["cancel_after_ms"] = cancel_after_ms
    return record


def _uniform(rng, count):
    return [_record("uniform", i, f"tenant-{i % 2}", i * 7,
                    rng.randint(8, 24), rng.randint(8, 16), rng.randrange(1 << 30))
            for i in range(count)]


def _mixed_length(rng, count):
    lengths = ((8, 8), (32, 16), (96, 32), (16, 4))
    return [_record("mixed-length", i, f"tenant-{i % 3}", i * 5,
                    *lengths[i % len(lengths)], rng.randrange(1 << 30),
                    priority="batch" if i % 4 == 3 else "interactive")
            for i in range(count)]


def _burst(rng, count):
    return [_record("burst", i, f"tenant-{i % 3}", (i // 4) * 100,
                    rng.randint(12, 32), rng.randint(8, 24), rng.randrange(1 << 30))
            for i in range(count)]


def _noisy_neighbor(rng, count):
    records = []
    for i in range(count):
        noisy = i % 3 != 0
        records.append(_record(
            "noisy-neighbor", i, "noisy" if noisy else "interactive", i * 9,
            rng.randint(48, 96) if noisy else rng.randint(8, 16),
            rng.randint(32, 64) if noisy else rng.randint(4, 12), rng.randrange(1 << 30),
            priority="batch" if noisy else "interactive"))
    return records


def _shared_prefix(rng, count):
    return [_record("shared-prefix", i, f"tenant-{i % 2}", i * 6,
                    rng.randint(24, 40), rng.randint(8, 20), rng.randrange(1 << 30),
                    prefix_group="prefix-a" if i % 4 != 3 else "prefix-b")
            for i in range(count)]


def _cancellation_heavy(rng, count):
    return [_record("cancellation-heavy", i, f"tenant-{i % 2}", i * 4,
                    rng.randint(16, 48), rng.randint(12, 32), rng.randrange(1 << 30),
                    cancel_after_ms=None if i % 5 == 0 else rng.randint(1, 12))
            for i in range(count)]


def _long_context(rng, count):
    return [_record("long-context", i, f"tenant-{i % 2}", i * 25,
                    rng.randint(384, 768), rng.randint(32, 96), rng.randrange(1 << 30),
                    priority="batch" if i % 2 else "interactive")
            for i in range(count)]


_GENERATORS: dict[str, Callable] = {
    "uniform": _uniform, "mixed-length": _mixed_length, "burst": _burst,
    "noisy-neighbor": _noisy_neighbor, "shared-prefix": _shared_prefix,
    "cancellation-heavy": _cancellation_heavy, "long-context": _long_context,
}


def validate_trace(records: Iterable[dict], workload: str | None = None) -> tuple[dict, ...]:
    """校验 trace v1 字段、唯一 ID 和非递减到达时间。"""
    if workload is not None and workload not in WORKLOAD_NAMES:
        raise ValueError(f"unknown workload: {workload}")
    validated, request_ids, previous_arrival = [], set(), 0
    for index, record in enumerate(records):
        required = {"trace_version", "request_id", "tenant_id", "arrival_ms",
                    "prompt_tokens", "expected_output_tokens", "seed"}
        if not isinstance(record, dict) or required.difference(record):
            raise ValueError(f"record {index} is missing required trace fields")
        if record["trace_version"] != 1:
            raise ValueError(f"record {index} has unsupported trace_version")
        request_id = record["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError(f"record {index} has invalid or duplicate request_id")
        request_ids.add(request_id)
        if type(record["arrival_ms"]) is not int or record["arrival_ms"] < previous_arrival:
            raise ValueError(f"record {index} arrival_ms must be non-decreasing")
        previous_arrival = record["arrival_ms"]
        for field in ("arrival_ms", "prompt_tokens", "expected_output_tokens", "seed"):
            if type(record[field]) is not int or record[field] < 0:
                raise ValueError(f"record {index} {field} must be non-negative int")
        if record.get("max_new_tokens", record["expected_output_tokens"]) < 1:
            raise ValueError(f"record {index} max_new_tokens must be positive")
        if record.get("priority", "interactive") not in {"interactive", "batch"}:
            raise ValueError(f"record {index} has invalid priority")
        cancel = record.get("cancel_after_ms")
        if cancel is not None and (type(cancel) is not int or cancel < 0):
            raise ValueError(f"record {index} cancel_after_ms must be non-negative int")
        validated.append(dict(record))
    return tuple(validated)


def generate_workload(name: str, seed: int = 7, count: int = 12) -> tuple[dict, ...]:
    """按名称、seed 和 count 生成并校验固定 workload。"""
    if name not in _GENERATORS:
        raise ValueError(f"unknown workload: {name}")
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
        raise ValueError("seed must be non-negative and count must be positive")
    return validate_trace(_GENERATORS[name](random.Random(seed), count), name)


def write_trace(records: Iterable[dict], output: Path) -> None:
    """以稳定 JSON 编码写出 JSONL trace。"""
    validated = validate_trace(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                                  for r in validated), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=WORKLOAD_NAMES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_trace(generate_workload(args.workload, args.seed, args.count), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
