"""Evidence helpers for the Phase 1 Colab notebook."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.analyze import analyze


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sim_run(output: Path, reference: Path) -> None:
    from cachepilot.executors import (
        LogicalClock,
        SimExecutor,
        SimExecutorConfig,
        SimRequest,
    )

    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((reference / "manifest.json").read_text())
    manifest.update(run_id="colab-sim", trace_id="colab-sim-v1", warmup=False)
    manifest["hardware"].update(
        gpu="none",
        gpu_count=0,
        gpu_memory_bytes=0,
        driver="not_applicable",
        cuda="not_applicable",
        topology="not_applicable",
    )
    manifest["software"]["executor"] = "SimExecutor"
    manifest["strategy"].update(
        executor="SimExecutor",
        admission="strict",
        scheduler="fcfs",
        prefix_mode="blind",
    )
    executor = SimExecutor(
        SimExecutorConfig(
            tick_ns=1_000_000,
            block_size=16,
            max_batch_size=2,
            prefill_tokens_per_tick=32,
            decode_tokens_per_tick=1,
            output_buffer_tokens=16,
            client_drain_tokens_per_tick=16,
            seed=7,
        ),
        LogicalClock(),
    )
    traces = []
    for index, prompt_tokens in enumerate((32, 128, 256, 512)):
        request_id = f"sim-{index}"
        executor.submit(SimRequest(request_id, prompt_tokens, 8, seed=7 + index))
        traces.append(
            {
                "trace_version": 1,
                "request_id": request_id,
                "tenant_id": f"team-{'a' if index % 2 == 0 else 'b'}",
                "arrival_ms": 0,
                "prompt_tokens": prompt_tokens,
                "expected_output_tokens": 8,
                "seed": 7 + index,
            }
        )
    snapshot = executor.run_until_idle()
    assert snapshot.stats.current_logical_blocks == 0
    records = []
    for trace, request in zip(traces, snapshot.requests):
        events = [
            event for event in snapshot.events if event.request_id == request.request_id
        ]
        tokens = [
            event.at_ns / 1e6
            for event in events
            if event.kind.value == "token_generated"
        ]
        prefill = next(
            event.at_ns / 1e6
            for event in events
            if event.kind.value == "prefill_started"
        )
        record = {key: value for key, value in trace.items() if key != "trace_version"}
        record.update(
            record_type="request",
            schema_version=1,
            run_id=manifest["run_id"],
            terminal_state=request.state.value,
            completion_tokens=request.delivered_tokens,
            queue_ms=prefill,
            ttft_ms=tokens[0],
            tpot_ms=(tokens[-1] - tokens[0]) / (len(tokens) - 1),
            total_ms=request.terminal_at_ns / 1e6,
            worker_id=snapshot.worker_id,
            logical_hit=False,
            physical_hit=None,
            reserved_blocks_peak=request.peak_logical_blocks,
            estimated_gpu_seconds=None,
            estimated_cost=None,
            cost_is_estimate=True,
        )
        records.append(record)
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "trace.jsonl", traces)
    write_jsonl(output / "requests.jsonl", records)
    write_json(output / "sim_snapshot.json", asdict(snapshot))
    write_json(
        output / "summary.json",
        analyze(
            output / "manifest.json",
            output / "requests.jsonl",
            output / "trace.jsonl",
        ),
    )


def gpu_chaos(output: Path, dtype: str) -> None:
    import httpx
    import torch
    import uvicorn

    from benchmarks.gpu_environment import collect_environment, validate_environment
    from cachepilot.server import create_gpu_app
    from cachepilot.runtime.registry import RequestNotFoundError

    root = Path(__file__).resolve().parents[1]
    environment = collect_environment(root)
    failures = validate_environment(environment, require_torch=True)
    if failures:
        raise RuntimeError("; ".join(failures))
    if output.exists():
        raise FileExistsError(output)
    report = {
        "status": "FAIL",
        "environment": environment,
        "dtype": dtype,
        "evidence_kind": "real_torch_http_with_controlled_faults",
        "cases": [],
        "oom_scope": "real CUDA allocator failure inside model.forward; not a natural workload capacity boundary",
    }
    write_json(output, report)
    app = create_gpu_app(dtype=dtype)
    runtime = app.state.gateway_runtime
    executor = runtime.backend
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    original_forward = executor.model.forward

    def wait_for(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise TimeoutError("GPU chaos condition not reached")

    def request(name, *, stream=False, deadline=60000, tokens=512):
        return {
            "headers": {
                "X-Tenant-ID": "team-a",
                "X-Request-ID": name,
                "X-Deadline-Ms": str(deadline),
            },
            "json": {
                "model": runtime.settings.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a long numbered list of animals.",
                    }
                ],
                "max_tokens": tokens,
                "stream": stream,
            },
        }

    def post(name, **kwargs):
        return client.post("/v1/chat/completions", **request(name, **kwargs))

    def executing(name):
        try:
            snapshot = runtime.registry.get(name)
        except RequestNotFoundError:
            return False
        return snapshot.terminal or (
            snapshot.state.value == "EXECUTING" and executor._generation_lock.locked()
        )

    def settle():
        acquired = executor._generation_lock.acquire(timeout=120)
        if not acquired:
            raise TimeoutError("Torch generation thread did not stop")
        executor._generation_lock.release()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    try:
        wait_for(lambda: server.started)
        assert post("chaos-warmup", tokens=8).status_code == 200
        settle()
        baseline_bytes = torch.cuda.memory_allocated()
        for name, expected in (
            ("cancel", "CANCELLED"),
            ("disconnect", "CANCELLED"),
            ("timeout", "TIMED_OUT"),
            ("exception", "FAILED"),
            ("oom", "FAILED"),
        ):
            request_id = f"gpu-chaos-{name}"
            injection = {}
            if name == "cancel":
                future = pool.submit(post, request_id)
                wait_for(lambda: executing(request_id))
                for _ in range(2):
                    response = client.post(
                        f"/v1/requests/{request_id}/cancel",
                        headers={"X-Tenant-ID": "team-a"},
                    )
                    assert response.status_code in (200, 202)
                future.result(timeout=120)
            elif name == "disconnect":
                with client.stream(
                    "POST", "/v1/chat/completions", **request(request_id, stream=True)
                ) as response:
                    assert response.status_code == 200
                    wait_for(lambda: executing(request_id))
            elif name == "timeout":
                post(request_id, deadline=100)
            else:

                def fail_forward(*args, **kwargs):
                    injection["forward_entered"] = True
                    if name == "oom":
                        total = torch.cuda.get_device_properties(0).total_memory
                        try:
                            torch.empty(
                                total + 1024**3, dtype=torch.uint8, device="cuda:0"
                            )
                        except torch.cuda.OutOfMemoryError:
                            injection["cuda_oom_observed"] = True
                            raise
                        raise RuntimeError("expected CUDA allocation to fail")
                    raise RuntimeError("controlled model.forward failure")

                executor.model.forward = fail_forward
                try:
                    post(request_id, tokens=8)
                finally:
                    executor.model.forward = original_forward
                assert injection.get("forward_entered")
                if name == "oom":
                    assert injection.get("cuda_oom_observed")
            wait_for(lambda: runtime.registry.get(request_id).terminal)
            settle()
            snapshot = runtime.registry.get(request_id)
            terminal = {"FINISHED", "CANCELLED", "TIMED_OUT", "REJECTED", "FAILED"}
            transitions = sum(
                event.state.value in terminal for event in snapshot.events
            )
            admission = runtime.admission.snapshot()
            allocated = torch.cuda.memory_allocated()
            case = {
                "name": name,
                "expected": expected,
                "terminal_state": snapshot.state.value,
                "terminal_transitions": transitions,
                "reserved_blocks": admission.reserved_blocks,
                "active_sequences": admission.active_sequences,
                "baseline_allocated_bytes": baseline_bytes,
                "allocated_bytes_after": allocated,
                "allocation_tolerance_bytes": 16 * 1024**2,
                "injection": injection,
                "query": runtime.query(request_id, "team-a"),
                "events": [asdict(event) for event in snapshot.events],
                "executor_cancel_reason": executor.cancel_reasons.get(request_id),
            }
            case["status"] = (
                "PASS"
                if (
                    snapshot.state.value == expected
                    and transitions == 1
                    and admission.reserved_blocks == 0
                    and admission.active_sequences == 0
                    and allocated <= baseline_bytes + 16 * 1024**2
                )
                else "FAIL"
            )
            recovery = post(f"{request_id}-recovery", tokens=8)
            case["recovery_http_status"] = recovery.status_code
            case["recovery_query"] = runtime.query(
                f"{request_id}-recovery", "team-a"
            )
            with runtime._schedule_lock:
                case["executing_requests_after_recovery"] = sorted(
                    runtime._executing_requests
                )
                case["schedule_events_after_recovery"] = sorted(
                    runtime._schedule_events
                )
            if name in {"cancel", "disconnect", "timeout"}:
                expected_reason = "explicit" if name == "cancel" else name
                if case["executor_cancel_reason"] != expected_reason:
                    case["status"] = "FAIL"
            if (
                recovery.status_code != 200
                or case["recovery_query"]["state"] != "FINISHED"
                or case["executing_requests_after_recovery"]
                or case["schedule_events_after_recovery"]
            ):
                case["status"] = "FAIL"
            report["cases"].append(case)
            write_json(output, report)
            assert case["status"] == "PASS", (
                f"{name}: terminal/release/recovery check failed"
            )
        report["status"] = "PASS"
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        executor.model.forward = original_forward
        server.should_exit = True
        thread.join(timeout=15)
        client.close()
        listener.close()
        pool.shutdown(wait=True)
        write_json(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sim", "gpu-chaos"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    if args.action == "sim":
        if args.reference is None:
            parser.error("sim requires --reference")
        sim_run(args.output, args.reference)
    else:
        gpu_chaos(args.output, args.dtype)


if __name__ == "__main__":
    main()
