#!/usr/bin/env python3
"""Run deterministic Gateway fault-injection checks.

The checks use the real Gateway/Registry/Admission implementation with a tiny
fault backend. They validate terminal-state uniqueness and reservation release;
GPU-specific OOM reproduction remains an additional server-side run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi.testclient import TestClient

from cachepilot.gateway.api import GatewaySettings, create_app
from cachepilot.gateway.backends import GeneratedText
from cachepilot.runtime.admission import TenantAdmissionLimits


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class _Counter:
    def count_prompt_tokens(self, request: Any) -> int:
        return 3


class _FaultBackend:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.cancel_reasons: dict[str, str] = {}

    async def is_ready(self) -> bool:
        return True

    async def cancel(self, request_id: str, reason: str = "explicit") -> None:
        self.cancel_reasons.setdefault(request_id, reason)

    async def generate(self, request: Any) -> AsyncIterator[GeneratedText]:
        if self.mode == "exception":
            raise RuntimeError("injected executor failure")
        if self.mode == "oom":
            raise RuntimeError("CUDA out of memory: injected")
        if self.mode == "timeout":
            await asyncio.sleep(0.2)
        yield GeneratedText("ok")


def _settings() -> GatewaySettings:
    return GatewaySettings(
        model_id=MODEL,
        context_limit=256,
        total_kv_blocks=32,
        safety_kv_blocks=2,
        max_active_sequences=2,
        max_queued_requests=2,
        tenant_limits={"team-a": TenantAdmissionLimits(2, 128, 2)},
    )


def _body() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "fault"}],
        "stream": False,
        "max_tokens": 8,
    }


def _check_terminal(app: Any, request_id: str, expected: str) -> dict[str, Any]:
    runtime = app.state.gateway_runtime
    snapshot = runtime.registry.get(request_id)
    terminal_states = {"FINISHED", "CANCELLED", "TIMED_OUT", "REJECTED", "FAILED"}
    terminal_events = [
        event for event in snapshot.events if event.state.value in terminal_states
    ]
    admission = runtime.admission.snapshot()
    if snapshot.state.value != expected:
        raise RuntimeError(
            f"{request_id}: expected {expected}, got {snapshot.state.value}"
        )
    if len(terminal_events) != 1:
        raise RuntimeError(f"{request_id}: expected one terminal transition")
    if admission.reserved_blocks != 0 or admission.active_sequences != 0:
        raise RuntimeError(f"{request_id}: admission reservation leaked")
    return {
        "request_id": request_id,
        "terminal_state": snapshot.state.value,
        "terminal_transitions": len(terminal_events),
        "reserved_blocks": admission.reserved_blocks,
        "active_sequences": admission.active_sequences,
    }


def _run_case(
    name: str, mode: str, expected_state: str, *, cancel: bool = False
) -> dict[str, Any]:
    backend = _FaultBackend(mode)
    app = create_app(_settings(), backend=backend, token_counter=_Counter())
    client = TestClient(app)
    request_id = f"chaos-{name}"
    headers = {"X-Tenant-ID": "team-a", "X-Request-ID": request_id}
    if mode == "timeout":
        headers["X-Deadline-Ms"] = "10"
    if cancel:
        app.state.gateway_runtime.prepare(_body(), headers)
        asyncio.run(
            app.state.gateway_runtime.cancel(
                request_id, "team-a", reason="disconnect"
            )
        )
        return _check_terminal(app, request_id, expected_state) | {
            "http_status": None, "error_code": None
        }
    response = client.post("/v1/chat/completions", headers=headers, json=_body())
    payload = response.json()
    return _check_terminal(app, request_id, expected_state) | {
        "http_status": response.status_code,
        "error_code": payload.get("error", {}).get("code"),
    }


def run() -> dict[str, Any]:
    cases = [
        _run_case("cancel", "normal", "CANCELLED", cancel=True),
        _run_case("disconnect", "normal", "CANCELLED", cancel=True),
        _run_case("timeout", "timeout", "TIMED_OUT"),
        _run_case("exception", "exception", "FAILED"),
        _run_case("oom", "oom", "FAILED"),
    ]
    return {
        "artifact_type": "cachepilot_chaos_validation",
        "schema_version": 1,
        "status": "PASS",
        "cases": cases,
        "validated_in_process": True,
        "note": (
            "OOM case uses an injected CUDA out-of-memory exception; repeat on GPU "
            "for hardware evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/phase1-chaos/chaos_report.json")
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"CHAOS_VALIDATION=PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
