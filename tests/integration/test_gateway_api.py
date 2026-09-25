import json
import unittest

from fastapi.testclient import TestClient

from cachepilot.gateway.api import GatewaySettings, create_app
from cachepilot.gateway.backends import DeterministicChatBackend
from cachepilot.runtime.admission import TenantAdmissionLimits
from cachepilot.runtime.registry import RequestNotFoundError


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class FixedTokenCounter:
    def __init__(self, prompt_tokens=3):
        self.prompt_tokens = prompt_tokens

    def count_prompt_tokens(self, request):
        return self.prompt_tokens


def settings(*, max_active_tokens=128):
    return GatewaySettings(
        model_id=MODEL,
        context_limit=256,
        total_kv_blocks=32,
        safety_kv_blocks=2,
        max_active_sequences=2,
        max_queued_requests=2,
        tenant_limits={
            "team-a": TenantAdmissionLimits(2, max_active_tokens, 2),
            "team-b": TenantAdmissionLimits(2, max_active_tokens, 2),
        },
    )


def body(*, stream=True, max_tokens=8):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
        "max_tokens": max_tokens,
    }


class GatewayAPITests(unittest.TestCase):
    def make_client(self, *, selected_settings=None):
        app = create_app(
            selected_settings or settings(),
            token_counter=FixedTokenCounter(),
        )
        return app, TestClient(app)

    def test_health_ready_and_metrics(self):
        app, client = self.make_client()

        self.assertEqual(200, client.get("/healthz").status_code)
        self.assertEqual("ready", client.get("/readyz").json()["status"])
        metrics = client.get("/metrics")
        self.assertEqual(200, metrics.status_code)
        self.assertIn("cachepilot_gateway_up 1", metrics.text)
        self.assertIsNotNone(app.state.gateway_runtime)

    def test_non_streaming_chat_and_tenant_scoped_query(self):
        _, client = self.make_client()
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "team-a", "X-Request-ID": "normal-1"},
            json=body(stream=False),
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("chat.completion", payload["object"])
        self.assertEqual("Echo: Hello", payload["choices"][0]["message"]["content"])
        self.assertEqual("normal-1", response.headers["X-Request-ID"])

        query = client.get(
            "/v1/requests/normal-1",
            headers={"X-Tenant-ID": "team-a"},
        )
        self.assertEqual(200, query.status_code)
        self.assertEqual("FINISHED", query.json()["state"])
        hidden = client.get(
            "/v1/requests/normal-1",
            headers={"X-Tenant-ID": "team-b"},
        )
        self.assertEqual(404, hidden.status_code)
        self.assertEqual("request_not_found", hidden.json()["error"]["code"])

    def test_sse_is_parseable_and_ends_with_done(self):
        _, client = self.make_client()
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "X-Tenant-ID": "team-a",
                "X-Request-ID": "stream-1",
                "Accept": "text/event-stream",
            },
            json=body(stream=True),
        ) as response:
            self.assertEqual(200, response.status_code)
            self.assertTrue(
                response.headers["content-type"].startswith("text/event-stream")
            )
            lines = [line for line in response.iter_lines() if line]

        data_lines = [line for line in lines if line.startswith("data: ")]
        self.assertEqual("data: [DONE]", data_lines[-1])
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in data_lines[:-1]
        ]
        self.assertEqual("assistant", chunks[0]["choices"][0]["delta"]["role"])
        self.assertEqual(
            "Echo: Hello",
            "".join(
                chunk["choices"][0]["delta"].get("content", "")
                for chunk in chunks
            ),
        )
        self.assertEqual("stop", chunks[-1]["choices"][0]["finish_reason"])
        self.assertIn("usage", chunks[-1])

    def test_invalid_request_never_registers_or_reserves(self):
        app, client = self.make_client()
        runtime = app.state.gateway_runtime
        invalid = body()
        invalid["tools"] = []

        before = runtime.admission.snapshot()
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "team-a", "X-Request-ID": "invalid-1"},
            json=invalid,
        )
        after = runtime.admission.snapshot()

        self.assertEqual(400, response.status_code)
        self.assertEqual("unknown_field", response.json()["error"]["code"])
        self.assertEqual(0, before.reserved_blocks)
        self.assertEqual(0, after.reserved_blocks)
        self.assertEqual(0, after.active_sequences)
        with self.assertRaises(RequestNotFoundError):
            runtime.registry.get("invalid-1")

    def test_unknown_tenant_is_rejected_before_reservation(self):
        app, client = self.make_client()
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "unknown", "X-Request-ID": "unknown-1"},
            json=body(),
        )

        self.assertEqual(403, response.status_code)
        self.assertEqual("tenant_not_authorized", response.json()["error"]["code"])
        snapshot = app.state.gateway_runtime.admission.snapshot()
        self.assertEqual(0, snapshot.reserved_blocks)
        self.assertEqual(0, snapshot.active_sequences)

    def test_quota_rejection_releases_all_admission_state(self):
        app, client = self.make_client(
            selected_settings=settings(max_active_tokens=4)
        )
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "team-a", "X-Request-ID": "quota-1"},
            json=body(max_tokens=2),
        )

        self.assertEqual(429, response.status_code)
        self.assertEqual("tenant_quota_exceeded", response.json()["error"]["code"])
        snapshot = app.state.gateway_runtime.admission.snapshot()
        self.assertEqual(0, snapshot.reserved_blocks)
        self.assertEqual(0, snapshot.active_sequences)
        status = client.get(
            "/v1/requests/quota-1",
            headers={"X-Tenant-ID": "team-a"},
        )
        self.assertEqual("REJECTED", status.json()["state"])

    def test_deadline_finishes_as_timeout_and_releases_reservation(self):
        selected_settings = settings()
        app = create_app(
            selected_settings,
            backend=DeterministicChatBackend(token_delay_seconds=0.01),
            token_counter=FixedTokenCounter(),
        )
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            headers={
                "X-Tenant-ID": "team-a",
                "X-Request-ID": "deadline-1",
                "X-Deadline-Ms": "1",
            },
            json=body(stream=False),
        )

        self.assertEqual(504, response.status_code)
        self.assertEqual("deadline_exceeded", response.json()["error"]["code"])
        self.assertEqual(
            "TIMED_OUT",
            client.get(
                "/v1/requests/deadline-1",
                headers={"X-Tenant-ID": "team-a"},
            ).json()["state"],
        )
        self.assertEqual(0, app.state.gateway_runtime.admission.snapshot().reserved_blocks)

    def test_cancel_is_idempotent_and_terminal_requests_conflict(self):
        app, client = self.make_client()
        runtime = app.state.gateway_runtime
        runtime.prepare(
            body(),
            {"X-Tenant-ID": "team-a", "X-Request-ID": "cancel-1"},
        )

        first = client.post(
            "/v1/requests/cancel-1/cancel",
            headers={"X-Tenant-ID": "team-a"},
        )
        second = client.post(
            "/v1/requests/cancel-1/cancel",
            headers={"X-Tenant-ID": "team-a"},
        )

        self.assertEqual(202, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual("CANCELLED", second.json()["state"])
        self.assertEqual(0, runtime.admission.snapshot().reserved_blocks)

        finished = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "team-a", "X-Request-ID": "finished-1"},
            json=body(stream=False),
        )
        self.assertEqual(200, finished.status_code)
        conflict = client.post(
            "/v1/requests/finished-1/cancel",
            headers={"X-Tenant-ID": "team-a"},
        )
        self.assertEqual(409, conflict.status_code)
        self.assertEqual("request_terminal", conflict.json()["error"]["code"])


if __name__ == "__main__":
    unittest.main()
