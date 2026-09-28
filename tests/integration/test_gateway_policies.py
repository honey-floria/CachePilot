import asyncio
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient

from cachepilot.cache.prefix_index import PrefixScopeKey
from cachepilot.gateway.api import GatewaySettings, create_app
from cachepilot.gateway.backends import GeneratedText
from cachepilot.runtime.admission import TenantAdmissionLimits


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TokenCounter:
    revision = "a" * 40

    def count_prompt_tokens(self, request):
        return len(request.messages[0].content)

    def prefix_key(self, request):
        return PrefixScopeKey.create(
            tenant_id=request.tenant_id,
            model_id=request.model,
            model_revision=self.revision,
            tokenizer_revision=self.revision,
            quantization_config="none",
            tokenized_prefix=map(ord, request.messages[0].content),
        )


class OutputBackend:
    def __init__(self):
        self.order = []
        self.active = 0
        self.peak = 0
        self.delay = 0
        self.cancel_reasons = {}

    async def is_ready(self):
        return True

    async def cancel(self, request_id, reason="explicit"):
        self.cancel_reasons[request_id] = reason

    async def generate(self, request):
        self.order.append(request.request_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            yield GeneratedText(
                "answer", token_count=24 if "tail" in request.request_id else 2
            )
        finally:
            self.active -= 1


def config(**overrides):
    return replace(
        GatewaySettings(
            model_id=MODEL,
            execution_slots=1,
            tenant_limits={
                name: TenantAdmissionLimits(8, 8192, 32)
                for name in ("team-a", "team-b")
            },
        ),
        **overrides,
    )


def payload(tokens=64, stream=False, content="abc"):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": tokens,
        "stream": stream,
    }


def headers(request_id, tenant="team-a", deadline=30000):
    return {
        "X-Request-ID": request_id,
        "X-Tenant-ID": tenant,
        "X-Deadline-Ms": str(deadline),
    }


class GatewayPolicyTests(unittest.TestCase):
    def test_adaptive_learns_stream_and_nonstream_and_grows_tail(self):
        app = create_app(
            config(admission_strategy="adaptive"),
            backend=OutputBackend(),
            token_counter=TokenCounter(),
        )
        with TestClient(app) as client:
            for index in range(3):
                response = client.post(
                    "/v1/chat/completions",
                    headers=headers(f"warm-{index}"),
                    json=payload(stream=index % 2 == 0),
                )
                self.assertEqual(response.status_code, 200)
            response = client.post(
                "/v1/chat/completions", headers=headers("tail"), json=payload()
            )
            self.assertEqual(response.status_code, 200)
            query = client.get("/v1/requests/tail", headers=headers("tail")).json()
        self.assertFalse(query["policy"]["fallback_to_strict"])
        self.assertEqual(query["policy"]["estimated_output_tokens"], 10)
        self.assertEqual(query["ledger"]["reservation_peak_blocks"], 2)
        self.assertEqual(query["adaptive"]["estimation_count"], 4)
        self.assertEqual(query["adaptive"]["underestimation_count"], 1)
        self.assertEqual(
            app.state.gateway_runtime.admission.snapshot().reserved_blocks, 0
        )

    def test_adaptive_growth_failure_becomes_terminal_and_releases(self):
        app = create_app(
            config(
                admission_strategy="adaptive", total_kv_blocks=2, safety_kv_blocks=1
            ),
            backend=OutputBackend(),
            token_counter=TokenCounter(),
        )
        runtime = app.state.gateway_runtime
        for _ in range(3):
            runtime.admission.observe_output("team-a", 3, 2)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions", headers=headers("tail"), json=payload()
            )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(runtime.registry.get("tail").state.value, "FAILED")
        self.assertEqual(runtime.admission.snapshot().reserved_blocks, 0)
        self.assertEqual(
            runtime.ledger.snapshot("tail").error_code, "admission_capacity_exceeded"
        )

    def test_prefix_hits_require_completed_same_tenant_and_version(self):
        counter = TokenCounter()
        app = create_app(
            config(scheduler_strategy="wfq", prefix_mode="aware"),
            backend=OutputBackend(),
            token_counter=counter,
        )
        with TestClient(app) as client:
            hits = []
            for name, tenant in (
                ("cold", "team-a"),
                ("hit", "team-a"),
                ("other", "team-b"),
            ):
                self.assertEqual(
                    client.post(
                        "/v1/chat/completions",
                        headers=headers(name, tenant),
                        json=payload(),
                    ).status_code,
                    200,
                )
                query = client.get(
                    f"/v1/requests/{name}", headers=headers(name, tenant)
                ).json()
                hits.append(query["ledger"]["logical_hit"])
                self.assertIsNone(query["ledger"]["physical_hit"])
            counter.revision = "b" * 40
            client.post(
                "/v1/chat/completions", headers=headers("new-version"), json=payload()
            )
            query = client.get(
                "/v1/requests/new-version", headers=headers("new-version")
            ).json()
        self.assertEqual(hits, [False, True, False])
        self.assertFalse(query["ledger"]["logical_hit"])
        self.assertEqual(query["ledger"]["logical_hit_source"], "tenant_prefix_index")


class GatewaySchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiting_stream_never_starts_backend(self):
        class ConnectedClient:
            async def is_disconnected(self):
                return False

        backend = OutputBackend()
        backend.delay = 0.03
        runtime = create_app(
            config(), backend=backend, token_counter=TokenCounter()
        ).state.gateway_runtime
        active = runtime.prepare(payload(), headers("active"))
        waiting = runtime.prepare(payload(stream=True), headers("waiting"))

        async def collect():
            return [item async for item in runtime.stream(waiting, ConnectedClient())]

        active_task = asyncio.create_task(runtime.complete(active))
        await asyncio.sleep(0)
        stream_task = asyncio.create_task(collect())
        await asyncio.sleep(0)
        await runtime.cancel("waiting", "team-a")
        chunks = await stream_task
        await active_task
        self.assertIn("request_cancelled", "".join(chunks))
        self.assertEqual(backend.order, ["active"])
        self.assertEqual(runtime._executing_requests, set())
        self.assertEqual(runtime.admission.snapshot().reserved_blocks, 0)

    async def test_scheduler_owns_execution_order_and_single_slot(self):
        for strategy, expected in (
            ("fcfs", ["long-0", "long-1", "short"]),
            ("wfq", ["short", "long-0", "long-1"]),
        ):
            backend = OutputBackend()
            backend.delay = 0.005
            runtime = create_app(
                config(scheduler_strategy=strategy),
                backend=backend,
                token_counter=TokenCounter(),
            ).state.gateway_runtime
            prepared = [
                runtime.prepare(payload(tokens), headers(name, tenant))
                for name, tenant, tokens in (
                    ("long-0", "team-a", 64),
                    ("long-1", "team-a", 64),
                    ("short", "team-b", 8),
                )
            ]
            await asyncio.gather(*(runtime.complete(request) for request in prepared))
            self.assertEqual(backend.order, expected)
            self.assertEqual(backend.peak, 1)
            self.assertGreater(
                runtime.query(
                    expected[-1], "team-a" if strategy == "wfq" else "team-b"
                )["policy"]["scheduler_wait_ms"],
                5,
            )
            self.assertEqual(runtime._executing_requests, set())

    async def test_prefix_boost_reaches_real_dispatch_and_ledger(self):
        backend = OutputBackend()
        runtime = create_app(
            config(scheduler_strategy="wfq", prefix_mode="aware"),
            backend=backend,
            token_counter=TokenCounter(),
        ).state.gateway_runtime
        warm = runtime.prepare(
            payload(32, content="shared" * 10), headers("warm", "team-b")
        )
        await runtime.complete(warm)
        cold = runtime.prepare(payload(80, content="abc"), headers("cold"))
        hit = runtime.prepare(
            payload(32, content="shared" * 10), headers("hit", "team-b")
        )
        await asyncio.gather(runtime.complete(cold), runtime.complete(hit))
        self.assertTrue(runtime.query("hit", "team-b")["ledger"]["logical_hit"])
        self.assertEqual(
            runtime.query("hit", "team-b")["policy"]["logical_hit_tokens"], 60
        )
        self.assertTrue(runtime.query("hit", "team-b")["policy"]["cache_boosted"])
        self.assertEqual(backend.order, ["warm", "hit", "cold"])

    async def test_waiting_timeout_and_cancellation_leave_no_slots(self):
        backend = OutputBackend()
        backend.delay = 0.03
        runtime = create_app(
            config(), backend=backend, token_counter=TokenCounter()
        ).state.gateway_runtime
        active = runtime.prepare(payload(), headers("active"))
        timeout = runtime.prepare(payload(), headers("timeout", deadline=1))
        cancelled = runtime.prepare(payload(), headers("cancelled"))
        task = asyncio.create_task(runtime.complete(active))
        await asyncio.sleep(0)
        await runtime.cancel("cancelled", "team-a")
        results = await asyncio.gather(
            runtime.complete(timeout),
            runtime.complete(cancelled),
            return_exceptions=True,
        )
        await task
        self.assertTrue(all(isinstance(result, Exception) for result in results))
        self.assertEqual(runtime.registry.get("timeout").state.value, "TIMED_OUT")
        self.assertEqual(runtime.registry.get("cancelled").state.value, "CANCELLED")
        self.assertEqual(runtime.admission.snapshot().reserved_blocks, 0)
        self.assertEqual(runtime._executing_requests, set())
        self.assertEqual(runtime._schedule_events, {})
