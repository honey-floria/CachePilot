import json
import random
import threading
import unittest
from pathlib import Path

from cachepilot.cache import PrefixScopeKey, TenantPrefixIndex
from cachepilot.gateway.contracts import ChatMessage, ValidatedChatRequest
from cachepilot.runtime.admission import (
    AdmissionStatus,
    StrictAdmissionConfig,
    StrictAdmissionController,
    TenantAdmissionLimits,
)
from cachepilot.runtime.deadlines import (
    DeadlinePolicy,
    RequestDeadlineManager,
    TimeoutReason,
)
from cachepilot.runtime.kv_planner import KVModelSpec, KVPlanner
from cachepilot.runtime.registry import RequestRegistry
from cachepilot.runtime.resources import ResourceLeaseManager
from cachepilot.runtime.state_machine import RequestState, TERMINAL_STATES


class FakeClock:
    def __init__(self):
        self.now_ns = 0

    def advance_ms(self, value):
        self.now_ns += value * 1_000_000

    def __call__(self):
        return self.now_ns


class RuntimeInvariantTests(unittest.TestCase):
    def make_request(self, request_id):
        return ValidatedChatRequest(
            request_id=request_id,
            tenant_id="tenant-a",
            priority="interactive",
            deadline_ms=30_000,
            idempotency_key=None,
            model="model-a",
            messages=(ChatMessage(role="user", content="hello"),),
            stream=True,
            max_tokens=32,
        )

    def advance_to(self, registry, request_id, target):
        states = (
            RequestState.TOKENIZED,
            RequestState.QUEUED,
            RequestState.ADMITTED,
            RequestState.ROUTED,
            RequestState.EXECUTING,
        )
        for index, state in enumerate(states, 1):
            registry.transition(request_id, state, "advance-{0}".format(index))
            if state is target:
                break

    def test_repeated_terminal_races_have_one_winner_and_release_once(self):
        for iteration in range(32):
            registry = RequestRegistry(resource_capacity_blocks=4)
            request_id = "race-{0}".format(iteration)
            registry.register(self.make_request(request_id))
            self.advance_to(registry, request_id, RequestState.ADMITTED)
            registry.reserve(request_id, 2)
            registry.transition(request_id, RequestState.ROUTED, "advance-route")
            registry.transition(request_id, RequestState.EXECUTING, "advance-execute")
            barrier = threading.Barrier(3)
            outcomes = []
            lock = threading.Lock()

            def terminate(target):
                barrier.wait()
                try:
                    result = registry.transition(
                        request_id, target, "terminal-{0}".format(target.value)
                    )
                    outcome = result.applied
                except ValueError:
                    outcome = False
                with lock:
                    outcomes.append(outcome)

            threads = [
                threading.Thread(target=terminate, args=(RequestState.CANCELLED,)),
                threading.Thread(target=terminate, args=(RequestState.FINISHED,)),
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2)

            snapshot = registry.get(request_id)
            lease = registry.resource_snapshot(request_id)
            self.assertEqual(2, len(outcomes))
            self.assertEqual(1, sum(outcomes))
            self.assertIn(snapshot.state, TERMINAL_STATES)
            self.assertEqual(1, sum(event.state in TERMINAL_STATES for event in snapshot.events))
            self.assertEqual(0, lease.logical_blocks)
            self.assertEqual(0, registry._resources.total_logical_blocks)

    def test_deadline_is_inclusive_and_request_budget_wins_ties(self):
        clock = FakeClock()
        released = []
        manager = RequestDeadlineManager(
            DeadlinePolicy(queue_timeout_ms=10, execution_timeout_ms=20),
            lambda request_id: released.append(request_id) or True,
            monotonic_ns=clock,
        )
        manager.track_queued("queue-boundary", request_deadline_ms=100)
        clock.advance_ms(9)
        self.assertEqual((), manager.expire_due())
        clock.advance_ms(1)
        event = manager.expire_due()[0]
        self.assertEqual(TimeoutReason.QUEUE_DEADLINE, event.reason)

        manager.track_queued("request-boundary", request_deadline_ms=10)
        clock.advance_ms(10)
        event = manager.expire_due()[0]
        self.assertEqual(TimeoutReason.REQUEST_DEADLINE, event.reason)
        self.assertEqual(["queue-boundary", "request-boundary"], released)

    def test_tenant_limits_remain_hard_under_deterministic_submit_release_trace(self):
        planner = KVPlanner(KVModelSpec(2, 2, 8, "float16", 8, 128))
        controller = StrictAdmissionController(
            planner,
            StrictAdmissionConfig(
                total_blocks=40,
                safety_blocks=2,
                max_active_sequences=20,
                max_queued_requests=40,
                tenant_limits={
                    "tenant-a": TenantAdmissionLimits(2, 20, 8),
                    "tenant-b": TenantAdmissionLimits(3, 24, 8),
                },
            ),
        )
        rng = random.Random(20260924)
        accepted = []
        for index in range(100):
            tenant = "tenant-a" if index % 2 == 0 else "tenant-b"
            decision = controller.submit(
                "property-{0}".format(index), tenant, rng.randrange(0, 5), 1
            )
            if decision.status is AdmissionStatus.ADMITTED:
                accepted.append(decision.request_id)
            snapshot = controller.snapshot()
            for tenant_id, max_sequences, max_tokens in (
                ("tenant-a", 2, 20),
                ("tenant-b", 3, 24),
            ):
                active_sequences = dict(snapshot.tenant_active_sequences).get(tenant_id, 0)
                active_tokens = dict(snapshot.tenant_active_tokens).get(tenant_id, 0)
                self.assertLessEqual(active_sequences, max_sequences)
                self.assertLessEqual(active_tokens, max_tokens)
            if accepted and index % 3 == 0:
                request_id = accepted.pop(0)
                self.assertTrue(controller.release(request_id))
                self.assertFalse(controller.release(request_id))
        for request_id in accepted:
            controller.release(request_id)
        self.assertEqual(0, controller.snapshot().active_sequences)
        self.assertEqual(0, controller.snapshot().reserved_blocks)

    def test_duplicate_events_release_and_cache_invalidation_are_idempotent(self):
        manager = ResourceLeaseManager(capacity_blocks=3)
        manager.reserve("request-1", 2)
        first = manager.release("request-1")
        second = manager.release("request-1")
        self.assertEqual(first, second)
        self.assertEqual(0, manager.total_logical_blocks)

        index = TenantPrefixIndex()
        key = PrefixScopeKey.create(
            tenant_id="tenant-a",
            model_id="model-a",
            model_revision="rev-a",
            tokenizer_revision="tok-a",
            quantization_config="none",
            tokenized_prefix=(1, 2, 3),
        )
        self.assertTrue(index.record(key))
        self.assertFalse(index.record(key))
        self.assertEqual(1, index.invalidate_scope(key.scope))
        self.assertEqual(0, index.invalidate_scope(key.scope))
        self.assertFalse(index.lookup(key).logical_hit)

    def test_reproducible_cancel_finish_regression_trace(self):
        trace_path = Path(__file__).parents[1] / "regression" / "traces" / "cancel_finish_race.json"
        trace = json.loads(trace_path.read_text())
        self.assertEqual(1, trace["trace_version"])
        rng = random.Random(trace["seed"])
        outcomes = []
        for round_index in range(trace["rounds"]):
            registry = RequestRegistry(resource_capacity_blocks=trace["capacity_blocks"])
            request_id = "{0}-{1}".format(trace["request_id_prefix"], round_index)
            registry.register(self.make_request(request_id))
            self.advance_to(registry, request_id, RequestState.ADMITTED)
            registry.reserve(request_id, trace["reserved_blocks"])
            registry.transition(request_id, RequestState.ROUTED, "advance-route")
            registry.transition(request_id, RequestState.EXECUTING, "advance-execute")
            barrier = threading.Barrier(3)
            applied = []
            lock = threading.Lock()

            def run(target):
                barrier.wait()
                try:
                    result = registry.transition(request_id, target, "race-" + target.value)
                    with lock:
                        applied.append(result.applied)
                except ValueError:
                    with lock:
                        applied.append(False)

            targets = [RequestState.CANCELLED, RequestState.FINISHED]
            rng.shuffle(targets)
            threads = [threading.Thread(target=run, args=(target,)) for target in targets]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2)
            outcomes.append((sum(applied), registry.resource_snapshot(request_id).logical_blocks))
        expected = trace["expected"]
        self.assertEqual(
            [(expected["terminal_winners"], expected["remaining_logical_blocks"])]
            * trace["rounds"],
            outcomes,
        )


if __name__ == "__main__":
    unittest.main()
