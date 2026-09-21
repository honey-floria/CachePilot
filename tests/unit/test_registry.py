import threading
import unittest

from cachepilot.gateway.contracts import (
    ChatMessage,
    ClaimStatus,
    ValidatedChatRequest,
)
from cachepilot.runtime.registry import RequestNotFoundError, RequestRegistry
from cachepilot.runtime.state_machine import (
    InvalidTransitionError,
    RequestState,
    TERMINAL_STATES,
)


class RequestRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = RequestRegistry()

    def make_request(
        self,
        request_id="request-1",
        *,
        tenant_id="team-a",
        idempotency_key="operation-1",
        content="Hello",
    ):
        return ValidatedChatRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            priority="interactive",
            deadline_ms=30_000,
            idempotency_key=idempotency_key,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=(ChatMessage(role="user", content=content),),
            stream=True,
            max_tokens=256,
        )

    def advance_to_executing(self, request_id="request-1"):
        states = (
            RequestState.TOKENIZED,
            RequestState.QUEUED,
            RequestState.ADMITTED,
            RequestState.ROUTED,
            RequestState.EXECUTING,
        )
        for index, state in enumerate(states, 1):
            self.registry.transition(
                request_id,
                state,
                "event_{0}".format(index),
            )

    def test_registers_and_queries_state_and_event_log(self):
        request = self.make_request()

        result = self.registry.register(request, "event_received")
        snapshot = self.registry.get(request.request_id)

        self.assertEqual(ClaimStatus.ACCEPTED, result.status)
        self.assertEqual(RequestState.RECEIVED, snapshot.state)
        self.assertFalse(snapshot.terminal)
        self.assertEqual("event_received", snapshot.events[0].event_id)
        self.assertEqual(request, snapshot.request)

    def test_duplicate_request_id_is_rejected_without_idempotency_key(self):
        request = self.make_request(idempotency_key=None)
        self.registry.register(request)

        result = self.registry.register(request)

        self.assertEqual(ClaimStatus.REQUEST_ID_CONFLICT, result.status)
        self.assertEqual(request.request_id, result.request_id)

    def test_idempotency_key_returns_original_in_progress_request(self):
        original = self.make_request("request-1")
        duplicate = self.make_request("request-2")
        self.registry.register(original)

        result = self.registry.register(duplicate)

        self.assertEqual(ClaimStatus.IDEMPOTENCY_IN_PROGRESS, result.status)
        self.assertEqual(original.request_id, result.request_id)
        with self.assertRaises(RequestNotFoundError):
            self.registry.get(duplicate.request_id)

    def test_idempotency_key_rejects_a_different_fingerprint(self):
        self.registry.register(self.make_request("request-1"))

        result = self.registry.register(
            self.make_request("request-2", content="Different")
        )

        self.assertEqual(ClaimStatus.IDEMPOTENCY_KEY_CONFLICT, result.status)
        self.assertEqual("request-1", result.request_id)

    def test_terminal_idempotent_request_is_not_replayed(self):
        request = self.make_request("request-1")
        self.registry.register(request)
        self.registry.transition(
            request.request_id,
            RequestState.CANCELLED,
            "event_cancelled",
        )

        result = self.registry.register(self.make_request("request-2"))

        self.assertEqual(
            ClaimStatus.IDEMPOTENCY_REPLAY_UNAVAILABLE,
            result.status,
        )
        self.assertEqual(request.request_id, result.request_id)

    def test_idempotency_key_is_tenant_scoped_and_queryable(self):
        first = self.make_request("request-1", tenant_id="team-a")
        second = self.make_request("request-2", tenant_id="team-b")

        self.assertEqual(
            ClaimStatus.ACCEPTED,
            self.registry.register(first).status,
        )
        self.assertEqual(
            ClaimStatus.ACCEPTED,
            self.registry.register(second).status,
        )
        snapshot = self.registry.get_by_idempotency_key(
            "team-a",
            "operation-1",
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual("request-1", snapshot.request_id)

    def test_tenant_query_does_not_reveal_cross_tenant_request(self):
        request = self.make_request()
        self.registry.register(request)

        self.assertIsNotNone(
            self.registry.get_for_tenant(request.request_id, "team-a")
        )
        self.assertIsNone(
            self.registry.get_for_tenant(request.request_id, "team-b")
        )
        self.assertIsNone(
            self.registry.get_for_tenant("missing", "team-a")
        )

    def test_registry_updates_state_event_log_and_token_count(self):
        request = self.make_request()
        self.registry.register(request)
        self.advance_to_executing()
        self.assertTrue(
            self.registry.record_token_emission(request.request_id, "token-1")
        )

        snapshot = self.registry.get(request.request_id)

        self.assertEqual(RequestState.EXECUTING, snapshot.state)
        self.assertEqual(1, snapshot.emitted_token_count)
        self.assertEqual(6, len(snapshot.events))

    def test_concurrent_terminal_events_produce_one_terminal_state(self):
        request = self.make_request()
        self.registry.register(request)
        self.advance_to_executing()
        barrier = threading.Barrier(3)
        outcomes = []
        outcomes_lock = threading.Lock()

        def terminate(target):
            barrier.wait()
            try:
                result = self.registry.transition(
                    request.request_id,
                    target,
                    "terminal_{0}".format(target.value),
                )
                outcome = ("applied", result.current_state)
            except InvalidTransitionError:
                outcome = ("rejected", target)
            with outcomes_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=terminate, args=(target,))
            for target in (
                RequestState.CANCELLED,
                RequestState.FINISHED,
                RequestState.FAILED,
            )
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        snapshot = self.registry.get(request.request_id)
        applied = [outcome for outcome in outcomes if outcome[0] == "applied"]
        terminal_events = [
            event for event in snapshot.events if event.state in TERMINAL_STATES
        ]
        self.assertEqual(3, len(outcomes))
        self.assertEqual(1, len(applied))
        self.assertEqual(1, len(terminal_events))
        self.assertTrue(snapshot.terminal)
        self.assertEqual(applied[0][1], snapshot.state)


if __name__ == "__main__":
    unittest.main()
