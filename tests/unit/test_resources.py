import unittest

from cachepilot.gateway.contracts import ChatMessage, ValidatedChatRequest
from cachepilot.runtime.registry import RequestRegistry
from cachepilot.runtime.resources import (
    CapacityExceededError,
    LeaseReleasedError,
    ResourceLeaseManager,
)
from cachepilot.runtime.state_machine import RequestState


class ResourceLeaseManagerTests(unittest.TestCase):
    def test_reserve_grow_and_release_restore_capacity(self):
        manager = ResourceLeaseManager(capacity_blocks=10)
        first = manager.reserve("request-1", 3)
        grown = manager.grow("request-1", 2)
        attached = manager.attach_physical_handle("request-1", "gpu-handle-1")
        released = manager.release("request-1")
        repeated = manager.release("request-1")

        self.assertEqual(3, first.logical_blocks)
        self.assertEqual(5, grown.logical_blocks)
        self.assertEqual(5, grown.peak_logical_blocks)
        self.assertEqual(("gpu-handle-1",), attached.physical_handles)
        self.assertEqual(0, released.logical_blocks)
        self.assertEqual(5, released.peak_logical_blocks)
        self.assertEqual((), released.physical_handles)
        self.assertTrue(released.released)
        self.assertEqual(released, repeated)
        self.assertEqual(0, manager.total_logical_blocks)

    def test_capacity_is_shared_by_logical_blocks_only(self):
        manager = ResourceLeaseManager(capacity_blocks=4)
        manager.reserve("request-1", 3)
        manager.attach_physical_handle("request-1", "physical-1")

        with self.assertRaises(CapacityExceededError):
            manager.reserve("request-2", 2)

        manager.release("request-1")
        self.assertEqual(0, manager.total_logical_blocks)

    def test_released_lease_cannot_grow_or_attach(self):
        manager = ResourceLeaseManager()
        manager.reserve("request-1", 1)
        manager.release("request-1")

        with self.assertRaises(LeaseReleasedError):
            manager.grow("request-1", 1)
        with self.assertRaises(LeaseReleasedError):
            manager.attach_physical_handle("request-1", "physical-1")


class RegistryResourceOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.registry = RequestRegistry(resource_capacity_blocks=20)

    def make_request(self, request_id="request-1"):
        return ValidatedChatRequest(
            request_id=request_id,
            tenant_id="team-a",
            priority="interactive",
            deadline_ms=30_000,
            idempotency_key=None,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=(ChatMessage(role="user", content="Hello"),),
            stream=True,
            max_tokens=256,
        )

    def admit(self, request_id="request-1"):
        self.registry.register(self.make_request(request_id))
        for index, state in enumerate(
            (
                RequestState.TOKENIZED,
                RequestState.QUEUED,
                RequestState.ADMITTED,
            ),
            1,
        ):
            self.registry.transition(request_id, state, "event_{0}".format(index))

    def test_only_admitted_requests_can_reserve(self):
        self.registry.register(self.make_request())
        with self.assertRaises(ValueError):
            self.registry.reserve("request-1", 2)

        self.admit()
        reservation = self.registry.reserve("request-1", 2)
        self.assertEqual(2, reservation.logical_blocks)

    def test_terminal_paths_release_resources_to_baseline(self):
        terminals = (
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.TIMED_OUT,
            RequestState.REJECTED,
            RequestState.FAILED,
        )
        for terminal in terminals:
            with self.subTest(terminal=terminal):
                registry = RequestRegistry(resource_capacity_blocks=20)
                request_id = "request-{0}".format(terminal.value.lower())
                self.registry = registry
                self.admit(request_id)
                registry.reserve(request_id, 4)
                registry.grow_reservation(request_id, 2)
                registry.attach_physical_handle(request_id, "physical-1")
                if terminal is RequestState.FINISHED:
                    registry.transition(request_id, RequestState.ROUTED, "route")
                    registry.transition(request_id, RequestState.EXECUTING, "execute")
                registry.transition(request_id, terminal, "terminal")

                snapshot = registry.resource_snapshot(request_id)
                self.assertEqual(0, snapshot.logical_blocks)
                self.assertEqual(0, registry._resources.total_logical_blocks)
                self.assertTrue(snapshot.released)

    def test_release_is_idempotent_after_terminal_transition(self):
        self.admit()
        self.registry.reserve("request-1", 3)
        self.registry.transition("request-1", RequestState.CANCELLED, "cancel")

        first = self.registry.release_resources("request-1")
        second = self.registry.release_resources("request-1")

        self.assertEqual(first, second)
        self.assertEqual(0, first.logical_blocks)


if __name__ == "__main__":
    unittest.main()
