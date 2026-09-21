import unittest

from cachepilot.runtime.admission import (
    AdmissionStatus,
    StrictAdmissionConfig,
    StrictAdmissionController,
    TenantAdmissionLimits,
)
from cachepilot.runtime.deadlines import (
    DeadlinePhase,
    DeadlinePolicy,
    RequestDeadlineManager,
    TimeoutReason,
)
from cachepilot.runtime.kv_planner import KVModelSpec, KVPlanner


class FakeClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def advance_ms(self, milliseconds):
        self.now_ns += milliseconds * 1_000_000


class RequestDeadlineManagerTests(unittest.TestCase):
    def make_controller(self, max_active_sequences=1):
        planner = KVPlanner(KVModelSpec(2, 2, 8, "float16", 16, 256))
        return StrictAdmissionController(
            planner,
            StrictAdmissionConfig(
                total_blocks=10,
                safety_blocks=1,
                max_active_sequences=max_active_sequences,
                max_queued_requests=2,
                tenant_limits={
                    "team-a": TenantAdmissionLimits(4, 256, 2),
                },
                retry_after_ms=750,
            ),
        )

    def test_queue_deadline_releases_queued_request(self):
        clock = FakeClock()
        controller = self.make_controller()
        controller.submit("active", "team-a", 1, 1)
        queued = controller.submit("queued", "team-a", 1, 1)
        self.assertEqual(AdmissionStatus.QUEUED, queued.status)
        manager = RequestDeadlineManager(
            DeadlinePolicy(queue_timeout_ms=50, execution_timeout_ms=100),
            controller.release,
            monotonic_ns=clock,
        )
        manager.track_queued("queued", request_deadline_ms=500)

        clock.advance_ms(50)
        events = manager.expire_due()

        self.assertEqual(1, len(events))
        self.assertEqual(TimeoutReason.QUEUE_DEADLINE, events[0].reason)
        self.assertTrue(events[0].resources_released)
        self.assertEqual(0, controller.snapshot().queued_requests)

    def test_execution_deadline_releases_active_reservation(self):
        clock = FakeClock()
        controller = self.make_controller()
        controller.submit("request-1", "team-a", 16, 16)
        manager = RequestDeadlineManager(
            DeadlinePolicy(queue_timeout_ms=50, execution_timeout_ms=100),
            controller.release,
            monotonic_ns=clock,
        )
        manager.track_queued("request-1", request_deadline_ms=500)
        self.assertIsNone(manager.mark_executing("request-1"))

        clock.advance_ms(100)
        event = manager.expire_due()[0]

        self.assertEqual(DeadlinePhase.EXECUTING, event.phase)
        self.assertEqual(TimeoutReason.EXECUTION_DEADLINE, event.reason)
        self.assertEqual(0, controller.snapshot().reserved_blocks)

    def test_request_deadline_caps_phase_deadlines(self):
        clock = FakeClock()
        released = []
        manager = RequestDeadlineManager(
            DeadlinePolicy(queue_timeout_ms=100, execution_timeout_ms=100),
            lambda request_id: not released.append(request_id),
            monotonic_ns=clock,
        )
        manager.track_queued("request-1", request_deadline_ms=30)

        clock.advance_ms(30)
        event = manager.expire_due()[0]

        self.assertEqual(TimeoutReason.REQUEST_DEADLINE, event.reason)
        self.assertEqual(["request-1"], released)

    def test_mark_executing_cannot_revive_expired_queue_request(self):
        clock = FakeClock()
        released = []
        manager = RequestDeadlineManager(
            DeadlinePolicy(queue_timeout_ms=10, execution_timeout_ms=100),
            lambda request_id: not released.append(request_id),
            monotonic_ns=clock,
        )
        manager.track_queued("request-1", request_deadline_ms=500)
        clock.advance_ms(11)

        event = manager.mark_executing("request-1")

        self.assertEqual(TimeoutReason.QUEUE_DEADLINE, event.reason)
        self.assertEqual(["request-1"], released)
        self.assertEqual(0, manager.snapshot().queued_requests)

    def test_queue_full_rejection_contains_retry_suggestion(self):
        controller = self.make_controller()
        controller.submit("active", "team-a", 1, 1)
        controller.submit("queued-1", "team-a", 1, 1)
        controller.submit("queued-2", "team-a", 1, 1)

        rejected = controller.submit("rejected", "team-a", 1, 1)

        self.assertEqual(AdmissionStatus.REJECTED, rejected.status)
        self.assertEqual(750, rejected.retry_after_ms)


if __name__ == "__main__":
    unittest.main()
