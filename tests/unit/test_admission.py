import threading
import unittest

from cachepilot.runtime.admission import (
    AdmissionError,
    AdmissionReason,
    AdmissionStatus,
    StrictAdmissionConfig,
    StrictAdmissionController,
    TenantAdmissionLimits,
)
from cachepilot.runtime.kv_planner import KVModelSpec, KVPlanner


class StrictAdmissionControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        total_blocks=12,
        safety_blocks=2,
        max_active_sequences=3,
        max_queued_requests=4,
        tenant_active_sequences=2,
        tenant_active_tokens=128,
        tenant_queued_requests=2,
    ):
        planner = KVPlanner(
            KVModelSpec(
                num_hidden_layers=2,
                num_key_value_heads=2,
                head_dim=8,
                dtype="float16",
                block_size=16,
                context_limit=256,
            )
        )
        config = StrictAdmissionConfig(
            total_blocks=total_blocks,
            safety_blocks=safety_blocks,
            max_active_sequences=max_active_sequences,
            max_queued_requests=max_queued_requests,
            tenant_limits={
                "team-a": TenantAdmissionLimits(
                    max_active_sequences=tenant_active_sequences,
                    max_active_tokens=tenant_active_tokens,
                    max_queued_requests=tenant_queued_requests,
                ),
                "team-b": TenantAdmissionLimits(2, 128, 2),
            },
        )
        return StrictAdmissionController(planner, config)

    def test_strict_admission_reserves_prompt_plus_max_new_tokens(self):
        controller = self.make_controller()

        decision = controller.submit("request-1", "team-a", 33, 16)

        self.assertEqual(AdmissionStatus.ADMITTED, decision.status)
        self.assertEqual(AdmissionReason.ADMITTED, decision.reason)
        self.assertEqual(49, decision.plan.total_tokens)
        self.assertEqual(4, decision.plan.logical_blocks)
        snapshot = controller.snapshot()
        self.assertEqual(1, snapshot.active_sequences)
        self.assertEqual(4, snapshot.reserved_blocks)
        self.assertEqual((("team-a", 49),), snapshot.tenant_active_tokens)

    def test_safety_blocks_are_never_reservable(self):
        controller = self.make_controller(
            total_blocks=6,
            safety_blocks=2,
            tenant_active_tokens=256,
        )

        first = controller.submit("request-1", "team-a", 33, 16)
        second = controller.submit("request-2", "team-a", 1, 1)

        self.assertEqual(AdmissionStatus.ADMITTED, first.status)
        self.assertEqual(AdmissionStatus.QUEUED, second.status)
        self.assertEqual(AdmissionReason.KV_CAPACITY, second.reason)
        self.assertEqual(4, controller.snapshot().reserved_blocks)
        self.assertEqual(4, controller.snapshot().usable_blocks)

    def test_active_limit_queues_then_retry_after_release(self):
        controller = self.make_controller(
            max_active_sequences=1,
            tenant_active_sequences=1,
        )
        controller.submit("request-1", "team-a", 16, 16)

        queued = controller.submit("request-2", "team-a", 16, 16)
        self.assertEqual(AdmissionStatus.QUEUED, queued.status)
        self.assertEqual(AdmissionReason.MAX_ACTIVE_SEQUENCES, queued.reason)

        self.assertTrue(controller.release("request-1"))
        admitted = controller.retry_queued("request-2")
        self.assertEqual(AdmissionStatus.ADMITTED, admitted.status)
        self.assertEqual(0, controller.snapshot().queued_requests)
        self.assertEqual(1, controller.snapshot().active_sequences)

    def test_tenant_token_quota_is_enforced_across_active_requests(self):
        controller = self.make_controller(
            tenant_active_sequences=3,
            tenant_active_tokens=64,
        )
        controller.submit("request-1", "team-a", 24, 16)

        decision = controller.submit("request-2", "team-a", 16, 16)

        self.assertEqual(AdmissionStatus.QUEUED, decision.status)
        self.assertEqual(AdmissionReason.TENANT_ACTIVE_TOKENS, decision.reason)

    def test_tenant_concurrency_is_enforced_independently(self):
        controller = self.make_controller(
            max_active_sequences=3,
            tenant_active_sequences=1,
            tenant_active_tokens=256,
        )
        controller.submit("request-a1", "team-a", 1, 1)
        controller.submit("request-b1", "team-b", 1, 1)

        decision = controller.submit("request-a2", "team-a", 1, 1)

        self.assertEqual(AdmissionStatus.QUEUED, decision.status)
        self.assertEqual(AdmissionReason.TENANT_CONCURRENCY, decision.reason)

    def test_permanently_impossible_requests_are_rejected(self):
        controller = self.make_controller(
            total_blocks=5,
            safety_blocks=1,
            tenant_active_tokens=48,
        )

        too_many_blocks = controller.submit("request-1", "team-a", 64, 16)
        too_many_tenant_tokens = controller.submit("request-2", "team-a", 40, 16)
        context = controller.submit("request-3", "team-a", 250, 16)
        unknown_tenant = controller.submit("request-4", "team-c", 1, 1)

        self.assertEqual(
            AdmissionReason.REQUEST_EXCEEDS_KV_CAPACITY,
            too_many_blocks.reason,
        )
        self.assertEqual(
            AdmissionReason.TENANT_REQUEST_EXCEEDS_TOKEN_QUOTA,
            too_many_tenant_tokens.reason,
        )
        self.assertEqual(AdmissionReason.CONTEXT_LIMIT_EXCEEDED, context.reason)
        self.assertEqual(
            AdmissionReason.TENANT_NOT_CONFIGURED,
            unknown_tenant.reason,
        )

    def test_global_and_tenant_queues_are_bounded(self):
        controller = self.make_controller(
            max_active_sequences=1,
            max_queued_requests=2,
            tenant_active_sequences=1,
            tenant_queued_requests=1,
        )
        controller.submit("active-a", "team-a", 1, 1)

        queued_a = controller.submit("queued-a", "team-a", 1, 1)
        rejected_a = controller.submit("rejected-a", "team-a", 1, 1)
        queued_b = controller.submit("queued-b", "team-b", 1, 1)
        rejected_global = controller.submit("rejected-b", "team-b", 1, 1)

        self.assertEqual(AdmissionStatus.QUEUED, queued_a.status)
        self.assertEqual(AdmissionReason.TENANT_QUEUE_FULL, rejected_a.reason)
        self.assertEqual(AdmissionStatus.QUEUED, queued_b.status)
        self.assertEqual(AdmissionReason.QUEUE_FULL, rejected_global.reason)

    def test_release_is_idempotent_and_restores_all_counters(self):
        controller = self.make_controller()
        controller.submit("request-1", "team-a", 20, 12)

        self.assertTrue(controller.release("request-1"))
        self.assertFalse(controller.release("request-1"))
        snapshot = controller.snapshot()
        self.assertEqual(0, snapshot.active_sequences)
        self.assertEqual(0, snapshot.reserved_blocks)
        self.assertEqual((), snapshot.tenant_active_sequences)
        self.assertEqual((), snapshot.tenant_active_tokens)
        with self.assertRaises(AdmissionError):
            controller.submit("request-1", "team-a", 1, 1)

    def test_concurrent_submissions_cannot_exceed_hard_limits(self):
        controller = self.make_controller(
            total_blocks=10,
            safety_blocks=2,
            max_active_sequences=4,
            max_queued_requests=20,
            tenant_active_sequences=20,
            tenant_active_tokens=1000,
            tenant_queued_requests=20,
        )
        barrier = threading.Barrier(10)
        decisions = []
        decisions_lock = threading.Lock()

        def submit(index):
            barrier.wait()
            decision = controller.submit(
                "request-{0}".format(index), "team-a", 16, 16
            )
            with decisions_lock:
                decisions.append(decision)

        threads = [
            threading.Thread(target=submit, args=(index,)) for index in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        admitted = [
            decision
            for decision in decisions
            if decision.status is AdmissionStatus.ADMITTED
        ]
        snapshot = controller.snapshot()
        self.assertEqual(10, len(decisions))
        self.assertEqual(4, len(admitted))
        self.assertEqual(4, snapshot.active_sequences)
        self.assertEqual(8, snapshot.reserved_blocks)


if __name__ == "__main__":
    unittest.main()
