import unittest

from cachepilot.runtime.adaptive_admission import (
    AdaptiveAdmissionConfig,
    AdaptiveAdmissionController,
    GrowthReason,
    GrowthStatus,
)
from cachepilot.runtime.admission import (
    AdmissionReason,
    AdmissionStatus,
    StrictAdmissionConfig,
    TenantAdmissionLimits,
)
from cachepilot.runtime.kv_planner import KVModelSpec, KVPlanner


class AdaptiveAdmissionControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        total_blocks=20,
        safety_blocks=2,
        min_samples=5,
        safety_margin_tokens=5,
        tenant_active_tokens=512,
    ):
        planner = KVPlanner(
            KVModelSpec(2, 2, 8, "float16", 16, 256)
        )
        strict = StrictAdmissionConfig(
            total_blocks=total_blocks,
            safety_blocks=safety_blocks,
            max_active_sequences=8,
            max_queued_requests=8,
            tenant_limits={
                "team-a": TenantAdmissionLimits(8, tenant_active_tokens, 8),
                "team-b": TenantAdmissionLimits(8, 512, 8),
            },
        )
        return AdaptiveAdmissionController(
            planner,
            AdaptiveAdmissionConfig(
                strict=strict,
                prompt_bucket_boundaries=(32, 128),
                min_samples_per_bucket=min_samples,
                safety_margin_tokens=safety_margin_tokens,
                max_samples_per_bucket=20,
            ),
        )

    def seed(self, controller, tenant_id="team-a", prompt_tokens=32):
        for output_tokens in (5, 10, 15, 20, 25):
            controller.observe_output(
                tenant_id,
                prompt_tokens,
                output_tokens,
            )

    def test_insufficient_samples_fall_back_to_strict(self):
        controller = self.make_controller()

        decision = controller.submit("request-1", "team-a", 32, 100)

        self.assertEqual(AdmissionStatus.ADMITTED, decision.status)
        self.assertTrue(decision.fallback_to_strict)
        self.assertEqual(100, decision.estimated_output_tokens)
        self.assertEqual(9, decision.plan.logical_blocks)
        self.assertEqual(1, controller.adaptive_snapshot().fallback_count)

    def test_bucket_p95_plus_margin_reduces_reservation(self):
        controller = self.make_controller()
        self.seed(controller)

        decision = controller.submit("request-1", "team-a", 32, 100)

        self.assertEqual(AdmissionStatus.ADMITTED, decision.status)
        self.assertFalse(decision.fallback_to_strict)
        self.assertEqual(30, decision.estimated_output_tokens)
        self.assertEqual(4, decision.plan.logical_blocks)

    def test_history_is_isolated_by_tenant_and_prompt_bucket(self):
        controller = self.make_controller()
        self.seed(controller, "team-a", 32)

        other_tenant = controller.submit("request-b", "team-b", 32, 100)
        other_bucket = controller.submit("request-a", "team-a", 33, 100)

        self.assertTrue(other_tenant.fallback_to_strict)
        self.assertTrue(other_bucket.fallback_to_strict)

    def test_generation_growth_expands_reservation_within_hard_limits(self):
        controller = self.make_controller(min_samples=3, safety_margin_tokens=0)
        for _ in range(3):
            controller.observe_output("team-a", 16, 8)
        decision = controller.submit("request-1", "team-a", 16, 64)
        self.assertEqual(2, decision.plan.logical_blocks)

        growth = controller.reserve_generated_tokens("request-1", 17)

        self.assertEqual(GrowthStatus.CONTINUE, growth.status)
        self.assertEqual(GrowthReason.RESERVATION_GROWN, growth.reason)
        self.assertEqual(3, growth.reserved_blocks)
        self.assertEqual(3, controller.snapshot().reserved_blocks)

    def test_long_tail_must_stop_before_crossing_kv_capacity(self):
        controller = self.make_controller(
            total_blocks=5,
            safety_blocks=1,
            min_samples=3,
            safety_margin_tokens=0,
        )
        for _ in range(3):
            controller.observe_output("team-a", 16, 8)
        controller.submit("adaptive", "team-a", 16, 64)
        controller.submit("occupier", "team-b", 16, 8)

        growth = controller.reserve_generated_tokens("adaptive", 17)

        self.assertEqual(GrowthStatus.STOP_REQUIRED, growth.status)
        self.assertEqual(GrowthReason.KV_CAPACITY, growth.reason)
        self.assertEqual(4, controller.snapshot().reserved_blocks)

    def test_long_tail_must_stop_before_crossing_tenant_quota(self):
        controller = self.make_controller(
            min_samples=3,
            safety_margin_tokens=0,
            tenant_active_tokens=30,
        )
        for _ in range(3):
            controller.observe_output("team-a", 16, 8)
        controller.submit("request-1", "team-a", 16, 64)

        growth = controller.reserve_generated_tokens("request-1", 15)

        self.assertEqual(GrowthStatus.STOP_REQUIRED, growth.status)
        self.assertEqual(GrowthReason.TENANT_ACTIVE_TOKENS, growth.reason)

    def test_completion_records_original_estimation_error_and_sample(self):
        controller = self.make_controller(min_samples=3, safety_margin_tokens=0)
        for _ in range(3):
            controller.observe_output("team-a", 16, 8)
        controller.submit("request-1", "team-a", 16, 64)
        controller.reserve_generated_tokens("request-1", 12)

        self.assertTrue(controller.complete("request-1", 12))

        metrics = controller.adaptive_snapshot()
        self.assertEqual(1, metrics.estimation_count)
        self.assertEqual(1, metrics.underestimation_count)
        self.assertEqual(4, metrics.signed_error_tokens)
        self.assertEqual(4, metrics.absolute_error_tokens)
        self.assertEqual(0, controller.snapshot().reserved_blocks)

    def test_completion_cannot_bypass_growth_reservation(self):
        controller = self.make_controller(min_samples=3, safety_margin_tokens=0)
        for _ in range(3):
            controller.observe_output("team-a", 16, 8)
        controller.submit("request-1", "team-a", 16, 64)

        with self.assertRaisesRegex(ValueError, "growth must be reserved"):
            controller.complete("request-1", 9)

        self.assertEqual(2, controller.snapshot().reserved_blocks)

    def test_hard_context_limit_is_not_relaxed_by_adaptive_estimate(self):
        controller = self.make_controller()
        self.seed(controller)

        decision = controller.submit("request-1", "team-a", 200, 64)

        self.assertEqual(AdmissionStatus.REJECTED, decision.status)
        self.assertEqual(AdmissionReason.CONTEXT_LIMIT_EXCEEDED, decision.reason)


if __name__ == "__main__":
    unittest.main()
