import unittest
from fractions import Fraction

from cachepilot.runtime.scheduler import (
    CacheBoostConfig,
    FCFSScheduler,
    PrefixAwareWFQScheduler,
    SchedulerError,
    SchedulingPriority,
    SchedulingRequest,
    WFQScheduler,
)


class LogicalClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def advance(self, nanoseconds):
        self.now_ns += nanoseconds


def request(request_id, tenant_id, priority="interactive", service_cost=1):
    return SchedulingRequest(
        request_id=request_id,
        tenant_id=tenant_id,
        priority=priority,
        service_cost=service_cost,
    )


class FCFSSchedulerTests(unittest.TestCase):
    def test_interactive_precedes_batch_and_each_class_is_fcfs(self):
        clock = LogicalClock()
        scheduler = FCFSScheduler(clock)
        scheduler.enqueue(request("batch-a", "tenant-a", "batch"))
        clock.advance(1)
        scheduler.enqueue(request("interactive-b", "tenant-b"))
        clock.advance(1)
        scheduler.enqueue(request("interactive-a", "tenant-a"))

        selected = [scheduler.select().request.request_id for _ in range(3)]

        self.assertEqual(
            ["interactive-b", "interactive-a", "batch-a"], selected
        )

    def test_tenant_subqueues_preserve_fifo_order(self):
        clock = LogicalClock()
        scheduler = FCFSScheduler(clock)
        for queued_request in (
            request("a-1", "tenant-a"),
            request("b-1", "tenant-b"),
            request("a-2", "tenant-a"),
        ):
            scheduler.enqueue(queued_request)

        snapshot = scheduler.snapshot()
        selected = [scheduler.select().request.request_id for _ in range(3)]

        self.assertEqual(3, snapshot.queued_requests)
        self.assertEqual(
            (
                ("interactive", "tenant-a", 2),
                ("interactive", "tenant-b", 1),
            ),
            snapshot.tenant_counts,
        )
        self.assertEqual(["a-1", "b-1", "a-2"], selected)
        self.assertIsNone(scheduler.select())


class WFQSchedulerTests(unittest.TestCase):
    def test_weighted_virtual_finish_tags_determine_order(self):
        clock = LogicalClock()
        scheduler = WFQScheduler(
            {"tenant-heavy": 4, "tenant-light": 1},
            max_starvation_ns=100,
            monotonic_ns=clock,
        )
        scheduler.enqueue(request("light-1", "tenant-light", service_cost=4))
        scheduler.enqueue(request("heavy-1", "tenant-heavy", service_cost=4))
        scheduler.enqueue(request("heavy-2", "tenant-heavy", service_cost=4))

        decisions = [scheduler.select() for _ in range(3)]

        self.assertEqual(
            ["heavy-1", "heavy-2", "light-1"],
            [decision.request.request_id for decision in decisions],
        )
        self.assertEqual(Fraction(1), decisions[0].virtual_finish)
        self.assertEqual(Fraction(2), decisions[1].virtual_finish)
        self.assertEqual(Fraction(4), decisions[2].virtual_finish)

    def test_low_weight_tenant_is_forced_after_max_starvation(self):
        clock = LogicalClock()
        scheduler = WFQScheduler(
            {"tenant-heavy": 100, "tenant-light": 1},
            max_starvation_ns=10,
            monotonic_ns=clock,
        )
        scheduler.enqueue(request("light", "tenant-light", service_cost=1000))
        clock.advance(1)
        scheduler.enqueue(request("heavy-1", "tenant-heavy"))
        self.assertEqual("heavy-1", scheduler.select().request.request_id)

        clock.advance(9)
        scheduler.enqueue(request("heavy-2", "tenant-heavy"))
        decision = scheduler.select()

        self.assertEqual("light", decision.request.request_id)
        self.assertTrue(decision.starvation_promoted)
        self.assertEqual(10, decision.queue_wait_ns)

    def test_starvation_promotion_crosses_priority_classes(self):
        clock = LogicalClock()
        scheduler = WFQScheduler(
            {}, max_starvation_ns=5, monotonic_ns=clock
        )
        scheduler.enqueue(request("batch", "tenant-b", "batch"))
        clock.advance(4)
        scheduler.enqueue(request("interactive", "tenant-i"))
        self.assertEqual("interactive", scheduler.select().request.request_id)

        clock.advance(1)
        scheduler.enqueue(request("interactive-2", "tenant-i"))
        decision = scheduler.select()

        self.assertEqual("batch", decision.request.request_id)
        self.assertTrue(decision.starvation_promoted)

    def test_fixed_trace_replays_identically(self):
        def replay():
            clock = LogicalClock()
            scheduler = WFQScheduler(
                {"a": 1, "b": 3},
                max_starvation_ns=20,
                monotonic_ns=clock,
            )
            trace = (
                (0, request("a-1", "a", service_cost=6)),
                (0, request("b-1", "b", service_cost=6)),
                (2, request("batch-a", "a", "batch", 1)),
                (4, request("b-2", "b", service_cost=3)),
            )
            for timestamp, queued_request in trace:
                clock.now_ns = timestamp
                scheduler.enqueue(queued_request)
            clock.now_ns = 10
            return tuple(
                scheduler.select().request.request_id for _ in range(len(trace))
            )

        self.assertEqual(replay(), replay())
        self.assertEqual(("b-1", "b-2", "a-1", "batch-a"), replay())

    def test_duplicate_ids_invalid_config_and_backwards_clock_are_rejected(self):
        clock = LogicalClock()
        scheduler = WFQScheduler({}, 10, clock)
        scheduler.enqueue(request("request-1", "tenant-a"))

        with self.assertRaises(SchedulerError):
            scheduler.enqueue(request("request-1", "tenant-b"))
        with self.assertRaises(SchedulerError):
            WFQScheduler({"tenant-a": 0}, 10, clock)

        clock.now_ns = -1
        with self.assertRaises(SchedulerError):
            scheduler.select()

    def test_string_priority_is_normalized(self):
        queued_request = request("request-1", "tenant-a", "batch")

        self.assertIs(SchedulingPriority.BATCH, queued_request.priority)


class PrefixAwareWFQSchedulerTests(unittest.TestCase):
    def make_scheduler(self, clock, **overrides):
        values = {
            "max_consecutive_boosts": 1,
            "max_bypass_wait_ns": 10,
            "min_tenant_share": Fraction(0),
        }
        values.update(overrides)
        return PrefixAwareWFQScheduler(
            {},
            max_starvation_ns=100,
            cache_boost=CacheBoostConfig(**values),
            monotonic_ns=clock,
        )

    def test_logical_hit_can_boundedly_bypass_baseline_wfq(self):
        clock = LogicalClock()
        scheduler = self.make_scheduler(clock)
        scheduler.enqueue(request("baseline", "tenant-a", service_cost=2))
        scheduler.enqueue(
            SchedulingRequest(
                "cached",
                "tenant-b",
                "interactive",
                service_cost=4,
                cache_hit_tokens=4,
            )
        )

        decision = scheduler.select()

        self.assertEqual("cached", decision.request.request_id)
        self.assertTrue(decision.cache_boosted)

    def test_max_consecutive_boosts_forces_baseline_selection(self):
        clock = LogicalClock()
        scheduler = self.make_scheduler(clock)
        scheduler.enqueue(request("baseline", "tenant-a", service_cost=2))
        for tenant_id in ("tenant-b", "tenant-c"):
            scheduler.enqueue(
                SchedulingRequest(
                    "cached-" + tenant_id,
                    tenant_id,
                    "interactive",
                    service_cost=4,
                    cache_hit_tokens=4,
                )
            )

        first = scheduler.select()
        second = scheduler.select()

        self.assertTrue(first.cache_boosted)
        self.assertEqual("baseline", second.request.request_id)
        self.assertFalse(second.cache_boosted)

    def test_wait_boundary_prevents_cache_bypass(self):
        clock = LogicalClock()
        scheduler = self.make_scheduler(clock)
        scheduler.enqueue(request("baseline", "tenant-a", service_cost=2))
        clock.advance(10)
        scheduler.enqueue(
            SchedulingRequest(
                "cached",
                "tenant-b",
                "interactive",
                service_cost=4,
                cache_hit_tokens=4,
            )
        )

        decision = scheduler.select()

        self.assertEqual("baseline", decision.request.request_id)
        self.assertFalse(decision.cache_boosted)

    def test_minimum_tenant_share_prevents_repeated_bypass(self):
        clock = LogicalClock()
        scheduler = self.make_scheduler(
            clock,
            max_consecutive_boosts=10,
            min_tenant_share=Fraction(1, 4),
        )
        scheduler.enqueue(request("baseline", "tenant-a", service_cost=2))
        for tenant_id in ("tenant-b", "tenant-c"):
            scheduler.enqueue(
                SchedulingRequest(
                    "cached-" + tenant_id,
                    tenant_id,
                    "interactive",
                    service_cost=4,
                    cache_hit_tokens=4,
                )
            )

        first = scheduler.select()
        second = scheduler.select()

        self.assertTrue(first.cache_boosted)
        self.assertEqual("baseline", second.request.request_id)
        self.assertFalse(second.cache_boosted)


if __name__ == "__main__":
    unittest.main()
