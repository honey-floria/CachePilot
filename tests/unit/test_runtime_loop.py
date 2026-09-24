import unittest
from fractions import Fraction

from cachepilot.executors import LogicalClock, SimExecutor, SimExecutorConfig
from cachepilot.runtime.loop import (
    RequestExceedsLoopCapacityError,
    RuntimeLoop,
    RuntimeLoopConfig,
    RuntimeRequest,
)
from cachepilot.runtime.scheduler import (
    CacheBoostConfig,
    FCFSScheduler,
    PrefixAwareWFQScheduler,
)


class RuntimeLoopTests(unittest.TestCase):
    def make_loop(
        self,
        *,
        max_active_sequences=2,
        max_batched_tokens=3,
        max_kv_blocks=5,
        prefill_tokens_per_tick=4,
        decode_tokens_per_tick=2,
    ):
        clock = LogicalClock()
        executor = SimExecutor(
            SimExecutorConfig(
                tick_ns=10,
                block_size=4,
                max_batch_size=max_active_sequences,
                prefill_tokens_per_tick=prefill_tokens_per_tick,
                decode_tokens_per_tick=decode_tokens_per_tick,
                output_buffer_tokens=8,
                client_drain_tokens_per_tick=8,
                seed=23,
            ),
            clock,
        )
        scheduler = FCFSScheduler(clock)
        loop = RuntimeLoop(
            RuntimeLoopConfig(
                max_active_sequences=max_active_sequences,
                max_batched_tokens=max_batched_tokens,
                max_kv_blocks=max_kv_blocks,
            ),
            scheduler,
            executor,
        )
        return loop

    @staticmethod
    def request(request_id, prompt_tokens, output_tokens, tenant_id="tenant-a"):
        return RuntimeRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            priority="interactive",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    def test_mixed_lengths_never_exceed_three_hard_limits(self):
        loop = self.make_loop()
        loop.submit(self.request("long", 8, 8))
        loop.submit(self.request("short-1", 1, 1, "tenant-b"))
        loop.submit(self.request("short-2", 1, 1, "tenant-c"))

        results = []
        while loop.has_work:
            results.append(loop.step())

        final = loop.snapshot()
        self.assertTrue(results)
        for result in results:
            self.assertLessEqual(result.after.active_sequences, 2)
            self.assertLessEqual(result.after.batched_tokens, 3)
            self.assertLessEqual(result.after.kv_blocks, 5)
            self.assertLessEqual(result.after.reserved_kv_blocks, 5)
        self.assertEqual(3, final.peak_batched_tokens)
        self.assertLessEqual(final.peak_kv_blocks, 5)
        self.assertEqual(2, final.peak_active_sequences)
        self.assertEqual(
            ("short-1", "short-2", "long"),
            final.completed_request_ids,
        )
        self.assertEqual((), final.pending_request_ids)
        self.assertEqual((), final.active_request_ids)

    def test_completion_is_reclaimed_before_same_tick_admission(self):
        loop = self.make_loop()
        loop.submit(self.request("long", 8, 8))
        loop.submit(self.request("short-1", 1, 1))
        loop.submit(self.request("short-2", 1, 1))

        matching_result = None
        for _ in range(20):
            result = loop.step()
            if result.reclaimed_request_ids:
                matching_result = result
                break

        self.assertIsNotNone(matching_result)
        self.assertEqual(("short-1",), matching_result.reclaimed_request_ids)
        self.assertEqual(("short-2",), matching_result.admitted_request_ids)
        self.assertEqual(2, matching_result.before.active_sequences)
        self.assertEqual(5, matching_result.before.reserved_kv_blocks)

    def test_batch_token_budget_partially_advances_prefill(self):
        loop = self.make_loop(
            max_active_sequences=1,
            max_batched_tokens=2,
            max_kv_blocks=4,
            prefill_tokens_per_tick=8,
        )
        loop.submit(self.request("request-1", 8, 1))

        result = loop.step()
        request = result.executor.requests[0]

        self.assertEqual((("request-1", 2),), result.work_allocations)
        self.assertEqual(2, result.after.batched_tokens)
        self.assertEqual(2, request.prompt_tokens_processed)

    def test_rotation_prevents_one_request_from_owning_every_tick(self):
        loop = self.make_loop(
            max_batched_tokens=1,
            max_kv_blocks=4,
            prefill_tokens_per_tick=4,
        )
        loop.submit(self.request("request-a", 4, 1, "tenant-a"))
        loop.submit(self.request("request-b", 4, 1, "tenant-b"))

        first = loop.step()
        second = loop.step()

        self.assertEqual((("request-a", 1),), first.work_allocations)
        self.assertEqual((("request-b", 1),), second.work_allocations)

    def test_request_larger_than_kv_hard_limit_is_rejected(self):
        loop = self.make_loop(max_kv_blocks=2)

        with self.assertRaises(RequestExceedsLoopCapacityError):
            loop.submit(self.request("too-large", 8, 1))

        self.assertFalse(loop.has_work)

    def test_same_trace_replays_with_identical_ticks_and_statistics(self):
        def replay():
            loop = self.make_loop()
            loop.submit(self.request("long", 8, 8, "tenant-a"))
            loop.submit(self.request("short", 1, 2, "tenant-b"))
            loop.submit(self.request("tail", 2, 1, "tenant-c"))
            ticks = []
            while loop.has_work:
                ticks.append(loop.step())
            return tuple(ticks), loop.snapshot()

        first_ticks, first_snapshot = replay()
        second_ticks, second_snapshot = replay()

        self.assertEqual(first_ticks, second_ticks)
        self.assertEqual(first_snapshot, second_snapshot)

    def test_runtime_passes_logical_cache_hit_to_prefix_aware_scheduler(self):
        clock = LogicalClock()
        executor = SimExecutor(
            SimExecutorConfig(10, 4, 1, 4, 1, 8, 8),
            clock,
        )
        scheduler = PrefixAwareWFQScheduler(
            {},
            max_starvation_ns=100,
            cache_boost=CacheBoostConfig(1, 50, Fraction(0)),
            monotonic_ns=clock,
        )
        loop = RuntimeLoop(RuntimeLoopConfig(1, 4, 8), scheduler, executor)
        loop.submit(self.request("baseline", 2, 0, "tenant-a"))
        loop.submit(
            RuntimeRequest(
                "cached",
                "tenant-b",
                "interactive",
                prompt_tokens=4,
                output_tokens=0,
                cache_hit_tokens=4,
            )
        )

        result = loop.step()

        self.assertEqual(("cached",), result.admitted_request_ids)


if __name__ == "__main__":
    unittest.main()
