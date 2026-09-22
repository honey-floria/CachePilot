import unittest

from cachepilot.executors import (
    LogicalClock,
    SimEventKind,
    SimExecutor,
    SimExecutorConfig,
    SimRequest,
    SimRequestState,
    SimulationLimitError,
    WorkerUnavailableError,
)


class SimExecutorTests(unittest.TestCase):
    def make_executor(self, **overrides):
        values = {
            "tick_ns": 10,
            "block_size": 4,
            "max_batch_size": 2,
            "prefill_tokens_per_tick": 3,
            "decode_tokens_per_tick": 1,
            "output_buffer_tokens": 8,
            "client_drain_tokens_per_tick": 8,
            "seed": 17,
        }
        values.update(overrides)
        return SimExecutor(SimExecutorConfig(**values), LogicalClock())

    def test_logical_clock_drives_prefill_decode_and_kv_growth(self):
        executor = self.make_executor()
        executor.submit(SimRequest("request-1", prompt_tokens=5, output_tokens=3))

        executor.step()
        after_first_tick = executor.request_snapshot("request-1")
        after_first_tick_ns = executor.clock()
        executor.step()
        after_prefill = executor.request_snapshot("request-1")
        final = executor.run_until_idle()

        self.assertEqual(10, after_first_tick_ns)
        self.assertEqual(60, executor.clock())
        self.assertEqual(3, after_first_tick.prompt_tokens_processed)
        self.assertEqual(1, after_first_tick.logical_blocks)
        self.assertEqual(SimRequestState.DECODING, after_prefill.state)
        self.assertEqual(2, after_prefill.logical_blocks)
        self.assertEqual(SimRequestState.FINISHED, final.requests[0].state)
        self.assertEqual(3, final.stats.generated_tokens)
        self.assertEqual(3, final.stats.delivered_tokens)
        self.assertEqual(0, final.stats.current_logical_blocks)
        self.assertEqual(2, final.stats.peak_logical_blocks)

    def test_continuous_batch_refills_slot_while_long_request_decodes(self):
        executor = self.make_executor(
            prefill_tokens_per_tick=4,
            max_batch_size=2,
        )
        executor.submit(SimRequest("long", 1, 4))
        executor.submit(SimRequest("short", 1, 1))
        executor.submit(SimRequest("next", 1, 1))

        executor.step()
        executor.step()
        executor.step()
        snapshot = executor.snapshot()

        self.assertEqual(SimRequestState.DECODING, snapshot.requests[0].state)
        self.assertEqual(SimRequestState.FINISHED, snapshot.requests[1].state)
        self.assertEqual(SimRequestState.DECODING, snapshot.requests[2].state)
        prefill_started = [
            event.request_id
            for event in snapshot.events
            if event.kind is SimEventKind.PREFILL_STARTED
        ]
        self.assertEqual(["long", "short", "next"], prefill_started)

    def test_slow_client_applies_backpressure_then_finishes(self):
        executor = self.make_executor(
            prefill_tokens_per_tick=1,
            decode_tokens_per_tick=2,
            output_buffer_tokens=2,
            client_drain_tokens_per_tick=1,
        )
        executor.submit(SimRequest("slow", 0, 5))

        final = executor.run_until_idle()
        event_kinds = tuple(event.kind for event in final.events)

        self.assertIn(SimEventKind.CLIENT_BACKPRESSURE, event_kinds)
        self.assertEqual(SimRequestState.FINISHED, final.requests[0].state)
        self.assertEqual(5, final.requests[0].delivered_tokens)
        self.assertGreater(final.stats.ticks, 4)

    def test_zero_speed_client_can_be_manually_drained(self):
        executor = self.make_executor(client_drain_tokens_per_tick=0)
        executor.submit(SimRequest("paused-client", 0, 1))

        with self.assertRaises(SimulationLimitError):
            executor.run_until_idle(max_ticks=4)

        self.assertEqual(1, executor.drain_client("paused-client", 1))
        snapshot = executor.snapshot()
        self.assertEqual(SimRequestState.FINISHED, snapshot.requests[0].state)

    def test_cancel_releases_kv_and_stops_future_tokens(self):
        executor = self.make_executor(prefill_tokens_per_tick=8)
        executor.submit(SimRequest("cancelled", 8, 8))
        executor.step()
        before_cancel = executor.request_snapshot("cancelled")

        self.assertEqual(2, before_cancel.logical_blocks)
        self.assertTrue(executor.cancel("cancelled"))
        self.assertFalse(executor.cancel("cancelled"))
        snapshot = executor.snapshot()

        self.assertEqual(SimRequestState.CANCELLED, snapshot.requests[0].state)
        self.assertEqual(0, snapshot.requests[0].logical_blocks)
        self.assertEqual(0, snapshot.stats.generated_tokens)
        self.assertFalse(executor.has_work)

    def test_automatic_cancellation_uses_submission_relative_time(self):
        executor = self.make_executor()
        executor.submit(
            SimRequest("auto-cancel", 20, 2, cancel_after_ns=10)
        )

        executor.step()
        executor.step()

        snapshot = executor.request_snapshot("auto-cancel")
        self.assertEqual(SimRequestState.CANCELLED, snapshot.state)
        self.assertEqual(10, snapshot.terminal_at_ns)
        self.assertEqual(0, snapshot.logical_blocks)

    def test_worker_failure_fails_active_and_queued_requests(self):
        executor = self.make_executor(max_batch_size=1)
        executor.submit(SimRequest("active", 8, 2))
        executor.submit(SimRequest("queued", 8, 2))
        executor.step()

        self.assertTrue(executor.fail_worker("test-failure"))
        snapshot = executor.snapshot()

        self.assertFalse(snapshot.healthy)
        self.assertEqual(2, snapshot.stats.failed_requests)
        self.assertEqual(0, snapshot.stats.current_logical_blocks)
        self.assertEqual((), snapshot.queued_request_ids)
        self.assertEqual((), snapshot.active_request_ids)
        with self.assertRaises(WorkerUnavailableError):
            executor.step()
        with self.assertRaises(WorkerUnavailableError):
            executor.submit(SimRequest("late", 1, 1))

    def test_same_config_trace_and_seed_replay_identically(self):
        def replay():
            executor = self.make_executor(
                decode_tokens_per_tick=2,
                output_buffer_tokens=3,
                client_drain_tokens_per_tick=1,
                seed=99,
            )
            executor.submit(SimRequest("a", 5, 4, seed=7))
            executor.submit(
                SimRequest("b", 2, 6, seed=8, cancel_after_ns=30)
            )
            executor.submit(
                SimRequest(
                    "c",
                    1,
                    2,
                    seed=9,
                    client_drain_tokens_per_tick=0,
                )
            )
            for _ in range(8):
                executor.step()
            executor.drain_client("c", 2)
            return executor.run_until_idle()

        first = replay()
        second = replay()

        self.assertEqual(first, second)
        self.assertEqual(99, first.seed)
        self.assertEqual(1, first.stats.cancelled_requests)
        self.assertEqual(2, first.stats.finished_requests)


if __name__ == "__main__":
    unittest.main()
