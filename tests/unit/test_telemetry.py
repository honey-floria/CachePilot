import unittest

from cachepilot.telemetry import TelemetryCollector


class TelemetryCollectorTests(unittest.TestCase):
    def test_trace_records_admission_stages_durations_and_errors(self):
        now = [0]

        def clock_ns():
            return now[0]

        telemetry = TelemetryCollector(clock_ns=clock_ns)
        telemetry.start_request("request-1", "team-a", "model-a")
        now[0] = 1_000_000
        telemetry.set_prompt_tokens("request-1", 12)
        telemetry.mark_stage("request-1", "tokenized")
        telemetry.mark_stage("request-1", "queued")
        now[0] = 6_000_000
        telemetry.record_admission(
            "request-1",
            "ADMITTED",
            "admitted",
            logical_kv_blocks=3,
        )
        telemetry.mark_stage("request-1", "executing")
        now[0] = 13_000_000
        telemetry.record_token("request-1", 2)
        now[0] = 23_000_000
        telemetry.record_error("request-1", "executor_failed", "executor")
        telemetry.finish("request-1", "FAILED")

        trace = telemetry.snapshot("request-1")
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual("admitted", trace.admission_reason)
        self.assertEqual(3, trace.logical_kv_blocks)
        self.assertEqual(5.0, trace.queue_ms)
        self.assertEqual(7.0, trace.prefill_ms)
        self.assertEqual(10.0, trace.decode_ms)
        self.assertEqual(10.0, trace.tpot_ms)
        self.assertEqual(23.0, trace.total_ms)
        self.assertEqual("executor_failed", trace.error_code)
        self.assertEqual("FAILED", trace.terminal_state)

    def test_prometheus_labels_are_low_cardinality(self):
        telemetry = TelemetryCollector(clock_ns=lambda: 0)
        telemetry.start_request("request-secret", "team-a", "model-a")
        telemetry.mark_stage("request-secret", "queued")
        telemetry.record_admission(
            "request-secret", "ADMITTED", "admitted", logical_kv_blocks=2
        )
        telemetry.mark_stage("request-secret", "executing")
        telemetry.record_token("request-secret", 1)
        telemetry.record_error("request-secret", "executor_failed", "executor")
        telemetry.finish("request-secret", "FINISHED")

        rendered = telemetry.render_prometheus(
            active_sequences=0,
            reserved_kv_blocks=0,
        )
        self.assertIn(
            'cachepilot_requests_total{model="model-a",state="FINISHED",'
            'tenant="team-a"}',
            rendered,
        )
        self.assertNotIn("request_id=", rendered)
        self.assertNotIn("prompt=", rendered)
        self.assertNotIn("prefix_key=", rendered)
        self.assertNotIn("request-secret", rendered)
        self.assertIn(
            'cachepilot_errors_total{code="executor_failed",stage="executor"}',
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
