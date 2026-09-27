import unittest

from cachepilot.telemetry import (
    LedgerError,
    RequestLedger,
    TelemetryCollector,
    validate_recalculated_cost,
)


class RequestLedgerTests(unittest.TestCase):
    def _trace(self):
        now = [0]

        def clock_ns():
            return now[0]

        telemetry = TelemetryCollector(clock_ns=clock_ns)
        telemetry.start_request("request-1", "team-a", "model-a")
        telemetry.set_prompt_tokens("request-1", 12)
        telemetry.mark_stage("request-1", "queued")
        now[0] = 2_000_000
        telemetry.record_admission(
            "request-1", "ADMITTED", "admitted", logical_kv_blocks=3
        )
        telemetry.mark_stage("request-1", "executing")
        now[0] = 12_000_000
        telemetry.record_token("request-1", 2)
        now[0] = 22_000_000
        telemetry.finish("request-1", "FINISHED")
        return telemetry.snapshot("request-1")

    def test_record_contains_strategy_hits_and_reproducible_estimated_cost(self):
        trace = self._trace()
        assert trace is not None
        ledger = RequestLedger(
            strategy_version="strict-fcfs-v2",
            gpu_hour_price=3.60,
        )

        record = ledger.record_trace(trace)
        payload = record.as_dict()

        self.assertEqual("strict-fcfs-v2", payload["strategy_version"])
        self.assertEqual(3, payload["reservation_peak_blocks"])
        self.assertFalse(payload["logical_hit"])
        self.assertEqual("not_configured", payload["logical_hit_source"])
        self.assertIsNone(payload["physical_hit"])
        self.assertEqual("unobservable", payload["physical_hit_source"])
        self.assertTrue(payload["cost_is_estimate"])
        self.assertEqual("prefill_plus_decode_wall_time", payload["cost_basis"])
        self.assertEqual(3.60 * 0.020 / 3600.0, payload["estimated_cost"])
        validate_recalculated_cost(payload)

    def test_recording_terminal_trace_is_idempotent_and_exportable(self):
        trace = self._trace()
        assert trace is not None
        ledger = RequestLedger(strategy_version="gateway-v1")
        first = ledger.record_trace(trace)
        second = ledger.record_trace(trace)

        self.assertIs(first, second)
        self.assertEqual(1, len(ledger.records()))
        exported = ledger.export_jsonl()
        self.assertEqual(1, len(exported.splitlines()))
        self.assertIn('"cost_is_estimate": true', exported)

    def test_observed_hits_require_explicit_sources(self):
        trace = self._trace()
        assert trace is not None
        ledger = RequestLedger(strategy_version="gateway-v1")

        with self.assertRaises(LedgerError):
            ledger.record_trace(trace, logical_hit=True)
        with self.assertRaises(LedgerError):
            ledger.record_trace(trace, physical_hit=True)

    def test_prometheus_cost_totals_are_low_cardinality(self):
        trace = self._trace()
        assert trace is not None
        ledger = RequestLedger(
            strategy_version="gateway-v1",
            gpu_hour_price=3.60,
        )
        ledger.record_trace(trace)

        rendered = ledger.render_prometheus()

        self.assertIn(
            'cachepilot_estimated_gpu_seconds_total{model="model-a",'
            'tenant="team-a"} 0.02',
            rendered,
        )
        self.assertIn(
            'cachepilot_estimated_cost_total{currency="USD",model="model-a",'
            'tenant="team-a"}',
            rendered,
        )
        self.assertNotIn("request_id=", rendered)


if __name__ == "__main__":
    unittest.main()
