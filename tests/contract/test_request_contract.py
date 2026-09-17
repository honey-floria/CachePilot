import json
import unittest
from pathlib import Path

from cachepilot.cache.prefix_index import PrefixScopeKey, TenantPrefixIndex
from cachepilot.gateway.contracts import (
    ClaimStatus,
    ContractViolation,
    DEFAULT_DEADLINE_MS,
    DEFAULT_MAX_TOKENS,
    IdempotencyGuard,
    validate_chat_completion_request,
)
from cachepilot.gateway.intake import ChatRequestIntake


CONFIGURED_MODEL = "test-model-revision"
UNSUPPORTED_OPENAI_FIELDS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "modalities",
    "audio",
    "n",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "response_format",
    "stream_options",
    "prediction",
    "reasoning_effort",
    "user",
    "service_tier",
    "store",
    "metadata",
)


def valid_body():
    return {
        "model": CONFIGURED_MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }


def valid_headers(**overrides):
    headers = {"X-Tenant-ID": "team-a"}
    headers.update(overrides)
    return headers


class RequestValidationTests(unittest.TestCase):
    def assert_violation(self, expected_code, body=None, headers=None):
        with self.assertRaises(ContractViolation) as raised:
            validate_chat_completion_request(
                valid_body() if body is None else body,
                valid_headers() if headers is None else headers,
                configured_model=CONFIGURED_MODEL,
            )
        self.assertEqual(expected_code, raised.exception.code)
        return raised.exception

    def test_minimal_request_is_normalized(self):
        request = validate_chat_completion_request(
            valid_body(),
            valid_headers(),
            configured_model=CONFIGURED_MODEL,
        )

        self.assertEqual("team-a", request.tenant_id)
        self.assertEqual("interactive", request.priority)
        self.assertEqual(DEFAULT_DEADLINE_MS["interactive"], request.deadline_ms)
        self.assertEqual(DEFAULT_MAX_TOKENS, request.max_tokens)
        self.assertTrue(request.stream)
        self.assertTrue(request.request_id.startswith("req_"))

    def test_explicit_headers_are_preserved(self):
        request = validate_chat_completion_request(
            {**valid_body(), "max_tokens": 64},
            valid_headers(
                **{
                    "X-Request-ID": "client-request:1",
                    "X-Priority": "batch",
                    "X-Deadline-Ms": "120000",
                    "Idempotency-Key": "job-42",
                }
            ),
            configured_model=CONFIGURED_MODEL,
        )

        self.assertEqual("client-request:1", request.request_id)
        self.assertEqual("batch", request.priority)
        self.assertEqual(120000, request.deadline_ms)
        self.assertEqual("job-42", request.idempotency_key)
        self.assertEqual(64, request.max_tokens)

    def test_unknown_top_level_field_is_rejected_even_when_null(self):
        body = valid_body()
        body["tools"] = None
        violation = self.assert_violation("unknown_field", body=body)
        self.assertEqual("tools", violation.param)

    def test_unknown_top_level_field_is_rejected_when_empty_array(self):
        body = valid_body()
        body["tools"] = []
        violation = self.assert_violation("unknown_field", body=body)
        self.assertEqual("tools", violation.param)

    def test_typo_and_multi_result_fields_are_rejected(self):
        for field, value in (("max_token", 32), ("n", 2)):
            body = valid_body()
            body[field] = value
            with self.subTest(field=field):
                violation = self.assert_violation("unknown_field", body=body)
                self.assertEqual(field, violation.param)

    def test_every_explicitly_unsupported_openai_field_is_rejected(self):
        for field in UNSUPPORTED_OPENAI_FIELDS:
            body = valid_body()
            body[field] = None
            with self.subTest(field=field):
                violation = self.assert_violation("unknown_field", body=body)
                self.assertEqual(field, violation.param)

    def test_unknown_message_field_is_rejected(self):
        body = valid_body()
        body["messages"][0]["name"] = "caller"
        self.assert_violation("unknown_field", body=body)

    def test_multimodal_content_is_rejected(self):
        body = valid_body()
        body["messages"][0]["content"] = [
            {"type": "image_url", "image_url": {"url": "https://invalid.test/a.png"}}
        ]
        self.assert_violation("text_content_required", body=body)

    def test_stream_false_is_rejected(self):
        body = valid_body()
        body["stream"] = False
        self.assert_violation("streaming_required", body=body)

    def test_unconfigured_model_is_rejected(self):
        body = valid_body()
        body["model"] = "some-other-model"
        violation = self.assert_violation("model_not_found", body=body)
        self.assertEqual(404, violation.status_code)

    def test_missing_tenant_is_rejected(self):
        self.assert_violation("tenant_required", headers={})

    def test_invalid_priority_and_deadline_are_rejected(self):
        self.assert_violation(
            "invalid_priority",
            headers=valid_headers(**{"X-Priority": "urgent"}),
        )
        self.assert_violation(
            "invalid_deadline",
            headers=valid_headers(**{"X-Deadline-Ms": "1.5"}),
        )

    def test_integer_fields_are_not_loosely_coerced(self):
        for invalid_value in (True, "32", 0, 4097):
            body = valid_body()
            body["max_tokens"] = invalid_value
            expected = (
                "invalid_type"
                if invalid_value is True or isinstance(invalid_value, str)
                else "value_out_of_range"
            )
            with self.subTest(value=invalid_value):
                self.assert_violation(expected, body=body)

    def test_empty_and_wrongly_typed_messages_are_rejected(self):
        for messages in ([], "hello", None, {}):
            body = valid_body()
            body["messages"] = messages
            with self.subTest(messages=messages):
                self.assert_violation("invalid_messages", body=body)

    def test_invalid_message_role_is_rejected(self):
        body = valid_body()
        body["messages"][0]["role"] = "tool"
        self.assert_violation("invalid_role", body=body)

    def test_default_and_explicit_defaults_have_same_fingerprint(self):
        implicit = validate_chat_completion_request(
            valid_body(),
            valid_headers(**{"X-Request-ID": "request-one"}),
            configured_model=CONFIGURED_MODEL,
        )
        explicit = validate_chat_completion_request(
            {**valid_body(), "max_tokens": DEFAULT_MAX_TOKENS},
            valid_headers(
                **{
                    "X-Request-ID": "request-two",
                    "X-Priority": "interactive",
                    "X-Deadline-Ms": str(DEFAULT_DEADLINE_MS["interactive"]),
                }
            ),
            configured_model=CONFIGURED_MODEL,
        )

        self.assertEqual(implicit.fingerprint(), explicit.fingerprint())


class ScopeBoundaryTests(unittest.TestCase):
    class RecordingLifecycleSink:
        def __init__(self):
            self.registry_writes = 0
            self.reservation_writes = 0

        def accept(self, request):
            self.registry_writes += 1
            self.reservation_writes += 1

    def test_invalid_request_never_reaches_lifecycle_state(self):
        sink = self.RecordingLifecycleSink()
        intake = ChatRequestIntake(configured_model=CONFIGURED_MODEL, sink=sink)
        body = valid_body()
        body["tools"] = []

        with self.assertRaises(ContractViolation):
            intake.submit(body, valid_headers())

        self.assertEqual(0, sink.registry_writes)
        self.assertEqual(0, sink.reservation_writes)

    def test_valid_request_is_dispatched_only_after_normalization(self):
        sink = self.RecordingLifecycleSink()
        intake = ChatRequestIntake(configured_model=CONFIGURED_MODEL, sink=sink)

        request = intake.submit(valid_body(), valid_headers())

        self.assertEqual(DEFAULT_MAX_TOKENS, request.max_tokens)
        self.assertEqual(1, sink.registry_writes)
        self.assertEqual(1, sink.reservation_writes)


class TenantPrefixIsolationTests(unittest.TestCase):
    def make_key(self, tenant_id):
        return PrefixScopeKey.create(
            tenant_id=tenant_id,
            model_id=CONFIGURED_MODEL,
            model_revision="model-commit",
            tokenizer_revision="tokenizer-commit",
            quantization_config="none",
            tokenized_prefix=(10, 20, 30),
        )

    def test_identical_prefix_does_not_hit_across_tenants(self):
        index = TenantPrefixIndex()
        team_a_key = self.make_key("team-a")
        team_b_key = self.make_key("team-b")

        index.record(team_a_key)

        self.assertTrue(index.contains(team_a_key))
        self.assertFalse(index.contains(team_b_key))


class DuplicateSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.guard = IdempotencyGuard()

    def make_request(self, request_id, *, tenant="team-a", key="operation-1", content="Hello"):
        body = valid_body()
        body["messages"][0]["content"] = content
        return validate_chat_completion_request(
            body,
            {
                "X-Tenant-ID": tenant,
                "X-Request-ID": request_id,
                "Idempotency-Key": key,
            },
            configured_model=CONFIGURED_MODEL,
        )

    def test_same_key_and_fingerprint_does_not_create_second_request(self):
        original = self.make_request("request-1")
        duplicate = self.make_request("request-2")

        first = self.guard.claim(original)
        second = self.guard.claim(duplicate)

        self.assertEqual(ClaimStatus.ACCEPTED, first.status)
        self.assertEqual(ClaimStatus.IDEMPOTENCY_IN_PROGRESS, second.status)
        self.assertEqual("request-1", second.request_id)

    def test_same_key_with_different_fingerprint_is_a_conflict(self):
        self.guard.claim(self.make_request("request-1"))
        result = self.guard.claim(self.make_request("request-2", content="Different"))

        self.assertEqual(ClaimStatus.IDEMPOTENCY_KEY_CONFLICT, result.status)
        self.assertEqual("request-1", result.request_id)

    def test_terminal_request_is_not_replayed(self):
        original = self.make_request("request-1")
        self.guard.claim(original)
        self.guard.mark_terminal(original.request_id)

        result = self.guard.claim(self.make_request("request-2"))

        self.assertEqual(ClaimStatus.IDEMPOTENCY_REPLAY_UNAVAILABLE, result.status)
        self.assertEqual("request-1", result.request_id)

    def test_idempotency_keys_are_tenant_scoped(self):
        first = self.guard.claim(self.make_request("request-1", tenant="team-a"))
        second = self.guard.claim(self.make_request("request-2", tenant="team-b"))

        self.assertEqual(ClaimStatus.ACCEPTED, first.status)
        self.assertEqual(ClaimStatus.ACCEPTED, second.status)

    def test_request_id_is_unique_without_an_idempotency_key(self):
        first = validate_chat_completion_request(
            valid_body(),
            valid_headers(**{"X-Request-ID": "request-1"}),
            configured_model=CONFIGURED_MODEL,
        )
        duplicate = validate_chat_completion_request(
            valid_body(),
            valid_headers(**{"X-Request-ID": "request-1"}),
            configured_model=CONFIGURED_MODEL,
        )

        self.assertEqual(ClaimStatus.ACCEPTED, self.guard.claim(first).status)
        self.assertEqual(
            ClaimStatus.REQUEST_ID_CONFLICT,
            self.guard.claim(duplicate).status,
        )


class OpenAPIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository_root = Path(__file__).resolve().parents[2]
        with (repository_root / "contracts" / "openapi.json").open(
            encoding="utf-8"
        ) as file:
            cls.openapi = json.load(file)

    def test_openapi_declares_strict_request_objects(self):
        schemas = self.openapi["components"]["schemas"]
        request_schema = schemas["ChatCompletionRequest"]
        message_schema = schemas["ChatMessage"]

        self.assertFalse(request_schema["additionalProperties"])
        self.assertFalse(message_schema["additionalProperties"])
        self.assertEqual(
            {"model", "messages", "stream", "max_tokens"},
            set(request_schema["properties"]),
        )
        self.assertTrue(request_schema["properties"]["stream"]["const"])

    def test_all_internal_openapi_references_resolve(self):
        def walk(value):
            if isinstance(value, dict):
                reference = value.get("$ref")
                if reference is not None:
                    self.assertTrue(reference.startswith("#/"), reference)
                    target = self.openapi
                    for segment in reference[2:].split("/"):
                        target = target[segment.replace("~1", "/").replace("~0", "~")]
                    self.assertIsNotNone(target)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.openapi)

    def test_openapi_declares_control_headers_and_duplicate_response(self):
        operation = self.openapi["paths"]["/v1/chat/completions"]["post"]
        parameter_refs = {parameter["$ref"] for parameter in operation["parameters"]}

        self.assertIn("#/components/parameters/TenantId", parameter_refs)
        self.assertIn("#/components/parameters/RequestId", parameter_refs)
        self.assertIn("#/components/parameters/Priority", parameter_refs)
        self.assertIn("#/components/parameters/DeadlineMs", parameter_refs)
        self.assertIn("#/components/parameters/IdempotencyKey", parameter_refs)
        self.assertIn("409", operation["responses"])

    def test_openapi_fixes_sse_termination(self):
        response = self.openapi["paths"]["/v1/chat/completions"]["post"]["responses"][
            "200"
        ]
        contract = response["x-sse-contract"]

        self.assertEqual("data: [DONE]", contract["normalTermination"][-1])
        self.assertEqual("data: [DONE]", contract["errorTermination"][-1])
        self.assertIn("text/event-stream", response["content"])

    def test_openapi_exposes_query_and_cancel_endpoints(self):
        self.assertIn("/v1/requests/{request_id}", self.openapi["paths"])
        self.assertIn("/v1/requests/{request_id}/cancel", self.openapi["paths"])
        cancel_responses = self.openapi["paths"]["/v1/requests/{request_id}/cancel"][
            "post"
        ]["responses"]
        self.assertEqual({"200", "202", "400", "404", "409"}, set(cancel_responses))


if __name__ == "__main__":
    unittest.main()
