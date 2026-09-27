import json
import unittest
from dataclasses import dataclass

from fastapi.testclient import TestClient

from cachepilot.executors import VllmExecutor, VllmExecutorConfig
from cachepilot.gateway.api import GatewaySettings, create_app
from cachepilot.runtime.admission import TenantAdmissionLimits


@dataclass
class Candidate:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None = None


@dataclass
class Output:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[Candidate, ...]
    finished: bool = False


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del messages
        if kwargs["tokenize"]:
            return [1, 2, 3]
        return "prompt"


class Engine:
    def __init__(self):
        self.request_id = None

    async def generate(self, **kwargs):
        self.request_id = kwargs["request_id"]
        yield Output(
            self.request_id,
            (1, 2, 3),
            (Candidate("vLLM", (4,)),),
        )
        yield Output(
            self.request_id,
            (1, 2, 3),
            (Candidate(" works", (5,), "stop"),),
            finished=True,
        )

    async def abort(self, request_id):
        del request_id


class VllmGatewayTests(unittest.TestCase):
    def test_sse_request_id_and_usage_match_vllm_adapter(self):
        model = "Qwen/Qwen2.5-0.5B-Instruct"
        engine = Engine()
        executor = VllmExecutor(
            VllmExecutorConfig(model),
            engine=engine,
            tokenizer=Tokenizer(),
            sampling_params_factory=lambda max_tokens: {"max_tokens": max_tokens},
        )
        app = create_app(
            GatewaySettings(
                model_id=model,
                context_limit=128,
                total_kv_blocks=32,
                safety_kv_blocks=2,
                max_active_sequences=2,
                max_queued_requests=2,
                tenant_limits={
                    "team-a": TenantAdmissionLimits(2, 128, 2),
                },
            ),
            backend=executor,
            token_counter=executor,
        )

        with TestClient(app).stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "X-Tenant-ID": "team-a",
                "X-Request-ID": "gateway-vllm-1",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 4,
            },
        ) as response:
            data = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        chunks = [json.loads(line) for line in data[:-1]]
        text = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
        )
        self.assertEqual("vLLM works", text)
        self.assertEqual("gateway-vllm-1", engine.request_id)
        self.assertEqual("gateway-vllm-1", chunks[0]["id"])
        self.assertEqual("data: [DONE]", "data: " + data[-1])
        self.assertEqual(3, chunks[-1]["usage"]["prompt_tokens"])
        self.assertEqual(2, chunks[-1]["usage"]["completion_tokens"])
        self.assertEqual(2, executor.usage("gateway-vllm-1").completion_tokens)


if __name__ == "__main__":
    unittest.main()
