import asyncio
import unittest
from dataclasses import dataclass

from cachepilot.executors import (
    VllmExecutor,
    VllmExecutorConfig,
    VllmExecutorError,
)
from cachepilot.gateway.contracts import ChatMessage, ValidatedChatRequest


@dataclass
class FakeSamplingParams:
    max_tokens: int
    output_kind: str = "delta"


@dataclass
class FakeCandidate:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None = None


@dataclass
class FakeOutput:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[FakeCandidate, ...]
    finished: bool = False


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        if kwargs["tokenize"]:
            return [10, 11, 12]
        return "<chat>hello</chat>"


class FakeEngine:
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or []
        self.error = error
        self.generate_call = None
        self.aborted = []
        self.release = asyncio.Event()
        self.block_after_first = False

    async def generate(self, **kwargs):
        self.generate_call = kwargs
        if self.error is not None:
            raise self.error
        for index, output in enumerate(self.outputs):
            yield output
            if index == 0 and self.block_after_first:
                await self.release.wait()

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.release.set()

    async def check_health(self):
        return None


def request(request_id="vllm-1", max_tokens=8):
    return ValidatedChatRequest(
        request_id=request_id,
        tenant_id="team-a",
        priority="interactive",
        deadline_ms=1000,
        idempotency_key=None,
        model="model",
        messages=(ChatMessage("user", "hello"),),
        stream=True,
        max_tokens=max_tokens,
    )


def make_executor(engine):
    return VllmExecutor(
        VllmExecutorConfig("model"),
        engine=engine,
        tokenizer=FakeTokenizer(),
        sampling_params_factory=lambda max_tokens: FakeSamplingParams(max_tokens),
    )


class VllmExecutorTests(unittest.TestCase):
    def test_forwards_request_id_prompt_stream_and_usage(self):
        engine = FakeEngine(
            [
                FakeOutput(
                    "vllm-1",
                    (10, 11, 12),
                    (FakeCandidate("hello", (20,)),),
                ),
                FakeOutput(
                    "vllm-1",
                    (10, 11, 12),
                    (FakeCandidate(" world", (21, 22), "stop"),),
                    finished=True,
                ),
            ]
        )
        executor = make_executor(engine)

        generated = asyncio.run(self._collect(executor.generate(request())))

        self.assertEqual([("hello", 1), (" world", 2)], generated)
        self.assertEqual("vllm-1", engine.generate_call["request_id"])
        self.assertEqual("<chat>hello</chat>", engine.generate_call["prompt"])
        self.assertEqual(8, engine.generate_call["sampling_params"].max_tokens)
        self.assertEqual(3, executor.count_prompt_tokens(request()))
        usage = executor.usage("vllm-1")
        self.assertEqual(3, usage.prompt_tokens)
        self.assertEqual(3, usage.completion_tokens)
        self.assertEqual(6, usage.total_tokens)
        self.assertEqual("stop", usage.finish_reason)
        self.assertTrue(usage.finished)
        self.assertTrue(executor.supports_batching)
        self.assertEqual("vllm", executor.batching_owner)
        self.assertFalse(executor.implements_batching)

    def test_cancel_aborts_same_active_request_and_stops_delivery(self):
        engine = FakeEngine(
            [
                FakeOutput(
                    "vllm-cancel",
                    (1, 2),
                    (FakeCandidate("first", (3,)),),
                ),
                FakeOutput(
                    "vllm-cancel",
                    (1, 2),
                    (FakeCandidate("late", (4,)),),
                ),
            ]
        )
        engine.block_after_first = True
        executor = make_executor(engine)

        async def run():
            iterator = executor.generate(request("vllm-cancel"))
            first = await anext(iterator)
            pending = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            await executor.cancel("vllm-cancel", reason="disconnect")
            await executor.cancel("vllm-cancel", reason="explicit")
            with self.assertRaises(StopAsyncIteration):
                await pending
            return first

        first = asyncio.run(run())

        self.assertEqual("first", first.text)
        self.assertEqual(["vllm-cancel"], engine.aborted)
        self.assertEqual("disconnect", executor.cancel_reasons["vllm-cancel"])
        self.assertEqual(1, executor.usage("vllm-cancel").completion_tokens)

    def test_request_id_mismatch_and_engine_errors_are_stable(self):
        mismatch = make_executor(
            FakeEngine(
                [
                    FakeOutput(
                        "wrong-id",
                        (1,),
                        (FakeCandidate("bad", (2,)),),
                    )
                ]
            )
        )
        failed = make_executor(FakeEngine(error=RuntimeError("worker failed")))

        with self.assertRaisesRegex(VllmExecutorError, "request ID mismatch"):
            asyncio.run(self._collect(mismatch.generate(request())))
        with self.assertRaisesRegex(VllmExecutorError, "generation failed"):
            asyncio.run(self._collect(failed.generate(request("vllm-error"))))

    @staticmethod
    async def _collect(iterator):
        return [(item.text, item.token_count) async for item in iterator]


if __name__ == "__main__":
    unittest.main()
