import asyncio
import unittest

from cachepilot.executors import TorchExecutor, TorchExecutorConfig
from cachepilot.gateway.contracts import ChatMessage, ValidatedChatRequest


class Matrix:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))

    def long(self):
        return self

    def cumsum(self, dim):
        del dim
        return Matrix(
            [
                [sum(row[:index + 1]) for index in range(len(row))]
                for row in self.values
            ]
        )

    def __sub__(self, value):
        return Matrix([[item - value for item in row] for row in self.values])

    def __eq__(self, value):
        return Matrix([[item == value for item in row] for row in self.values])

    def masked_fill(self, mask, value):
        return Matrix(
            [
                [
                    value if mask.values[row][column] else self.values[row][column]
                    for column in range(len(self.values[row]))
                ]
                for row in range(len(self.values))
            ]
        )

    def to(self, device):
        del device
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Matrix(self.values[index])
        row = self.values[index]
        return Matrix(row) if row and isinstance(row[0], list) else row


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "formatted prompt"

    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return {
            "input_ids": Matrix([[0, 1, 2]]),
            "attention_mask": Matrix([[0, 1, 1]]),
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return {3: "A", 4: "B"}.get(token_ids[0], "")


class FakeModel:
    generation_config = None

    def __init__(self):
        self.generate_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return Matrix([[0, 1, 2, 3, 4, 9]])


def request(max_tokens=4):
    return ValidatedChatRequest(
        request_id="torch-1",
        tenant_id="team-a",
        priority="interactive",
        deadline_ms=1000,
        idempotency_key=None,
        model="model",
        messages=(ChatMessage("user", "hello"),),
        stream=False,
        max_tokens=max_tokens,
    )


class TorchExecutorTests(unittest.TestCase):
    def test_single_request_passes_mask_positions_and_stops_at_eos(self):
        model = FakeModel()
        executor = TorchExecutor(
            TorchExecutorConfig("model", device="cpu"),
            model=model,
            tokenizer=FakeTokenizer(),
        )

        generated = asyncio.run(self._collect(executor.generate(request())))

        self.assertEqual(["A", "B"], generated)
        self.assertEqual(
            [[0, 0, 1]],
            model.generate_kwargs["position_ids"].values,
        )
        self.assertEqual([[0, 1, 1]], model.generate_kwargs["attention_mask"].values)
        self.assertEqual(4, model.generate_kwargs["max_new_tokens"])
        self.assertFalse(executor.supports_batching)

    def test_cancel_prevents_delivery_of_generated_tokens(self):
        model = FakeModel()
        executor = TorchExecutor(
            TorchExecutorConfig("model", device="cpu"),
            model=model,
            tokenizer=FakeTokenizer(),
        )

        async def run():
            await executor.cancel("torch-1", reason="disconnect")
            return [item async for item in executor.generate(request())]

        self.assertEqual([], asyncio.run(run()))

    @staticmethod
    async def _collect(iterator):
        return [item.text async for item in iterator]


if __name__ == "__main__":
    unittest.main()
