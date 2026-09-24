import unittest

from cachepilot.cache import (
    PrefixIndexError,
    PrefixScopeKey,
    TenantPrefixIndex,
)


class TenantPrefixIndexTests(unittest.TestCase):
    def make_key(self, tokens, **overrides):
        values = {
            "tenant_id": "tenant-a",
            "model_id": "model-a",
            "model_revision": "model-rev-1",
            "tokenizer_revision": "tokenizer-rev-1",
            "quantization_config": "none",
            "tokenized_prefix": tokens,
        }
        values.update(overrides)
        return PrefixScopeKey.create(**values)

    def test_lookup_returns_longest_prefix_and_not_physical_hit(self):
        index = TenantPrefixIndex()
        short = self.make_key((10, 20))
        long = self.make_key((10, 20, 30))
        index.record(short)
        index.record(long)

        result = index.lookup(self.make_key((10, 20, 30, 40)))

        self.assertTrue(result.logical_hit)
        self.assertEqual(3, result.matched_tokens)
        self.assertEqual(long, result.matched_key)
        self.assertIsNone(result.physical_hit)

    def test_tenant_model_tokenizer_quantization_and_tokens_are_isolated(self):
        index = TenantPrefixIndex()
        index.record(self.make_key((1, 2, 3)))
        variants = (
            self.make_key((1, 2, 3, 4), tenant_id="tenant-b"),
            self.make_key((1, 2, 3, 4), model_id="model-b"),
            self.make_key((1, 2, 3, 4), model_revision="model-rev-2"),
            self.make_key(
                (1, 2, 3, 4), tokenizer_revision="tokenizer-rev-2"
            ),
            self.make_key((1, 2, 3, 4), quantization_config="int8"),
            self.make_key((1, 9, 3, 4)),
        )

        for key in variants:
            with self.subTest(key=key):
                result = index.lookup(key)
                self.assertFalse(result.logical_hit)
                self.assertEqual(0, result.matched_tokens)
                self.assertIsNone(result.physical_hit)

    def test_record_discard_invalidate_and_snapshot_are_deterministic(self):
        index = TenantPrefixIndex()
        first = self.make_key((1,))
        second = self.make_key((1, 2))

        self.assertTrue(index.record(first))
        self.assertFalse(index.record(first))
        self.assertTrue(index.record(second))
        index.lookup(self.make_key((1, 2, 3)))
        index.lookup(self.make_key((9,)))
        snapshot = index.snapshot()

        self.assertEqual(1, snapshot.scopes)
        self.assertEqual(2, snapshot.entries)
        self.assertEqual(1, snapshot.logical_hits)
        self.assertEqual(1, snapshot.logical_misses)
        self.assertTrue(index.discard(first))
        self.assertFalse(index.discard(first))
        self.assertEqual(1, index.invalidate_scope(second.scope))
        self.assertEqual(0, index.snapshot().entries)

    def test_invalid_scope_and_token_ids_are_rejected(self):
        with self.assertRaises(PrefixIndexError):
            self.make_key(())
        with self.assertRaises(PrefixIndexError):
            self.make_key((1, -1))
        with self.assertRaises(PrefixIndexError):
            self.make_key((1,), model_revision="")


if __name__ == "__main__":
    unittest.main()
