import unittest

from cachepilot.utils import CommonUtils


class CustomValidationError(ValueError):
    pass


class CommonUtilsTests(unittest.TestCase):
    def test_validation_returns_valid_values(self):
        self.assertEqual(
            "request-1",
            CommonUtils.require_identifier("request-1", "request_id"),
        )
        self.assertEqual(2, CommonUtils.require_positive_int(2, "count"))
        self.assertEqual(
            0,
            CommonUtils.require_non_negative_int(0, "count"),
        )

    def test_validation_uses_caller_exception_type_and_rejects_bool(self):
        with self.assertRaisesRegex(CustomValidationError, "request_id"):
            CommonUtils.require_identifier("", "request_id", CustomValidationError)
        with self.assertRaises(CustomValidationError):
            CommonUtils.require_positive_int(True, "count", CustomValidationError)
        with self.assertRaises(CustomValidationError):
            CommonUtils.require_non_negative_int(
                -1,
                "count",
                CustomValidationError,
            )

    def test_integer_math_helpers_are_exact(self):
        self.assertEqual(0, CommonUtils.ceil_div(0, 4))
        self.assertEqual(3, CommonUtils.ceil_div(9, 4))
        self.assertEqual(12_000_000, CommonUtils.milliseconds_to_ns(12))


if __name__ == "__main__":
    unittest.main()
