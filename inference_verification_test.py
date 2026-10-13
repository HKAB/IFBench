# coding=utf-8
"""Unit tests for local verification in the concurrent inference script."""

import unittest

from generate_and_verify import build_summary, verify_response


class InferenceVerificationTest(unittest.TestCase):

    def setUp(self):
        self.row = {
            "key": 1,
            "prompt": "Nêu một màu sắc. Không được dùng dấu phẩy.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }

    def test_verify_response_records_strict_and_loose_results(self):
        passed = verify_response(self.row, "Màu xanh")
        failed = verify_response(self.row, "Màu xanh, màu đỏ")
        self.assertTrue(passed["strict_follow_all"])
        self.assertTrue(passed["loose_follow_all"])
        self.assertFalse(failed["strict_follow_all"])
        self.assertFalse(failed["loose_follow_all"])

    def test_summary_counts_errors_and_instruction_scores(self):
        result = {
            **self.row,
            "response": "Màu xanh",
            "model": "test-model",
            "strict_follow_all": True,
            "strict_follow_instruction_list": [True],
            "loose_follow_all": True,
            "loose_follow_instruction_list": [True],
            "error": None,
        }
        summary = build_summary(
            [result], model="test-model", input_file="sample.jsonl")
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["strict_accuracy"], 1.0)
        self.assertEqual(
            summary["per_instruction"]["punctuation:no_comma"]["strict_correct"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
