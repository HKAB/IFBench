# coding=utf-8
"""Tests for the Vietnamese RL dataset generator."""

import json
import tempfile
import unittest
from pathlib import Path

from generate_vi_rl_dataset import DEFAULT_BASE_PROMPTS, generate_dataset
from ifbench import instructions_registry
from ifbench import instructions


class VietnameseRLDatasetGenerationTest(unittest.TestCase):

    def test_unique_word_checker_counts_all_words(self):
        checker = instructions.UniqueWordCountChecker("count:unique_word_count")
        checker.build_description(N=3)
        self.assertTrue(checker.check_following("mây gió nắng mây"))
        self.assertFalse(checker.check_following("mây mây gió"))

    def test_generated_splits_are_disjoint_and_rebuildable(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = generate_dataset(
                output_dir=output_dir,
                prompts=list(DEFAULT_BASE_PROMPTS),
                train_size=80,
                val_size=75,
                seed=17,
            )
            train = [json.loads(line) for line in
                     (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(line) for line in
                   (output_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(train), 80)
            self.assertEqual(len(val), 75)
            self.assertTrue({row["prompt"] for row in train}.isdisjoint(
                row["prompt"] for row in val))
            self.assertFalse(set(manifest["instruction_ids"]) -
                             instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS)

            for row in train + val:
                self.assertEqual(row["messages"][0]["content"], row["prompt"])
                self.assertEqual(len(row["instruction_id_list"]), len(row["kwargs"]))
                for instruction_id, kwargs in zip(
                        row["instruction_id_list"], row["kwargs"]):
                    checker = instructions_registry.INSTRUCTION_DICT[instruction_id](
                        instruction_id)
                    description = checker.build_description(**kwargs)
                    self.assertTrue(description)
                    self.assertEqual(checker.get_instruction_args() or {}, kwargs)


if __name__ == "__main__":
    unittest.main()
