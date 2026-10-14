# coding=utf-8
"""Tests for the Vietnamese RL dataset generator."""

import json
import tempfile
import unittest
from pathlib import Path

from generate_vi_rl_dataset import (
    CORE_INSTRUCTION_IDS,
    DEFAULT_BASE_PROMPTS,
    HARD_INSTRUCTION_IDS,
    TASK_SETS,
    generate_dataset,
)
from ifbench import instructions_registry
from ifbench import instructions


class VietnameseRLDatasetGenerationTest(unittest.TestCase):

    def test_default_prompt_bank_is_large_and_unique(self):
        self.assertGreaterEqual(len(DEFAULT_BASE_PROMPTS), 100)
        self.assertEqual(len(set(DEFAULT_BASE_PROMPTS)),
                         len(DEFAULT_BASE_PROMPTS))
        self.assertTrue(all(len(prompt.split()) >= 2
                            for prompt in DEFAULT_BASE_PROMPTS))

    def test_audited_task_sets_are_disjoint_and_complete(self):
        self.assertEqual(len(CORE_INSTRUCTION_IDS), 39)
        self.assertEqual(len(HARD_INSTRUCTION_IDS), 13)
        self.assertTrue(CORE_INSTRUCTION_IDS.isdisjoint(HARD_INSTRUCTION_IDS))
        self.assertEqual(
            TASK_SETS["all"], CORE_INSTRUCTION_IDS | HARD_INSTRUCTION_IDS)
        self.assertTrue(TASK_SETS["all"].issubset(
            instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS))

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
                             CORE_INSTRUCTION_IDS)
            self.assertEqual(manifest["task_set"], "core")
            self.assertEqual(manifest["available_instruction_count"], 39)
            self.assertEqual(set(manifest["instruction_ids"]), CORE_INSTRUCTION_IDS)

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

    def test_hard_dataset_uses_only_hard_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_dataset(
                output_dir=Path(directory),
                prompts=list(DEFAULT_BASE_PROMPTS),
                train_size=20,
                val_size=10,
                seed=23,
                task_set="hard",
            )
            self.assertEqual(manifest["task_set"], "hard")
            self.assertEqual(manifest["available_instruction_count"], 13)
            self.assertEqual(set(manifest["instruction_ids"]), HARD_INSTRUCTION_IDS)


if __name__ == "__main__":
    unittest.main()
