# coding=utf-8
"""Kiểm thử hợp đồng bản địa hóa tiếng Việt cho toàn bộ registry."""

import unittest

from ifbench import classic_instructions
from ifbench import instructions
from ifbench import instructions_registry
from ifbench import instructions_util


_SPECIAL_BUILD_ARGS = {
    "combination:repeat_prompt": {
        "prompt_to_repeat": "Hãy kể một câu chuyện ngắn về quê hương."
    },
    "ratio:overlap": {
        "reference_text": "Mây trắng trôi trên bầu trời xanh.",
        "percentage": 50,
    },
    "repeat:repeat_change": {
        "prompt_to_repeat": "Hãy viết một câu về mùa thu."
    },
    "repeat:repeat_span": {
        "prompt_to_repeat": "một hai ba bốn năm",
        "n_start": 1,
        "n_end": 3,
    },
}

_ENGLISH_INSTRUCTION_FRAGMENTS = (
    "the response", "your response", "your answer", "answer with",
    "include keyword", "write each", "generate ", "do not ",
    "only output", "repeat the request", "copy the span",
)


class VietnameseRegistryContractTest(unittest.TestCase):

    def _build_all(self):
        for instruction_id, checker_cls in instructions_registry.INSTRUCTION_DICT.items():
            checker = checker_cls(instruction_id)
            description = checker.build_description(
                **_SPECIAL_BUILD_ARGS.get(instruction_id, {}))
            yield instruction_id, checker, description

    def test_every_registered_description_is_vietnamese(self):
        """Mỗi tác vụ có mô tả tiếng Việt, không còn mệnh lệnh tiếng Anh."""
        for instruction_id, _, description in self._build_all():
            with self.subTest(instruction_id=instruction_id):
                self.assertIsInstance(description, str)
                self.assertTrue(description.strip())
                self.assertTrue(any(ord(char) > 127 for char in description))
                lowered = description.casefold()
                self.assertFalse(any(
                    fragment in lowered
                    for fragment in _ENGLISH_INSTRUCTION_FRAGMENTS
                ))

    def test_every_registered_argument_contract_round_trips(self):
        """Hai giao diện tham số của từng tác vụ luôn nhất quán sau bản địa hóa."""
        for instruction_id, checker, _ in self._build_all():
            with self.subTest(instruction_id=instruction_id):
                keys = checker.get_instruction_args_keys()
                args = checker.get_instruction_args()
                self.assertIsInstance(keys, list)
                if args is None:
                    self.assertEqual(keys, [])
                else:
                    self.assertEqual(set(args), set(keys))

    def test_registry_has_explicit_safe_or_experimental_status(self):
        all_ids = set(instructions_registry.INSTRUCTION_DICT)
        safe = set(instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS)
        experimental = set(instructions_registry.VIETNAMESE_EXPERIMENTAL_REASONS)
        self.assertEqual(safe | experimental, all_ids)
        self.assertFalse(safe & experimental)
        self.assertEqual(len(all_ids), 83)


class VietnameseArtifactBehaviorTest(unittest.TestCase):

    def test_generated_keywords_are_vietnamese_and_single_unit(self):
        for _ in range(20):
            keywords = instructions_util.generate_keywords(5)
            self.assertEqual(len(keywords), len(set(keywords)))
            self.assertTrue(all(" " not in keyword for keyword in keywords))
            self.assertTrue(all(
                keyword in instructions_util.VIETNAMESE_WORD_LIST
                for keyword in keywords
            ))

    def test_vietnamese_conjunctions_positive_and_english_negative(self):
        checker = instructions.ConjunctionCountChecker("count:conjunctions")
        checker.build_description(small_n=3)
        self.assertTrue(checker.check_following("Tôi đi và bạn ở nhưng họ về nên ta chờ."))
        self.assertFalse(checker.check_following("I go and you stay but they leave, so we wait."))

    def test_vietnamese_names_positive_and_foreign_names_negative(self):
        checker = instructions.PersonNameCountChecker("count:person_names")
        checker.build_description(N=3)
        self.assertTrue(checker.check_following("An gặp Bình và Chi."))
        self.assertFalse(checker.check_following("Emma met Liam and Sophia."))

    def test_vietnamese_alphabet_positive_and_wrong_order_negative(self):
        checker = instructions.AlphabetLoopChecker("words:alphabet")
        checker.build_description()
        self.assertTrue(checker.check_following("A ăn âm bờ ca dê đê em êm gỗ."))
        self.assertFalse(checker.check_following("A âm ăn bờ."))

    def test_vietnamese_verb_start_positive_and_unlisted_negative(self):
        checker = instructions.StartWithVerbChecker("words:start_verb")
        checker.build_description()
        self.assertTrue(checker.check_following("Đọc kỹ phần này."))
        self.assertFalse(checker.check_following("Phần này cần được đọc kỹ."))

    def test_localized_exact_response_artifacts(self):
        constrained = classic_instructions.ConstrainedResponseChecker(
            "detectable_format:constrained_response")
        constrained.build_description()
        self.assertTrue(constrained.check_following("Câu trả lời của tôi là có."))
        self.assertFalse(constrained.check_following("My answer is yes."))

        template = instructions.OutputTemplateChecker("format:output_template")
        template.build_description()
        self.assertTrue(template.check_following(
            "Câu trả lời của tôi: Nội dung. "
            "Kết luận của tôi: Kết luận. "
            "Triển vọng tương lai: Tích cực."
        ))
        self.assertFalse(template.check_following(
            "My Answer: text My Conclusion: end Future Outlook: good"
        ))

    def test_localized_csv_headers_positive_and_english_negative(self):
        checker = instructions.CityCSVChecker("custom:csv_city")
        checker.build_description()
        rows = "\n".join(f"{i},Việt Nam,Huế,2026,{i}" for i in range(1, 8))
        self.assertTrue(checker.check_following(
            "Mã,Quốc gia,Thành phố,Năm,Số lượng\n" + rows))
        self.assertFalse(checker.check_following(
            "ID,Country,City,Year,Count\n" + rows))

    def test_nested_parentheses_reject_unclosed_outer_groups(self):
        checker = instructions.NestedParenthesesChecker("format:parentheses")
        checker.build_description()
        self.assertTrue(checker.check_following("({[({nội dung})]})"))
        self.assertFalse(checker.check_following("({[({nội dung})]"))


if __name__ == "__main__":
    unittest.main()
