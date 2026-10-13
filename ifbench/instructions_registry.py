# Copyright 2025 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

from ifbench import classic_instructions, instructions


_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"


INSTRUCTION_DICT = {
    # --- Classic Google IFEval verifiers ---
    _KEYWORD + "existence": classic_instructions.KeywordChecker,
    _KEYWORD + "frequency": classic_instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": classic_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": classic_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": classic_instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": classic_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": classic_instructions.ParagraphChecker,
    _LENGTH + "number_words": classic_instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": classic_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": classic_instructions.PlaceholderChecker,
    _CONTENT + "postscript": classic_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": classic_instructions.BulletListChecker,
    _FORMAT + "constrained_response": classic_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": classic_instructions.HighlightSectionChecker,
    _FORMAT + "multiple_sections": classic_instructions.SectionChecker,
    _FORMAT + "json_format": classic_instructions.JsonFormat,
    _FORMAT + "title": classic_instructions.TitleChecker,
    _COMBINATION + "two_responses": classic_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": classic_instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": classic_instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": classic_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": classic_instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": classic_instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": classic_instructions.CommaChecker,
    _STARTEND + "quotation": classic_instructions.QuotationChecker,
    # --- IFBench OOD verifiers ---
    "count:word_count_range": instructions.WordCountRangeChecker,
    "count:unique_word_count" : instructions.UniqueWordCountChecker,
    "ratio:stop_words" : instructions.StopWordPercentageChecker,
    "ratio:sentence_type" : instructions.SentTypeRatioChecker,
    "ratio:sentence_balance" : instructions.SentBalanceChecker,
    "count:conjunctions" : instructions.ConjunctionCountChecker,
    "count:person_names" : instructions.PersonNameCountChecker,
    "ratio:overlap" : instructions.NGramOverlapChecker,
    "count:numbers" : instructions.NumbersCountChecker,
    "words:alphabet" : instructions.AlphabetLoopChecker,
    "words:vowel" : instructions.SingleVowelParagraphChecker,
    "words:consonants" : instructions.ConsonantClusterChecker,
    "sentence:alliteration_increment" : instructions.IncrementingAlliterationChecker,
    "words:palindrome" : instructions.PalindromeChecker,
    "count:punctuation" : instructions.PunctuationCoverChecker,
    "format:parentheses" : instructions.NestedParenthesesChecker,
    "format:quotes" : instructions.NestedQuotesChecker,
    "words:prime_lengths" : instructions.PrimeLengthsChecker,
    "format:options" : instructions.OptionsResponseChecker,
    "format:newline" : instructions.NewLineWordsChecker,
    "format:emoji" : instructions.EmojiSentenceChecker,
    "ratio:sentence_words" : instructions.CharacterCountUniqueWordsChecker,
    "count:words_japanese" : instructions.NthWordJapaneseChecker,
    "words:start_verb" : instructions.StartWithVerbChecker,
    "words:repeats" : instructions.LimitedWordRepeatChecker,
    "sentence:keyword" : instructions.IncludeKeywordChecker,
    "count:pronouns" : instructions.PronounCountChecker,
    "words:odd_even_syllables" : instructions.AlternateParitySyllablesChecker,
    "words:last_first" : instructions.LastWordFirstNextChecker,
    "words:paragraph_last_first" : instructions.ParagraphLastFirstWordMatchChecker,
    "sentence:increment" : instructions.IncrementingWordCountChecker,
    "words:no_consecutive" : instructions.NoConsecutiveFirstLetterChecker,
    "format:line_indent" : instructions.IndentStairsChecker,
    "format:quote_unquote" : instructions.QuoteExplanationChecker,
    "format:list" : instructions.SpecialBulletPointsChecker,
    "format:thesis" : instructions.ItalicsThesisChecker,
    "format:sub-bullets" : instructions.SubBulletPointsChecker,
    "format:no_bullets_bullets" : instructions.SomeBulletPointsChecker,
    "custom:multiples" : instructions.PrintMultiplesChecker,
    "custom:mcq_count_length": instructions.MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": instructions.ReverseNewlineChecker,
    "custom:word_reverse": instructions.WordReverseOrderChecker,
    "custom:character_reverse": instructions.CharacterReverseOrderChecker,
    "custom:sentence_alphabet": instructions.SentenceAlphabetChecker,
    "custom:european_capitals_sort": instructions.EuropeanCapitalsSortChecker,
    "custom:csv_city": instructions.CityCSVChecker,
    "custom:csv_special_character": instructions.SpecialCharacterCSVChecker,
    "custom:csv_quotes": instructions.QuotesCSVChecker,
    "custom:date_format_list": instructions.DateFormatListChecker,
    "count:keywords_multiple" : instructions.KeywordsMultipleChecker,
    "words:keywords_specific_position" : instructions.KeywordSpecificPositionChecker,
    "words:words_position" : instructions.WordsPositionChecker,
    "repeat:repeat_change" : instructions.RepeatChangeChecker,
    "repeat:repeat_simple" : instructions.RepeatSimpleChecker,
    "repeat:repeat_span" : instructions.RepeatSpanChecker,
    "format:title_case" : instructions.TitleCaseChecker,
    "format:output_template" : instructions.OutputTemplateChecker,
    "format:no_whitespace" : instructions.NoWhitespaceChecker,
}

# Các bộ kiểm tra dưới đây chỉ dựa trên cấu trúc hoặc phép đếm xác định và đã
# được đối chiếu trực tiếp với câu chỉ dẫn tiếng Việt. Đây là danh sách bảo thủ
# nên dùng khi tạo dữ liệu huấn luyện tiếng Việt.
VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS = frozenset({
    "keywords:existence", "keywords:frequency", "keywords:forbidden_words",
    "keywords:letter_frequency", "length_constraints:number_sentences",
    "length_constraints:number_paragraphs", "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders", "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections", "detectable_format:json_format",
    "detectable_format:title", "combination:two_responses",
    "combination:repeat_prompt", "startend:end_checker",
    "change_case:capital_word_frequency", "punctuation:no_comma",
    "startend:quotation", "count:word_count_range",
    "count:unique_word_count", "ratio:sentence_type",
    "ratio:sentence_balance", "count:conjunctions", "count:person_names",
    "ratio:overlap", "count:numbers", "words:alphabet", "words:vowel",
    "words:consonants", "sentence:alliteration_increment",
    "count:punctuation", "format:parentheses", "format:quotes",
    "words:prime_lengths", "format:options", "format:newline",
    "ratio:sentence_words", "words:start_verb", "words:repeats",
    "sentence:keyword", "words:odd_even_syllables", "words:last_first",
    "words:paragraph_last_first", "sentence:increment",
    "words:no_consecutive", "format:line_indent", "format:quote_unquote",
    "format:list", "format:thesis", "format:sub-bullets",
    "format:no_bullets_bullets", "custom:multiples",
    "custom:reverse_newline", "custom:sentence_alphabet", "custom:csv_city",
    "custom:csv_special_character", "custom:csv_quotes",
    "count:keywords_multiple", "words:keywords_specific_position",
    "words:words_position", "repeat:repeat_change", "repeat:repeat_simple",
    "repeat:repeat_span", "format:title_case", "format:output_template",
    "format:no_whitespace",
})

# Không dùng các mục này cho dữ liệu được gắn nhãn hoàn toàn tự động. Lý do là
# bộ kiểm tra chỉ xấp xỉ ngôn ngữ/ngữ nghĩa hoặc yêu cầu kiến thức thực tế mà
# một phép kiểm tra chuỗi không thể chứng minh đầy đủ.
VIETNAMESE_EXPERIMENTAL_REASONS = {
    "language:response_language": "langdetect chỉ ước lượng ngôn ngữ, nhất là với câu ngắn",
    "change_case:english_capital": "phụ thuộc langdetect ngoài phép kiểm tra chữ hoa",
    "change_case:english_lowercase": "phụ thuộc langdetect ngoài phép kiểm tra chữ thường",
    "ratio:stop_words": "khái niệm hư từ tiếng Việt không có một danh sách chuẩn duy nhất",
    "words:palindrome": "rất khó tạo đủ từ đơn tiếng Việt tự nhiên thỏa điều kiện",
    "format:emoji": "ranh giới câu quanh emoji không có quy ước duy nhất",
    "count:words_japanese": "là nhiệm vụ trộn tiếng Nhật, không thuần tiếng Việt",
    "count:pronouns": "ranh giới và chức năng đại từ tiếng Việt phụ thuộc ngữ cảnh",
    "custom:mcq_count_length": "bộ kiểm tra không xác minh chủ đề hay tính đúng đắn của câu hỏi",
    "custom:word_reverse": "bộ kiểm tra chỉ tìm đáp án con, không chứng minh toàn câu hợp lệ",
    "custom:character_reverse": "bộ kiểm tra chỉ tìm đáp án con, không chứng minh toàn câu hợp lệ",
    "custom:european_capitals_sort": "phụ thuộc danh sách địa chính trị và cách gọi địa danh",
    "custom:date_format_list": "bộ kiểm tra định dạng không xác minh đủ mọi trận chiến",
}

assert VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS.isdisjoint(
    VIETNAMESE_EXPERIMENTAL_REASONS)
assert VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS | set(
    VIETNAMESE_EXPERIMENTAL_REASONS) == set(INSTRUCTION_DICT)
