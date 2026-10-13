# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Library of classic Google IFEval instructions.

Verbatim from google-research/google-research/instruction_following_eval/
instructions.py, with two changes: the `Instruction` base class is imported
from `instructions` (IFBench's module) instead of redefined, and the util
import is rewritten to the flat top-level layout used by this package.
"""
import collections
import json
import random
import re
import string
import unicodedata
from typing import Dict, Optional, Sequence, Union

from absl import logging
import langdetect

from ifbench import instructions_util
from ifbench.instructions import Instruction


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")
_COMPARISON_RELATION_VI = {
    "less than": "ít hơn",
    "at least": "ít nhất",
}

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "Câu trả lời của tôi là có.", "Câu trả lời của tôi là không.",
    "Câu trả lời của tôi là có thể.")

# The options of starter keywords.
_STARTER_OPTIONS = ("Tôi cho rằng", "Câu trả lời của tôi là", "Tôi tin rằng",
                    "Theo ý kiến của tôi", "Tôi nghĩ rằng", "Tôi cảm thấy rằng",
                    "Theo góc nhìn của tôi", "Theo tôi thấy", "Theo hiểu biết của tôi")

# The options of ending keywords.
_ENDING_OPTIONS = ("Bạn còn câu hỏi nào khác không?",
                   "Tôi có thể giúp gì thêm cho bạn không?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Phần", "PHẦN")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("T.B.", "T.B.2")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class ResponseLanguageChecker(Instruction):
  """Check the language of the entire response."""

  def build_description(self, *, language = None):
    self._language = language
    if self._language is None:
      self._language = "vi"
    self._description_pattern = (
        "TOÀN BỘ câu trả lời phải được viết bằng {language}; không được dùng " +
        "ngôn ngữ nào khác.")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    return {"language": self._language}

  def get_instruction_args_keys(self):
    return ["language"]

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return langdetect.detect(value) == self._language
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )
      return False


class NumberOfSentences(Instruction):
  """Check the number of sentences."""

  def build_description(self, *, num_sentences = None, relation = None):
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or
        self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Câu trả lời phải có {relation} {num_sentences} câu.")
    return self._description_pattern.format(
        relation=_COMPARISON_RELATION_VI[self._comparison_relation],
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["num_sentences", "relation"]

  def check_following(self, value):
    num_sentences = instructions_util.count_sentences(value)
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_sentences < self._num_sentences_threshold
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders = None):
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "Câu trả lời phải có ít nhất {num_placeholders} chỗ trống được biểu " +
        "diễn bằng ngoặc vuông, chẳng hạn [địa chỉ].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    return ["num_placeholders"]

  def check_following(self, value):
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Câu trả lời phải có đúng {num_bullets} mục đầu dòng. " +
        "Dùng cú pháp Markdown như sau:\n" +
        "* Đây là mục 1. \n" +
        "* Đây là mục 2")
    return self._description_pattern.format(num_bullets=self._num_bullets)

  def get_instruction_args(self):
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    return ["num_bullets"]

  def check_following(self, value):
    bullet_lists = re.findall(r"^\s*\*\s+\S.*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-\s+\S.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
  """Checks the constrained response."""

  def build_description(self):
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = (
        "Chỉ trả lời bằng một trong các phương án sau: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = value.strip()
    return value in self._constrained_responses


class ConstrainedStartChecker(Instruction):
  """Checks the response start."""

  def build_description(self, *, starter = None):
    self._starter = starter.strip() if isinstance(starter, str) else starter
    if self._starter is None:
      self._starter = random.choice(_STARTER_OPTIONS)
    self._description_pattern = (
        "Trong cuộc trò chuyện, mỗi khi đến lượt, hãy luôn bắt đầu bằng đúng " +
        "cụm từ: {starter}")
    return self._description_pattern.format(starter=self._starter)

  def get_instruction_args(self):
    return {"starter": self._starter}

  def get_instruction_args_keys(self):
    return ["starter"]

  def check_following(self, value):
    response_pattern = r"^\s*" + re.escape(self._starter)
    response_with_constrained_start = re.search(response_pattern, value)
    return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights = None):
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "Hãy làm nổi bật ít nhất {num_highlights} đoạn bằng Markdown, ví dụ " +
        "*đoạn được làm nổi bật*.")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    return {"num_highlights": self._num_highlights}

  def get_instruction_args_keys(self):
    return ["num_highlights"]

  def check_following(self, value):
    highlights = re.findall(r"(?<!\*)\*{1,2}([^\n*]+?)\*{1,2}(?!\*)", value)
    return sum(bool(highlight.strip()) for highlight in highlights) >= self._num_highlights


class SectionChecker(Instruction):
  """Checks the sections."""

  def build_description(self, *, section_spliter = None, num_sections = None):
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    self._description_pattern = (
        "Câu trả lời phải có ít nhất {num_sections} phần. Đánh dấu đầu mỗi " +
        "phần bằng {section_spliter} X, ví dụ:\n" +
        "{section_spliter} 1\n" +
        "[nội dung phần 1]\n" +
        "{section_spliter} 2\n" +
        "[nội dung phần 2]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter)

  def get_instruction_args(self):
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections}

  def get_instruction_args_keys(self):
    return ["section_spliter", "num_sections"]

  def check_following(self, value):
    section_splitter_patten = r"(?:^|\n)\s*" + re.escape(
        self._section_spliter) + r"\s+\d+\s*(?:\n|$)"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
  """Checks the paragraphs."""

  def build_description(self, *, num_paragraphs = None):
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "Câu trả lời phải có đúng {num_paragraphs} đoạn. " +
        "Phân cách các đoạn bằng đường kẻ Markdown: ***")

    return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    return ["num_paragraphs"]

  def check_following(self, value):
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False

    return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
  """Checks the postscript."""

  def build_description(self, *, postscript_marker = None):
    self._postscript_marker = postscript_marker.strip() if isinstance(
        postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = (
        "Ở cuối câu trả lời, hãy thêm một phần tái bút bắt đầu bằng {postscript}")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    return ["postscript_marker"]

  def check_following(self, value):
    value = value.lower()
    postscript_pattern = r"(?:^|\n)\s*" + re.escape(
        self._postscript_marker.lower()) + r"[^\n]*\s*$"
    postscript = re.findall(postscript_pattern, value)
    return True if postscript else False


class RephraseChecker(Instruction):
  """Checks the repharse."""

  def build_description(self, *, original_message):
    if not self.is_change(original_message):
      raise ValueError(f"Message {original_message} does not contain changes "
                       "in the form of *change me*.")

    self._reference_without_change = original_message
    self._description = ("Viết lại: chỉ được thay đổi các từ hoặc câu nằm " +
                         "giữa hai dấu sao, chẳng hạn *hãy đổi tôi*.")
    return self._description

  def get_instruction_args(self):
    return {"original_message": self._reference_without_change}

  def get_instruction_args_keys(self):
    return ["original_message"]

  def check_following(self, value):
    if not self.is_change(value):
      raise ValueError(f"value {value} does not contain "
                       "changes in the form of *change me*.")

    response_without_changes = self.strip_changes(value)
    reference_without_changes = self.strip_changes(
        self._reference_without_change)

    return response_without_changes == reference_without_changes

  def is_change(self, response):
    return re.search(r"\*.*\*", response)

  def strip_changes(self, response):
    return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
  """Check the exisitence of certain keywords."""

  def build_description(self, *, keywords = None):
    if not keywords:
      self._keywords = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("Câu trả lời phải chứa tất cả từ khóa sau: {keywords}.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    return ["keywords"]

  def check_following(self, value):
    for keyword in self._keywords:
      if not re.search(r"\b" + re.escape(keyword) + r"\b",
                       value, flags=re.IGNORECASE):
        return False
    return True


class KeywordFrequencyChecker(Instruction):
  """Check the keyword frequency."""

  def build_description(self, *, keyword = None, frequency = None,
                        relation = None):
    if not keyword:
      self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Trong câu trả lời, từ {keyword} phải xuất hiện {relation} " +
        "{frequency} lần.")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=_COMPARISON_RELATION_VI[self._comparison_relation],
        frequency=self._frequency)

  def get_instruction_args(self):
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    actual_occurrences = len(re.findall(
        r"\b" + re.escape(self._keyword) + r"\b", value, flags=re.IGNORECASE))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return actual_occurrences < self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
  """Checks the number of words."""

  def build_description(self, *, num_words = None, relation = None):
    self._num_words = num_words
    if self._num_words is None or self._num_words < 0:
      self._num_words = random.randint(
          _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
      )

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Trả lời bằng {relation} {num_words} đơn vị từ; mỗi chuỗi chữ cái hoặc "
        "chữ số Unicode liên tiếp được tính là một đơn vị.")

    return self._description_pattern.format(
        relation=_COMPARISON_RELATION_VI[self._comparison_relation],
        num_words=self._num_words)

  def get_instruction_args(self):
    return {"num_words": self._num_words,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["num_words", "relation"]

  def check_following(self, value):
    num_words = instructions_util.count_words(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_words < self._num_words
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_words >= self._num_words


class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):
    self._description_pattern = (
        "Toàn bộ đầu ra phải là một giá trị JSON hợp lệ. Có thể bọc JSON trong "
        "hàng rào mã Markdown ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True


class ParagraphFirstWordCheck(Instruction):
  """Check the paragraph and the first word of the nth paragraph."""

  def build_description(self, num_paragraphs = None, nth_paragraph = None,
                        first_word = None):
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
    self._first_word = self._first_word.lower()

    self._description_pattern = (
        "Câu trả lời phải có đúng {num_paragraphs} đoạn. Chỉ dùng đúng hai ký " +
        "tự xuống dòng (tức '\\n\\n' trong Python) để phân cách các đoạn. " +
        "Đoạn thứ {nth_paragraph} phải bắt đầu bằng từ {first_word}.")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False
    else:
      return False

    first_word = ""
    punctuation = {".", ",", "?", "!", "'", '"'}

    word = paragraph.split()[0].strip()
    word = word.lstrip("'")
    word = word.lstrip('"')

    for letter in word:
      if letter in punctuation:
        break
      first_word += letter.lower()

    return (
        num_paragraphs == self._num_paragraphs
        and first_word == self._first_word
    )


class KeySentenceChecker(Instruction):
  """Check the existence of certain key sentences."""

  def build_description(self, key_sentences = None, num_sentences = None):
    if not key_sentences:
      self._key_sentences = set(["Hiện tại, như vậy là ổn."])
    else:
      self._key_sentences = key_sentences

    if not num_sentences:
      self._num_sentences = random.randint(1, len(self._key_sentences))
    else:
      self._num_sentences = num_sentences

    self._description_pattern = (
        "Hãy chèn đúng {num_sentences} câu trong tập câu sau: {key_sentences}"
    )

    return self._description_pattern.format(
        num_sentences=self._num_sentences, key_sentences=self._key_sentences
    )

  def get_instruction_args(self):
    return {"num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences)}

  def get_instruction_args_keys(self):
    return ["num_sentences", "key_sentences"]

  def check_following(self, value):
    count = 0
    sentences = instructions_util.split_into_sentences(value)
    for sentence in self._key_sentences:
      if sentence in sentences:
        count += 1

    return count == self._num_sentences


class ForbiddenWords(Instruction):
  """Checks that specified words are not used in response."""

  def build_description(self, forbidden_words = None):
    if not forbidden_words:
      self._forbidden_words = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "Không được dùng bất kỳ từ khóa nào sau đây trong câu trả lời: {forbidden_words}."
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words
    )

  def get_instruction_args(self):
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    return ["forbidden_words"]

  def check_following(self, value):
    for word in self._forbidden_words:
      if re.search(r"\b" + re.escape(word) + r"\b", value, flags=re.IGNORECASE):
        return False
    return True


class RephraseParagraph(Instruction):
  """Checks that the paragraph is rephrased."""

  def build_description(self, *, original_paragraph, low, high):
    self._original_paragraph = original_paragraph
    self._low = low
    self._high = high

    self._description = ("Hãy diễn đạt lại đoạn sau: {original_paragraph}\n" +
                         "Câu trả lời phải có từ {low} đến {high} từ trùng với " +
                         "đoạn gốc. Hai từ được xem là trùng khi toàn bộ chữ " +
                         "cái giống nhau nếu không phân biệt hoa, thường; ví " +
                         "dụ 'Nhà' trùng với 'nhà' nhưng khác 'nhàu'.")

    return self._description.format(original_paragraph=original_paragraph,
                                    low=self._low, high=self._high)

  def get_instruction_args(self):
    return {"original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high}

  def get_instruction_args_keys(self):
    return ["original_paragraph", "low", "high"]

  def check_following(self, value):
    val_words = re.findall(r"\w+", value.lower())
    original_words = re.findall(r"\w+", self._original_paragraph.lower())
    similar_words = 0

    dict_val = collections.Counter(val_words)
    dict_original = collections.Counter(original_words)

    for word in dict_original:
      similar_words += min(dict_original[word], dict_val[word])

    return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
  """Check that two responses were given."""

  def build_description(self):
    self._description_pattern = (
        "Đưa ra hai câu trả lời khác nhau. Chỉ phân cách hai câu trả lời bằng "
        "đúng 6 dấu sao: ******."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    valid_responses = list()
    responses = value.split("******")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False
      else:
        valid_responses.append(response)
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    )


class RepeatPromptThenAnswer(Instruction):
  """Checks that Prompt is first repeated then answered."""

  def build_description(self, *, prompt_to_repeat = None):
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeat must be set.")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "Trước tiên, lặp lại nguyên văn yêu cầu rồi mới trả lời. (1) Không "
        "viết bất kỳ từ hay ký tự nào trước phần lặp lại. (2) Câu chỉ dẫn này "
        "không thuộc yêu cầu cần lặp lại."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    return ["prompt_to_repeat"]

  def check_following(self, value):
    stripped_value = value.strip()
    prompt = self._prompt_to_repeat.strip()
    if not stripped_value.casefold().startswith(prompt.casefold()):
      return False
    return bool(stripped_value[len(prompt):].strip())


class EndChecker(Instruction):
  """Checks that the prompt ends with a given phrase."""

  def build_description(self, *, end_phrase = None):
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Kết thúc câu trả lời bằng đúng cụm từ sau: {ender} Không được viết "
        "thêm bất kỳ từ nào sau cụm từ đó.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    return ["end_phrase"]

  def check_following(self, value):
    value = value.strip().strip("\"").lower()
    self._end_phrase = self._end_phrase.strip().lower()
    return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):
    self._description_pattern = (
        "Câu trả lời phải có một tiêu đề được bọc trong hai cặp ngoặc nhọn, "
        "chẳng hạn <<bài thơ niềm vui>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True
    return False


class LetterFrequencyChecker(Instruction):
  """Checks letter frequency."""

  def build_description(self, *, letter = None, let_frequency = None,
                        let_relation = None):
    if (
        not letter
        or len(letter) > 1
        or not letter.isalpha()
    ):
      self._letter = random.choice(
          list("aăâbcdđeêghiklmnoôơpqrstuưvxy"))
    else:
      self._letter = letter.strip()
    self._letter = self._letter.lower()

    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {let_relation} is given."
      )
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "Trong câu trả lời, chữ cái {letter} phải xuất hiện {let_relation}"
        " {let_frequency} lần."
    )

    return self._description_pattern.format(
        letter=self._letter,
        let_frequency=self._frequency,
        let_relation=_COMPARISON_RELATION_VI[self._comparison_relation],
    )

  def get_instruction_args(self):
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    value = unicodedata.normalize("NFC", value).lower()
    self._letter = unicodedata.normalize("NFC", self._letter)
    letters = collections.Counter(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return letters[self._letter] < self._frequency
    else:
      return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all capital letters."""

  def build_description(self):
    self._description_pattern = (
        "Toàn bộ câu trả lời phải bằng tiếng Việt và mọi chữ cái phải viết hoa."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return value.isupper() and langdetect.detect(value) == "vi"
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )
      return False


class LowercaseLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all lowercase letters."""

  def build_description(self):
    self._description_pattern = (
        "Toàn bộ câu trả lời phải bằng tiếng Việt và mọi chữ cái phải viết "
        "thường; không được có chữ cái viết hoa."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return value.islower() and langdetect.detect(value) == "vi"
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )
      return False


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):
    self._description_pattern = (
        "Không được dùng dấu phẩy ở bất kỳ đâu trong câu trả lời."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
  """Checks frequency of words with all capital letters."""

  def build_description(self, capital_frequency = None,
                        capital_relation = None):
    self._frequency = capital_frequency
    if self._frequency is None:
      self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

    self._comparison_relation = capital_relation
    if capital_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif capital_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {capital_relation} is given."
      )

    self._description_pattern = (
        "Trong câu trả lời, số từ được viết hoàn toàn bằng chữ in hoa phải là"
        " {relation} {frequency}."
    )

    return self._description_pattern.format(
        frequency=self._frequency,
        relation=_COMPARISON_RELATION_VI[self._comparison_relation]
    )

  def get_instruction_args(self):
    return {
        "capital_frequency": self._frequency,
        "capital_relation": self._comparison_relation,
    }

  def get_instruction_args_keys(self):
    return ["capital_frequency", "capital_relation"]

  def check_following(self, value):
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    capital_words = [word for word in words if word.isupper()]

    capital_words = len(capital_words)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return capital_words < self._frequency
    else:
      return capital_words >= self._frequency


class QuotationChecker(Instruction):
  """Checks response is wrapped with double quotation marks."""

  def build_description(self):
    self._description_pattern = (
        "Bọc toàn bộ câu trả lời trong một cặp dấu ngoặc kép."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'
