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

"""Utility library of instructions."""

import random
import re

import immutabledict

LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "tiếng Anh",
    "es": "tiếng Tây Ban Nha",
    "pt": "tiếng Bồ Đào Nha",
    "ar": "tiếng Ả Rập",
    "hi": "tiếng Hindi",
    "fr": "tiếng Pháp",
    "ru": "tiếng Nga",
    "de": "tiếng Đức",
    "ja": "tiếng Nhật",
    "it": "tiếng Ý",
    "bn": "tiếng Bengal",
    "uk": "tiếng Ukraina",
    "th": "tiếng Thái",
    "ur": "tiếng Urdu",
    "ta": "tiếng Tamil",
    "te": "tiếng Telugu",
    "bg": "tiếng Bulgaria",
    "ko": "tiếng Hàn",
    "pl": "tiếng Ba Lan",
    "he": "tiếng Hebrew",
    "fa": "tiếng Ba Tư",
    "vi": "tiếng Việt",
    "ne": "tiếng Nepal",
    "sw": "tiếng Swahili",
    "kn": "tiếng Kannada",
    "mr": "tiếng Marathi",
    "gu": "tiếng Gujarat",
    "pa": "tiếng Punjab",
    "ml": "tiếng Malayalam",
    "fi": "tiếng Phần Lan",
})

# Từ khóa mặc định dùng để sinh chỉ dẫn tiếng Việt. Mỗi mục là một đơn vị
# cách nhau bằng khoảng trắng để các bộ kiểm tra vị trí/tần suất không bị mơ hồ
# bởi bài toán tách từ tiếng Việt.
VIETNAMESE_WORD_LIST = [
    "ánh", "bàn", "biển", "bình", "bút", "cam", "cây", "cầu", "chân",
    "chim", "chuông", "cửa", "đá", "đèn", "đất", "đêm", "đỏ", "đường",
    "gió", "giấy", "hạt", "hoa", "hồ", "làng", "lá", "lửa", "mây",
    "mưa", "nắng", "núi", "phố", "sách", "sáng", "sông", "sóng",
    "sương", "thuyền", "trăng", "trời", "vườn", "xanh", "yên", "ấm",
    "bền", "chậm", "đẹp", "êm", "gần", "khỏe", "mới", "nhẹ", "rộng",
    "sạch", "sâu", "thật", "tươi", "vui", "cao", "dài", "ngắn", "nhanh",
    "đọc", "đi", "đứng", "hát", "học", "làm", "mở", "nghe", "nhìn",
    "nói", "viết", "xây", "chọn", "giữ", "mang", "tạo", "tìm", "trồng",
    "bạn", "cha", "chị", "con", "em", "mẹ", "người", "ông", "thầy",
    "tôi", "anh", "cô", "trẻ", "nhà", "quê", "nước", "rừng", "đảo",
    "bếp", "chợ", "lớp", "trường", "viện", "ga", "bến", "vịnh", "đồi",
]

# Tên cũ được giữ để mã bên ngoài đang nhập WORD_LIST không bị hỏng.
WORD_LIST = VIETNAMESE_WORD_LIST

VIETNAMESE_STOPWORDS = frozenset({
    "ai", "bị", "bởi", "cả", "các", "cái", "cần", "chỉ", "cho", "chưa",
    "có", "còn", "cũng", "của", "đã", "đang", "để", "đến", "đều", "do",
    "đó", "được", "gì", "hay", "hơn", "khi", "không", "là", "lại", "lên",
    "mà", "mỗi", "một", "này", "nên", "những", "như", "nhưng", "ở", "phải",
    "ra", "rằng", "rất", "rồi", "sẽ", "thì", "trên", "trong", "từ", "và",
    "vẫn", "về", "vì", "với", "vừa", "theo", "tại", "nào", "nhiều", "ít",
})

def split_into_sentences(text):
    """Tách câu theo `.`, `?`, `!` khi sau dấu câu là khoảng trắng.

    Quy tắc cố định này phù hợp với chỉ dẫn tiếng Việt và tránh phụ thuộc vào
    bộ tách câu Punkt được huấn luyện cho tiếng Anh.
    """
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def count_words(text):
    """Đếm các chuỗi chữ cái/chữ số Unicode (không tính dấu gạch dưới)."""
    return len(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def count_sentences(text):
    """Đếm câu bằng cùng quy tắc với :func:`split_into_sentences`."""
    return len(split_into_sentences(text))


def count_stopwords(text):
    """Đếm hư từ tiếng Việt theo một danh sách công khai, cố định."""
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    num_stopwords = len([t for t in tokens if t.lower() in VIETNAMESE_STOPWORDS])
    return num_stopwords

def generate_keywords(num_keywords):
    """Sinh một số từ khóa tiếng Việt đơn âm tiết."""
    return random.sample(VIETNAMESE_WORD_LIST, k=num_keywords)
