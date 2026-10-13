#!/usr/bin/env python3
"""Generate deterministic Vietnamese IFBench train/validation JSONL files.

The default configuration needs no input file.  It combines a small bundled
bank of natural Vietnamese requests with the 70 verifier-safe instructions in
``VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS``.  A custom TXT or JSONL prompt bank
can be supplied with ``--base-prompts``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ifbench import instructions_registry
from ifbench.instructions_util import VIETNAMESE_WORD_LIST


DEFAULT_BASE_PROMPTS = (
    "Giải thích vì sao cây xanh cần ánh sáng để phát triển.",
    "Nêu ba cách tiết kiệm điện trong gia đình.",
    "So sánh ưu điểm của xe buýt và xe đạp khi đi lại trong thành phố.",
    "Viết một đoạn giới thiệu ngắn về văn hóa uống trà ở Việt Nam.",
    "Đề xuất kế hoạch học từ vựng trong bốn tuần cho người mới bắt đầu.",
    "Giải thích hiện tượng cầu vồng bằng ngôn ngữ dễ hiểu.",
    "Trình bày lợi ích của việc đọc sách mỗi ngày.",
    "Gợi ý một thực đơn lành mạnh cho bữa sáng.",
    "Mô tả khung cảnh một khu chợ quê vào buổi sớm.",
    "Nêu các bước chuẩn bị cho một buổi thuyết trình hiệu quả.",
    "Giải thích sự khác nhau giữa thời tiết và khí hậu.",
    "Đề xuất cách tổ chức góc học tập gọn gàng và yên tĩnh.",
    "Viết lời khuyên giúp một người hình thành thói quen tập thể dục.",
    "Tóm tắt vai trò của rừng ngập mặn đối với môi trường ven biển.",
    "Lập kế hoạch cho chuyến tham quan bảo tàng kéo dài nửa ngày.",
    "Mô tả cách nấu một món canh rau đơn giản.",
    "Giải thích vì sao cần kiểm chứng nguồn tin trước khi chia sẻ.",
    "Nêu một số biện pháp giảm rác thải nhựa ở trường học.",
    "Viết đoạn văn khuyến khích mọi người sử dụng thư viện công cộng.",
    "Đề xuất hoạt động gắn kết cho một nhóm làm việc từ xa.",
    "Giải thích cách lập ngân sách cá nhân cơ bản cho sinh viên.",
    "Mô tả vòng tuần hoàn của nước trong tự nhiên.",
    "So sánh việc học trực tiếp và học trực tuyến.",
    "Gợi ý cách chăm sóc một cây non trong mùa khô.",
    "Viết một câu chuyện ngắn về tình bạn và lòng trung thực.",
    "Nêu các nguyên tắc ứng xử lịch sự trong cuộc họp.",
    "Giải thích cách hoạt động của pin mặt trời ở mức phổ thông.",
    "Đề xuất lịch trình khám phá Huế trong một ngày.",
    "Trình bày ý nghĩa của việc bảo tồn các làng nghề truyền thống.",
    "Gợi ý phương pháp ghi chép giúp nhớ bài lâu hơn.",
    "Mô tả một buổi chiều mưa ở Hà Nội.",
    "Giải thích lợi ích của việc ngủ đủ giấc.",
    "Nêu cách xử lý khi hai thành viên trong nhóm bất đồng ý kiến.",
    "Đề xuất ba hoạt động khoa học đơn giản cho trẻ em tại nhà.",
    "Viết phần giới thiệu cho một câu lạc bộ nhiếp ảnh.",
    "Giải thích vì sao đa dạng sinh học quan trọng.",
    "Lập danh sách việc cần làm trước một chuyến đi xa.",
    "Mô tả cách pha một cốc nước chanh ít đường.",
    "Nêu lợi ích và hạn chế của làm việc tại nhà.",
    "Đề xuất cách giúp khu phố có thêm không gian xanh.",
)

# Các nhiệm vụ này đã tự chứa nội dung cần trả lời. Thêm một câu hỏi nền sẽ
# tạo hai yêu cầu cạnh tranh nhau, nên chỉ dùng lời dẫn trung tính.
SELF_CONTAINED_IDS = frozenset({
    "custom:multiples",
    "custom:reverse_newline",
    "custom:sentence_alphabet",
    "custom:csv_city",
    "custom:csv_special_character",
    "custom:csv_quotes",
    "detectable_format:constrained_response",
    "format:options",
    "repeat:repeat_simple",
})

FRAMES = (
    "Hãy thực hiện yêu cầu sau.",
    "Vui lòng làm đúng chỉ dẫn dưới đây.",
    "Thực hiện chính xác nhiệm vụ sau.",
    "Hãy trả lời theo yêu cầu được nêu bên dưới.",
)


def _load_prompts(path: Path | None, prompt_field: str) -> list[str]:
    if path is None:
        return list(DEFAULT_BASE_PROMPTS)
    if not path.is_file():
        raise ValueError(f"Không tìm thấy tệp câu hỏi nền: {path}")

    prompts: list[str] = []
    if path.suffix.casefold() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if prompt_field not in value or not isinstance(value[prompt_field], str):
                raise ValueError(
                    f"Dòng {line_number} không có trường chuỗi {prompt_field!r}.")
            prompts.append(value[prompt_field].strip())
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError("Tệp câu hỏi nền không có câu hỏi hợp lệ.")
    if any(len(prompt.split()) < 2 for prompt in prompts):
        raise ValueError("Mỗi câu hỏi nền phải có ít nhất hai từ.")
    return prompts


def _sample_words(rng: random.Random, count: int) -> list[str]:
    return rng.sample(VIETNAMESE_WORD_LIST, count)


def _build_kwargs(instruction_id: str, base_prompt: str,
                  rng: random.Random) -> dict[str, Any]:
    """Use moderate, learnable values instead of several very large defaults."""
    words = _sample_words(rng, 6)
    factories: dict[str, dict[str, Any]] = {
        "keywords:existence": {"keywords": words[:2]},
        "keywords:frequency": {
            "keyword": words[0], "frequency": rng.randint(1, 3),
            "relation": "at least",
        },
        "keywords:forbidden_words": {"forbidden_words": words[:2]},
        "keywords:letter_frequency": {
            "letter": rng.choice("aemnothiu"),
            "let_frequency": rng.randint(2, 5),
            "let_relation": "at least",
        },
        "length_constraints:number_sentences": {
            "num_sentences": rng.randint(2, 6), "relation": "at least",
        },
        "length_constraints:number_paragraphs": {
            "num_paragraphs": rng.randint(2, 4),
        },
        "length_constraints:number_words": {
            "num_words": rng.randint(40, 90), "relation": "at least",
        },
        "length_constraints:nth_paragraph_first_word": {
            "num_paragraphs": 3, "nth_paragraph": rng.randint(1, 3),
            "first_word": words[0],
        },
        "detectable_content:number_placeholders": {
            "num_placeholders": rng.randint(1, 3),
        },
        "detectable_content:postscript": {
            "postscript_marker": rng.choice(("Tái bút:", "Ghi chú cuối:", "P.S.:")),
        },
        "detectable_format:number_bullet_lists": {"num_bullets": rng.randint(2, 5)},
        "detectable_format:number_highlighted_sections": {
            "num_highlights": rng.randint(1, 3),
        },
        "detectable_format:multiple_sections": {
            "section_spliter": "Phần", "num_sections": rng.randint(2, 4),
        },
        "combination:repeat_prompt": {"prompt_to_repeat": base_prompt},
        "startend:end_checker": {
            "end_phrase": rng.choice(("Hết.", "Xin cảm ơn.", "Kết thúc.")),
        },
        "change_case:capital_word_frequency": {
            "capital_frequency": rng.randint(2, 5), "capital_relation": "at least",
        },
        "count:word_count_range": {
            "min_words": (minimum := rng.randint(40, 80)),
            "max_words": minimum + rng.randint(10, 25),
        },
        "count:unique_word_count": {"N": rng.randint(25, 50)},
        "count:conjunctions": {"small_n": rng.randint(2, 4)},
        "count:person_names": {"N": rng.randint(2, 5)},
        "ratio:overlap": {
            "reference_text": base_prompt, "percentage": rng.choice((25, 30, 35)),
        },
        "count:numbers": {"N": rng.randint(2, 5)},
        "format:options": {"options": "a), b), c), d)"},
        "words:repeats": {"small_n": rng.randint(3, 5)},
        "sentence:keyword": {"word": words[0], "N": rng.randint(2, 4)},
        "sentence:increment": {"small_n": rng.randint(2, 4)},
        "format:list": {"sep": rng.choice(("MỤC", "PHẦN", "Ý"))},
        "words:keywords_specific_position": {
            "keyword": words[0], "n": rng.randint(2, 4), "m": rng.randint(3, 8),
        },
        "words:words_position": {"keyword": words[0]},
        "repeat:repeat_change": {"prompt_to_repeat": base_prompt},
        "repeat:repeat_span": {
            "prompt_to_repeat": base_prompt,
            "n_start": (start := rng.randint(0, len(base_prompt.split()) - 2)),
            "n_end": rng.randint(start + 1, len(base_prompt.split()) - 1),
        },
    }
    return factories.get(instruction_id, {})


def _make_example(instruction_id: str, base_prompt: str, key: int,
                  rng: random.Random) -> dict[str, Any]:
    checker_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    checker = checker_cls(instruction_id)
    description = checker.build_description(
        **_build_kwargs(instruction_id, base_prompt, rng))
    kwargs = checker.get_instruction_args() or {}

    if instruction_id in SELF_CONTAINED_IDS:
        prompt = f"{rng.choice(FRAMES)}\n\n{description}"
    elif instruction_id == "repeat:repeat_change":
        prompt = description
    elif instruction_id == "ratio:overlap":
        prompt = f"{rng.choice(FRAMES)}\n\n{description}"
    elif instruction_id == "repeat:repeat_span":
        prompt = f"Văn bản nguồn: {base_prompt}\n\n{description}"
    else:
        prompt = f"{base_prompt}\n\nYêu cầu bắt buộc:\n{description}"

    return {
        "key": key,
        "instruction_id_list": [instruction_id],
        "prompt": prompt,
        "kwargs": [kwargs],
        "messages": [{"role": "user", "content": prompt}],
        "verifier": {
            "type": "ifbench",
            "instruction_id_list": [instruction_id],
            "kwargs": [kwargs],
        },
    }


def _generate_examples(count: int, prompts: list[str], seed: int,
                       key_offset: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    instruction_ids = sorted(
        instructions_registry.VIETNAMESE_RECOMMENDED_INSTRUCTION_IDS)
    rng.shuffle(instruction_ids)
    results: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    attempts = 0
    while len(results) < count:
        if attempts > max(10_000, count * 100):
            raise RuntimeError("Không thể tạo đủ mẫu không trùng lặp.")
        instruction_id = instruction_ids[attempts % len(instruction_ids)]
        base_prompt = prompts[(attempts // len(instruction_ids)) % len(prompts)]
        example = _make_example(
            instruction_id, base_prompt, key_offset + len(results), rng)
        attempts += 1
        if example["prompt"] in seen_prompts:
            continue
        seen_prompts.add(example["prompt"])
        results.append(example)
    rng.shuffle(results)
    return results


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def generate_dataset(*, output_dir: Path, prompts: list[str], train_size: int,
                     val_size: int, seed: int) -> dict[str, Any]:
    if train_size < 1 or val_size < 1:
        raise ValueError("train-size và val-size phải lớn hơn 0.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sinh một tập duy nhất không trùng rồi mới chia để không rò rỉ câu hỏi qua
    # train/validation, kể cả với các nhiệm vụ có mô tả cố định.
    examples = _generate_examples(train_size + val_size, prompts, seed, 0)
    train = examples[:train_size]
    val = examples[train_size:]

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    counts = Counter(
        instruction_id
        for row in train + val
        for instruction_id in row["instruction_id_list"])
    manifest = {
        "language": "vi",
        "seed": seed,
        "train_size": len(train),
        "val_size": len(val),
        "base_prompt_count": len(prompts),
        "instruction_ids": sorted(counts),
        "instruction_counts": dict(sorted(counts.items())),
        "experimental_ids_included": False,
        "format": "IFBench fields plus messages and verifier metadata",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo tập train/validation tiếng Việt cho RL với IFBench.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/vi_rl"))
    parser.add_argument("--base-prompts", type=Path,
                        help="Tệp TXT (mỗi dòng một câu) hoặc JSONL; không bắt buộc.")
    parser.add_argument("--prompt-field", default="prompt",
                        help="Tên trường câu hỏi khi --base-prompts là JSONL.")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompts = _load_prompts(args.base_prompts, args.prompt_field)
    manifest = generate_dataset(
        output_dir=args.output_dir,
        prompts=prompts,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(
        f"Đã tạo {manifest['train_size']} mẫu train và "
        f"{manifest['val_size']} mẫu validation trong {args.output_dir}.")
    print(f"Số tác vụ an toàn được dùng: {len(manifest['instruction_ids'])}.")


if __name__ == "__main__":
    main()
