#!/usr/bin/env python3
"""Generate model responses concurrently and verify them with IFBench.

The server only needs to implement the OpenAI-compatible ``/v1/models`` and
``/v1/chat/completions`` endpoints. Results are appended as requests finish,
so an interrupted run can continue with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

import evaluation_lib
from config import get_settings


_THREAD_LOCAL = threading.local()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"key", "prompt", "instruction_id_list", "kwargs"}
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"Dòng {line_number} trong {path} thiếu: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Không có mẫu nào trong {path}.")
    return rows


def _client(api_base: str, api_key: str, timeout: float,
            max_retries: int) -> OpenAI:
    """Return one reusable official OpenAI client per worker thread."""
    identity = (api_base, api_key, timeout, max_retries)
    if getattr(_THREAD_LOCAL, "identity", None) != identity:
        _THREAD_LOCAL.client = OpenAI(
            base_url=api_base.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        _THREAD_LOCAL.identity = identity
    return _THREAD_LOCAL.client


def discover_model(api_base: str, api_key: str, timeout: float,
                   max_retries: int) -> str:
    """Select the only served model, or require --model if several exist."""
    models = _client(api_base, api_key, timeout, max_retries).models.list().data
    model_ids = [model.id for model in models]
    if len(model_ids) == 1:
        return model_ids[0]
    if not model_ids:
        raise RuntimeError("Máy chủ không trả về model nào từ /models.")
    raise RuntimeError(
        "Máy chủ có nhiều model; hãy chọn bằng --model: " + ", ".join(model_ids))


def request_response(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    timeout: float,
    max_retries: int,
) -> str:
    """Request one chat completion with the official OpenAI client."""
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        request["seed"] = seed
    completion = _client(
        api_base, api_key, timeout, max_retries).chat.completions.create(**request)
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Model trả về nội dung rỗng.")
    return content


def verify_response(row: dict[str, Any], response: str) -> dict[str, Any]:
    """Run both strict and loose IFBench checks for a generated response."""
    def make_input() -> evaluation_lib.InputExample:
        # Copy kwargs because the upstream strict evaluator normalizes it in-place.
        return evaluation_lib.InputExample(
            key=row["key"],
            instruction_id_list=list(row["instruction_id_list"]),
            prompt=row["prompt"],
            kwargs=[dict(value) for value in row["kwargs"]],
        )

    prompt_to_response = {row["prompt"]: response}
    strict = evaluation_lib.test_instruction_following_strict(
        make_input(), prompt_to_response)
    loose = evaluation_lib.test_instruction_following_loose(
        make_input(), prompt_to_response)
    return {
        "strict_follow_all": strict.follow_all_instructions,
        "strict_follow_instruction_list": strict.follow_instruction_list,
        "loose_follow_all": loose.follow_all_instructions,
        "loose_follow_instruction_list": loose.follow_instruction_list,
    }


def _process_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        response = request_response(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            prompt=row["prompt"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        verification = verify_response(row, response)
        error = None
    except Exception as exc:  # Keep other workers running and make failures resumable.
        response = ""
        verification = verify_response(row, response)
        error = f"{type(exc).__name__}: {exc}"

    return {
        "key": row["key"],
        "prompt": row["prompt"],
        "response": response,
        "instruction_id_list": row["instruction_id_list"],
        "kwargs": row["kwargs"],
        "model": args.model,
        **verification,
        "error": error,
    }


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_final(path: Path, rows: list[dict[str, Any]],
                 order: dict[str, int]) -> None:
    rows.sort(key=lambda row: order[row["prompt"]])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_summary(rows: list[dict[str, Any]], *, model: str,
                  input_file: Path) -> dict[str, Any]:
    completed = [row for row in rows if not row["error"]]
    per_instruction: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "strict_correct": 0, "loose_correct": 0})
    for row in rows:
        for instruction_id, strict_ok, loose_ok in zip(
                row["instruction_id_list"],
                row["strict_follow_instruction_list"],
                row["loose_follow_instruction_list"]):
            values = per_instruction[instruction_id]
            values["total"] += 1
            values["strict_correct"] += int(strict_ok)
            values["loose_correct"] += int(loose_ok)

    total = len(rows)
    return {
        "model": model,
        "input_file": str(input_file),
        "total": total,
        "completed": len(completed),
        "errors": total - len(completed),
        "strict_correct": sum(row["strict_follow_all"] for row in rows),
        "strict_accuracy": (
            sum(row["strict_follow_all"] for row in rows) / total if total else 0),
        "loose_correct": sum(row["loose_follow_all"] for row in rows),
        "loose_accuracy": (
            sum(row["loose_follow_all"] for row in rows) / total if total else 0),
        "per_instruction": dict(sorted(per_instruction.items())),
    }


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Gọi model song song và chấm câu trả lời bằng IFBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path,
                        help="Mặc định: results/<model>-<input>-verified.jsonl")
    parser.add_argument("--summary-file", type=Path,
                        help="Mặc định: cùng tên output với đuôi .summary.json")
    parser.add_argument("--api-base", default=settings.api_base)
    parser.add_argument(
        "--api-key", default=settings.api_key or os.environ.get("OPENAI_API_KEY"),
        help="Nên đặt bằng API_KEY hoặc OPENAI_API_KEY thay vì ghi vào lệnh.")
    parser.add_argument("--model", default=settings.model or None,
                        help="Nếu bỏ trống, tự chọn khi server chỉ có một model.")
    parser.add_argument("--workers", type=int, default=settings.workers)
    parser.add_argument("--temperature", type=float, default=settings.temperature)
    parser.add_argument("--max-tokens", type=int, default=settings.max_tokens)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int,
                        help="Chỉ chạy N mẫu đầu tiên để kiểm tra nhanh.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Cần --api-key, API_KEY trong .env, hoặc OPENAI_API_KEY.")
    if args.workers < 1:
        parser.error("--workers phải lớn hơn 0.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0.")
    return args


def main() -> None:
    args = _parse_args()
    if not args.model:
        args.model = discover_model(
            args.api_base, args.api_key, args.request_timeout, args.max_retries)

    rows = _read_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[:args.limit]
    order = {row["prompt"]: index for index, row in enumerate(rows)}

    if args.output_file is None:
        safe_model = args.model.replace("/", "-")
        args.output_file = Path("results") / (
            f"{safe_model}-{args.input_file.stem}-verified.jsonl")
    if args.summary_file is None:
        args.summary_file = args.output_file.with_suffix(".summary.json")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(args.output_file) if args.resume else []
    existing = [row for row in existing if row["prompt"] in order and not row["error"]]
    completed_prompts = {row["prompt"] for row in existing}
    remaining = [row for row in rows if row["prompt"] not in completed_prompts]
    mode = "a" if args.resume else "w"

    print(f"Model: {args.model}")
    print(f"Input: {args.input_file} ({len(rows)} mẫu)")
    print(f"Workers: {args.workers}; còn lại: {len(remaining)}")
    results = list(existing)
    with args.output_file.open(mode, encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_row, row, args): row
                       for row in remaining}
            with tqdm(total=len(futures), desc="Sinh và chấm") as progress:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    progress.update(1)

    _write_final(args.output_file, results, order)
    summary = build_summary(results, model=args.model, input_file=args.input_file)
    args.summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "total", "completed", "errors", "strict_accuracy", "loose_accuracy")},
        ensure_ascii=False, indent=2))
    print(f"Kết quả: {args.output_file}")
    print(f"Tóm tắt: {args.summary_file}")
    if args.fail_on_error and summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
