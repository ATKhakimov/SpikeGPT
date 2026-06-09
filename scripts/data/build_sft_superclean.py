"""Build a stricter one-turn SFT dataset from normalized SFT JSONL files."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import cyrillic_ratio, document_hash, normalize_text


SERVICE_FIELD_RE = re.compile(r"(?i)(['\"](?:role|content)['\"]\s*:|role['\"]?\s*:\s*['\"]?(?:user|assistant|system))")
CODE_RE = re.compile(
    r"(?i)\b("
    r"python|javascript|typescript|html|css|sql|"
    r"php|file_get_contents|file_put_contents|shutdown_function|"
    r"phpstorm|mysql|wordpress|linux|root|csv|xls|api|pyrogram|"
    r"heroku|pythonanywhere|mysqldump|telegram client|"
    r"def |class |import |return |function|console\.log|rfind|"
    r"botype|offunctions|parning|traceback|localhost"
    r")\b"
)
BAD_PHRASE_RE = re.compile(
    r"(?i)("
    r"я не могу помочь|не могу вам помочь|as an ai|language model|"
    r"генераци[яю] кода|экспериментальн[а-я]+ код|"
    r"openai|chatgpt|gpt-4|role':|\"role\"|content':|\"content\""
    r")"
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def parse_dict_like_text(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if not (text.startswith("{") and "content" in text and "role" in text):
        return text
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
        return normalize_text(parsed["content"])
    return text


def normalize_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = str(message.get("role") or "").strip().lower()
    if role in {"human", "prompter"}:
        role = "user"
    if role in {"bot", "gpt", "model"}:
        role = "assistant"
    if role not in {"user", "assistant"}:
        return None
    content = parse_dict_like_text(str(message.get("content") or ""))
    content = normalize_text(content)
    if not content:
        return None
    return {"role": role, "content": content}


def first_user_assistant_pair(messages: list[Any]) -> tuple[str, str] | None:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = normalize_message(message)
        if item:
            normalized.append(item)
    for idx, message in enumerate(normalized[:-1]):
        if message["role"] == "user" and normalized[idx + 1]["role"] == "assistant":
            return message["content"], normalized[idx + 1]["content"]
    return None


def repeated_word_fraction(text: str) -> float:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if len(words) < 12:
        return 0.0
    counts = Counter(words)
    return max(counts.values()) / len(words)


def reject_reason(user: str, assistant: str) -> str | None:
    joined = f"{user}\n{assistant}"
    if len(user) < 8:
        return "user_too_short"
    if len(user) > 280:
        return "user_too_long"
    if len(assistant) < 45:
        return "assistant_too_short"
    if len(assistant) > 1100:
        return "assistant_too_long"
    if cyrillic_ratio(user) < 0.45:
        return "low_user_cyrillic"
    if cyrillic_ratio(assistant) < 0.55:
        return "low_assistant_cyrillic"
    if cyrillic_ratio(joined) < 0.58:
        return "low_cyrillic"
    if SERVICE_FIELD_RE.search(joined):
        return "service_role_content_artifact"
    if BAD_PHRASE_RE.search(joined):
        return "bad_phrase"
    if CODE_RE.search(joined):
        return "code_or_garbage_token"
    if joined.count("`") > 1 or "```" in joined:
        return "markdown_code"
    if joined.count("{") + joined.count("}") + joined.count("[") + joined.count("]") > 6:
        return "too_many_brackets"
    if URL_RE.search(joined) or EMAIL_RE.search(joined):
        return "url_or_email"
    if assistant.count("\n") > 12:
        return "too_many_answer_lines"
    if repeated_word_fraction(assistant) > 0.18:
        return "word_repetition"
    if assistant.count("?") > 4:
        return "too_many_questions"
    return None


def quality_score(user: str, assistant: str) -> float:
    answer_len = len(assistant)
    target_len_score = 1.0 - min(abs(answer_len - 420) / 700, 1.0)
    cyr_score = min(cyrillic_ratio(user + "\n" + assistant) / 0.9, 1.0)
    repetition_penalty = repeated_word_fraction(assistant)
    line_penalty = min(assistant.count("\n") / 16, 0.4)
    return target_len_score * 0.45 + cyr_score * 0.45 - repetition_penalty * 0.6 - line_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=["data/sft/spikerugpt_sft_clean_final.jsonl"])
    parser.add_argument("--out", default="data/sft/spikerugpt_sft_superclean_v2.jsonl")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-examples", type=int, default=45000)
    parser.add_argument("--max-per-source", type=int, default=24000)
    parser.add_argument("--sample-report", default="ARTICLE/sft_v2_superclean/dataset_samples.md")
    args = parser.parse_args()

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    counters = Counter()
    reject_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_by_source = Counter()

    for input_path in args.input:
        with Path(input_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                counters["rows"] += 1
                row = json.loads(line)
                source = str(row.get("source") or row.get("dataset") or "unknown")
                pair = first_user_assistant_pair(row.get("messages") or [])
                if not pair:
                    counters["reject_no_pair"] += 1
                    reject_by_source[source]["no_pair"] += 1
                    continue
                user, assistant = pair
                reason = reject_reason(user, assistant)
                if reason:
                    counters[f"reject_{reason}"] += 1
                    reject_by_source[source][reason] += 1
                    continue
                row_hash = document_hash(user + "\n\n" + assistant)
                if row_hash in seen:
                    counters["reject_duplicate"] += 1
                    reject_by_source[source]["duplicate"] += 1
                    continue
                seen.add(row_hash)
                candidates.append(
                    {
                        "source": source,
                        "dataset": row.get("dataset"),
                        "messages": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant},
                        ],
                        "hash": row_hash,
                        "quality_score": quality_score(user, assistant),
                    }
                )
                accepted_by_source[source] += 1

    candidates.sort(key=lambda item: (item["quality_score"], len(item["messages"][1]["content"])), reverse=True)
    selected = []
    selected_by_source = Counter()
    for item in candidates:
        source = item["source"]
        if selected_by_source[source] >= args.max_per_source:
            continue
        selected.append(item)
        selected_by_source[source] += 1
        if len(selected) >= args.target_examples:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in selected:
            clean_item = dict(item)
            clean_item.pop("quality_score", None)
            f.write(json.dumps(clean_item, ensure_ascii=False, sort_keys=True) + "\n")

    assistant_lengths = [len(item["messages"][1]["content"]) for item in selected]
    user_lengths = [len(item["messages"][0]["content"]) for item in selected]

    def percentile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        values = sorted(values)
        idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
        return values[idx]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": args.input,
        "output": str(out_path),
        "target_examples": args.target_examples,
        "written_examples": len(selected),
        "policy": {
            "format": "single user-assistant turn only",
            "repair": "dict-like {'content': ..., 'role': ...} strings are parsed and replaced by content",
            "max_user_chars": 280,
            "assistant_chars": [45, 1100],
            "min_cyrillic_ratio": 0.58,
            "rejects": [
                "role/content artifacts",
                "code-like fragments",
                "markdown code/backticks",
                "urls/emails",
                "refusal boilerplate",
                "high repetition",
                "very long multiline answers",
            ],
        },
        "counters": dict(counters),
        "candidates_after_filter": len(candidates),
        "accepted_by_source_before_cap": dict(accepted_by_source),
        "selected_by_source": dict(selected_by_source),
        "reject_by_source": {source: dict(counts) for source, counts in reject_by_source.items()},
        "length_chars": {
            "user_median": percentile(user_lengths, 0.50),
            "user_p95": percentile(user_lengths, 0.95),
            "assistant_median": percentile(assistant_lengths, 0.50),
            "assistant_p95": percentile(assistant_lengths, 0.95),
            "assistant_max": max(assistant_lengths) if assistant_lengths else None,
        },
    }
    manifest_path = Path(args.manifest or f"{out_path}.manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = Path(args.sample_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SFT v2 superclean dataset samples",
        "",
        f"Output: `{out_path}`",
        f"Written examples: {len(selected):,}",
        "",
        "## Composition",
        "",
    ]
    for source, count in selected_by_source.most_common():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(["", "## Samples", ""])
    for idx, item in enumerate(selected[:12], 1):
        user = item["messages"][0]["content"]
        assistant = item["messages"][1]["content"]
        lines.extend(
            [
                f"### {idx}. {item['source']}",
                "",
                "User:",
                "",
                "```text",
                user[:700],
                "```",
                "",
                "Assistant:",
                "",
                "```text",
                assistant[:1000],
                "```",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(selected):,} examples to {out_path}", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)
    print(f"Wrote samples to {report_path}", flush=True)


if __name__ == "__main__":
    main()
