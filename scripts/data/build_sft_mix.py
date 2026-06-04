"""Build a normalized SFT JSONL mix from configured instruction datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from common import (
    document_hash,
    first_field_text,
    is_enabled,
    load_hf_stream,
    load_plan,
    normalize_text,
    source_name,
    weighted_quota,
)


def score_ok(row: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    min_score = source.get("min_score")
    if min_score is None:
        return True
    for field in source.get("score_fields", ["score"]):
        value = row.get(field)
        if value is None:
            continue
        try:
            return float(value) >= float(min_score)
        except (TypeError, ValueError):
            return False
    return False


def language_ok(row: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    values = source.get("language_values")
    field = source.get("language_field")
    if not values or not field:
        return True
    return str(row.get(field, "")).lower() in {str(v).lower() for v in values}


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user", "prompter"}:
        return "user"
    if role in {"assistant", "bot", "gpt", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def build_from_messages(messages: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        return None
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = normalize_role(str(message.get("role", "")))
        content = normalize_text(str(message.get("content", "")))
        if role and content:
            normalized.append({"role": role, "content": content})
    user_count = sum(1 for message in normalized if message["role"] == "user")
    assistant_count = sum(1 for message in normalized if message["role"] == "assistant")
    if user_count < 1 or assistant_count < 1:
        return None
    text_for_hash = "\n".join(m["role"] + ":" + m["content"] for m in normalized)
    return {"messages": normalized, "hash": document_hash(text_for_hash)}


def build_from_dialogue(dialogue: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(dialogue, list) or len(dialogue) < 2:
        return None
    normalized = []
    for index, item in enumerate(dialogue):
        content = normalize_text(str(item))
        if not content:
            continue
        role = "user" if index % 2 == 0 else "assistant"
        normalized.append({"role": role, "content": content})
    if len(normalized) < 2 or normalized[0]["role"] != "user":
        return None
    if not any(message["role"] == "assistant" for message in normalized):
        return None
    text_for_hash = "\n".join(m["role"] + ":" + m["content"] for m in normalized)
    return {"messages": normalized, "hash": document_hash(text_for_hash)}


def build_messages(row: Mapping[str, Any], source: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not score_ok(row, source) or not language_ok(row, source):
        return None

    example = build_from_messages(row.get("messages"))
    if example:
        return example

    dialogue_field = source.get("dialogue_field")
    if dialogue_field:
        example = build_from_dialogue(row.get(dialogue_field))
        if example:
            return example

    instruction = first_field_text(row, source.get("instruction_fields", []))
    extra_input = first_field_text(row, source.get("input_fields", []))
    output = first_field_text(row, source.get("output_fields", []))

    if not instruction or not output:
        return None

    user_content = normalize_text(instruction)
    if extra_input:
        user_content = normalize_text(user_content + "\n\n" + extra_input)
    assistant_content = normalize_text(output)

    if len(user_content) < 5 or len(assistant_content) < 5:
        return None

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    text_for_hash = user_content + "\n\n" + assistant_content
    return {"messages": messages, "hash": document_hash(text_for_hash)}


def length_ok(example: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    min_user_chars = int(source.get("min_user_chars", 5))
    max_user_chars = source.get("max_user_chars")
    min_assistant_chars = int(source.get("min_assistant_chars", 5))
    max_assistant_chars = source.get("max_assistant_chars")

    users = [m["content"] for m in example["messages"] if m["role"] == "user"]
    assistants = [m["content"] for m in example["messages"] if m["role"] == "assistant"]
    if not users or not assistants:
        return False
    if any(len(text) < min_user_chars for text in users):
        return False
    if any(len(text) < min_assistant_chars for text in assistants):
        return False
    if max_user_chars is not None and any(len(text) > int(max_user_chars) for text in users):
        return False
    if max_assistant_chars is not None and any(len(text) > int(max_assistant_chars) for text in assistants):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-examples", type=int, default=None)
    parser.add_argument("--dedup", choices=["none", "memory"], default="memory")
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    args = parser.parse_args()

    plan = load_plan(args.config)
    section = plan["sft_mix"]
    sources = [s for s in section["sources"] if is_enabled(s)]
    target_examples = int(args.target_examples or section["target_examples"])
    out_path = Path(args.out or section["output"])
    manifest_path = Path(args.manifest or f"{out_path}.manifest.json")
    quotas = weighted_quota(target_examples, sources)
    seen = set() if args.dedup == "memory" else None
    if seen is not None:
        for exclude_path in args.exclude_jsonl:
            with open(exclude_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    value = record.get("hash")
                    if value:
                        seen.add(value)
    manifest = {
        "config": args.config,
        "output": str(out_path),
        "target_examples": target_examples,
        "sources": [],
    }

    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for source in sources:
            name = source_name(source)
            quota = quotas[name]
            written = 0
            scanned = 0
            print(f"[sft] {name}: quota={quota:,} examples", flush=True)

            for row in load_hf_stream(source):
                scanned += 1
                if args.max_rows_per_source and scanned > args.max_rows_per_source:
                    break
                example = build_messages(row, source)
                if not example:
                    continue
                if not length_ok(example, source):
                    continue
                if seen is not None:
                    if example["hash"] in seen:
                        continue
                    seen.add(example["hash"])
                record = {
                    "source": name,
                    "dataset": source.get("dataset"),
                    "messages": example["messages"],
                    "hash": example["hash"],
                }
                fout.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                fout.write("\n")
                written += 1
                total += 1
                if written >= quota:
                    break

            manifest["sources"].append(
                {
                    "name": name,
                    "dataset": source.get("dataset"),
                    "split": source.get("split"),
                    "quota_examples": quota,
                    "written_examples": written,
                    "scanned_rows": scanned,
                }
            )

    manifest["written_examples"] = total
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {total:,} SFT examples to {out_path}", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
