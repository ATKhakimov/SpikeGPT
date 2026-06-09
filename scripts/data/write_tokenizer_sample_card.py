"""Write a Hugging Face dataset card for the tokenizer sample artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


REASON_LABELS = {
    "rejected_exact_duplicate": "Exact duplicate",
    "rejected_low_alpha_ratio": "Low alphabetic ratio",
    "rejected_low_cyrillic_ratio": "Low Cyrillic ratio",
    "rejected_low_unique_word_fraction": "Low unique-word fraction",
    "rejected_near_duplicate": "Near duplicate",
    "rejected_repeated_lines": "Repeated lines",
    "rejected_spam_keyword": "Spam keyword",
    "rejected_too_long": "Too long",
    "rejected_too_many_digits": "Too many digits",
    "rejected_too_many_emails": "Too many emails",
    "rejected_too_many_phones": "Too many phones",
    "rejected_too_many_short_lines": "Too many short lines",
    "rejected_too_many_urls": "Too many URLs",
    "rejected_too_much_punctuation": "Too much punctuation",
    "rejected_too_short": "Too short",
}


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_mb(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} MB"


def source_rows(sources: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        "| Source | Dataset | Split | Documents/chunks | Written | Quota |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for source in sources:
        rows.append(
            "| {name} | `{dataset}` | `{split}` | {documents} | {written} | {quota} |".format(
                name=source["name"],
                dataset=source.get("dataset"),
                split=source.get("split"),
                documents=fmt_int(int(source.get("documents", 0))),
                written=fmt_mb(int(source.get("written_bytes", 0))),
                quota=fmt_mb(int(source.get("quota_bytes", 0))),
            )
        )
    return "\n".join(rows)


def totals(sources: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for source in sources:
        for key, value in source.get("filter_stats", {}).items():
            out[key] = out.get(key, 0) + int(value)
    return out


def reject_rows(stats: Mapping[str, int]) -> str:
    rows = [
        "| Reject reason | Count |",
        "|---|---:|",
    ]
    for key in sorted(k for k in stats if k.startswith("rejected_")):
        rows.append(f"| {REASON_LABELS.get(key, key)} (`{key}`) | {fmt_int(int(stats[key]))} |")
    return "\n".join(rows)


def per_source_reject_rows(sources: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        "| Source | Rows seen | Accepted chunks | Rejected chunks | Top reject reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for source in sources:
        stats = source.get("filter_stats", {})
        rejected = {k: int(v) for k, v in stats.items() if k.startswith("rejected_") and int(v)}
        rejected_total = sum(rejected.values())
        top = sorted(rejected.items(), key=lambda item: item[1], reverse=True)[:4]
        top_text = ", ".join(f"`{k}`={fmt_int(v)}" for k, v in top) or "-"
        rows.append(
            "| {name} | {rows_seen} | {accepted} | {rejected} | {top} |".format(
                name=source["name"],
                rows_seen=fmt_int(int(stats.get("rows_seen", 0))),
                accepted=fmt_int(int(stats.get("accepted_chunks", 0))),
                rejected=fmt_int(rejected_total),
                top=top_text,
            )
        )
    return "\n".join(rows)


def card_text(manifest: Mapping[str, Any], sample_filename: str, manifest_filename: str) -> str:
    sources = manifest["sources"]
    stats = totals(sources)
    accepted = int(stats.get("accepted_chunks", 0))
    rejected = sum(int(v) for k, v in stats.items() if k.startswith("rejected_"))
    written = int(manifest["written_bytes"])
    target = int(manifest["target_bytes"])
    return f"""---
license: other
language:
- ru
task_categories:
- text-generation
pretty_name: SpikeRuGPT tokenizer sample
size_categories:
- 100K<n<1M
tags:
- russian
- tokenizer-training
- sentencepiece
- spikerugpt
---

# SpikeRuGPT Tokenizer Sample

This dataset is a cleaned Russian text sample prepared for training the
`SpikeRuGPT` SentencePiece tokenizer. It is intended as a tokenizer-training
artifact, not as a final public pretraining corpus.

## Files

- `{sample_filename}`: cleaned plain-text sample.
- `{manifest_filename}`: exact source mix, byte quotas, and cleaning counters.

## Summary

- Target bytes: {fmt_int(target)}
- Written bytes: {fmt_int(written)} ({fmt_mb(written)})
- Accepted chunks: {fmt_int(accepted)}
- Rejected chunks: {fmt_int(rejected)}
- Exact deduplication: `{manifest.get("dedup")}`
- Near deduplication: `{manifest.get("near_dedup")}`
- Config file: `{manifest.get("config")}`

## Source Mix

{source_rows(sources)}

## Cleaning Pipeline

The sample was built with `scripts/data/build_tokenizer_sample.py` and the
shared cleaning functions in `scripts/data/common.py`.

Processing steps:

1. Extract source-specific text fields.
2. Normalize line endings, spaces, non-breaking spaces, and empty lines.
3. Remove lightweight HTML and Markdown markup.
4. Remove short boilerplate lines such as cookie/privacy/menu/login prompts.
5. Chunk long documents instead of dropping them wholesale.
6. Apply Russian/text-quality filters:
   - minimum length;
   - maximum chunk length;
   - minimum alphabetic ratio;
   - minimum Cyrillic ratio;
   - maximum digit ratio;
   - maximum punctuation ratio;
   - repeated-line filtering;
   - short-line filtering;
   - URL, email, and phone-count limits;
   - minimum unique-word fraction;
   - spam/SEO/adult/betting keyword filtering.
7. Deduplicate by exact normalized text hash.
8. Deduplicate near-duplicates with SimHash.

## Rejection Totals

{reject_rows(stats)}

## Per-Source Filtering

{per_source_reject_rows(sources)}

## Intended Use

Use this sample to train and compare Russian tokenizer candidates for a
small/medium SpikeGPT-style language model, especially a 32k SentencePiece BPE
tokenizer.

## Limitations and Release Notes

This artifact mixes public Hugging Face datasets with source-specific licenses
and terms. Before publishing derivative models or broader corpora, review the
license and provenance of each upstream dataset. Keep this repository private
until that review is complete.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/tokenizer_sample/spikerugpt_tokenizer_sample.manifest.json",
    )
    parser.add_argument("--sample-filename", default="spikerugpt_tokenizer_sample.txt")
    parser.add_argument(
        "--out",
        default="data/tokenizer_sample/hf_dataset/README.md",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        card_text(manifest, args.sample_filename, manifest_path.name),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
