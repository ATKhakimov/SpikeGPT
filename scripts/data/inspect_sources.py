"""Inspect configured Hugging Face data sources without building datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from common import collect_sources, extract_text, is_enabled, load_plan, normalize_text, source_name


def hf_metadata_kwargs(config: str | None, trust_remote_code: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if config:
        kwargs["config_name"] = config
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def inspect_hf_source(source: Mapping[str, Any], max_samples: int) -> Dict[str, Any]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    dataset = source["dataset"]
    config = source.get("config")
    split = source.get("split", "train")
    trust_remote_code = bool(source.get("trust_remote_code", False))
    result: Dict[str, Any] = {
        "name": source_name(source),
        "enabled": is_enabled(source),
        "kind": source.get("kind", "hf"),
        "dataset": dataset,
        "config": config,
        "split": split,
        "ok": False,
    }

    try:
        config_kwargs = hf_metadata_kwargs(None, trust_remote_code)
        configs = get_dataset_config_names(dataset, **config_kwargs)
        result["config_count"] = len(configs)
        result["config_seen"] = config in configs if config else None
        result["config_preview"] = configs[:10]
    except Exception as exc:
        result["config_error"] = repr(exc)

    try:
        split_kwargs = hf_metadata_kwargs(config, trust_remote_code)
        result["available_splits"] = get_dataset_split_names(dataset, **split_kwargs)
        result["split_seen"] = split in result["available_splits"]
    except Exception as exc:
        result["split_error"] = repr(exc)

    try:
        load_kwargs = {
            "split": split,
            "streaming": True,
            "trust_remote_code": trust_remote_code,
        }
        if config:
            load_kwargs["name"] = config
        stream = load_dataset(dataset, **load_kwargs)
        samples = []
        for i, row in enumerate(stream):
            if i >= max_samples:
                break
            text = extract_text(row, source, {"text_fields": ["text"]})
            samples.append(
                {
                    "row_index": i,
                    "columns": sorted(row.keys()),
                    "text_chars": len(normalize_text(text)) if text else 0,
                    "has_text": bool(text),
                }
            )
        result["samples"] = samples
        result["ok"] = True
    except Exception as exc:
        result["sample_error"] = repr(exc)

    return result


def write_report(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["tokenizer_sample", "pretrain_mix", "validation_splits", "sft_mix"],
    )
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Inspect sources with enabled: false as well.",
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--out", default="reports/data_source_inspection.jsonl")
    args = parser.parse_args()

    plan = load_plan(args.config)
    wanted = set(args.names or [])
    sources = collect_sources(plan, args.sections)
    if wanted:
        sources = [s for s in sources if source_name(s) in wanted]
    if not args.include_disabled:
        skipped = [source_name(s) for s in sources if not is_enabled(s)]
        sources = [s for s in sources if is_enabled(s)]
        for name in skipped:
            print(f"[skip disabled] {name}", flush=True)

    results = []
    for source in sources:
        if source.get("kind", "hf") != "hf":
            results.append(
                {
                    "name": source_name(source),
                    "kind": source.get("kind"),
                    "ok": False,
                    "error": "Only kind=hf is implemented in inspect_sources.py",
                }
            )
            continue
        print(f"[inspect] {source_name(source)} -> {source['dataset']}", flush=True)
        results.append(inspect_hf_source(source, args.max_samples))

    write_report(Path(args.out), results)
    ok_count = sum(1 for row in results if row.get("ok"))
    print(f"Wrote {len(results)} source reports to {args.out}; ok={ok_count}", flush=True)


if __name__ == "__main__":
    main()
