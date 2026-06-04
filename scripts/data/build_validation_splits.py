"""Build fixed text validation splits from configured sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from common import iter_clean_texts, is_enabled, load_plan, source_name, weighted_quota


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--output-dir", default="data/validation_text")
    parser.add_argument("--chars-per-token-estimate", type=float, default=3.2)
    parser.add_argument("--dedup", choices=["none", "memory"], default="memory")
    parser.add_argument("--near-dedup", choices=["none", "simhash"], default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    args = parser.parse_args()

    plan = load_plan(args.config)
    defaults: Dict[str, Any] = dict(plan.get("defaults", {}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set() if args.dedup == "memory" else None
    near_dedup = args.near_dedup or defaults.get("near_dedup", "none")
    seen_simhashes = {} if near_dedup == "simhash" else None

    for split_cfg in plan.get("validation_splits", []):
        split_name = split_cfg["name"]
        target_tokens = int(split_cfg["target_tokens"])
        target_bytes = int(target_tokens * args.chars_per_token_estimate)
        sources = [s for s in split_cfg["sources"] if is_enabled(s)]
        quotas = weighted_quota(target_bytes, sources)
        out_path = output_dir / f"{split_name}.jsonl"
        manifest_path = output_dir / f"{split_name}.manifest.json"

        split_docs = 0
        split_bytes = 0
        manifest = {
            "name": split_name,
            "target_tokens": target_tokens,
            "target_bytes_estimate": target_bytes,
            "dedup": args.dedup,
            "near_dedup": near_dedup,
            "sources": [],
        }

        with open(out_path, "w", encoding="utf-8") as fout:
            for source in sources:
                name = source_name(source)
                quota = quotas[name]
                source_docs = 0
                source_bytes = 0
                filter_stats: Dict[str, int] = {}
                print(f"[validation:{split_name}] {name}: quota={quota:,} bytes", flush=True)

                for text, meta in iter_clean_texts(
                    source,
                    defaults,
                    seen_hashes=seen,
                    seen_simhashes=seen_simhashes,
                    stats=filter_stats,
                ):
                    if source_bytes >= quota:
                        break
                    row = dict(meta)
                    row["validation_split"] = split_name
                    row["text"] = text
                    fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    fout.write("\n")
                    encoded_len = len(text.encode("utf-8"))
                    source_bytes += encoded_len
                    split_bytes += encoded_len
                    source_docs += 1
                    split_docs += 1

                    if args.max_docs_per_source and source_docs >= args.max_docs_per_source:
                        break

                manifest["sources"].append(
                    {
                        "name": name,
                        "dataset": source.get("dataset"),
                        "config": source.get("config"),
                        "split": source.get("split"),
                        "quota_bytes": quota,
                        "written_bytes": source_bytes,
                        "documents": source_docs,
                        "filter_stats": filter_stats,
                    }
                )

        manifest["written_bytes"] = split_bytes
        manifest["documents"] = split_docs
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {out_path}; docs={split_docs:,} bytes={split_bytes:,}", flush=True)


if __name__ == "__main__":
    main()
