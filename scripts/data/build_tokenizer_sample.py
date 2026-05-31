"""Build a plain-text sample for SentencePiece tokenizer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from common import iter_clean_texts, is_enabled, load_plan, source_name, weighted_quota


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-bytes", type=int, default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--dedup", choices=["none", "memory"], default="memory")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    plan = load_plan(args.config)
    defaults: Dict[str, Any] = dict(plan.get("defaults", {}))
    section = plan["tokenizer_sample"]
    sources = [s for s in section["sources"] if is_enabled(s)]
    target_bytes = int(args.target_bytes or section["target_bytes"])
    out_path = Path(args.out or section["output"])
    manifest_path = Path(args.manifest or f"{out_path}.manifest.json")
    quotas = weighted_quota(target_bytes, sources)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.dedup == "memory" else None
    manifest = {
        "config": args.config,
        "output": str(out_path),
        "target_bytes": target_bytes,
        "sources": [],
    }

    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for source in sources:
            name = source_name(source)
            quota = quotas[name]
            source_bytes = 0
            source_docs = 0
            print(f"[sample] {name}: quota={quota:,} bytes", flush=True)

            for text, meta in iter_clean_texts(source, defaults, seen_hashes=seen):
                encoded_len = len(text.encode("utf-8")) + 2
                if source_bytes >= quota:
                    break
                fout.write(text)
                fout.write("\n\n")
                source_bytes += encoded_len
                total_bytes += encoded_len
                source_docs += 1

                if args.max_docs_per_source and source_docs >= args.max_docs_per_source:
                    break
                if source_docs % 10000 == 0:
                    print(
                        f"  {name}: docs={source_docs:,} bytes={source_bytes:,}/{quota:,}",
                        flush=True,
                    )

            manifest["sources"].append(
                {
                    "name": name,
                    "dataset": source.get("dataset"),
                    "config": source.get("config"),
                    "split": source.get("split"),
                    "quota_bytes": quota,
                    "written_bytes": source_bytes,
                    "documents": source_docs,
                }
            )

    manifest["written_bytes"] = total_bytes
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote sample to {out_path}; bytes={total_bytes:,}", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
