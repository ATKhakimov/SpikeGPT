"""Tokenize configured pretraining sources into local binary shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from common import iter_clean_texts, is_enabled, load_plan, source_name, weighted_quota


class TokenizerAdapter:
    def __init__(self, kind: str, path: str):
        self.kind = kind
        self.path = path
        if kind == "hf":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.eos_id = self.tokenizer.eos_token_id
            if self.eos_id is None:
                raise ValueError("HF tokenizer must define eos_token_id")
            self.vocab_size = len(self.tokenizer)
        elif kind == "sentencepiece":
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor(model_file=path)
            self.eos_id = int(self.tokenizer.eos_id())
            if self.eos_id < 0:
                raise ValueError("SentencePiece tokenizer must define eos_id")
            self.vocab_size = int(self.tokenizer.vocab_size())
        else:
            raise ValueError(f"Unknown tokenizer kind: {kind}")

    def encode(self, text: str) -> List[int]:
        if self.kind == "hf":
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        return list(self.tokenizer.encode(text, out_type=int))


class ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, dtype: np.dtype, tokens_per_shard: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.dtype = dtype
        self.tokens_per_shard = tokens_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.shard_tokens = 0
        self.total_tokens = 0
        self.current = None
        self.manifest_rows: List[Dict[str, Any]] = []
        self._open_next()

    def _path(self) -> Path:
        return self.output_dir / f"{self.prefix}-{self.shard_index:05d}.bin"

    def _open_next(self) -> None:
        if self.current is not None:
            self.current.close()
        self.shard_tokens = 0
        self.current_path = self._path()
        self.current = open(self.current_path, "wb")

    def write(self, ids: List[int]) -> None:
        offset = 0
        while offset < len(ids):
            room = self.tokens_per_shard - self.shard_tokens
            chunk = ids[offset : offset + room]
            np.asarray(chunk, dtype=self.dtype).tofile(self.current)
            offset += len(chunk)
            self.shard_tokens += len(chunk)
            self.total_tokens += len(chunk)
            if self.shard_tokens >= self.tokens_per_shard:
                self._finish_current()
                self.shard_index += 1
                self._open_next()

    def _finish_current(self) -> None:
        self.current.flush()
        self.manifest_rows.append(
            {
                "path": str(self.current_path),
                "dtype": str(np.dtype(self.dtype)),
                "tokens": self.shard_tokens,
            }
        )

    def close(self) -> List[Dict[str, Any]]:
        if self.current is not None:
            if self.shard_tokens > 0:
                self._finish_current()
            else:
                self.current.close()
                self.current_path.unlink(missing_ok=True)
                self.current = None
                return self.manifest_rows
            self.current.close()
            self.current = None
        return self.manifest_rows


def choose_dtype(name: str, vocab_size: int) -> np.dtype:
    if name == "auto":
        return np.dtype("uint16" if vocab_size <= 65535 else "uint32")
    return np.dtype(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--tokenizer-kind", choices=["hf", "sentencepiece"], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="data/tokenized/pretrain")
    parser.add_argument("--prefix", default="spikerugpt-pretrain")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tokens-per-shard", type=int, default=100_000_000)
    parser.add_argument("--dtype", choices=["auto", "uint16", "uint32"], default="auto")
    parser.add_argument("--dedup", choices=["none", "memory"], default="memory")
    parser.add_argument("--near-dedup", choices=["none", "simhash"], default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--progress-docs", type=int, default=1000)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    plan = load_plan(args.config)
    defaults: Dict[str, Any] = dict(plan.get("defaults", {}))
    section = plan["pretrain_mix"]
    sources = [s for s in section["sources"] if is_enabled(s)]
    max_tokens = int(args.max_tokens or section["target_tokens"])

    tokenizer = TokenizerAdapter(args.tokenizer_kind, args.tokenizer)
    dtype = choose_dtype(args.dtype, tokenizer.vocab_size)
    if dtype == np.dtype("uint16") and tokenizer.vocab_size > 65535:
        raise ValueError("uint16 cannot store tokenizer ids for vocab_size > 65535")

    writer = ShardWriter(Path(args.output_dir), args.prefix, dtype, args.tokens_per_shard)
    quotas = weighted_quota(max_tokens, sources)
    seen = set() if args.dedup == "memory" else None
    near_dedup = args.near_dedup or defaults.get("near_dedup", "none")
    seen_simhashes = {} if near_dedup == "simhash" else None
    source_reports = []
    started_at = time.monotonic()

    try:
        for source in sources:
            name = source_name(source)
            quota = quotas[name]
            source_tokens = 0
            source_docs = 0
            filter_stats: Dict[str, int] = {}
            source_started_at = time.monotonic()
            print(f"[pretrain] {name}: quota={quota:,} tokens", flush=True)

            for text, meta in iter_clean_texts(
                source,
                defaults,
                seen_hashes=seen,
                seen_simhashes=seen_simhashes,
                stats=filter_stats,
            ):
                ids = tokenizer.encode(text)
                if not ids:
                    continue
                ids.append(tokenizer.eos_id)
                if source_tokens + len(ids) > quota:
                    break
                writer.write(ids)
                source_tokens += len(ids)
                source_docs += 1

                if args.max_docs_per_source and source_docs >= args.max_docs_per_source:
                    break
                if args.progress_docs > 0 and source_docs % args.progress_docs == 0:
                    elapsed = max(time.monotonic() - source_started_at, 1e-9)
                    total_elapsed = max(time.monotonic() - started_at, 1e-9)
                    source_pct = 100.0 * source_tokens / quota if quota else 100.0
                    total_pct = 100.0 * writer.total_tokens / max_tokens if max_tokens else 100.0
                    print(
                        "  "
                        f"{name}: docs={source_docs:,} "
                        f"tokens={source_tokens:,}/{quota:,} ({source_pct:.1f}%) "
                        f"total={writer.total_tokens:,}/{max_tokens:,} ({total_pct:.1f}%) "
                        f"source_tok/s={source_tokens / elapsed:,.0f} "
                        f"total_tok/s={writer.total_tokens / total_elapsed:,.0f}",
                        flush=True,
                    )

            source_reports.append(
                {
                    "name": name,
                    "dataset": source.get("dataset"),
                    "config": source.get("config"),
                    "split": source.get("split"),
                    "quota_tokens": quota,
                    "written_tokens": source_tokens,
                    "documents": source_docs,
                    "filter_stats": filter_stats,
                }
            )
    finally:
        shards = writer.close()

    manifest = {
        "config": args.config,
        "tokenizer_kind": args.tokenizer_kind,
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(dtype),
        "max_tokens": max_tokens,
        "written_tokens": writer.total_tokens,
        "tokens_per_shard": args.tokens_per_shard,
        "dedup": args.dedup,
        "near_dedup": near_dedup,
        "sources": source_reports,
        "shards": shards,
    }
    manifest_path = Path(args.manifest or Path(args.output_dir) / f"{args.prefix}.manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {writer.total_tokens:,} tokens to {len(shards)} shards", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
