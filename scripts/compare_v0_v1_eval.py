"""Compare v0 and v1 SpikeRuGPT checkpoints on shared validation text.

The models use different tokenizers, so token-level perplexity is not directly
comparable. This script also reports bits per byte (BPB), which is a better
cross-tokenizer metric for article notes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional, neuron, surrogate  # noqa: E402


def iter_texts(validation_dir: Path, split: str, max_docs: int):
    path = validation_dir / f"{split}.jsonl"
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if emitted >= max_docs:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or ""
            if text.strip():
                emitted += 1
                yield text


def make_sequences(texts, encode, eos_id: int, ctx_len: int, max_sequences: int):
    produced = 0
    bytes_seen = 0
    for text in texts:
        ids = list(encode(text))
        ids.append(eos_id)
        bytes_seen += len(text.encode("utf-8"))
        if len(ids) < ctx_len + 1:
            continue
        for offset in range(0, len(ids) - ctx_len - 1, ctx_len):
            yield np.asarray(ids[offset : offset + ctx_len + 1], dtype=np.int64), bytes_seen
            produced += 1
            if produced >= max_sequences:
                return


def build_model(vocab_size: int, ctx_len: int, old_lif: bool) -> GPT:
    model = GPT(GPTConfig(vocab_size, ctx_len, model_type="RWKV", n_layer=12, n_embd=512)).cuda()
    if old_lif:
        for block in model.blocks:
            block.lif1 = neuron.MultiStepLIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(alpha=2.0),
                backend="torch",
                v_threshold=1.0,
            ).cuda()
            block.lif2 = neuron.MultiStepLIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(alpha=2.0),
                backend="torch",
                v_threshold=1.0,
            ).cuda()
    return model


def load_v0(path: Path, ctx_len: int) -> GPT:
    model = build_model(50258, ctx_len, old_lif=True)
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"v0 load_state missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def load_v1(path: Path, ctx_len: int) -> GPT:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_model(int(config.get("vocab_size", 32000)), ctx_len, old_lif=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model: GPT, sequences, batch_size: int) -> dict:
    losses = []
    total_tokens = 0
    max_bytes_seen = 0
    batch = []
    for seq, bytes_seen in sequences:
        batch.append(seq)
        max_bytes_seen = max(max_bytes_seen, bytes_seen)
        if len(batch) < batch_size:
            continue
        x = torch.tensor(np.stack([s[:-1] for s in batch]), dtype=torch.long, device="cuda")
        y = torch.tensor(np.stack([s[1:] for s in batch]), dtype=torch.long, device="cuda")
        loss = model(x, y)
        functional.reset_net(model)
        loss_value = float(loss.item())
        tokens = int(y.numel())
        losses.append((loss_value, tokens))
        total_tokens += tokens
        batch.clear()
    if batch:
        x = torch.tensor(np.stack([s[:-1] for s in batch]), dtype=torch.long, device="cuda")
        y = torch.tensor(np.stack([s[1:] for s in batch]), dtype=torch.long, device="cuda")
        loss = model(x, y)
        functional.reset_net(model)
        loss_value = float(loss.item())
        tokens = int(y.numel())
        losses.append((loss_value, tokens))
        total_tokens += tokens
    if not losses:
        raise RuntimeError("No validation sequences produced")
    loss = sum(value * tokens for value, tokens in losses) / total_tokens
    bpb = loss * total_tokens / max(max_bytes_seen, 1) / math.log(2)
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "tokens": total_tokens,
        "bytes": max_bytes_seen,
        "bpb": bpb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", default="data/validation_text")
    parser.add_argument("--splits", nargs="+", default=["val_wiki", "val_lit", "val_habr"])
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--max-sequences", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--v0-checkpoint", default="models/v0_spikerugpt_100m_taiga/spikegpt-ru-175.pth")
    parser.add_argument("--v0-tokenizer", default="models/v0_spikerugpt_100m_taiga/tokenizer/tokenizer.json")
    parser.add_argument("--v1-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-12h/final.pt")
    parser.add_argument("--v1-tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--out", default="reports/v0_v1_eval_small.json")
    args = parser.parse_args()

    import sentencepiece as spm

    validation_dir = Path(args.validation_dir)
    v0_tokenizer = Tokenizer.from_file(args.v0_tokenizer)
    v1_tokenizer = spm.SentencePieceProcessor(model_file=args.v1_tokenizer)

    results = {
        "ctx_len": args.ctx_len,
        "max_docs": args.max_docs,
        "max_sequences": args.max_sequences,
        "batch_size": args.batch_size,
        "splits": {},
    }

    models = {
        "v0": {
            "model": load_v0(Path(args.v0_checkpoint), args.ctx_len),
            "encode": lambda text: v0_tokenizer.encode(text).ids,
            "eos_id": 50257,
        },
        "v1": {
            "model": load_v1(Path(args.v1_checkpoint), args.ctx_len),
            "encode": lambda text: v1_tokenizer.encode(text, out_type=int),
            "eos_id": int(v1_tokenizer.eos_id()),
        },
    }

    try:
        for split in args.splits:
            results["splits"][split] = {}
            for name, item in models.items():
                texts = iter_texts(validation_dir, split, args.max_docs)
                sequences = make_sequences(
                    texts,
                    item["encode"],
                    item["eos_id"],
                    args.ctx_len,
                    args.max_sequences,
                )
                metrics = evaluate_model(item["model"], sequences, args.batch_size)
                results["splits"][split][name] = metrics
                print(
                    f"{split} {name} loss={metrics['loss']:.4f} "
                    f"ppl={metrics['ppl']:.1f} bpb={metrics['bpb']:.4f} "
                    f"tokens={metrics['tokens']} bytes={metrics['bytes']}",
                    flush=True,
                )
    finally:
        del models
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
