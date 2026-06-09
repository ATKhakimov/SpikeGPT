"""Compare two v1 SpikeRuGPT checkpoints during pretraining."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig, LearnableLIFNode  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


def load_model(path: Path, ctx_len: int) -> tuple[GPT, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = GPT(
        GPTConfig(
            int(config.get("vocab_size", 32000)),
            ctx_len,
            model_type="RWKV",
            n_layer=int(config.get("n_layer", 12)),
            n_embd=int(config.get("n_embd", 512)),
        )
    ).cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


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


def make_sequences(texts, encode: Callable[[str], list[int]], eos_id: int, ctx_len: int, max_sequences: int):
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


@torch.no_grad()
def evaluate(model: GPT, sequences, batch_size: int) -> dict:
    losses = []
    total_tokens = 0
    max_bytes_seen = 0
    batch = []
    for seq, bytes_seen in sequences:
        batch.append(seq)
        max_bytes_seen = max(max_bytes_seen, bytes_seen)
        if len(batch) < batch_size:
            continue
        arr = np.stack(batch)
        x = torch.tensor(arr[:, :-1], dtype=torch.long, device="cuda")
        y = torch.tensor(arr[:, 1:], dtype=torch.long, device="cuda")
        loss = model(x, y)
        functional.reset_net(model)
        tokens = int(y.numel())
        losses.append((float(loss.item()), tokens))
        total_tokens += tokens
        batch.clear()
    if batch:
        arr = np.stack(batch)
        x = torch.tensor(arr[:, :-1], dtype=torch.long, device="cuda")
        y = torch.tensor(arr[:, 1:], dtype=torch.long, device="cuda")
        loss = model(x, y)
        functional.reset_net(model)
        tokens = int(y.numel())
        losses.append((float(loss.item()), tokens))
        total_tokens += tokens
    if not losses:
        raise RuntimeError("no eval sequences")
    loss = sum(value * tokens for value, tokens in losses) / total_tokens
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "tokens": total_tokens,
        "bytes": max_bytes_seen,
        "bpb": loss * total_tokens / max(max_bytes_seen, 1) / math.log(2),
    }


class ActivityCollector:
    def __init__(self):
        self.stats = defaultdict(lambda: {"spikes": 0.0, "count": 0})
        self.channels = defaultdict(lambda: None)
        self.channel_count = defaultdict(int)
        self.handles = []

    def _hook(self, key: str):
        def hook(_module, _inputs, output):
            spikes = (output.detach() > 0).float()
            self.stats[key]["spikes"] += float(spikes.sum().item())
            self.stats[key]["count"] += int(spikes.numel())
            by_channel = spikes.sum(dim=(0, 1)).detach().cpu()
            self.channels[key] = by_channel if self.channels[key] is None else self.channels[key] + by_channel
            self.channel_count[key] += int(spikes.shape[0] * spikes.shape[1])

        return hook

    def register(self, model: GPT) -> None:
        for layer, block in enumerate(model.blocks):
            self.handles.append(block.lif1.register_forward_hook(self._hook(f"{layer:02d}.lif1")))
            self.handles.append(block.lif2.register_forward_hook(self._hook(f"{layer:02d}.lif2")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def rows(self) -> list[dict]:
        rows = []
        for key, stat in sorted(self.stats.items()):
            layer, branch = key.split(".")
            channel_rates = self.channels[key].numpy() / max(self.channel_count[key], 1)
            rows.append(
                {
                    "layer": int(layer),
                    "branch": branch,
                    "firing_rate": stat["spikes"] / max(stat["count"], 1),
                    "silent_channel_fraction": float((channel_rates == 0).mean()),
                    "channel_rate_mean": float(channel_rates.mean()),
                    "channel_rate_std": float(channel_rates.std()),
                }
            )
        return rows


@torch.no_grad()
def analyze_activity(model: GPT, sequences, batch_size: int) -> list[dict]:
    collector = ActivityCollector()
    collector.register(model)
    batch = []
    try:
        for seq, _bytes_seen in sequences:
            batch.append(seq)
            if len(batch) < batch_size:
                continue
            arr = np.stack(batch)
            x = torch.tensor(arr[:, :-1], dtype=torch.long, device="cuda")
            _ = model(x)
            functional.reset_net(model)
            batch.clear()
        if batch:
            arr = np.stack(batch)
            x = torch.tensor(arr[:, :-1], dtype=torch.long, device="cuda")
            _ = model(x)
            functional.reset_net(model)
    finally:
        collector.close()
    return collector.rows()


def lif_params(model: GPT) -> list[dict]:
    rows = []
    for layer, block in enumerate(model.blocks):
        for branch in ("lif1", "lif2"):
            lif = getattr(block, branch)
            if isinstance(lif, LearnableLIFNode):
                rows.append(
                    {
                        "layer": layer,
                        "branch": branch,
                        "tau": float(lif.tau.detach().cpu().item()),
                        "threshold": float(lif.threshold.detach().cpu().item()),
                    }
                )
    return rows


def forward_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    x = model.atan(model.emb(idx))
    x = model.blocks(x)
    x = model.ln_out(x)
    return model.head(x)


@torch.no_grad()
def generate(model: GPT, ids: list[int], decode, ctx_len: int, length: int, seed: int) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
    out = []
    for _ in range(length):
        logits = forward_logits(model, ctx[:, -ctx_len:])[0, -1, :].float()
        functional.reset_net(model)
        for token_id in set(ctx[0].tolist()):
            if logits[token_id] > 0:
                logits[token_id] /= 1.15
            else:
                logits[token_id] *= 1.15
        probs = F.softmax(logits / 0.8, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > 0.9] = 0
        sorted_probs /= sorted_probs.sum().clamp_min(1e-12)
        next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
        out.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]], dtype=torch.long, device="cuda")], dim=1)
    return decode(out)


def plot_results(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(results["checkpoints"].keys())

    fig, ax = plt.subplots(figsize=(9, 4))
    splits = list(next(iter(results["eval"].values())).keys())
    x = np.arange(len(splits))
    width = 0.35
    for i, label in enumerate(labels):
        vals = [results["eval"][label][split]["loss"] for split in splits]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=label)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Loss")
    ax.set_title("Intermediate checkpoint validation loss")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v1_intermediate_validation_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for label in labels:
        rows = results["activity"][label]["_mean_by_layer"]
        ax.plot([r["layer"] for r in rows], [r["firing_rate"] for r in rows], marker="o", label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Firing rate")
    ax.set_title("Intermediate checkpoint firing rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v1_intermediate_firing_rate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=[
        "v1_3h=checkpoints/autonomous/autonomous-ctx1024-3h/final.pt",
        "v1_12h=checkpoints/autonomous/autonomous-ctx1024-12h/final.pt",
    ])
    parser.add_argument("--tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--validation-dir", default="data/validation_text")
    parser.add_argument("--splits", nargs="+", default=["val_wiki", "val_lit", "val_habr"])
    parser.add_argument("--eval-ctx-len", type=int, default=512)
    parser.add_argument("--activity-ctx-len", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generation-length", type=int, default=60)
    parser.add_argument("--out-json", default="reports/v1_intermediate_comparison.json")
    parser.add_argument("--out-md", default="ARTICLE/v1_intermediate_comparison.md")
    parser.add_argument("--figure-dir", default="ARTICLE/figures")
    args = parser.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    encode = lambda text: sp.encode(text, out_type=int)
    decode = lambda ids: sp.decode(ids)
    eos_id = int(sp.eos_id())
    validation_dir = Path(args.validation_dir)
    prompts = [
        "Почему русский язык сложен для языковых моделей?",
        "Объясни простыми словами, что такое нейроморфные вычисления.",
    ]

    specs = {}
    for item in args.checkpoints:
        label, path = item.split("=", 1)
        specs[label] = Path(path)

    results = {"checkpoints": {}, "eval": {}, "activity": {}, "lif_parameters": {}, "generations": {}}
    for label, path in specs.items():
        print(f"loading {label}: {path}", flush=True)
        model, checkpoint = load_model(path, args.eval_ctx_len)
        results["checkpoints"][label] = {
            "path": str(path),
            "step": int(checkpoint.get("step", 0)),
            "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        }
        results["eval"][label] = {}
        for split in args.splits:
            texts = iter_texts(validation_dir, split, args.max_docs)
            seqs = make_sequences(texts, encode, eos_id, args.eval_ctx_len, args.max_sequences)
            metrics = evaluate(model, seqs, args.batch_size)
            results["eval"][label][split] = metrics
            print(f"{label} {split} loss={metrics['loss']:.4f} bpb={metrics['bpb']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

        model, _checkpoint = load_model(path, args.activity_ctx_len)
        results["lif_parameters"][label] = lif_params(model)
        results["activity"][label] = {}
        all_activity = []
        for split in args.splits:
            texts = iter_texts(validation_dir, split, args.max_docs)
            seqs = make_sequences(texts, encode, eos_id, args.activity_ctx_len, args.max_sequences)
            rows = analyze_activity(model, seqs, args.batch_size)
            results["activity"][label][split] = rows
            all_activity.extend(rows)
            mean_rate = sum(r["firing_rate"] for r in rows) / len(rows)
            mean_silent = sum(r["silent_channel_fraction"] for r in rows) / len(rows)
            print(f"{label} {split} firing={mean_rate:.4f} silent={mean_silent:.4f}", flush=True)
        mean_by_layer = []
        for layer in sorted({r["layer"] for r in all_activity}):
            rows = [r for r in all_activity if r["layer"] == layer]
            mean_by_layer.append(
                {
                    "layer": layer,
                    "firing_rate": sum(r["firing_rate"] for r in rows) / len(rows),
                    "silent_channel_fraction": sum(r["silent_channel_fraction"] for r in rows) / len(rows),
                }
            )
        results["activity"][label]["_mean_by_layer"] = mean_by_layer
        results["generations"][label] = {}
        for i, prompt in enumerate(prompts):
            generated = generate(model, encode(prompt), decode, args.activity_ctx_len, args.generation_length, 20260602 + i)
            results["generations"][label][prompt] = prompt + generated
        del model
        torch.cuda.empty_cache()

    plot_results(results, Path(args.figure_dir))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# v1 intermediate checkpoint comparison",
        "",
        "Сравнение промежуточного `3h` checkpoint и текущего `12h` checkpoint.",
        "",
        "## Validation",
        "",
        "| Split | v1 3h loss | v1 12h loss | Delta | v1 3h BPB | v1 12h BPB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = list(specs.keys())
    for split in args.splits:
        a = results["eval"][labels[0]][split]
        b = results["eval"][labels[1]][split]
        lines.append(
            f"| {split} | {a['loss']:.4f} | {b['loss']:.4f} | {b['loss'] - a['loss']:+.4f} | {a['bpb']:.4f} | {b['bpb']:.4f} |"
        )
    lines.extend(["", "## Spiking Activity", "", "| Split | v1 3h firing | v1 12h firing | v1 3h silent | v1 12h silent |", "|---|---:|---:|---:|---:|"])
    for split in args.splits:
        a_rows = results["activity"][labels[0]][split]
        b_rows = results["activity"][labels[1]][split]
        a_rate = sum(r["firing_rate"] for r in a_rows) / len(a_rows)
        b_rate = sum(r["firing_rate"] for r in b_rows) / len(b_rows)
        a_silent = sum(r["silent_channel_fraction"] for r in a_rows) / len(a_rows)
        b_silent = sum(r["silent_channel_fraction"] for r in b_rows) / len(b_rows)
        lines.append(f"| {split} | {a_rate:.4f} | {b_rate:.4f} | {a_silent:.4f} | {b_silent:.4f} |")
    lines.extend(["", "## Generations", ""])
    for prompt in prompts:
        lines.extend([f"### {prompt}", ""])
        for label in labels:
            lines.extend([f"**{label}**", "", f"```text\n{results['generations'][label][prompt]}\n```", ""])
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)
    print(f"wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
