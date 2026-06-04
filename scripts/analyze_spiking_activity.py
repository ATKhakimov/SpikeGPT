"""Analyze SpikeGPT spiking activity for article figures.

Produces JSON/CSV metrics and PNG plots:
- firing rate by layer and LIF branch;
- silent channel fraction;
- v1 learnable tau / threshold profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig, LearnableLIFNode  # noqa: E402
from src.spikingjelly.clock_driven import functional, neuron, surrogate  # noqa: E402


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
    model.eval()
    return model


def load_v0(path: Path, ctx_len: int) -> GPT:
    model = build_model(50258, ctx_len, old_lif=True)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def load_v1(path: Path, ctx_len: int) -> GPT:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_model(int(config.get("vocab_size", 32000)), ctx_len, old_lif=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


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


def make_sequences(
    texts,
    encode: Callable[[str], list[int]],
    eos_id: int,
    ctx_len: int,
    max_sequences: int,
):
    produced = 0
    for text in texts:
        ids = list(encode(text))
        ids.append(eos_id)
        if len(ids) < ctx_len + 1:
            continue
        for offset in range(0, len(ids) - ctx_len - 1, ctx_len):
            yield np.asarray(ids[offset : offset + ctx_len + 1], dtype=np.int64)
            produced += 1
            if produced >= max_sequences:
                return


class SpikeCollector:
    def __init__(self, model_name: str, split: str):
        self.model_name = model_name
        self.split = split
        self.stats = defaultdict(lambda: {"spikes": 0.0, "count": 0, "samples": 0})
        self.channel_spikes = defaultdict(lambda: None)
        self.channel_count = defaultdict(int)
        self.handles = []

    def hook(self, layer: int, branch: str):
        key = f"layer{layer:02d}.{branch}"

        def _hook(_module, _inputs, output):
            spikes = (output.detach() > 0).to(torch.float32)
            self.stats[key]["spikes"] += float(spikes.sum().item())
            self.stats[key]["count"] += int(spikes.numel())
            self.stats[key]["samples"] += 1
            by_channel = spikes.sum(dim=(0, 1)).detach().cpu()
            if self.channel_spikes[key] is None:
                self.channel_spikes[key] = by_channel
            else:
                self.channel_spikes[key] += by_channel
            self.channel_count[key] += int(spikes.shape[0] * spikes.shape[1])

        return _hook

    def register(self, model: GPT) -> None:
        for layer, block in enumerate(model.blocks):
            self.handles.append(block.lif1.register_forward_hook(self.hook(layer, "lif1")))
            self.handles.append(block.lif2.register_forward_hook(self.hook(layer, "lif2")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rows(self) -> list[dict]:
        rows = []
        for key, stat in sorted(self.stats.items()):
            layer_text, branch = key.split(".")
            layer = int(layer_text.replace("layer", ""))
            channel_spikes = self.channel_spikes[key]
            channel_count = max(self.channel_count[key], 1)
            channel_rates = channel_spikes.numpy() / channel_count
            rows.append(
                {
                    "model": self.model_name,
                    "split": self.split,
                    "layer": layer,
                    "branch": branch,
                    "firing_rate": stat["spikes"] / max(stat["count"], 1),
                    "silent_channel_fraction": float((channel_rates == 0).mean()),
                    "channel_rate_mean": float(channel_rates.mean()),
                    "channel_rate_std": float(channel_rates.std()),
                    "channel_rate_p90": float(np.quantile(channel_rates, 0.9)),
                    "samples": stat["samples"],
                }
            )
        return rows


@torch.no_grad()
def run_activity(model: GPT, sequences, batch_size: int, collector: SpikeCollector) -> None:
    collector.register(model)
    try:
        batch = []
        for seq in sequences:
            batch.append(seq)
            if len(batch) < batch_size:
                continue
            x = torch.tensor(np.stack([s[:-1] for s in batch]), dtype=torch.long, device="cuda")
            _ = model(x)
            functional.reset_net(model)
            batch.clear()
        if batch:
            x = torch.tensor(np.stack([s[:-1] for s in batch]), dtype=torch.long, device="cuda")
            _ = model(x)
            functional.reset_net(model)
    finally:
        collector.close()


def tau_rows(model_name: str, model: GPT) -> list[dict]:
    rows = []
    for layer, block in enumerate(model.blocks):
        for branch in ("lif1", "lif2"):
            lif = getattr(block, branch)
            if isinstance(lif, LearnableLIFNode):
                rows.append(
                    {
                        "model": model_name,
                        "layer": layer,
                        "branch": branch,
                        "tau": float(lif.tau.detach().cpu().item()),
                        "threshold": float(lif.threshold.detach().cpu().item()),
                    }
                )
            else:
                rows.append(
                    {
                        "model": model_name,
                        "layer": layer,
                        "branch": branch,
                        "tau": float(getattr(lif, "tau", 2.0)),
                        "threshold": float(getattr(lif, "v_threshold", 1.0)),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict], tau_data: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = sorted({row["split"] for row in rows})
    models = sorted({row["model"] for row in rows})
    branches = ["lif1", "lif2"]

    for split in splits:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, branch in zip(axes, branches):
            for model in models:
                subset = [r for r in rows if r["split"] == split and r["branch"] == branch and r["model"] == model]
                subset.sort(key=lambda r: r["layer"])
                ax.plot(
                    [r["layer"] for r in subset],
                    [r["firing_rate"] for r in subset],
                    marker="o",
                    label=model,
                )
            ax.set_title(f"{split} {branch}")
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Firing rate")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"spiking_firing_rate_{split}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        by_layer = []
        for layer in sorted({r["layer"] for r in subset}):
            vals = [r["firing_rate"] for r in subset if r["layer"] == layer]
            by_layer.append((layer, sum(vals) / len(vals)))
        ax.plot([x for x, _ in by_layer], [y for _, y in by_layer], marker="o", label=model)
    ax.set_title("Mean firing rate by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Firing rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spiking_firing_rate_mean_by_layer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        by_layer = []
        for layer in sorted({r["layer"] for r in subset}):
            vals = [r["silent_channel_fraction"] for r in subset if r["layer"] == layer]
            by_layer.append((layer, sum(vals) / len(vals)))
        ax.plot([x for x, _ in by_layer], [y for _, y in by_layer], marker="o", label=model)
    ax.set_title("Silent channel fraction by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spiking_silent_channels_by_layer.png", dpi=180)
    plt.close(fig)

    v1_tau = [r for r in tau_data if r["model"] == "v1"]
    if v1_tau:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        for branch in branches:
            subset = [r for r in v1_tau if r["branch"] == branch]
            subset.sort(key=lambda r: r["layer"])
            ax1.plot([r["layer"] for r in subset], [r["tau"] for r in subset], marker="o", label=f"{branch} tau")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Tau")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        for branch in branches:
            subset = [r for r in v1_tau if r["branch"] == branch]
            subset.sort(key=lambda r: r["layer"])
            ax2.plot(
                [r["layer"] for r in subset],
                [r["threshold"] for r in subset],
                marker="x",
                linestyle="--",
                label=f"{branch} threshold",
            )
        ax2.set_ylabel("Threshold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.set_title("v1 learnable LIF parameters")
        fig.tight_layout()
        fig.savefig(out_dir / "spiking_v1_tau_threshold_profile.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", default="data/validation_text")
    parser.add_argument("--splits", nargs="+", default=["val_wiki", "val_lit", "val_habr"])
    parser.add_argument("--ctx-len", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=60)
    parser.add_argument("--max-sequences", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--v0-checkpoint", default="models/v0_spikerugpt_100m_taiga/spikegpt-ru-175.pth")
    parser.add_argument("--v0-tokenizer", default="models/v0_spikerugpt_100m_taiga/tokenizer/tokenizer.json")
    parser.add_argument("--v1-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-12h/final.pt")
    parser.add_argument("--v1-tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--out-json", default="reports/spiking_activity_v0_v1.json")
    parser.add_argument("--out-csv", default="reports/spiking_activity_v0_v1.csv")
    parser.add_argument("--figure-dir", default="ARTICLE/figures")
    args = parser.parse_args()

    import sentencepiece as spm

    validation_dir = Path(args.validation_dir)
    v0_tokenizer = Tokenizer.from_file(args.v0_tokenizer)
    v1_tokenizer = spm.SentencePieceProcessor(model_file=args.v1_tokenizer)
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

    rows = []
    tau_data = []
    try:
        for name, item in models.items():
            tau_data.extend(tau_rows(name, item["model"]))
        for split in args.splits:
            for name, item in models.items():
                collector = SpikeCollector(name, split)
                texts = iter_texts(validation_dir, split, args.max_docs)
                sequences = make_sequences(texts, item["encode"], item["eos_id"], args.ctx_len, args.max_sequences)
                run_activity(item["model"], sequences, args.batch_size, collector)
                split_rows = collector.rows()
                rows.extend(split_rows)
                mean_rate = sum(r["firing_rate"] for r in split_rows) / max(len(split_rows), 1)
                mean_silent = sum(r["silent_channel_fraction"] for r in split_rows) / max(len(split_rows), 1)
                print(
                    f"{split} {name} mean_firing_rate={mean_rate:.4f} "
                    f"mean_silent_channel_fraction={mean_silent:.4f}",
                    flush=True,
                )
                functional.reset_net(item["model"])
    finally:
        del models
        torch.cuda.empty_cache()

    result = {
        "config": {
            "ctx_len": args.ctx_len,
            "max_docs": args.max_docs,
            "max_sequences": args.max_sequences,
            "batch_size": args.batch_size,
            "splits": args.splits,
        },
        "activity": rows,
        "lif_parameters": tau_data,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.out_csv), rows)
    write_csv(Path(args.out_csv).with_name("spiking_lif_parameters_v0_v1.csv"), tau_data)
    make_plots(rows, tau_data, Path(args.figure_dir))
    print(f"wrote {args.out_json}", flush=True)
    print(f"wrote {args.out_csv}", flush=True)
    print(f"wrote figures to {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
