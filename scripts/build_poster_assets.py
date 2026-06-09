"""Build poster-ready tables, figures, and summary notes for SpikeRuGPT."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ARTICLE" / "poster_assets"
FIG = OUT / "figures"


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_activity(rows: list[dict]) -> dict[str, float]:
    return {
        "firing_rate": sum(r["firing_rate"] for r in rows) / len(rows),
        "silent_channel_fraction": sum(r["silent_channel_fraction"] for r in rows) / len(rows),
    }


def copy_figure(src: str, name: str | None = None) -> str:
    source = ROOT / src
    target = FIG / (name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target.relative_to(OUT))


def build_figures(rows_models: list[dict], rows_activity: list[dict], rows_eval: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    labels = [r["label"] for r in rows_models]
    params = [r["params_m"] for r in rows_models]
    tokens = [r["train_tokens_b"] for r in rows_models]

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    width = 0.38
    ax1.bar(x - width / 2, params, width, label="Parameters, M", color="#3b82f6")
    ax1.set_ylabel("Parameters, M")
    ax1.set_xticks(x, labels, rotation=12, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, tokens, width, label="Training tokens, B", color="#f97316")
    ax2.set_ylabel("Training tokens, B")
    ax1.set_title("Model scale and training tokens")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    path = FIG / "poster_model_scale.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    made.append(str(path.relative_to(OUT)))

    splits = sorted({r["split"] for r in rows_eval})
    models = ["v0_taiga_100m", "base_1b_74m", "sft_v2_superclean"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(splits))
    width = 0.25
    for idx, model in enumerate(models):
        vals = []
        for split in splits:
            match = [r for r in rows_eval if r["model"] == model and r["split"] == split]
            vals.append(match[0]["bpb"] if match else float("nan"))
        ax.bar(x + (idx - 1) * width, vals, width, label=model)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Bits per byte")
    ax.set_title("Validation BPB, cross-tokenizer comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "poster_validation_bpb.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    made.append(str(path.relative_to(OUT)))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    models = ["v0_taiga_100m", "base_1b_74m", "sft_v2_superclean", "original_en_216m"]
    vals = []
    labels = []
    for model in models:
        subset = [r for r in rows_activity if r["model"] == model]
        if not subset:
            continue
        vals.append(sum(r["firing_rate"] for r in subset) / len(subset))
        labels.append(model)
    bars = ax.bar(labels, [v * 100 for v in vals], color=["#0f766e", "#3b82f6", "#60a5fa", "#ef4444"][: len(vals)])
    ax.set_ylabel("Mean firing rate, %")
    ax.set_title("Spiking activity")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    path = FIG / "poster_firing_rate_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    made.append(str(path.relative_to(OUT)))

    return made


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    eval_final = read_json("reports/eval/step_43674_metrics.json")
    v0_v1_eval = read_json("reports/v0_v1_eval_small.json")
    trajectory = read_json("reports/v0_v1_trajectory_summary.json")
    sft_activity = read_json("ARTICLE/sft_v2_superclean/base_vs_sft_v2_activity.json")
    sft_report = read_json("reports/sft/sft-step43674-v2-superclean.json")

    rows_models = [
        {
            "label": "v0_taiga_100m",
            "params_m": 92.43,
            "train_tokens_b": 1.8,
            "tokenizer": "ruGPT-3 BPE 50k",
            "notes": "old Taiga continuation baseline",
        },
        {
            "label": "base_1b_74m",
            "params_m": 73.73,
            "train_tokens_b": 0.945,
            "tokenizer": "SentencePiece BPE 32k",
            "notes": "controlled v1 pretrain checkpoint",
        },
        {
            "label": "sft_v2_superclean",
            "params_m": 73.73,
            "train_tokens_b": 0.945,
            "tokenizer": "SentencePiece BPE 32k",
            "notes": "super-clean SFT, diagnostic only",
        },
        {
            "label": "original_en_216m",
            "params_m": 215.40,
            "train_tokens_b": 5.0,
            "tokenizer": "GPT-NeoX 20B BPE",
            "notes": "original English OpenWebText model",
        },
    ]
    write_csv(OUT / "model_comparison.csv", rows_models)

    rows_eval = []
    for split, block in v0_v1_eval["splits"].items():
        for model, label in [("v0", "v0_taiga_100m"), ("v1", "base_1b_74m")]:
            rows_eval.append(
                {
                    "split": split,
                    "model": label,
                    "loss": block[model]["loss"],
                    "ppl": block[model]["ppl"],
                    "bpb": block[model]["bpb"],
                }
            )
    for split, block in sft_activity["eval"]["sft_v2"].items():
        rows_eval.append(
            {
                "split": split,
                "model": "sft_v2_superclean",
                "loss": block["loss"],
                "ppl": math.exp(block["loss"]),
                "bpb": block["bpb"],
            }
        )
    write_csv(OUT / "validation_metrics.csv", rows_eval)

    rows_activity = [
        {"split": split, "model": "v0_taiga_100m", **vals}
        for split, vals in trajectory["activity"].items()
        if "v0" in vals
        for vals in [vals["v0"]]
    ]
    for split, vals in trajectory["activity"].items():
        rows_activity.append({"split": split, "model": "base_1b_74m", **vals["v1_12h"]})
    for split, rows in sft_activity["activity"]["sft_v2"].items():
        if split.startswith("_"):
            continue
        rows_activity.append({"split": split, "model": "sft_v2_superclean", **mean_activity(rows)})
    rows_activity.append(
        {
            "split": "openwebtext",
            "model": "original_en_216m",
            "firing_rate": 0.217,
            "silent_channel_fraction": 0.783,
        }
    )
    write_csv(OUT / "spiking_activity.csv", rows_activity)

    rows_sft = [
        {
            "run": "sft_v1_dirty",
            "examples": 65056,
            "supervised_val_loss": 3.8272,
            "supervised_val_ppl": 45.93,
            "generation_artifacts": "role/content and code-like artifacts observed",
            "interpretation": "diagnostic failed SFT due to dirty data",
        },
        {
            "run": "sft_v2_superclean",
            "examples": 45000,
            "supervised_val_loss": sft_report["validation"]["loss"],
            "supervised_val_ppl": sft_report["validation"]["ppl"],
            "generation_artifacts": "technical artifacts removed",
            "interpretation": "format cleaned, semantic quality still weak",
        },
    ]
    write_csv(OUT / "sft_ablation.csv", rows_sft)

    copied = [
        copy_figure("analysis/figures/training_curve.png"),
        copy_figure("analysis/figures/sparsity_summary.png"),
        copy_figure("analysis/figures/spike_sparsity.png"),
        copy_figure("analysis/figures/sparsity_heatmap.png"),
        copy_figure("analysis/figures/lif_tau_evolution.png"),
        copy_figure("analysis/figures/lif_tau_final.png"),
        copy_figure("ARTICLE/figures/v0_v1_trajectory_firing_rate.png"),
        copy_figure("ARTICLE/figures/training_loss_by_tokens.png"),
    ]
    made = build_figures(rows_models, rows_activity, rows_eval)

    summary = [
        "# Poster Assets: SpikeRuGPT",
        "",
        "This folder contains poster-ready tables, figures, and compact text blocks.",
        "",
        "## Core Message",
        "",
        "SpikeRuGPT demonstrates that a SpikeGPT/RWKV-style spiking language model can be adapted to Russian text. The strongest result is not chat quality, but the combination of Russian base-language modeling, continuation examples, and measurable neuromorphic sparsity via LIF firing rate.",
        "",
        "## Recommended Poster Title",
        "",
        "SpikeRuGPT: Russian Spiking Language Modeling and Neuromorphic Sparsity Analysis",
        "",
        "## Main Quantitative Results",
        "",
        "- v0 Taiga model: 92.43M actual parameters, ruGPT-3 BPE 50k, about 1.8B Taiga tokens, model-card validation PPL around 67; thesis/local notes report best PPL 59.79.",
        "- v1 controlled base: 73.73M parameters, SentencePiece BPE 32k, 944.6M tokens seen, validation PPL 69.90 on wiki / 135.14 on lit / 137.96 on habr.",
        "- Original English SpikeGPT: 215.40M parameters, reported 5B OpenWebText tokens, mean firing rate around 21.7%.",
        "- Russian v0 firing rate: around 33.2% in original analysis, suggesting higher spike activity than English in the reported setup.",
        "- Current v1 base firing rate: around 9.3-9.8% on local validation probes; SFT v2 barely changes this global activity.",
        "",
        "## Honest Interpretation",
        "",
        "- v0 remains the best Russian continuation baseline in this repo.",
        "- v1 improves engineering reproducibility, tokenizer efficiency, and controlled data filtering, but current generation quality is weaker than v0.",
        "- SFT v2 removes formatting artifacts from dirty SFT, but does not solve semantic quality for a 74M base model.",
        "- The poster should frame this as an engineering/research trajectory, not as a finished assistant model.",
        "",
        "## Tables",
        "",
        "- `model_comparison.csv`: model scale and training setup.",
        "- `validation_metrics.csv`: loss/PPL/BPB for v0/base/SFT probes.",
        "- `spiking_activity.csv`: firing rate and silent-channel fraction.",
        "- `sft_ablation.csv`: dirty vs super-clean SFT.",
        "",
        "## Figures",
        "",
    ]
    for item in copied + made:
        summary.append(f"- `{item}`")

    summary.extend(
        [
            "",
            "## Poster Layout Suggestion",
            "",
            "1. Left column: motivation, SpikeGPT architecture, Russian adaptation.",
            "2. Middle column: training setup, tokenizer/parameter comparison, validation metrics.",
            "3. Right column: spiking activity, firing-rate plots, continuation examples, limitations.",
            "",
            "## Short Figure Captions",
            "",
            "- `poster_model_scale.png`: Parameter count and training-token scale for Russian v0/v1 and English original SpikeGPT.",
            "- `poster_validation_bpb.png`: Cross-tokenizer validation comparison using bits per byte.",
            "- `poster_firing_rate_summary.png`: Mean LIF firing rate across compared models/probes.",
            "- `sparsity_summary.png`: Russian versus English spike activity from the original v0 analysis.",
            "- `v0_v1_trajectory_firing_rate.png`: v1 pretraining trajectory toward lower firing rate.",
            "",
            "## Limitations To State Explicitly",
            "",
            "- v0, v1, and English original differ in model size, tokenizer, data, and training tokens.",
            "- Token-level PPL is not directly comparable across tokenizers; use BPB for cross-tokenizer comparison.",
            "- Firing rate is an activity/efficiency proxy, not a direct text-quality metric.",
            "- SFT results are diagnostic; current SFT model should not be presented as final assistant quality.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
