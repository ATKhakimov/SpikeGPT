"""Summarize v0 vs v1 intermediate trajectory from existing reports."""

from __future__ import annotations

import json
from pathlib import Path


def mean_activity(rows: list[dict]) -> dict:
    return {
        "firing_rate": sum(row["firing_rate"] for row in rows) / len(rows),
        "silent_channel_fraction": sum(row["silent_channel_fraction"] for row in rows) / len(rows),
    }


def main() -> None:
    v0_v1 = json.loads(Path("reports/v0_v1_eval_small.json").read_text(encoding="utf-8"))
    intermediate = json.loads(Path("reports/v1_intermediate_comparison.json").read_text(encoding="utf-8"))
    spiking = json.loads(Path("reports/spiking_activity_v0_v1.json").read_text(encoding="utf-8"))

    splits = ["val_wiki", "val_lit", "val_habr"]
    models = ["v0", "v1_3h", "v1_12h"]
    eval_table: dict[str, dict[str, dict]] = {split: {} for split in splits}
    for split in splits:
        eval_table[split]["v0"] = v0_v1["splits"][split]["v0"]
        eval_table[split]["v1_3h"] = intermediate["eval"]["v1_3h"][split]
        eval_table[split]["v1_12h"] = intermediate["eval"]["v1_12h"][split]

    activity_table: dict[str, dict[str, dict]] = {split: {} for split in splits}
    for split in splits:
        for model in ("v0", "v1"):
            rows = [row for row in spiking["activity"] if row["split"] == split and row["model"] == model]
            activity_table[split]["v0" if model == "v0" else "v1_12h"] = mean_activity(rows)
        rows = intermediate["activity"]["v1_3h"][split]
        activity_table[split]["v1_3h"] = mean_activity(rows)

    out = {
        "eval": eval_table,
        "activity": activity_table,
        "notes": {
            "v0": "Public proof-of-concept model trained on claimed ~1.8B Taiga tokens.",
            "v1_3h": "Current pipeline intermediate checkpoint, ~20.9M tokens seen.",
            "v1_12h": "Current pipeline 12h checkpoint, ~104.8M tokens seen.",
        },
    }
    Path("reports/v0_v1_trajectory_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = Path("ARTICLE/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(splits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        ax.bar(x + (i - 1) * width, [eval_table[split][model]["loss"] for split in splits], width, label=model)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Loss")
    ax.set_title("v0 vs v1 trajectory: validation loss")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "v0_v1_trajectory_validation_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        ax.bar(x + (i - 1) * width, [eval_table[split][model]["bpb"] for split in splits], width, label=model)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Bits per byte")
    ax.set_title("v0 vs v1 trajectory: BPB")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "v0_v1_trajectory_bpb.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        ax.bar(
            x + (i - 1) * width,
            [activity_table[split][model]["firing_rate"] for split in splits],
            width,
            label=model,
        )
    ax.set_xticks(x, splits)
    ax.set_ylabel("Mean firing rate")
    ax.set_title("v0 vs v1 trajectory: spiking activity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "v0_v1_trajectory_firing_rate.png", dpi=180)
    plt.close(fig)

    lines = [
        "# v0 vs v1 trajectory comparison",
        "",
        "Сводка по трем точкам: старая публичная v0, промежуточная v1 3h и текущая v1 12h.",
        "",
        "## Validation",
        "",
        "| Split | v0 loss | v1 3h loss | v1 12h loss | v0 BPB | v1 3h BPB | v1 12h BPB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        lines.append(
            "| {split} | {v0_loss:.4f} | {v13_loss:.4f} | {v112_loss:.4f} | "
            "{v0_bpb:.4f} | {v13_bpb:.4f} | {v112_bpb:.4f} |".format(
                split=split,
                v0_loss=eval_table[split]["v0"]["loss"],
                v13_loss=eval_table[split]["v1_3h"]["loss"],
                v112_loss=eval_table[split]["v1_12h"]["loss"],
                v0_bpb=eval_table[split]["v0"]["bpb"],
                v13_bpb=eval_table[split]["v1_3h"]["bpb"],
                v112_bpb=eval_table[split]["v1_12h"]["bpb"],
            )
        )
    lines.extend(
        [
            "",
            "## Spiking Activity",
            "",
            "| Split | v0 firing | v1 3h firing | v1 12h firing | v0 silent | v1 3h silent | v1 12h silent |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in splits:
        lines.append(
            "| {split} | {v0_rate:.4f} | {v13_rate:.4f} | {v112_rate:.4f} | "
            "{v0_silent:.4f} | {v13_silent:.4f} | {v112_silent:.4f} |".format(
                split=split,
                v0_rate=activity_table[split]["v0"]["firing_rate"],
                v13_rate=activity_table[split]["v1_3h"]["firing_rate"],
                v112_rate=activity_table[split]["v1_12h"]["firing_rate"],
                v0_silent=activity_table[split]["v0"]["silent_channel_fraction"],
                v13_silent=activity_table[split]["v1_3h"]["silent_channel_fraction"],
                v112_silent=activity_table[split]["v1_12h"]["silent_channel_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v0 пока лучше v1 по LM loss/BPB, что ожидаемо: v0 заявлена как обученная на ~1.8B Taiga tokens.",
            "- v1 12h заметно лучше v1 3h по всем проверенным split-ам.",
            "- v1 при обучении становится более sparse: firing rate снижается, silent-channel fraction растет.",
            "- Это хороший промежуточный вывод: v1 еще не догнала v0 по качеству, но trajectory правильная.",
            "",
            "Figures:",
            "",
            "```text",
            "ARTICLE/figures/v0_v1_trajectory_validation_loss.png",
            "ARTICLE/figures/v0_v1_trajectory_bpb.png",
            "ARTICLE/figures/v0_v1_trajectory_firing_rate.png",
            "```",
            "",
        ]
    )
    Path("ARTICLE/v0_v1_trajectory_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/v0_v1_trajectory_summary.json")
    print("wrote ARTICLE/v0_v1_trajectory_comparison.md")


if __name__ == "__main__":
    main()
