"""Plot SpikeRuGPT training curves from autonomous metrics JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_train_steps(path: Path, label: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "train_step":
                continue
            rows.append(
                {
                    "label": label,
                    "time": row.get("time"),
                    "step": int(row["step"]),
                    "tokens_seen": int(row.get("tokens_seen", 0)),
                    "loss": float(row["loss"]),
                    "avg_recent": float(row.get("avg_recent", row["loss"])),
                    "tok_per_sec": float(row.get("tok_per_sec", 0.0)),
                    "peak_mem_gb": float(row.get("peak_mem_gb", 0.0)),
                    "grad_norm": float(row.get("grad_norm", 0.0)),
                }
            )
    return rows


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    running = []
    for value in values:
        running.append(value)
        if len(running) > window:
            running.pop(0)
        out.append(sum(running) / len(running))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "autonomous-ctx1024-3h=reports/autonomous-ctx1024-3h.metrics.jsonl",
            "autonomous-ctx1024-12h=reports/autonomous-ctx1024-12h.metrics.jsonl",
            "bf16-batch22-smoke2=reports/bf16-batch22-smoke2.metrics.jsonl",
        ],
        help="Run specs as label=path.",
    )
    parser.add_argument("--out-dir", default="ARTICLE/figures")
    parser.add_argument("--summary", default="reports/training_curve_summary.json")
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    labels = []
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"run spec must be label=path: {spec}")
        label, path_text = spec.split("=", 1)
        path = Path(path_text)
        if not path.exists():
            print(f"skip missing {path}", flush=True)
            continue
        rows = read_train_steps(path, label)
        if not rows:
            print(f"skip empty {path}", flush=True)
            continue
        labels.append(label)
        all_rows.extend(rows)
        print(f"{label}: {len(rows)} train_step rows", flush=True)

    if not all_rows:
        raise RuntimeError("No train_step rows found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in labels:
        rows = sorted([row for row in all_rows if row["label"] == label], key=lambda row: row["step"])
        x = [row["tokens_seen"] / 1_000_000 for row in rows]
        y = smooth([row["loss"] for row in rows], args.smooth_window)
        ax.plot(x, y, label=label)
    ax.set_title("Training loss by tokens seen")
    ax.set_xlabel("Tokens seen, M")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_by_tokens.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in labels:
        rows = sorted([row for row in all_rows if row["label"] == label], key=lambda row: row["step"])
        x = [row["step"] for row in rows]
        y = smooth([row["loss"] for row in rows], args.smooth_window)
        ax.plot(x, y, label=label)
    ax.set_title("Training loss by step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_by_step.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label in labels:
        rows = sorted([row for row in all_rows if row["label"] == label], key=lambda row: row["step"])
        x = [row["tokens_seen"] / 1_000_000 for row in rows]
        y = smooth([row["tok_per_sec"] for row in rows], args.smooth_window)
        ax.plot(x, y, label=label)
    ax.set_title("Training throughput")
    ax.set_xlabel("Tokens seen, M")
    ax.set_ylabel("Tokens/sec")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_throughput_by_tokens.png", dpi=180)
    plt.close(fig)

    summary = {}
    for label in labels:
        rows = sorted([row for row in all_rows if row["label"] == label], key=lambda row: row["step"])
        summary[label] = {
            "points": len(rows),
            "first_step": rows[0]["step"],
            "last_step": rows[-1]["step"],
            "first_tokens_seen": rows[0]["tokens_seen"],
            "last_tokens_seen": rows[-1]["tokens_seen"],
            "first_loss": rows[0]["loss"],
            "last_loss": rows[-1]["loss"],
            "min_loss": min(row["loss"] for row in rows),
            "median_tok_per_sec": sorted(row["tok_per_sec"] for row in rows)[len(rows) // 2],
            "max_peak_mem_gb": max(row["peak_mem_gb"] for row in rows),
        }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'training_loss_by_tokens.png'}", flush=True)
    print(f"wrote {out_dir / 'training_loss_by_step.png'}", flush=True)
    print(f"wrote {out_dir / 'training_throughput_by_tokens.png'}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
