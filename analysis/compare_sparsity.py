"""
Spike Firing Rate Comparison: Russian vs English SpikeGPT
==========================================================
Runs (or loads from JSON cache) both sparsity analyses and prints a side-by-side
comparison table, then saves comparison_results.json.

Run from the repo root:
    python analysis/compare_sparsity.py

Options (env vars):
    FORCE_RERUN=1   — ignore JSON caches and rerun both analyses
"""

import sys
import os
import json
import subprocess
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(ANALYSIS_DIR)

RU_JSON  = os.path.join(ANALYSIS_DIR, "spike_stats.json")
EN_JSON  = os.path.join(ANALYSIS_DIR, "spike_stats_en.json")
OUT_JSON = os.path.join(ANALYSIS_DIR, "comparison_results.json")

FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"


# ── helpers ───────────────────────────────────────────────────────────────────

def run_analysis(script: str, label: str) -> None:
    """Run a child analysis script and stream its output."""
    print(f"\n{'='*60}")
    print(f"Running {label} analysis: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        check=True,
    )


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nan_mean(values):
    vals = [v for v in values if v == v]   # filter NaN
    return sum(vals) / len(vals) if vals else float("nan")


# ── run analyses if needed ─────────────────────────────────────────────────────

if FORCE_RERUN or not os.path.exists(RU_JSON):
    run_analysis(os.path.join(ANALYSIS_DIR, "spike_sparsity.py"), "Russian")

if FORCE_RERUN or not os.path.exists(EN_JSON):
    run_analysis(os.path.join(ANALYSIS_DIR, "spike_sparsity_en.py"), "English")


# ── load results ──────────────────────────────────────────────────────────────

ru_data = load_json(RU_JSON)
en_data = load_json(EN_JSON)

ru_layers_raw = ru_data["layers"]
en_layers_raw = en_data["layers"]

# JSON keys are strings; convert to int-keyed dicts
ru_layers = {int(k): v for k, v in ru_layers_raw.items()}
en_layers = {int(k): v for k, v in en_layers_raw.items()}

n_ru = len(ru_layers)   # 12
n_en = len(en_layers)   # 18

ru_overall = ru_data.get("overall_mean_firing_rate", float("nan"))
en_overall = en_data.get("overall_mean_firing_rate", float("nan"))


# ── comparison table ──────────────────────────────────────────────────────────

def fmt(v, width=8):
    if v != v:   # NaN
        return f"{'N/A':>{width}}"
    return f"{v:.4f}".rjust(width)


# We print by relative position so the user can compare apples-to-apples.
# For each model we also show the absolute layer index in brackets.
# Number of rows = max(n_ru, n_en) where we step through 0..max-1 and map
# each i to the nearest layer: floor(i / max * n_model).

N_ROWS = max(n_ru, n_en)   # 18

header_line1 = (
    f"{'':>5}  "
    f"{'Russian (12L-512D)':^27}  "
    f"{'English (18L-768D)':^27}"
)
header_line2 = (
    f"{'Rel':>5}  "
    f"{'[L]':>4} {'LIF1':>8} {'LIF2':>8}  "
    f"{'[L]':>4} {'LIF1':>8} {'LIF2':>8}"
)
separator = "-" * len(header_line2)

print()
print("=" * 60)
print("       SPIKE FIRING RATE COMPARISON")
print("=" * 60)
print(header_line1)
print(header_line2)
print(separator)

# Accumulate per-model means for final summary
ru_lif1_list, ru_lif2_list = [], []
en_lif1_list, en_lif2_list = [], []

comparison_rows = []

for i in range(N_ROWS):
    rel = i / (N_ROWS - 1) if N_ROWS > 1 else 0.0   # 0.0 … 1.0

    # Nearest absolute layer index for each model
    ru_idx = min(round(rel * (n_ru - 1)), n_ru - 1)
    en_idx = i   # English has 18 layers = N_ROWS, direct mapping

    ru_l  = ru_layers.get(ru_idx, {})
    en_l  = en_layers.get(en_idx, {})

    ru_lif1 = ru_l.get("lif1", {}).get("mean", float("nan"))
    ru_lif2 = ru_l.get("lif2", {}).get("mean", float("nan"))
    en_lif1 = en_l.get("lif1", {}).get("mean", float("nan"))
    en_lif2 = en_l.get("lif2", {}).get("mean", float("nan"))

    print(
        f"{rel:>5.2f}  "
        f"[{ru_idx:>2}] {fmt(ru_lif1)} {fmt(ru_lif2)}  "
        f"[{en_idx:>2}] {fmt(en_lif1)} {fmt(en_lif2)}"
    )

    comparison_rows.append({
        "rel_position":    round(rel, 4),
        "ru_layer":        ru_idx,
        "en_layer":        en_idx,
        "ru_lif1_mean":    ru_lif1,
        "ru_lif2_mean":    ru_lif2,
        "en_lif1_mean":    en_lif1,
        "en_lif2_mean":    en_lif2,
    })

    ru_lif1_list.append(ru_lif1)
    ru_lif2_list.append(ru_lif2)
    en_lif1_list.append(en_lif1)
    en_lif2_list.append(en_lif2)

print(separator)

# Per-model overall means (from their own JSON, which covers all unique layers)
print(
    f"{'Mean':>5}  "
    f"{'':>4} {fmt(ru_overall):>8} {'':>8}  "
    f"{'':>4} {fmt(en_overall):>8} {'':>8}"
)
print()


# ── summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("                      SUMMARY")
print("=" * 60)
print(f"  Russian model  overall mean firing rate : {ru_overall:.4f}  "
      f"({ru_overall*100:.1f}% active / {(1-ru_overall)*100:.1f}% silent)")
print(f"  English model  overall mean firing rate : {en_overall:.4f}  "
      f"({en_overall*100:.1f}% active / {(1-en_overall)*100:.1f}% silent)")
print()

if ru_overall == ru_overall and en_overall == en_overall:   # not NaN
    if ru_overall < en_overall:
        higher_sparse  = "Russian"
        lower_sparse   = "English"
        diff_abs  = en_overall - ru_overall
        diff_pct  = diff_abs / en_overall * 100
        sparsity_ru = (1 - ru_overall) * 100
        sparsity_en = (1 - en_overall) * 100
        sparsity_diff = sparsity_ru - sparsity_en
    else:
        higher_sparse  = "English"
        lower_sparse   = "Russian"
        diff_abs  = ru_overall - en_overall
        diff_pct  = diff_abs / ru_overall * 100
        sparsity_ru = (1 - ru_overall) * 100
        sparsity_en = (1 - en_overall) * 100
        sparsity_diff = sparsity_en - sparsity_ru

    print(f"  CONCLUSION: The {higher_sparse} model has HIGHER sparsity.")
    print(f"    Firing rate difference : {diff_abs:.4f} ({diff_pct:.1f}% relative)")
    print(f"    Sparsity Russian       : {sparsity_ru:.1f}%  silent neurons")
    print(f"    Sparsity English       : {sparsity_en:.1f}%  silent neurons")
    print(f"    Sparsity advantage     : {abs(sparsity_diff):.1f} pp in favour of {higher_sparse}")
    print()


# ── save JSON ─────────────────────────────────────────────────────────────────

output = {
    "russian": {
        "model":                "SpikeGPT-RU (12L-512D)",
        "n_layer":              n_ru,
        "n_embd":               512,
        "vocab_size":           50258,
        "overall_mean_firing_rate": ru_overall,
        "sparsity_pct":         (1 - ru_overall) * 100,
    },
    "english": {
        "model":                "ridger/SpikeGPT-OpenWebText-216M (18L-768D)",
        "n_layer":              n_en,
        "n_embd":               768,
        "vocab_size":           50257,
        "overall_mean_firing_rate": en_overall,
        "sparsity_pct":         (1 - en_overall) * 100,
    },
    "comparison_rows": comparison_rows,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Comparison results saved to: {OUT_JSON}")
