"""
Spike Sparsity Analysis
=======================
Loads the latest SpikeGPT checkpoint, runs 100 random batches through the model,
and measures the firing rate (fraction of active spikes) for every LIF neuron
in every Block.

Run from the repo root:
    python analysis/spike_sparsity.py

Requires a CUDA-capable GPU and at least one checkpoint in checkpoints/.
"""

import sys
import os
import json
import glob
import math

# Allow imports from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU.")
    sys.exit(1)


# ── locate latest checkpoint (numeric sort, same logic as train.py) ────────────
checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
checkpoint_files = sorted(
    glob.glob(os.path.join(checkpoint_dir, "spikegpt-ru-*.pth")),
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[-1]),
)
if not checkpoint_files:
    print(f"ERROR: No checkpoints found in {checkpoint_dir}")
    sys.exit(1)

latest_ckpt = checkpoint_files[-1]
print(f"Using checkpoint: {latest_ckpt}")


# ── locate training data ───────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path_full = os.path.join(REPO_ROOT, "data", "ru_train_full.npy")
data_path_base = os.path.join(REPO_ROOT, "data", "ru_train.npy")

if os.path.exists(data_path_full):
    data_path = data_path_full
elif os.path.exists(data_path_base):
    data_path = data_path_base
else:
    # Fallback: look in REPO_ROOT itself (some setups put .npy files there)
    data_path_full_root = os.path.join(REPO_ROOT, "ru_train_full.npy")
    data_path_base_root = os.path.join(REPO_ROOT, "ru_train.npy")
    if os.path.exists(data_path_full_root):
        data_path = data_path_full_root
    elif os.path.exists(data_path_base_root):
        data_path = data_path_base_root
    else:
        print("ERROR: No training data found. Looked for:")
        print(f"  {data_path_full}")
        print(f"  {data_path_base}")
        sys.exit(1)

print(f"Using data   : {data_path}")


# ── model config (must match the checkpoint) ───────────────────────────────────
os.environ["VOCAB_SIZE"] = "50258"

CTX_LEN   = 1024
N_LAYER   = 12
N_EMBD    = 512
MODEL_TYPE = "RWKV"
VOCAB_SIZE = 50258

N_BATCHES  = 100
BATCH_SIZE = 4   # small: we only need statistics, not training throughput


# ── import model & functional reset ───────────────────────────────────────────
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional


# ── build & load model ─────────────────────────────────────────────────────────
print("Building model …")
config = GPTConfig(
    VOCAB_SIZE, CTX_LEN,
    model_type=MODEL_TYPE,
    n_layer=N_LAYER,
    n_embd=N_EMBD,
)
model = GPT(config).cuda()
model.eval()

state_dict = torch.load(latest_ckpt, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
print("Checkpoint loaded.\n")


# ── register forward hooks ─────────────────────────────────────────────────────
# We collect the *output* of each LIF node.  The output is a binary spike tensor
# of shape [T_steps, B, C] (time-major, as produced by LearnableLIFNode).

spike_accum: dict[str, list] = {}   # hook_name -> list of per-batch mean firing rates

def make_hook(name: str):
    def hook(module, input, output):
        # output: [T_steps, B, C]  (LearnableLIFNode) or [T, B, C]
        # firing rate = fraction of 1s over all elements
        rate = output.detach().float().mean().item()
        if name not in spike_accum:
            spike_accum[name] = []
        spike_accum[name].append(rate)
    return hook

hooks = []
# Identify all LIF nodes inside Block instances
for block_idx, block in enumerate(model.blocks):
    for lif_name in ("lif1", "lif2"):
        lif_node = getattr(block, lif_name, None)
        if lif_node is None:
            continue
        hook_name = f"block_{block_idx}.{lif_name}"
        h = lif_node.register_forward_hook(make_hook(hook_name))
        hooks.append(h)

print(f"Registered {len(hooks)} forward hooks on LIF nodes.\n")


# ── prepare random batches ─────────────────────────────────────────────────────
print(f"Loading data from {data_path} …")
data = np.load(data_path, mmap_mode="r")
total_tokens = len(data)
print(f"  Total tokens in dataset: {total_tokens:,}")

rng = np.random.default_rng(seed=42)

def sample_batch(batch_size: int, ctx_len: int) -> torch.Tensor:
    """Sample a random batch of token sequences."""
    max_start = total_tokens - ctx_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    seqs = np.stack([data[s : s + ctx_len].astype(np.int64) for s in starts])
    return torch.from_numpy(seqs).cuda()


# ── inference loop ─────────────────────────────────────────────────────────────
print(f"Running {N_BATCHES} batches (batch_size={BATCH_SIZE}, ctx_len={CTX_LEN}) …")

with torch.no_grad():
    for batch_idx in range(N_BATCHES):
        x = sample_batch(BATCH_SIZE, CTX_LEN)
        _ = model(x)                       # forward; hooks collect spike outputs
        functional.reset_net(model)        # reset LIF membrane potentials

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{N_BATCHES}")

# ── remove hooks ──────────────────────────────────────────────────────────────
for h in hooks:
    h.remove()

print()


# ── compute statistics & print table ──────────────────────────────────────────
results = {}  # layer_id -> {lif1: {mean, std}, lif2: {mean, std}}

n_layers = len(model.blocks)
for block_idx in range(n_layers):
    results[block_idx] = {}
    for lif_name in ("lif1", "lif2"):
        key = f"block_{block_idx}.{lif_name}"
        rates = spike_accum.get(key, [])
        if rates:
            mean_rate = float(np.mean(rates))
            std_rate  = float(np.std(rates))
        else:
            mean_rate = float("nan")
            std_rate  = float("nan")
        results[block_idx][lif_name] = {"mean": mean_rate, "std": std_rate}

# Also collect learnable tau/threshold if available
for block_idx, block in enumerate(model.blocks):
    for lif_name in ("lif1", "lif2"):
        lif_node = getattr(block, lif_name, None)
        if lif_node is None or not hasattr(lif_node, "tau"):
            continue
        tau_val = lif_node.tau.item()
        thr_val = lif_node.threshold.item()
        results[block_idx][lif_name]["tau"]       = tau_val
        results[block_idx][lif_name]["threshold"] = thr_val

# Print header
header = (
    f"{'Layer':>5} | "
    f"{'LIF1 rate':>12} {'±':>2} {'std':>8} | "
    f"{'LIF2 rate':>12} {'±':>2} {'std':>8} | "
    f"{'tau1':>6} {'thr1':>6} | "
    f"{'tau2':>6} {'thr2':>6}"
)
separator = "-" * len(header)
print(separator)
print(header)
print(separator)

for layer_id in range(n_layers):
    r = results[layer_id]
    l1 = r.get("lif1", {})
    l2 = r.get("lif2", {})

    def fmt(v):
        return f"{v:.4f}" if v == v else " nan  "  # nan check

    tau1 = l1.get("tau", float("nan"))
    tau2 = l2.get("tau", float("nan"))
    thr1 = l1.get("threshold", float("nan"))
    thr2 = l2.get("threshold", float("nan"))

    print(
        f"{layer_id:>5} | "
        f"{fmt(l1.get('mean', float('nan'))):>12} {'±':>2} {fmt(l1.get('std', float('nan'))):>8} | "
        f"{fmt(l2.get('mean', float('nan'))):>12} {'±':>2} {fmt(l2.get('std', float('nan'))):>8} | "
        f"{tau1:>6.3f} {thr1:>6.3f} | "
        f"{tau2:>6.3f} {thr2:>6.3f}"
    )

print(separator)
print()

# Compute overall statistics
all_means_lif1 = [results[i]["lif1"]["mean"] for i in range(n_layers) if "lif1" in results[i]]
all_means_lif2 = [results[i]["lif2"]["mean"] for i in range(n_layers) if "lif2" in results[i]]
overall_mean = np.nanmean(all_means_lif1 + all_means_lif2)
print(f"Overall mean firing rate (all LIF nodes): {overall_mean:.4f}")
print(f"  → {overall_mean*100:.1f}% spikes active  |  {(1-overall_mean)*100:.1f}% silent (sparse)")
print()


# ── save to JSON ───────────────────────────────────────────────────────────────
output_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "spike_stats.json")

payload = {
    "checkpoint": latest_ckpt,
    "data_file":  data_path,
    "n_batches":  N_BATCHES,
    "batch_size": BATCH_SIZE,
    "ctx_len":    CTX_LEN,
    "overall_mean_firing_rate": float(overall_mean),
    "layers": results,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {output_path}")
