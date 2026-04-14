"""
Spike Sparsity Analysis — English Model (ridger/SpikeGPT-OpenWebText-216M)
===========================================================================
Downloads (or re-uses) the 216M English SpikeGPT checkpoint, runs 100 random
batches of synthetic tokens through the model, and measures the firing rate
(fraction of active spikes) for every LIF neuron in every Block.

Run from the repo root:
    python analysis/spike_sparsity_en.py

Requires a CUDA-capable GPU.
"""

import sys
import os
import json
import math

# Allow imports from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU.")
    sys.exit(1)


# ── English model config ───────────────────────────────────────────────────────
os.environ["VOCAB_SIZE"] = "50277"

CTX_LEN    = 1024
N_LAYER    = 18
N_EMBD     = 768
MODEL_TYPE = "RWKV"
VOCAB_SIZE  = 50277

N_BATCHES  = 100
BATCH_SIZE = 4
DATASET_ID = "Skylion007/openwebtext"

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
CKPT_PATH      = os.path.join(CHECKPOINT_DIR, "spikegpt-en-216M.pth")
OUTPUT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_stats_en.json")


# ── Step 1: Download checkpoint if not present ─────────────────────────────────
if not os.path.exists(CKPT_PATH):
    print("Checkpoint not found locally. Downloading from HuggingFace …")
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import os as _os

        HF_TOKEN = _os.environ.get("HF_TOKEN", "") or None

        # First list files to find the actual .pth filename
        api = HfApi()
        repo_files = list(api.list_repo_files(
            "ridger/SpikeGPT-OpenWebText-216M",
            token=HF_TOKEN,
        ))
        print(f"Files in repo: {repo_files}")

        pth_files = [f for f in repo_files if f.endswith(".pth")]
        if not pth_files:
            print("ERROR: No .pth file found in ridger/SpikeGPT-OpenWebText-216M")
            sys.exit(1)

        remote_filename = pth_files[0]
        print(f"Downloading: {remote_filename}")

        downloaded = hf_hub_download(
            repo_id="ridger/SpikeGPT-OpenWebText-216M",
            filename=remote_filename,
            token=HF_TOKEN,
        )

        # Copy to our canonical path
        import shutil
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        shutil.copy(downloaded, CKPT_PATH)
        print(f"Saved to: {CKPT_PATH}")

    except Exception as e:
        print(f"ERROR downloading model: {e}")
        sys.exit(1)
else:
    print(f"Using cached checkpoint: {CKPT_PATH}")


# ── Step 2: Import model & functional reset ────────────────────────────────────
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional


# ── Step 3: Build & load model ─────────────────────────────────────────────────
print("Building English model (n_layer=18, n_embd=768, vocab_size=50277) …")
config = GPTConfig(
    VOCAB_SIZE, CTX_LEN,
    model_type=MODEL_TYPE,
    n_layer=N_LAYER,
    n_embd=N_EMBD,
)
model = GPT(config).cuda()
model.eval()

state_dict = torch.load(CKPT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
if unexpected:
    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
print("Checkpoint loaded.\n")


# ── Step 4: Register forward hooks ────────────────────────────────────────────
spike_accum: dict = {}   # hook_name -> list of per-batch mean firing rates

def make_hook(name: str):
    def hook(module, input, output):
        # output: [T_steps, B, C]  — binary spikes from LearnableLIFNode
        rate = output.detach().float().mean().item()
        if name not in spike_accum:
            spike_accum[name] = []
        spike_accum[name].append(rate)
    return hook

hooks = []
for block_idx, block in enumerate(model.blocks):
    for lif_name in ("lif1", "lif2"):
        lif_node = getattr(block, lif_name, None)
        if lif_node is None:
            continue
        hook_name = f"block_{block_idx}.{lif_name}"
        h = lif_node.register_forward_hook(make_hook(hook_name))
        hooks.append(h)

print(f"Registered {len(hooks)} forward hooks on LIF nodes.\n")


# ── Step 5: Load real English text from OpenWebText2 ──────────────────────────
print(f"Loading tokenizer and dataset ({DATASET_ID}) …")

from transformers import AutoTokenizer
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN", "") or None

# NeoX-style GPT-2 tokenizer (50277 tokens)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)

# Stream dataset to avoid downloading everything
dataset = load_dataset(DATASET_ID, split="train", streaming=True,
                       token=HF_TOKEN)

# Tokenize texts into a flat buffer, then slice into ctx_len chunks
token_buffer = []
NEED_TOKENS = N_BATCHES * BATCH_SIZE * CTX_LEN + CTX_LEN  # a bit extra

print(f"Tokenizing texts until we have {NEED_TOKENS:,} tokens …")
for sample in dataset:
    text = sample.get("text", "") or ""
    if not text.strip():
        continue
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_buffer.extend(ids)
    if len(token_buffer) >= NEED_TOKENS:
        break

token_buffer = token_buffer[:NEED_TOKENS]
print(f"  Got {len(token_buffer):,} tokens from real English text.")

all_tokens = torch.tensor(token_buffer, dtype=torch.long)
# Split into (N_BATCHES * BATCH_SIZE) chunks of CTX_LEN
n_chunks = N_BATCHES * BATCH_SIZE
chunks = all_tokens[:n_chunks * CTX_LEN].view(n_chunks, CTX_LEN)

def get_batch(batch_idx: int) -> torch.Tensor:
    start = batch_idx * BATCH_SIZE
    return chunks[start:start + BATCH_SIZE].cuda()


# ── Step 6: Inference loop ─────────────────────────────────────────────────────
print(f"\nRunning {N_BATCHES} batches (batch_size={BATCH_SIZE}, ctx_len={CTX_LEN}) …")

with torch.no_grad():
    for batch_idx in range(N_BATCHES):
        x = get_batch(batch_idx)
        _ = model(x)                       # forward; hooks collect spike outputs
        functional.reset_net(model)        # reset LIF membrane potentials

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{N_BATCHES}")

# Remove hooks
for h in hooks:
    h.remove()

print()


# ── Step 7: Compute statistics & print table ──────────────────────────────────
results = {}   # layer_id -> {lif1: {mean, std, tau, threshold}, lif2: …}

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

# Collect learnable tau / threshold
for block_idx, block in enumerate(model.blocks):
    for lif_name in ("lif1", "lif2"):
        lif_node = getattr(block, lif_name, None)
        if lif_node is None or not hasattr(lif_node, "tau"):
            continue
        tau_val = lif_node.tau.item()
        thr_val = lif_node.threshold.item()
        results[block_idx][lif_name]["tau"]       = tau_val
        results[block_idx][lif_name]["threshold"] = thr_val

# Print table
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

def fmt(v):
    return f"{v:.4f}" if v == v else " nan  "

for layer_id in range(n_layers):
    r = results[layer_id]
    l1 = r.get("lif1", {})
    l2 = r.get("lif2", {})
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

all_means_lif1 = [results[i]["lif1"]["mean"] for i in range(n_layers) if "lif1" in results[i]]
all_means_lif2 = [results[i]["lif2"]["mean"] for i in range(n_layers) if "lif2" in results[i]]
overall_mean = float(np.nanmean(all_means_lif1 + all_means_lif2))
print(f"Overall mean firing rate (all LIF nodes): {overall_mean:.4f}")
print(f"  → {overall_mean*100:.1f}% spikes active  |  {(1-overall_mean)*100:.1f}% silent (sparse)")
print()


# ── Step 8: Save to JSON ───────────────────────────────────────────────────────
payload = {
    "model":       "ridger/SpikeGPT-OpenWebText-216M",
    "checkpoint":  CKPT_PATH,
    "n_layer":     N_LAYER,
    "n_embd":      N_EMBD,
    "vocab_size":  VOCAB_SIZE,
    "n_batches":   N_BATCHES,
    "batch_size":  BATCH_SIZE,
    "ctx_len":     CTX_LEN,
    "input_type":  "real_text_openwebtext2",
    "overall_mean_firing_rate": overall_mean,
    "layers": results,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {OUTPUT_PATH}")
