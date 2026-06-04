"""Benchmark SpikeGPT training throughput for batch / precision choices."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from scripts.run_autonomous_training import BinShardBatcher  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


def build_model(vocab_size: int, ctx_len: int, n_layer: int, n_embd: int) -> GPT:
    config = GPTConfig(vocab_size, ctx_len, model_type="RWKV", n_layer=n_layer, n_embd=n_embd)
    return GPT(config).cuda()


def load_model_checkpoint(path: str, model: GPT) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)


def run_case(args: argparse.Namespace, batch_size: int, precision: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    np.random.seed(123)
    torch.manual_seed(123)

    batcher = BinShardBatcher(args.manifest, args.ctx_len, batch_size)
    model = build_model(args.vocab_size, args.ctx_len, args.n_layer, args.n_embd)
    load_model_checkpoint(args.checkpoint, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

    autocast_enabled = precision == "bf16"
    losses = []
    tokens = 0
    started = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            x_cpu, y_cpu = batcher.next_batch(batch_size)
            x = x_cpu.cuda(non_blocking=True)
            y = y_cpu.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = model(x, y)
            functional.reset_net(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens += batch_size * args.ctx_len
            losses.append(float(loss.item()))
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"case=batch{batch_size}-{precision} step={step}/{args.steps} "
                f"loss={losses[-1]:.6f} grad_norm={float(grad_norm):.4f} "
                f"tok/s={tokens / elapsed:.0f} "
                f"peak_mem_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}",
                flush=True,
            )
        elapsed = time.monotonic() - started
        return {
            "batch_size": batch_size,
            "precision": precision,
            "ok": True,
            "steps": args.steps,
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens / max(elapsed, 1e-9),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "avg_loss": sum(losses) / len(losses),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "batch_size": batch_size,
            "precision": precision,
            "ok": False,
            "error": "oom",
            "detail": str(exc),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    except Exception as exc:
        return {
            "batch_size": batch_size,
            "precision": precision,
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        }
    finally:
        del model, optimizer, batcher
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-12h/final.pt")
    parser.add_argument("--out", default="reports/speed_probe_ctx1024_74m.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batches", default="16,18,20,22,24")
    parser.add_argument("--precisions", default="fp32,bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    results = []
    for precision in [item.strip() for item in args.precisions.split(",") if item.strip()]:
        for batch_size in [int(item) for item in args.batches.split(",") if item.strip()]:
            print(f"\n=== case batch={batch_size} precision={precision} ===", flush=True)
            result = run_case(args, batch_size, precision)
            results.append(result)
            print("result", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if not result.get("ok") and result.get("error") == "oom":
                print(f"stopping larger batches for precision={precision} after OOM", flush=True)
                break

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
