"""Short SpikeGPT pilot training run on tokenized .bin shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


class BinShardBatcher:
    def __init__(self, manifest_path: str | os.PathLike[str], ctx_len: int, batch_size: int):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.dtype = np.dtype(manifest["dtype"])
        self.shards: List[np.memmap] = []
        self.lengths: List[int] = []
        for shard in manifest["shards"]:
            arr = np.memmap(shard["path"], dtype=self.dtype, mode="r")
            if len(arr) > ctx_len + 1:
                self.shards.append(arr)
                self.lengths.append(len(arr))
        if not self.shards:
            raise ValueError("No usable shards in manifest")
        weights = np.asarray(self.lengths, dtype=np.float64)
        self.weights = weights / weights.sum()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for _ in range(self.batch_size):
            shard_index = int(np.random.choice(len(self.shards), p=self.weights))
            arr = self.shards[shard_index]
            start = np.random.randint(0, len(arr) - self.ctx_len - 1)
            seq = np.asarray(arr[start : start + self.ctx_len + 1], dtype=np.int64)
            xs.append(seq[:-1])
            ys.append(seq[1:])
        x = torch.tensor(np.stack(xs), dtype=torch.long)
        y = torch.tensor(np.stack(ys), dtype=torch.long)
        return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json",
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default="checkpoints/pilot-300m-smoke.pth")
    parser.add_argument("--report", default="reports/pilot_300m_smoke.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this pilot")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    batcher = BinShardBatcher(args.manifest, args.ctx_len, args.batch_size)
    config = GPTConfig(
        args.vocab_size,
        args.ctx_len,
        model_type="RWKV",
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    model = GPT(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

    losses = []
    tokens_seen = 0
    started_at = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        x_cpu, y_cpu = batcher.next_batch()
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        functional.reset_net(model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        tokens_seen += args.batch_size * args.ctx_len

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            recent = losses[-args.log_every :]
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"step={step:05d}/{args.steps} "
                f"loss={loss_value:.6f} "
                f"avg_recent={sum(recent) / len(recent):.6f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"tok/s={tokens_seen / elapsed:,.0f} "
                f"peak_mem_gb={mem_gb:.2f}",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    report = {
        "manifest": args.manifest,
        "vocab_size": args.vocab_size,
        "ctx_len": args.ctx_len,
        "batch_size": args.batch_size,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "steps": args.steps,
        "lr": args.lr,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.monotonic() - started_at,
        "tokens_per_sec": tokens_seen / max(time.monotonic() - started_at, 1e-9),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "checkpoint": str(out_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
