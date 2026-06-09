"""Run a fixed-batch overfit smoke test on the tokenized SpikeRuGPT data."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


def build_batch(data: np.ndarray, *, batch_size: int, ctx_len: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    stride = ctx_len + 1
    for i in range(batch_size):
        start = offset + i * stride
        seq = data[start : start + stride].astype(np.int64)
        if len(seq) != stride:
            raise ValueError("Not enough data to build fixed batch")
        xs.append(seq[:-1])
        ys.append(seq[1:])
    x = torch.tensor(np.stack(xs), dtype=torch.long)
    y = torch.tensor(np.stack(ys), dtype=torch.long)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pretrain_smoke.npy")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--ctx-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="checkpoints/smoke-overfit.pth")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    data = np.load(args.data, mmap_mode="r")
    x_cpu, y_cpu = build_batch(data, batch_size=args.batch_size, ctx_len=args.ctx_len, offset=0)
    x = x_cpu.cuda(non_blocking=True)
    y = y_cpu.cuda(non_blocking=True)

    config = GPTConfig(
        args.vocab_size,
        args.ctx_len,
        model_type="RWKV",
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    model = GPT(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

    initial_loss = None
    started_at = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        functional.reset_net(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        value = float(loss.item())
        if initial_loss is None:
            initial_loss = value
        print(
            f"step={step:04d} loss={value:.6f} "
            f"delta={value - initial_loss:+.6f} "
            f"elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
