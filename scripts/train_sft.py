"""Supervised fine-tuning for SpikeRuGPT with assistant-only loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig, RWKV_HEAD_QK_DIM  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


IGNORE_INDEX = -100


@dataclass
class SftExample:
    ids: list[int]
    mask: list[bool]
    source: str


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        row = {"event": event, "time": datetime.now(timezone.utc).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def heartbeat(message: str) -> None:
    print(message, flush=True)


def role_header(role: str) -> str:
    if role == "system":
        return "Система:\n"
    if role == "user":
        return "Инструкция:\n"
    if role == "assistant":
        return "Ответ:\n"
    return f"{role}:\n"


def encode_sft_row(row: dict[str, Any], sp, ctx_len: int) -> SftExample | None:
    ids: list[int] = []
    mask: list[bool] = []
    eos_id = int(sp.eos_id())
    has_assistant = False
    source = str(row.get("source") or row.get("dataset") or "unknown")

    for message in row.get("messages", []):
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            continue
        prefix_ids = sp.encode(role_header(role), out_type=int)
        ids.extend(prefix_ids)
        mask.extend([False] * len(prefix_ids))

        content_ids = sp.encode(content, out_type=int)
        ids.extend(content_ids)
        is_assistant = role == "assistant"
        mask.extend([is_assistant] * len(content_ids))
        if is_assistant:
            has_assistant = True
            ids.append(eos_id)
            mask.append(True)
        else:
            sep_ids = sp.encode("\n\n", out_type=int)
            ids.extend(sep_ids)
            mask.extend([False] * len(sep_ids))

    if not has_assistant or len(ids) < 2:
        return None
    if len(ids) > ctx_len + 1:
        return None
    if sum(mask[1:]) == 0:
        return None
    return SftExample(ids=ids, mask=mask, source=source)


def load_examples(path: Path, tokenizer_path: Path, ctx_len: int) -> tuple[list[SftExample], dict[str, int]]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    examples = []
    counters = {"rows": 0, "encoded": 0, "skipped": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            counters["rows"] += 1
            row = json.loads(line)
            item = encode_sft_row(row, sp, ctx_len)
            if item is None:
                counters["skipped"] += 1
                continue
            counters["encoded"] += 1
            examples.append(item)
    return examples, counters


def split_examples(
    examples: list[SftExample],
    *,
    seed: int,
    val_fraction: float,
    max_train_examples: int | None,
    max_val_examples: int | None,
) -> tuple[list[SftExample], list[SftExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    if max_train_examples is not None:
        train = train[:max_train_examples]
    if max_val_examples is not None:
        val = val[:max_val_examples]
    return train, val


def collate_batch(items: list[SftExample]) -> tuple[torch.Tensor, torch.Tensor, int]:
    max_len = max(len(item.ids) for item in items)
    x = torch.zeros((len(items), max_len - 1), dtype=torch.long)
    y = torch.full((len(items), max_len - 1), IGNORE_INDEX, dtype=torch.long)
    supervised_tokens = 0
    for row, item in enumerate(items):
        ids = item.ids
        mask = item.mask
        x[row, : len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        label_mask = torch.tensor(mask[1:], dtype=torch.bool)
        supervised_tokens += int(label_mask.sum().item())
        y[row, : len(ids) - 1] = torch.where(label_mask, labels, torch.full_like(labels, IGNORE_INDEX))
    return x, y, supervised_tokens


def make_dynamic_batches(
    examples: list[SftExample],
    *,
    rng: random.Random,
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[list[SftExample]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    batches: list[list[SftExample]] = []
    batch: list[SftExample] = []
    max_len = 0
    for item in shuffled:
        item_len = len(item.ids) - 1
        next_max_len = max(max_len, item_len)
        would_exceed_size = len(batch) >= max_batch_size
        would_exceed_tokens = bool(batch) and next_max_len * (len(batch) + 1) > max_batch_tokens
        if would_exceed_size or would_exceed_tokens:
            batches.append(batch)
            batch = []
            max_len = 0
            next_max_len = item_len
        batch.append(item)
        max_len = next_max_len
    if batch:
        batches.append(batch)
    return batches


def forward_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    idx = idx.to(model.emb.weight.device)
    model.step += 1
    _, time_steps = idx.size()
    assert time_steps <= model.ctx_len, "Cannot forward, because len(input) > model ctx_len."
    x = model.atan(model.emb(idx))
    x = model.blocks(x)
    x = model.ln_out(x)
    if RWKV_HEAD_QK_DIM > 0:
        q = model.head_q(x)[:, :time_steps, :]
        k = model.head_k(x)[:, :time_steps, :]
        c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
        c = c.masked_fill(model.copy_mask[:time_steps, :time_steps] == 0, 0)
        return model.head(x) + c
    return model.head(x)


def load_model(checkpoint_path: Path, ctx_len: int) -> tuple[GPT, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = GPT(
        GPTConfig(
            int(config.get("vocab_size", 32000)),
            ctx_len,
            model_type="RWKV",
            n_layer=int(config.get("n_layer", 12)),
            n_embd=int(config.get("n_embd", 512)),
        )
    ).cuda()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


@torch.no_grad()
def evaluate(
    model: GPT,
    examples: list[SftExample],
    batch_size: int,
    max_batch_tokens: int,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    eval_batches = make_dynamic_batches(
        examples,
        rng=random.Random(0),
        max_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    for items in eval_batches:
        x_cpu, y_cpu, supervised_tokens = collate_batch(items)
        if supervised_tokens == 0:
            continue
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = forward_logits(model, x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.cuda().reshape(-1), ignore_index=IGNORE_INDEX)
        functional.reset_net(model)
        total_loss += float(loss.item()) * supervised_tokens
        total_tokens += supervised_tokens
        batches += 1
    mean_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "supervised_tokens": total_tokens,
        "batches": batches,
        "examples": len(examples),
    }


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    epoch: int,
    examples_seen: int,
    best_val_loss: float,
    final: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "step": step,
            "epoch": epoch,
            "examples_seen": examples_seen,
            "best_val_loss": best_val_loss,
            "final": final,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "format": "spikerugpt_sft",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt")
    parser.add_argument("--sft-data", default="data/sft/spikerugpt_sft_clean_final.jsonl")
    parser.add_argument("--tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--run-id", default="sft-step43674-v1")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=18000)
    parser.add_argument("--max-eval-batch-tokens", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--betas", default="0.9,0.99")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/sft/{args.run_id}")
    report_path = Path(args.report or f"reports/sft/{args.run_id}.json")
    metrics_path = Path(args.metrics_jsonl or f"reports/sft/{args.run_id}.metrics.jsonl")
    metrics = JsonlLogger(metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    heartbeat(f"stage=sft_start run_id={args.run_id}")
    examples, counters = load_examples(Path(args.sft_data), Path(args.tokenizer), args.ctx_len)
    train_examples, val_examples = split_examples(
        examples,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
    )
    source_counts: dict[str, int] = {}
    for item in train_examples:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    config = vars(args) | {
        "checkpoint_dir": str(checkpoint_dir),
        "report": str(report_path),
        "metrics_jsonl": str(metrics_path),
        "encoded_counters": counters,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_source_counts": source_counts,
    }
    report_path.write_text(json.dumps({"status": "running", "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics.write("run_started", config=config)

    model, base_checkpoint = load_model(Path(args.base_checkpoint), args.ctx_len)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(float(x) for x in args.betas.split(",")),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    base_info = {
        "base_step": int(base_checkpoint.get("step", 0)),
        "base_tokens_seen": int(base_checkpoint.get("tokens_seen", 0)),
        "base_saved_at": base_checkpoint.get("saved_at"),
    }
    metrics.write("base_loaded", **base_info)

    best_val_loss = float("inf")
    step = 0
    examples_seen = 0
    losses: list[float] = []
    started = time.monotonic()
    progress = None
    if not args.no_progress_bar:
        try:
            from tqdm import tqdm

            total_steps = math.ceil(len(train_examples) / max(1, args.batch_size)) * args.epochs
            progress = tqdm(total=total_steps, desc=args.run_id, dynamic_ncols=True)
        except Exception as exc:
            heartbeat(f"stage=sft progress_bar=0 error={type(exc).__name__}")

    try:
        for epoch in range(args.epochs):
            rng = random.Random(args.seed + epoch)
            epoch_batches = make_dynamic_batches(
                train_examples,
                rng=rng,
                max_batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            if progress is not None:
                progress.total = len(epoch_batches) * args.epochs
                progress.refresh()
            for items in epoch_batches:
                x_cpu, y_cpu, supervised_tokens = collate_batch(items)
                if supervised_tokens == 0:
                    continue
                x = x_cpu.cuda(non_blocking=True)
                y = y_cpu.cuda(non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = forward_logits(model, x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.cuda().reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                functional.reset_net(model)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite SFT loss at step {step + 1}: {loss.item()}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                step += 1
                examples_seen += len(items)
                loss_value = float(loss.item())
                losses.append(loss_value)
                elapsed = max(time.monotonic() - started, 1e-9)
                recent = losses[-args.log_every :]
                tok_per_sec = examples_seen / elapsed
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        avg=f"{sum(recent) / len(recent):.4f}",
                        mem_gb=f"{peak_mem_gb:.1f}",
                        refresh=False,
                    )

                if step == 1 or step % args.log_every == 0:
                    avg_recent = sum(recent) / len(recent)
                    heartbeat(
                        f"stage=sft step={step} epoch={epoch + 1} loss={loss_value:.6f} "
                        f"avg_recent={avg_recent:.6f} grad_norm={float(grad_norm):.4f} "
                        f"examples_seen={examples_seen} ex/s={tok_per_sec:.2f} mem_gb={peak_mem_gb:.2f}"
                    )
                    metrics.write(
                        "train_step",
                        step=step,
                        epoch=epoch + 1,
                        loss=loss_value,
                        avg_recent=avg_recent,
                        grad_norm=float(grad_norm),
                        examples_seen=examples_seen,
                        examples_per_sec=tok_per_sec,
                        peak_mem_gb=peak_mem_gb,
                        supervised_tokens=supervised_tokens,
                    )

                if step % args.eval_every == 0:
                    val = evaluate(model, val_examples, args.eval_batch_size, args.max_eval_batch_tokens)
                    metrics.write("validation", step=step, **val)
                    heartbeat(f"stage=sft_eval step={step} val_loss={val['loss']:.6f} val_ppl={val['ppl']:.2f}")
                    if float(val["loss"]) < best_val_loss:
                        best_val_loss = float(val["loss"])
                        save_checkpoint(
                            checkpoint_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            step=step,
                            epoch=epoch + 1,
                            examples_seen=examples_seen,
                            best_val_loss=best_val_loss,
                            final=False,
                        )
                        metrics.write("checkpoint_saved", checkpoint=str(checkpoint_dir / "best.pt"), step=step, best=True)

                if step % args.save_every == 0:
                    save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=step,
                        epoch=epoch + 1,
                        examples_seen=examples_seen,
                        best_val_loss=best_val_loss,
                        final=False,
                    )
                    metrics.write("checkpoint_saved", checkpoint=str(checkpoint_dir / "latest.pt"), step=step, best=False)
    finally:
        if progress is not None:
            progress.close()

    final_val = evaluate(model, val_examples, args.eval_batch_size, args.max_eval_batch_tokens)
    if float(final_val["loss"]) < best_val_loss:
        best_val_loss = float(final_val["loss"])
        save_checkpoint(
            checkpoint_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            step=step,
            epoch=args.epochs,
            examples_seen=examples_seen,
            best_val_loss=best_val_loss,
            final=False,
        )
    save_checkpoint(
        checkpoint_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        epoch=args.epochs,
        examples_seen=examples_seen,
        best_val_loss=best_val_loss,
        final=True,
    )
    save_checkpoint(
        checkpoint_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        epoch=args.epochs,
        examples_seen=examples_seen,
        best_val_loss=best_val_loss,
        final=True,
    )

    summary = {
        "status": "ok",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base": base_info,
        "train": {
            "step": step,
            "epochs": args.epochs,
            "examples_seen": examples_seen,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "avg_last_50_loss": sum(losses[-50:]) / len(losses[-50:]) if losses else None,
            "elapsed_sec": time.monotonic() - started,
            "examples_per_sec": examples_seen / max(time.monotonic() - started, 1e-9),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "latest_checkpoint": str(checkpoint_dir / "latest.pt"),
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "final_checkpoint": str(checkpoint_dir / "final.pt"),
        },
        "validation": final_val,
        "config": config,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics.write("train_done", **summary["train"], validation=final_val)
    heartbeat(
        f"stage=sft_done step={step} final_loss={summary['train']['final_loss']:.6f} "
        f"val_loss={final_val['loss']:.6f} val_ppl={final_val['ppl']:.2f}"
    )


if __name__ == "__main__":
    main()
