"""Autonomous SpikeGPT training runner for local tokenized shards.

This runner is intentionally independent from the legacy train.py path. It
reads the manifest produced by scripts/data/build_pretrain_shards.py, trains
from .bin shards, emits heartbeat lines for Codex /ps, runs small validation
checks, and optionally uploads reports/checkpoints to a private Hugging Face
repo.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


@dataclass
class RunConfig:
    run_id: str
    manifest: str
    tokenizer: str
    validation_dir: str
    vocab_size: int
    ctx_len: int
    n_layer: int
    n_embd: int
    precision: str
    batch_size: int | None
    batch_candidates: list[int]
    probe_steps: int
    max_wall_time_sec: int
    min_steps: int
    max_steps: int
    lr: float
    log_every: int
    save_every_sec: int
    eval_splits: list[str]
    eval_batches: int
    eval_batch_size: int
    checkpoint_dir: str
    report: str
    metrics_jsonl: str
    hf_repo_id: str | None
    hf_private: bool
    upload_checkpoints: bool
    dry_run: bool
    resume_from: str | None
    progress_bar: bool


class BinShardBatcher:
    def __init__(self, manifest_path: str | os.PathLike[str], ctx_len: int, batch_size: int):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.manifest = manifest
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.dtype = np.dtype(manifest["dtype"])
        self.shards: list[np.memmap] = []
        self.lengths: list[int] = []
        for shard in manifest["shards"]:
            arr = np.memmap(shard["path"], dtype=self.dtype, mode="r")
            if len(arr) > ctx_len + 1:
                self.shards.append(arr)
                self.lengths.append(len(arr))
        if not self.shards:
            raise ValueError("No usable shards in manifest")
        weights = np.asarray(self.lengths, dtype=np.float64)
        self.weights = weights / weights.sum()

    def next_batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        size = batch_size or self.batch_size
        xs = []
        ys = []
        for _ in range(size):
            shard_index = int(np.random.choice(len(self.shards), p=self.weights))
            arr = self.shards[shard_index]
            start = np.random.randint(0, len(arr) - self.ctx_len - 1)
            seq = np.asarray(arr[start : start + self.ctx_len + 1], dtype=np.int64)
            xs.append(seq[:-1])
            ys.append(seq[1:])
        x = torch.tensor(np.stack(xs), dtype=torch.long)
        y = torch.tensor(np.stack(ys), dtype=torch.long)
        return x, y


class JsonlLogger:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def heartbeat(message: str) -> None:
    print(message, flush=True)


def ensure_environment(args: RunConfig, metrics: JsonlLogger) -> None:
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "manifest_exists": Path(args.manifest).exists(),
        "tokenizer_exists": Path(args.tokenizer).exists(),
        "validation_dir_exists": Path(args.validation_dir).exists(),
    }
    if torch.cuda.is_available():
        checks.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            }
        )
    metrics.write("environment_gate", **checks)
    heartbeat(
        "stage=environment_gate "
        + " ".join(f"{key}={value}" for key, value in checks.items())
    )
    failed = [key for key, value in checks.items() if key.endswith("_exists") and not value]
    if not checks["cuda_available"]:
        failed.append("cuda_available")
    if failed:
        raise RuntimeError(f"Environment gate failed: {', '.join(failed)}")


def build_model(args: RunConfig) -> GPT:
    config = GPTConfig(
        args.vocab_size,
        args.ctx_len,
        model_type="RWKV",
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    return GPT(config).cuda()


def use_bf16(args: RunConfig) -> bool:
    return args.precision == "bf16"


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_batch_probe(args: RunConfig, metrics: JsonlLogger) -> int:
    if args.batch_size:
        heartbeat(f"stage=batch_probe selected_batch={args.batch_size} mode=fixed")
        metrics.write("batch_probe_fixed", selected_batch=args.batch_size)
        return args.batch_size

    selected = None
    probe_error = None
    for candidate in args.batch_candidates:
        batcher = None
        model = None
        optimizer = None
        x_cpu = None
        y_cpu = None
        x = None
        y = None
        loss = None
        free_cuda()
        try:
            batcher = BinShardBatcher(args.manifest, args.ctx_len, candidate)
            model = build_model(args)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
            started = time.monotonic()
            last_loss = None
            for step in range(1, args.probe_steps + 1):
                x_cpu, y_cpu = batcher.next_batch(candidate)
                x = x_cpu.cuda(non_blocking=True)
                y = y_cpu.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16(args)):
                    loss = model(x, y)
                functional.reset_net(model)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite probe loss: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.item())
            elapsed = max(time.monotonic() - started, 1e-9)
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            status = {
                "candidate": candidate,
                "ok": True,
                "loss": last_loss,
                "peak_mem_gb": peak_mem_gb,
                "tok_per_sec": candidate * args.ctx_len * args.probe_steps / elapsed,
            }
            heartbeat(
                f"stage=batch_probe candidate={candidate} ok=1 "
                f"loss={last_loss:.6f} peak_mem_gb={peak_mem_gb:.2f} "
                f"tok/s={status['tok_per_sec']:.0f}"
            )
            metrics.write("batch_probe_candidate", **status)
            if peak_mem_gb <= 28.0:
                selected = candidate
            else:
                break
        except torch.cuda.OutOfMemoryError as exc:
            probe_error = f"OOM at batch {candidate}: {exc}"
            heartbeat(f"stage=batch_probe candidate={candidate} ok=0 error=oom")
            metrics.write("batch_probe_candidate", candidate=candidate, ok=False, error="oom")
            break
        except Exception as exc:
            probe_error = f"{type(exc).__name__} at batch {candidate}: {exc}"
            heartbeat(f"stage=batch_probe candidate={candidate} ok=0 error={type(exc).__name__}")
            metrics.write(
                "batch_probe_candidate",
                candidate=candidate,
                ok=False,
                error=type(exc).__name__,
                detail=str(exc),
            )
            break
        finally:
            del model, optimizer, batcher, x, y, loss, x_cpu, y_cpu
            free_cuda()

    if selected is None:
        selected = 4
        heartbeat(f"stage=batch_probe selected_batch={selected} mode=fallback error={probe_error!r}")
    else:
        heartbeat(f"stage=batch_probe selected_batch={selected} mode=auto")
    metrics.write("batch_probe_selected", selected_batch=selected, error=probe_error)
    return selected


def save_training_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    args: RunConfig,
    step: int,
    tokens_seen: int,
    losses: list[float],
    final: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(args),
            "step": step,
            "tokens_seen": tokens_seen,
            "losses": losses,
            "final": final,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def load_checkpoint_for_resume(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, list[float]]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint.get("step", 0)), int(checkpoint.get("tokens_seen", 0)), list(checkpoint.get("losses", []))
    model.load_state_dict(checkpoint)
    return 0, 0, []


def train_wall_time(args: RunConfig, batch_size: int, metrics: JsonlLogger) -> dict[str, Any]:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"
    final_path = checkpoint_dir / "final.pt"
    batcher = BinShardBatcher(args.manifest, args.ctx_len, batch_size)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

    start_step = 0
    tokens_seen = 0
    losses: list[float] = []
    if latest_path.exists():
        start_step, tokens_seen, losses = load_checkpoint_for_resume(latest_path, model=model, optimizer=optimizer)
        heartbeat(f"stage=train resume=1 step={start_step} tokens_seen={tokens_seen}")
        metrics.write("train_resume", checkpoint=str(latest_path), step=start_step, tokens_seen=tokens_seen)
    elif args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_step, tokens_seen, losses = load_checkpoint_for_resume(resume_path, model=model, optimizer=optimizer)
        heartbeat(f"stage=train resume=1 checkpoint={resume_path} step={start_step} tokens_seen={tokens_seen}")
        metrics.write("train_resume", checkpoint=str(resume_path), step=start_step, tokens_seen=tokens_seen)
    start_tokens_seen = tokens_seen
    run_loss_start_index = len(losses)

    started_at = time.monotonic()
    last_save = started_at
    torch.cuda.reset_peak_memory_stats()
    stop_reason = "max_steps"
    step = start_step
    progress = None
    progress_last_elapsed = 0.0
    if args.progress_bar:
        try:
            from tqdm import tqdm

            progress = tqdm(
                total=args.max_wall_time_sec,
                initial=0,
                desc=args.run_id,
                unit="s",
                dynamic_ncols=True,
                leave=True,
            )
        except Exception as exc:
            heartbeat(f"stage=train progress_bar=0 error={type(exc).__name__}")

    try:
        while step < args.max_steps:
            if step > start_step and step >= args.min_steps and time.monotonic() - started_at >= args.max_wall_time_sec:
                stop_reason = "wall_time"
                break
            step += 1
            x_cpu, y_cpu = batcher.next_batch(batch_size)
            x = x_cpu.cuda(non_blocking=True)
            y = y_cpu.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16(args)):
                loss = model(x, y)
            functional.reset_net(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)
            tokens_seen += batch_size * args.ctx_len
            elapsed = max(time.monotonic() - started_at, 1e-9)
            recent = losses[-args.log_every :]
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            tok_per_sec = (tokens_seen - start_tokens_seen) / elapsed
            eta_sec = max(args.max_wall_time_sec - elapsed, 0)

            if progress is not None:
                target_elapsed = min(elapsed, float(args.max_wall_time_sec))
                progress.update(max(0.0, target_elapsed - progress_last_elapsed))
                progress_last_elapsed = target_elapsed
                progress.set_postfix(
                    step=step,
                    loss=f"{loss_value:.4f}",
                    tok_s=f"{tok_per_sec:.0f}",
                    mem_gb=f"{peak_mem_gb:.1f}",
                    refresh=False,
                )

            if step == 1 or step % args.log_every == 0:
                line = (
                    f"stage=train step={step} loss={loss_value:.6f} "
                    f"avg_recent={sum(recent) / len(recent):.6f} "
                    f"grad_norm={float(grad_norm):.4f} tok/s={tok_per_sec:.0f} "
                    f"peak_mem_gb={peak_mem_gb:.2f} eta_min={eta_sec / 60:.1f}"
                )
                heartbeat(line)
                metrics.write(
                    "train_step",
                    step=step,
                    loss=loss_value,
                    avg_recent=sum(recent) / len(recent),
                    grad_norm=float(grad_norm),
                    tok_per_sec=tok_per_sec,
                    peak_mem_gb=peak_mem_gb,
                    eta_sec=eta_sec,
                    tokens_seen=tokens_seen,
                )

            if time.monotonic() - last_save >= args.save_every_sec:
                save_training_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    tokens_seen=tokens_seen,
                    losses=losses,
                    final=False,
                )
                heartbeat(f"stage=train checkpoint={latest_path} step={step}")
                metrics.write("checkpoint_saved", checkpoint=str(latest_path), step=step, final=False)
                last_save = time.monotonic()
    finally:
        if progress is not None:
            progress.close()

    save_training_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        args=args,
        step=step,
        tokens_seen=tokens_seen,
        losses=losses,
        final=True,
    )
    save_training_checkpoint(
        latest_path,
        model=model,
        optimizer=optimizer,
        args=args,
        step=step,
        tokens_seen=tokens_seen,
        losses=losses,
        final=True,
    )
    elapsed_total = time.monotonic() - started_at
    run_losses = losses[run_loss_start_index:]
    train_summary = {
        "step": step,
        "start_step": start_step,
        "stop_reason": stop_reason,
        "initial_loss": run_losses[0] if run_losses else None,
        "final_loss": run_losses[-1] if run_losses else None,
        "min_loss": min(run_losses) if run_losses else None,
        "history_initial_loss": losses[0] if losses else None,
        "history_min_loss": min(losses) if losses else None,
        "tokens_seen": tokens_seen,
        "run_tokens_seen": tokens_seen - start_tokens_seen,
        "elapsed_sec": elapsed_total,
        "tokens_per_sec": (step - start_step) * batch_size * args.ctx_len / max(elapsed_total, 1e-9),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "checkpoint": str(final_path),
        "latest_checkpoint": str(latest_path),
    }
    final_loss_text = (
        f"{train_summary['final_loss']:.6f}" if train_summary["final_loss"] is not None else "None"
    )
    heartbeat(
        f"stage=train done=1 stop_reason={stop_reason} step={step} "
        f"final_loss={final_loss_text} "
        f"tok/s={train_summary['tokens_per_sec']:.0f}"
    )
    metrics.write("train_done", **train_summary)
    return {"summary": train_summary, "model": model}


def load_sentencepiece(tokenizer_path: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for validation") from exc
    processor = spm.SentencePieceProcessor()
    processor.Load(tokenizer_path)
    return processor


def iter_validation_sequences(
    *,
    validation_dir: Path,
    split: str,
    tokenizer_path: str,
    ctx_len: int,
    max_sequences: int,
) -> Iterable[np.ndarray]:
    sp = load_sentencepiece(tokenizer_path)
    eos_id = int(sp.eos_id())
    path = validation_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    produced = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if produced >= max_sequences:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or ""
            ids = sp.EncodeAsIds(text)
            ids.append(eos_id)
            if len(ids) < ctx_len + 1:
                continue
            for offset in range(0, len(ids) - ctx_len - 1, ctx_len):
                seq = np.asarray(ids[offset : offset + ctx_len + 1], dtype=np.int64)
                if len(seq) == ctx_len + 1:
                    produced += 1
                    yield seq
                    if produced >= max_sequences:
                        break


@torch.no_grad()
def evaluate_validation(args: RunConfig, model: GPT, metrics: JsonlLogger) -> dict[str, Any]:
    model.eval()
    results: dict[str, Any] = {}
    validation_dir = Path(args.validation_dir)
    max_sequences = args.eval_batches * args.eval_batch_size
    for split in args.eval_splits:
        losses = []
        sequences = list(
            iter_validation_sequences(
                validation_dir=validation_dir,
                split=split,
                tokenizer_path=args.tokenizer,
                ctx_len=args.ctx_len,
                max_sequences=max_sequences,
            )
        )
        if not sequences:
            results[split] = {"ok": False, "error": "no_sequences"}
            heartbeat(f"stage=validation split={split} ok=0 error=no_sequences")
            continue
        for index in range(0, len(sequences), args.eval_batch_size):
            batch = sequences[index : index + args.eval_batch_size]
            if len(batch) < args.eval_batch_size:
                break
            arr = np.stack(batch)
            x = torch.tensor(arr[:, :-1], dtype=torch.long, device="cuda")
            y = torch.tensor(arr[:, 1:], dtype=torch.long, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16(args)):
                loss = model(x, y)
            functional.reset_net(model)
            losses.append(float(loss.item()))
        if losses:
            mean_loss = sum(losses) / len(losses)
            result = {
                "ok": True,
                "batches": len(losses),
                "sequences": len(losses) * args.eval_batch_size,
                "loss": mean_loss,
                "ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
            }
            results[split] = result
            heartbeat(
                f"stage=validation split={split} ok=1 "
                f"loss={mean_loss:.6f} ppl={result['ppl']:.2f} batches={len(losses)}"
            )
            metrics.write("validation_split", split=split, **result)
        else:
            results[split] = {"ok": False, "error": "no_full_batches"}
            heartbeat(f"stage=validation split={split} ok=0 error=no_full_batches")
    model.train()
    return results


def verify_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
            model_tensors = len(model_state) if hasattr(model_state, "__len__") else None
            return {"ok": True, "format": "autonomous", "model_tensors": model_tensors}
        if isinstance(checkpoint, dict):
            return {"ok": True, "format": "state_dict", "model_tensors": len(checkpoint)}
        return {"ok": False, "error": "unexpected_checkpoint_type", "type": type(checkpoint).__name__}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def conservative_gate(
    summary: dict[str, Any],
    validation: dict[str, Any],
    checkpoint_validation: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    initial = summary.get("initial_loss")
    final = summary.get("final_loss")
    if initial is None or final is None:
        reasons.append("missing_train_loss")
    elif not final < initial:
        reasons.append(f"loss_not_decreased initial={initial:.6f} final={final:.6f}")
    checkpoint = summary.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        reasons.append("missing_final_checkpoint")
    elif not checkpoint_validation.get("ok"):
        reasons.append(f"checkpoint_not_readable error={checkpoint_validation.get('error')}")
    if summary.get("peak_mem_gb", 0) <= 0:
        reasons.append("missing_gpu_memory_metric")
    ok_validations = [name for name, result in validation.items() if result.get("ok")]
    if not ok_validations:
        reasons.append("no_successful_validation_split")
    return not reasons, reasons


def write_report(args: RunConfig, payload: dict[str, Any]) -> None:
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upload_to_hf(args: RunConfig, files: list[Path], metrics: JsonlLogger) -> dict[str, Any]:
    if not args.hf_repo_id:
        return {"ok": False, "skipped": True, "reason": "hf_repo_id_not_set"}
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        api = HfApi(token=token) if token else HfApi()
        api.create_repo(args.hf_repo_id, repo_type="model", private=args.hf_private, exist_ok=True)
        uploaded = []
        for path in files:
            if not path.exists():
                continue
            if path.suffix in {".pt", ".pth"} and not args.upload_checkpoints:
                continue
            remote_path = f"runs/{args.run_id}/{path.name}"
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote_path,
                repo_id=args.hf_repo_id,
                repo_type="model",
            )
            uploaded.append(remote_path)
            heartbeat(f"stage=backup uploaded={remote_path}")
        result = {"ok": True, "repo_id": args.hf_repo_id, "uploaded": uploaded}
        metrics.write("hf_backup_done", **result)
        return result
    except Exception as exc:
        result = {
            "ok": False,
            "skipped": False,
            "error": type(exc).__name__,
            "detail": str(exc),
        }
        metrics.write("hf_backup_failed", **result)
        heartbeat(f"stage=backup ok=0 error={type(exc).__name__}")
        return result


def parse_args() -> RunConfig:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"autonomous-{timestamp}")
    parser.add_argument("--manifest", default="data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json")
    parser.add_argument("--tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--validation-dir", default="data/validation_text")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-candidates", default="4,8,12,16,24,32")
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--max-wall-time-sec", type=int, default=3 * 60 * 60)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-sec", type=int, default=45 * 60)
    parser.add_argument("--eval-splits", default="val_mixed,val_wiki,val_habr,val_lit")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--hf-repo-id", default=os.environ.get("SPIKERUGPT_HF_RUN_REPO", "Koras1k/spikerugpt-autonomous-runs"))
    parser.add_argument("--hf-public", action="store_true")
    parser.add_argument("--no-upload-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    ns = parser.parse_args()

    checkpoint_dir = ns.checkpoint_dir or f"checkpoints/autonomous/{ns.run_id}"
    report = ns.report or f"reports/{ns.run_id}.json"
    metrics_jsonl = ns.metrics_jsonl or f"reports/{ns.run_id}.metrics.jsonl"
    return RunConfig(
        run_id=ns.run_id,
        manifest=ns.manifest,
        tokenizer=ns.tokenizer,
        validation_dir=ns.validation_dir,
        vocab_size=ns.vocab_size,
        ctx_len=ns.ctx_len,
        n_layer=ns.n_layer,
        n_embd=ns.n_embd,
        precision=ns.precision,
        batch_size=ns.batch_size,
        batch_candidates=[int(item) for item in ns.batch_candidates.split(",") if item.strip()],
        probe_steps=ns.probe_steps,
        max_wall_time_sec=ns.max_wall_time_sec,
        min_steps=ns.min_steps,
        max_steps=ns.max_steps,
        lr=ns.lr,
        log_every=ns.log_every,
        save_every_sec=ns.save_every_sec,
        eval_splits=[item.strip() for item in ns.eval_splits.split(",") if item.strip()],
        eval_batches=ns.eval_batches,
        eval_batch_size=ns.eval_batch_size,
        checkpoint_dir=checkpoint_dir,
        report=report,
        metrics_jsonl=metrics_jsonl,
        hf_repo_id=ns.hf_repo_id,
        hf_private=not ns.hf_public,
        upload_checkpoints=not ns.no_upload_checkpoints,
        dry_run=ns.dry_run,
        resume_from=ns.resume_from,
        progress_bar=ns.progress_bar,
    )


def main() -> None:
    args = parse_args()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    metrics = JsonlLogger(args.metrics_jsonl)
    final_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "config": asdict(args),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_report(args, final_payload)
    metrics.write("run_started", config=asdict(args))
    heartbeat(f"stage=start run_id={args.run_id} precision={args.precision} report={args.report}")

    try:
        ensure_environment(args, metrics)
        if args.dry_run:
            final_payload.update({"status": "dry_run_ok", "finished_at": datetime.now(timezone.utc).isoformat()})
            write_report(args, final_payload)
            heartbeat("stage=done status=dry_run_ok")
            return

        selected_batch = run_batch_probe(args, metrics)
        train_result = train_wall_time(args, selected_batch, metrics)
        validation = evaluate_validation(args, train_result["model"], metrics)
        checkpoint_validation = verify_checkpoint(train_result["summary"]["checkpoint"])
        metrics.write("checkpoint_verified", **checkpoint_validation)
        heartbeat(
            f"stage=checkpoint_verify ok={int(bool(checkpoint_validation.get('ok')))} "
            f"format={checkpoint_validation.get('format')} "
            f"model_tensors={checkpoint_validation.get('model_tensors')}"
        )
        gate_ok, gate_reasons = conservative_gate(train_result["summary"], validation, checkpoint_validation)

        final_payload.update(
            {
                "status": "ok" if gate_ok else "gate_failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "selected_batch": selected_batch,
                "train": train_result["summary"],
                "validation": validation,
                "checkpoint_validation": checkpoint_validation,
                "gate": {"ok": gate_ok, "reasons": gate_reasons},
            }
        )
        write_report(args, final_payload)

        files = [
            Path(args.report),
            Path(args.metrics_jsonl),
            Path(train_result["summary"]["checkpoint"]),
            Path(train_result["summary"]["latest_checkpoint"]),
        ]
        backup = upload_to_hf(args, files, metrics)
        final_payload["backup"] = backup
        write_report(args, final_payload)

        if gate_ok:
            final_loss_text = (
                f"{train_result['summary']['final_loss']:.6f}"
                if train_result["summary"]["final_loss"] is not None
                else "None"
            )
            heartbeat(
                f"stage=done status=ok selected_batch={selected_batch} "
                f"final_loss={final_loss_text} "
                f"report={args.report}"
            )
        else:
            heartbeat(f"stage=done status=gate_failed reasons={';'.join(gate_reasons)} report={args.report}")
            raise SystemExit(2)
    except Exception as exc:
        final_payload.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_report(args, final_payload)
        metrics.write("run_failed", error=type(exc).__name__, detail=str(exc))
        heartbeat(f"stage=done status=failed error={type(exc).__name__} detail={str(exc)} report={args.report}")
        raise


if __name__ == "__main__":
    main()
