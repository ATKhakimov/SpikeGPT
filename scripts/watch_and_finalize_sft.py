"""Watch an SFT run, print hourly status, then evaluate and upload artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": type(exc).__name__, "detail": str(exc)}


def latest_train_step(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    last = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") in {"train_step", "validation", "checkpoint_saved", "train_done"}:
                last = row
    return last


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    log("run: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def generate_samples(args: argparse.Namespace, checkpoint: Path) -> None:
    script = f"""
import json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional

checkpoint_path = Path({str(checkpoint)!r})
tokenizer_path = Path({args.tokenizer!r})
out_json = Path({args.generations_json!r})
out_md = Path({args.generations_md!r})
ctx_len = {args.ctx_len}
length = {args.generation_length}
temperature = {args.temperature}
top_p = {args.top_p}
seed = {args.seed}

prompts = [
    "Объясни простыми словами, что такое нейроморфные вычисления.",
    "Составь краткий план статьи о русскоязычной SpikeGPT-модели.",
    "Почему русский язык сложен для языковых моделей?",
    "Напиши короткое резюме: модель обучалась на русскоязычном корпусе и затем проходила SFT.",
]

def load_model(path):
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("config", {{}})
    model = GPT(GPTConfig(
        int(cfg.get("vocab_size", 32000)),
        ctx_len,
        model_type="RWKV",
        n_layer=int(cfg.get("n_layer", 12)),
        n_embd=int(cfg.get("n_embd", 512)),
    )).cuda().eval()
    model.load_state_dict(ck["model_state"])
    return model

def forward_logits(model, idx):
    x = model.atan(model.emb(idx))
    x = model.blocks(x)
    x = model.ln_out(x)
    return model.head(x)

@torch.no_grad()
def sample(model, ids, decode, local_seed):
    torch.manual_seed(local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)
    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
    out = []
    for _ in range(length):
        logits = forward_logits(model, ctx[:, -ctx_len:])[0, -1, :].float()
        functional.reset_net(model)
        for tok_id in set(ctx[0].tolist()):
            if 0 <= tok_id < logits.numel():
                logits[tok_id] = logits[tok_id] / 1.12 if logits[tok_id] > 0 else logits[tok_id] * 1.12
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        nxt = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
        out.append(nxt)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], dtype=torch.long, device="cuda")], dim=1)
    return decode(out)

sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
model = load_model(checkpoint_path)
rows = []
for i, prompt in enumerate(prompts):
    formatted = "Инструкция:\\n" + prompt + "\\n\\nОтвет:\\n"
    ids = sp.encode(formatted, out_type=int)
    generated = sample(model, ids, lambda x: sp.decode(x), seed + i)
    rows.append({{"prompt": prompt, "formatted_prompt": formatted, "generation": generated, "full_text": formatted + generated}})

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps({{"checkpoint": str(checkpoint_path), "examples": rows}}, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# SFT generation samples", "", f"Checkpoint: `{{checkpoint_path}}`", ""]
for i, row in enumerate(rows, 1):
    lines.extend([f"## Prompt {{i}}", "", "```text", row["formatted_prompt"] + row["generation"], "```", ""])
out_md.write_text("\\n".join(lines), encoding="utf-8")
print(f"wrote {{out_json}}")
print(f"wrote {{out_md}}")
"""
    run([sys.executable, "-c", script])


def upload_artifacts(args: argparse.Namespace, report: dict, checkpoint: Path) -> None:
    from huggingface_hub import HfApi
    import torch

    repo_id = args.hf_repo_id
    run_id = args.run_id
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    model_only = Path(args.checkpoint_dir) / "model_state_sft.pt"
    if not model_only.exists():
        ck = torch.load(checkpoint, map_location="cpu")
        torch.save(
            {
                "model_state": ck["model_state"],
                "config": ck.get("config", {}),
                "step": ck.get("step"),
                "epoch": ck.get("epoch"),
                "examples_seen": ck.get("examples_seen"),
                "best_val_loss": ck.get("best_val_loss"),
                "saved_at": ck.get("saved_at"),
                "format": "spikerugpt_sft_model_state_only",
            },
            model_only,
        )

    files = [
        checkpoint,
        Path(args.checkpoint_dir) / "best.pt",
        model_only,
        Path(args.report),
        Path(args.metrics_jsonl),
        Path(args.lm_eval_json),
        Path(args.lm_eval_md),
        Path(args.generations_json),
        Path(args.generations_md),
        Path(args.analysis_summary),
        Path(args.finalizer_log),
    ]
    uploaded = []
    for path in files:
        if not path.exists():
            continue
        remote = f"runs/{run_id}/sft/{path.name}"
        log(f"upload {path} -> {repo_id}/{remote}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote, repo_id=repo_id, repo_type="model")
        uploaded.append(remote)

    summary = {
        "repo_id": repo_id,
        "run_id": run_id,
        "uploaded": uploaded,
        "finished_at": now(),
        "train": report.get("train", {}),
        "validation": report.get("validation", {}),
    }
    out = Path(args.upload_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"runs/{run_id}/sft/{out.name}",
        repo_id=repo_id,
        repo_type="model",
    )
    log("upload_done " + json.dumps(summary, ensure_ascii=False))


def write_analysis_summary(args: argparse.Namespace, report: dict) -> None:
    base = read_json(Path(args.base_eval_json))
    sft_lm = read_json(Path(args.lm_eval_json))
    generations = read_json(Path(args.generations_json))
    out = Path(args.analysis_summary)
    out.parent.mkdir(parents=True, exist_ok=True)

    train = report.get("train", {})
    validation = report.get("validation", {})
    lines = [
        "# SpikeRuGPT v1 SFT Final Analysis",
        "",
        f"Generated: `{now()}`",
        "",
        "## Files",
        "",
        f"- SFT report: `{args.report}`",
        f"- SFT metrics: `{args.metrics_jsonl}`",
        f"- SFT LM eval: `{args.lm_eval_json}`",
        f"- SFT generations: `{args.generations_md}`",
        "",
        "## SFT Training",
        "",
        f"- steps: `{train.get('step')}`",
        f"- examples_seen: `{train.get('examples_seen')}`",
        f"- final_train_loss: `{train.get('final_loss')}`",
        f"- avg_last_50_loss: `{train.get('avg_last_50_loss')}`",
        f"- sft_val_loss: `{validation.get('loss')}`",
        f"- sft_val_ppl: `{validation.get('ppl')}`",
        "",
        "## Base vs SFT LM Perplexity",
        "",
        "| Split | Base loss | Base PPL | SFT loss | SFT PPL | Delta PPL |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    base_validation = base.get("validation", {})
    sft_validation = sft_lm.get("validation", {})
    for split in ["val_mixed", "val_wiki", "val_habr", "val_lit"]:
        b = base_validation.get(split, {})
        s = sft_validation.get(split, {})
        if not b or not s:
            continue
        base_ppl = float(b.get("ppl", 0.0))
        sft_ppl = float(s.get("ppl", 0.0))
        lines.append(
            f"| {split} | {float(b.get('loss', 0.0)):.4f} | {base_ppl:.2f} | "
            f"{float(s.get('loss', 0.0)):.4f} | {sft_ppl:.2f} | {sft_ppl - base_ppl:+.2f} |"
        )

    lines.extend(["", "## Generation Samples", ""])
    for i, row in enumerate(generations.get("examples", []), 1):
        lines.extend(
            [
                f"### Prompt {i}",
                "",
                f"Prompt: `{row.get('prompt', '')}`",
                "",
                "```text",
                row.get("full_text", ""),
                "```",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote_analysis_summary {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="sft-step43674-v1-b64tok18k")
    parser.add_argument("--report", default="reports/sft/sft-step43674-v1-b64tok18k.json")
    parser.add_argument("--metrics-jsonl", default="reports/sft/sft-step43674-v1-b64tok18k.metrics.jsonl")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sft/sft-step43674-v1-b64tok18k")
    parser.add_argument("--tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--hf-repo-id", default="Koras1k/spikerugpt-autonomous-runs")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--status-interval-sec", type=int, default=3600)
    parser.add_argument("--analysis-dir", default="ARTICLE/sft_v1_final_analysis")
    parser.add_argument("--base-eval-json", default="reports/eval/step_43674_metrics.json")
    parser.add_argument("--lm-eval-json", default=None)
    parser.add_argument("--lm-eval-md", default=None)
    parser.add_argument("--generations-json", default=None)
    parser.add_argument("--generations-md", default=None)
    parser.add_argument("--analysis-summary", default=None)
    parser.add_argument("--upload-summary", default=None)
    parser.add_argument("--finalizer-log", default="logs/sft-finalizer.log")
    parser.add_argument("--generation-length", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    args.lm_eval_json = args.lm_eval_json or str(analysis_dir / "sft_lm_eval.json")
    args.lm_eval_md = args.lm_eval_md or str(analysis_dir / "sft_lm_eval.md")
    args.generations_json = args.generations_json or str(analysis_dir / "sft_generations.json")
    args.generations_md = args.generations_md or str(analysis_dir / "sft_generations.md")
    args.analysis_summary = args.analysis_summary or str(analysis_dir / "README.md")
    args.upload_summary = args.upload_summary or str(analysis_dir / "hf_upload.json")

    last_status = 0.0
    report_path = Path(args.report)
    metrics_path = Path(args.metrics_jsonl)
    log(f"watch_start run_id={args.run_id}")

    while True:
        report = read_json(report_path)
        status = report.get("status")
        step = latest_train_step(metrics_path)
        if time.monotonic() - last_status >= args.status_interval_sec:
            log(f"status={status} latest={json.dumps(step, ensure_ascii=False)}")
            last_status = time.monotonic()
        if status == "ok":
            break
        if status not in {None, "running", "unreadable"}:
            raise SystemExit(f"SFT run ended with non-ok status: {status}")
        time.sleep(args.poll_sec)

    report = read_json(report_path)
    checkpoint = Path(report["train"]["final_checkpoint"])
    log(f"sft_complete checkpoint={checkpoint}")

    eval_script = Path("/workspace/scripts/evaluate_checkpoint_metrics.py")
    if eval_script.exists():
        run(
            [
                sys.executable,
                str(eval_script),
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                args.tokenizer,
                "--metrics-jsonl",
                args.metrics_jsonl,
                "--max-sequences",
                "32",
                "--activity-sequences",
                "8",
                "--train-batches",
                "16",
                "--out-json",
                args.lm_eval_json,
                "--out-md",
                args.lm_eval_md,
            ]
        )
    else:
        log(f"skip_lm_eval missing={eval_script}")

    generate_samples(args, checkpoint)
    write_analysis_summary(args, report)

    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.update(env)
    upload_artifacts(args, report, checkpoint)
    log("finalizer_done")


if __name__ == "__main__":
    main()
