"""Generate side-by-side base/SFT samples for SFT quality checks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig, RWKV_HEAD_QK_DIM  # noqa: E402
from src.spikingjelly.clock_driven import functional  # noqa: E402


DEFAULT_PROMPTS = [
    "Объясни простыми словами, что такое нейроморфные вычисления.",
    "Составь краткий план статьи о русскоязычной SpikeGPT-модели.",
    "Почему русский язык сложен для языковых моделей?",
    "Напиши короткое резюме: модель обучалась на русскоязычном корпусе и затем проходила SFT.",
    "Что такое perplexity в языковой модели? Ответь кратко.",
    "Дай три практических совета для очистки русскоязычного датасета.",
]

EASY_PROMPTS = [
    "Что такое Солнце?",
    "Где находится Москва?",
    "Кто написал роман Война и мир?",
    "Что такое вода?",
    "Сколько дней в неделе?",
    "Что такое Байкал?",
    "Кто такой Пушкин?",
    "Что такое электричество?",
    "Какая столица России?",
    "Что такое дерево?",
]


def load_model(path: Path, ctx_len: int) -> GPT:
    checkpoint = torch.load(path, map_location="cpu")
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
    model.eval()
    return model


def forward_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    model.step += 1
    _, time_steps = idx.size()
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


@torch.no_grad()
def generate(
    *,
    model: GPT,
    ids: list[int],
    decode: Callable[[list[int]], str],
    ctx_len: int,
    length: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
    generated: list[int] = []
    for _ in range(length):
        logits = forward_logits(model, ctx[:, -ctx_len:])[0, -1, :].float()
        functional.reset_net(model)

        if repetition_penalty != 1.0:
            for tok_id in set(ctx[0].tolist()):
                if tok_id >= logits.numel():
                    continue
                if logits[tok_id] > 0:
                    logits[tok_id] /= repetition_penalty
                else:
                    logits[tok_id] *= repetition_penalty

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
        generated.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]], dtype=torch.long, device="cuda")], dim=1)
    return decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt")
    parser.add_argument("--sft-v1-checkpoint", default="checkpoints/sft/sft-step43674-v1-b64tok18k/final.pt")
    parser.add_argument("--sft-v2-checkpoint", default="checkpoints/sft/sft-step43674-v2-superclean/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.75)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--out-json", default="ARTICLE/sft_v2_superclean/base_vs_sft_v1_v2_generations.json")
    parser.add_argument("--out-md", default="ARTICLE/sft_v2_superclean/base_vs_sft_v1_v2_generations.md")
    parser.add_argument("--prompt-set", choices=["default", "easy"], default="default")
    args = parser.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    decode = lambda ids: sp.decode(ids)

    prompts = EASY_PROMPTS if args.prompt_set == "easy" else DEFAULT_PROMPTS

    checkpoints = [
        ("base", Path(args.base_checkpoint), False),
        ("sft_v1_dirty", Path(args.sft_v1_checkpoint), True),
        ("sft_v2_superclean", Path(args.sft_v2_checkpoint), True),
    ]
    rows: list[dict[str, str]] = []
    config = vars(args)

    for label, checkpoint, instruction_format in checkpoints:
        print(f"loading {label}: {checkpoint}", flush=True)
        model = load_model(checkpoint, args.ctx_len)
        try:
            for index, prompt in enumerate(prompts):
                if instruction_format:
                    prefix = f"Инструкция:\n{prompt}\n\nОтвет:\n"
                else:
                    prefix = prompt
                print(f"{label} prompt {index + 1}/{len(prompts)}", flush=True)
                text = generate(
                    model=model,
                    ids=sp.encode(prefix, out_type=int),
                    decode=decode,
                    ctx_len=args.ctx_len,
                    length=args.length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed + index,
                )
                rows.append(
                    {
                        "label": label,
                        "checkpoint": str(checkpoint),
                        "prompt": prompt,
                        "prefix": prefix,
                        "generation": text,
                        "full_text": prefix + text,
                    }
                )
        finally:
            del model
            torch.cuda.empty_cache()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"config": config, "examples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    by_prompt: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row)

    lines = [
        "# Base vs SFT v1 vs SFT v2 generations",
        "",
        (
            f"Sampling: temperature={args.temperature}, top_p={args.top_p}, "
            f"repetition_penalty={args.repetition_penalty}, length={args.length}"
        ),
        "",
    ]
    for prompt, prompt_rows in by_prompt.items():
        lines.extend(["## " + prompt, ""])
        for row in prompt_rows:
            lines.extend([f"### {row['label']}", "", "```text", row["full_text"], "```", ""])
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
