"""Generate side-by-side v0/v1 samples for article notes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig  # noqa: E402
from src.spikingjelly.clock_driven import functional, neuron, surrogate  # noqa: E402


def build_model(vocab_size: int, ctx_len: int, old_lif: bool) -> GPT:
    model = GPT(GPTConfig(vocab_size, ctx_len, model_type="RWKV", n_layer=12, n_embd=512)).cuda()
    if old_lif:
        for block in model.blocks:
            block.lif1 = neuron.MultiStepLIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(alpha=2.0),
                backend="torch",
                v_threshold=1.0,
            ).cuda()
            block.lif2 = neuron.MultiStepLIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(alpha=2.0),
                backend="torch",
                v_threshold=1.0,
            ).cuda()
    model.eval()
    return model


def load_v0(path: Path, ctx_len: int) -> GPT:
    model = build_model(50258, ctx_len, old_lif=True)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def load_v1(path: Path, ctx_len: int) -> GPT:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_model(int(config.get("vocab_size", 32000)), ctx_len, old_lif=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def forward_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    x = model.atan(model.emb(idx))
    x = model.blocks(x)
    x = model.ln_out(x)
    return model.head(x)


@torch.no_grad()
def generate(
    *,
    model: GPT,
    ids: list[int],
    decode,
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
    generated = []
    for _ in range(length):
        logits = forward_logits(model, ctx[:, -ctx_len:])
        functional.reset_net(model)
        logits = logits[0, -1, :].float()

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
        next_id = sorted_ids[torch.multinomial(sorted_probs, 1)].item()
        generated.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]], dtype=torch.long, device="cuda")], dim=1)

    return decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--v0-checkpoint", default="models/v0_spikerugpt_100m_taiga/spikegpt-ru-175.pth")
    parser.add_argument("--v0-tokenizer", default="models/v0_spikerugpt_100m_taiga/tokenizer/tokenizer.json")
    parser.add_argument("--v1-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-12h/final.pt")
    parser.add_argument("--v1-tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--out-json", default="reports/v0_v1_generations.json")
    parser.add_argument("--out-md", default="ARTICLE/v0_v1_generation_examples.md")
    args = parser.parse_args()

    import sentencepiece as spm

    prompts = [
        "Объясни простыми словами, что такое нейроморфные вычисления.",
        "Почему русский язык сложен для языковых моделей?",
        "Напиши короткий абзац о будущем искусственного интеллекта в России.",
    ]

    v0_tokenizer = Tokenizer.from_file(args.v0_tokenizer)
    v1_tokenizer = spm.SentencePieceProcessor(model_file=args.v1_tokenizer)

    print("loading v0", flush=True)
    v0 = load_v0(Path(args.v0_checkpoint), args.ctx_len)
    print("loading v1", flush=True)
    v1 = load_v1(Path(args.v1_checkpoint), args.ctx_len)

    rows = []
    try:
        for index, prompt in enumerate(prompts):
            print(f"prompt {index + 1}/{len(prompts)} v0", flush=True)
            v0_text = generate(
                model=v0,
                ids=v0_tokenizer.encode(prompt).ids,
                decode=lambda ids: v0_tokenizer.decode(ids),
                ctx_len=args.ctx_len,
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + index,
            )
            print(f"prompt {index + 1}/{len(prompts)} v1", flush=True)
            v1_text = generate(
                model=v1,
                ids=v1_tokenizer.encode(prompt, out_type=int),
                decode=lambda ids: v1_tokenizer.decode(ids),
                ctx_len=args.ctx_len,
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + index,
            )
            rows.append({"prompt": prompt, "v0": prompt + v0_text, "v1": prompt + v1_text})
    finally:
        del v0, v1
        torch.cuda.empty_cache()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"config": vars(args), "examples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v0 vs v1 generation examples",
        "",
        f"Дата: 2026-06-02",
        f"Sampling: temperature={args.temperature}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}, length={args.length}",
        "",
        "Важно: v1 здесь является ранним 12h checkpoint, а не финальной моделью после 1B continuation/SFT.",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Prompt {i}",
                "",
                f"```text\n{row['prompt']}\n```",
                "",
                "### v0",
                "",
                f"```text\n{row['v0']}\n```",
                "",
                "### v1 12h",
                "",
                f"```text\n{row['v1']}\n```",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
