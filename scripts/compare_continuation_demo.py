"""Compare v0/base/SFT continuations on the original demo.py prompt set."""

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
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from src.model import GPT, GPTConfig, RWKV_HEAD_QK_DIM  # noqa: E402
from src.spikingjelly.clock_driven import functional, neuron, surrogate  # noqa: E402


DEMOS = [
    ("Проза: природа", "Осенний лес был тих и задумчив. Жёлтые листья медленно", 200, 0.85, 0.9),
    ("Продолжение: диалог", "— Ты понимаешь, что это невозможно? — спросил он.\n— Почему же,", 220, 0.85, 0.9),
    ("Проза: городская ночь", "Поздним вечером улицы Москвы опустели. Только одинокий фонарь освещал", 200, 0.85, 0.9),
    ("Новости: происшествие", "В Санкт-Петербурге сегодня утром", 150, 0.75, 0.85),
    ("Новости: наука", "Учёные Московского государственного университета объявили об открытии нового", 150, 0.75, 0.85),
    ("Новости: политика", "Государственная Дума приняла закон о", 150, 0.75, 0.85),
    ("История: Наполеон", "В 1812 году армия Наполеона вошла в Москву и обнаружила, что город", 180, 0.8, 0.85),
    ("История: революция", "В октябре 1917 года в Петрограде произошли события, которые навсегда изменили", 180, 0.8, 0.85),
    ("История: Вторая мировая", "Летом 1941 года немецкие войска перешли границу СССР. Советские солдаты", 180, 0.8, 0.85),
]


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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"v0 load_state missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def load_v1(path: Path, ctx_len: int) -> GPT:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_model(int(config.get("vocab_size", 32000)), ctx_len, old_lif=False)
    model.load_state_dict(checkpoint["model_state"])
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
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    stop_on_double_newline: bool,
) -> str:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
    generated: list[int] = []
    for _ in range(max_tokens):
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
        if stop_on_double_newline:
            text = decode(generated)
            if "\n\n" in text:
                return text.split("\n\n")[0].strip()
        ctx = torch.cat([ctx, torch.tensor([[next_id]], dtype=torch.long, device="cuda")], dim=1)
    return decode(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--v0-checkpoint", default="models/v0_spikerugpt_100m_taiga/spikegpt-ru-175.pth")
    parser.add_argument("--v0-tokenizer", default="models/v0_spikerugpt_100m_taiga/tokenizer/tokenizer.json")
    parser.add_argument("--base-checkpoint", default="checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt")
    parser.add_argument("--sft-v2-checkpoint", default="checkpoints/sft/sft-step43674-v2-superclean/final.pt")
    parser.add_argument("--v1-tokenizer", default="tokenizer/spikerugpt-bpe-32k.model")
    parser.add_argument("--out-json", default="ARTICLE/sft_v2_superclean/continuation_v0_base_sft_v2.json")
    parser.add_argument("--out-md", default="ARTICLE/sft_v2_superclean/continuation_v0_base_sft_v2.md")
    args = parser.parse_args()

    import sentencepiece as spm

    v0_tokenizer = Tokenizer.from_file(args.v0_tokenizer)
    v1_tokenizer = spm.SentencePieceProcessor(model_file=args.v1_tokenizer)

    specs = [
        ("v0_taiga_100m", Path(args.v0_checkpoint), load_v0, lambda text: v0_tokenizer.encode(text).ids, lambda ids: v0_tokenizer.decode(ids)),
        ("base_1b_74m", Path(args.base_checkpoint), load_v1, lambda text: v1_tokenizer.encode(text, out_type=int), lambda ids: v1_tokenizer.decode(ids)),
        ("sft_v2_superclean", Path(args.sft_v2_checkpoint), load_v1, lambda text: v1_tokenizer.encode(text, out_type=int), lambda ids: v1_tokenizer.decode(ids)),
    ]

    rows = []
    for model_label, checkpoint, loader, encode, decode in specs:
        print(f"loading {model_label}: {checkpoint}", flush=True)
        model = loader(checkpoint, args.ctx_len)
        try:
            for idx, (title, prompt, max_tokens, temperature, top_p) in enumerate(DEMOS):
                print(f"{model_label} {idx + 1}/{len(DEMOS)} {title}", flush=True)
                continuation = generate(
                    model=model,
                    ids=encode(prompt),
                    decode=decode,
                    ctx_len=args.ctx_len,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed + idx,
                    stop_on_double_newline=True,
                )
                rows.append(
                    {
                        "model": model_label,
                        "checkpoint": str(checkpoint),
                        "title": title,
                        "prompt": prompt,
                        "continuation": continuation,
                        "full_text": f"{prompt} {continuation}".strip(),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
        finally:
            del model
            torch.cuda.empty_cache()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"config": vars(args), "examples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Original demo continuation comparison",
        "",
        "Prompt set copied from `demo.py`. This is a base-LM continuation test, not instruction following.",
        "",
        f"Repetition penalty: {args.repetition_penalty}",
        "",
    ]
    for title, prompt, max_tokens, temperature, top_p in DEMOS:
        lines.extend(
            [
                f"## {title}",
                "",
                f"Parameters: `temperature={temperature}`, `top_p={top_p}`, `max_tokens={max_tokens}`",
                "",
                "Prompt:",
                "",
                "```text",
                prompt,
                "```",
                "",
            ]
        )
        for model_label, *_ in specs:
            row = next(r for r in rows if r["model"] == model_label and r["title"] == title)
            lines.extend([f"### {model_label}", "", "```text", row["full_text"], "```", ""])

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
