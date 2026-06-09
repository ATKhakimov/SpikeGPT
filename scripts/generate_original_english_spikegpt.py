"""Generate continuations from the original English SpikeGPT 216M checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ["VOCAB_SIZE"] = "50277"

from src.model import GPT, GPTConfig, RWKV_HEAD_QK_DIM  # noqa: E402
from src.spikingjelly.clock_driven import functional, neuron, surrogate  # noqa: E402


PROMPTS = [
    ("Prose: autumn", "The old forest was quiet, and the yellow leaves slowly", 120, 0.8, 0.9),
    ("Dialogue", "“You know this is impossible,” he said.\n“Why not,”", 140, 0.8, 0.9),
    ("City night", "Late at night, the streets of London were empty. A single street lamp", 120, 0.8, 0.9),
    ("News", "This morning in New York, officials announced", 100, 0.75, 0.85),
    ("Science", "Researchers at the university discovered a new method", 100, 0.75, 0.85),
    ("History", "In October 1917, events in Petrograd changed", 120, 0.8, 0.85),
]


def ensure_checkpoint(path: Path) -> None:
    if path.exists():
        print(f"using cached checkpoint: {path}", flush=True)
        return
    from huggingface_hub import hf_hub_download

    print("downloading ridger/SpikeGPT-OpenWebText-216M / SpikeGPT-216M.pth", flush=True)
    downloaded = hf_hub_download(
        repo_id="ridger/SpikeGPT-OpenWebText-216M",
        filename="SpikeGPT-216M.pth",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded, path)
    print(f"saved checkpoint: {path}", flush=True)


def load_model(checkpoint_path: Path, ctx_len: int) -> GPT:
    model = GPT(GPTConfig(50277, ctx_len, model_type="RWKV", n_layer=18, n_embd=768)).cuda()
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
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"unexpected keys: {len(unexpected)}", flush=True)
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
    max_tokens: int,
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
        text = decode(generated)
        if "\n\n" in text:
            return text.split("\n\n")[0].strip()
        ctx = torch.cat([ctx, torch.tensor([[next_id]], dtype=torch.long, device="cuda")], dim=1)
    return decode(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/spikegpt-en-216M.pth")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--out-json", default="ARTICLE/original_english_spikegpt_generations.json")
    parser.add_argument("--out-md", default="ARTICLE/original_english_spikegpt_generations.md")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    checkpoint = Path(args.checkpoint)
    ensure_checkpoint(checkpoint)

    print("loading tokenizer: EleutherAI/gpt-neox-20b", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)

    print("loading English SpikeGPT 216M", flush=True)
    model = load_model(checkpoint, args.ctx_len)

    rows = []
    for idx, (title, prompt, max_tokens, temperature, top_p) in enumerate(PROMPTS):
        print(f"{idx + 1}/{len(PROMPTS)} {title}", flush=True)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        continuation = generate(
            model=model,
            ids=ids,
            decode=lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=True),
            ctx_len=args.ctx_len,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed + idx,
        )
        rows.append(
            {
                "title": title,
                "prompt": prompt,
                "continuation": continuation,
                "full_text": f"{prompt} {continuation}".strip(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"config": vars(args), "examples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Original English SpikeGPT 216M Generations",
        "",
        "Model: `ridger/SpikeGPT-OpenWebText-216M`",
        "Tokenizer: `EleutherAI/gpt-neox-20b`",
        f"Checkpoint: `{checkpoint}`",
        f"Repetition penalty: `{args.repetition_penalty}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['title']}",
                "",
                f"Parameters: `temperature={row['temperature']}`, `top_p={row['top_p']}`, `max_tokens={row['max_tokens']}`",
                "",
                "```text",
                row["full_text"],
                "```",
                "",
            ]
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
