"""
Генерация текста русской SpikeGPT модели.

Использование:
    python generate.py --prompt "Москва — это"
    python generate.py --prompt "Нейронные сети" --length 200 --temperature 0.9
"""

import os, sys, glob, argparse, math, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional

# ── Параметры модели (должны совпадать с train.py) ─────────────────────────
N_LAYER    = 12
N_EMBD     = 512
CTX_LEN    = 1024
MODEL_TYPE = "RWKV"
VOCAB_SIZE = 50258

TOKENIZER_PATH  = "tokenizer/rugpt3"
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_GLOB = "spikegpt-ru-*.pth"


def load_latest_checkpoint():
    files = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB)),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[-1])
    )
    if not files:
        print(f"ERROR: нет чекпоинтов в {CHECKPOINT_DIR}/")
        sys.exit(1)
    return files[-1]


@torch.no_grad()
def generate(model, tokenizer, prompt, length=200, temperature=1.0, top_p=0.9,
             repetition_penalty=1.3):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ctx = torch.tensor([ids], dtype=torch.long).cuda()

    generated = []
    for step in range(length):
        inp = ctx[:, -CTX_LEN:]
        logits = _forward_logits(model, inp)
        functional.reset_net(model)

        # берём логиты последнего токена
        logits = logits[0, -1, :]  # [vocab]

        # repetition penalty — штрафуем уже встреченные токены
        if repetition_penalty != 1.0:
            seen = set(ctx[0].tolist())
            for tok_id in seen:
                if logits[tok_id] > 0:
                    logits[tok_id] /= repetition_penalty
                else:
                    logits[tok_id] *= repetition_penalty

        logits = logits / max(temperature, 1e-6)

        # top-p (nucleus) сэмплинг
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_ids[torch.multinomial(sorted_probs, 1)].item()

        generated.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]]).cuda()], dim=1)

        if (step + 1) % 20 == 0:
            print(f"  [{step+1}/{length} токенов]", flush=True)

    return tokenizer.decode(generated, skip_special_tokens=True)


def _forward_logits(model, idx):
    """Прогон без вычисления loss — возвращает logits [B, T, vocab]."""
    from src.model import L2Wrap
    from src.spikingjelly.clock_driven.surrogate import ATan as atan_cls
    import torch.nn as nn

    x = model.atan(model.emb(idx))
    x = model.blocks(x)
    x = model.ln_out(x)
    x = model.head(x)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default="Москва — это")
    parser.add_argument("--length",      type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="путь к .pth файлу (по умолчанию — последний чекпоинт)")
    parser.add_argument("--max-tokens", type=int,   default=None,
                        help="жёсткий лимит токенов (обрезает по первому \n\n если достигнут)")
    args = parser.parse_args()

    os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)

    # ── Загрузка токенизатора ──────────────────────────────────────────────
    print(f"Загрузка токенизатора из {TOKENIZER_PATH} …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    # ── Загрузка модели ────────────────────────────────────────────────────
    ckpt = args.checkpoint or load_latest_checkpoint()
    print(f"Загрузка чекпоинта: {ckpt}")
    config = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type=MODEL_TYPE,
                       n_layer=N_LAYER, n_embd=N_EMBD)
    model = GPT(config).cuda()

    # Возвращаем оригинальные LIF узлы (модель обучалась с MultiStepLIFNode,
    # а не с LearnableLIFNode который мы добавили позже)
    from src.spikingjelly.clock_driven import neuron, surrogate
    for block in model.blocks:
        block.lif1 = neuron.MultiStepLIFNode(
            tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
            backend='cupy', v_threshold=1.
        ).cuda()
        block.lif2 = neuron.MultiStepLIFNode(
            tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
            backend='cupy', v_threshold=1.
        ).cuda()

    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Модель загружена ({sum(p.numel() for p in model.parameters()):,} параметров)\n")

    # ── Генерация ──────────────────────────────────────────────────────────
    print(f"Промпт: «{args.prompt}»")
    print(f"Генерация {args.length} токенов (temperature={args.temperature}, top_p={args.top_p}):\n")
    print("-" * 60)
    result = generate(model, tokenizer, args.prompt,
                      length=args.length, temperature=args.temperature, top_p=args.top_p)
    print("-" * 60)
    print("\nИТОГ:")
    print(args.prompt + result)


if __name__ == "__main__":
    main()
