"""
Демонстрационная сессия SpikeGPT Russian.
Запуск: python demo.py
Результаты сохраняются в demo_results.md
"""

import os, sys, glob, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional, neuron, surrogate

CHECKPOINT  = "checkpoints/spikegpt-ru-261.pth"
TOKENIZER   = "tokenizer/rugpt3"
N_LAYER, N_EMBD, CTX_LEN, VOCAB_SIZE = 12, 512, 1024, 50258
OUTPUT_FILE = "demo_results.md"

DEMOS = [
    # (название, промпт, max_tokens, temperature, top_p)
    ("Проза: природа",
     "Осенний лес был тих и задумчив. Жёлтые листья медленно",
     200, 0.85, 0.9),

    ("Продолжение: диалог",
     "— Ты понимаешь, что это невозможно? — спросил он.\n— Почему же,",
     220, 0.85, 0.9),

    ("Проза: городская ночь",
     "Поздним вечером улицы Москвы опустели. Только одинокий фонарь освещал",
     200, 0.85, 0.9),

    ("Новости: происшествие",
     "В Санкт-Петербурге сегодня утром",
     150, 0.75, 0.85),

    ("Новости: наука",
     "Учёные Московского государственного университета объявили об открытии нового",
     150, 0.75, 0.85),

    ("Новости: политика",
     "Государственная Дума приняла закон о",
     150, 0.75, 0.85),

    ("История: Наполеон",
     "В 1812 году армия Наполеона вошла в Москву и обнаружила, что город",
     180, 0.8, 0.85),

    ("История: революция",
     "В октябре 1917 года в Петрограде произошли события, которые навсегда изменили",
     180, 0.8, 0.85),

    ("История: Вторая мировая",
     "Летом 1941 года немецкие войска перешли границу СССР. Советские солдаты",
     180, 0.8, 0.85),
]


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_p=0.9,
             repetition_penalty=1.3, stop_on_double_newline=True):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ctx = torch.tensor([ids], dtype=torch.long).cuda()
    generated = []

    for step in range(max_tokens):
        inp = ctx[:, -CTX_LEN:]
        x = model.atan(model.emb(inp))
        x = model.blocks(x)
        x = model.ln_out(x)
        logits = model.head(x)[0, -1, :]
        functional.reset_net(model)

        if repetition_penalty != 1.0:
            for tok_id in set(ctx[0].tolist()):
                logits[tok_id] = logits[tok_id] / repetition_penalty if logits[tok_id] > 0 \
                                  else logits[tok_id] * repetition_penalty

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_ids[torch.multinomial(sorted_probs, 1)].item()

        generated.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]]).cuda()], dim=1)

        # стоп по двойному переносу строки
        decoded_so_far = tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_double_newline and "\n\n" in decoded_so_far:
            decoded_so_far = decoded_so_far.split("\n\n")[0]
            return decoded_so_far.strip()

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)

    print("Загрузка токенизатора …")
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    print(f"Загрузка модели из {CHECKPOINT} …")
    cfg   = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type="RWKV", n_layer=N_LAYER, n_embd=N_EMBD)
    model = GPT(cfg).cuda()
    for block in model.blocks:
        block.lif1 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                              backend='cupy', v_threshold=1.).cuda()
        block.lif2 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                              backend='cupy', v_threshold=1.).cuda()
    sd = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Модель загружена ({sum(p.numel() for p in model.parameters()):,} параметров)\n")

    results = []
    for i, (title, prompt, max_tok, temp, top_p) in enumerate(DEMOS, 1):
        print(f"[{i}/{len(DEMOS)}] {title} …", flush=True)
        answer = generate(model, tok, prompt, max_tokens=max_tok,
                          temperature=temp, top_p=top_p)
        results.append((title, prompt, answer, max_tok, temp, top_p))
        print(f"  {prompt[:50]}… → {answer[:60]}…\n")

    # ── Сохранение в Markdown ──────────────────────────────────────────────
    md = "# SpikeGPT Russian — Демонстрация генерации\n\n"
    md += f"> Модель: `{CHECKPOINT}` | Параметры: 100M | Корпус: Тайга ~1.8B токенов\n\n---\n\n"

    for title, prompt, answer, max_tok, temp, top_p in results:
        md += f"## {title}\n\n"
        md += f"**Параметры:** `temperature={temp}`, `top_p={top_p}`, `max_tokens={max_tok}`\n\n"
        md += f"**Промпт:** {prompt}\n\n"
        md += f"**Продолжение:** {prompt} {answer}\n\n"
        md += "---\n\n"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nРезультаты сохранены в: {OUTPUT_FILE}")
    print("\n" + "="*60)
    print("ИТОГОВЫЕ ПРИМЕРЫ:")
    print("="*60)
    for title, prompt, answer, *_ in results:
        print(f"\n[{title}]")
        print(f"  {prompt} {answer}")


if __name__ == "__main__":
    main()
