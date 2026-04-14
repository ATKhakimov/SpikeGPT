"""
Сравнение двух чекпоинтов на одинаковых промптах.
Запуск: python analysis/compare_checkpoints.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional, neuron, surrogate

CKPT_A     = "checkpoints/spikegpt-ru-175.pth"   # до lr-restart бага, PPL ~67
CKPT_B     = "checkpoints/spikegpt-ru-181.pth"   # сразу после рестарта lr
CKPT_C     = "checkpoints/spikegpt-ru-261.pth"   # лучший, PPL 59.79
TOKENIZER  = "tokenizer/rugpt3"
N_LAYER, N_EMBD, CTX_LEN, VOCAB_SIZE = 12, 512, 1024, 50258

PROMPTS = [
    # Художественная проза
    ("Проза: осень",    "Осенний лес был тих и задумчив. Жёлтые листья медленно"),
    ("Проза: ночь",     "Поздним вечером улицы Москвы опустели. Только одинокий фонарь освещал"),
    ("Проза: море",     "Он стоял на берегу и смотрел на море. Волны"),
    # Диалог
    ("Диалог: спор",    "— Ты понимаешь, что это невозможно? — спросил он.\n— Почему же,"),
    ("Диалог: встреча", "— Давно не виделись, — сказала она, улыбнувшись.\n— Да, очень давно. Я"),
    # История
    ("История: 1917",   "В октябре 1917 года в Петрограде произошли события, которые навсегда изменили"),
    ("История: война",  "Летом 1941 года немецкие войска перешли границу СССР. Советские солдаты"),
    ("История: Пётр",   "Пётр Великий основал Санкт-Петербург в 1703 году. Новая столица"),
    # Новости / публицистика
    ("Новости: наука",  "Учёные Московского государственного университета объявили об открытии нового"),
    ("Новости: эконом", "По данным Центрального банка России, инфляция в стране"),
    ("Новости: спорт",  "Сборная России по футболу вчера провела товарищеский матч против"),
    # Описание / рассуждение
    ("Рассуждение",     "Искусственный интеллект изменит мир так же радикально, как"),
    ("Природа",         "Байкал — самое глубокое озеро в мире. Его воды"),
]

MAX_TOKENS  = 120
TEMPERATURE = 0.85
TOP_P       = 0.9
REP_PENALTY = 1.3


def load_model(ckpt_path):
    os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)
    cfg   = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type="RWKV", n_layer=N_LAYER, n_embd=N_EMBD)
    model = GPT(cfg).cuda()
    for block in model.blocks:
        block.lif1 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                              backend='cupy', v_threshold=1.).cuda()
        block.lif2 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                              backend='cupy', v_threshold=1.).cuda()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS,
             temperature=TEMPERATURE, top_p=TOP_P, rep_penalty=REP_PENALTY):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ctx = torch.tensor([ids], dtype=torch.long).cuda()
    generated = []
    for _ in range(max_tokens):
        inp = ctx[:, -CTX_LEN:]
        x = model.atan(model.emb(inp))
        x = model.blocks(x)
        x = model.ln_out(x)
        logits = model.head(x)[0, -1, :]
        functional.reset_net(model)

        if rep_penalty != 1.0:
            for tok_id in set(ctx[0].tolist()):
                logits[tok_id] = logits[tok_id] / rep_penalty if logits[tok_id] > 0 \
                                  else logits[tok_id] * rep_penalty

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_ids[torch.multinomial(sorted_probs, 1)].item()
        generated.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([[next_id]]).cuda()], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("Загрузка токенизатора …")
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    print(f"Загрузка {CKPT_A} (эпоха 175) …")
    model_a = load_model(CKPT_A)
    print(f"Загрузка {CKPT_B} (эпоха 181) …")
    model_b = load_model(CKPT_B)
    print(f"Загрузка {CKPT_C} (эпоха 261) …\n")
    model_c = load_model(CKPT_C)

    sep  = "═" * 76
    sep2 = "─" * 76

    for i, (title, prompt) in enumerate(PROMPTS, 1):
        print(sep)
        print(f"  [{i}/{len(PROMPTS)}] {title}")
        print(f"  Промпт: {prompt[:70]}")
        print(sep)

        out_a = generate(model_a, tok, prompt)
        out_b = generate(model_b, tok, prompt)
        out_c = generate(model_c, tok, prompt)

        print(f"\n  ── Эпоха 175  PPL ~67  (до рестарта lr) {'─'*28}")
        print(f"  {prompt} {out_a}")
        print(f"\n  ── Эпоха 181  PPL ~?   (сразу после рестарта lr) {'─'*22}")
        print(f"  {prompt} {out_b}")
        print(f"\n  ── Эпоха 261  PPL 59.8 (лучший чекпоинт) {'─'*27}")
        print(f"  {prompt} {out_c}")
        print()

    print(sep)
    print("Готово.")


if __name__ == "__main__":
    main()
