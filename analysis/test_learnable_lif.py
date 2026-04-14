"""
Быстрая проверка гипотезы LearnableLIF.

Загружаем spikegpt-ru-261.pth, заменяем LIF-узлы на LearnableLIFNode,
замораживаем все параметры кроме LIF (log_tau, log_threshold),
прогоняем ~50 батчей через real-text данные и смотрим, куда сойдутся
tau и threshold по слоям.

Гипотеза: нижние слои должны получить меньший tau (быстрая динамика),
           верхние — больший tau (медленная интеграция).

Запуск: python analysis/test_learnable_lif.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

# ── Конфиг ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/spikegpt-ru-261.pth"
DATA_PATH    = "data/ru_train_full.npy"
N_LAYER      = 12
N_EMBD       = 512
CTX_LEN      = 1024
VOCAB_SIZE   = 50258
N_BATCHES    = 50        # число батчей для fine-tune LIF-параметров
BATCH_SIZE   = 4
LR_LIF       = 1e-2     # только LIF-параметры обучаем с высоким lr

os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)

from src.model import GPT, GPTConfig, LearnableLIFNode

# ── Загрузка данных ────────────────────────────────────────────────────────────
print(f"Загрузка данных из {DATA_PATH} …")
tokens = np.load(DATA_PATH, mmap_mode='r')
total  = len(tokens)
print(f"  Токенов: {total:,}")

def get_batch():
    ix = torch.randint(total - CTX_LEN - 1, (BATCH_SIZE,))
    x  = torch.stack([torch.from_numpy(tokens[i:i+CTX_LEN].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(tokens[i+1:i+CTX_LEN+1].astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()

# ── Строим модель с LearnableLIFNode ──────────────────────────────────────────
print("Строим модель с LearnableLIFNode …")
cfg   = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type="RWKV", n_layer=N_LAYER, n_embd=N_EMBD)
model = GPT(cfg).cuda()

# Убеждаемся что lif1/lif2 — LearnableLIFNode (они уже такие по умолчанию)
for i, block in enumerate(model.blocks):
    assert isinstance(block.lif1, LearnableLIFNode), f"block {i} lif1 не LearnableLIFNode!"
    assert isinstance(block.lif2, LearnableLIFNode), f"block {i} lif2 не LearnableLIFNode!"

# ── Загружаем чекпоинт (strict=False — LIF-параметры не в чекпоинте) ──────────
print(f"Загружаем чекпоинт {CHECKPOINT} …")
sd = torch.load(CHECKPOINT, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
lif_missing = [k for k in missing if 'lif' in k]
print(f"  Отсутствующих ключей: {len(missing)}  (LIF: {len(lif_missing)})")
print(f"  Лишних ключей: {len(unexpected)}")
model.train()

# ── Замораживаем всё кроме LIF-параметров ─────────────────────────────────────
lif_params = []
for name, param in model.named_parameters():
    if 'lif' in name:
        param.requires_grad_(True)
        lif_params.append(param)
    else:
        param.requires_grad_(False)

print(f"\nОбучаемых параметров (только LIF): {sum(p.numel() for p in lif_params)}")
print(f"  ({len(lif_params)} тензоров: log_tau и log_threshold для {N_LAYER * 2} LIF-узлов)")

# ── Начальные значения tau / threshold ────────────────────────────────────────
def print_lif_table(title: str):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"{'Layer':>5}  {'lif1 tau':>9} {'lif1 thr':>9}  {'lif2 tau':>9} {'lif2 thr':>9}")
    print(f"{'-'*50}")
    taus_all = []
    for i, block in enumerate(model.blocks):
        with torch.no_grad():
            t1 = block.lif1.tau.item()
            h1 = block.lif1.threshold.item()
            t2 = block.lif2.tau.item()
            h2 = block.lif2.threshold.item()
        taus_all.extend([t1, t2])
        print(f"{i:>5}  {t1:>9.4f} {h1:>9.4f}  {t2:>9.4f} {h2:>9.4f}")
    print(f"\n  Средний tau: {np.mean(taus_all):.4f}  ±{np.std(taus_all):.4f}")

print_lif_table("НАЧАЛЬНЫЕ ЗНАЧЕНИЯ (до обучения LIF)")

# ── Оптимизатор только для LIF-параметров ─────────────────────────────────────
optimizer = torch.optim.Adam(lif_params, lr=LR_LIF)

# ── Цикл обучения LIF-параметров ──────────────────────────────────────────────
print(f"\nОбучение LIF-параметров ({N_BATCHES} батчей, lr={LR_LIF}) …\n")
losses = []
for step in range(N_BATCHES):
    x, y = get_batch()
    optimizer.zero_grad()
    # Прямой проход через model.forward (возвращает L2Wrap(loss, logits))
    # Нам нужен loss — он хранится как первый элемент, но L2Wrap прозрачен для forward
    out = model(x, targets=y)
    # out — это loss (L2Wrap в forward уже вернул loss)
    out.backward()
    torch.nn.utils.clip_grad_norm_(lif_params, 1.0)
    optimizer.step()
    losses.append(out.item())

    if (step + 1) % 10 == 0:
        mean_loss = np.mean(losses[-10:])
        # Средний tau по всем LIF-узлам
        with torch.no_grad():
            avg_tau = np.mean([block.lif1.tau.item() for block in model.blocks] +
                              [block.lif2.tau.item() for block in model.blocks])
        print(f"  Шаг {step+1:3d}/{N_BATCHES}  loss={mean_loss:.4f}  avg_tau={avg_tau:.4f}")

# ── Финальные значения ─────────────────────────────────────────────────────────
print_lif_table("ФИНАЛЬНЫЕ ЗНАЧЕНИЯ (после обучения LIF)")

# ── Проверка гипотезы ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ПРОВЕРКА ГИПОТЕЗЫ: нижние слои → меньший tau, верхние → больший tau")
print("="*70)
tau_by_layer = []
for i, block in enumerate(model.blocks):
    with torch.no_grad():
        avg = (block.lif1.tau.item() + block.lif2.tau.item()) / 2
    tau_by_layer.append(avg)

# Корреляция tau с номером слоя
layers = np.arange(N_LAYER)
corr = np.corrcoef(layers, tau_by_layer)[0, 1]
print(f"\n  Корреляция (layer_id, avg_tau) = {corr:+.4f}")
if corr > 0.3:
    print("  ✓ ГИПОТЕЗА ПОДТВЕРЖДЕНА: tau растёт с глубиной слоя")
elif corr < -0.3:
    print("  ✗ ОБРАТНАЯ ЗАВИСИМОСТЬ: tau убывает с глубиной слоя")
else:
    print("  ~ НЕТ ВЫРАЖЕННОЙ ЗАВИСИМОСТИ: tau не коррелирует с глубиной")

print("\n  tau по слоям (avg lif1+lif2):")
for i, t in enumerate(tau_by_layer):
    bar = "█" * int(t * 10)
    print(f"    Layer {i:2d}: {t:.4f}  {bar}")
