"""
Чистая проверка гипотезы LearnableLIF — обучение с нуля.

Обучаем маленькую модель (4 слоя, d=128, ~3M параметров) с LearnableLIFNode
на реальных данных Тайга. Все параметры обучаются вместе, включая tau и threshold.
Смотрим, куда сойдутся tau по слоям к концу обучения.

Гипотеза: нижние слои → меньший tau (быстрая динамика / лексика),
           верхние слои → больший tau (медленная интеграция / синтаксис).

Запуск: python analysis/train_small_learnable_lif.py
Время: ~10–15 минут на A100 / RTX 3090
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import json

# ── Конфиг маленькой модели ───────────────────────────────────────────────────
N_LAYER    = 4
N_EMBD     = 128       # C=128, B*C % min(C,1024)=0 ✓ для любого батча
CTX_LEN    = 512       # < T_MAX=1024 в model.py
VOCAB_SIZE = 50258

BATCH_SIZE = 16
N_STEPS    = 2000      # ~10 мин на A100
LOG_EVERY  = 100       # шаги между логами tau
LR         = 6e-4

DATA_PATH  = "data/ru_train_full.npy"
OUT_JSON   = "analysis/learnable_lif_small_run.json"

os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)

from src.model import GPT, GPTConfig, LearnableLIFNode

# ── Данные ────────────────────────────────────────────────────────────────────
print(f"Загрузка данных {DATA_PATH} …")
tokens = np.load(DATA_PATH, mmap_mode='r')
total  = len(tokens)
print(f"  Токенов в корпусе: {total:,}\n")

def get_batch():
    ix = torch.randint(total - CTX_LEN - 1, (BATCH_SIZE,))
    x  = torch.stack([torch.from_numpy(tokens[i:i+CTX_LEN].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(tokens[i+1:i+CTX_LEN+1].astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()

# ── Модель ────────────────────────────────────────────────────────────────────
print(f"Строим модель: {N_LAYER}L × {N_EMBD}D  (LearnableLIF, все параметры обучаются)")
cfg   = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type="RWKV",
                  n_layer=N_LAYER, n_embd=N_EMBD)
model = GPT(cfg).cuda()
model.train()

n_params = sum(p.numel() for p in model.parameters())
lif_params_names = [n for n, _ in model.named_parameters() if 'lif' in n]
print(f"  Всего параметров:   {n_params:,}")
print(f"  LIF-параметров:     {len(lif_params_names)}  ({', '.join(lif_params_names[:4])} …)\n")

# ── Оптимизатор (обучаем ВСЕ параметры) ───────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Утилита: вытащить tau/threshold по слоям ─────────────────────────────────
def snapshot_lif():
    snap = []
    for i, block in enumerate(model.blocks):
        with torch.no_grad():
            snap.append({
                "layer": i,
                "lif1_tau":  block.lif1.tau.item(),
                "lif1_thr":  block.lif1.threshold.item(),
                "lif2_tau":  block.lif2.tau.item(),
                "lif2_thr":  block.lif2.threshold.item(),
            })
    return snap

def print_lif(snap, step, loss):
    print(f"\n{'─'*62}")
    print(f"  Шаг {step:4d}  |  loss={loss:.4f}")
    print(f"  {'Layer':>5}  {'lif1 τ':>8} {'lif1 θ':>8}  {'lif2 τ':>8} {'lif2 θ':>8}")
    for s in snap:
        print(f"  {s['layer']:>5}  {s['lif1_tau']:>8.4f} {s['lif1_thr']:>8.4f}"
              f"  {s['lif2_tau']:>8.4f} {s['lif2_thr']:>8.4f}")
    taus = [s['lif1_tau'] for s in snap] + [s['lif2_tau'] for s in snap]
    layers = [s['layer'] for s in snap] * 2
    corr = float(np.corrcoef(layers, taus)[0, 1])
    print(f"  Корреляция (layer, τ): {corr:+.4f}", end="")
    if corr > 0.3:   print("  ← τ растёт с глубиной ✓")
    elif corr < -0.3: print("  ← τ убывает с глубиной (обратная)")
    else:             print("  ← нет выраженной зависимости")

# ── Начальный снимок ──────────────────────────────────────────────────────────
history = []  # list of {step, loss, lif_snap}

snap0 = snapshot_lif()
print_lif(snap0, step=0, loss=float('nan'))
history.append({"step": 0, "loss": None, "lif": snap0})

# ── Цикл обучения ─────────────────────────────────────────────────────────────
print(f"\nОбучение {N_STEPS} шагов (lr={LR}, batch={BATCH_SIZE}, ctx={CTX_LEN}) …")
losses = []
t0 = time.time()

for step in range(1, N_STEPS + 1):
    x, y = get_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, targets=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    if step % LOG_EVERY == 0:
        mean_loss = float(np.mean(losses[-LOG_EVERY:]))
        snap = snapshot_lif()
        print_lif(snap, step=step, loss=mean_loss)
        history.append({"step": step, "loss": mean_loss, "lif": snap})

        elapsed = time.time() - t0
        eta = elapsed / step * (N_STEPS - step)
        print(f"  Прошло: {elapsed/60:.1f} мин  |  Осталось: ~{eta/60:.1f} мин")

# ── Финальный анализ ──────────────────────────────────────────────────────────
final_snap = history[-1]["lif"]
print(f"\n{'='*62}")
print("ИТОГОВЫЙ АНАЛИЗ")
print(f"{'='*62}")

taus  = [s['lif1_tau'] for s in final_snap] + [s['lif2_tau'] for s in final_snap]
thrs  = [s['lif1_thr'] for s in final_snap] + [s['lif2_thr'] for s in final_snap]
lyrs  = list(range(N_LAYER)) * 2
corr_tau = float(np.corrcoef(lyrs, taus)[0, 1])
corr_thr = float(np.corrcoef(lyrs, thrs)[0, 1])

print(f"\nКорреляция (layer, tau):       {corr_tau:+.4f}")
print(f"Корреляция (layer, threshold): {corr_thr:+.4f}")

print("\nTau по слоям (avg lif1+lif2):")
for s in final_snap:
    avg_tau = (s['lif1_tau'] + s['lif2_tau']) / 2
    bar = "█" * max(1, int(avg_tau * 8))
    print(f"  Layer {s['layer']}: τ={avg_tau:.4f}  {bar}")

print(f"\nStart tau (layer 0): {(final_snap[0]['lif1_tau'] + final_snap[0]['lif2_tau'])/2:.4f}")
print(f"End   tau (layer {N_LAYER-1}): {(final_snap[-1]['lif1_tau'] + final_snap[-1]['lif2_tau'])/2:.4f}")

if abs(corr_tau) < 0.2:
    print("\n→ tau не коррелирует с глубиной слоя.")
    print("  Возможно, tau=2.0 уже оптимален для всех слоёв,")
    print("  или модель слишком маленькая для дифференциации динамики.")
elif corr_tau > 0.3:
    print("\n→ ГИПОТЕЗА ПОДТВЕРЖДЕНА: τ растёт с глубиной.")
    print("  Нижние слои предпочитают быстрые нейроны (лексика),")
    print("  верхние — медленную интеграцию (синтаксис/семантика).")
else:
    print("\n→ ОБРАТНАЯ ЗАВИСИМОСТЬ: τ убывает с глубиной.")
    print("  Интерпретация: верхние слои обрабатывают более локальные признаки.")

# ── Сохранение результатов ────────────────────────────────────────────────────
result = {
    "model": f"SpikeGPT-small-{N_LAYER}L-{N_EMBD}D",
    "n_params": n_params,
    "n_steps":  N_STEPS,
    "lr":       LR,
    "batch_size": BATCH_SIZE,
    "ctx_len":    CTX_LEN,
    "final_loss": float(np.mean(losses[-100:])),
    "corr_tau":   corr_tau,
    "corr_thr":   corr_thr,
    "history":    history,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nРезультаты сохранены: {OUT_JSON}")
print(f"Общее время обучения: {(time.time()-t0)/60:.1f} мин")
