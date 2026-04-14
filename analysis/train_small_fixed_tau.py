"""
Baseline: та же маленькая модель (4L×128D), но tau зафиксирован = 2.0 для всех слоёв.
Сохраняет loss history в analysis/fixed_tau_small_run.json для сравнения с LearnableLIF.

Запуск: python analysis/train_small_fixed_tau.py
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json

N_LAYER    = 4
N_EMBD     = 128
CTX_LEN    = 512
VOCAB_SIZE = 50258

BATCH_SIZE = 16
N_STEPS    = 2000
LOG_EVERY  = 100
LR         = 6e-4
FIXED_TAU  = 2.0

DATA_PATH = "data/ru_train_full.npy"
OUT_JSON  = "analysis/fixed_tau_small_run.json"

os.environ["VOCAB_SIZE"] = str(VOCAB_SIZE)

from src.model import GPT, GPTConfig

print(f"Загрузка данных {DATA_PATH} …")
tokens = np.load(DATA_PATH, mmap_mode='r')
total  = len(tokens)
print(f"  Токенов: {total:,}\n")

def get_batch():
    ix = torch.randint(total - CTX_LEN - 1, (BATCH_SIZE,))
    x  = torch.stack([torch.from_numpy(tokens[i:i+CTX_LEN].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(tokens[i+1:i+CTX_LEN+1].astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()

print(f"Строим модель: {N_LAYER}L × {N_EMBD}D  (Fixed τ={FIXED_TAU})")
cfg   = GPTConfig(VOCAB_SIZE, CTX_LEN, model_type="RWKV",
                  n_layer=N_LAYER, n_embd=N_EMBD)
model = GPT(cfg).cuda()

# Замораживаем LIF-параметры — tau и threshold не обучаются
for name, param in model.named_parameters():
    if 'lif' in name:
        param.requires_grad_(False)

# Инициализируем все LIF с одинаковым tau=2.0 (честный baseline)
import torch.nn.functional as F
from src.model import LearnableLIFNode
with torch.no_grad():
    for block in model.blocks:
        for lif in (block.lif1, block.lif2):
            # log_tau такой, чтобы tau = 1 + softplus(log_tau) = 2.0
            target_tau = FIXED_TAU
            # softplus(x) = log(1 + exp(x)) = 1.0  =>  x ≈ 0.5413
            lif.log_tau.fill_(math.log(math.exp(target_tau - 1.0) - 1.0))
            # threshold = softplus(log_threshold) = 1.0  =>  log_threshold = log(exp(1)-1)
            lif.log_threshold.fill_(math.log(math.exp(1.0) - 1.0))

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Обучаемых параметров: {n_trainable:,}  (LIF заморожены)\n")

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR
)

model.train()
print(f"Обучение {N_STEPS} шагов (Fixed τ={FIXED_TAU}) …\n")
losses   = []
history  = []
t0 = time.time()

for step in range(1, N_STEPS + 1):
    x, y = get_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, targets=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()
    losses.append(loss.item())

    if step % LOG_EVERY == 0:
        mean_loss = float(np.mean(losses[-LOG_EVERY:]))
        history.append({"step": step, "loss": mean_loss})
        elapsed = time.time() - t0
        eta     = elapsed / step * (N_STEPS - step)
        print(f"  Шаг {step:4d}/{N_STEPS}  loss={mean_loss:.4f}"
              f"  [{elapsed/60:.1f} мин  ETA {eta/60:.1f} мин]")

result = {
    "model":      f"SpikeGPT-small-{N_LAYER}L-{N_EMBD}D-FixedTau{FIXED_TAU}",
    "fixed_tau":  FIXED_TAU,
    "n_steps":    N_STEPS,
    "lr":         LR,
    "batch_size": BATCH_SIZE,
    "ctx_len":    CTX_LEN,
    "final_loss": float(np.mean(losses[-100:])),
    "history":    history,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nFinal loss: {result['final_loss']:.4f}")
print(f"Результаты сохранены: {OUT_JSON}")
print(f"Время: {(time.time()-t0)/60:.1f} мин")
