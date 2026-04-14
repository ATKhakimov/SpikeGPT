"""
Визуализация результатов LearnableLIF.

Читает:
  analysis/learnable_lif_small_run.json   (LearnableLIF, 4L×128D)
  analysis/fixed_tau_small_run.json       (Fixed τ=2.0, 4L×128D)

Создаёт 3 фигуры в analysis/figures/:
  lif_loss_comparison.png   — кривые loss: LearnableLIF vs Fixed τ
  lif_tau_evolution.png     — τ по слоям во время обучения
  lif_tau_final.png         — финальные τ по слоям (lif1 vs lif2)

Запуск: python analysis/plot_learnable_lif.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LEARN_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "learnable_lif_small_run.json")
FIXED_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixed_tau_small_run.json")
LOG_PATH   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "wik8-0.01.txt")

# ── Цвета и стиль ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
})

LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 4 слоя
LIF1_COLOR   = "#2196F3"
LIF2_COLOR   = "#FF5722"

# ── Загрузка данных ────────────────────────────────────────────────────────────
print("Загрузка JSON …")
with open(LEARN_JSON, encoding="utf-8") as f:
    learn = json.load(f)

has_fixed = os.path.exists(FIXED_JSON)
if has_fixed:
    with open(FIXED_JSON, encoding="utf-8") as f:
        fixed = json.load(f)
    print(f"  Fixed τ JSON найден: {FIXED_JSON}")
else:
    print(f"  Fixed τ JSON не найден — на графике loss покажем только LearnableLIF")

# Парсим лог обучения большой модели (100M, fixed τ=2.0)
big_train_epochs, big_train_loss = [], []
big_valid_epochs, big_valid_loss = [], []
big_valid_ppl = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                epoch = int(parts[0])
                split = parts[1]
                loss  = float(parts[2])
                ppl   = float(parts[3])
            except:
                continue
            if split == "train":
                big_train_epochs.append(epoch)
                big_train_loss.append(loss)
            elif split == "valid":
                big_valid_epochs.append(epoch)
                big_valid_loss.append(loss)
                big_valid_ppl.append(ppl)
    print(f"  Лог большой модели: {len(big_train_epochs)} train, {len(big_valid_epochs)} valid эпох")
else:
    print(f"  Лог {LOG_PATH} не найден")

n_layer = learn["n_layer"] if "n_layer" in learn else 4

# ── История tau по слоям ───────────────────────────────────────────────────────
steps_tau  = [h["step"] for h in learn["history"] if h["step"] > 0]

# shape: [n_checkpoints, n_layer]
lif1_tau_hist = np.array([
    [h["lif"][i]["lif1_tau"] for i in range(n_layer)]
    for h in learn["history"] if h["step"] > 0
])
lif2_tau_hist = np.array([
    [h["lif"][i]["lif2_tau"] for i in range(n_layer)]
    for h in learn["history"] if h["step"] > 0
])

# Финальный снапшот
final_lif = learn["history"][-1]["lif"]

# ══════════════════════════════════════════════════════════════════════════════
# Фигура 1a: Кривая обучения большой модели (100M, Fixed τ=2.0)
# ══════════════════════════════════════════════════════════════════════════════
print("Строим lif_big_model_training.png …")

if big_train_epochs:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(big_train_epochs, big_train_loss, color="#2196F3", lw=1.5,
            alpha=0.6, label="Train loss")
    if big_valid_epochs:
        ax.plot(big_valid_epochs, big_valid_loss, color="#FF5722", lw=2.0,
                marker="o", markersize=4, label="Valid loss")

        best_idx = int(np.argmin(big_valid_loss))
        ax.axvline(big_valid_epochs[best_idx], color="gray", lw=1.2,
                   linestyle=":", alpha=0.8)
        ax.annotate(
            f"Best valid loss={big_valid_loss[best_idx]:.3f}\n"
            f"PPL={big_valid_ppl[best_idx]:.2f}  (epoch {big_valid_epochs[best_idx]})",
            xy=(big_valid_epochs[best_idx], big_valid_loss[best_idx]),
            xytext=(20, 30), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            fontsize=9, color="#333333",
        )

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Кривая обучения SpikeGPT Russian (100M, Fixed τ=2.0)\n"
                 "12L × 512D, ruGPT-3 BPE, корпус Тайга ~1.8B токенов, NVIDIA A100")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "lif_big_model_training.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранено: {out}")
else:
    print("  Пропускаем (нет данных лога)")

# ══════════════════════════════════════════════════════════════════════════════
# Фигура 1b: Нормализованное сравнение loss (LearnableLIF vs Fixed τ=2.0)
# Нормировка: loss / loss[0] — сравниваем скорость сходимости, не абсолют
# ══════════════════════════════════════════════════════════════════════════════
print("Строим lif_loss_comparison.png …")

learn_steps = [h["step"] for h in learn["history"] if h["step"] > 0 and h["loss"] is not None]
learn_loss  = [h["loss"] for h in learn["history"] if h["step"] > 0 and h["loss"] is not None]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# — Левый: абсолютный loss маленькой модели ───────────────────────────────
ax = axes[0]
ax.plot(learn_steps, learn_loss, color="#2196F3", lw=2.0,
        label="LearnableLIF 4L×128D")
if has_fixed:
    fixed_steps = [h["step"] for h in fixed["history"]]
    fixed_loss  = [h["loss"] for h in fixed["history"]]
    ax.plot(fixed_steps, fixed_loss, color="#FF5722", lw=2.0,
            linestyle="--", label="Fixed τ=2.0  4L×128D")
    dl = fixed_loss[-1] - learn_loss[-1]
    ax.annotate(f"Δ={dl:.3f}", xy=(learn_steps[-1], (learn_loss[-1]+fixed_loss[-1])/2),
                xytext=(-90, 0), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=1),
                fontsize=9, color="gray")
ax.set_xlabel("Шаг обучения")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Абсолютный loss\n(одинаковая архитектура 4L×128D)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# — Правый: нормализованный loss обеих моделей ────────────────────────────
ax = axes[1]
if learn_loss:
    norm_learn = [l / learn_loss[0] for l in learn_loss]
    ax.plot(learn_steps, norm_learn, color="#2196F3", lw=2.0,
            label="LearnableLIF 4L×128D")

if has_fixed and fixed_loss:
    norm_fixed = [l / fixed_loss[0] for l in fixed_loss]
    ax.plot(fixed_steps, norm_fixed, color="#FF5722", lw=2.0,
            linestyle="--", label="Fixed τ=2.0  4L×128D")

if big_train_loss:
    # Берём loss по эпохам, нормируем на первое значение
    # Масштабируем эпохи → шаги (приблизительно, для визуализации)
    steps_per_epoch = learn_steps[-1] / len(learn_steps) * (len(big_train_epochs) / len(big_train_epochs))
    # Отображаем эпохи в шаги пропорционально последнему шагу LearnableLIF
    max_step  = learn_steps[-1] if learn_steps else 2000
    max_epoch = big_train_epochs[-1]
    scaled_epochs = [e / max_epoch * max_step for e in big_train_epochs]
    norm_big  = [l / big_train_loss[0] for l in big_train_loss]
    ax.plot(scaled_epochs, norm_big, color="#4CAF50", lw=1.5, alpha=0.7,
            linestyle="-.", label="Fixed τ=2.0  100M (эпохи→шаги)")

ax.set_xlabel("Шаг обучения (нормализован)")
ax.set_ylabel("Loss / Loss₀  (скорость сходимости)")
ax.set_title("Нормализованный loss\n(сравнение скорости сходимости)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

fig.suptitle("LearnableLIF vs Fixed τ=2.0 — сравнение обучения", fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "lif_loss_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранено: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Фигура 2: Эволюция τ по слоям во время обучения
# ══════════════════════════════════════════════════════════════════════════════
print("Строим lif_tau_evolution.png …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, tau_hist, title, marker in zip(
    axes,
    [lif1_tau_hist, lif2_tau_hist],
    ["lif1 (после TimeMix / внимание)", "lif2 (после ChannelMix / FFN)"],
    ["o", "s"],
):
    for i in range(n_layer):
        ax.plot(steps_tau, tau_hist[:, i],
                color=LAYER_COLORS[i], lw=1.8,
                marker=marker, markersize=3, markevery=3,
                label=f"Layer {i}")
    # Пунктир tau=2.0 (baseline)
    ax.axhline(2.0, color="gray", lw=1.2, linestyle=":", label="τ=2.0 (fixed)")
    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel("τ (membrane time constant)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

fig.suptitle("Эволюция обучаемого τ по слоям во время обучения\n(4L × 128D, LearnableLIF)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "lif_tau_evolution.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранено: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Фигура 3: Финальный τ по слоям (lif1 vs lif2)
# ══════════════════════════════════════════════════════════════════════════════
print("Строим lif_tau_final.png …")

layers = list(range(n_layer))
tau1_final = [final_lif[i]["lif1_tau"] for i in layers]
tau2_final = [final_lif[i]["lif2_tau"] for i in layers]
thr1_final = [final_lif[i]["lif1_thr"] for i in layers]
thr2_final = [final_lif[i]["lif2_thr"] for i in layers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# — τ по слоям ─────────────────────────────────────────────────────────────
x = np.arange(n_layer)
w = 0.35
bars1 = ax1.bar(x - w/2, tau1_final, w, label="lif1 (TimeMix)", color=LIF1_COLOR, alpha=0.85)
bars2 = ax1.bar(x + w/2, tau2_final, w, label="lif2 (ChannelMix)", color=LIF2_COLOR, alpha=0.85)
ax1.axhline(2.0, color="gray", lw=1.2, linestyle=":", label="τ=2.0 (fixed baseline)")

# Значения над столбцами
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8.5)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8.5)

ax1.set_xticks(x)
ax1.set_xticklabels([f"Layer {i}" for i in layers])
ax1.set_ylabel("τ (membrane time constant)")
ax1.set_title("Финальный τ по слоям\n(LearnableLIF, 2000 шагов)")
ax1.legend()
ax1.set_ylim(1.0, max(max(tau1_final), max(tau2_final)) + 0.3)

# — threshold по слоям ────────────────────────────────────────────────────
bars3 = ax2.bar(x - w/2, thr1_final, w, label="lif1 threshold", color=LIF1_COLOR, alpha=0.85)
bars4 = ax2.bar(x + w/2, thr2_final, w, label="lif2 threshold", color=LIF2_COLOR, alpha=0.85)
ax2.axhline(1.0, color="gray", lw=1.2, linestyle=":", label="θ=1.0 (fixed baseline)")

for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8.5)
for bar in bars4:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8.5)

ax2.set_xticks(x)
ax2.set_xticklabels([f"Layer {i}" for i in layers])
ax2.set_ylabel("Порог возбуждения θ")
ax2.set_title("Финальный threshold по слоям\n(LearnableLIF, 2000 шагов)")
ax2.legend()
ax2.set_ylim(0.5, max(max(thr1_final), max(thr2_final)) + 0.1)

fig.suptitle("Выученные параметры LIF-нейронов по слоям\n"
             "Гипотеза: нижние слои → малый τ (лексика), верхние → большой τ (синтаксис)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "lif_tau_final.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Сохранено: {out}")

print("\nВсе графики готовы:")
for name in ["lif_loss_comparison.png", "lif_tau_evolution.png", "lif_tau_final.png"]:
    print(f"  analysis/figures/{name}")
