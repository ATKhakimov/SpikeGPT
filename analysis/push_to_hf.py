"""
Публикация SpikeGPT Russian на HuggingFace Hub.

Запуск:
    HF_TOKEN=hf_ваш_токен python analysis/push_to_hf.py
"""

import os, sys, shutil
from pathlib import Path

HF_REPO_ID  = "Koras1k/spikerugpt-100M-Taiga"
CHECKPOINT  = "checkpoints/spikegpt-ru-175.pth"
TOKENIZER   = "tokenizer/rugpt3"
ROOT        = Path(__file__).parent.parent

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: передай токен через HF_TOKEN=hf_... python analysis/push_to_hf.py")
    sys.exit(1)

from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)

# ── 1. Создаём репозиторий ────────────────────────────────────────────────────
print(f"Создаём репозиторий {HF_REPO_ID} …")
create_repo(HF_REPO_ID, token=HF_TOKEN, repo_type="model",
            private=False, exist_ok=True)
print("  OK")

# ── 2. Model card ─────────────────────────────────────────────────────────────
model_card = """\
---
language: ru
license: mit
tags:
  - spiking-neural-network
  - rwkv
  - russian
  - neuromorphic
  - language-model
datasets:
  - taiga
library_name: pytorch
---

# SpikeGPT Russian — 100M

Адаптация импульсной языковой модели **SpikeGPT** ([Zhu et al., 2023](https://arxiv.org/abs/2302.13939)) для русского языка.

## Описание

SpikeGPT основана на архитектуре **RWKV** с бинарными событийно-управляемыми **LIF-нейронами** (Leaky Integrate-and-Fire), что делает её пригодной для нейроморфного аппаратного обеспечения (Intel Loihi, BrainScaleS).

Данная модель — первая публичная версия SpikeGPT, обученная на русскоязычном корпусе.

## Конфигурация

| Параметр | Значение |
|---|---|
| Архитектура | SpikeGPT (RWKV + MultiStepLIF) |
| Параметры | ~100M (12 слоёв, d_model=512) |
| Токенизатор | ruGPT-3 Large BPE (vocab=50 258) |
| Корпус | Тайга: taiga_stripped_rest + taiga_stripped_proza |
| Объём данных | ~1.8B токенов |
| Длина контекста | 1 024 токена |
| Оборудование | NVIDIA A100 SXM 80GB |
| Чекпоинт | Эпоха 175 |

## Использование

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("Koras1k/spikerugpt-100M-Taiga")

# Загрузка модели — см. github.com/Koras1k/SpikeGPT
# (требует src/model.py и CUDA-ядро wkv_cuda.cu)
```

Полный код генерации: [github.com/Koras1k/SpikeGPT](https://github.com/Koras1k/SpikeGPT)

## Результаты

| Метрика | Значение |
|---|---|
| Valid Perplexity | ~67 (эпоха 175) |
| Firing rate (LIF) | 33.2% активных нейронов |
| Молчащие нейроны | 66.8% |

Сравнение нейроморфной спарсити с английской моделью (SpikeGPT-OpenWebText-216M):
- Русский: 33.2% активных нейронов
- Английский: 21.7% активных нейронов
- Русский язык требует на **53% больше спайков** — следствие морфологической сложности.

## Цитирование

```bibtex
@article{zhu2023spikegpt,
    title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
    author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
    journal = {arXiv preprint arXiv:2302.13939},
    year    = {2023}
}
```
"""

card_path = ROOT / "HF_README.md"
card_path.write_text(model_card, encoding="utf-8")

# ── 3. Загружаем файлы ────────────────────────────────────────────────────────
uploads = [
    (str(card_path),               "README.md"),
    (str(ROOT / CHECKPOINT),       "spikegpt-ru-175.pth"),
]

# Токенизатор
for f in (ROOT / TOKENIZER).iterdir():
    if f.is_file():
        uploads.append((str(f), f"tokenizer/{f.name}"))

print(f"\nЗагружаем {len(uploads)} файлов в {HF_REPO_ID} …")
for local_path, repo_path in uploads:
    size_mb = os.path.getsize(local_path) / 1024**2
    print(f"  {repo_path}  ({size_mb:.1f} MB) …", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
    )
    print("OK")

# Убираем временный файл
card_path.unlink()

print(f"\nГотово! Модель доступна по адресу:")
print(f"  https://huggingface.co/{HF_REPO_ID}")
