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

## Конфигурация основной модели v0

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

Файлы основной модели:

- `spikegpt-ru-175.pth`
- `tokenizer/`

## Дополнительная SFT-версия v1

В этот же репозиторий добавлена экспериментальная SFT-версия:

- checkpoint: `sft-v2-superclean/final.pt`
- tokenizer: `tokenizer-sp32k/spikerugpt-bpe-32k.model`
- описание: `sft-v2-superclean/README.md`
- отчеты: `sft-v2-superclean/reports/`

Важно: это не дообучение старого `spikegpt-ru-175.pth`. SFT v2 построена на новой компактной base-модели SpikeRuGPT v1:

| Параметр | Значение |
|---|---|
| Архитектура | SpikeGPT/RWKV + LIF |
| Параметры | 73.7M |
| Слоёв | 12 |
| d_model | 512 |
| Контекст | 1024 |
| Tokenizer | SentencePiece BPE, vocab=32k |
| Base checkpoint | step 43674 |
| Base tokens seen | 944,590,848 |
| SFT dataset | 45,000 short one-turn Russian instruction examples |
| SFT validation loss | 4.0997 |
| SFT validation PPL | 60.32 |

SFT v2 собрана как диагностический эксперимент по очистке инструкционных данных. Она устраняет видимые технические артефакты, обнаруженные в предыдущей SFT v1, но не является полноценной ассистентской моделью. Для надежных factual QA и сложных инструкций требуется более сильное предобучение, больший масштаб модели или более специализированное SFT/RLHF-обучение.

SHA256:

| Файл | SHA256 |
|---|---|
| `sft-v2-superclean/final.pt` | `a5ddc7f00111f0a721ea5373c0f2a8e75aeb8984525c917efb1877897cc313b9` |
| `tokenizer-sp32k/spikerugpt-bpe-32k.model` | `ee47e1dd17fa209f91342a78308e40b85539ff597719ee8e2c786092571ecd8d` |
| `tokenizer-sp32k/spikerugpt-bpe-32k.vocab` | `09723941d23ff20869d54f735044eb75d2ae12f7f4cd7d056d168eb11bea9c35` |

## Использование

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Загрузка токенизатора v0
tokenizer = AutoTokenizer.from_pretrained("Koras1k/spikerugpt-100M-Taiga")

# Загрузка модели — см. github.com/ATKhakimov/SpikeRuGPT
# (требует src/model.py и CUDA-ядро wkv_cuda.cu)
```

Полный код генерации: [github.com/ATKhakimov/SpikeRuGPT](https://github.com/ATKhakimov/SpikeRuGPT)

## Результаты v0

| Метрика | Значение |
|---|---|
| Valid Perplexity | ~67 (эпоха 175) |
| Firing rate (LIF) | 33.2% активных нейронов |
| Молчащие нейроны | 66.8% |

Сравнение нейроморфной спарсити с английской моделью (SpikeGPT-OpenWebText-216M):

- Русский: 33.2% активных нейронов
- Английский: 21.7% активных нейронов
- Русский язык требует на **53% больше спайков** в данной экспериментальной постановке.

## Цитирование

```bibtex
@article{zhu2023spikegpt,
    title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
    author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
    journal = {arXiv preprint arXiv:2302.13939},
    year    = {2023}
}
```
