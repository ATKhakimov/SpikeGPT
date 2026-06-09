# SpikeRuGPT

SpikeRuGPT — русскоязычная адаптация архитектуры SpikeGPT: autoregressive language model в стиле RWKV с импульсными LIF-нейронами.

Репозиторий содержит код первой русскоязычной SpikeGPT-модели, воспроизводимый пайплайн v1, эксперименты с SFT, анализ спайковой активности и материалы для статьи/постера.

![SpikeGPT architecture](static/spikegpt.png)

## Статус

Это исследовательский репозиторий, а не production-ready чат-модель.

Главный результат сейчас — русскоязычное базовое языковое моделирование и измеримый анализ нейроморфной спарсити. SFT-чекпоинт добавлен как диагностический артефакт: он лучше держит формат и убирает видимые артефакты загрязненных instruction-данных, но пока не является надежной factual QA/chat-моделью.

## Модели и веса

Веса не хранятся в git. Публичные артефакты лежат на Hugging Face:

- v0 Taiga checkpoint и tokenizer: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga
- v1 SFT v2 superclean checkpoint: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga/tree/main/sft-v2-superclean

| Линия | Параметры | Tokenizer | Данные | Роль |
|---|---:|---|---|---|
| `v0_taiga_100m` | 92.4M | ruGPT-3 BPE, 50k | Taiga, около 1.8B токенов | первая русскоязычная baseline-модель |
| `v1_base_74m` | 73.7M | SentencePiece BPE, 32k | очищенный смешанный русский корпус, около 0.95B показанных токенов | воспроизводимый pretrain-пайплайн |
| `v1_sft_v2_superclean` | 73.7M | SentencePiece BPE, 32k | 45k коротких одноходовых русских инструкций | диагностический SFT |
| `SpikeGPT-OpenWebText-216M` | 215.4M | GPT-NeoX tokenizer | OpenWebText | англоязычная референсная модель |

## Ключевые результаты

| Эксперимент | Результат |
|---|---|
| v0 Taiga validation perplexity | best validation PPL 59.79 |
| Спайковая активность v0 на русском | mean firing rate 33.2% |
| Англоязычная SpikeGPT reference activity | mean firing rate 21.7% |
| v1 base final evaluation | val_wiki PPL 69.90, val_mixed PPL 118.27 |
| SFT v2 supervised validation | loss 4.0997, PPL 60.32 |

Основные выводы:

- SpikeGPT-подобные импульсные языковые модели можно обучать на русском тексте.
- В нашей постановке русский текст дал более высокий firing rate, чем англоязычная reference-модель, поэтому язык и корпус влияют на нейроморфный event budget.
- Для малых моделей tokenizer важен особенно сильно: 32k SentencePiece BPE уменьшает число параметров без заметной потери плотности кодирования на измеренном validation-фрагменте.
- Строгая очистка SFT-данных убирает `role/content` и code-like артефакты, но SFT сам по себе не компенсирует слабость маленькой base-модели.

## Графики

### Динамика обучения

![Training loss by tokens](ARTICLE/figures/training_loss_by_tokens.png)

### Сравнение спайковой активности

![Sparsity summary](analysis/figures/sparsity_summary.png)

### Послойная спайковая активность

![Spike sparsity by layer](analysis/figures/spike_sparsity.png)

### Обучаемые LIF-параметры

![Learnable LIF tau](analysis/figures/lif_tau_final.png)

Больше графиков и таблиц для статьи/постера:

- [`ARTICLE/poster_assets/`](ARTICLE/poster_assets/)
- [`ARTICLE/figures/`](ARTICLE/figures/)
- [`analysis/figures/`](analysis/figures/)

## Структура репозитория

```text
src/                    модель SpikeGPT, trainer utilities, vendored spikingjelly subset
cuda/                   CUDA-ядра WKV
train.py                original v0 training entrypoint
generate.py             generation entrypoint для v0
demo.py                 continuation-prompt demo для v0
scripts/                v1 data/training/evaluation/SFT/analysis tools
scripts/data/           inspection, filtering, tokenizer и shard builders
configs/                конфиги источников данных и SFT
analysis/               старый v0 sparsity/LIF analysis и графики
ARTICLE/                статья, technical logs, poster assets и SFT-анализ
NLU/                    original SpikeGPT NLU evaluation scripts
static/                 статические изображения проекта
```

Локальные run-артефакты игнорируются:

- `data/`
- `tokenizer/`
- `checkpoints/`
- `models/`
- `reports/`
- `logs/`

Они исключены из git намеренно, потому что реальные training/eval runs создают большие файлы.

## Карта документации

- Черновик статьи: [`ARTICLE/spikerugpt_conference_article_draft.md`](ARTICLE/spikerugpt_conference_article_draft.md)
- Технический лог обучения: [`ARTICLE/spikerugpt_technical_log.md`](ARTICLE/spikerugpt_technical_log.md)
- План обучения и данных: [`ARTICLE/spikerugpt_training_plan.md`](ARTICLE/spikerugpt_training_plan.md)
- Сравнение v0/v1: [`ARTICLE/spikerugpt_v0_v1_comparison.md`](ARTICLE/spikerugpt_v0_v1_comparison.md)
- Poster assets: [`ARTICLE/poster_assets/README.md`](ARTICLE/poster_assets/README.md)
- SFT v2 analysis: [`ARTICLE/sft_v2_superclean/README.md`](ARTICLE/sft_v2_superclean/README.md)
- Data pipeline docs: [`scripts/data/README.md`](scripts/data/README.md)

## Установка

Сначала установите PyTorch под вашу версию CUDA, затем зависимости проекта:

```bash
pip install -r requirements.txt
pip install -r requirements_data.txt
```

Для RTX 50xx / CUDA 12.8 окружений есть отдельные заметки:

```bash
requirements_runpod_cu128.txt
```

Некоторые скрипты скачивают датасеты/модели с Hugging Face. Для них нужен `HF_TOKEN` или авторизация через `huggingface_hub`.

## v1 Data and Training Pipeline

Проверить источники данных:

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl
```

Собрать sample для tokenizer-а и обучить 32k SentencePiece:

```bash
python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --out data/tokenizer_sample/spikerugpt_tokenizer_sample.txt

python scripts/data/train_sentencepiece.py \
  --input data/tokenizer_sample/spikerugpt_tokenizer_sample.txt \
  --model-prefix tokenizer/spikerugpt-bpe-32k \
  --vocab-size 32000
```

Собрать pretraining shards:

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_1b \
  --max-tokens 1000000000
```

Запустить автономный pretrain:

```bash
python scripts/run_autonomous_training.py \
  --manifest data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --run-id autonomous-ctx1024-1b-bf16-5d \
  --precision bf16
```

Собрать superclean SFT dataset и запустить SFT:

```bash
python scripts/data/build_sft_superclean.py \
  --input data/sft/spikerugpt_sft_clean_final.jsonl \
  --out data/sft/spikerugpt_sft_superclean_v2.jsonl

python scripts/train_sft.py \
  --base-checkpoint checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt \
  --sft-data data/sft/spikerugpt_sft_superclean_v2.jsonl \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --run-id sft-step43674-v2-superclean
```

## Генерация v0

Оригинальный v0-код ожидает локальный checkpoint и tokenizer. Скачайте их с Hugging Face, затем:

```bash
python generate.py \
  --prompt "Осенний лес был тих и задумчив." \
  --checkpoint checkpoints/spikegpt-ru-175.pth \
  --temperature 0.85 \
  --top_p 0.9
```

## Анализ

Скрипты для spiking activity и сравнения моделей:

```bash
python scripts/analyze_spiking_activity.py
python scripts/compare_v0_v1_eval.py
python scripts/compare_continuation_demo.py
python scripts/compare_sft_generations.py
python scripts/build_poster_assets.py
```

Статья пересобирается из markdown:

```bash
python scripts/article/build_conference_docx.py
```

Сгенерированный `.docx` намеренно игнорируется git.

## Цитирование

Если используете этот репозиторий, цитируйте оригинальную статью SpikeGPT:

```bibtex
@article{zhu2023spikegpt,
    title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
    author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
    journal = {arXiv preprint arXiv:2302.13939},
    year    = {2023}
}
```

## Лицензия

Репозиторий распространяется под лицензией MIT. См. [`LICENSE`](LICENSE).
