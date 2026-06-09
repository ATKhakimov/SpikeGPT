# SpikeRuGPT

**SpikeRuGPT** — исследовательский проект по адаптации импульсной языковой модели SpikeGPT к русскому языку и анализу нейроморфной активности при обработке русскоязычного текста.

Модель относится к семейству autoregressive language models и использует RWKV-подобное рекуррентное смешение токенов в сочетании с LIF-нейронами. Такая архитектура позволяет рассматривать языковое моделирование не только через perplexity и качество генерации, но и через внутреннюю событийную активность: firing rate, долю молчащих нейронов и послойный профиль спайков.


![Архитектура SpikeGPT](static/spikegpt.png)

## Цель работы

Цель проекта — проверить применимость SpikeGPT-подобной импульсной архитектуры к русскому языку и оценить, как морфологически богатый язык влияет на нейроморфную активность модели.

## Научный вклад

В рамках проекта:

- обучена русскоязычная SpikeGPT-подобная модель на корпусе «Тайга»;
- проведено сравнение спайковой активности русскоязычной модели и открытой англоязычной SpikeGPT-OpenWebText-216M;
- показано, что русскоязычная модель в данной постановке имеет более высокий firing rate;
- исследована послойная структура спайковой активности;
- проведена серия экспериментов с обучаемыми параметрами LIF-нейронов;
- подготовлен воспроизводимый контур для дальнейших экспериментов с tokenizer-ом, корпусом и инструкционным дообучением.

## Основные результаты

Первая русскоязычная модель SpikeRuGPT была обучена на корпусе «Тайга» объемом около 1,8 млрд токенов. Модель достигла validation perplexity 59,79 и продемонстрировала способность генерировать связный русский текст в режиме продолжения.

| Результат | Значение |
|---|---:|
| Размер основной русскоязычной модели | 92,4M параметров |
| Обучающий корпус | «Тайга», около 1,8B токенов |
| Лучшая validation perplexity | 59,79 |
| Средний firing rate русскоязычной модели | 33,2% |
| Средний firing rate англоязычной SpikeGPT-OpenWebText-216M | 21,7% |
| Относительное увеличение числа спайков для русского | около 1,53x |

Дополнительно была проведена воспроизводимая серия экспериментов с компактной моделью на 73,7M параметров, SentencePiece BPE-словарем на 32 тыс. токенов и очищенным русскоязычным корпусом объемом около 0,95B показанных токенов. Эта серия использовалась для проверки подготовки данных, оценки tokenizer-а, анализа промежуточных checkpoint-ов и экспериментов с instruction tuning.

## Нейроморфная спарсити

Firing rate измеряет среднюю долю LIF-нейронов, породивших spike-событие при обработке текста. Для нейроморфного аппаратного обеспечения эта величина важна, поскольку число спайков связано с числом событийных операций.

![Сравнение firing rate](analysis/figures/sparsity_summary.png)

В проведенной постановке русскоязычная модель имеет более высокий firing rate, чем англоязычная модель для сравнения. Это не является универсальным утверждением о всех языках и всех архитектурах, поскольку модели различаются размером и обучающими корпусами. Однако результат показывает, что язык и корпус оценки могут существенно влиять на оценку энергобюджета спайковой языковой модели.

Послойный анализ показывает U-образный профиль активности: высокая активность в начальных слоях, спад в средних и рост ближе к выходу.

![Послойная спайковая активность](analysis/figures/spike_sparsity.png)

Наиболее выраженное различие между русскоязычной и англоязычной моделями наблюдается в средних слоях. Возможная интерпретация состоит в том, что русская модель вынуждена активнее кодировать морфологические зависимости: падеж, согласование, грамматическую роль слова и связи между удаленными токенами.

## Обучение

Динамика обучения первой русскоязычной модели показывает устойчивую сходимость на корпусе «Тайга».

![Кривая обучения](analysis/figures/training_curve.png)

Для компактной версии модели дополнительно фиксировалась динамика loss по числу показанных токенов.

![Loss по токенам](ARTICLE/figures/training_loss_by_tokens.png)

## LIF-динамика

В отдельной серии экспериментов исследовалась модификация LearnableLIF, в которой мембранная постоянная времени `tau` и порог возбуждения являются обучаемыми параметрами. В экспериментах наблюдался рост `tau` с глубиной слоя.

![Финальные значения tau](analysis/figures/lif_tau_final.png)

Этот результат согласуется с иерархической интерпретацией языковой модели: нижние слои обрабатывают более локальные признаки, а верхние интегрируют более длинный контекст.

## Модели и артефакты

Веса моделей не хранятся в git. Публичные артефакты доступны на Hugging Face:

- основная модель SpikeRuGPT на корпусе «Тайга»: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga
- дополнительная SFT-версия компактной модели: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga/tree/main/sft-v2-superclean

| Модель | Параметры | Tokenizer | Данные | Назначение |
|---|---:|---|---|---|
| SpikeRuGPT Taiga | 92,4M | ruGPT-3 BPE, 50k | «Тайга», около 1,8B токенов | основная русскоязычная модель |
| SpikeRuGPT compact | 73,7M | SentencePiece BPE, 32k | очищенный смешанный русский корпус, около 0,95B токенов | воспроизводимая серия экспериментов |
| SpikeRuGPT compact SFT | 73,7M | SentencePiece BPE, 32k | 45k очищенных русских инструкций | анализ влияния instruction tuning и очистки данных |
| SpikeGPT English | 215,4M | GPT-NeoX tokenizer | OpenWebText | англоязычная модель для сравнения |

SFT-эксперименты рассматриваются как дополнительная часть исследования данных. Они показывают, что строгая очистка инструкционных примеров устраняет служебные артефакты формата, но качество ответов малой модели по-прежнему ограничивается уровнем базового предобучения.

## Ограничения

Сравнение русскоязычной и англоязычной моделей не является строго контролируемым языковым экспериментом: модели различаются размером, tokenizer-ом и обучающими корпусами. Поэтому результаты по firing rate следует рассматривать как практическое наблюдение для данной экспериментальной постановки.

Сгенерированный текст не должен рассматриваться как источник фактических сведений. Основная модель обучалась как базовая языковая модель продолжения текста, а не как диалоговая система.

## Структура репозитория

```text
src/                    архитектура SpikeGPT, LIF/RWKV-блоки и training utilities
cuda/                   CUDA-ядра WKV-оператора
train.py                обучение основной модели
generate.py             генерация текста основной моделью
demo.py                 демонстрационные continuation-prompt-ы
scripts/                подготовка данных, обучение, SFT, оценка и анализ
scripts/data/           фильтрация корпусов, tokenizer, validation splits, token shards
configs/                конфигурации источников данных и SFT-наборов
analysis/               анализ спарсити, LearnableLIF и графики для первой модели
ARTICLE/                статья, технические заметки, материалы для постера и дополнительные отчеты
NLU/                    оригинальные скрипты оценки SpikeGPT на NLU-задачах
static/                 статические изображения
```

Крупные локальные артефакты обучения исключены из git: `data/`, `tokenizer/`, `checkpoints/`, `models/`, `reports/`, `logs/`.

## Документация

- Черновик статьи: [`ARTICLE/spikerugpt_conference_article_draft.md`](ARTICLE/spikerugpt_conference_article_draft.md)
- Технический лог обучения: [`ARTICLE/spikerugpt_technical_log.md`](ARTICLE/spikerugpt_technical_log.md)
- План подготовки данных и обучения: [`ARTICLE/spikerugpt_training_plan.md`](ARTICLE/spikerugpt_training_plan.md)
- Сравнение русскоязычных версий: [`ARTICLE/spikerugpt_v0_v1_comparison.md`](ARTICLE/spikerugpt_v0_v1_comparison.md)
- Материалы для постера: [`ARTICLE/poster_assets/README.md`](ARTICLE/poster_assets/README.md)
- Анализ SFT v2: [`ARTICLE/sft_v2_superclean/README.md`](ARTICLE/sft_v2_superclean/README.md)
- Описание data pipeline: [`scripts/data/README.md`](scripts/data/README.md)

## Воспроизведение

Сначала установите PyTorch под вашу версию CUDA, затем зависимости проекта:

```bash
pip install -r requirements.txt
pip install -r requirements_data.txt
```

Для окружений с CUDA 12.8 и RTX 50xx см.:

```bash
requirements_runpod_cu128.txt
```

Некоторые скрипты используют Hugging Face для загрузки моделей и датасетов. Для приватных или gated-источников требуется `HF_TOKEN` или авторизация через `huggingface_hub`.

### Подготовка данных компактной серии

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl

python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --out data/tokenizer_sample/spikerugpt_tokenizer_sample.txt

python scripts/data/train_sentencepiece.py \
  --input data/tokenizer_sample/spikerugpt_tokenizer_sample.txt \
  --model-prefix tokenizer/spikerugpt-bpe-32k \
  --vocab-size 32000
```

### Сборка pretraining shards

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_1b \
  --max-tokens 1000000000
```

### Обучение компактной модели

```bash
python scripts/run_autonomous_training.py \
  --manifest data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --run-id autonomous-ctx1024-1b-bf16-5d \
  --precision bf16
```

### Генерация основной моделью

После загрузки checkpoint-а и tokenizer-а с Hugging Face:

```bash
python generate.py \
  --prompt "Осенний лес был тих и задумчив." \
  --checkpoint checkpoints/spikegpt-ru-175.pth \
  --temperature 0.85 \
  --top_p 0.9
```

### Анализ

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

Сгенерированный `.docx` не хранится в git.

## Цитирование

При использовании репозитория следует цитировать исходную работу SpikeGPT:

```bibtex
@article{zhu2023spikegpt,
    title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
    author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
    journal = {arXiv preprint arXiv:2302.13939},
    year    = {2023}
}
```

## Лицензия

Код распространяется под лицензией MIT. См. [`LICENSE`](LICENSE).
