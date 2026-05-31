# SpikeRuGPT: план дообучения и подготовки данных

Версия: 0.1  
Дата фиксации: 2026-06-01  
Назначение: рабочий план для перехода от текущей 100M proof-of-concept модели к воспроизводимому обучению более сильной русскоязычной SpikeGPT-модели на сервере с RTX 5090.

## 1. Главная цель

Текущая модель SpikeRuGPT показывает, что русскоязычная spiking language model работоспособна, но текущий результат лучше рассматривать как proof of concept, а не как финальную модель.

Основная цель следующего этапа:

- собрать воспроизводимый data pipeline;
- заменить слишком дорогой tokenizer на более подходящий русский/многоисточниковый tokenizer;
- обучить стабильную базовую модель;
- затем сделать quality annealing;
- только после этого делать SFT;
- параллельно логировать spiking-метрики, чтобы работа оставалась научной, а не только инженерной.

Целевой кандидат:

```text
name: SpikeRuGPT-150M-v1
architecture: SpikeGPT / RWKV-style + LIF
parameters: ~150M
context length: 1024
tokenizer: SentencePiece BPE 32k
precision: bf16
hardware: 1x RTX 5090
```

Текущая 100M модель остается baseline.

## 2. Почему не копировать Tiny-LLM напрямую

Tiny-LLM полезен как ориентир по режиму обучения: маленькую модель можно сильно улучшить большим количеством токенов. Но его архитектура и задача не совпадают с нашей.

Источник: [arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM)

Полезный вывод:

- маленькая модель может получать заметный выигрыш от большого token budget;
- контекст 1024 достаточен для первого сильного эксперимента;
- tokenizer 32k выглядит разумнее, чем 50k для малой/средней модели;
- качество данных и длительность обучения важнее экзотической архитектуры.

Но для нашей задачи важнее опыт из русскоязычной Habr-статьи:

- обучение с нуля на русском;
- практическая фильтрация данных;
- tokenizer;
- SFT;
- сохранение generations/checkpoints;
- понимание, что объем и качество корпуса сильнее влияют на результат, чем единичные архитектурные изменения.

Источник: [Habr, статья 1037532](https://habr.com/ru/articles/1037532/)

## 3. Целевая стратегия

Не скачивать все подряд. Вместо этого:

1. Проверить источники и способы загрузки.
2. Собрать tokenizer sample.
3. Обучить новый tokenizer.
4. Сделать фиксированные validation splits.
5. Провести маленькие smoke-тесты.
6. Провести 300M-token tokenizer/model pilot.
7. Запустить основной base pretraining.
8. Сделать quality annealing.
9. Сделать SFT.
10. Провести eval и ablations.

Критичный принцип: SFT нельзя делать до нормальной base model. Иначе модель научится форме диалога поверх слабого языкового ядра.

## 4. Проверенные источники данных

### 4.1 Pretraining

| Источник | Доступность | Роль | Комментарий |
|---|---:|---|---|
| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | HF, public | основной web corpus | брать `rus_Cyrl`, желательно streaming |
| [deepvk/cultura_ru_edu](https://huggingface.co/datasets/deepvk/cultura_ru_edu) | HF, public | качественный русский educational/web текст | есть `train` и `validation` |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | HF, public | энциклопедический стиль | брать русский config, например `20231101.ru` |
| [HPLT/HPLT3.0](https://huggingface.co/datasets/HPLT/HPLT3.0) | HF + external shards | большой web corpus | Dataset Viewer может быть нестабилен, лучше использовать direct shards/map |
| [cointegrated/taiga_stripped_proza](https://huggingface.co/datasets/cointegrated/taiga_stripped_proza) | HF, public | литература/проза | полезно для long-form русского |
| [cointegrated/taiga_stripped_rest](https://huggingface.co/datasets/cointegrated/taiga_stripped_rest) | HF, public | новости, журналы, субтитры, соцтекст | не брать без фильтрации |
| [IlyaGusev/habr](https://huggingface.co/datasets/IlyaGusev/habr) | HF, public | технический стиль | использовать малой долей |
| [PleIAs/Russian-PD](https://huggingface.co/datasets/PleIAs/Russian-PD) | HF, public | public-domain тексты | полезно для quality mix |
| [RussianNLP/wikiomnia](https://huggingface.co/datasets/RussianNLP/wikiomnia) | HF, public | QA/энциклопедический материал | loader может быть нестандартный |

### 4.2 SFT

| Источник | Доступность | Роль | Комментарий |
|---|---:|---|---|
| [IlyaGusev/saiga_scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored) | HF, public | основной SFT-кандидат | фильтровать по score |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | HF, public | multilingual instruction data | брать русский `language_code=rus` |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | HF, public | диалоговые данные | фильтровать русские ветки |
| [IlyaGusev/oasst1_ru_main_branch](https://huggingface.co/datasets/IlyaGusev/oasst1_ru_main_branch) | HF, public | готовый русский subset OASST | loader может быть нестандартный |
| [Den4ikAI/russian_instructions](https://huggingface.co/datasets/Den4ikAI/russian_instructions) | HF, public | instruction tuning | JSON формат |
| [IlyaGusev/ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca) | HF, public | optional SFT | использовать только после проверки лицензий |
| [IlyaGusev/ru_turbo_saiga](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga) | HF, public | optional SFT | использовать только после проверки лицензий |

`ru_turbo_alpaca` и `ru_turbo_saiga` лучше не делать основой публичной модели до проверки происхождения и лицензий. Их можно держать как optional/private ablation.

## 5. Tokenizer plan

Текущий tokenizer на 50k токенов слишком дорогой для 100M/150M модели, особенно если embedding и lm head не tied.

Цель: обучить новый SentencePiece BPE tokenizer на 32k токенов.

Tokenizer sample: 1-3 GB чистого текста.

Пример смеси:

```text
FineWeb2 rus_Cyrl       35%
Cultura RU Edu          25%
Wikipedia ru            15%
Taiga                   15%
Habr                     5%
Russian-PD / HPLT        5%
```

Что сравнивать с текущим tokenizer:

- chars/token;
- bytes/token;
- bits per byte;
- validation loss на одинаковом 300M-token pilot;
- tokens/sec;
- доля параметров в embedding/head;
- качество sample generations;
- firing rate profile.

Решение по tokenizer принимается только после pilot, не по ощущениям.

## 6. Validation splits

Validation надо зафиксировать до основного обучения.

Предлагаемые splits:

```text
val_web_clean    FineWeb2/Cultura
val_edu          Cultura RU Edu
val_wiki         Wikipedia
val_lit          Taiga proza
val_habr         Habr
val_mixed        смесь всех доменов
val_sft_prompts  фиксированный ручной список prompts
```

Минимальный размер:

```text
val_web_clean    5M tokens
val_edu          5M tokens
val_wiki         5M tokens
val_lit          5M tokens
val_habr         2M tokens
val_mixed        10M tokens
```

Все validation documents должны быть исключены из train через exact hash и near-dedup.

## 7. Фильтрация и дедупликация

Для pretraining:

```text
language: ru / rus_Cyrl
length: 200-20000 chars
cyrillic ratio: высокий
remove: SEO, boilerplate, casino, betting, adult spam, repeated lines
remove: слишком много URL/email/phone
dedup: normalized exact hash
near-dedup: SimHash или MinHash
eval contamination: удалить fixed eval prompts и benchmark-like data
```

Для SFT:

```text
оставлять пары instruction/response с нормальной длиной
убирать пустые, токсичные, мусорные и повторяющиеся ответы
фильтровать machine-translated artifacts
отдельно маркировать multi-turn и single-turn
не смешивать SFT с pretraining без явного stage marker
```

## 8. Формат локальных данных

Не тренироваться напрямую из HF streaming.

Правильный pipeline:

```text
HF / external source
-> inspect source
-> normalize text
-> filter
-> dedup
-> tokenize
-> write local shards
-> train from local shards
```

Формат shards:

```text
data/
  raw_manifest.jsonl
  tokenizer_sample/
  tokenized/
    pretrain/
      source=fineweb2_ru/
      source=cultura_ru_edu/
      source=wikipedia_ru/
      ...
    validation/
    sft/
```

Для tokenizer vocab < 65536 можно хранить token ids как `uint16`, если training code это поддержит. Если нет, использовать `uint32` и не оптимизировать преждевременно.

Каждый shard должен иметь manifest:

```json
{
  "source": "HuggingFaceFW/fineweb-2",
  "config": "rus_Cyrl",
  "split": "train",
  "license": "source license from dataset card",
  "documents": 100000,
  "tokens": 123456789,
  "tokenizer": "spikerugpt-bpe-32k-v1",
  "filters": ["langid", "length", "dedup"],
  "created_at": "2026-06-01"
}
```

## 9. Training stages

### Stage 0: environment gate

На RTX 5090:

```text
OS: Linux
PyTorch: CUDA 12.8 compatible build
precision: bf16
TORCH_CUDA_ARCH_LIST: 12.0
custom CUDA/WKV kernels: compile for sm_120
```

Проверки:

- CUDA видит 5090;
- bf16 работает;
- training step проходит;
- WKV kernel собирается;
- generation работает после checkpoint load.

### Stage 1: one-batch overfit

Цель: проверить, что модель, tokenizer, loss, optimizer и checkpointing не сломаны.

Критерий успеха:

```text
loss стабильно падает на одном batch
нет NaN/Inf
checkpoint reload дает тот же loss/generation
```

### Stage 2: 1M-token overfit

Цель: поймать ошибки в data loader, positions/context, tokenizer decode, optimizer schedule.

Критерий успеха:

```text
train loss сильно ниже validation loss
generation воспроизводит стиль маленького корпуса
```

### Stage 3: 30M smoke model

Цель: проверить полный training loop на малом бюджете.

Критерий успеха:

```text
loss падает на всех validation slices
firing rate не вырождается в 0% или 100%
samples улучшаются от checkpoint к checkpoint
```

### Stage 4: 300M tokenizer pilot

Сравнить:

```text
current tokenizer: ruGPT-3 BPE 50k
new tokenizer: SentencePiece BPE 32k
```

На одинаковых данных и одинаковой архитектуре.

Решение:

- если 32k tokenizer выигрывает или не хуже по loss/BPB, но быстрее и дешевле по параметрам, берем 32k;
- если 50k явно лучше, проверяем tied embeddings и повторяем расчет параметров.

### Stage 5: base pretraining

Первый серьезный запуск: 8B-12B tokens.

Рекомендуемая смесь:

```text
FineWeb2 rus_Cyrl          40-45%
Cultura RU Edu             25%
HPLT top-quality rus       10%
Taiga                       8%
Wikipedia                   6%
Russian-PD                  3-5%
Habr / technical / news     2-3%
WikiOmnia plain/context     1-2%
```

Если HPLT задерживает pipeline, временно заменить его FineWeb2 + Cultura. HPLT не должен блокировать весь запуск.

### Stage 6: quality annealing

После base pretraining: 0.5B-1B tokens на более качественной смеси.

```text
Cultura RU Edu             35-40%
Wikipedia                  20%
Russian-PD                 15%
WikiOmnia                  10%
Taiga/Habr longform        10%
QA/explanatory              5-10%
```

Цель:

- улучшить связность;
- улучшить фактичность;
- уменьшить web-noise;
- подготовить модель к SFT.

### Stage 7: SFT

SFT после base + annealing.

Начальный размер: 70k-120k examples.

Смесь:

```text
Saiga scored             20k-40k
russian_instructions     10k-30k
Aya RU                    2k-10k
OASST RU                  3k-8k
WikiOmnia QA             20k-50k
manual/synthetic          later
```

Обязательные правила:

- фильтровать Saiga по score;
- не смешивать низкокачественные synthetic examples с чистым SFT;
- держать отдельный validation set для SFT;
- сохранять generations на фиксированных prompts после каждого checkpoint.

### Stage 8: optional alignment

DPO/ORPO/RLHF не нужны на первом проходе.

Их можно рассматривать только после:

- стабильной base model;
- стабильного SFT;
- понятного набора preference data;
- ручной оценки generations.

## 10. Метрики

Обязательные training metrics:

```text
train loss
validation loss by domain
tokens/sec
GPU memory
grad norm
learning rate
NaN/Inf checks
checkpoint size
```

Обязательные language metrics:

```text
perplexity
bits per byte
chars/token
fixed prompt generations
MERA/RuMMLU-style eval later
```

Обязательные spiking metrics:

```text
global firing rate
firing rate by layer
firing rate by token position
silent neurons/channels
overactive neurons/channels
firing rate on different domains
```

Для статьи особенно важны:

- сравнение firing rate RU vs EN на сопоставимых условиях;
- нормировка spikes/token, spikes/char, spikes/byte;
- доверительные интервалы;
- анализ по жанрам;
- анализ по морфологическим категориям;
- ablation fixed LIF vs learnable tau/theta.

## 11. Минимальные ablations для статьи

Если хватит времени/GPU:

```text
A0: current 100M baseline, old tokenizer
A1: 100M, new 32k tokenizer
A2: 150M, fixed LIF
A3: 150M, learnable tau only
A4: 150M, learnable threshold only
A5: 150M, learnable tau + threshold
```

Научно полезные сравнения:

- одинаковый token budget;
- одинаковый validation set;
- одинаковая архитектура, кроме LIF-варианта;
- генерации на одинаковых prompts;
- firing rate с confidence intervals.

## 12. Риски

### Dataset risk

Некоторые источники доступны, но имеют нестандартные loaders или нестабильный Dataset Viewer:

- HPLT;
- WikiOmnia;
- ru_turbo_alpaca;
- ru_turbo_saiga;
- oasst1_ru_main_branch.

Решение: в `inspect_sources.py` делать не только HF Dataset Viewer check, но и реальный `load_dataset(..., streaming=True)` или fallback на files API.

### License risk

Перед публичным релизом модели отдельно проверить:

- FineWeb2 license/data terms;
- HPLT terms;
- Taiga licenses;
- Wikipedia/Wikimedia attribution/share-alike;
- Russian-PD terms;
- generated instruction datasets;
- OpenAI-generated derivatives in `ru_turbo_*`.

### Architecture risk

В текущем коде важно отдельно проверить:

- tied vs untied embeddings;
- корректность инициализации `LearnableLIFNode`;
- реальное значение tau/threshold после параметризации через `softplus`;
- совместимость CUDA kernels с RTX 5090.

## 13. Что писать в коде первым

Минимальный порядок реализации:

```text
configs/data_sources.yaml
scripts/inspect_sources.py
scripts/build_tokenizer_sample.py
scripts/train_tokenizer_spm.py
scripts/eval_tokenizer.py
scripts/build_validation_splits.py
scripts/build_pretrain_shards.py
scripts/build_sft_mix.py
```

После этого:

```text
tokenizer switch in training code
tied embeddings option
domain validation logging
firing rate logging
generation snapshots
checkpoint metadata
5090 environment script
```

## 14. Первый конкретный milestone

Milestone 1 считается закрытым, когда есть:

- `data_sources.yaml` со всеми источниками;
- `inspect_sources.py`, который проверяет доступность источников;
- tokenizer sample 1-3 GB;
- обученный `spikerugpt-bpe-32k-v1`;
- отчет сравнения 32k vs текущий 50k tokenizer;
- зафиксированные validation splits;
- one-batch overfit на новом tokenizer.

После этого можно переноситься на сервер с RTX 5090 и запускать pilot.

