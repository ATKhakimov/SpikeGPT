# Server handoff: SpikeRuGPT data-prep and training

Дата: 2026-06-01  
Цель: передать работу другому агенту/инженеру на сервере с RTX 5090 без необходимости читать всю историю чата.

## 1. Что читать первым

1. `ARTICLE/spikerugpt_training_plan.md`
2. `ARTICLE/spikerugpt_technical_log.md`
3. `configs/data_sources.yaml`
4. `scripts/data/README.md`
5. `scripts/data/common.py`
6. `scripts/data/inspect_sources.py`

Главная идея: не скачивать все датасеты сразу. Сначала проверить источники, схемы, splits и первые строки, затем собрать tokenizer sample, validation splits и только потом pretraining shards.

## 2. Текущее состояние

Готово:

- зафиксирован общий план обучения в `ARTICLE/spikerugpt_training_plan.md`;
- добавлен конфиг источников `configs/data_sources.yaml`;
- добавлены скрипты подготовки данных в `scripts/data/`;
- добавлен отдельный список зависимостей `requirements_data.txt`;
- `.gitignore` поправлен так, чтобы `scripts/data/` не игнорировался;
- HPLT, WikiOmnia и часть сложных SFT-источников пока отключены через `enabled: false`.

Не сделано:

- скрипты подготовки данных не запускались;
- tokenizer 32k еще не обучен;
- validation splits еще не собраны;
- pretraining shards еще не собраны;
- training loop еще не адаптирован под новый tokenizer/shards;
- лицензии и provenance для части датасетов еще не проверены;
- HPLT/WikiOmnia требуют отдельной проверки схемы и загрузчика.

## 3. Первая команда на сервере

После установки зависимостей первая реальная команда должна быть инспекцией источников:

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl
```

Эта команда должна проверить:

- доступность HF dataset repos;
- наличие нужных configs;
- наличие нужных splits;
- streaming load;
- columns/schema первых строк;
- возможность извлечь текст через заданные `text_fields`.

Если эта команда падает на одном источнике, не надо сразу править training code. Сначала поправить `configs/data_sources.yaml` или добавить source-specific adapter.

## 4. Что пока не запускать

Не запускать сразу:

```bash
python scripts/data/build_pretrain_shards.py ...
```

Причина: до инспекции источников неизвестны реальные схемы некоторых датасетов, а полный pretrain mix может начать большой download/streaming.

Не запускать SFT mix до проверки схем:

```bash
python scripts/data/build_sft_mix.py ...
```

Причина: разные instruction datasets используют разные поля и форматы conversation tree.

Не запускать полный training run до:

- tokenizer comparison;
- one-batch overfit;
- 1M-token overfit;
- 30M smoke run;
- проверки формата данных в training loader.

## 5. Рекомендуемый порядок запуска

### Step 0: окружение

Проверить:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Для RTX 5090 нужен современный PyTorch/CUDA stack с поддержкой Blackwell/sm_120. Если custom CUDA/WKV kernels используются в обучении, их надо собирать под `sm_120`.

### Step 1: зависимости data-prep

```bash
pip install -r requirements_data.txt
```

Если используется отдельное training окружение, лучше поставить data-prep зависимости туда же или сделать отдельный venv.

### Step 2: inspect sources

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl
```

Ожидаемый результат: JSONL-отчет, где у основных источников `ok=true`.

Основные источники, которые должны заработать первыми:

- `fineweb2_ru`;
- `cultura_ru_edu`;
- `wikipedia_ru`;
- `taiga_proza`;
- `taiga_rest`;
- `habr`;
- `russian_pd`;
- `saiga_scored`;
- `russian_instructions`;
- `aya_ru`.

### Step 3: tokenizer sample

Сначала можно сделать маленький smoke sample:

```bash
python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --target-bytes 100000000 \
  --max-docs-per-source 20000 \
  --out data/tokenizer_sample/smoke_sample.txt
```

Если smoke sample нормальный, собирать полный sample:

```bash
python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --out data/tokenizer_sample/spikerugpt_tokenizer_sample.txt
```

### Step 4: SentencePiece tokenizer

```bash
python scripts/data/train_sentencepiece.py \
  --input data/tokenizer_sample/spikerugpt_tokenizer_sample.txt \
  --model-prefix tokenizer/spikerugpt-bpe-32k \
  --vocab-size 32000
```

После обучения tokenizer нужно сравнить со старым ruGPT-3 tokenizer по:

- chars/token;
- bytes/token;
- BPB/BPC;
- tokenization examples;
- tokens/sec;
- доля параметров в embedding/head.

### Step 5: validation splits

```bash
python scripts/data/build_validation_splits.py \
  --config configs/data_sources.yaml \
  --output-dir data/validation_text
```

Потом эти validation texts надо токенизировать тем же tokenizer и исключить из train через hash/dedup.

### Step 6: маленький pretrain shard

Не начинать с 10B tokens. Сначала сделать маленький shard:

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_smoke \
  --max-tokens 10000000 \
  --tokens-per-shard 10000000
```

Этот shard нужен для:

- проверки dtype;
- проверки eos;
- проверки decode samples;
- one-batch overfit;
- 1M-token overfit.

### Step 7: pilot shard

После smoke:

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_300m \
  --max-tokens 300000000 \
  --tokens-per-shard 100000000
```

Этот этап нужен для сравнения старого 50k tokenizer и нового 32k tokenizer.

## 6. Важное несовпадение с текущим training code

Сейчас старый `train.py` ожидает примерно такой путь:

```text
data/ru_train_full.npy
```

Новый pipeline пишет:

```text
data/tokenized/pretrain/*.bin
manifest.json
```

Значит, перед обучением надо выбрать один из вариантов:

1. временно конвертировать `.bin` shards в один `.npy` для smoke/pilot;
2. добавить loader для нескольких `.bin` shards;
3. добавить полноценный `.bin/.idx` формат через `src/binidx.py`;
4. адаптировать training code под manifest.

Для первого smoke-теста проще вариант 1. Для полного обучения лучше вариант 2 или 3.

## 7. Датасеты, требующие осторожности

### HPLT

В `configs/data_sources.yaml` источник `hplt3_ru_top` отключен:

```yaml
enabled: false
```

Причина: HF Dataset Viewer может быть нестабилен, а для HPLT лучше использовать direct sorted shards/map. Включать только после отдельного adapter-а.

### WikiOmnia

Отключен в pretrain/SFT mix до проверки схемы:

```yaml
enabled: false
```

Причина: может потребоваться QA-to-text adapter.

### ru_turbo_alpaca / ru_turbo_saiga

Не включены в основной SFT mix. Использовать только после проверки лицензий и происхождения данных.

## 8. Что проверить в коде модели позже

После data-prep, но до полного обучения:

- сделать option для нового vocab size 32000;
- проверить tied embeddings vs untied embeddings;
- проверить `LearnableLIFNode` и фактическую инициализацию `tau`/`threshold`;
- добавить logging firing rate по слоям;
- добавить validation по доменам;
- добавить сохранение fixed prompt generations;
- проверить checkpoint metadata.

## 9. Definition of done для server data-prep

Data-prep этап можно считать готовым, когда есть:

- `reports/data_source_inspection.jsonl`;
- tokenizer sample;
- `tokenizer/spikerugpt-bpe-32k.model`;
- validation text splits;
- tokenized validation splits;
- маленький pretrain smoke shard;
- manifest для каждого собранного артефакта;
- короткий отчет, какие источники реально сработали, какие отключены и почему.

После этого можно переходить к one-batch overfit и training code adaptation.
