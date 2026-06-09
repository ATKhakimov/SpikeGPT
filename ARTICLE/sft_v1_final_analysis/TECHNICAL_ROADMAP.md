# SpikeRuGPT v1: Technical Training Roadmap

Дата фиксации: 2026-06-09  
Статус: pretrain base зафиксирован, SFT запущен, финальный post-SFT анализ автоматизирован через watcher.

Этот документ фиксирует инженерный путь обучения SpikeRuGPT v1: какие решения были приняты, с какими техническими проблемами столкнулись, как они были решены, и какие артефакты нужны для статьи/доклада. Это не статья, а рабочий технический отчет.

## 1. Цель проекта

Изначальная цель была не просто запустить старый SpikeGPT-код, а получить воспроизводимый русскоязычный pipeline:

1. Собрать и очистить русскоязычный pretrain corpus.
2. Обучить собственный tokenizer под русский текст.
3. Провести pretrain небольшой SpikeGPT/RWKV-like модели.
4. Зафиксировать base checkpoint.
5. Провести SFT на коротких русскоязычных instruction-примерах.
6. Сравнить v0/v1/base/SFT и собрать материалы для статьи.

Практическое ограничение: один доступный сервер с RTX 5090 и ограничение по времени около двух недель. Поэтому основной кандидат был зафиксирован как модель порядка 74M параметров, а не попытка обучить 1B-параметровую модель.

## 2. Версии модели

| Версия | Назначение | Статус |
|---|---|---|
| v0 | старая proof-of-concept модель `Koras1k/spikerugpt-100M-Taiga` | baseline для сравнения и тезисов |
| v1 base | текущая 74M SpikeGPT после нового pretrain pipeline | зафиксирована на `step=43674` |
| v1 SFT | instruction-tuned версия v1 base | обучение запущено, post-run анализ автоматизирован |

Важно для статьи: v0 лучше описывать как предварительную проверку идеи, а v1 как инженерно воспроизводимую версию с нормальной подготовкой данных, tokenizer, логами, eval и backup.

## 3. Архитектура и модельные нюансы

Текущая v1:

```text
architecture: SpikeGPT / RWKV-style blocks + LIF nodes
n_layer: 12
n_embd: 512
ctx_len: 1024
vocab_size: 32000
parameters: ~73.7M
precision: bf16
hardware: NVIDIA GeForce RTX 5090, 32607 MiB VRAM
```

Архитектурно модель использует RWKV-like блоки и spiking/LIF-механизм. Внутри `src/model.py` есть два LIF-узла на block (`lif1`, `lif2`). Это важно для анализа, потому что кроме обычных LM-метрик можно смотреть:

- firing rate по слоям;
- долю silent channels;
- параметры learnable LIF (`tau`, `threshold`);
- изменение активности после pretrain/SFT.

Практический вывод по архитектуре: для статьи ценность не только в loss/PPL, но и в том, что можно показать, как spiking-активность ведет себя на русскоязычных доменах.

## 4. Tokenizer

Был выбран собственный SentencePiece BPE tokenizer на 32k vocab:

```text
tokenizer/spikerugpt-bpe-32k.model
```

Причины:

- старый tokenizer/vocab был слишком дорог для маленькой модели;
- embedding/head занимают значимую долю параметров;
- русский текст требует плотной токенизации;
- при малой модели лучше не тратить capacity на избыточный vocab.

Практический эффект нового tokenizer по раннему сравнению:

```text
old tokenizer tokens: 14.10M
new tokenizer tokens: 6.42M
token reduction: ~54%
```

Это один из ключевых инженерных выводов: для маленькой русскоязычной модели tokenizer влияет не только на удобство, но и на реальный token budget обучения.

## 5. Pretrain Data Pipeline

Pipeline был сделан локальным и воспроизводимым:

```text
HF/source datasets
-> schema inspection
-> text normalization
-> quality filters
-> exact dedup
-> near dedup
-> SentencePiece tokenization
-> uint16 .bin shards
-> training from local manifest
```

Мы специально не обучались напрямую из HF streaming, потому что:

- сетевые сбои ломают длинный training;
- сложно повторить точный набор данных;
- нельзя нормально проверять contamination/dedup;
- локальные shards дают стабильный throughput.

Финальный pretrain manifest:

```text
data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json
written_tokens: 977,192,802
dtype: uint16
tokenizer: tokenizer/spikerugpt-bpe-32k.model
```

Состав корпуса:

| Source | Tokens | Documents |
|---|---:|---:|
| fineweb2_ru | 485,874,429 | 556,616 |
| cultura_ru_edu | 282,485,004 | 228,631 |
| wikipedia_ru | 67,793,729 | 31,010 |
| taiga_proza | 56,495,556 | 23,593 |
| russian_pd | 45,190,997 | 6,998 |
| habr | 28,247,740 | 20,755 |
| taiga_lenta | 11,105,347 | 35,174 |

Итоговый корпус оказался примерно 0.98B токенов. Для модели ~74M это примерно 13 токенов на параметр. Это не Chinchilla-optimal 20 токенов/параметр, но достаточно хороший режим для ограниченного бюджета и практической цели.

## 6. Фильтрация данных

Было принято решение фильтровать строго, потому что данных достаточно, а модель маленькая. Для 74M модели шум вреднее, чем для большой: capacity меньше, и мусор быстрее ухудшает распределение.

Основные типы reject в pretrain corpus:

| Reject reason | Count |
|---|---:|
| too many short lines | 102,072 |
| spam keyword | 40,900 |
| too short | 4,666 |
| too many phones | 3,407 |
| near duplicate | 2,247 |
| low cyrillic ratio | 2,190 |
| repeated lines | 1,323 |
| low alpha ratio | 204 |
| exact duplicate | 140 |
| too many emails | 53 |
| low unique word fraction | 27 |
| too long | 21 |

Фильтры включали:

- минимальную долю кириллицы;
- отсечение SEO/spam/adult/casino/betting;
- отсечение документов с большим числом URL/email/телефонов;
- repeated lines;
- exact duplicate;
- near duplicate через SimHash;
- ограничения длины и качества текста.

## 7. Pretrain Training

Основной run:

```text
run_id: autonomous-ctx1024-1b-bf16-5d
checkpoint_dir: checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d
manifest: data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json
precision: bf16
batch_size: 22
ctx_len: 1024
lr: 3e-4
```

Стабильная скорость после оптимизации:

```text
~2760-2770 tok/s
peak PyTorch memory: ~23.2 GB
nvidia-smi used/reserved: до ~31.7 GB
```

Последний надежный base checkpoint:

```text
checkpoint: checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt
step: 43674
tokens_seen: 944,590,848
coverage: ~96.7% от 977M corpus
size: 885 MB
```

По метрикам training успел дойти до `step=43790`, но надежный сохраненный checkpoint остался на `step=43674`.

## 8. Технические проблемы pretrain и решения

### 8.1 Background-процессы и `/ps`

Сначала часть задач запускалась так, что пользователь не видел прогресс через `/ps`. После этого длинные процессы стали запускаться через `tmux`, а рядом оставлялся `tail -f` логов как background terminal.

Итоговый принцип:

```text
long run -> tmux session
progress -> tqdm/log
/ps visibility -> background tail -f
```

### 8.2 BrokenPipe при остановке

Когда обучение остановили вручную через tmux/pipe, скрипт поймал:

```text
BrokenPipeError: [Errno 32] Broken pipe
```

Из-за этого финальный checkpoint после interrupt не был записан. Последний безопасный checkpoint остался:

```text
step=43674
tokens_seen=944,590,848
```

Потеря относительно последней train metric была около 2.6M токенов, то есть порядка 15-16 минут. Это было признано некритичным.

Вывод: для будущих запусков лучше иметь штатный stop-флаг или отдельный signal handler, который сначала сохраняет checkpoint, а потом закрывает progress/tqdm.

### 8.3 Hugging Face upload

Обычный upload full checkpoint через Python API завис/полз очень медленно около 93%.

Решение:

```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1
```

После этого upload прошел быстро. На HF были сохранены:

```text
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/latest.pt
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/model_state_step_43674.pt
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/eval_metrics.json
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/eval_metrics.md
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/metrics.jsonl
runs/autonomous-ctx1024-1b-bf16-5d/snapshots/step_43674/training_last_24h_avg_loss.png
```

## 9. Base Evaluation

Base checkpoint `step=43674` был провалидирован на fixed splits.

| Split | Loss | PPL |
|---|---:|---:|
| val_mixed | 4.7730 | 118.27 |
| val_wiki | 4.2470 | 69.90 |
| val_habr | 4.9269 | 137.96 |
| val_lit | 4.9063 | 135.14 |
| train_sample | 4.7286 | 113.13 |

Среднее:

```text
mean validation loss: 4.7133
arithmetic mean validation PPL: 115.32
train/val loss gap: -0.0152
```

Сравнение с промежуточным `step=39298`:

| Split | PPL step 39298 | PPL step 43674 | Delta |
|---|---:|---:|---:|
| train_sample | 131.29 | 113.13 | -18.15 |
| val_mixed | 136.83 | 118.27 | -18.56 |
| val_wiki | 103.87 | 69.90 | -33.97 |
| val_habr | 145.00 | 137.96 | -7.05 |
| val_lit | 156.17 | 135.14 | -21.03 |

Вывод: base checkpoint рабочий. Ухудшения на validation не видно, PPL для модели такого размера выглядит достаточно сильным.

## 10. Spiking Activity

Для base checkpoint была снята spiking activity на validation domains.

Средний firing rate:

| Split | Firing rate | Silent channel fraction |
|---|---:|---:|
| val_mixed | 0.0928 | 0.4637 |
| val_wiki | 0.0960 | 0.4651 |
| val_habr | 0.0940 | 0.4715 |
| val_lit | 0.0959 | 0.4793 |

Идейная интерпретация firing rate: это доля активных spike-событий в LIF-узлах. Низкий firing rate означает sparse event-like computation, высокий firing rate означает более плотную активацию. Для статьи это можно использовать как нейроморфный слой анализа, отличающий проект от обычной dense LM.

Осторожность: по одному firing rate нельзя делать сильный вывод о “понимании” модели. Это диагностическая метрика активности, а не метрика качества текста.

## 11. SFT Data

Финальный SFT dataset:

```text
data/sft/spikerugpt_sft_clean_final.jsonl
examples: 65,056
encoded usable examples: 64,893
skipped by ctx/format during SFT encoding: 163
```

Состав:

| Source | Examples |
|---|---:|
| russian_instructions | 26,119 |
| russian_easy_instructions | 18,909 |
| saiga_scored | 9,495 |
| ru_turbo_alpaca | 7,551 |
| ru_turbo_saiga | 2,982 |

OASST был исключен, потому что после фильтрации давал слишком мало полезных примеров и не стоил усложнения смеси.

Длины после токенизации:

```text
median full example: 108 tokens
p90: 249
p95: 293
p99: 433
max: 2122
over ctx_len+1: 34 examples (~0.05%)
median answer: 80 tokens
p95 answer: 247
```

Вывод: SFT dataset хорошо подходит для `ctx_len=1024`; почти все примеры помещаются без обрезки.

## 12. SFT Training Design

Для SFT был написан отдельный trainer:

```text
scripts/train_sft.py
```

Главный технический момент: loss считается только по assistant-ответу, а не по всей инструкции.

Формат:

```text
Система:
...

Инструкция:
...

Ответ:
...
```

Labels:

```text
system/user/prefix tokens -> ignore_index = -100
assistant content + eos -> supervised tokens
```

Это важно, потому что если считать loss по всей строке, модель учится предсказывать инструкцию, а не только ответ. Для instruction tuning это менее корректно.

Текущий SFT запуск:

```text
run_id: sft-step43674-v1-b64tok18k
base: checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt
epochs: 1
lr: 2e-5
batch_size cap: 64
max_batch_tokens: 18000
eval_every: 250
save_every: 250
train_examples: 63,596
val_examples: 1,000
```

## 13. Технические проблемы SFT и решения

### 13.1 Свободная VRAM оказалась обманчивой

Первый smoke с batch 24 показывал невысокую память на первых шагах:

```text
step=1 peak_mem_gb ~7.4
```

Но дальше peak вырос:

```text
step=30 peak_mem_gb ~20.2
nvidia-smi used/reserved ~24-28 GB
```

Причина: SFT-примеры имеют разную длину, а batch padding идет до самого длинного примера в батче. Поэтому реальная память зависит не только от числа примеров, но и от `batch_size * max_sequence_length`.

### 13.2 Fixed batch OOM

Был проведен batch probe:

```text
bs=32 ok, peak_gb=20.59
bs=40 OOM, peak_gb=26.55
```

При полном shuffle фиксированный `batch=32` тоже оказался рискованным: первый batch мог случайно собрать длинные примеры и ловить OOM в backward.

### 13.3 Dynamic batching

Решение: перейти от fixed batch по количеству примеров к dynamic batching с ограничением по padded token budget.

Текущий режим:

```text
batch_size cap: 64
max_batch_tokens: 18000
max_eval_batch_tokens: 16000
```

Так короткие примеры идут большими батчами, а длинные автоматически уменьшают фактический batch. Это убрало OOM и сохранило нормальную скорость.

### 13.4 Текущий SFT прогресс

На момент фиксации:

```text
step: 110 / 1721
examples_seen: 4,114
avg_recent: 4.5270
peak_mem_gb: 18.8167
examples_per_sec: ~9.66
tmux: sft_step43674_v1
watcher tmux: sft_finalizer
```

Watcher пишет статус раз в час и после завершения запускает:

1. SFT LM/PPL eval.
2. Instruction generations.
3. Summary README для статьи.
4. HF upload.

Финальная папка анализа:

```text
ARTICLE/sft_v1_final_analysis/
```

## 14. Roadmap обучения: фактический

### Stage A: подготовка данных

Статус: выполнено.

- проверены источники;
- собран tokenizer sample;
- обучен SentencePiece BPE 32k;
- собраны validation splits;
- собраны pretrain shards;
- собран SFT mix.

### Stage B: pretrain smoke/pilot

Статус: выполнено.

- проверена загрузка shards;
- проверен bf16 на RTX 5090;
- проверены checkpoints;
- подобран batch;
- подтверждено, что loss падает.

### Stage C: основной pretrain

Статус: выполнено до надежного checkpoint `step=43674`.

- показано ~944.6M токенов;
- это ~96.7% финального корпуса;
- base checkpoint сохранен локально и выгружен на HF;
- base PPL подтверждает, что модель рабочая.

### Stage D: SFT

Статус: выполняется.

- trainer с assistant-only loss написан;
- smoke пройден;
- fixed batch OOM решен через dynamic batching;
- full SFT запущен.

### Stage E: post-SFT analysis

Статус: автоматизирован, ожидает завершения SFT.

Планируемые артефакты:

```text
ARTICLE/sft_v1_final_analysis/README.md
ARTICLE/sft_v1_final_analysis/sft_lm_eval.json
ARTICLE/sft_v1_final_analysis/sft_lm_eval.md
ARTICLE/sft_v1_final_analysis/sft_generations.json
ARTICLE/sft_v1_final_analysis/sft_generations.md
ARTICLE/sft_v1_final_analysis/hf_upload.json
```

## 15. Что важно вынести в статью

Потенциально сильные инженерные тезисы:

1. Для малой русскоязычной spiking LM tokenizer критичен: 32k SentencePiece сильно уменьшил token count.
2. Маленькая модель лучше выигрывает от чистых данных, чем от максимально большого, но шумного корпуса.
3. Pretrain почти на 1B русскоязычных токенов дал рабочий base checkpoint с `val_mixed PPL ~118`.
4. Spiking-модель можно анализировать не только через loss/PPL, но и через firing rate/silent channels.
5. SFT для такой модели требует коротких инструкций, assistant-only loss и осторожности с dynamic batching.
6. Практическая инженерия оказалась существенной частью работы: tmux, checkpointing, HF backup, upload acceleration, OOM handling.

## 16. Риски и ограничения

Ограничения, которые надо честно указать:

- модель маленькая, не конкурент крупным LLM;
- SFT dataset частично включает синтетические/переводные instruction данные;
- PPL после SFT может ухудшиться на обычном LM continuation, это нормально, если instruction behavior улучшится;
- firing rate не является прямой метрикой качества;
- сравнение с v0 осложнено разными tokenizer и training corpus;
- pretrain остановлен на 96.7% корпуса, а не на полном 100% проходе, но разница мала.

## 17. Следующие действия

После завершения SFT:

1. Проверить `ARTICLE/sft_v1_final_analysis/README.md`.
2. Проверить `sft_generations.md` руками на мусор/повторы.
3. Сравнить base vs SFT:
   - instruction behavior;
   - LM PPL;
   - firing rate;
   - длина и стабильность ответов.
4. Выгрузить финальный SFT checkpoint на HF.
5. Обновить конференционную статью:
   - добавить pipeline diagram;
   - добавить таблицу pretrain/SFT metrics;
   - добавить 1-2 графика loss;
   - добавить firing-rate график;
   - явно описать v0 как proof-of-concept, v1 как improved pipeline.

