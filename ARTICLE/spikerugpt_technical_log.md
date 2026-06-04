# SpikeRuGPT technical log

Дата фиксации: 2026-06-02  
Назначение: рабочий технический журнал проекта, чтобы не держать решения, нюансы и текущий статус только в чате.

Этот документ дополняет `ARTICLE/spikerugpt_training_plan.md` и `ARTICLE/server_handoff.md`. План описывает желаемую стратегию, handoff описывает стартовое состояние на сервере, а этот файл фиксирует фактически принятые решения, текущие результаты, ограничения и следующие действия.

## 1. Текущая рамка проекта

Цель на ближайшие две недели: получить русскоязычную SpikeGPT-модель, за которую не стыдно как за инженерный и исследовательский результат, а не пытаться любой ценой догнать большие dense LLM.

Практическое решение после обсуждения:

- не переключаться сейчас на модель порядка 1B параметров;
- не начинать заново 150M/более крупную модель без жесткой причины;
- сфокусироваться на текущей 74M-модели как v1;
- использовать уже сделанный 12-часовой pretrain как базу для продолжения;
- довести pretraining, затем quality annealing, затем SFT;
- параллельно собирать артефакты для конференционной статьи.

Версии модели:

| Версия | Смысл | Статус |
|---|---|---|
| v0 | старая тестовая модель `Koras1k/spikerugpt-100M-Taiga` | baseline/proof of concept для тезисов |
| v1 | текущая улучшенная 74M-модель на новом pipeline | основной кандидат |

Важно для статьи: v0 надо подавать как первый proof of concept, а v1 как улучшенную итерацию с нормальным pipeline, tokenizer, фильтрацией и воспроизводимыми логами.

## 2. Аппаратная и программная среда

Основной сервер:

```text
GPU: NVIDIA GeForce RTX 5090
VRAM: 32 GB class, nvidia-smi показывает 32607 MiB total
CPU: используется для сборки shard-ов параллельно с GPU-задачами
OS/user workspace: /workspace/SpikeGPT
```

Операционный принцип:

- длинные задачи запускать через `tmux`, чтобы они жили независимо от активного чата;
- для видимости через `/ps` держать tail/log-процессы в background terminal;
- не оставлять GPU простаивать, если CPU занят подготовкой данных;
- тяжелую сборку данных запускать с `nice` и `ionice`, чтобы она не душила интерактивную работу и GPU-pretrain.

HF:

- авторизация Hugging Face настроена через ранее переданный токен;
- сам токен в файлы, логи и отчеты не записывать;
- приватные backup/upload использовать только через cached auth/env.

## 3. Архитектура текущей v1

Текущий основной кандидат:

```text
architecture: SpikeGPT / RWKV-style + LIF
layers: 12
n_embd: 512
context length: 1024
tokenizer: SentencePiece BPE 32k
model size: примерно 74M
```

Почему сейчас 74M:

- модель уже обучается стабильно;
- за две недели реалистичнее довести 74M до демонстрационного качества;
- переход на 150M или 1B параметров увеличивает риск не успеть получить законченный результат;
- текущий pretrain уже дал хороший сигнал по loss, и его можно продолжать.

Вывод по русскому качеству: 74M не станет полноценным ChatGPT-классом, но должна нормально порождать русский текст и после SFT отвечать в простых диалоговых сценариях. Ожидание надо формулировать честно: маленькая русскоязычная spiking LM, а не универсальный ассистент.

## 4. Tokenizer

Решение: обучать свой SentencePiece BPE 32k.

Причины:

- старый 50k tokenizer слишком дорог для маленькой модели;
- у малой модели большая доля параметров уходит в embedding/head;
- русский текст должен кодироваться плотнее;
- новый tokenizer заметно снижает token budget на тот же объем текста.

Фактическое сравнение старого и нового tokenizer:

```text
old tokenizer:
  tokens: 14.10M
  chars/token: 1.95
  bytes/token: 3.53

new spikerugpt-bpe-32k:
  tokens: 6.42M
  chars/token: 4.29
  bytes/token: 7.75

token count reduction: -54.46%
```

Артефакты:

```text
tokenizer/spikerugpt-bpe-32k.model
tokenizer/spikerugpt-bpe-32k.vocab
reports/tokenizer_comparison.md
```

Практический вывод: новый tokenizer сильно выгоднее для русского текста и текущего размера модели.

## 5. Pretraining data

Основной pipeline:

```text
HF/source data
-> inspect schemas
-> normalize text
-> filter
-> exact dedup
-> near dedup
-> tokenize with SentencePiece
-> write local .bin shards
-> train only from local shards
```

Не тренируемся напрямую из HF streaming. Это было зафиксировано как принцип, потому что training должен быть воспроизводимым, локальным и не зависеть от сетевых провалов.

### 5.1 Готовый 300M corpus

Готов 300M-token pretrain corpus:

```text
path: data/tokenized/pretrain_300m
manifest: data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json
total tokens: 299,985,616
HF backup: Koras1k/spikerugpt-pretrain-300m-v1
```

Состав 300M:

| Источник | Tokens | Доля |
|---|---:|---:|
| fineweb2_ru | 145,761,449 | 48.59% |
| cultura_ru_edu | 84,744,890 | 28.25% |
| wikipedia_ru | 20,338,310 | 6.78% |
| taiga_proza | 16,947,203 | 5.65% |
| russian_pd | 13,553,755 | 4.52% |
| taiga_lenta | 10,169,379 | 3.39% |
| habr | 8,470,630 | 2.82% |

### 5.2 Текущая сборка 1B corpus

Запущена сборка 1B-token corpus:

```bash
PYTHONUNBUFFERED=1 ionice -c2 -n7 nice -n 10 python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_1b \
  --max-tokens 1000000000 \
  --tokens-per-shard 100000000 \
  --progress-docs 5000 2>&1 | tee reports/build_pretrain_1b.log
```

Текущее состояние на момент фиксации:

```text
tmux session: spikerugpt_build_1b
log: reports/build_pretrain_1b.log
progress: около 41.3% от 1B
current source: fineweb2_ru
current speed: около 95.6k tokens/sec
```

Оценка: при такой скорости полная сборка 1B занимает около 2.9 часа чистого времени. С учетом уже выполненных 41% осталось примерно 1.7 часа, если скорость сохранится.

## 6. Очистка и фильтрация данных

Решение: фильтровать строго, потому что данных достаточно. Лучше потерять часть данных, чем обучать маленькую модель на мусоре.

Основные фильтры pretraining:

- язык: русский / `rus_Cyrl`;
- длина документа: отсечь слишком короткие и слишком длинные документы;
- достаточная доля кириллицы;
- удаление HTML/markdown boilerplate;
- удаление документов с высокой долей URL/email/телефонов;
- удаление SEO, casino, betting, adult spam и похожего мусора;
- ограничение повторяющихся строк и повторяющихся n-грамм;
- exact dedup по нормализованному hash;
- near dedup через SimHash;
- отдельные validation splits не должны протекать в train.

Код:

```text
scripts/data/common.py
scripts/data/build_pretrain_shards.py
configs/data_sources.yaml
```

Важный нюанс: фильтры подбирались с учетом размера модели. Для 74M-модели шум вреднее, чем для большой модели, потому что capacity меньше и мусор быстрее съедает полезный signal.

## 7. Validation

Validation splits зафиксированы отдельно:

```text
data/validation_text
```

После 12-часового pretrain были получены:

| Split | Loss | PPL |
|---|---:|---:|
| val_mixed | 5.654484 | 285.57 |
| val_wiki | 5.451165 | 233.03 |
| val_habr | 5.839232 | 343.52 |
| val_lit | 5.590524 | 267.88 |

Интерпретация:

- loss падает устойчиво;
- wiki проще остальных доменов;
- habr тяжелее, вероятно из-за технических терминов, форматирования и смешанного стиля;
- это хороший сигнал для pretraining, но еще не показатель диалогового качества.

## 8. Training state

Завершенный 12h run:

```text
run_id: autonomous-ctx1024-12h
checkpoint: checkpoints/autonomous/autonomous-ctx1024-12h/final.pt
latest: checkpoints/autonomous/autonomous-ctx1024-12h/latest.pt
report: reports/autonomous-ctx1024-12h.json
HF backup: Koras1k/spikerugpt-autonomous-runs/runs/autonomous-ctx1024-12h/
```

Основные метрики:

```text
status: ok
selected_batch: 16
step: 6396
start_step: 1277
stop_reason: wall_time
tokens_seen: 104,792,064
elapsed: 43,205.75 sec
speed: 1941.17 tok/s
initial_loss: 10.540096
final_loss: 5.910952
min_loss: 5.729897
peak_mem_gb: 20.1248
checkpoint_verified: ok
```

Вывод: checkpoint пригоден как база для продолжения. Это не финальная модель, а стабильный pretrain checkpoint.

Loss/throughput curves:

```text
script: scripts/plot_training_curves.py
figures:
  ARTICLE/figures/training_loss_by_tokens.png
  ARTICLE/figures/training_loss_by_step.png
  ARTICLE/figures/training_throughput_by_tokens.png
summary:
  reports/training_curve_summary.json
```

Текущие данные для графика:

| Run | Points | Tokens range | Loss range | Median tok/s |
|---|---:|---:|---:|---:|
| autonomous-ctx1024-3h | 128 | 0.02M -> 20.81M | 10.5401 -> 7.3479 | 1936.6 |
| autonomous-ctx1024-12h | 512 | 20.97M -> 104.69M | 7.3157 -> 6.0257 | 1940.8 |
| bf16-batch22-smoke2 | 1 | 104.88M | 5.8506 | 2418.6 |

После большого `bf16` continuation этот же скрипт надо запустить с новым metrics-файлом, например:

```bash
python scripts/plot_training_curves.py \
  --runs \
  autonomous-ctx1024-3h=reports/autonomous-ctx1024-3h.metrics.jsonl \
  autonomous-ctx1024-12h=reports/autonomous-ctx1024-12h.metrics.jsonl \
  bf16-continuation=reports/<new-run-id>.metrics.jsonl
```

Так на одном графике будет видно падение loss до speed-mode и после перехода на `batch=22 bf16`.

Intermediate checkpoint comparison:

```text
script: scripts/compare_v1_checkpoints.py
report: ARTICLE/v1_intermediate_comparison.md
json: reports/v1_intermediate_comparison.json
figures:
  ARTICLE/figures/v1_intermediate_validation_loss.png
  ARTICLE/figures/v1_intermediate_firing_rate.png
```

Ключевой вывод `3h -> 12h`:

```text
val_wiki loss: 7.4983 -> 5.6233
val_lit  loss: 6.9081 -> 5.5918
val_habr loss: 7.4192 -> 5.8409

val_wiki firing: 12.56% -> 9.55%
val_lit  firing: 13.38% -> 9.70%
val_habr firing: 12.80% -> 9.63%
```

Это хороший sanity check: текущий v1 действительно улучшается при продолжении pretrain, и одновременно становится более sparse по spike activity. Генерации на instruction-like prompts все еще плохие, что ожидаемо до SFT.

## 9. Speed probe

Speed probe был запущен после 12h checkpoint, чтобы не запускать следующий длинный этап на неоптимальном batch/precision.

Команда:

```bash
PYTHONUNBUFFERED=1 python scripts/benchmark_training_speed.py \
  --checkpoint checkpoints/autonomous/autonomous-ctx1024-12h/final.pt \
  --manifest data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json \
  --steps 8 --batches 16,18,20,22,24 --precisions fp32,bf16 \
  2>&1 | tee reports/speed_probe_ctx1024_74m.log
```

Артефакты:

```text
reports/speed_probe_ctx1024_74m.log
reports/speed_probe_ctx1024_74m.json
```

Результаты:

| Batch | Precision | Status | Tok/s | Peak memory |
|---:|---|---|---:|---:|
| 16 | fp32 | ok | 1880 | 20.12 GB |
| 18 | fp32 | ok | 2151 | 22.54 GB |
| 20 | fp32 | ok | 2390 | 24.95 GB |
| 22 | fp32 | OOM | n/a | 26.81 GB before fail |
| 16 | bf16 | ok | 1881 | 17.14 GB |
| 18 | bf16 | ok | 2101 | 19.16 GB |
| 20 | bf16 | ok | 2347 | 21.19 GB |
| 22 | bf16 | ok | 2562 | 23.21 GB |
| 24 | bf16 | OOM | n/a | 24.68 GB before fail |

Решение:

```text
recommended next mode: batch=22, precision=bf16
expected speedup vs old batch=16 fp32: about +36%
```

Нюанс: `batch=24 bf16` показывает перспективную скорость на первом шаге, но падает по OOM. Для длинного запуска его нельзя использовать без дополнительных оптимизаций памяти.

Что надо сделать перед следующим длинным pretrain:

- `bf16` режим добавлен в `scripts/run_autonomous_training.py`;
- выставить batch 22;
- проверить, что loss/grad logging не ломается под autocast;
- сохранить конфиг запуска в report.

Preflight smoke после добавления `bf16`:

```text
run_id: bf16-batch22-smoke2
checkpoint source: checkpoints/autonomous/autonomous-ctx1024-12h/final.pt
manifest: data/tokenized/pretrain_300m/spikerugpt-pretrain.manifest.json
precision: bf16
batch_size: 22
steps: 6396 -> 6400
final_loss: 5.850621
avg_recent: 5.891558
grad_norm: 0.3651
speed: 2319-2419 tok/s in runner smoke
peak_mem_gb: 23.21
validation: ok on val_mixed, val_wiki, val_habr, val_lit
checkpoint_verify: ok
report: reports/bf16-batch22-smoke2.json
```

Interpretation: the main autonomous runner now supports the speed-probe-selected mode. This reduces the risk that the next long continuation fails for a trivial precision/batch issue.

## 10. SFT plan

SFT делаем только после нормального base pretraining. Иначе получится форма ассистента поверх слабого языкового ядра.

Текущие источники и решения:

| Источник | Решение | Комментарий |
|---|---|---|
| `IlyaGusev/saiga_scored` | оставить как основной | хороший русский SFT, но не делать 70-90% смеси |
| `Den4ikAI/russian_instructions` | оставить | полезный instruction source |
| `attn-signs/russian-easy-instructions` | добавить | Apache-2.0, 32,603 examples, хорош как простой русский instruction data |
| `IlyaGusev/oasst1_ru_main_branch` | оставить | готовая русская ветка OASST |
| `IlyaGusev/ru_turbo_alpaca` | optional/умеренно | полезно, но следить за provenance/licensing |
| `IlyaGusev/ru_turbo_saiga` | optional/умеренно | полезно, но следить за provenance/licensing |
| `CohereLabs/aya_dataset` / `aya_ru` | скорее отключить | multilingual source, мало пользы при наличии более чистых русских источников |
| `wikiomnia` | disabled | нестандартный loader/формат, не нужен в первом SFT |

Рекомендованный SFT mix:

```text
saiga_scored                  35-37%
russian_instructions          20%
russian_easy_instructions     15-16%
oasst1_ru_main_branch         12%
ru_turbo_alpaca               10%
ru_turbo_saiga                 5%
aya_ru                         0%
```

Текущий статус кода:

- `scripts/data/build_sft_mix.py` поддерживает `messages`;
- поддерживает `dialogue` arrays;
- нормализует роли `human/user/prompter -> user`;
- нормализует роли `assistant/bot/gpt/model -> assistant`;
- сохраняет `system`;
- делает hash/dedup;
- добавлена поддержка `data_files` и `.jsonl.zst` через `zstandard`.

Smoke SFT уже проверялся:

```text
data/sft/smoke_sft_mix.jsonl: 257 examples
data/sft/smoke_sft_mix_ilya.jsonl: 686 examples
```

Примерный состав `smoke_sft_mix_ilya`:

```text
saiga_scored: 350
russian_instructions: 155
aya_ru: 9
oasst1_ru_main_branch: 77
ru_turbo_alpaca: 54
ru_turbo_saiga: 41
```

Следующее действие по SFT:

- обновить `configs/data_sources.yaml`: добавить `russian_easy_instructions`, отключить `aya_ru` или снизить до нуля;
- пересобрать smoke SFT;
- вручную просмотреть samples;
- только после pretrain/annealing собирать финальный SFT corpus.

## 11. Сравнение с оригинальной SpikeGPT-работой

Вывод по сравнению:

- оригинальная работа показывает, что SpikeGPT-подход жизнеспособен;
- в их настройке был другой масштаб железа и, вероятно, более подготовленный training stack;
- наша задача не повторить их compute profile, а получить русскоязычную reproducible-итерацию;
- главным вкладом для нас становится русский tokenizer, data pipeline, фильтрация, spiking-метрики и честное сравнение v0/v1.

Важный разговорный вывод:

- RTX 5090 очень мощная, но одиночная карта ограничена памятью, batch и реализацией kernels;
- 4x V100 могут выигрывать по aggregate memory и параллелизму, но это не автоматически быстрее для нашей текущей реализации;
- реальная скорость определяется не только TFLOPS, а memory, kernels, batch, precision, data pipeline и distributed overhead.

## 12. LIF loop and speed ideas

Обсуждалась оптимизация LIF loop.

Позиция:

- LIF loop может быть узким местом;
- сначала надо зафиксировать baseline speed и memory;
- затем отдельно делать профилирование;
- изменения в ядре LIF опасны, потому что можно ускорить код и незаметно изменить динамику модели.

Приоритеты оптимизации:

1. Batch/precision tuning. Уже сделано, найден `batch=22 bf16`.
2. Включить bf16 в основной runner.
3. Проверить memory fragmentation env, например `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. Профилировать training step.
5. Только потом трогать LIF loop/kernel-level optimization.

## 13. Conference/article track

Контекст:

- тезисы по v0 уже были поданы и приняты;
- дедлайн черновика статьи фактически уже просрочен;
- финальная статья нужна примерно через две недели;
- сейчас цель не Habr-пост, а конференционная статья;
- Habr-style narrative можно использовать позже как популярное изложение.

Как честно описывать работу:

- v0: первая тестовая русскоязычная SpikeGPT-модель, proof of concept;
- v1: новая воспроизводимая итерация с улучшенным tokenizer, строгой очисткой, validation, backup и SFT pipeline;
- результаты v1 не надо обещать заранее, пока обучение не закончено;
- можно описать pipeline и промежуточные метрики как инженерный вклад.

Что взять из тезисов v0:

```text
dataset: Taiga-like Russian corpus, около 1.8B tokens
claim: первая русскоязычная SpikeGPT-style model
spiking metric: firing rate Russian 33.2% vs English 21.7%
relative firing rate increase: +53%
tau hierarchy Pearson: +0.996
```

Риск: надо сверить расхождение validation perplexity между тезисами и HF card v0. В обсуждении всплывали значения около 59.79 и около 67. Это надо привести к одному объяснению перед статьей.

Отдельные notes по сравнению v0/v1:

```text
ARTICLE/spikerugpt_v0_v1_comparison.md
reports/v0_v1_eval_small.json
reports/v0_v1_eval_small.log
scripts/compare_v0_v1_eval.py
```

## 14. Current background processes

На момент фиксации:

| tmux session | Назначение | Статус |
|---|---|---|
| `spikerugpt_build_1b` | сборка 1B pretrain shards | идет |
| `spikerugpt_1b_auto_pretrain` | watcher: дождаться 1B manifest, запустить early4h, затем 5d continuation | идет |
| `spikerugpt_speed_probe` | speed probe | завершен, session может остаться открытой |
| `spikerugpt_base_12h` | старый 12h run | завершен, session может остаться открытой |
| `spikerugpt_pilot_monitor` | старый мониторинг | устаревшее |
| `tqdm_demo` | тест `/ps`/progress | устаревшее |

Проверка статуса:

```bash
tmux ls
tail -n 50 reports/build_pretrain_1b.log
tail -n 50 reports/wait_and_launch_1b_pretrain.log
tail -n 50 reports/speed_probe_ctx1024_74m.log
nvidia-smi
```

Autostart watcher:

```text
script: scripts/wait_and_launch_1b_pretrain.sh
tmux: spikerugpt_1b_auto_pretrain
watch log: reports/wait_and_launch_1b_pretrain.log
early run: autonomous-ctx1024-1b-bf16-early4h
long run: autonomous-ctx1024-1b-bf16-5d
```

Логика watcher-а:

```text
1. wait until data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json exists
2. require >= 990M written tokens
3. start 4h early gate from checkpoints/autonomous/autonomous-ctx1024-12h/final.pt
4. if early report status/gate ok, start 5d continuation from early final.pt
5. save logs/reports/checkpoints and upload runs to HF backup repo
```

## 15. Immediate next actions

Следующий технический порядок:

1. Дождаться окончания `data/tokenized/pretrain_1b`.
2. Проверить manifest, размеры shard-ов и decode sample.
3. Запустить continuation только после manifest/decode gate.
4. Запустить продолжение pretrain с:

```text
resume_from: checkpoints/autonomous/autonomous-ctx1024-12h/final.pt
manifest: data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json
batch_size: 22
precision: bf16
context: 1024
```

5. В первые часы continuation смотреть early gates:

```text
loss finite and trending down
grad_norm finite and not exploding
tok/s around expected 2.3k-2.6k
peak_mem_gb around 23-24 GB
validation not worse than 12h checkpoint by a large margin
fixed-prompt generations less noisy over time
```

6. Во время pretrain обновить SFT config:

```text
add: russian_easy_instructions
disable: aya_ru
rebalance: saiga_scored около 35-37%
```

7. Собрать SFT smoke и показать примеры.
8. После следующего pretrain этапа сделать:

```text
validation
generation samples
loss graph
checkpoint backup to HF
short comparison v0 vs v1
```

9. После base/annealing запустить SFT.

## 16. Known risks

| Риск | Что делать |
|---|---|
| Маленькая модель может плохо отвечать после SFT | держать ожидания как demo/small SLM, а не assistant-grade LLM |
| Слишком шумный corpus | строгая фильтрация, validation по доменам |
| Простой GPU во время CPU data prep | заранее запускать независимые GPU-задачи: speed probe, generation eval, short continuation |
| OOM на длинном run | использовать `batch=22 bf16`, не `batch=24 bf16` |
| Сломанный bf16 runner | сначала smoke на малом числе steps |
| Слабая статья без финальных результатов | описать v0 как proof of concept, v1 как reproducible improved pipeline, добавить промежуточные метрики |
| Лицензии SFT | turbo-источники держать optional/private до проверки provenance |

## 17. Decisions to not reopen without new evidence

- Не переходить сейчас на 1B параметров.
- Не бросать текущий 74M checkpoint.
- Не делать SFT до нормального base checkpoint.
- Не использовать `batch=24 bf16` для длинного запуска.
- Не возвращаться к старому 50k tokenizer.
- Не делать Aya основой русского SFT mix.
