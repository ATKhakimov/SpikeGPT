# SpikeRuGPT v0 vs v1 comparison notes

Дата фиксации: 2026-06-02  
Назначение: материал для конференционной статьи и последующего описания эволюции модели.

## 1. Что сравниваем

| Версия | Источник | Роль |
|---|---|---|
| v0 | `Koras1k/spikerugpt-100M-Taiga` | предыдущая публичная proof-of-concept модель |
| v1 | текущий локальный checkpoint `autonomous-ctx1024-12h` | новая воспроизводимая итерация, еще не финальная |

v0 скачана локально:

```text
models/v0_spikerugpt_100m_taiga
models/v0_spikerugpt_100m_taiga/spikegpt-ru-175.pth
```

HF metadata:

```text
repo: Koras1k/spikerugpt-100M-Taiga
sha: e10cf8202708f74ef6173a3d872fd4a236a864f6
license: MIT
checkpoint size: 369,795,555 bytes
```

## 2. v0 model card facts

Из model card v0:

```text
architecture: SpikeGPT / RWKV + MultiStepLIF
claimed parameters: ~100M
actual checkpoint parameters: 92.43M
layers: 12
d_model: 512
tokenizer: ruGPT-3 Large BPE, vocab=50,258
corpus: Taiga, taiga_stripped_rest + taiga_stripped_proza
training data: ~1.8B tokens
context length: 1024
hardware: NVIDIA A100 SXM 80GB
checkpoint: epoch 175
valid perplexity: ~67 in card
firing rate: 33.2%
silent neurons: 66.8%
```

Important article nuance: thesis/local notes mention a validation perplexity around `59.79`, while the HF model card says `~67`. This mismatch should be explained before final submission. Possible reason: different checkpoint, validation split, or reporting stage.

## 3. Static architecture comparison

Both checkpoints use the same high-level shape:

```text
layers: 12
n_embd: 512
context length: 1024
architecture family: SpikeGPT/RWKV-style + LIF
```

But the tokenizer/vocab changed:

| Metric | v0 | v1 |
|---|---:|---:|
| Vocab size | 50,258 | 32,000 |
| Total params | 92.43M | 73.73M |
| Embedding params | 25.73M | 16.38M |
| Head params | 25.73M | 16.38M |
| Embedding + head | 51.46M | 32.77M |

Parameter interpretation:

- v0 spends about `55.7%` of all parameters on embedding + lm head;
- v1 spends about `44.4%` of all parameters on embedding + lm head;
- switching to 32k vocab saves about `18.7M` parameters in embedding/head;
- this is a major efficiency argument for a small SpikeGPT model.

Article wording: v1 is not just “smaller”; it reallocates a larger fraction of model capacity away from vocabulary tables and toward the recurrent/spiking body.

## 4. Tokenizer comparison

Tokenizer density was checked on the same local validation text sample:

```text
texts: 1200
chars: 8,208,103
bytes: 14,800,256
```

| Tokenizer | Tokens | Chars/token | Bytes/token |
|---|---:|---:|---:|
| v0 ruGPT-3 BPE 50k | 1,895,156 | 4.331 | 7.810 |
| v1 SentencePiece BPE 32k | 1,878,879 | 4.369 | 7.877 |

Interpretation:

- downloaded v0 tokenizer is already fairly dense on this Russian sample;
- v1 tokenizer is only slightly denser on this sample;
- the stronger v1 argument is not huge token-count reduction against this exact v0 tokenizer;
- the stronger argument is similar or slightly better density with much smaller vocab and lower parameter cost.

Prompt-level token counts:

| Prompt | v0 tokens | v1 tokens |
|---|---:|---:|
| `Москва — это` | 3 | 3 |
| `Нейроморфные вычисления позволяют` | 8 | 6 |
| `Напиши краткое объяснение, что такое SpikeGPT.` | 15 | 15 |
| `Почему русский язык сложен для языковых моделей?` | 10 | 10 |

## 5. Preliminary validation comparison

Script:

```text
scripts/compare_v0_v1_eval.py
```

Command:

```bash
PYTHONUNBUFFERED=1 python scripts/compare_v0_v1_eval.py \
  --ctx-len 512 \
  --max-docs 80 \
  --max-sequences 16 \
  --batch-size 2 \
  --splits val_wiki val_lit val_habr \
  2>&1 | tee reports/v0_v1_eval_small.log
```

Artifacts:

```text
reports/v0_v1_eval_small.log
reports/v0_v1_eval_small.json
```

Generation examples:

```text
scripts/compare_v0_v1_generations.py
reports/v0_v1_generations.json
reports/v0_v1_generations.log
ARTICLE/v0_v1_generation_examples.md
```

Why BPB matters: v0 and v1 use different tokenizers, so token-level perplexity is not directly comparable. `bits/byte` is a better cross-tokenizer metric.

Small eval results:

| Split | Model | Loss | PPL | BPB |
|---|---|---:|---:|---:|
| val_wiki | v0 | 4.3841 | 80.2 | 0.6101 |
| val_wiki | v1 12h | 5.6233 | 276.8 | 0.7826 |
| val_lit | v0 | 3.9927 | 54.2 | 0.6026 |
| val_lit | v1 12h | 5.5918 | 268.2 | 0.8439 |
| val_habr | v0 | 5.0682 | 158.9 | 0.5599 |
| val_habr | v1 12h | 5.8409 | 344.1 | 0.6452 |

Interpretation:

- v0 is currently better on these language-modeling metrics;
- this is expected, because v0 is an epoch-175 model trained on the model-card claim of about `1.8B` Taiga tokens;
- current v1 checkpoint has only about `104.8M` tokens seen in the 12h run;
- v1 is not yet supposed to beat v0 on raw language modeling quality;
- v1 should be re-evaluated after the 1B-token continuation and quality annealing.

Article wording: v0 remains a strong proof-of-concept baseline, while v1 is the controlled reproduction/improvement track. At the current checkpoint, v1 improves engineering reproducibility and parameter efficiency, not yet final perplexity.

## 6. Important compatibility note

v0 checkpoint loads as a raw `OrderedDict` and is compatible with the current `src.model.GPT` shape after setting:

```text
vocab_size: 50258
n_layer: 12
n_embd: 512
lif: MultiStepLIFNode tau=2.0
```

Original generation code expected `backend="cupy"` for `MultiStepLIFNode`, but CuPy is not installed in this environment. For evaluation, `scripts/compare_v0_v1_eval.py` uses:

```text
backend="torch"
```

This preserves the intended node type but is slower than CuPy.

## 7. Suggested article framing

Possible narrative:

1. v0 demonstrated that a Russian SpikeGPT-style model can be trained and publicly released.
2. v0 was useful but had proof-of-concept limitations:
   - Taiga-heavy data;
   - older ruGPT-3 tokenizer;
   - large vocabulary cost for a small model;
   - less reproducible data/training pipeline;
   - model card and thesis metrics need reconciliation.
3. v1 rebuilds the experiment more carefully:
   - explicit data inspection;
   - stricter filtering and deduplication;
   - fixed validation splits;
   - 32k SentencePiece tokenizer;
   - local tokenized shards;
   - autonomous training runner;
   - HF backup of artifacts;
   - speed/memory probes;
   - planned SFT mix.
4. Current v1 12h checkpoint is not final and does not yet beat v0 on BPB.
5. The fair final comparison should be repeated after v1 continuation pretraining and quality annealing.

## 8. Next comparison tasks

Before final article:

- run the same `scripts/compare_v0_v1_eval.py` after v1 1B continuation;
- add `val_mixed` and maybe a larger sample size;
- generate fixed prompt outputs from v0 and v1;
- compare firing-rate statistics on the same prompt set;
- resolve v0 perplexity mismatch: thesis value vs HF model card value;
- add a short qualitative table with 5-8 prompts.

## 9. Preliminary spiking activity analysis

Script:

```text
scripts/analyze_spiking_activity.py
```

Command:

```bash
PYTHONUNBUFFERED=1 python scripts/analyze_spiking_activity.py \
  --ctx-len 256 \
  --max-docs 60 \
  --max-sequences 12 \
  --batch-size 2 \
  --splits val_wiki val_lit val_habr \
  2>&1 | tee reports/spiking_activity_v0_v1.log
```

Artifacts:

```text
reports/spiking_activity_v0_v1.json
reports/spiking_activity_v0_v1.csv
reports/spiking_lif_parameters_v0_v1.csv
ARTICLE/figures/spiking_firing_rate_val_wiki.png
ARTICLE/figures/spiking_firing_rate_val_lit.png
ARTICLE/figures/spiking_firing_rate_val_habr.png
ARTICLE/figures/spiking_firing_rate_mean_by_layer.png
ARTICLE/figures/spiking_silent_channels_by_layer.png
ARTICLE/figures/spiking_v1_tau_threshold_profile.png
```

Summary:

| Split | Model | Mean firing rate | Mean silent-channel fraction |
|---|---|---:|---:|
| val_wiki | v0 | 0.1537 | 0.1900 |
| val_wiki | v1 12h | 0.0953 | 0.4907 |
| val_lit | v0 | 0.1553 | 0.1895 |
| val_lit | v1 12h | 0.0971 | 0.5211 |
| val_habr | v0 | 0.1510 | 0.1721 |
| val_habr | v1 12h | 0.0960 | 0.5072 |

Interpretation:

- v1 currently has substantially lower firing activity than v0 on all checked domains;
- v1 also has a much larger fraction of silent channels in this small probe;
- this supports a neuromorphic/sparsity discussion, but should not be framed as a quality win by itself;
- lower firing rate may indicate better event sparsity, undertraining, stronger thresholds, or different LIF dynamics;
- final interpretation requires repeating the same analysis after the 1B continuation and quality annealing.

v1 learnable LIF parameter profile:

| Branch | Tau first layer | Tau last layer | Threshold first layer | Threshold last layer |
|---|---:|---:|---:|---:|
| lif1 | 2.0307 | 3.2229 | 0.8863 | 0.6977 |
| lif2 | 1.5343 | 2.7185 | 0.8050 | 0.7848 |

Article angle:

- v0 can be described as the initial Russian SpikeGPT baseline with fixed MultiStepLIF dynamics;
- v1 introduces learnable LIF parameters and shows a clear layer-wise timescale hierarchy;
- the increasing tau profile is useful for the neuromorphic part of the paper because it suggests slower integration in upper layers;
- firing-rate plots can be included as experimental evidence that the model remains event-sparse.
