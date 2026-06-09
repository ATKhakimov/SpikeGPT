# SFT v2 superclean analysis

Date: 2026-06-09

## Dataset

SFT v2 was rebuilt from the previous SFT mix with strict one-turn filtering and repair of malformed
`russian_easy_instructions` records.

- Output: `data/sft/spikerugpt_sft_superclean_v2.jsonl`
- Examples: 45,000
- `role/content` artifacts in dataset quality gate: 0
- backtick/service-token artifacts in dataset quality gate: 0
- explicit code markers in dataset quality gate: 0

Composition:

- `russian_instructions`: 23,670
- `russian_easy_instructions`: 9,574
- `ru_turbo_alpaca`: 6,383
- `saiga_scored`: 3,071
- `ru_turbo_saiga`: 2,302

Length profile:

- median user chars: 57
- p95 user chars: 129
- median assistant chars: 310
- p95 assistant chars: 719
- max assistant chars: 844

## Training

Run: `sft-step43674-v2-superclean`

- base checkpoint: `checkpoints/autonomous/autonomous-ctx1024-1b-bf16-5d/latest.pt`
- final checkpoint: `checkpoints/sft/sft-step43674-v2-superclean/final.pt`
- steps: 690
- elapsed: about 18 minutes
- final train loss: 4.1768
- final supervised validation loss: 4.0997
- final supervised validation PPL: 60.32
- peak VRAM: 17.59 GB

## Generation Check

Comparison artifact:

- `ARTICLE/sft_v2_superclean/base_vs_sft_v1_v2_generations.md`
- `ARTICLE/sft_v2_superclean/base_vs_sft_v1_v2_generations.json`
- `ARTICLE/sft_v2_superclean/easy_base_vs_sft_v1_v2_generations.md`
- `ARTICLE/sft_v2_superclean/easy_base_vs_sft_v1_v2_generations.json`
- `ARTICLE/sft_v2_superclean/continuation_v0_base_sft_v2.md`
- `ARTICLE/sft_v2_superclean/continuation_v0_base_sft_v2.json`

Sampling:

- temperature: 0.45
- top_p: 0.75
- repetition_penalty: 1.18
- length: 100 tokens

Automatic artifact check over six prompts:

| model | role/content | backticks | code-like noise |
|---|---:|---:|---:|
| base | 0/6 | 0/6 | 0/6 |
| sft_v1_dirty | 0/6 | 1/6 | 1/6 |
| sft_v2_superclean | 0/6 | 0/6 | 0/6 |

Automatic artifact check over ten easy factual prompts:

| model | role/content | backticks | code-like noise |
|---|---:|---:|---:|
| base | 0/10 | 0/10 | 0/10 |
| sft_v1_dirty | 0/10 | 0/10 | 0/10 |
| sft_v2_superclean | 0/10 | 0/10 | 0/10 |

## Interpretation

SFT v2 successfully removes the visible technical corruption seen in SFT v1. In particular, the
`perplexity` prompt no longer produces `botype('python')`, markdown/backtick runs, or serialized
role/content fragments.

However, SFT v2 does not yet produce semantically reliable answers. The model often follows the
instruction wrapper but drifts into generic assistant text, service-style phrases, repetition, or
unrelated topics. This suggests that the main remaining limitation is not just SFT dataset format
corruption, but the combination of:

- small 74M model capacity;
- weak base-model factual/instruction competence after pretraining;
- SFT data still being stylistically generic and partially translated;
- only one short SFT pass at conservative learning rate.

For article purposes, SFT v1 versus SFT v2 is a useful ablation: cleaning the SFT data removes
format artifacts, but does not by itself solve semantic quality for a small SpikeGPT model.

An additional easy factual prompt set confirms this. Even very simple prompts such as `Что такое
Солнце?`, `Где находится Москва?`, `Сколько дней в неделе?`, and `Какая столица России?` are not
answered reliably by base, SFT v1, or SFT v2. This means the previous prompt set was a stress test,
but the failure is not only caused by overly complex prompts.

The original v0 evaluation style was also reproduced: plain continuation prompts copied from
`demo.py`, such as prose fragments, news leads, and historical leads. This is a fairer test for a
base language model than instruction QA. On these prompts, the old `v0_taiga_100m` often keeps a
more coherent prose/news continuation style, while `base_1b_74m` and especially `sft_v2_superclean`
frequently drift into generic web/service/news-like fragments. This supports the interpretation
that the current pretraining mix is broader but noisier than the Taiga-focused v0 setup, and that
SFT v2 should not be used as the main quality showcase.

## Spiking Activity After SFT

Artifact:

- `ARTICLE/sft_v2_superclean/base_vs_sft_v2_activity.json`
- `ARTICLE/sft_v2_superclean/base_vs_sft_v2_activity.md`

Small validation/activity probe, `ctx_len=512` for LM loss and `ctx_len=256` for LIF activity:

| Split | base loss | sft v2 loss | base BPB | sft v2 BPB | base firing | sft v2 firing | base silent | sft v2 silent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val_wiki | 4.5513 | 4.6404 | 0.7175 | 0.7315 | 0.0964 | 0.0977 | 0.4701 | 0.4674 |
| val_lit | 4.8371 | 4.9411 | 0.7412 | 0.7571 | 0.0960 | 0.0976 | 0.4897 | 0.4896 |
| val_habr | 5.0921 | 5.1943 | 0.5038 | 0.5139 | 0.0926 | 0.0944 | 0.4792 | 0.4788 |

Interpretation: SFT v2 slightly worsens base-LM continuation loss/BPB, but barely changes global
spiking activity. Mean firing rate stays around 9.3-9.8%, and silent-channel fraction stays around
47-49%. This suggests that this short SFT mainly changes the language-model head/behavioral
distribution while preserving the sparse LIF activity regime.

## Next Options

Recommended next experiment:

- continue from `sft-step43674-v2-superclean/final.pt`;
- train one additional short epoch on the same superclean data;
- use LR `1e-5` or `1.5e-5`;
- generate fixed prompts every 150-300 steps;
- stop if repetitions or refusal-like generic answers get worse.

Alternative:

- reduce SFT to 15k-25k highest-confidence examples;
- prefer direct factual/explanatory QA;
- remove vague self-help/business-style answers;
- retrain a shorter SFT pass and compare generations.
