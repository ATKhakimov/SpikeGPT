# SpikeRuGPT v1 SFT v2 superclean

This directory contains an experimental instruction-tuned checkpoint for the compact SpikeRuGPT v1 line.

It is uploaded into `Koras1k/spikerugpt-100M-Taiga` for convenience, but it is **not** a fine-tune of the old 100M Taiga checkpoint. It uses a newer 73.7M-parameter SpikeGPT/RWKV+LIF base model with a 32k SentencePiece BPE tokenizer.

## Files

- `final.pt`: final SFT v2 checkpoint.
- `reports/sft-step43674-v2-superclean.json`: training and validation summary.
- `reports/sft-step43674-v2-superclean.metrics.jsonl`: SFT training metrics.
- `reports/base_step43674_metrics.md`: base model LM evaluation before SFT.
- `analysis/`: generation comparisons and spiking-activity probes.

Tokenizer files are in the sibling directory:

- `../tokenizer-sp32k/spikerugpt-bpe-32k.model`
- `../tokenizer-sp32k/spikerugpt-bpe-32k.vocab`

## Base Model

| Parameter | Value |
|---|---:|
| Parameters | 73.7M |
| Layers | 12 |
| Hidden size | 512 |
| Context length | 1024 |
| Vocabulary | 32,000 |
| Base step | 43,674 |
| Base tokens seen | 944,590,848 |

Base LM evaluation before SFT:

| Split | Loss | PPL |
|---|---:|---:|
| val_mixed | 4.7730 | 118.27 |
| val_wiki | 4.2470 | 69.90 |
| val_habr | 4.9269 | 137.96 |
| val_lit | 4.9063 | 135.14 |
| train_sample | 4.7286 | 113.13 |

## SFT Dataset

SFT v2 was rebuilt from the previous SFT mix using stricter one-turn filtering and repair of malformed `russian_easy_instructions` records.

| Source | Examples |
|---|---:|
| russian_instructions | 23,670 |
| russian_easy_instructions | 9,574 |
| ru_turbo_alpaca | 6,383 |
| saiga_scored | 3,071 |
| ru_turbo_saiga | 2,302 |

Total: 45,000 examples.

Length profile:

- median user chars: 57
- p95 user chars: 129
- median assistant chars: 310
- p95 assistant chars: 719
- max assistant chars: 844

Quality gate:

- `role/content` artifacts: 0
- backtick/service-token artifacts: 0
- explicit code markers: 0

## SFT Training

| Metric | Value |
|---|---:|
| Run ID | sft-step43674-v2-superclean |
| Steps | 690 |
| Epochs | 1 |
| Examples seen | 44,100 |
| Initial train loss | 4.5728 |
| Final train loss | 4.1768 |
| Min train loss | 4.1202 |
| Elapsed | 1,088.5 s |
| Peak VRAM | 17.59 GB |
| Supervised validation loss | 4.0997 |
| Supervised validation PPL | 60.32 |

## Interpretation

SFT v2 successfully removes the visible technical corruption seen in the earlier dirty SFT attempt: serialized role/content fragments, markdown/code noise and Python-like artifacts are no longer observed in the fixed prompt checks.

However, this checkpoint should be treated as a research artifact rather than a production assistant model. It often follows the instruction format better than the base model, but factual reliability and semantic answer quality remain weak. The main limitation is the strength of the compact base model, not only SFT dataset formatting.

In continuation-style prompts, the older Taiga-focused v0 model often produces more natural Russian prose. The v1/SFT line is more useful as a reproducible research pipeline for studying data cleaning, tokenizer choice and spiking activity.

## Checksums

| File | SHA256 |
|---|---|
| `final.pt` | `a5ddc7f00111f0a721ea5373c0f2a8e75aeb8984525c917efb1877897cc313b9` |
| `../tokenizer-sp32k/spikerugpt-bpe-32k.model` | `ee47e1dd17fa209f91342a78308e40b85539ff597719ee8e2c786092571ecd8d` |
| `../tokenizer-sp32k/spikerugpt-bpe-32k.vocab` | `09723941d23ff20869d54f735044eb75d2ae12f7f4cd7d056d168eb11bea9c35` |
