# SpikeRuGPT Data Pipeline

This directory contains data-preparation scripts for the next SpikeRuGPT training run.

The scripts are intended to be run on the training machine, not on a laptop with limited disk/network.

## 1. Inspect sources

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl
```

This checks configs, splits, and a few streamed rows for each configured source.

## 2. Build tokenizer sample

```bash
python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --out data/tokenizer_sample/spikerugpt_tokenizer_sample.txt
```

## 3. Train SentencePiece tokenizer

```bash
python scripts/data/train_sentencepiece.py \
  --input data/tokenizer_sample/spikerugpt_tokenizer_sample.txt \
  --model-prefix tokenizer/spikerugpt-bpe-32k \
  --vocab-size 32000
```

## 4. Build validation text splits

```bash
python scripts/data/build_validation_splits.py \
  --config configs/data_sources.yaml \
  --output-dir data/validation_text
```

## 5. Build pretraining token shards

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain \
  --max-tokens 10000000000
```

## 6. Build SFT mix

```bash
python scripts/data/build_sft_mix.py \
  --config configs/data_sources.yaml \
  --out data/sft/spikerugpt_sft_mix.jsonl
```

Notes:

- HPLT and WikiOmnia are currently disabled in the config until their schemas/loaders are inspected on the server.
- `ru_turbo_*` datasets are intentionally not included in the first SFT mix until licensing/provenance is checked.
- Exact deduplication is implemented in memory for first-pass pilots. Large production runs should replace it with a disk-backed hash store.
- Text builders share the strict cleaning path in `common.py`: markup cleanup, light boilerplate removal, long-document chunking, language/quality filters, spam keyword filters, URL/email/phone limits, exact deduplication, SimHash near-deduplication, and per-source `filter_stats` in manifests.
- Long high-quality sources such as Wikipedia, Taiga prose, and Russian-PD are chunked instead of being rejected only because the original document exceeds `max_chars`.
