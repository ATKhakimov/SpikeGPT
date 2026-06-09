# SpikeRuGPT

SpikeRuGPT is a Russian adaptation of the SpikeGPT architecture: an RWKV-style autoregressive language model with LIF spiking neurons.

The repository contains the original Russian 100M proof-of-concept model code, the newer reproducible v1 training pipeline, SFT experiments, and article/poster materials for the neuromorphic sparsity analysis.

## Status

This is a research repository, not a production assistant model.

The strongest current result is Russian base language modeling plus measurable spiking activity analysis. The SFT checkpoint is included as a diagnostic artifact: it improves formatting and removes visible dataset-corruption artifacts, but it is not yet a reliable factual QA/chat model.

## Model Artifacts

Weights are not stored in git. Public artifacts are on Hugging Face:

- v0 Taiga checkpoint and tokenizer: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga
- v1 SFT v2 superclean checkpoint: https://huggingface.co/Koras1k/spikerugpt-100M-Taiga/tree/main/sft-v2-superclean

| Line | Parameters | Tokenizer | Training data | Role |
|---|---:|---|---|---|
| `v0_taiga_100m` | 92.4M | ruGPT-3 BPE, 50k | Taiga, about 1.8B tokens | first Russian SpikeGPT baseline |
| `v1_base_74m` | 73.7M | SentencePiece BPE, 32k | filtered mixed Russian corpus, about 0.95B tokens seen | reproducible pretraining pipeline |
| `v1_sft_v2_superclean` | 73.7M | SentencePiece BPE, 32k | 45k short one-turn Russian instructions | SFT/data-cleaning diagnostic |
| `SpikeGPT-OpenWebText-216M` | 215.4M | GPT-NeoX tokenizer | OpenWebText, original English line | reference model for comparison |

## Key Results

| Experiment | Result |
|---|---|
| v0 Taiga validation perplexity | best validation PPL 59.79 |
| Russian v0 spiking activity | mean firing rate 33.2% |
| English SpikeGPT reference activity | mean firing rate 21.7% |
| v1 base final evaluation | val_wiki PPL 69.90, val_mixed PPL 118.27 |
| SFT v2 supervised validation | loss 4.0997, PPL 60.32 |

Main interpretation:

- SpikeGPT-style spiking language models can be trained on Russian text.
- Russian text produced higher measured firing rate than the English reference in this setup, so language and corpus choice matter for neuromorphic event budgets.
- Smaller vocabularies are useful for small models: a 32k SentencePiece tokenizer saves parameters without materially hurting Russian token density in the measured validation sample.
- Strict SFT cleaning removes visible `role/content` and code-like artifacts, but SFT alone does not compensate for a weak small base model.

## Repository Layout

```text
src/                    SpikeGPT model, trainer utilities, vendored spikingjelly subset
cuda/                   WKV CUDA kernels
train.py                original v0 training entrypoint
generate.py             v0 generation entrypoint
demo.py                 continuation-prompt demo for v0
scripts/                v1 data, training, evaluation, SFT and analysis tools
scripts/data/           dataset inspection, filtering, tokenizer and shard builders
configs/                data-source and SFT configs
analysis/               original v0 sparsity/LIF analysis scripts and figures
ARTICLE/                paper draft, technical logs, poster assets and SFT analysis
NLU/                    original SpikeGPT NLU evaluation scripts
static/                 static project images
```

Ignored local directories:

- `data/`
- `tokenizer/`
- `checkpoints/`
- `models/`
- `reports/`
- `logs/`

These are intentionally excluded because real runs create large local artifacts.

## Documentation Map

- Conference article draft: [`ARTICLE/spikerugpt_conference_article_draft.md`](ARTICLE/spikerugpt_conference_article_draft.md)
- Technical training log: [`ARTICLE/spikerugpt_technical_log.md`](ARTICLE/spikerugpt_technical_log.md)
- Training/data plan: [`ARTICLE/spikerugpt_training_plan.md`](ARTICLE/spikerugpt_training_plan.md)
- v0/v1 comparison notes: [`ARTICLE/spikerugpt_v0_v1_comparison.md`](ARTICLE/spikerugpt_v0_v1_comparison.md)
- Poster assets: [`ARTICLE/poster_assets/README.md`](ARTICLE/poster_assets/README.md)
- SFT v2 analysis: [`ARTICLE/sft_v2_superclean/README.md`](ARTICLE/sft_v2_superclean/README.md)
- Data pipeline docs: [`scripts/data/README.md`](scripts/data/README.md)

## Setup

Install PyTorch for your CUDA version first, then install the project dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_data.txt
```

For RTX 50xx / CUDA 12.8 environments, use the dedicated environment notes in:

```bash
requirements_runpod_cu128.txt
```

Some scripts require Hugging Face access for dataset/model downloads. Set `HF_TOKEN` or login through `huggingface_hub` before running data-preparation jobs.

## v1 Data and Training Pipeline

Inspect configured data sources:

```bash
python scripts/data/inspect_sources.py \
  --config configs/data_sources.yaml \
  --out reports/data_source_inspection.jsonl
```

Build tokenizer sample and train the 32k SentencePiece tokenizer:

```bash
python scripts/data/build_tokenizer_sample.py \
  --config configs/data_sources.yaml \
  --out data/tokenizer_sample/spikerugpt_tokenizer_sample.txt

python scripts/data/train_sentencepiece.py \
  --input data/tokenizer_sample/spikerugpt_tokenizer_sample.txt \
  --model-prefix tokenizer/spikerugpt-bpe-32k \
  --vocab-size 32000
```

Build pretraining shards:

```bash
python scripts/data/build_pretrain_shards.py \
  --config configs/data_sources.yaml \
  --tokenizer-kind sentencepiece \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --output-dir data/tokenized/pretrain_1b \
  --max-tokens 1000000000
```

Run autonomous pretraining:

```bash
python scripts/run_autonomous_training.py \
  --manifest data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json \
  --tokenizer tokenizer/spikerugpt-bpe-32k.model \
  --run-id autonomous-ctx1024-1b-bf16-5d \
  --precision bf16
```

Build the superclean SFT dataset and train SFT:

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

## v0 Generation

The original v0 code expects a local checkpoint and tokenizer. Download them from Hugging Face, then run:

```bash
python generate.py \
  --prompt "Осенний лес был тих и задумчив." \
  --checkpoint checkpoints/spikegpt-ru-175.pth \
  --temperature 0.85 \
  --top_p 0.9
```

## Analysis

Spiking activity and model-comparison scripts:

```bash
python scripts/analyze_spiking_activity.py
python scripts/compare_v0_v1_eval.py
python scripts/compare_continuation_demo.py
python scripts/compare_sft_generations.py
python scripts/build_poster_assets.py
```

The article can be rebuilt from markdown:

```bash
python scripts/article/build_conference_docx.py
```

The generated `.docx` is intentionally ignored by git.

## Citation

If you use this repository, cite the original SpikeGPT paper:

```bibtex
@article{zhu2023spikegpt,
    title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
    author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
    journal = {arXiv preprint arXiv:2302.13939},
    year    = {2023}
}
```

## License

This repository is released under the MIT license. See [`LICENSE`](LICENSE).
