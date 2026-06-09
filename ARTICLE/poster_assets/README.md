# Poster Assets: SpikeRuGPT

This folder contains poster-ready tables, figures, and compact text blocks.

## Core Message

SpikeRuGPT demonstrates that a SpikeGPT/RWKV-style spiking language model can be adapted to Russian text. The strongest result is not chat quality, but the combination of Russian base-language modeling, continuation examples, and measurable neuromorphic sparsity via LIF firing rate.

## Recommended Poster Title

SpikeRuGPT: Russian Spiking Language Modeling and Neuromorphic Sparsity Analysis

## Main Quantitative Results

- v0 Taiga model: 92.43M actual parameters, ruGPT-3 BPE 50k, about 1.8B Taiga tokens, model-card validation PPL around 67; thesis/local notes report best PPL 59.79.
- v1 controlled base: 73.73M parameters, SentencePiece BPE 32k, 944.6M tokens seen, validation PPL 69.90 on wiki / 135.14 on lit / 137.96 on habr.
- Original English SpikeGPT: 215.40M parameters, reported 5B OpenWebText tokens, mean firing rate around 21.7%.
- Russian v0 firing rate: around 33.2% in original analysis, suggesting higher spike activity than English in the reported setup.
- Current v1 base firing rate: around 9.3-9.8% on local validation probes; SFT v2 barely changes this global activity.

## Honest Interpretation

- v0 remains the best Russian continuation baseline in this repo.
- v1 improves engineering reproducibility, tokenizer efficiency, and controlled data filtering, but current generation quality is weaker than v0.
- SFT v2 removes formatting artifacts from dirty SFT, but does not solve semantic quality for a 74M base model.
- The poster should frame this as an engineering/research trajectory, not as a finished assistant model.

## Tables

- `model_comparison.csv`: model scale and training setup.
- `validation_metrics.csv`: loss/PPL/BPB for v0/base/SFT probes.
- `spiking_activity.csv`: firing rate and silent-channel fraction.
- `sft_ablation.csv`: dirty vs super-clean SFT.

## Figures

- `figures/training_curve.png`
- `figures/sparsity_summary.png`
- `figures/spike_sparsity.png`
- `figures/sparsity_heatmap.png`
- `figures/lif_tau_evolution.png`
- `figures/lif_tau_final.png`
- `figures/v0_v1_trajectory_firing_rate.png`
- `figures/training_loss_by_tokens.png`
- `figures/poster_model_scale.png`
- `figures/poster_validation_bpb.png`
- `figures/poster_firing_rate_summary.png`

## Poster Layout Suggestion

1. Left column: motivation, SpikeGPT architecture, Russian adaptation.
2. Middle column: training setup, tokenizer/parameter comparison, validation metrics.
3. Right column: spiking activity, firing-rate plots, continuation examples, limitations.

## Short Figure Captions

- `poster_model_scale.png`: Parameter count and training-token scale for Russian v0/v1 and English original SpikeGPT.
- `poster_validation_bpb.png`: Cross-tokenizer validation comparison using bits per byte.
- `poster_firing_rate_summary.png`: Mean LIF firing rate across compared models/probes.
- `sparsity_summary.png`: Russian versus English spike activity from the original v0 analysis.
- `v0_v1_trajectory_firing_rate.png`: v1 pretraining trajectory toward lower firing rate.

## Limitations To State Explicitly

- v0, v1, and English original differ in model size, tokenizer, data, and training tokens.
- Token-level PPL is not directly comparable across tokenizers; use BPB for cross-tokenizer comparison.
- Firing rate is an activity/efficiency proxy, not a direct text-quality metric.
- SFT results are diagnostic; current SFT model should not be presented as final assistant quality.
