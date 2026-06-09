# Scripts

This directory contains the reproducible SpikeRuGPT v1 pipeline.

## Data

See [`scripts/data/README.md`](data/README.md) for dataset inspection, filtering, tokenizer training, validation split creation, pretraining shard construction and SFT-mix construction.

## Training

| Script | Purpose |
|---|---|
| `run_autonomous_training.py` | resumable v1 pretraining loop with metrics, checkpoints and optional HF backup |
| `train_pilot.py` | small pilot pretraining run |
| `train_smoke_overfit.py` | fixed-batch overfit smoke test |
| `train_sft.py` | supervised fine-tuning with assistant-only loss |
| `watch_and_finalize_sft.py` | SFT watcher/finalizer used during long background runs |
| `wait_and_launch_1b_pretrain.sh` | tmux-oriented helper for launching the 1B-token run after shard creation |

## Evaluation and Analysis

| Script | Purpose |
|---|---|
| `compare_v0_v1_eval.py` | evaluate v0/v1 checkpoints on shared validation text |
| `compare_v0_v1_generations.py` | compare continuation generations |
| `compare_v1_checkpoints.py` | compare intermediate v1 checkpoints |
| `compare_continuation_demo.py` | reproduce v0-style continuation prompts across checkpoints |
| `compare_sft_generations.py` | compare base/SFT generations and artifact checks |
| `analyze_spiking_activity.py` | LIF firing-rate and silent-channel probes |
| `plot_training_curves.py` | plot pretraining metrics |
| `build_poster_assets.py` | build poster tables and figures |
| `generate_original_english_spikegpt.py` | run the open English SpikeGPT reference model |

## Article

`article/build_conference_docx.py` converts `ARTICLE/spikerugpt_conference_article_draft.md` into a conference-style `.docx`.
