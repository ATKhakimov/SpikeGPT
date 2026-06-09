# Article and Research Notes

This directory contains the paper draft, technical notes, poster assets and analysis artifacts for SpikeRuGPT.

## Primary Files

| File | Purpose |
|---|---|
| `spikerugpt_conference_article_draft.md` | current conference article draft |
| `spikerugpt_training_plan.md` | original training/data plan |
| `spikerugpt_technical_log.md` | factual engineering log of the training run |
| `spikerugpt_v0_v1_comparison.md` | comparison notes for v0 and v1 |
| `server_handoff.md` | server handoff state used during the run |

## Analysis Directories

| Directory | Purpose |
|---|---|
| `figures/` | article figures and training plots |
| `poster_assets/` | poster-ready tables and figures |
| `sft_v1_final_analysis/` | first SFT attempt and artifact analysis |
| `sft_v2_superclean/` | cleaned SFT dataset, generations and activity probes |
| `hf_upload/` | Hugging Face model-card fragments used for uploads |
| `templates/` | conference/sample formatting references |

## Generated Documents

Binary drafts such as `.doc`, `.docx` and notebooks are ignored by git. Keep the markdown article as the source of truth and rebuild the `.docx` with:

```bash
python scripts/article/build_conference_docx.py
```

from the repository root.
