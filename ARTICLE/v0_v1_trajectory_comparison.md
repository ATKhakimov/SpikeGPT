# v0 vs v1 trajectory comparison

Сводка по трем точкам: старая публичная v0, промежуточная v1 3h и текущая v1 12h.

## Validation

| Split | v0 loss | v1 3h loss | v1 12h loss | v0 BPB | v1 3h BPB | v1 12h BPB |
|---|---:|---:|---:|---:|---:|---:|
| val_wiki | 4.3841 | 7.4983 | 5.6233 | 0.6101 | 1.0436 | 0.7826 |
| val_lit | 3.9927 | 6.9081 | 5.5918 | 0.6026 | 1.0426 | 0.8439 |
| val_habr | 5.0682 | 7.4192 | 5.8409 | 0.5599 | 0.8196 | 0.6452 |

## Spiking Activity

| Split | v0 firing | v1 3h firing | v1 12h firing | v0 silent | v1 3h silent | v1 12h silent |
|---|---:|---:|---:|---:|---:|---:|
| val_wiki | 0.1537 | 0.1256 | 0.0953 | 0.1900 | 0.4195 | 0.4907 |
| val_lit | 0.1553 | 0.1338 | 0.0971 | 0.1895 | 0.4401 | 0.5211 |
| val_habr | 0.1510 | 0.1280 | 0.0960 | 0.1721 | 0.4039 | 0.5072 |

## Interpretation

- v0 пока лучше v1 по LM loss/BPB, что ожидаемо: v0 заявлена как обученная на ~1.8B Taiga tokens.
- v1 12h заметно лучше v1 3h по всем проверенным split-ам.
- v1 при обучении становится более sparse: firing rate снижается, silent-channel fraction растет.
- Это хороший промежуточный вывод: v1 еще не догнала v0 по качеству, но trajectory правильная.

Figures:

```text
ARTICLE/figures/v0_v1_trajectory_validation_loss.png
ARTICLE/figures/v0_v1_trajectory_bpb.png
ARTICLE/figures/v0_v1_trajectory_firing_rate.png
```
