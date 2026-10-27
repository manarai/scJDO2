# orient/S13 — whitened OT flow J

## Fixed gate
* **FAIL** — 0 of 4 linear embeddings pass every v3_transition seed (required ≥3).

| embedding | n | min AUC | median AUC | max AUC | all seeds pass |
|---|---:|---:|---:|---:|---|
| pca10 | 3 | 0.2171 | 0.3198 | 0.3736 | no |
| pca20 | 3 | 0.3736 | 0.4264 | 0.5836 | no |
| pca50 | 3 | 0.3239 | 0.3891 | 0.4861 | no |
| fa20 | 3 | 0.3000 | 0.4901 | 0.5333 | no |

## Held-out prediction (truth-free)
Flow fit: first sampling pair only; integrate middle-day cells to final day in 1-hour Euler steps. Best method for each dataset × embedding:

| dataset | embedding | method | energy_distance |
| --- | --- | --- | --- |
| larry | fa30 | mean_displacement | 0.515522 |
| larry | pca30 | whitened_flow | 0.751157 |
| v3_transition_s0 | fa20 | mean_displacement | 1.343283 |
| v3_transition_s0 | pca10 | mean_displacement | 1.470987 |
| v3_transition_s0 | pca20 | no_movement | 2.086663 |
| v3_transition_s0 | pca50 | no_movement | 2.042409 |
| v3_transition_s1 | fa20 | mean_displacement | 1.002119 |
| v3_transition_s1 | pca10 | mean_displacement | 1.050155 |
| v3_transition_s1 | pca20 | no_movement | 1.911018 |
| v3_transition_s1 | pca50 | no_movement | 1.873678 |
| v3_transition_s2 | fa20 | mean_displacement | 0.839790 |
| v3_transition_s2 | pca10 | mean_displacement | 0.752941 |
| v3_transition_s2 | pca20 | no_movement | 1.714929 |
| v3_transition_s2 | pca50 | no_movement | 1.685533 |

## Scope
Synthetic units use every S09/S10 cell. LARRY uses S11’s fixed `default_rng(0)` sample of 4,000 cells per time point as amended in S13 Revision 1.

## CPU
Timed unit CPU total from `cpu_ledger.csv`: **0.699 CPU-h** (task budget: 8.000 CPU-h).
