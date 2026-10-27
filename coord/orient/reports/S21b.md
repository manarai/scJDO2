# S21b

## Summary

Re-sampled the saved dyngen simulations at T=8; no new simulation was run.
All three bifurcating datasets met the ≥100 fork-cell selection rule: {'bifurcating_v2_s10_T8': 329, 'bifurcating_v2_s16_T8': 746, 'bifurcating_v2_s18_T8': 776}.

## What ran

Part A: `generate_experiment` only, with `experiment_synchronised(num_timepoints=8)` and the original dataset seeds; raw counts and truth exported through the unchanged S19 exporter.
Part B: median-library normalisation, log1p, PCA-15, S14 reference flow J in raw coordinates, drift seeds 0–2, and S10 score J (1500 epochs; exact ESM Laplacian).
Part C: truth was read only after all nine datasets had completed their fits. `hours = 48 × (day − 1)`.

## Gate numbers

| Gate | Result | Number |
|---|---:|---:|
| F1 flow-J fork AUC ≥ 0.85 | FAIL | minimum 0.3122803141875306 |
| F2 cycle rotation > linear, all seeds | FAIL | 6/9 seed comparisons |

### F1 scores

| Dataset | Drift seed | Fork cells | Committed cells | Fork AUC |
|---|---:|---:|---:|---:|
| bifurcating_v2_s10_T8 | 0 | 329 | 1981 | 1.0000 |
| bifurcating_v2_s10_T8 | 1 | 329 | 1981 | 1.0000 |
| bifurcating_v2_s10_T8 | 2 | 329 | 1981 | 1.0000 |
| bifurcating_v2_s16_T8 | 0 | 746 | 797 | 0.9860 |
| bifurcating_v2_s16_T8 | 1 | 746 | 797 | 0.9158 |
| bifurcating_v2_s16_T8 | 2 | 746 | 797 | 0.9301 |
| bifurcating_v2_s18_T8 | 0 | 776 | 525 | 0.3123 |
| bifurcating_v2_s18_T8 | 1 | 776 | 525 | 0.3421 |
| bifurcating_v2_s18_T8 | 2 | 776 | 525 | 0.3827 |

### F2 comparisons

| Cycle | Drift seed | Cycle median rotation | Linear median rotation | Pass |
|---|---:|---:|---:|---:|
| cycle_s0_T8 | 0 | 0.023975 | 0.096966 | False |
| cycle_s0_T8 | 1 | 0.022595 | 0.092090 | False |
| cycle_s0_T8 | 2 | 0.020327 | 0.078949 | False |
| cycle_s1_T8 | 0 | 0.308135 | 0.080511 | True |
| cycle_s1_T8 | 1 | 0.319063 | 0.084478 | True |
| cycle_s1_T8 | 2 | 0.191880 | 0.067961 | True |
| cycle_s2_T8 | 0 | 0.370177 | 0.077238 | True |
| cycle_s2_T8 | 1 | 0.397058 | 0.088928 | True |
| cycle_s2_T8 | 2 | 0.357649 | 0.073820 | True |

### Report-only pre-branch and linear fractions

| Dataset | Drift seed | Region | Cells | Fraction with Re λ₁ > 0 |
|---|---:|---|---:|---:|
| bifurcating_v2_s10_T8 | 0 | pre_branch | 971 | 0.0237 |
| bifurcating_v2_s10_T8 | 1 | pre_branch | 971 | 0.0649 |
| bifurcating_v2_s10_T8 | 2 | pre_branch | 971 | 0.0278 |
| bifurcating_v2_s16_T8 | 0 | pre_branch | 1268 | 1.0000 |
| bifurcating_v2_s16_T8 | 1 | pre_branch | 1268 | 1.0000 |
| bifurcating_v2_s16_T8 | 2 | pre_branch | 1268 | 1.0000 |
| bifurcating_v2_s18_T8 | 0 | pre_branch | 919 | 0.0000 |
| bifurcating_v2_s18_T8 | 1 | pre_branch | 919 | 0.0000 |
| bifurcating_v2_s18_T8 | 2 | pre_branch | 919 | 0.0000 |
| linear_s0_T8 | 0 | linear_all | 5000 | 0.0000 |
| linear_s0_T8 | 1 | linear_all | 5000 | 0.0000 |
| linear_s0_T8 | 2 | linear_all | 5000 | 0.0000 |
| linear_s1_T8 | 0 | linear_all | 5000 | 0.0000 |
| linear_s1_T8 | 1 | linear_all | 5000 | 0.0000 |
| linear_s1_T8 | 2 | linear_all | 5000 | 0.0000 |
| linear_s2_T8 | 0 | linear_all | 5000 | 0.0000 |
| linear_s2_T8 | 1 | linear_all | 5000 | 0.0000 |
| linear_s2_T8 | 2 | linear_all | 5000 | 0.0000 |

### Score-J AUC and rotation

| Dataset | Score-J fork AUC | Median rotation |
|---|---:|---:|
| bifurcating_v2_s10_T8 | 0.978927470544642 | 0.000000 |
| bifurcating_v2_s16_T8 | 0.5821949603237341 | 0.000000 |
| bifurcating_v2_s18_T8 | 0.8495336278841432 | 0.000000 |
| cycle_s0_T8 | NA | 0.000000 |
| linear_s0_T8 | NA | 0.000000 |
| cycle_s1_T8 | NA | 0.000000 |
| linear_s1_T8 | NA | 0.000000 |
| cycle_s2_T8 | NA | 0.000000 |
| linear_s2_T8 | NA | 0.000000 |

## S21 → S21b change

Every numeric score and J_ref field is compared row-wise in `s21_change_scores.csv` and `s21_change_jref.csv` (matched by dataset, operator, and drift seed; S21b's `_T8` suffix is removed). The source values, new values, and deltas are recorded for every field.

## J_ref diagnosis

All 36 operator rows are in `jref_diagnosis.csv`; columns include norm ratio, optimal scalar and relative error, median per-cell cosine, and Spearman λ₁ correlation. Full per-row values are also in the JSON report.

## Deviations

- The initial R launch omitted R_LIBS_USER and stopped before loading dyngen; the following attempt was interrupted after a default temporary reference download. All successful resampling units used the private .sandbox/Rlib and .sandbox/tmp.
- The first preparation launch was denied before preprocessing because the workspace sandbox blocked task-folder writes; the units were rerun after explicit task-folder authorization. No analysis outputs from that attempt were used.
The S21 fit and scoring calculations were reused unchanged; only S21b paths and dataset keys were selected, with a report adapter for the new T=8 outputs.

## Observations

T=8 fork counts were {'bifurcating_v2_s10_T8': 329, 'bifurcating_v2_s16_T8': 746, 'bifurcating_v2_s18_T8': 776}. S21 had 22 fork cells for s10 and zero for s16/s18; S21b now samples the specified fork region in all three datasets.

## Suggestions

If F2 remains negative, inspect dyngen's cycle milestone progression under the selected synchronized snapshots before proposing a new cycle-generation design.

## Questions

None.

## Artifacts

- `benchmarks/dyngen/MANIFEST.csv` (nine T=8 entries appended)
- `benchmarks/dyngen/s21b/part_a_T.csv`
- `benchmarks/dyngen/s21b/scores.csv`
- `benchmarks/dyngen/s21b/jref_diagnosis.csv`
- `benchmarks/dyngen/s21b/s21_change_scores.csv`
- `benchmarks/dyngen/s21b/s21_change_jref.csv`
- `benchmarks/figures/dyngen/s21b_lambda1_pca.png`
- `benchmarks/figures/dyngen/s21b_auc_bars.png`
