# S02 end-to-end orientation results

## What ran
- For every S01 manifest input (four presets × seeds 0–2) and PCA/FA embedding, fitted the frozen 1,500-epoch ESM EnergyDriftAdapter (hidden 96, depth 3, batch 256, lr 5e-4, beta 1.0, exact Laplacian; seed 0).
- Fitted LabelledRateModel defaults (seed 0) on observed `new`/`total` counts and pulse hours; derived raw count velocity, mapped it through the stored observed-count transform, and smoothed it with default seed-0 MeasuredDrift.
- Ran lineage, smoothed-labelling, combined, and raw-labelling sensitivity orientation calls. Generator truth was read only after those fits to score signs.

## Gates (verbatim)
- On default / PCA / smoothed labelling, for all three seeds: lineage sign agreement ≥ 0.90 and oriented fraction ≥ 0.3; labelling ≥ 0.85 and ≥ 0.3; combined ≥ 0.90 and ≥ 0.5.
- On stationary_cycle / PCA / smoothed combined, non_equilibrium is ≥ 0.8 of cells with an identified equilibrium flag for every seed.
- J hash is unchanged in every `orient` call. Symmetry check maximum is ≤ 1e-4 for every energy fit.

## Gate results
| gate | minimum | per_seed | minimum_observed | pass |
|---|---|---|---|---|
| default pca smoothed lineage sign_agreement | 0.9 | 0.8087, 0.7763, 0.7735 | 0.7734851565677104 | False |
| default pca smoothed lineage oriented_fraction | 0.3 | 0.4569, 0.4531, 0.4298 | 0.42983709631323236 | True |
| default pca smoothed labelling sign_agreement | 0.85 | 0.8609, 0.8410, 0.8447 | 0.8410119292472233 | False |
| default pca smoothed labelling oriented_fraction | 0.3 | 0.7405, 0.7409, 0.7079 | 0.7079165475850243 | True |
| default pca smoothed lineage+labelling sign_agreement | 0.9 | 0.8334, 0.8078, 0.8027 | 0.8027154906397861 | False |
| default pca smoothed lineage+labelling oriented_fraction | 0.5 | 0.8043, 0.8152, 0.7822 | 0.7822234924264075 | True |
| stationary_cycle pca smoothed combined non_equilibrium among identified | 0.8 | 1.0000, 1.0000, 1.0000 | 1.0 | True |
| J hash unchanged in every orient call | 1.0 | True | 1.0 | True |

## PCA summary (mean and minimum over seeds)
| preset | source_set | labelling_variant | n_rows | sign_agreement_mean | sign_agreement_min | sign_agreement_moving_mean | oriented_fraction_mean | oriented_fraction_min | conflict_rate_mean | rotation_share_median_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| batch | labelling | smoothed | 3 | 0.8411 | 0.8113 | 0.8832 | 0.6604 | 0.5791 | — | 0.6748 |
| batch | lineage | none | 3 | 0.7618 | 0.7167 | 0.8301 | 0.3949 | 0.3325 | — | — |
| batch | lineage+labelling | smoothed | 3 | 0.7978 | 0.7596 | 0.8553 | 0.7434 | 0.6618 | 0.0532 | 0.6748 |
| default | labelling | smoothed | 3 | 0.8489 | 0.841 | 0.891 | 0.7298 | 0.7079 | — | 0.6633 |
| default | lineage | none | 3 | 0.7861 | 0.7735 | 0.8317 | 0.4466 | 0.4298 | — | — |
| default | lineage+labelling | smoothed | 3 | 0.8146 | 0.8027 | 0.8622 | 0.8005 | 0.7822 | 0.0412 | 0.6633 |
| stationary_cycle | labelling | smoothed | 3 | 0.9724 | 0.9663 | 0.9982 | 0.6222 | 0.5166 | — | 0.7416 |
| stationary_cycle | lineage | none | 3 | — | — | — | 0.0 | 0.0 | — | — |
| stationary_cycle | lineage+labelling | smoothed | 3 | 0.9724 | 0.9663 | 0.9982 | 0.6222 | 0.5166 | — | 0.7416 |
| stress | labelling | smoothed | 3 | 0.8294 | 0.819 | 0.8677 | 0.7154 | 0.704 | — | 0.66 |
| stress | lineage | none | 3 | 0.7592 | 0.7445 | 0.7984 | 0.1994 | 0.1971 | — | — |
| stress | lineage+labelling | smoothed | 3 | 0.8094 | 0.7971 | 0.8491 | 0.7528 | 0.747 | 0.0841 | 0.66 |

## PCA equilibrium and stationarity flags (combined, smoothed; mean over seeds)
| preset | fraction_equilibrium_mean | fraction_non_equilibrium_mean | fraction_not_identified_mean | fraction_stationary_mean | fraction_non_stationary_mean |
|---|---|---|---|---|---|
| batch | 0.0 | 1.0 | 0.0 | 0.9117 | 0.0883 |
| default | 0.0 | 1.0 | 0.0 | 0.9124 | 0.0876 |
| stationary_cycle | 0.0 | 1.0 | 0.0 | 0.9227 | 0.0773 |
| stress | 0.0 | 1.0 | 0.0 | 0.8958 | 0.1042 |

## PCA raw-versus-smoothed labelling sensitivity (mean and minimum over seeds)
| preset | labelling_variant | sign_agreement_mean | sign_agreement_min | sign_agreement_moving_mean | oriented_fraction_mean | rotation_share_median_mean | labelled_fit_nll_mean |
|---|---|---|---|---|---|---|---|
| batch | raw | 0.866 | 0.8327 | 0.9314 | 0.578 | 0.6918 | -34.3899 |
| batch | smoothed | 0.8411 | 0.8113 | 0.8832 | 0.6604 | 0.6748 | -34.3899 |
| default | raw | 0.8962 | 0.8664 | 0.9438 | 0.6278 | 0.6928 | -25.6462 |
| default | smoothed | 0.8489 | 0.841 | 0.891 | 0.7298 | 0.6633 | -25.6462 |
| stationary_cycle | raw | 0.9628 | 0.9539 | 0.9984 | 0.6461 | 0.6918 | -0.0074 |
| stationary_cycle | smoothed | 0.9724 | 0.9663 | 0.9982 | 0.6222 | 0.7416 | -0.0074 |
| stress | raw | 0.8357 | 0.7886 | 0.8876 | 0.6167 | 0.6904 | -27.2731 |
| stress | smoothed | 0.8294 | 0.819 | 0.8677 | 0.7154 | 0.66 | -27.2731 |

## Complete outputs
- `results.csv`: one row per dataset × embedding × source set × labelling variant.
- `summary.csv`: mean and minimum-over-seed summaries per preset × embedding × source set.
- Fitted H5AD outputs and Torch state dictionaries: `benchmarks/synthetic_multimodal/data/s02/` (local, gitignored).
- Figures: `benchmarks/figures/orient/s02/`.
