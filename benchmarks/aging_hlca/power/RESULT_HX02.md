# EXPLORATORY — hlca/HX02 power analysis

## What ran
Real H03 drift-subsample cells were used with synthetic, known-size injections only; no model was fitted. Each grid row rebuilt deterministic equal-count donor operators (`n_common=50`, 100 draws, 30-cell centroid) and used 50 seeded 80%-donor subsamples with 200 age permutations per target test.

## Definitions
For each type: `ref` = the H04 reference operator (pooled cells of donors aged < 45, `compare.centroid_operator`, 30 cells); `u` = its soft axis (eigenvector of its largest eigenvalue); `σ_u` = standard deviation of the young pooled cells' projections on `u`. Age is mapped to `t = (age − 10) / 66` (0 at 10 years, 1 at 76). Injections apply to **every cell** of the type, using that cell's donor age:
1. **Single-axis softening** (target: `isotropy_trend`), fold `F`:
   `J' = J + (F^(−t) − 1) · (uᵀ J u) · u uᵀ` — the curvature along `u`
   is divided by `F^t`; other directions unchanged.
   `F ∈ {1.25, 1.5, 2, 3, 5, 8}`.
2. **Uniform softening** (target: `stiffness_trend`), fold `F`:
   `J' = F^(−t) · J`. `F ∈ {1.1, 1.25, 1.5, 2, 3}`.
3. **Drift along the young soft axis** (target: `anticipation_trend`),
   size `D` (in units of `σ_u` over the age range):
   `z' = z + D · σ_u · t · u`. `D ∈ {0.5, 1, 2, 4, 8, 16}`.
Plus a **null row** (no injection, `F = 1`, `D = 0`) for the baseline pass rate.

## MDE table
| ann_level_3 | mde_single_axis_F | mde_uniform_F | mde_drift_D | null_pass_rate_iso | null_pass_rate_stiff | null_pass_rate_ant |
|---|---|---|---|---|---|---|
| AT2 | 3 | 1.1 | 2 | 0 | 0 | 0.02 |
| Basal | > max | 1.5 | 4 | 0 | 0 | 0.04 |
| Dendritic cells | > max | 1.25 | 1 | 0 | 0 | 0.02 |
| EC arterial | > max | 1.5 | 2 | 0 | 0 | 0 |
| EC capillary | > max | 2 | 0.5 | 0.02 | 0 | 0.34 |
| EC venous | > max | 1.1 | 4 | 0 | 0.24 | 0 |
| Fibroblasts | > max | 3 | 2 | 0 | 0 | 0 |
| Innate lymphoid cell NK | > max | 1.5 | 1 | 0 | 0 | 0.06 |
| Macrophages | > max | 1.5 | 4 | 0 | 0 | 0.08 |
| Monocytes | > max | 1.25 | 4 | 0 | 0.22 | 0 |
| Multiciliated lineage | > max | 1.5 | 2 | 0.2 | 0 | 0 |
| Secretory | > max | 1.5 | 4 | 0 | 0 | 0 |
| T cell lineage | > max | 1.1 | 4 | 0 | 0.96 | 0 |

Across the 13 types, the median MDE was >8× for single-axis softening, ×1.5 for uniform softening, and 2 σ for drift along the soft axis, over ages 10–76.

## Null rows
| ann_level_3 | null_pass_rate_iso | null_pass_rate_stiff | null_pass_rate_ant |
|---|---|---|---|
| AT2 | 0 | 0 | 0.02 |
| Basal | 0 | 0 | 0.04 |
| Dendritic cells | 0 | 0 | 0.02 |
| EC arterial | 0 | 0 | 0 |
| EC capillary | 0.02 | 0 | 0.34 |
| EC venous | 0 | 0.24 | 0 |
| Fibroblasts | 0 | 0 | 0 |
| Innate lymphoid cell NK | 0 | 0 | 0.06 |
| Macrophages | 0 | 0 | 0.08 |
| Monocytes | 0 | 0.22 | 0 |
| Multiciliated lineage | 0.2 | 0 | 0 |
| Secretory | 0 | 0 | 0 |
| T cell lineage | 0 | 0.96 | 0 |

Null pass rates exceeding 0.20: EC capillary/ant=0.34; EC venous/stiff=0.24; Monocytes/stiff=0.22; T cell lineage/stiff=0.96.
