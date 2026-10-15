# Aging basins — A01 results

## §1 Data

The unmodified `scripts/fetch_tms_aging.py` fetched Tabula Muris Senis from
cellxgene Census `2024-07-01`. Both observed dataset IDs match the frozen
PREREG §1 identifiers. Raw H5AD matrices are gitignored local artifacts; their
checksums are in `coord/aging/reports/A01.json`.

| Tissue | Dataset ID | Raw cells | Raw genes | `4 weeks` excluded (cells / donors) | Prefix-30 excluded (cells / donors) | Young, prefix 3 (cells / donors) | Old, prefixes 18/21/24 (cells / donors) |
|---|---|---:|---:|---:|---:|---:|---:|
| Lung | `0fb7916e-7a68-4a4c-a441-3ab3989f29a7` | 24,540 | 52,437 | 2,508 / 2 | 11,093 / 4 | 3,612 / 4 | 7,327 / 6 |
| Marrow | `0bd1a1de-3aee-40e0-b2ec-86c7a30c7149` | 40,220 | 52,437 | 3,027 / 2 | 13,496 / 3 | 3,490 / 2 | 20,207 / 10 |

There are no labels outside the §2b `4 weeks` exclusion and the §2c
prefix-based young/old mapping. The final old group restores the frozen
18–24-month scope by excluding prefix-30 donors before any donor floor or
downsampling.

## §2a Prereg correction (downsampling)

PREREG §2a corrected the original downsampling direction: the larger age group
is downsampled to the smaller with seed 0, after applying the bookkeeping floor
and before the unchanged ≥200-cells-per-age eligibility gate. Its original
per-`(donor, cell_type)` pairing unit cannot operate in TMS because each donor
has one age; PREREG §2b supersedes only that unit.

## §2b Prereg amendment: age labels and downsampling unit

The executed downsampling unit is per cell type. After the age exclusions and
`(donor, cell_type) < 20` bookkeeping floor, the larger group is sampled to the
smaller group with seed 0. Sampling uses a stable CRC32 per tissue/cell-type
stream and water-fills across donors, so donors contribute as evenly as their
available counts allow. The committed `*_after.csv` tables record each
contribution; `*_cell_filter.csv` records every cell-level filter reason.

## §2c Prereg amendment: old = 18–24 months; HVG span

PREREG §2c was executed in both tissues: young donors have prefix `3`; old
donors have prefixes `18`, `21`, or `24`; `4 weeks` and prefix-`30` donors are
excluded before the donor floor. For both tissues the required pooled HVG call
was `scanpy.pp.highly_variable_genes(flavor="seurat_v3", n_top_genes=2000,
batch_key="donor_id", span=1.0)`.

| Tissue | Balanced cells before ≥200 gate (young / old) | Eligible types | Final eligible cells (young / old) | Processed AnnData |
|---|---:|---:|---:|---|
| Lung | 1,687 / 1,687 | 2 | 648 / 648 | 1,296 × 2,000 HVGs; `.raw` = 1,296 × 52,437; counts layer retained |
| Marrow | 3,443 / 3,443 | 6 | 2,694 / 2,694 | 5,388 × 2,000 HVGs; `.raw` = 5,388 × 52,437; counts layer retained |

The A01 feasibility gate passes for both tissues: lung has 2 eligible types
(`fibroblast of lung`, `natural killer cell`), and marrow has 6 (`granulocyte`,
`granulocytopoietic cell`, `hematopoietic precursor cell`, `monocyte`,
`precursor B cell`, `promonocyte`). No model was fitted in A01.
## §3 Mapping path
A02 fitted the required 15-dimensional, two-layer negative-binomial scVI models
on CPU with `batch_key="donor_id"`, seed 0, `max_epochs=400`, and early stopping
(patience 30). Each tissue used a young-only reference, old query data mapped with
`scvi.model.SCVI.load_query_data`, and a standalone-old control. Query fitting
used the scArches online-update procedure with `weight_decay=0.0`; all reference
encoder/decoder expression parameters that procedure freezes were checked after
query training. The maximum absolute change was exactly 0 for both tissues.

| Tissue | Fit | Epochs run | Best validation ELBO | Wall time (s) | Mapping path |
|---|---|---:|---:|---:|---|
| Lung | young reference | 45 | 1052.336 | 2.904 | young reference |
| Lung | old query/shared | 167 | 939.062 | 8.716 | `scvi.model.SCVI.load_query_data` |
| Lung | standalone old | 43 | 990.037 | 2.938 | standalone old |
| Marrow | young reference | 364 | 945.792 | 91.375 | young reference |
| Marrow | old query/shared | 130 | 922.048 | 25.011 | `scvi.model.SCVI.load_query_data` |
| Marrow | standalone old | 264 | 904.736 | 64.046 | standalone old |

The local, gitignored `benchmarks/aging/{lung,marrow}_shared.h5ad` objects retain
all balanced cells and three embeddings: `X_scVI_young` (young cells),
`X_scVI_old` (old cells), and `X_scVI_shared` (both ages in the young-reference
space). The frozen-reference audits covered 1,070,606 lung and 1,070,094 marrow
parameters; `{lung,marrow}_ref_weights_max_abs_change = 0.0`.

## §4 Per-type tables (shared and standalone)
The §4b isotropy and §5 anticipation tests were run once in the frozen scArches shared space (`X_scVI_shared`; 30-cell centroid operators, 1,000 split-half null draws, seed 0). Standalone embeddings are not cross-age coordinate systems, so their table is limited to axis-free per-age quantities. Every `substate_alignment` is `NaN`: the Census input has no annotation below `cell_type`, and no post-hoc substate clustering was performed.

| Tissue | Cell type | n young / old | S | iso q95 | Iso | |cos(u,d)| | ant q95 | Ant | Shared holder young / old | Correction artifact | Aging mode |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|---|
| lung | fibroblast of lung | 263 / 263 | 0.017 | 0.028 | fail | 0.119 | 0.918 | fail | 0.913 / 0.977 | no | mixed |
| lung | natural killer cell | 385 / 385 | 0.028 | 0.040 | fail | 0.929 | 0.956 | fail | 0.914 / 0.901 | no | mixed |
| marrow | granulocyte | 704 / 704 | 0.080 | 0.121 | fail | 0.714 | 0.732 | fail | 0.686 / 0.803 | no | mixed |
| marrow | granulocytopoietic cell | 727 / 727 | 1.666 | 1.233 | pass | 0.587 | 0.616 | fail | 0.195 / 0.436 | no | mixed |
| marrow | hematopoietic precursor cell | 285 / 285 | 2.498 | 1.980 | pass | 0.200 | 0.432 | fail | 0.137 / 0.133 | no | mixed |
| marrow | monocyte | 488 / 488 | 2.242 | 2.952 | fail | 0.089 | 0.353 | fail | 0.061 / 0.051 | no | mixed |
| marrow | precursor B cell | 259 / 259 | 1.023 | 0.743 | pass | 0.108 | 0.362 | fail | 0.220 / 0.108 | no | mixed |
| marrow | promonocyte | 231 / 231 | 3.177 | 3.452 | fail | 0.470 | 0.463 | pass | 0.082 / 0.017 | no | mixed |

### Standalone axis-free summaries
The full sorted centroid spectra are `;`-joined in the committed standalone CSVs; the table below gives their scalar summaries alongside the locked holder fraction.

| Tissue | Cell type | Age | n | lead eig | median |λ| | anisotropy | Holder fraction | Correction artifact |
|---|---|---|---:|---:|---:|---:|---:|---|
| lung | fibroblast of lung | old | 263 | -5.068 | 204.901 | 40.428 | 0.901 | no |
| lung | fibroblast of lung | young | 263 | -7.474 | 303.942 | 40.669 | 0.970 | no |
| lung | natural killer cell | old | 385 | -8.214 | 272.741 | 33.206 | 0.935 | no |
| lung | natural killer cell | young | 385 | -12.841 | 309.113 | 24.072 | 0.909 | no |
| marrow | granulocyte | old | 704 | -1.354 | 4.808 | 3.551 | 0.751 | no |
| marrow | granulocyte | young | 704 | -1.367 | 4.236 | 3.099 | 0.705 | no |
| marrow | granulocytopoietic cell | old | 727 | -0.473 | 3.678 | 7.777 | 0.107 | no |
| marrow | granulocytopoietic cell | young | 727 | 0.441 | 1.910 | 4.333 | 0.165 | no |
| marrow | hematopoietic precursor cell | old | 285 | 0.965 | 1.621 | 1.681 | 0.042 | no |
| marrow | hematopoietic precursor cell | young | 285 | -0.013 | 1.881 | 139.982 | 0.133 | no |
| marrow | monocyte | old | 488 | 0.041 | 3.142 | 76.258 | 0.064 | no |
| marrow | monocyte | young | 488 | -0.279 | 2.290 | 8.205 | 0.094 | no |
| marrow | precursor B cell | old | 259 | 0.712 | 1.991 | 2.797 | 0.015 | no |
| marrow | precursor B cell | young | 259 | 0.066 | 1.914 | 28.812 | 0.046 | no |
| marrow | promonocyte | old | 231 | -0.460 | 2.140 | 4.651 | 0.342 | no |
| marrow | promonocyte | young | 231 | -0.091 | 1.767 | 19.450 | 0.056 | no |

| Scope | Non-artifact eligible types | Isotropy pass fraction | Anticipation pass fraction |
|---|---:|---:|---:|
| lung | 2 | 0.000 | 0.000 |
| marrow | 6 | 0.500 | 0.167 |
| pooled | 8 | 0.375 | 0.125 |

Correction-artifact count: **0 of 8** eligible types. No type crossed the locked 0.5 holder-fraction threshold between shared and its same-age standalone fit.

The descriptive aging-mode clustering selected `k=2` by silhouette across the eight operator-coordinate vectors. One mode contained granulocytopoietic cell; the other contained the remaining seven types. Both mode centroids were labelled `mixed` by the pre-specified descriptive rules. These labels are not outcome gates.

## §4b Isotropy amendment
This amendment was made before any aging data run. For each type, the statistic was `S = max_k |log(|λ_old,k| / |λ_young,k|) − median(log(|λ_old,k| / |λ_young,k|))|` over cosine-matched centroid-operator directions. Its null used 1,000 random split-halves within young cells, with halves of `min(n_old, floor(n_young/2))` cells. A type passed only when `S > iso_null_q95`; `rho` and `cv_ratio` in the compare CSVs are provenance quantities, not gates. This rejects uniform softening, which preserves eigenvalue rank ordering under the retired §4a statistic.

## §5 Interpretation
**isotropic noise increase; no aging claim beyond stiffness**

Pooled non-artifact majorities were assessed over 8 types: isotropy 0.375 and anticipation 0.125. Neither exceeds the strict `> 0.5` majority rule. Per tissue, lung was 0/2 for both tests; marrow was 3/6 for isotropy and 1/6 for anticipation. The header is the frozen §6 row selected from those fractions and does not reinterpret individual type-level results.

## §6 Figure inventory
Every rendered figure is a 200-dpi PDF/PNG pair under `benchmarks/figures/aging/`; the source-file mapping is the committed `MANIFEST.csv`. The table lists all 78 pairs by path.

| Spec item | Tissue | Age | Paths |
|---|---|---|---|
| §7.5 aging_vectors | both | young_old | `benchmarks/figures/aging/comparison/aging_vectors.pdf`; `benchmarks/figures/aging/comparison/aging_vectors.png` |
| §7.1 spectrum_ladder groups | lung | young_old | `benchmarks/figures/aging/lung/spectrum_ladder_young_old.pdf`; `benchmarks/figures/aging/lung/spectrum_ladder_young_old.png` |
| §7.2 soft_axis_map displacement | lung | young_old | `benchmarks/figures/aging/lung/soft_axis_map_young_old.pdf`; `benchmarks/figures/aging/lung/soft_axis_map_young_old.png` |
| §7.3 basin_index_by_type | lung | young_old | `benchmarks/figures/aging/lung/basin_index_age_facet.pdf`; `benchmarks/figures/aging/lung/basin_index_age_facet.png` |
| §7.4 ledger_map | lung | young | `benchmarks/figures/aging/lung/young/ledger_map.pdf`; `benchmarks/figures/aging/lung/young/ledger_map.png` |
| §7b figure_operator_space | lung | young | `benchmarks/figures/aging/lung/young/operator_space.pdf`; `benchmarks/figures/aging/lung/young/operator_space.png` |
| §7b figure_ledger_summary | lung | young | `benchmarks/figures/aging/lung/young/ledger_summary.pdf`; `benchmarks/figures/aging/lung/young/ledger_summary.png` |
| §7b spectrum_ladder single group | lung | young | `benchmarks/figures/aging/lung/young/spectrum_ladder.pdf`; `benchmarks/figures/aging/lung/young/spectrum_ladder.png` |
| §7b soft_axis_map | lung | young | `benchmarks/figures/aging/lung/young/soft_axis_map.pdf`; `benchmarks/figures/aging/lung/young/soft_axis_map.png` |
| §7b basin_index_by_type single group | lung | young | `benchmarks/figures/aging/lung/young/basin_index.pdf`; `benchmarks/figures/aging/lung/young/basin_index.png` |
| §7b Gaussian-oracle curvature ratio | lung | young | `benchmarks/figures/aging/lung/young/gaussian_oracle_curvature_ratio.pdf`; `benchmarks/figures/aging/lung/young/gaussian_oracle_curvature_ratio.png` |
| §7b scVI ELBO QC | lung | young | `benchmarks/figures/aging/lung/young/qc_scvi_elbo_summary.pdf`; `benchmarks/figures/aging/lung/young/qc_scvi_elbo_summary.png` |
| §7b d_latent choice | lung | young | `benchmarks/figures/aging/lung/young/qc_d_latent.pdf`; `benchmarks/figures/aging/lung/young/qc_d_latent.png` |
| §7b within-cluster σ versus drift scale | lung | young | `benchmarks/figures/aging/lung/young/qc_sigma.pdf`; `benchmarks/figures/aging/lung/young/qc_sigma.png` |
| §7b symmetry check histogram | lung | young | `benchmarks/figures/aging/lung/young/qc_symmetry.pdf`; `benchmarks/figures/aging/lung/young/qc_symmetry.png` |
| §7b tangent-rank distribution | lung | young | `benchmarks/figures/aging/lung/young/qc_rank.pdf`; `benchmarks/figures/aging/lung/young/qc_rank.png` |
| §7b inter-cluster gap control | lung | young | `benchmarks/figures/aging/lung/young/qc_gap.pdf`; `benchmarks/figures/aging/lung/young/qc_gap.png` |
| §7b donor × type counts | lung | young | `benchmarks/figures/aging/lung/young/qc_donor_type_counts.pdf`; `benchmarks/figures/aging/lung/young/qc_donor_type_counts.png` |
| §7b standalone-versus-shared basin agreement | lung | young | `benchmarks/figures/aging/lung/young/qc_shared_standalone_basin_confusion.pdf`; `benchmarks/figures/aging/lung/young/qc_shared_standalone_basin_confusion.png` |
| §7b correction_artifact list | lung | young | `benchmarks/figures/aging/lung/young/qc_correction_artifacts.pdf`; `benchmarks/figures/aging/lung/young/qc_correction_artifacts.png` |
| §7.4 ledger_map | lung | old | `benchmarks/figures/aging/lung/old/ledger_map.pdf`; `benchmarks/figures/aging/lung/old/ledger_map.png` |
| §7b figure_operator_space | lung | old | `benchmarks/figures/aging/lung/old/operator_space.pdf`; `benchmarks/figures/aging/lung/old/operator_space.png` |
| §7b figure_ledger_summary | lung | old | `benchmarks/figures/aging/lung/old/ledger_summary.pdf`; `benchmarks/figures/aging/lung/old/ledger_summary.png` |
| §7b spectrum_ladder single group | lung | old | `benchmarks/figures/aging/lung/old/spectrum_ladder.pdf`; `benchmarks/figures/aging/lung/old/spectrum_ladder.png` |
| §7b soft_axis_map | lung | old | `benchmarks/figures/aging/lung/old/soft_axis_map.pdf`; `benchmarks/figures/aging/lung/old/soft_axis_map.png` |
| §7b basin_index_by_type single group | lung | old | `benchmarks/figures/aging/lung/old/basin_index.pdf`; `benchmarks/figures/aging/lung/old/basin_index.png` |
| §7b Gaussian-oracle curvature ratio | lung | old | `benchmarks/figures/aging/lung/old/gaussian_oracle_curvature_ratio.pdf`; `benchmarks/figures/aging/lung/old/gaussian_oracle_curvature_ratio.png` |
| §7b scVI ELBO QC | lung | old | `benchmarks/figures/aging/lung/old/qc_scvi_elbo_summary.pdf`; `benchmarks/figures/aging/lung/old/qc_scvi_elbo_summary.png` |
| §7b d_latent choice | lung | old | `benchmarks/figures/aging/lung/old/qc_d_latent.pdf`; `benchmarks/figures/aging/lung/old/qc_d_latent.png` |
| §7b within-cluster σ versus drift scale | lung | old | `benchmarks/figures/aging/lung/old/qc_sigma.pdf`; `benchmarks/figures/aging/lung/old/qc_sigma.png` |
| §7b symmetry check histogram | lung | old | `benchmarks/figures/aging/lung/old/qc_symmetry.pdf`; `benchmarks/figures/aging/lung/old/qc_symmetry.png` |
| §7b tangent-rank distribution | lung | old | `benchmarks/figures/aging/lung/old/qc_rank.pdf`; `benchmarks/figures/aging/lung/old/qc_rank.png` |
| §7b inter-cluster gap control | lung | old | `benchmarks/figures/aging/lung/old/qc_gap.pdf`; `benchmarks/figures/aging/lung/old/qc_gap.png` |
| §7b donor × type counts | lung | old | `benchmarks/figures/aging/lung/old/qc_donor_type_counts.pdf`; `benchmarks/figures/aging/lung/old/qc_donor_type_counts.png` |
| §7b standalone-versus-shared basin agreement | lung | old | `benchmarks/figures/aging/lung/old/qc_shared_standalone_basin_confusion.pdf`; `benchmarks/figures/aging/lung/old/qc_shared_standalone_basin_confusion.png` |
| §7b correction_artifact list | lung | old | `benchmarks/figures/aging/lung/old/qc_correction_artifacts.pdf`; `benchmarks/figures/aging/lung/old/qc_correction_artifacts.png` |
| §7b compare_summary | lung | young_old | `benchmarks/figures/aging/lung/compare_summary.pdf`; `benchmarks/figures/aging/lung/compare_summary.png` |
| §7b standalone-versus-shared quantities | lung | young_old | `benchmarks/figures/aging/lung/standalone_shared_quantities.pdf`; `benchmarks/figures/aging/lung/standalone_shared_quantities.png` |
| §7.6 energy_transect | lung | young_old | `benchmarks/figures/aging/lung/energy_transect_fibroblast_of_lung.pdf`; `benchmarks/figures/aging/lung/energy_transect_fibroblast_of_lung.png` |
| §7.6 energy_transect | lung | young_old | `benchmarks/figures/aging/lung/energy_transect_natural_killer_cell.pdf`; `benchmarks/figures/aging/lung/energy_transect_natural_killer_cell.png` |
| §7.1 spectrum_ladder groups | marrow | young_old | `benchmarks/figures/aging/marrow/spectrum_ladder_young_old.pdf`; `benchmarks/figures/aging/marrow/spectrum_ladder_young_old.png` |
| §7.2 soft_axis_map displacement | marrow | young_old | `benchmarks/figures/aging/marrow/soft_axis_map_young_old.pdf`; `benchmarks/figures/aging/marrow/soft_axis_map_young_old.png` |
| §7.3 basin_index_by_type | marrow | young_old | `benchmarks/figures/aging/marrow/basin_index_age_facet.pdf`; `benchmarks/figures/aging/marrow/basin_index_age_facet.png` |
| §7.4 ledger_map | marrow | young | `benchmarks/figures/aging/marrow/young/ledger_map.pdf`; `benchmarks/figures/aging/marrow/young/ledger_map.png` |
| §7b figure_operator_space | marrow | young | `benchmarks/figures/aging/marrow/young/operator_space.pdf`; `benchmarks/figures/aging/marrow/young/operator_space.png` |
| §7b figure_ledger_summary | marrow | young | `benchmarks/figures/aging/marrow/young/ledger_summary.pdf`; `benchmarks/figures/aging/marrow/young/ledger_summary.png` |
| §7b spectrum_ladder single group | marrow | young | `benchmarks/figures/aging/marrow/young/spectrum_ladder.pdf`; `benchmarks/figures/aging/marrow/young/spectrum_ladder.png` |
| §7b soft_axis_map | marrow | young | `benchmarks/figures/aging/marrow/young/soft_axis_map.pdf`; `benchmarks/figures/aging/marrow/young/soft_axis_map.png` |
| §7b basin_index_by_type single group | marrow | young | `benchmarks/figures/aging/marrow/young/basin_index.pdf`; `benchmarks/figures/aging/marrow/young/basin_index.png` |
| §7b Gaussian-oracle curvature ratio | marrow | young | `benchmarks/figures/aging/marrow/young/gaussian_oracle_curvature_ratio.pdf`; `benchmarks/figures/aging/marrow/young/gaussian_oracle_curvature_ratio.png` |
| §7b scVI ELBO QC | marrow | young | `benchmarks/figures/aging/marrow/young/qc_scvi_elbo_summary.pdf`; `benchmarks/figures/aging/marrow/young/qc_scvi_elbo_summary.png` |
| §7b d_latent choice | marrow | young | `benchmarks/figures/aging/marrow/young/qc_d_latent.pdf`; `benchmarks/figures/aging/marrow/young/qc_d_latent.png` |
| §7b within-cluster σ versus drift scale | marrow | young | `benchmarks/figures/aging/marrow/young/qc_sigma.pdf`; `benchmarks/figures/aging/marrow/young/qc_sigma.png` |
| §7b symmetry check histogram | marrow | young | `benchmarks/figures/aging/marrow/young/qc_symmetry.pdf`; `benchmarks/figures/aging/marrow/young/qc_symmetry.png` |
| §7b tangent-rank distribution | marrow | young | `benchmarks/figures/aging/marrow/young/qc_rank.pdf`; `benchmarks/figures/aging/marrow/young/qc_rank.png` |
| §7b inter-cluster gap control | marrow | young | `benchmarks/figures/aging/marrow/young/qc_gap.pdf`; `benchmarks/figures/aging/marrow/young/qc_gap.png` |
| §7b donor × type counts | marrow | young | `benchmarks/figures/aging/marrow/young/qc_donor_type_counts.pdf`; `benchmarks/figures/aging/marrow/young/qc_donor_type_counts.png` |
| §7b standalone-versus-shared basin agreement | marrow | young | `benchmarks/figures/aging/marrow/young/qc_shared_standalone_basin_confusion.pdf`; `benchmarks/figures/aging/marrow/young/qc_shared_standalone_basin_confusion.png` |
| §7b correction_artifact list | marrow | young | `benchmarks/figures/aging/marrow/young/qc_correction_artifacts.pdf`; `benchmarks/figures/aging/marrow/young/qc_correction_artifacts.png` |
| §7.4 ledger_map | marrow | old | `benchmarks/figures/aging/marrow/old/ledger_map.pdf`; `benchmarks/figures/aging/marrow/old/ledger_map.png` |
| §7b figure_operator_space | marrow | old | `benchmarks/figures/aging/marrow/old/operator_space.pdf`; `benchmarks/figures/aging/marrow/old/operator_space.png` |
| §7b figure_ledger_summary | marrow | old | `benchmarks/figures/aging/marrow/old/ledger_summary.pdf`; `benchmarks/figures/aging/marrow/old/ledger_summary.png` |
| §7b spectrum_ladder single group | marrow | old | `benchmarks/figures/aging/marrow/old/spectrum_ladder.pdf`; `benchmarks/figures/aging/marrow/old/spectrum_ladder.png` |
| §7b soft_axis_map | marrow | old | `benchmarks/figures/aging/marrow/old/soft_axis_map.pdf`; `benchmarks/figures/aging/marrow/old/soft_axis_map.png` |
| §7b basin_index_by_type single group | marrow | old | `benchmarks/figures/aging/marrow/old/basin_index.pdf`; `benchmarks/figures/aging/marrow/old/basin_index.png` |
| §7b Gaussian-oracle curvature ratio | marrow | old | `benchmarks/figures/aging/marrow/old/gaussian_oracle_curvature_ratio.pdf`; `benchmarks/figures/aging/marrow/old/gaussian_oracle_curvature_ratio.png` |
| §7b scVI ELBO QC | marrow | old | `benchmarks/figures/aging/marrow/old/qc_scvi_elbo_summary.pdf`; `benchmarks/figures/aging/marrow/old/qc_scvi_elbo_summary.png` |
| §7b d_latent choice | marrow | old | `benchmarks/figures/aging/marrow/old/qc_d_latent.pdf`; `benchmarks/figures/aging/marrow/old/qc_d_latent.png` |
| §7b within-cluster σ versus drift scale | marrow | old | `benchmarks/figures/aging/marrow/old/qc_sigma.pdf`; `benchmarks/figures/aging/marrow/old/qc_sigma.png` |
| §7b symmetry check histogram | marrow | old | `benchmarks/figures/aging/marrow/old/qc_symmetry.pdf`; `benchmarks/figures/aging/marrow/old/qc_symmetry.png` |
| §7b tangent-rank distribution | marrow | old | `benchmarks/figures/aging/marrow/old/qc_rank.pdf`; `benchmarks/figures/aging/marrow/old/qc_rank.png` |
| §7b inter-cluster gap control | marrow | old | `benchmarks/figures/aging/marrow/old/qc_gap.pdf`; `benchmarks/figures/aging/marrow/old/qc_gap.png` |
| §7b donor × type counts | marrow | old | `benchmarks/figures/aging/marrow/old/qc_donor_type_counts.pdf`; `benchmarks/figures/aging/marrow/old/qc_donor_type_counts.png` |
| §7b standalone-versus-shared basin agreement | marrow | old | `benchmarks/figures/aging/marrow/old/qc_shared_standalone_basin_confusion.pdf`; `benchmarks/figures/aging/marrow/old/qc_shared_standalone_basin_confusion.png` |
| §7b correction_artifact list | marrow | old | `benchmarks/figures/aging/marrow/old/qc_correction_artifacts.pdf`; `benchmarks/figures/aging/marrow/old/qc_correction_artifacts.png` |
| §7b compare_summary | marrow | young_old | `benchmarks/figures/aging/marrow/compare_summary.pdf`; `benchmarks/figures/aging/marrow/compare_summary.png` |
| §7b standalone-versus-shared quantities | marrow | young_old | `benchmarks/figures/aging/marrow/standalone_shared_quantities.pdf`; `benchmarks/figures/aging/marrow/standalone_shared_quantities.png` |
| §7.6 energy_transect | marrow | young_old | `benchmarks/figures/aging/marrow/energy_transect_granulocyte.pdf`; `benchmarks/figures/aging/marrow/energy_transect_granulocyte.png` |

The three instances that cannot be rendered without a prohibited refit or an unavailable stored artifact are recorded in `benchmarks/figures/aging/DEVIATIONS.md` and as manifest rows: the archetype sheet, ten-refit permutation null, and per-epoch scVI ELBO curves. Stored final-ELBO summaries are rendered in place of the unavailable curves.

## §7 Limits
* The §6 header is the frozen label for "both tests fail in the
  majority". Stiffness did not decrease (median |λ| old/young 0.997–1.14
  in seven types, 3.1 in granulocytopoietic cells), so this result
  reports no detectable direction-specific, anticipatory or uniform
  loss of operator stiffness with age, not a measured noise increase.
* Only 3 of the 8 eligible types are basins under the locked rule (lung
  fibroblast, lung NK, marrow granulocyte); all three are unchanged.
  The isotropy passes are in marrow transit states.
* scArches mapping into a young reference may shrink old–young
  differences; lung rests on 2 types and 648 reference cells; marrow on
  2 young donors; one scVI seed.

## §8 Reproducibility
* Input integrity: `scripts/render_aging_figures.py` verifies every A01–A03 report-listed SHA-256 before reading an artifact; its verification list is `benchmarks/figures/aging/input_checksums.json`.
* Figure generation: `. .venv/bin/activate && python scripts/render_aging_figures.py`. The renderer fits no model and uses only stored H5AD, CSV, and drift-checkpoint artifacts.
* Executed notebook: `JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python scripts/make_aging_notebook.py`, after installing the local `.venv` kernel named `scjdo2-a04`.
* Environment: Python 3.13.2; anndata 0.11.4, scanpy 1.12.4, scvi-tools 1.5.1, scarches 0.6.1, dynamo-release 1.5.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pandas 2.3.3, torch 2.14.1, nbformat 5.11.1, nbclient 0.11.0.
* The executed notebook is `notebooks/aging_basins.ipynb`; it contains 95 cells, 8 executed code cells, no execution errors, and 78 inline figure-atlas Markdown cells.
