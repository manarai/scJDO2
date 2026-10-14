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
