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
