# Aging basins — A01 results

## §1 Data

The unmodified `scripts/fetch_tms_aging.py` fetched Tabula Muris Senis from
cellxgene Census `2024-07-01`. Both observed dataset IDs match the frozen
PREREG §1 identifiers. The raw matrices are retained locally as gitignored
AnnData files and are listed in the A01 report with checksums.

| Tissue | Dataset ID | Raw cells | Raw genes | `4 weeks` excluded (donors) | Young: 3 months (cells / donors) | Old: 18 month + 20 months and over (cells / donors) |
|---|---|---:|---:|---:|---:|---:|
| Lung | `0fb7916e-7a68-4a4c-a441-3ab3989f29a7` | 24,540 | 52,437 | 2,508 / 2 | 3,612 / 4 | 18,420 / 10 |
| Marrow | `0bd1a1de-3aee-40e0-b2ec-86c7a30c7149` | 40,220 | 52,437 | 3,027 / 2 | 3,490 / 2 | 33,703 / 13 |

`4 weeks` cells are excluded by design under PREREG §2b. No other age label
outside the revised young/old sets was observed. The `20 month-old stage and
over` label is used as supplied by the Census and may include mice older than
24 months; this is recorded rather than corrected.

## §2a Prereg correction (downsampling)

PREREG §2a corrected the original direction: the larger group is downsampled
to the smaller with seed 0, after applying a bookkeeping floor of 20 cells and
before the unchanged ≥200-cells-per-age inclusion gate. Its original
per-`(donor, cell_type)` pairing unit cannot operate in TMS because each donor
has one age; PREREG §2b supersedes only that unit.

## §2b Prereg amendment: age labels and downsampling unit

The revised preprocessing implementation (`scripts/preprocess_tms_aging.py`)
executed the frozen §2b procedure: exclude `4 weeks`, retain young = 3 months
and old = 18 months plus 20 months and over, remove `(donor, cell_type)` sets
with fewer than 20 cells, then per cell type sample the larger age group to the
smaller group's count. Sampling uses seed 0 with a stable CRC32 per
tissue/cell-type stream and water-fills across donors, so available donors
contribute as evenly as their capacities allow. The committed `*_after.csv`
files report every donor/type contribution; cell-level reasons are in the
committed `*_cell_filter.csv` files.

| Tissue | Balanced cells before ≥200 gate (young / old) | Eligible types | Eligible selection (young / old) | HVG result |
|---|---:|---:|---:|---|
| Lung | 1,815 / 1,815 | 2 | 776 / 776 | **Failed** during the required `seurat_v3` HVG call; no processed AnnData was written. |
| Marrow | 3,443 / 3,443 | 6 | 2,694 / 2,694 | **Completed:** 5,388 × 2,000 processed AnnData; counts layer and 52,437-gene `.raw` retained. |

The ≥200 feasibility gate passes in both tissues (lung: 2 eligible types;
marrow: 6). However, lung preprocessing is incomplete: the exact required
`scanpy.pp.highly_variable_genes(flavor="seurat_v3", n_top_genes=2000,
batch_key="donor_id")` call raised `_loess.loess.fit` `ValueError: There are
other near singularities as well. 0.031008`. No alternate HVG flavor,
parameter, or data alteration was substituted, and A01 remains blocked pending
a revised instruction.
