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
