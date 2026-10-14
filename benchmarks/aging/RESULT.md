# Aging basins — A01 results

## §1 Data

The unmodified `scripts/fetch_tms_aging.py` fetched Tabula Muris Senis from
cellxgene Census `2024-07-01`. Both observed dataset IDs match the frozen
PREREG §1 identifiers.

| Tissue | Dataset ID | Raw cells | Raw genes | Young cells / donors | Old cells / donors | `4 weeks` cells left unmapped |
|---|---|---:|---:|---:|---:|---:|
| Lung | `0fb7916e-7a68-4a4c-a441-3ab3989f29a7` | 24,540 | 52,437 | 3,612 / 4 | 18,420 / 10 | 2,508 (10.22%) |
| Marrow | `0bd1a1de-3aee-40e0-b2ec-86c7a30c7149` | 40,220 | 52,437 | 3,490 / 2 | 33,703 / 13 | 3,027 (7.53%) |

The script assigns `age_group = young` for its 3-month labels and `old` for
its 18–24-month labels. All cells with `development_stage == "4 weeks"` remain
unmapped (`age_group` missing), as intended by the method specification's
instruction to drop 4-week cells. No other unmapped development-stage label was
observed.

A01 stopped after this fetch. Its task-level stop rule requires stopping when
any label outside the script's young/old sets is at least 5% of cells; `4 weeks`
exceeds that threshold in both tissues. Therefore no downsampling, HVG
annotation, or fitting was performed.

## §2a Prereg correction (downsampling)

The following frozen amendment would govern preprocessing, but was **not
executed** because A01 stopped in §1: per `(donor, cell_type)`, downsample the
larger age group to the size of the smaller group with seed 0; drop
`(donor, cell_type)` cells with fewer than 20 cells on either side (a
bookkeeping floor, not a gate); then retain only cell types with at least 200
cells in both age groups. The 200-cell inclusion gate is unchanged.
