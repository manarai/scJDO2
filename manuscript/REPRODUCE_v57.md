# Reproducing the v57 figures and number reports

## Environment (private to this worktree)

```bash
cd <repo root>                      # branch figures-v57
python3 -m venv .venv-fig           # Python 3.13 was used
.venv-fig/bin/pip install matplotlib pandas numpy scipy nbformat nbclient ipykernel
```

`.venv-fig/` is git-ignored. Nothing is registered globally: `run_all.py` puts the venv's `bin/` first on `PATH`,
so nbclient starts the venv's own `python3` kernel.

## Run

```bash
.venv-fig/bin/python notebooks/run_all.py              # all notebooks, in order, executed in place
.venv-fig/bin/python notebooks/run_all.py fig3_flow_operator   # a subset (does not reset provenance.json)
```

A full run deletes and rebuilds `manuscript/figures/provenance.json`, executes (per-cell timeout 600 s, first error aborts with a
non-zero exit and leaves the failing notebook's outputs in place):

1. `notebooks/fig1_framework.ipynb`
2. `notebooks/fig2_benchmark.ipynb`
3. `notebooks/fig3_flow_operator.ipynb`
4. `notebooks/fig4_dyngen.ipynb`
5. `notebooks/fig5_real_scale.ipynb`
6. `notebooks/fig6_score_operator.ipynb`
7. `notebooks/fig7_archetypes.ipynb`
8. `notebooks/supp_s10_diffusion.ipynb`
9. `notebooks/supp_s21_three_timepoints.ipynb`
10. `notebooks/supp_s13_whitening.ipynb`
11. `notebooks/supp_s11_operator_agreement.ipynb`
12. `notebooks/tables_v57.ipynb`
13. `notebooks/numbers_v57.ipynb`
14. `notebooks/build_reports.ipynb`

and writes `manuscript/figures/run_log.json` (wall time per notebook).

Outputs: `manuscript/figures/main/Fig1–7.{pdf,png,tiff}`, `manuscript/figures/supplementary/FigS1–S4.{pdf,png,tiff}`,
`manuscript/tables/TableS2_S22.{csv,md}` and `manuscript/tables/TableS3_S18d.{csv,md}` (Supplementary Tables S2, S3),
`manuscript/figures/provenance.json`, `manuscript/figures/values/*.json` (every number used in a legend), `manuscript/figures/README.md`,
`manuscript/FIGURE_LEGENDS_v57.md`, `manuscript/NUMBERS_v57.md`, `manuscript/figures/numbers_v57.csv`, this file.
Shared style and provenance helpers: `notebooks/_style.py`.

## Formats and resolution

- Set centrally in `notebooks/_style.py` (`DPI = 600`, `FORMATS = ("pdf", "png", "tiff")`, `save()`): vector PDF with TrueType-embedded fonts
  (`pdf.fonttype = 42`), PNG at 600 dpi, TIFF at 600 dpi (LZW compression, RGB). Widths: single column 89 mm, 1.5 column 120 mm,
  double column 183 mm, maximum height 240 mm (`save()` prints the final size and warns above 183 × 240 mm).
- At save time every text is clamped to ≥ 6 pt and every non-zero line width to ≥ 0.5 pt; bold panel letters are placed top-left, outside the
  axes, tick labels and title of their panel and aligned per row. Point labels that could collide use `repel_labels()`.

## Inputs

All inputs of the v57 figures (Fig1–7, Supplementary Figs S1–S4), of Supplementary Tables S2/S3 and of NUMBERS_v57 are committed result files under `benchmarks/` (S15 at
`benchmarks/synthetic_multimodal/s15/`, the single variable `S15_DIR` in `notebooks/_style.py`, overridable with `SCJDO2_S15_DIR`).
The scJDO v1 (v56) figures are not part of v57 (they remain in version 56 of the preprint); no v56 file is read.

## Runtime

<!-- runtime:start -->
(filled in by run_all.py after a full run)
<!-- runtime:end -->

## Pending (regenerate after each item lands)

- S22 (`benchmarks/real_flow/s22/`) and S18d (`benchmarks/perturb_dose/`) are merged: Fig6d, Supplementary Tables S2/S3 and their
  NUMBERS_v57 rows are computed from them (no longer pending).
- **S14c** (confirmatory transition test) ended as a construction failure (2 of 4 required seeds eligible; no model fitted); Fig3e shows it.
- **S16** (head-to-head with CellRank / spliceJAC / dynamo): no figure (run but not scored; not reported in the manuscript).
- dyngen s18 inversion diagnosis: pending (noted in the Fig4 legend).
