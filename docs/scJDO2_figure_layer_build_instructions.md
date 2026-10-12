# scJDO2 — Build instructions: analysis + figure API (Task 41)

**Repo:** `/Users/terooatt/Documents/scJDO2`
**Authoritative spec:** `docs/opspace_figure_spec.md` (the merged file). This document does not replace it; it is the build order, the function-by-function contract, and the additions agreed since the spec was merged. Where this file and the spec disagree, this file wins, and the disagreement is recorded in `V2_PLAN.md`.

**Precondition:** the rank-aware spectrum (tangent projector `P_c`, `off_tangent` ledger entries) is in and the Phase C pitchfork gate table has been produced. Do not start the figure layer before that.

Build in the order below. One commit per numbered section. Every function gets a docstring, a type signature, and a known-answer test on the equilibrated pitchfork.

---

## 0. Additions to the spec (new since merge)

1. **Tangent restriction everywhere.** All spectral quantities (eigenvalues, eigenvectors, Henrici, transient gain, drivers) are computed on `P_c J_c P_cᵀ` and lifted back to gene space through `P_c`. Off-tangent directions never appear as eigenvalues; genes whose weight in the lifted eigenvector comes entirely from off-tangent directions are reported with reason `off_tangent`.
2. **Sensitive-holder mark.** A cell is `sensitive` when `henrici` is above the dataset's 90th percentile (stored as a boolean in `obs["op_sensitive"]`). It is drawn as a dashed outer ring (radius 2× the marker, 0.75 pt, dash 3-2, same colour as the behaviour class) in Panels A and B. The eigenvalue colouring and behaviour class are unchanged.
3. **Complex eigenvector rendering.** For a complex leading mode, Panel C and the archetype sheet draw one-sided bars: length = |v_g|, colour = arg(v_g) on a cyclic map (matplotlib `twilight`), with a small horizontal cyclic colour bar labelled "early in cycle → late in cycle". No sign axis. Phase is reported relative to the gene with the largest magnitude (which is fixed to phase 0 by the sign convention).
4. **Relative-units label.** Every axis or column that carries an eigenvalue says "relative" unless `uns["scjdo2"]["clock_calibrated"]` is `True`. Tested.

---

## 1. `scjdo2/opspace.py` — analysis API

All functions take an `AnnData` already carrying the Phase C outputs (per-cell CSR `J_c`, ledger, `P_c`, `r_c`, latent `z`, velocity `v`). All write into `adata` under the keys listed and return `adata`. A `compute(adata, smooth_k=0, n_eig=5, min_fisher=None)` wrapper runs 1.1–1.6 in order.

### 1.1 `spectrum(adata, n_eig=5, fisher_mask=True, min_fisher=None, smooth_k=0)`
- If `smooth_k > 0`, apply `field.smooth_jacobians` first and record `uns["scjdo2"]["smooth_k"]`.
- Mask entries with Fisher below `min_fisher` (default 10th percentile of that cell's live entries) to zero; record the threshold per cell.
- Restrict to the tangent subspace, eigendecompose, sort by real part descending, drop the negative-Im member of each conjugate pair, keep `n_eig`.
- Lift eigenvectors to gene space; fix phase so the largest-magnitude entry is real and positive.
- Store: `obsm["op_eigvals"]` complex (cells × n_eig, NaN-padded if `r_c < n_eig`); `uns["op_eigvecs"]` zarr-backed complex (cells × n_eig × genes), sparse by live columns; `obs["op_rank"] = r_c`.

### 1.2 `invariants(adata, tol=None, horizon=None)`
Writes to `obs`: `lead_re`, `lead_im`, `n_stable`, `n_unstable`, `n_near_zero`, `n_complex_pairs`, `spectral_gap`, `henrici`, `transient_gain`, `op_confidence`. Henrici and transient gain on the tangent-restricted block. `tol` default = 0.1 × SD of `lead_re`; `horizon` default = 5 / median |lead_re|. `op_confidence` = median Fisher over entries feeding the leading eigenvector with weight > 5 % of max.

### 1.3 `behaviour(adata, tol=None, sensitive_pct=90)`
Writes `obs["op_behaviour"]` ∈ {holder, fork, drift, damped_oscillator, growing_oscillator} and `obs["op_sensitive"]` (bool). Same `tol` as 1.2 so the ground-truth labels in tests can share it.

### 1.4 `drivers(adata, which="lead", top=15, min_fisher=None, by="cell" | "archetype")`
Returns a tidy DataFrame: `unit` (cell id or archetype id), `gene`, `weight` (signed real part; NaN for complex modes), `magnitude`, `phase` (radians relative to the largest entry; NaN for real modes), `fisher`, `fisher_se`, `reason` (`ok`, `no_counts`, `no_variation`, `not_in_ball`, `off_tangent`). Sorted by magnitude within unit, filtered by `min_fisher`, top `top` rows per unit. Genes with a non-`ok` reason are kept in the table only if they would be in the top set by magnitude, flagged, never silently dropped.

### 1.5 `archetype_operators(adata)`
Requires Step 8 weights in `obsm["archetype_weights"]`. For each archetype: Fisher-weighted, membership-weighted mean of `J_c` and of `P_c` (re-orthonormalised), then 1.1–1.4 on that operator. Stores `uns["archetypes"]` with: behaviour name, eigenvalues, Henrici, transient gain, n cells (weight > 0.5), mean confidence, driver table. If weights are absent, warn and skip; downstream figures must not fail.

### 1.6 `field_gradient(adata, k=15)` and `lineage_path(adata, start_cells, n_steps=200, dt=None)`
As in the spec. `lineage_path` returns a DataFrame per path: step, nearest cell, `lead_re`, `lead_im`, `op_behaviour`, archetype weights, and writes `uns["lineage_paths"]`.

### 1.7 `export_tables(adata, out_dir)`
CSVs: `per_cell_invariants.csv`, `archetypes.csv`, `archetype_drivers.csv`, `lineage_paths.csv`, `ledger_summary.csv` (fraction of entries per reason code, overall and per cell decile of confidence). Eigenvalue columns suffixed `_relative` unless clock-calibrated.

---

## 2. `scjdo2/plot/style.py`

- `BEHAVIOUR_COLOURS = {holder: "#378ADD", drift: "#888780", fork: "#D85A30", damped_oscillator: "#7F77DD", growing_oscillator: "#7F77DD"}`
- `BEHAVIOUR_MARKERS`: filled circle for the first three; ringed circle (hollow with inner dot) for damped; ringed circle with centre dot for growing.
- `sensitive_ring(ax, x, y, colour, r)` helper: dashed circle, radius 2× marker, 0.75 pt, dash (3, 2).
- `DIVERGING = "RdBu_r"`, `CYCLIC = "twilight"`.
- `style()` context manager: 8 pt base font, 0.5 pt axes, no top/right spines, 180 mm figure width, PDF with embedded fonts.
- `confidence_alpha(values, lo, hi)`: rank-based rescale to `[lo, hi]`.
- `relative_label(adata, base)`: returns `f"{base} (relative)"` unless clock-calibrated.

---

## 3. `scjdo2/plot/figures.py`

Each function returns a `matplotlib.figure.Figure` and saves PDF + PNG (300 dpi) when `out` is given. Each has a `--dry` test that renders on the pitchfork and asserts: no overlapping text artists (bounding-box check), no empty axes, every axis labelled.

### 3.1 `figure_operator_space(adata, lineages=None, archetype=None, out=None)`
Layout: 2 rows; row 1 = Panel A (left, 55 % width) and Panel B (right); row 2 = Panel C full width. One shared categorical legend (behaviour colours + markers + the dashed sensitive ring) drawn once, top-right of Panel A.

**Panel A.**
- Coordinates: `obsm["X_umap"]` if present, else first two latent dims.
- Scatter 1: all cells, colour = `lead_re` on `DIVERGING`, norm centred at 0, symmetric limits at the 98th percentile of |lead_re|; alpha = `confidence_alpha(op_confidence, 0.15, 1)`; marker by behaviour. Colour bar: `relative_label(adata, "leading eigenvalue, Re")`.
- Scatter 2: hollow ring on cells with `lead_im > 0`.
- Scatter 3: `sensitive_ring` on cells with `op_sensitive`.
- Lineage paths from `uns["lineage_paths"]` as 0.5 pt grey curves with an arrowhead at the end.
- Annotation: the fork-class cell with the highest `lead_re`, labelled "fork" with a 0.5 pt leader line into clear space (choose the quadrant with fewest points).

**Panel B.**
- Scatter (`lead_re`, `lead_im`), colour + marker by behaviour, alpha by confidence, sensitive ring as in A.
- Horizontal line at Im = 0; vertical dashed line at Re = 0 labelled "Re = 0".
- Axis labels: `relative_label("Re, growth rate")`, `relative_label("Im, rotation rate")`.
- Region text, placed once in quiet space: "stable" (left), "unstable" (right), "oscillating" (upper).
- Lineage paths overlaid as lines with an arrowhead.
- Footnote: "conjugate partner at −Im omitted".

**Panel C.** Calls 3.4 for the chosen archetype (default: first archetype with behaviour fork; fallback: cell with max `lead_re`).

### 3.2 `figure_archetype_sheet(adata, out=None)`
One row per archetype, three columns: (i) mini complex plane with that archetype's eigenvalues and the Re = 0 line; (ii) driver bars via 3.4 with `top=10`; (iii) text block: behaviour name, sensitive flag, eigenvalues (relative), Henrici, transient gain, n cells, mean confidence. Rows sorted by behaviour then by n cells.

### 3.3 `figure_v1_vs_v2(adata_v2, v1_results, out=None)`
Left: v1 pseudotime-slice archetype activations along each branch from the Phase A baseline outputs. Right: Panel A from 3.1. Mark the fork on both. Shared title-free layout; panel letters only.

### 3.4 `plot_drivers(ax, driver_table, mode="real" | "complex", colour, top=15)`
- **real:** horizontal bars, signed weight, right = positive. Rows ordered by sign block then magnitude; thin rule between blocks. Alpha = `confidence_alpha(fisher, 0.2, 1)`; horizontal error bar at bar end = `fisher_se`. Non-`ok` genes drawn hollow with the reason in grey 7 pt text at the bar end. Axis text: "down together | up together".
- **complex:** one-sided bars, length = magnitude, colour = phase on `CYCLIC`, same alpha and error bar rules; a small cyclic colour bar below labelled "early in cycle → late in cycle". No sign axis.
- Mode is chosen automatically from whether `phase` is NaN.

### 3.5 `figure_ledger_summary(adata, out=None)` (small, supplementary)
Stacked bar of reason-code fractions per confidence decile, plus a histogram of `op_rank`. This is the figure that shows what the ledger excluded and why.

---

## 4. `scjdo2/cli.py`

```
scjdo2 opspace --input run_dir/ --out run_dir/opspace/ [--smooth-k 0] [--n-eig 5]
scjdo2 figure operator-space --input run_dir/ --out figs/ [--archetype ID] [--lineages cells.txt]
scjdo2 figure archetypes     --input run_dir/ --out figs/
scjdo2 figure compare        --v1 v1_dir/ --v2 run_dir/ --out figs/
scjdo2 figure ledger         --input run_dir/ --out figs/
```
All idempotent; all read stored results only; all fail loudly if a required key is missing and name the step that produces it.

---

## 5. Tests (`tests/test_opspace_figures.py`)

On the equilibrated pitchfork:
1. `spectrum`: no stored conjugate pairs; `op_rank` == 2 for ≥ 95 % of cells; restricted eigenvalues match latent eigenvalues within 10 % on cells above median confidence.
2. `behaviour`: with the shared `tol`, holder P/R ≥ 0.8; fork P/R reported against the oracle (Phase C criterion).
3. `drivers`: planted driver genes in the top 10 of the fork mode with precision ≥ 0.8 after Fisher filtering; every row has a reason code.
4. Phase convention: largest-magnitude entry real and positive in every stored eigenvector.
5. Complex rendering: on a synthetic damped oscillator (2D limit cycle lifted to 20 genes), `plot_drivers` selects complex mode and the phase ordering of genes matches the planted lag order.
6. Sensitive ring: cells above the 90th percentile Henrici carry the ring artist in both panels (count the artists).
7. Relative label: toggling `clock_calibrated` adds/removes "(relative)" on every eigenvalue axis and CSV column.
8. Figures: no overlapping text artists, no empty axes, PDF and PNG written.

On Paul15 (sanity, not gated): Gata1 and Spi1 opposite-signed in the fork archetype driver table; fork class concentrated in the top-decile `op_field_gradient` region (report the fraction).

---

## 6. Definition of done

- `scjdo2.opspace.compute` and all five figure functions importable and documented; `docs/opspace.md` lists every stored key.
- Five CLI commands run on the pitchfork and Paul15.
- Tests 1–8 pass.
- `benchmarks/figures/<dataset>/` contains operator_space, archetypes, v1_vs_v2, ledger for both datasets.
- `V2_PLAN.md` and `CHANGELOG.md` updated; any deviation from this file recorded with a one-line reason.
