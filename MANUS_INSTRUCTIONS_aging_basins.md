# Manus run instructions — scJDO2 aging-basins arm (v2, GitHub-based)

**Source:** <https://github.com/manarai/scJDO2>, branch
`aging-basins` (parent `main @ cb9298f`, tag `v0.5-basin-lock`).
**Work in your own environment**; do not assume any local machine,
path or conda env from earlier drafts.
**Output:** commits on `aging-basins` pushed to a fork, or a PR
against `manarai/scJDO2:aging-basins`. Never push to `main`; never
force-push; never `--amend` a pushed commit.

Everything below is **pre-registered**. `PREREG_aging_basins.md`
in the repo holds the frozen rules; this document adds two
amendments (§0.2, §0.3) that **must be committed before any data
run**. After that, no gate, statistic, threshold, null or
inclusion rule changes after seeing results. If a rule cannot
execute as written, stop, add a dated §Xa amendment to PREREG and
RESULT.md, commit, and only then continue.

---

## 0 — Setup and two amendments (commit before any fit)

### 0.1 Environment

```bash
git clone https://github.com/manarai/scJDO2 && cd scJDO2
git checkout aging-basins
python -m venv .venv && source .venv/bin/activate      # or conda; Python ≥ 3.10
pip install -e ".[compare]"
pip install cellxgene-census scvi-tools scanpy scarches
python -m pytest tests/test_compare_aging.py tests/test_measured_flow_figures.py -q
```

Record in `benchmarks/aging/ENV.md`: Python, torch, scvi-tools,
scanpy, cellxgene-census versions, CPU/GPU, RAM. If any test
fails, stop and report before writing scripts.

### 0.2 Amendment §2a — downsampling direction

PREREG §2 says "downsample young to the number of old"; young is
the smaller group in both TMS tissues. **Rule as executed:** per
`(donor, cell_type)`, downsample the **larger** age group to the
smaller, seed 0; drop `(donor, type)` cells with fewer than 20
cells on either side (bookkeeping floor, **not a gate**); then
keep only cell types with `≥ 200` cells in both age groups (the
inclusion gate, unchanged). Commit this as §2a in both PREREG and
`benchmarks/aging/RESULT.md`.

### 0.3 Amendment §4b — isotropy statistic (replaces §4a)

The §4a Spearman statistic does not test what §4 intends: under
uniform (isotropic) softening every eigenvalue scales by the same
factor, the rank order is preserved and ρ ≈ 1, so "pass = ρ >
null" would pass the very outcome the test is meant to reject.
Use instead:

* For cosine-matched direction pairs `k`:
  `r_k = log(|λ_old,k| / |λ_young,k|)`.
* **Statistic** `S = max_k |r_k − median(r)|` — how far the
  most-changed direction departs from the typical change.
  Isotropic softening gives `S ≈ 0`; single-axis loss gives large
  `S`.
* **Null** — **split-half within young**. Split the young cells
  of the type into two random halves at the old sample size,
  compute the centroid operator of each half from the fixed
  shared-space drift, compute `S` between the halves; repeat
  1000×. This is sampling noise under no aging.
* **Pass** = `S` above the null 95th percentile.
* `IsotropyResult` keeps `rho` and `cv_ratio` as **provenance
  columns**; neither is a gate.

Implement in `scjdo2/compare.py`, update
`tests/test_compare_aging.py` so that:

* isotropic 1.6× widening **fails** (`S` within null);
* single-axis 3× widening **passes**;
* identical Gaussians **fail**.

All three must pass before data. Commit as:

```
compare: §4b isotropy statistic (prereg amendment, before data run)
```

### 0.4 Anticipation test (unchanged, PREREG §5)

`|cos(u_young, d)|`, `u_young` = eigenvector of the least-negative
eigenvalue of the young centroid operator, `d = old centroid −
young centroid` in the shared space; null = 1000 random
half-splits of young at the old sample size; pass = above the
null 95th percentile. Also report, for annotated sub-states
enriched in old (log-odds > 0.5), the cosine of their centroid
offset with `u_young`.

---

## 1 — Data (`scripts/fetch_tms_aging.py`, exists; run it)

Tabula Muris Senis via `cellxgene_census` (census version pinned
in the script; record it). Tissues: lung and bone marrow. Keep
`obs['cell_type']`, `obs['donor_id']`, `obs['development_stage']`,
`obs['age_group']` (young = 3 months, old = 18–24 months; drop
4-week cells). Raw integer counts in `.X`. Record per-tissue
counts, donor counts and dataset IDs in RESULT.md §1.

---

## 2 — Preprocessing (`scripts/preprocess_tms_aging.py`, write it)

Per tissue:

* apply §0.2;
* store counts in `.layers["counts"]`;
* `scanpy.pp.highly_variable_genes(flavor="seurat_v3",
    n_top_genes=2000, batch_key="donor_id")` on the downsampled
  pooled matrix;
* `.raw = adata` **before** HVG subsetting;
* save `benchmarks/aging/data/{tissue}_processed.h5ad` and
  `benchmarks/aging/{tissue}_cell_filter.csv`.

---

## 3 — Shared reference space by scArches (`scripts/fit_tms_scvi.py`, write it)

This is the **reference-mapping design**: a frozen reference
trained on young, old mapped in, plus standalone fits as controls.

All fits: `n_latent=15`, `n_layers=2`, `gene_likelihood="nb"`,
`batch_key="donor_id"`, `max_epochs=400` (reduce to 200 on
CPU-only and say so), `early_stopping=True`, patience 30, seed 0.
Record validation ELBO per fit.

* **Reference (young only)** → `models/{tissue}_ref/`; latent
  `obsm["X_scVI_young"]` for young cells.
* **Query mapping (scArches)**:
  `scvi.model.SCVI.load_query_data(adata_old, ref_model)`, then
  `.train(max_epochs=200, plan_kwargs={"weight_decay": 0.0})` with
  the reference weights frozen (this is the scArches procedure
  inside scvi-tools; `scarches` package is the fallback if the
  API differs). Latent for both ages →
  `obsm["X_scVI_shared"]`.
* **Standalone old (old only)** → `obsm["X_scVI_old"]`.

Save `benchmarks/aging/{tissue}_shared.h5ad` with all three
representations. Record in RESULT.md §3 which mapping path ran.
Note for later: the same procedure maps human aging lung data
into the Human Lung Cell Atlas reference; that is the atlas arm,
not this run.

---

## 4 — Energy drift and operators (`scripts/fit_tms_drift.py`, write it)

Use `scjdo2.energy_drift` exactly as at `v0.5-basin-lock` (exact
score matching, symmetric J, no prior, no residual):

```python
drift = fit_energy_drift(z, n_epochs=1500, seed=0, beta=1.0, lr=5e-4)
J = drift.jacobian_autodiff(z)
sym = drift.symmetry_check(z, n_cells=100)    # record; expect ~1e-6
```

Three fits per tissue: **shared** (both ages,
`X_scVI_shared`), **standalone young**, **standalone old**. Save
`benchmarks/aging/{tissue}_{fit}.h5ad` and the drift models.
Then `scjdo2.opspace.classify_basin_latent(rule="lead_eig")` on
each; store `basin_index` per cell.

---

## 5 — Per-type comparison (`scripts/run_tms_compare.py`, write it)

```python
from scjdo2.compare import compare_types
df = compare_types(adata_shared, by="cell_type",
                    groups=("young", "old"),
                    group_key="age_group", use_rep="X_scVI_shared",
                    n_cells_centroid=30, min_cells=200, n_null=1000, seed=0)
```

Columns: `cell_type, n_young, n_old, S, iso_null_q95, iso_pass,
rho (provenance), cv_ratio (provenance), cos_abs_u_young_d,
ant_null_q95, ant_pass, substate_alignment,
basin_fraction_young, basin_fraction_old,
lambda_ratio_soft_axis, median_abs_lambda_young,
median_abs_lambda_old, anisotropy_young, anisotropy_old`.

**Correction-artifact flag (PREREG §3):** a type is
`correction_artifact` when its basin (`basin_fraction ≥ 0.5` under
the locked rule) exists in the shared space but not in the
standalone fit of the same age group, or vice versa. Report
these types but exclude them from pass fractions, stating the
count.

Save `benchmarks/aging/{tissue}_compare_shared.csv` and
`_compare_standalone.csv`. Pooled table across tissues in
`benchmarks/aging/compare_pooled.csv`.

Also compute **`aging_modes`** on the per-type operator-coordinate
vectors (sorted spectrum padded to 10, `median |λ|`,
`anisotropy`, `n_soft`, `basin_index`): k-means with `k` by
silhouette over 2–5, plus a **label per mode from its centroid**:

* `"uniform softening"` if all components shrink together and
  anisotropy changes `< 10 %`;
* `"single-axis loss"` if anisotropy rises `> 25 %` with
  `median |λ|` within `20 %`;
* `"unchanged"` if displacement is below the split-half noise;
* else `"mixed"`.

Labels are descriptions, not claims.

---

## 6 — Interpretation rule (frozen; copy verbatim into RESULT.md §5)

| Test 1 isotropy (`S > null_q95`) | Test 2 anticipation | Outcome header |
|---|---|---|
| pass in majority | pass in majority | **directional, anticipatory loss** |
| pass in majority | fail in majority | **specific loosening without anticipation** |
| fail in majority | fail in majority | **isotropic noise increase; no aging claim beyond stiffness** |
| fail in majority | pass in majority | **anticipation without specific loosening (corner case; report as such)** |

"Majority" = `> 50 %` of non-artifact eligible types, pooled
across tissues; per-tissue majorities also reported. Write the
matching header verbatim; no reinterpretation after the gates run.

---

## 7 — Figures (`scripts/render_aging_figures.py`, write it)

All from stored results; PDF + PNG at 200 dpi under
`benchmarks/figures/aging/`; footer on every figure:

```
v0.5-basin-lock · aging-basins · classify_basin_latent(rule='lead_eig')
· scArches shared space · d_latent=15 · units relative
```

1. `spectrum_ladder(adata_shared, by="cell_type",
    groups=("young","old"), oracle=True)` — extend the function
   if it lacks `groups=`; matched directions connected by thin
   lines.
2. `soft_axis_map(adata_shared, by="cell_type")` with the
   per-type **old-centroid displacement arrow** overlaid on the
   `u_young` arrow; extend the function and add a unit test
   before using it on real data.
3. `basin_index_by_type(adata_shared, by="cell_type",
    group_key="age_group")` — one strip per type, faceted by
   age.
4. `ledger_map(adata_shared)` per age group.
5. New `plot.aging_vectors(results,
    axes=("median_abs_lambda", "anisotropy"), style_by="tissue",
    colour_by="mode")` — one arrow per type from young to old
   coordinates; axes `"overall stiffness (relative) → stiffer"`
   and `"anisotropy (soft axis / median)"`; footer states it is
   a 2-D projection and modes were clustered on full
   coordinates. Add to `scjdo2/plot/figures_v05.py` with a
   synthetic unit test.
6. New, **if the energy adapter exposes `E(z)`** (check
   `energy_drift` for an `energy` or `potential` method):
   `plot.energy_transect(adata_shared, type, neighbour,
    groups=("young","old"))` — `E` along the line from the young
   centroid along `u_young` toward the nearest other centroid,
   young and old overlaid, floors and saddle marked, `λ` at each
   floor, `ΔE` with the note that `ΔE` includes abundance,
   projected cells as dots. If `E(z)` is not exposed, skip and
   record why.

### 7b — Full QC and analysis figure set (every figure the pipeline has)

For each tissue and each age group, in addition to §7, render
the complete existing figure set under
`benchmarks/figures/aging/{tissue}/{age}/`, **reading stored
results only**:

* `figure_operator_space` (snapshot defaults: Panel A
  `basin_index`, Panel B spectrum ladder, Panel C responders) and
  `figure_ledger_summary`.
* `figure_archetype_sheet` **if** archetype weights were computed
  (run the Step-8 semi-NMF adapter on the shared-space operators
  with `K` by held-out likelihood; otherwise record why skipped).
* `spectrum_ladder` (single group), `soft_axis_map`,
  `ledger_map`, `basin_index_by_type` (single group).
* `oracle_agreement` is pitchfork-only; instead render the
  **Gaussian-oracle curvature ratio per type** (learned vs
  `−Σ⁻¹` on the contractive directions) as a bar chart with the
  `1/3 – 3` band marked.
* **QC panels**, one figure each:
  * scVI ELBO curves per fit;
  * the `d_latent` choice (fixed at 15 here; state it);
  * within-cluster `σ` versus the drift's `σ` scale;
  * symmetry check histogram;
  * permutation null of `lead_re` (shuffled positions, 10
    refits) against the real distribution;
  * tangent-rank distribution;
  * donor × type cell-count matrix before and after
    downsampling;
  * standalone-versus-shared basin agreement per type as a
    confusion plot;
  * `correction_artifact` list.
* The inter-cluster gap control: behaviour counts in the lowest
  local-density decile versus cluster centres, as on PBMC.

Then, **per tissue across ages**: §7 items 1–6, plus
`compare_summary` (per-type table as a figure: `iso pass,
ant pass, S, cosine, λ ratio, mode, n per group`, sorted by
displacement) and the standalone-versus-shared comparison of
every per-type quantity.

Every file is listed in RESULT.md §6 **by path**, and every
figure appears in the notebook with a markdown cell naming what
it shows and the rule. **Nothing** is rendered from refitted or
edited results.

---

## 8 — Notebook (`notebooks/aging_basins.ipynb`)

Built by `scripts/make_aging_notebook.py` with `nbformat` and
executed **in place**. Cells:

1. setup and citation;
2. preprocessing summary (counts, donor × type matrix);
3. scVI metrics and mapping path;
4. symmetry checks;
5. per-type compare tables and majorities;
6. figures inline via the `figures_v05` calls;
7. the §6 rule row and outcome header.

Markdown above each result states what it shows and which rule
produced it.

---

## 9 — RESULT.md (`benchmarks/aging/RESULT.md`), sections in order

* §1 Data
* §2a Prereg correction (downsampling)
* §3 Mapping path
* §4 Per-type tables (shared and standalone, totals with pass
  fractions per tissue and pooled, artifact count)
* §4b Isotropy amendment (statistic, null, reason, timestamp,
  "made before any data run")
* §5 Interpretation header, verbatim
* §6 Figure paths
* §7 Limits (one scVI seed unless the table motivates reruns;
  human cohort deferred; donor-level replicate shifts in a
  side-car CSV when donor counts allow)
* §8 Autonomy log (every amendment with timestamp and the
  sentence stating it was made before the relevant run).

---

## 10 — Autonomy rules

* No criterion changes after results.
* Pre-stated numeric rules only.
* Negative outcomes are results.
* Never fabricate, simulate or impute data to improve a number.
* Stop and ask for: any change to PREREG §§1–6 after a run, any
  push to `main`, any force-push, deletion of committed files,
  or compute beyond a few GPU-hours (or ~24 CPU-hours) without
  approval.
* Do not merge `aging-basins` into `main`.

Commit cadence: one commit per numbered section, section tag in
the subject (`aging: §3 scArches shared space`). Keep each commit
runnable against `main @ cb9298f` plus prior `aging-basins`
commits.

Commit footer:

```
Co-Authored-By: Manus <noreply@manus.im>
```

Add the AI-generation note the repo's CONTRIBUTING file
specifies, if any.

---

## 11 — Deliverables

* `benchmarks/aging/RESULT.md` complete to §8.
* `benchmarks/aging/{lung,marrow}_compare_shared.csv`,
  `_compare_standalone.csv`, `compare_pooled.csv`,
  `aging_modes.csv`.
* `benchmarks/figures/aging/` with the §7 comparison figures and
  the full §7b per-tissue, per-age QC and analysis set, all
  listed by path in RESULT.md §6.
* `notebooks/aging_basins.ipynb`, executed in place.
* `benchmarks/aging/ENV.md`.
* A PR against `manarai/scJDO2:aging-basins` (or a pushed branch
  on a fork with the URL), and an end-of-run message of **≤ 10
  lines**: per-tissue and pooled pass fractions for both tests,
  artifact count, the §6 outcome header verbatim, and the three
  figures to look at first. **No reinterpretation after the
  header.**
