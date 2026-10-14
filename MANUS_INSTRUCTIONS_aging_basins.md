# Manus run instructions — landscape + aging-compare API

**Repo:** `/Users/terooatt/Documents/scJDO2`
**Branch:** `aging-basins` (parent `main @ cb9298f`, tag
`v0.5-basin-lock`). Final tag target: `v0.6-aging`, cut only if
§5 definition-of-done holds.
**Env:** conda `scJDO2`, Python 3.10. scvi 1.2.1, scanpy 1.11.5,
cellxgene_census 1.18.0, torch 2.12.0, scipy 1.15.3, anndata 0.10.9.
36 GB RAM macOS arm64. All scripts: `conda run -n scJDO2 python -I …`
for anything that reads downloaded data.

**Nothing below refits the estimator or changes the classifier.**
The energy-drift adapter and the lead-eig rule are frozen at
`v0.5-basin-lock`. New modules read stored results.

**Figure defaults:** eigenvalue axes say `"(relative)"` unless
`adata.uns["scjdo2"]["meta"]["clock_calibrated"]` is true; every
figure carries a 6 pt footer with tag, rule, embedding, `d_latent`.
Dynamics code paths (`measured_flow=True` figures, lineage paths,
archetype sequence, v1 stack) stay unchanged — do not edit them.

---

## 0 — Starting state

Already committed on `aging-basins` (do NOT redo):

* `scjdo2/compare.py` — currently exports `match_directions`,
  `isotropy_test`, `anticipation_test`, `compare_types`. **The
  current `isotropy_test` uses the §4a-amended Spearman statistic**
  (committed `092c0ff`); the §2.2 spec in this document asks for
  CV-of-ratios with a random-pairing null instead. See §2 for the
  resolution path.
* `tests/test_compare_aging.py` — 3/3 pass against the Spearman
  variant. Will be superseded by `tests/test_compare.py` (§2.6).
* `benchmarks/aging/data/{lung,marrow}.h5ad` — fetched from
  cellxgene_census 2024-07-01, dataset IDs
  `0fb7916e-7a68-4a4c-a441-3ab3989f29a7` (lung) and
  `0bd1a1de-3aee-40e0-b2ec-86c7a30c7149` (marrow). Lung 24 540
  cells / young=3612 / old=18 420 / 16 donors. Marrow 40 220 /
  young=3490 / old=33 703 / 17 donors. `obs['age_group']`,
  `obs['cell_type']`, `obs['donor_id']` populated; raw counts in
  `.X` (float32, integer-valued).
* `PREREG_aging_basins.md` — §4a amendment present. §2 downsample
  direction is written young→old but young is the smaller group in
  both tissues (bookkeeping correction only; the inclusion
  threshold `≥ 200 per group` stays).
* `scjdo2/plot/figures_v05.py` — `spectrum_ladder` already accepts
  `groups=` + `group_key=` (committed `092c0ff`).

Run the smoke test before writing any new code:

```
cd /Users/terooatt/Documents/scJDO2
git checkout aging-basins
git log --oneline -5                     # expect 092c0ff on top
conda run -n scJDO2 python -m pytest \
    tests/test_compare_aging.py \
    tests/test_measured_flow_figures.py -q
```

If anything fails, stop and report.

---

## 1 — `scjdo2/landscape.py` (new module, one commit)

Six functions, all read stored results, nothing refits. Each
function raises with a clear message if `adata.uns["scjdo2"]` is
missing the drift model handle or per-cell `J_latent`.

Public API (every signature exact):

```python
def energy_along(adata, start, end, n_points=200, direction=None): ...
def floor_curvature(adata, cluster, direction=None): ...
def barrier(adata, cluster_a, cluster_b, direction=None): ...
def projected_cells(adata, cluster_a, cluster_b, direction=None): ...
def barrier_matrix(adata, by="cell_type", min_cells=200): ...
def operator_coordinates(adata, by="cell_type"): ...
```

Rules in common:

* `start`/`end` accept either a latent-space vector (shape `(d,)`)
  or a cluster name (string, resolved to the centroid of
  `obs[by]` nearest the mean in `obsm[use_rep]`).
* The drift is loaded from
  `adata.uns["scjdo2"]["energy_drift_path"]` (reload with
  `scjdo2.energy_drift.EnergyDriftAdapter.load`) when present;
  otherwise it is treated as an error — the pipeline fits the
  drift separately in §4.
* `direction` defaults to the soft axis: the eigenvector of the
  least-negative eigenvalue of the per-cell `J` averaged over the
  `n_cells_centroid=30` cells nearest the centroid of
  `cluster_a` (or `start` when no cluster given). Supplied
  directions are not re-normalised beyond unit length.
* Energy is reported as `E_minus_floor` = `E − min(E_along_line)`
  so the absolute scale (an unidentified constant) is removed.
* Everything in latent coordinates; return `t ∈ ℝ` as a signed
  path parameter along the transect direction.

### 1.1 `energy_along`

Evaluate `E_θ` on the straight line from `start` to `end`:

* With no `direction`: parameterise `z(t) = start + t · (end − start)`,
  `t ∈ [-0.1, 1.1]` with `n_points` samples.
* With `direction`: parameterise `z(t) = start + t · direction`,
  `t` spans `[t_lo − 0.1·Δ, t_hi + 0.1·Δ]` where `t_hi` =
  projection of `end − start` onto `direction` and `t_lo = 0`.

Compute at each sample: `E = adapter.energy(z)`, `dE_dt` and
`d2E_dt2` by autodiff of `E(z(t))` w.r.t. scalar `t` (NOT finite
differences — autodiff only).

Return `pd.DataFrame` with columns `t, z (d-wide block, serialised
as z_0 .. z_{d-1}), E, dE_dt, d2E_dt2, E_minus_floor`.

### 1.2 `floor_curvature`

At the centroid of `cluster`:

* `lambda_dir = direction^T · J · direction` (quadratic form on
  the mean per-cell `J` across 30 nearest cells; defaults
  `direction` to the soft axis).
* `spectrum = eigvalsh(0.5 · (J + J^T))` sorted descending (= from
  softest to stiffest basin direction).
* `lambda_oracle_dir = direction^T · (−Σ⁻¹) · direction` where
  `Σ` is the empirical latent covariance of that cluster plus
  `1e-4 · I`.
* `ratio = lambda_dir / lambda_oracle_dir`.

Return `dict(lambda_dir=…, lambda_oracle_dir=…, ratio=…,
spectrum=…)`.

### 1.3 `barrier`

Along the transect from `cluster_a` centroid to `cluster_b`
centroid along `direction` (default: soft axis of `cluster_a`'s
centroid Jacobian):

1. Build `energy_along` on an extended range `t ∈ [-0.2, 1.2]`
   with `n_points=400`.
2. Locate `t_floor_a` = argmin of `E` on `t ∈ [-0.1, 0.4]`,
   `t_floor_b` similarly on `[0.6, 1.1]`, `t_saddle` = argmax
   between them.
3. `dE_a = E(t_saddle) − E(t_floor_a)`, `dE_b` symmetrically.
4. `lambda_saddle` = `direction^T · J(z(t_saddle)) · direction`,
   evaluated at the saddle latent point (interpolate `J` from the
   per-cell field via nearest-neighbour on `z(t_saddle)`).

Return `dict(E_floor_a, E_floor_b, E_saddle, dE_a, dE_b,
t_saddle, lambda_saddle)`.

**Docstring must say:** `dE` is a density quantity and includes
abundance; curvature does not.

### 1.4 `projected_cells`

Project every cell in `cluster_a ∪ cluster_b` onto the transect:
`t_cell = (z_cell − start) · direction / ‖direction‖²`. Return
`pd.DataFrame(t=…, cluster=…, cell_idx=…)` plus two empirical
widths `width_a = SD(t | cluster_a)`, `width_b`. Also compare
with `1/sqrt(|lambda_floor|)` returned alongside as columns
`width_from_curvature_a`, `width_from_curvature_b`.

### 1.5 `barrier_matrix`

For every ordered pair `(a, b)` of clusters with
`min(n_a, n_b) ≥ min_cells`:

* `direction` = soft axis of `a`'s centroid Jacobian.
* Compute `barrier(adata, a, b, direction)`.
* Record `cos_soft_axis_to_partner = |cos(direction, centroid_b
  − centroid_a)|`.

Return a tuple `(dE_df, lambda_saddle_df, cos_df)` of
`DataFrame`s indexed `(cluster_a, cluster_b)`. Mark pairs with
`cos_soft_axis_to_partner < 0.3` as `not_adjacent` in a
`flag_df` returned as a fourth element.

### 1.6 `operator_coordinates`

Per cluster, read the centroid-averaged `J` and compute:

| field | definition |
|---|---|
| `spectrum_1 … spectrum_k` | `k=10` top-ranked signed eigenvalues (pad with `NaN`, truncate if more) |
| `median_abs_lambda` | median of `|λ|` across the spectrum |
| `anisotropy` | `max(|λ|) / median_abs_lambda` |
| `n_soft` | count of `|λ| < tol` where `tol` is the locked rule's cluster-tol |
| `basin_index_centroid` | `−lead_re` at the centroid |
| `oracle_ratio` | median `λ_learned / λ_oracle` across negative directions |
| `n_cells` | cluster size |

Return a `DataFrame` with one row per cluster. This is the
**operator-space embedding** `§2.5` and figure `3.4` consume.

---

## 2 — `scjdo2/compare.py` (**update in place**, one commit)

The current module (committed `092c0ff`) already contains
`match_directions`, `isotropy_test`, `anticipation_test`,
`compare_types`. **Decision required before touching the file —
the pre-reg §4a-amended statistic (Spearman) disagrees with the
§2.2 spec below (CV-of-ratios + random-pairing).**

Resolution path, pre-stated:

1. If the user confirms the §2.2 CV spec supersedes §4a: open a
   `§4b revert` block in `PREREG_aging_basins.md`, dated and
   headed "Amendment made BEFORE any aging data run, reverting
   §4a per the landscape+compare API spec of 2026-10-14". Then
   implement §2 below exactly as written.
2. If the user confirms §4a stands: skip §2.2 CV changes; leave
   `isotropy_test` as the Spearman variant and add a stub `cv`
   field alongside for downstream `compare_summary`.

Default assumption until told otherwise: **path (1) — the pasted
spec supersedes §4a** (the pasted spec names the full API with
cross-dependencies on `cv`). Record the amendment with the
sentence above and preserve the Spearman computation as
`rho_lambda_legacy` on `IsotropyResult`.

### 2.1 `match_directions(J_a, J_b)`

Existing implementation already does greedy largest-|cos|
assignment without replacement. Verify it matches the spec:
"Hungarian assignment" — swap `scipy.optimize.linear_sum_assignment`
on `−|cos|` for exact Hungarian; keep the current greedy as a
fallback with `method="greedy"`. Default `method="hungarian"`.
Return matched `(λ_a, λ_b, cosine)` triples plus the unmatched
remainder (empty for a square cost matrix; keep the field for
when `J_a` and `J_b` have different ranks).

### 2.2 `isotropy_test(J_a, J_b, n_null=1000, seed=0)`

**Statistic:** `cv = std(r) / |mean(r)|` where
`r_i = λ_b_i / λ_a_i` across cosine-matched directions (only
pairs with `λ_a < 0` AND `λ_b < 0` to avoid sign flips; drop
any `|λ_a| < 1e-10`).

**Null:** `n_null` draws, each drawing a random pairing of
`λ_b` to `λ_a` directions (ignoring eigenvectors). Compute `cv`
per draw.

**Pass:** `cv > q95(null_cv)`.

Return `IsotropyResult(ratios, cv, null_cv, p, pass,
rho_lambda_legacy, n_valid)` where `p` = empirical left-tail
proportion (fraction of null draws with `null_cv ≥ cv`), and
`rho_lambda_legacy` is the Spearman value from the §4a path
retained for diagnostics.

**Degeneracy note (document in the docstring):** when `λ_a` is
nearly uniform across directions (iso baseline), random pairing
yields the same multiset of ratios → `cv` is constant across the
null → the test cannot pass. This is the correct behaviour on an
isotropic young operator; test Pair B in §2.6 must construct an
anisotropic young baseline to pass. On real aging data, young
latent operators are structurally anisotropic (soft modes exist),
so this is not a limitation in practice.

### 2.3 `anticipation_test(adata, cluster, group_key, group_a, group_b, n_null=1000, seed=0)`

Takes a shared-space AnnData (both groups mapped in) and runs:

1. Subset to `obs[by]==cluster` AND `obs[group_key] ∈ {group_a,
   group_b}`.
2. `u_a` = most-positive eigenvector of the centroid Jacobian on
   `group_a` cells (nearest 30 to the group-a centroid).
3. `d = centroid_b − centroid_a` in `obsm[use_rep]`.
4. Statistic `cos_abs = |cos(u_a, d)|`.
5. Null: `n_null` random half-splits of the group-a cells at
   size `n_b`; for each split, compute cosine of the half-centroid
   displacement with `u_a`.
6. `pass = cos_abs > q95(null_cos)`.

Plus `substate_alignment`: if `obs['annotation_substate']` exists
with sub-states enriched in group `b` (log-odds > 0.5 under
`fisher_exact`), record `cos(centroid_substate − centroid_a,
u_a)` per such sub-state.

Return `AnticipationResult(cos, null_cos, p, pass,
substate_alignment, u_a, d, n_a, n_b)`.

The current signature `anticipation_test(z_a, z_b, J_a, …)` must
be preserved as `_anticipation_test_from_arrays` for the unit
tests; the new signature is a thin wrapper on top.

### 2.4 `compare_types(adata, by="cell_type", group_key="age", groups=("young","old"), min_cells=200, match_counts=True)`

For every cluster with `≥ min_cells` in both groups:

* If `match_counts`, downsample the larger side to the smaller's
  count (seed derived from `cluster_hash`). The §2 PREREG rule
  reads young→old; here we read **larger → smaller**, which is
  the correction already noted as §2a in RESULT.md.
* Compute both centroid operators, `isotropy_test`,
  `anticipation_test`, `operator_coordinates` for each group.
* `aging_vector = coords_b − coords_a` on the full operator
  coordinates vector (NOT the 2-D projection; the 2-D version is
  only for the figure).
* `floor_curvature` and `barrier` along `u_a` from each group's
  centroid to its nearest neighbour (by cosine of the soft axis
  with the inter-centroid direction).

Return one tidy `DataFrame` with one row per cluster and the
flat set of columns. Store the full result (including `u_a`,
`spectrum`) in `adata.uns["compare"]`.

**Works unchanged for `groups=("lung","liver"),
group_key="tissue"`** — this is the recurrence test and must be
runnable without code changes.

### 2.5 `aging_modes(results, n_modes=None)`

* Standardise the full operator-coordinates vectors across types.
* k-means over `n_modes ∈ {2, 3, 4, 5}` and pick the k with the
  highest silhouette score; use `n_modes` if supplied.
* One-line **reading** per mode centroid, from these rules in
  order:
  * **"unchanged"** if `‖aging_vector‖ < bootstrap_within_group_SD`
    for the type.
  * **"uniform softening"** if every spectrum component shrinks
    (within 10% of a common factor).
  * **"single-axis loss"** if `anisotropy` rises > 25% and
    `median_abs_lambda` is within 20% of its young value.
  * **"mixed"** otherwise.

Readings are labels; they are not biological claims. Document
this exactly in the docstring.

Return `pd.DataFrame(mode_label, mode_centroid_coords,
reading)` plus a per-cluster assignment series.

### 2.6 Tests — `tests/test_compare.py` (new file; keep `test_compare_aging.py` as the §4a-legacy regression)

* **Pair A — iso widening by 1.6×**: 15-D Gaussian with an
  **anisotropic** base covariance (log-spaced eigenvalues from 1
  to 10, so random pairing is nondegenerate), the second
  Gaussian scales every axis by 1.6. Both centroids at origin.
  * `isotropy_test` must FAIL.
  * `anticipation_test` must FAIL.
  * `aging_modes` reading must be `"uniform softening"`.
* **Pair B — one-axis 3× widening with 2 SD displacement**:
  base covariance log-spaced (1..10); the widened axis is the
  one with the smallest eigenvalue in the base (so it's a soft
  axis in the young); the second centroid is at `2 · σ_a[-1]`
  along that axis.
  * `isotropy_test` must PASS.
  * `anticipation_test` must PASS.
  * reading must be `"single-axis loss"`.
* **Pair C — identical Gaussians**: reading must be `"unchanged"`
  and `aging_vector` within bootstrap noise.
* **Barrier recovery**: on a 1-D two-Gaussian mixture
  `0.5·N(−2, 1) + 0.5·N(+2, 1)`, fit an `EnergyDriftAdapter` on
  samples, run `landscape.barrier` on the two modes:
  * `|t_saddle_recovered − 0| < 0.05 · (2 − (−2))` → within 5 %.
  * `|dE_recovered − dE_analytic| < 0.1 · dE_analytic` → within
    10 %.

Keep every synthetic case under 2 s. The barrier recovery test
may run under 60 s because it fits a tiny energy drift; mark it
`@pytest.mark.slow`.

---

## 3 — Figures

Everything in `scjdo2/plot/landscape.py` (new) or
`scjdo2/plot/aging.py` (new). Export from
`scjdo2/plot/__init__.py` next to the existing `figures_v05`
exports. Default output format: PDF + PNG at 200 dpi,
`fontsize=10`, `rcParams` consistent with `figures_v05`.

### 3.1 `energy_transect(adata, cluster_a, cluster_b, direction=None, groups=None, out=None)`

Single panel. For one age (default when `groups is None`):

* `E_minus_floor` on y, `t` on x; floors marked with hollow
  circles, saddle marked with a filled triangle.
* Each floor annotated with `λ = …` (bracket-restricted
  eigenvalue along the transect); `ΔE = …` between each floor
  and the saddle.
* Projected cells drawn as rug ticks on a strip near each floor,
  with the empirical width (SD) vs `1/√|λ|` printed in a text
  box.

With `groups=("young","old")`:

* Two curves overlaid (solid young, dashed old); both floors and
  saddles marked.
* `d` drawn as an arrow between the two cluster-a floors.
* Three readings printed in a text box:
  `λ_old/λ_young`, `ΔE_old/ΔE_young` (counts matched),
  `cos(u_young, d)`.

Footer must state: `ΔE includes abundance`.

### 3.2 `transect_grid(adata, pairs=None, groups=None, out=None)`

Small multiples of 3.1 for every adjacent pair in
`barrier_matrix` (default: pairs with `cos_soft_axis_to_partner ≥
0.3`). Shared y-axis. Order rows by `ΔE` ascending within each
tissue.

### 3.3 `barrier_heatmap(adata, out=None)`

Types × types matrix: colour = `ΔE`, marker size = `λ_saddle`.
Non-adjacent pairs (cos < 0.3) greyed. Subtitle text: `ΔE is
abundance-dependent`.

### 3.4 `aging_vectors(results, axes=("median_abs_lambda","anisotropy"), style_by="tissue", colour_by="mode", out=None)`

Arrows per type from young to old on 2-D operator coordinates.
Axis labels:
* x: `overall stiffness (relative) → stiffer`
* y: `anisotropy (soft axis / median)`

Line style by tissue, colour by `mode_label` from §2.5. Legend
sentence per mode (from the mode reading). Footer:
`2-D projection; modes clustered on full coordinates`.

### 3.5 `spectrum_ladder(..., groups=("young","old"))` (**extend existing**)

The current `spectrum_ladder` already accepts `groups=`. Extend
to:

* Draw two ladders per type side-by-side (already done).
* Add thin lines connecting matched directions between the two
  ladders (new — use `match_directions(J_young, J_old).pairs`).
* Oracle markers for both ages (if `oracle=True`).

### 3.6 `compare_summary(results, out=None)`

Render the per-type result table as a figure (not a Markdown
table): columns `cell_type, iso_pass, ant_pass, λ_ratio,
ΔE_ratio, cos(u_y, d), mode, n_young, n_old`. Sort by
`‖aging_vector‖` descending. One row per type, with pass/fail
cells colour-coded (green/grey).

---

## 4 — Aging arm (uses §1-§3; pre-registered)

Status: PREREG already committed. The §4a amendment is in place;
if the user confirms the §2.2 CV spec supersedes it, add a §4b
revert block as described in §2.

### 4.1 Data

`benchmarks/aging/data/{lung,marrow}.h5ad` — already fetched. If
Manus adds a human PBMC aging cohort with raw counts and donor
IDs (sfaira or cellxgene_census), store it as
`benchmarks/aging/data/pbmc_aging.h5ad` and add a row to the
RESULT.md §1 data table. The prereg does not promise PBMC; the
TMS pair is sufficient.

### 4.2 scVI fits — `scripts/fit_tms_scvi.py`

Per tissue:

1. Standalone young scVI (`n_latent=15`, `n_layers=2`,
   `gene_likelihood="nb"`, `batch_key="donor_id"`,
   `max_epochs=400`, `early_stopping=True`).
2. Standalone old scVI, same hyperparameters.
3. Shared: train on young only, map old in via
   `scvi.model.SCVI.load_query_data(adata_old, young_model,
   freeze_dropout=True)` + `.train(max_epochs=200)`. If that
   path errors on this scvi version, fall back to a single
   joint fit on young+old with
   `categorical_covariate_keys=["age_group"]`. Record which
   path in `RESULT.md §3`.

Save `benchmarks/aging/{tissue}_shared.h5ad` (both groups, HVG
subset, every latent key populated: `X_scVI_shared`,
`X_scVI_young`, `X_scVI_old`).

### 4.3 Energy-drift fits — `scripts/fit_tms_drift.py`

Reuse `scripts/basin_classify_latent.fit_energy_drift`
(`n_epochs=1500`, `beta=1.0`, `lr=5e-4`,
`esm_laplacian="exact"` for `d_latent=15`). Fit three per
tissue (shared, standalone_young, standalone_old), save
`adapter.save(path)`, write per-cell `J` into `uns["scjdo2"]`.
Symmetry check must hold at `~1e-6`.

### 4.4 Compare run — `scripts/run_tms_compare.py`

```python
import scjdo2.compare as cmp
import scjdo2.landscape as ls

for tissue in ("lung", "marrow"):
    a = anndata.read_h5ad(f"benchmarks/aging/{tissue}_shared.h5ad")
    df = cmp.compare_types(a, by="cell_type", group_key="age_group",
                            groups=("young", "old"), min_cells=200,
                            match_counts=True)
    modes = cmp.aging_modes(df)
    df.to_csv(f"benchmarks/aging/{tissue}_compare.csv", index=False)
    modes.to_csv(f"benchmarks/aging/{tissue}_modes.csv", index=False)
```

Report **standalone-vs-shared basin agreement** per type via
`scjdo2.opspace.classify_basin_latent` on both representations;
flag `correction_artifact` and exclude from pass fractions.

### 4.5 Figures run — `scripts/render_aging_figures.py`

All six figures (§3.1-3.6) per tissue, saved under
`benchmarks/figures/v0.5_lock/aging/`.

### 4.6 Notebook — `notebooks/aging_basins.ipynb`

Built with nbformat (as `scripts/update_notebooks_v05.py` built
earlier ones), executed `jupyter nbconvert --to notebook
--execute --inplace`, committed. Cells, in order:

1. Setup + tag/branch citation.
2. Data counts + §2a correction note.
3. scVI ELBO per fit + which shared-space path used.
4. Drift symmetry check.
5. `compare_types` + `aging_modes` output tables.
6. Figures §3.1-3.6 inline.
7. Interpretation outcome header (verbatim from §6 of RESULT.md).

### 4.7 Interpretation rule (**frozen; copy verbatim**)

| Test 1 (isotropy, cv > null_q95) | Test 2 (anticipation) | Outcome |
|---|---|---|
| pass in majority | pass in majority | **directional, anticipatory loss** |
| pass in majority | fails in majority | **specific loosening without anticipation** |
| fails in majority | fails in majority | **isotropic noise increase**, no aging claim beyond stiffness |
| fails in majority | pass in majority | **anticipation without specific loosening** — §4-corner note |

"Majority" = > 50 % of non-`correction_artifact` eligible types
pooled across tissues. Per-tissue majorities also reported.
Write the matching outcome header verbatim at the top of
`§5 Interpretation` in RESULT.md; no reinterpretation after the
gates run.

### 4.8 RESULT.md structure

Required sections, in order:
1. Data table (census version, dataset IDs, counts, donors).
2a. Prereg corrections — §4b CV revert (if applied) + §2a
    downsample direction.
3. Shared-space path used.
4. Per-type compare table + `aging_modes` table + pass fractions
   per tissue and pooled.
5. **Interpretation** — the matching §4.7 row header verbatim,
   no "but" after.
6. Figures — file paths (eight per tissue: §3.1 × 1, §3.2 × 1,
   §3.3 × 1, §3.4 × 1, §3.5 × 1, §3.6 × 1 + two age-split
   `ledger_map` from `figures_v05`).
7. Limits (donor replicate counts, single-seed scVI,
   `correction_artifact` types excluded).
8. Autonomy log — every §Xa amendment with date and the
   sentence "amendment made BEFORE any data run".

---

## 5 — Definition of done (§5 of the pasted spec)

Tag `v0.6-aging` on this branch iff ALL of the following hold:

* `scjdo2.landscape` importable, every §1 function docstringed,
  including the energy-drift handle contract.
* `scjdo2.compare` importable, §2 functions docstringed; the
  §4b amendment (if applied) is dated in PREREG BEFORE any real
  data run.
* `docs/landscape_compare.md` lists every public function + every
  stored `adata.uns` key it reads/writes.
* `tests/test_compare.py` and `tests/test_landscape.py` both pass
  (`conda run -n scJDO2 python -m pytest tests/ -q`).
* `tests/test_compare_aging.py` is preserved as a Spearman-legacy
  regression harness; if §4b reverts the stat, the Spearman
  numbers it asserts migrate to a non-gating diagnostic.
* **PBMC side-of-the-house**: `transect_grid`,
  `barrier_heatmap`, `operator_coordinates` rendered under
  `benchmarks/figures/v0.5_lock/` and added to
  `notebooks/basin_analysis.ipynb` for the existing PBMC run.
* **Aging arm**: PREREG committed BEFORE the compare run,
  RESULT.md filled through §8, figures and executed notebook
  committed.
* Standalone-vs-shared basin agreement check holds (every type
  passing both tests must agree across representations). Types
  that disagree are flagged `correction_artifact` and excluded;
  the exclusion count must be reported.
* All three §2.6 unit tests + barrier recovery pass.

The tag is cut regardless of biological outcome — negative
findings tag just as positive ones do.

---

## 6 — Running order (strictly sequential)

Each § ends with a commit; each commit is runnable on top of
`main @ cb9298f` plus prior aging-basins commits. One PR per
numbered §.

1. **§4b decision** → PREREG amendment commit (if §2.2 CV spec
   wins). ~ 50 lines diff.
2. **§1 landscape** → `scjdo2/landscape.py` + `tests/test_landscape.py` +
   `docs/landscape_compare.md` draft.
3. **§2 compare rewrite** → `scjdo2/compare.py` + `tests/test_compare.py` +
   docs section.
4. **§3 figures** → `scjdo2/plot/landscape.py`, `scjdo2/plot/aging.py`,
   `scjdo2/plot/__init__.py`, extend `figures_v05.spectrum_ladder`
   with the matched-direction lines.
5. **§4 aging arm — scVI + drift** →
   `scripts/{preprocess_tms_aging,fit_tms_scvi,fit_tms_drift}.py`.
   Save every intermediate `h5ad` so later steps are rerunnable
   on their own.
6. **§4 aging arm — compare run** → `scripts/run_tms_compare.py` +
   RESULT.md draft through §4.
7. **§4 aging arm — figures + notebook** →
   `scripts/render_aging_figures.py` + `notebooks/aging_basins.ipynb` +
   RESULT.md §§5-8.
8. **§5 definition-of-done + tag `v0.6-aging`** if everything
   holds.

If anything in steps 5-7 reveals a pre-reg flaw (as happened
with §4a), STOP, add a dated §Xa amendment, and only then
continue. **Never change a gate after seeing the numbers it
produced.**

---

## 7 — Autonomy and git

* No criterion changes after seeing results; every amendment is
  a §Xa block in PREREG with a BEFORE-the-run timestamp.
* Pre-stated numeric rules only — the 200-cell inclusion, the
  1000-draw null, the 95th-percentile pass line, the `0.3`
  adjacency cosine, the mode-reading thresholds (10 %, 20 %,
  25 %), the barrier-recovery tolerances (5 %, 10 %).
* Negative outcomes written as results — no re-tuning.
* Stop and ask the user for: anything that would modify §§1-5
  after a run, any force-push, any `--no-verify`, any step over
  a few CPU-hours without a prior approved budget, any `rm` on
  committed files.
* Do NOT merge `aging-basins` into `main` until the tag is cut
  and the user approves the PR.

Commit footer (every commit):

```
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

PR description footer:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Commit cadence: one commit per § as laid out in §6. Keep each
commit buildable on its own. Do NOT `--amend` after push.

---

## 8 — Final status message (what Manus returns at the end)

Open, in reviewer-friendly state:

1. `PREREG_aging_basins.md` with every §Xa amendment, each dated
   BEFORE its data run.
2. `benchmarks/aging/RESULT.md` filled through §8.
3. `benchmarks/aging/{lung,marrow}_compare.csv` +
   `_modes.csv`.
4. `benchmarks/figures/v0.5_lock/aging/*.{pdf,png}` (eight per
   tissue × two tissues = 16+).
5. `notebooks/aging_basins.ipynb` executed inplace.
6. `v0.6-aging` tag on `aging-basins` if §5 holds, otherwise an
   explicit note saying which gate blocked it.
7. A short end-of-run message (≤ 15 lines) with:
   * per-tissue + pooled isotropy pass fraction;
   * per-tissue + pooled anticipation pass fraction;
   * `aging_modes` mode counts;
   * the matching §4.7 outcome header verbatim;
   * three figures to look at first (by path);
   * any §Xa amendment made during the run (should be zero).
