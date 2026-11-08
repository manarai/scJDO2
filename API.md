# scJDO2 API guide

How to run every scJDO2 analysis and figure from Python. Results are written
back into the `AnnData` (`obs`, `obsm`, `uns["scjdo2"]`). There are two routes,
by the kind of data:

* **Time course** (cells sampled at several times): the flow operator,
  `sj.flow` (section 0). This is the route evaluated in the App Note.
* **Snapshot** (one time point): the energy operator, `sj.fit`, and its
  behaviour landscape (sections 1–4).

```python
import scjdo2 as sj

sj.flow(adata, time_key="day")                             # 0. flow operator, lambda_1
sj.flow_groups(adata, "cell_type")                         #    lambda_1 per group
sj.flow_stability(adata, groupby="cell_type")              #    agreement across refits
sj.flow_archetypes(adata)                                  # 0b. operator shapes, label per cell
sj.flow_transitions(adata)                                 #     archetype -> archetype along the OT plan
sj.flow_transitions(adata, labels="cell_type")             #     the same for any label column
```

Snapshot:

```python
import scjdo2 as sj

drift = sj.fit(adata, use_rep="X_pca")                     # 1. operator field
sj.analyse(adata, cluster_key="cell_type", drift=drift)    # 2. behaviour landscape
sj.orient(adata, ["labelling", "lineage"])                 # 3. directions (optional)
sj.figures(adata, "figures/", cluster_key="cell_type")     # 4. every figure supported
```

or all of it in one call, saved to a run folder:

```python
summary = sj.run(adata, "runs/lung", cluster_key="cell_type",
                 sources=["labelling", "lineage"])                 # snapshot
summary = sj.run(adata, "runs/larry", cluster_key="cell_type",
                 time_key="day", stability_seeds=(0, 1, 2),
                 motifs=True)                                      # time course (+ 0b)
```

Each call wraps public functions in the modules listed at the end; you can
call those directly when you need finer control.

---

## What you need

| Analysis | Required in the AnnData |
|---|---|
| `flow` | a numeric sampling time in `obs` (≥ 2 times); log-normalised counts in `X` or a PCA in `obsm["X_pca"]` (10–20 components; not Factor Analysis) |
| `flow_groups`, `flow_stability` | `flow` done; a group column in `obs` |
| `flow_archetypes`, `flow_transitions`, `flow_integrate` | `flow` done (any version for archetypes; this version for `flow_integrate`, which needs the stored drift weights) |
| `flow_paths`, `flow_transitions(exact=False)` | `flow` run with `keep_plan` (default 100) |
| `fit` | a low-dimensional embedding in `obsm` (PCA, FA or scVI). In synthetic tests the fitted energy operator recovered a multi-well landscape only in 2-d; for d ≥ 4 read `J` as descriptive density curvature (see "Things to know") |
| `analyse` | `fit` done; a cell-type column in `obs` |
| `orient` | at least one direction source (table below) |
| energy figure | a 2-D basis, `obsm["X_umap"]` by default |

### Direction sources for `orient`

| Source | Output | Put in the AnnData | Notes |
|---|---|---|---|
| `labelling` | flow (ranked 1st) | `obsm["velocity_labelling"]`: velocity per hour **in the same embedding** | from metabolic labelling (new/total counts); see "Labelling velocity" below |
| `velocity` | flow (2nd) | `obsm["velocity_<basis>"]`, e.g. `velocity_pca` (scVelo's `velocity_embedding(basis="pca")`) | your own scVelo run; demoted automatically if it disagrees with labelling (§15) |
| `lineage` | developmental (1st) | `obs["clone"]` (barcode, missing if undetected) and `obs["day"]` | LARRY-style barcodes over ≥ 2 sampling days |
| `pseudotime` | developmental (2nd) | `obs["palantir_pseudotime"]`, `uns["palantir_start_cell"]` = the cell Palantir actually started from (the pseudotime minimum) | your own Palantir run; refused if it decreases with `obs["day"]` (§16) |

Keys can be renamed with keyword arguments (see `sj.orient`).

---

## 0. `sj.flow` — flow operator from a time course

```python
sj.flow(adata, time_key="day", use_rep="X_pca", n_comps=15)
```

Entropic optimal transport between consecutive sampling times gives each cell
a velocity; a smooth drift is fitted to those velocities, and its Jacobian is
the flow operator `J_flow` at every cell. Writes `obs["flow_lambda1"]` (leading
eigenvalue), `obs["flow_has_velocity"]` (False on the last time point, whose
operator is extrapolated), `obsm["flow_velocity"]`, `obsm["flow_drift"]`,
`obsm["flow_descriptors"]` (raw) and `obsm["flow_descriptor_ranks"]`,
`uns["scjdo2"]["J"]` and `uns["flow"]` (parameters, OT diagnostics, fit record).
All parameters and usage notes: [docs/flow.md](docs/flow.md).

`lambda_1` is a within-dataset branch-point score for **groups** of cells, not a
stability classification, and per-cell values vary between refits. Two calls
cover what to check:

```python
tab = sj.flow_groups(adata, "cell_type", n_boot=1000)
st  = sj.flow_stability(adata, seeds=(0, 1, 2), groupby="cell_type")
```

* `flow_groups` returns, per group: `n_cells`, `frac_extrapolated`,
  `median_lambda1` with a 95% bootstrap interval over cells (`ci_low`,
  `ci_high`; none below `min_cells=20`) and `rank`. The interval is conditional
  on the fitted operator. Stored in `uns["flow"]["groups"]`.
* `flow_stability` refits transport and drift once per seed (on a light copy;
  the stored operator is not changed) with the stored run's parameters, and
  returns the Spearman agreement of per-cell `lambda_1` between seeds (all
  pairs, median, minimum), the same for group medians with `groupby`, and
  whether the top group is the same in every refit. Writes
  `obs["flow_lambda1_rank_sd"]` (per-cell spread of the rank quantile across
  refits). Cost: one `flow` fit per seed.

`sj.figures` draws `flow_groups` (median `lambda_1` per group with its
interval, and `lambda_1` on `X_umap`) for a flow run; the landscape figures
need the energy operator and are skipped.

## 0b. Operator motifs along the time course

`J_flow` gives every cell an operator, but nothing in `J` links a day-2 cell to the
day-4 cells it becomes. The OT plan does, and `sj.flow` keeps it
(`obsp["flow_plan"]`). These calls chain operator **shapes** (archetypes) along it.
**New and not yet benchmarked**: archetype stability across replicates has not been
tested yet; read results within one dataset.

```python
sj.flow_archetypes(adata, n_archetypes="auto")             # semi-NMF of J_flow
tab   = sj.flow_transitions(adata, n_perm=1000)            # main readout
tab_c = sj.flow_transitions(adata, labels="cell_type")     # any obs column as labels
paths = sj.flow_paths(adata, n_paths=10_000, collapse=True)
res   = sj.flow_integrate(adata, start=np.flatnonzero(adata.obs["day"] == 2)[:20])
```

| call | what it does | writes / returns |
|---|---|---|
| `flow_archetypes` | each cell's operator divided by its Frobenius norm (shape, not speed; `normalize=None` keeps magnitudes), semi-NMF into `K` archetypes (`K` by held-out error over `k_candidates`; warns if it picks the largest), fitted on ≤ 4,000 cells and assigned to all by non-negative least squares | `obsm["flow_archetype_weights"]`, `obs["flow_archetype"]` (A1 = most common), `obs["flow_archetype_share"]` (1 = pure), `uns["flow"]["archetypes"]` (patterns, eigenvalues, K curve); returns the patterns (K, d, d) |
| `flow_transitions` | per pair of consecutive times: the mass cells labelled *k* send to cells labelled *l* (`prob` = P(l \| k)), against a null that shuffles source labels within the source time (sizes and target composition kept). Re-runs the transport with the stored settings, checks it reproduces the stored velocities, and contracts the **full** plan; `exact=False` uses the stored top-k plan | DataFrame: `from_time, to_time, from_label, to_label, n_from, mass, prob, expected, expected_prob, z, p_enriched, p_depleted, to_extrapolated`; `uns["flow"]["transitions"]` |
| `flow_paths` | chains sampled through the stored plan from the first time to the last; counts of label sequences vs a Markov chain on the same plan (`obs_over_expected` ≠ 1 = memory or rare routes). The stored plan keeps each cell's nearest destinations only, so frequencies are approximate | DataFrame: `path, count, frequency, expected, obs_over_expected`; `uns["flow"]["paths"]` |
| `flow_integrate` | RK4 paths of the fitted drift (rebuilt by `scjdo2.flow_operator.load_drift`), with λ₁ and archetype weights along each path; a deterministic path cannot branch, so use it for pictures | dict: `t`, `z` (m, steps+1, d), `lambda1`, `archetype_weights` |

Figures: `scjdo2.plot.figure_flow_archetypes(adata)` (per archetype: eigenvalues on
the complex plane above its matrix) and `scjdo2.plot.figure_flow_transitions(adata)`
(P(to | from) per time pair, `*` where p_enriched < 0.01). `sj.figures` draws both
when their inputs exist.

**Reading an archetype.** The eigenvalues of flow `J`: a positive real part is a
direction along which nearby cells spread apart (branch-like), a negative one a
direction they converge along; a non-zero imaginary part is local rotation, in
radians per unit of `time_key`. It is not a calibrated clock (`clock_calibrated` is
False) and has not been validated as a cycle. The matrix heatmap shows the
non-normal part (asymmetry across the diagonal: stretching, transient gain), which
the eigenvalues do not. Flow `J` carries direction from time itself; `sj.orient` is
for the energy operator, whose eigenvectors are unsigned.

## 1. `sj.fit` — the operator field

```python
drift = sj.fit(adata, use_rep="X_pca", n_epochs=1500, seed=0,
               with_rate_model=False)
```

Fits an energy `E(z)` by exact score matching on `obsm[use_rep]`; the drift
is the score `f = −β∇E` and the operator at each cell is its Jacobian
`J = −β Hess E` (symmetric: it describes the curvature of the cell density).

Writes `uns["scjdo2"]["J"]` (n × d × d), `obsm["X_drift_v"]`,
`obs["energy"]`, `uns["scjdo2"]["meta"]`. Returns the fitted drift.

* `with_rate_model=True` also fits the NB rate model on raw counts, the
  gene-space Jacobian and the Fisher ledger (for driver genes; slow).
* Other arguments: `hidden`, `depth`, `beta`, `lr`, `loss` ("esm" default).

## 2. `sj.analyse` — behaviour landscape

```python
info = sj.analyse(adata, cluster_key="cell_type", drift=drift,
                  resting_types=None)
```

Writes, per cell:

| Column | Meaning |
|---|---|
| `lead_re`, `spectral_gap`, `henrici`, ... | operator-space invariants |
| `op_behaviour_basin` | basin / fork / drift (lead-eigenvalue rule): labels of the density curvature, not stability |
| `op_basin_index` | continuous: positive = basin, larger = deeper |
| `op_frac_neg` / `zero` / `pos` | eigenvalue-sign fractions (ternary plot) |
| `op_stiffness`, `op_lambda_min` | well depth |
| `energy` | when `drift` is given |

The tolerance comes from a Gaussian oracle over `resting_types` (all types
by default) and is returned as `info["tol"]`.

## 3. `sj.orient` — directions

```python
out = sj.orient(adata, ["labelling", "velocity", "lineage", "pseudotime"],
                labelling_key="velocity_labelling", velocity_key="velocity_pca")
```

Gives each eigen-axis of `J` a sign at every cell, from measured evidence.
Two outputs, never merged with each other:

* **flow** — how the cell moves now: labelling > velocity. Also magnitude
  (labelling clock) and rotation `omega`.
* **developmental** — where development goes over days: lineage >
  pseudotime.

Within each output the higher-ranked source decides; disagreements are
recorded as conflicts, never averaged. Writes `obs["flow_orientation_class"]`
and `obs["dev_orientation_class"]` (measured / imputed / none),
`obs["*_conflict_flag"]`, `obs["flow_vs_dev_disagree"]`,
`obs["equilibrium_flag"]`, `obs["stationarity_flag"]`,
`obsm["orient_flow_sign"]`, `obsm["orient_dev_sign"]`, and the full ledger in
`uns["scjdo2"]["orient_meta"]` (coverage, reliability check outcome, guards,
`J` hash before/after). `J` is never modified.

Any `OrientConfig` field can be passed as a keyword: `clone_key`, `day_key`,
`pseudotime_key`, `root_cell`, `reliability_min` (`{}` switches the §15
check off), `n_axes`, `align_min`, ... Full rules:
[docs/ORIENTATION_SPEC.md](docs/ORIENTATION_SPEC.md).

## 4. `sj.figures` — every figure the state supports

```python
paths = sj.figures(adata, "figures/", cluster_key="cell_type",
                   condition_key=None, conditions=None,   # e.g. "treatment", ("control", "treated")
                   include=None, verbose=True)
```

Writes `<name>.pdf` and `.png` for each figure whose inputs exist and skips
the rest (`verbose=True` says why). `sj.FIGURES` lists them:

| Name | Shows | Needs |
|---|---|---|
| `operator_space` | cells in operator space, coloured by basin index | analyse |
| `landscape` | density on the (lead eigenvalue, spectral gap) plane | analyse |
| `landscape_3d` | the same as a 3-D surface with stiffness | analyse |
| `ternary` | eigenvalue-sign ternary + per-type composition, by condition | analyse |
| `energy` | energy landscape `E(z)` over a 2-D basis | fit + `X_umap` |
| `basin_index_by_type` | basin index per cell type | analyse |
| `spectrum_ladder` | per-type eigenvalue ladder vs Gaussian oracle (paired by condition) | fit |
| `soft_axis_map` | soft-axis arrows per type on the embedding | fit |
| `orientation` | flow / developmental classes on the embedding, coverage and conflicts by group, the ledger | orient |
| `flow_groups` | median λ₁ per group with its interval, and λ₁ on the embedding | flow + `cluster_key` |
| `flow_archetypes` | per archetype: eigenvalues on the complex plane and the matrix | flow + `flow_archetypes` |
| `flow_transitions` | P(to \| from) per time pair, enriched cells marked | flow + `flow_transitions` |

`include=["ternary", "orientation"]` renders a subset.

## 5. `sj.run` — everything at once

```python
summary = sj.run(adata, "runs/x", cluster_key="cell_type", use_rep="X_pca",
                 sources=["labelling", "lineage"],
                 orient_config={"labelling_key": "velocity_labelling"},
                 with_rate_model=False,
                 fit_kwargs={"n_epochs": 1500})
```

Fit (skipped if `J` is already stored, unless `refit=True`) → analyse →
orient (if `sources`) → figures → save. With `time_key=` the flow route runs
instead: `flow` → `flow_groups` → `flow_stability` (if `stability_seeds`) → with
`motifs=True`, `flow_archetypes` and `flow_transitions` (`transitions.csv`, and
`transitions_by_cluster.csv` by `cluster_key`) → figures → save. Writes to the run folder:

```
runs/x/
  adata.h5ad, phase_c.npz   # reload with scjdo2.io.load("runs/x")
  figures/                  # every supported figure, .pdf + .png
  summary.json              # settings, tolerance, behaviour fractions, orientation ledger
  per_cell.csv              # per-cell results
```

---

## Recipes

**Behaviour landscape of one tissue**
```python
sj.run(adata, "runs/pbmc", cluster_key="cell_type", use_rep="X_scVI")
```

**Development with lineage and labelling**
```python
drift = sj.fit(adata, use_rep="X_pca")
sj.analyse(adata, cluster_key="cell_type", drift=drift)
sj.orient(adata, ["labelling", "lineage", "pseudotime"])
sj.figures(adata, "figures/dev", cluster_key="day")
```

**Synthetic data with known truth** (for testing a pipeline)
```python
from scjdo2.data.synthetic_multimodal import make_multimodal, embed_linear
adata = make_multimodal("v2_default", seed=0)      # lineage, labelling, splicing, days
```
See [docs/SYNTHETIC_MULTIMODAL.md](docs/SYNTHETIC_MULTIMODAL.md) for which
keys are inputs and which are truth (never use truth as an input).

## Labelling velocity

`orient` needs the labelling velocity **in the embedding**. The pieces:

1. `scjdo2.labelled_rates.LabelledRateModel().fit(new, total, z, pulse_hours)`
   gives per-gene degradation rates `γ`.
2. Per cell and gene, transcription `α = γ·new / (1 − e^{−γ t})` and
   velocity `v = α − γ·total`.
3. Map `v` into the embedding by the chain rule through normalisation,
   `log1p` and the embedding's linear map.

Step 3 needs a **linear** embedding (PCA or FA) with its normalisation
target and loadings. `scripts/orient_end_to_end.py`
(`labelled_velocity`, `chain_rule_count_velocity`) does all three for the
synthetic data's stored embedding; it is not yet a library function, and a
non-linear embedding (scVI, UMAP) has no exact chain rule.

## Command line

```bash
scjdo2 run      --input data.h5ad --out runs/x --use-rep X_pca [--with-rate-model]
scjdo2 opspace  --input runs/x --out runs/x/opspace
scjdo2 figure   operator-space|archetypes|ledger|... --input runs/x --out figs/
```

The CLI covers fitting and the operator-space figures; the flow operator
and orientation are Python-only for now.

## Where each step lives

| Call | Module functions |
|---|---|
| `flow` | `scjdo2.flow_operator` (`ot_velocity`, `drift_jacobian`, `flow_descriptors`), `scjdo2.measured_drift.MeasuredDrift` |
| `flow_groups`, `flow_stability` | `scjdo2.api`; figure `scjdo2.plot.flow.figure_flow_groups` |
| `flow_archetypes`, `flow_transitions`, `flow_paths`, `flow_integrate` | `scjdo2.flow_motifs` (archetypes via `scjdo2.archetypes`; plan via `ot_velocity(plan_top_k=, plan_apply=)`; drift via `flow_operator.load_drift`); figures `scjdo2.plot.flow.figure_flow_archetypes`, `figure_flow_transitions` |
| `fit` | `scjdo2.pipeline.fit_latent_field`, `scjdo2.energy_drift.EnergyDriftAdapter`, `scjdo2.pipeline.fit_rate_and_gene_field` |
| `analyse` | `scjdo2.landscape.analyse` → `scjdo2.opspace` (`oracle_tol`, `spectrum`, `invariants`, `classify_basin_latent`, `behaviour_fractions`, `well_stiffness`, `store_energy`) |
| `orient` | `scjdo2.orient.orient`, `OrientConfig` |
| `figures` | `scjdo2.landscape.figures`, `scjdo2.plot` (`figure_*`, `basin_index_by_type`, `spectrum_ladder`, `soft_axis_map`, `figure_orientation`) |
| `run` | all of the above + `scjdo2.io.save` / `load`; `time_key=` selects the flow route |

Also available: `scjdo2.plot.plot_drivers` (gene drivers, needs the rate model),
`figure_archetype_sheet`, `scjdo2.clone` (lineage checks of the drift),
`scjdo2.measured_drift.MeasuredDrift`.

## Things to know

* **One operator per AnnData.** `flow` and `fit` both store their operator in
  `uns["scjdo2"]["J"]`, tagged by `uns["scjdo2"]["meta"]["operator"]`
  (`"flow"` or `"energy"`; `scjdo2.flow_operator.operator_kind(adata)` reads it).
  Each refuses to overwrite the other (pass `overwrite=True` to force it), and
  `analyse` refuses a flow operator. To compare both, fit them on
  `adata.copy()`.
* **Energy `J` is density curvature, not stability.** In synthetic systems with
  a known landscape, the fitted energy operator recovered the multi-well
  landscape in a 2-d fit but not in 20-d, and degraded with longer training.
  Basin / fork / drift labels describe the curvature of the cell density;
  they are not attractors or relaxation rates.
* **Steady-state populations.** Flow operators or potentials fitted to
  single-window metabolic-labelling velocity did not recover per-cell
  stability in simulated steady states; `flow` is for sampled time courses.
* `J` is symmetric: it cannot show rotation or absolute time. Rotation and
  clock come only from measured flow (labelling, velocity) via `orient`.
* Flow and developmental directions answer different questions and can
  disagree; that is reported, not resolved.
* RNA velocity works where genes switch on/off and reach steady states and
  can reverse where transcription ramps smoothly; the reliability check
  demotes it when labelling disagrees.
* Palantir replaces the start cell you give it by the nearest boundary cell
  unless `use_early_cell_as_start=True`; store the cell it actually used.
* FA embeddings depend slightly on the scikit-learn version; record it.
