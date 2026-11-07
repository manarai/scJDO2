# `scjdo2.flow` — flow operators from a sampled time course

`scjdo2.flow(adata, time_key=...)` estimates, for every cell, the **flow operator**
*J_flow = ∂f/∂z*: the Jacobian of a smooth drift *f* fitted to a velocity obtained from
the sampling times alone by entropic optimal transport (OT). No labels, splicing or
lineage barcodes are needed.

## Install

```bash
git clone https://github.com/manarai/scJDO2.git
cd scJDO2
pip install -e .            # Python >= 3.10; installs torch, POT, anndata, scanpy
```

## Quickstart (test data included)

```bash
python examples/flow_quickstart.py
```

This simulates a pitchfork time course (2,000 cells, five times), runs `sj.flow`, and
prints the leading eigenvalue at the branch point and in committed cells (AUC 1.00).
It takes a few seconds on a laptop CPU.

On your own data:

```python
import scanpy as sc
import scjdo2 as sj

adata = sc.read_h5ad("timecourse.h5ad")        # adata.obs["day"]: numeric sampling time
sc.pp.normalize_total(adata); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.tl.pca(adata, n_comps=15)

sj.flow(adata, time_key="day")                 # defaults = manuscript reference configuration

adata.obs["flow_lambda1"]                       # leading eigenvalue (max Re) per cell
adata.uns["scjdo2"]["J"]                        # (n_cells, 15, 15) flow operators
```

## What it does

1. **OT velocity** (`scjdo2.ot_velocity`). Balanced entropic OT (Sinkhorn, POT) between
   the cells of consecutive sampling times, with regularisation
   ε = `eps` × median squared distance of the pair. Each cell of the earlier time gets the
   barycentric velocity (barycentre of its image − z) / Δt. Cells of the last time get no
   velocity. Pairs with more than 64 × 10⁶ cell pairs use mini-batch OT (4,000-cell
   batches, each source cell in ten batches).
2. **Drift.** A multilayer perceptron *f(z)* (`MeasuredDrift`) is fitted to the velocities
   by mean-squared error and evaluated at every cell (the last time point by
   extrapolation).
3. **Flow operator.** *J_flow(z) = ∂f/∂z* by automatic differentiation (batched
   `torch.func.jacrev`).
4. **Descriptors** (`scjdo2.flow_descriptors`). Eight magnitude-free descriptors per
   cell, rank-transformed within the dataset (see *Invariance* below).

## Parameters

| parameter | default | meaning |
|---|---|---|
| `time_key` | — | `obs` column with a numeric sampling time; velocities are per unit of it |
| `use_rep` | `"X_pca"` | `obsm` key of the transport space; computed with `scanpy.tl.pca` if missing |
| `n_comps` | 15 | leading components used (10–20 tested); `None` = all |
| `eps` | 0.05 | entropic regularisation, fraction of the median squared distance |
| `solver` | `"auto"` | `"dense"`, `"minibatch"` or `"auto"` (mini-batch above 64 × 10⁶ pairs) |
| `hidden` | (96, 96, 96) | drift MLP hidden layers |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `n_epochs` | 1500 | drift optimisation steps (mini-batches of 256 cells) |
| `n_seeds` | 1 | > 1 averages Jacobians over fit seeds |
| `seed` | 0 | random seed (OT mini-batches and drift fit) |
| `device` | auto | `"cpu"`, `"cuda"` or `"mps"` for the drift fit (Jacobians on CPU) |
| `descriptors` | True | compute operator-space descriptors |
| `key_added` | `"flow"` | prefix of output keys |
| `copy` | False | return a modified copy |

## Outputs

| key | content |
|---|---|
| `obs["flow_lambda1"]` | leading eigenvalue (max real part) of *J_flow* |
| `obs["flow_has_velocity"]` | cell received an OT velocity (False on the last time) |
| `obsm["flow_velocity"]` | OT velocity (NaN on the last time point) |
| `obsm["flow_drift"]` | fitted drift *f(z)* at every cell |
| `obsm["flow_descriptors"]`, `obsm["flow_descriptor_ranks"]` | raw and rank-transformed descriptors |
| `uns["scjdo2"]["J"]` | *J_flow*, (n, d, d) float32; read by `scjdo2.opspace` |
| `obsp["flow_plan"]` | transport plan: row *i* = destination distribution of cell *i* over the next sampling time, its `keep_plan` (default 100) most likely destinations, renormalised (a truncation: on a 3,000-cell LARRY subsample the top 100 held a median 63% of each row's mass) |
| `uns["flow"]` | parameters, per-pair OT diagnostics (`ot_pairs`, incl. the plan's kept mass), drift-fit record, drift weights |

All outputs can be written with `adata.write_h5ad`.

## Invariance of the descriptors

Six descriptors are functions of the eigenvalues (sign of Re λ₁, number of eigenvalues
with positive real part, Re λ₁/max|λ|, rotational index max|Im λ|/max|λ|, spectral
gap/max|λ|, trace/Σ|λ|) and are invariant under any invertible linear change of the
embedding coordinates. Henrici non-normality and transient gain are invariant only under
orthogonal changes. Different embeddings (PCA, scVI, Factor Analysis) are **not** related
by such maps, so compare descriptors only within one embedding.

## How to use the result — and how not to

* **Use** λ₁ to compare groups of cells: on held-out synthetic data, λ₁ ranked fate-bifurcation
  cells above committed cells (AUC 0.89–1.00 with seed 0; 0.82–1.00 over seeds 0–2) with these defaults.
* **Do not** read the matrix entries as a regulatory Jacobian: *J_flow* does not recover
  the values of the true Jacobian (median relative error ≈ 1).
* **Use unwhitened PCA scores** (PCA of log-normalised counts, 10–20 components).
  The operator inverts on Factor Analysis and degrades in whitened or high-dimensional
  spaces.
* **Sampling density matters**: three sampling times were too coarse on an independent
  simulator; eight were sufficient.
* The clock is the unit of `time_key`; λ₁ values are comparable within a dataset only.
* **Balanced transport**: proliferation and death are not modelled; changes in composition
  between sampling times are treated as movement of cells.
* **Training length**: the default `n_epochs = 1500` (optimizer steps) is the evaluated setting.
  At 15,000 and 50,000 steps the LARRY results held (lineage Spearman 0.55 and 0.54; held-out
  energy distance 0.75 and 0.73, still below both baselines), but the minimum fork AUC fell to
  0.86 and 0.78.
* **Not for steady-state labeling data**: in simulated steady-state populations, flow operators
  fitted to single-window metabolic-labeling velocity did not recover per-cell stability, with
  or without kNN pooling of the velocity, and a potential fitted to that velocity was no better
  than a density estimate (and unstable under uneven sampling).
* **Fit-to-fit variation**: reported intervals are conditional on one fitted drift (`seed = 0`).
  Refitting with seeds 1 and 2 kept the group-level results (fork AUC 0.82–1.00 on the 11
  held-out datasets; LARRY lineage Spearman 0.55–0.56; held-out energy 0.69–0.75, below both
  baselines), but per-cell λ₁ rankings agreed only moderately between fits (median Spearman
  0.56, minimum −0.04). Compare per-cell values across refits (`seed = 1, 2, …`) before
  interpreting individual cells.

### Group summary and refit check

```python
tab = sj.flow_groups(adata, "cell_type")                     # median λ₁ per group, 95% bootstrap CI
st = sj.flow_stability(adata, seeds=(0, 1, 2), groupby="cell_type")
st["per_cell_spearman_median"], st["group_spearman_min"], st["top_group_same"]
```

`flow_groups` reports, per group, the number of cells, the fraction whose operator is
extrapolated (last sampling time), the median λ₁ with a bootstrap interval over cells (conditional
on the fitted operator) and its rank. `flow_stability` refits transport and drift once per seed
with the stored run's parameters (the stored operator is not changed) and reports the Spearman
agreement of per-cell λ₁ and of group medians between refits; it also writes
`obs["flow_lambda1_rank_sd"]`. `sj.run(adata, out_dir, cluster_key=..., time_key=...,
stability_seeds=(0, 1, 2))` runs `flow`, both checks and the `flow_groups` figure and saves the
run. Running `scjdo2.fit` (energy operator) on the same AnnData is refused, because both
operators use `uns["scjdo2"]["J"]`; use `adata.copy()`.

## Operator motifs along the time course (`scjdo2.flow_motifs`)

*J_flow* gives each cell an operator but no order: the operator of a day-2 cell and
those of the day-4 cells it becomes are not linked by *J* itself. The link is the OT
plan, which `sj.flow` now keeps. Archetypes (operator shapes) chained along the plan give
operator **sequences**:

```python
sj.flow_archetypes(adata)                 # K shapes of J_flow, obs["flow_archetype"] = A1..AK
tab = sj.flow_transitions(adata)          # archetype -> archetype per time pair, vs a shuffle null
paths = sj.flow_paths(adata)              # sampled chains through the plan vs the Markov prediction
res = sj.flow_integrate(adata, start=np.flatnonzero(adata.obs["day"] == 2)[:20])   # drift paths

from scjdo2.plot import figure_flow_archetypes, figure_flow_transitions
figure_flow_archetypes(adata, out="archetypes.png")     # eigenvalues + matrix per archetype
figure_flow_transitions(adata, out="transitions.png")   # P(to | from) per time pair
```

* **`flow_archetypes`**: semi-NMF of the per-cell operators (`scjdo2.archetypes`), each
  operator first divided by its Frobenius norm so archetypes are shapes, not speeds
  (`normalize=None` keeps magnitudes); *K* by held-out reconstruction error. Fitted on up to
  4,000 cells, then every cell is assigned by non-negative least squares. Writes
  `obsm["flow_archetype_weights"]`, `obs["flow_archetype"]` (largest weight; A1 the most
  common), `obs["flow_archetype_share"]` and `uns["flow"]["archetypes"]`.
* **`flow_transitions`** (the default readout): for each pair of consecutive times, the mass
  that cells labelled *k* send to cells labelled *l*, `prob = P(l | k)`, against a null that
  permutes the source labels within the source time (label sizes and target composition
  kept). It re-runs the transport of `sj.flow` with the same settings (checked against the
  stored velocities) and contracts the **full** plan with the labels, so nothing is truncated;
  `exact=False` uses the stored top-k plan instead. Any `obs` column can be the labels
  (`labels="cell_type"`).
* **`flow_paths`**: chains sampled through the stored (truncated) plan from the first time to
  the last; counts of label sequences, with `obs_over_expected` against a Markov chain on the
  same plan. Values far from 1 point to memory or rare routes the matrices average away.
  Because the stored plan keeps only each cell's nearest destinations, read path frequencies
  as approximate; the transition matrices are the exact readout.
* **`flow_integrate`**: RK4 paths of the fitted drift (`scjdo2.flow_operator.load_drift`
  rebuilds it from `uns["flow"]["drift_weights"]`), with λ₁ and the archetype weights along
  each path. A deterministic path cannot branch: use it as a picture of the mean flow.

**Status and caveats.** This readout is new and **not yet benchmarked**: in particular,
archetype shapes have not yet been shown to be stable across biological replicates, and
there is no cross-dataset frame yet. Read it within one dataset and check stability before
interpreting a motif. Targets at the last sampling time have extrapolated operators
(`to_extrapolated`). Balanced transport moves any change in composition between times as
cells changing state; with mini-batch transport, batch-to-batch composition noise adds a
small leak between labels (about 1% at the default 4,000-cell batches). The p-values are
conditional on one fitted plan and drift; they do not include refit variation.

## Reproducing the paper

`scripts/appnote_validate_api.py` runs `sj.flow` on the held-out synthetic test sets and
LARRY and compares with the stored results; `notebooks/run_all.py` regenerates every
figure of the preprint (see `manuscript/REPRODUCE_v57.md`).
