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
| `uns["flow"]` | parameters, per-pair OT diagnostics (`ot_pairs`), drift-fit record |

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

## Reproducing the paper

`scripts/appnote_validate_api.py` runs `sj.flow` on the held-out synthetic test sets and
LARRY and compares with the stored results; `notebooks/run_all.py` regenerates every
figure of the preprint (see `manuscript/REPRODUCE_v57.md`).
