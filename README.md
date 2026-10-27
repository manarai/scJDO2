# scJDO v2

Per-cell local operators for single-cell data. scJDO2 fits a drift field on a
latent embedding of the cells, takes its Jacobian `J_c` at every cell (an
*operator field*), and reads cell behaviour from the operator's spectrum:
**basin** (holder), **fork** or **drift**. A post-hoc **orientation layer**
then gives the operator directions from measured evidence (labelling,
lineage, RNA velocity, pseudotime), without changing it.

A successor to [scJDO](https://github.com/manarai/scJDO). The package does
not import v1; v1 is used only by the separate comparison harness in
`compare/` (see [docs/V1_REMOVAL.md](docs/V1_REMOVAL.md)).

## Flow operators from a time course in one call

```python
import scjdo2 as sj
sj.flow(adata, time_key="day")     # OT velocity -> drift -> J_flow per cell
adata.obs["flow_lambda1"]           # leading eigenvalue; adata.uns["scjdo2"]["J"] = operators
```

Quickstart with test data: `python examples/flow_quickstart.py`. Parameters,
outputs and usage notes: [docs/flow.md](docs/flow.md).

## How it works

1. **Embedding** (`scjdo2.embed`): latent positions `z_c` from the count
   matrix via scVI, factor analysis or PCA. Encoder frozen after the fit.
2. **Energy drift** (`scjdo2.energy_drift.EnergyDriftAdapter`): a learned
   scalar energy `E(z)` fitted by Hyvärinen exact score matching (default
   `loss="esm"`, exact Laplacian up to d = 20, Hutchinson probes above).
   The drift is the score `f(z) = −β ∇E(z)`.
3. **Operator field** (`scjdo2.pipeline.fit_latent_field`): `J_c = ∂f/∂z`
   at every cell by autodiff (default; a kNN ridge regression is
   available). `J = −β Hess E` is **symmetric by construction**: it
   describes the curvature of the cell density, has real eigenvalues, and
   cannot carry rotation. Stored in `adata.uns["scjdo2"]["J"]`.
4. **Behaviour** (`scjdo2.opspace`): invariants per cell (eigenvalues,
   spectral gap, Henrici index, transient gain) and the basin classifier
   `classify_basin_latent`. Its default rule, `"lead_eig"`, uses the sign of
   the leading eigenvalue against an oracle tolerance; the tangent-restricted
   `"magnitude"` rule is kept as an option. Behaviour is read from the
   **latent** operator (rule R15 in [V2_PLAN.md](V2_PLAN.md)).
5. **Gene level (optional, `--with-rate-model`)**: an NB rate model on raw
   counts with a smooth decoder (`scjdo2.rates`), the gene-space Jacobian
   (`scjdo2.field`) and a per-entry Fisher confidence ledger
   (`scjdo2.fisher`). Used for **driver genes** (the latent leading mode
   lifted to genes) and uncertainty, not for behaviour.
6. **Archetypes and grammar** (`scjdo2.archetypes`, `scjdo2.grammar`):
   semi-NMF archetypes of the operator field (reimplemented, no v1 import),
   named by eigenvalue signature; archetype sequences along the drift.

### Orientation layer (`scjdo2.orient`)

`f = −∇E` points toward higher density, not along development, so the
direction of each eigen-axis of `J` comes from separate evidence.
`orient(adata, sources, embedding_key)` returns two outputs that are never
merged with each other ([docs/ORIENTATION_SPEC.md](docs/ORIENTATION_SPEC.md)):

| Output | Question | Sources, by trust |
|---|---|---|
| **flow** | how the cell moves now | metabolic labelling > RNA velocity |
| **developmental** | where development goes over days | lineage barcodes > pseudotime |

* Within each output, the highest-ranked source with support decides each
  (cell, axis) sign; disagreements are recorded as conflicts, never
  blended. Disagreement between the two outputs is reported
  (`obsm["orient_flow_vs_dev"]`).
* Flow also gives magnitude (labelling clock) and rotation `omega` (the
  antisymmetric part of a local regression of measured velocity).
* **Reliability check** (§15): velocity that agrees with labelling below
  0.6 on the pairs both sign is demoted to a cross-check for that dataset.
* **Pseudotime guards** (§14, §16): the start cell must sit at the low end
  of the pseudotime, and pseudotime must increase with sampling day.
* Equilibrium and stationarity flags. `J` is read, never modified (hash
  checked).

Velocity and pseudotime are the user's own scVelo and Palantir runs; neither
is a dependency.

### Measured inputs

* `scjdo2.labelled_rates.LabelledRateModel`: joint NB fit on new and total
  counts with per-cell pulse length, giving per-gene degradation rates.
* `scjdo2.measured_drift.MeasuredDrift`: a smooth drift regressed on a
  measured velocity field.
* `scjdo2.clone`: lineage-barcode measurements of the drift.

## Synthetic data with ground truth (`scjdo2.data`)

* `synthetic_pitchfork`, `synthetic_oriented`: analytic test systems.
* `synthetic_multimodal.make_multimodal(preset, seed)`: one dynamical
  system producing lineage barcodes, labelling (new/total), spliced and
  unspliced counts, sampling days and inputs for pseudotime, with exact
  truth. Presets `default`, `stress`, `batch`, `stationary_cycle` (v1) and
  `v2_*` (realistic unspliced share, switching genes, overdispersed counts,
  sparse lineage sampling). See
  [docs/SYNTHETIC_MULTIMODAL.md](docs/SYNTHETIC_MULTIMODAL.md).

Loaders for real data: `pbmc`, `larry`, `setty_cd34`, `dynamo_hematopoiesis`.

## Install

```bash
cd scJDO2
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

For v1-vs-v2 comparison runs only, install v1 at its preprint tag with
`bash scripts/install_v1_compare.sh` (a separate worktree, so the live v1
repository is not moved).

## Use

```bash
scjdo2 run      --input data.h5ad --out runs/x --use-rep X_pca [--with-rate-model]
scjdo2 opspace  --input runs/x --out runs/x/opspace
scjdo2 figure   operator-space|archetypes|ledger --input runs/x --out figs/
scjdo2 compare  --v1 <dir> --v2 <dir> --out <dir>
```

```python
from scjdo2.pipeline import fit_latent_field
from scjdo2.orient import orient, OrientConfig

fit_latent_field(adata, use_rep="X_pca")            # energy drift + J
out = orient(adata, ["labelling", "lineage"], "X_pca",
             OrientConfig(labelling_key="velocity_labelling"))
```

## Status

* Energy drift and latent basin classifier locked (SUMMARY §5.7).
* Orientation layer implemented and tested on synthetic data with known
  truth (`coord/orient/`, findings in `coord/orient/FINDINGS.md`); the
  end-to-end evaluation on the v2 synthetic data is in progress.
* Open: fork detection at saddles is limited by neighbourhood width;
  real-data orientation not yet run.

History: [CHANGELOG.md](CHANGELOG.md), [SUMMARY.md](SUMMARY.md),
[V2_PLAN.md](V2_PLAN.md).

## What this is not

* Not a port of v1's Schrödinger-bridge, regulatory-inference or Koopman
  pipelines (only the Henrici index and transient gain are reimplemented,
  as formulas in `opspace.py`).
* Not a source of rotation or absolute timescale from a snapshot: the
  score-matched `J` is symmetric; rotation and clock come only from
  measured flow (labelling, velocity) through the orientation layer.
* scOpAlign (batch correction by operator-space alignment) is a separate
  tool.
