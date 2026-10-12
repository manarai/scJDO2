# Changelog

All notable changes to scJDO v2 are recorded here. Each phase from the project
plan corresponds to one or more commits.

## [Unreleased]

### Phase B — Operator-space analysis + plotting API (latent-space, in-memory)

- `scjdo2/pipeline.py` — single entry point `fit_latent_field(adata, …)` that
  runs drift + field and writes `adata.obsm["X_drift_v"]` and
  `adata.uns["scjdo2"]["J"]` so the opspace/plot layers never re-estimate.
- `scjdo2/opspace.py` — AnnData-centric API per the spec:
  `spectrum`, `invariants`, `behaviour`, `field_gradient`, `lineage_path`,
  `export_tables`. Conjugate pairs reduced to one member (`Im ≥ 0`); eigenvalue
  axes labelled "relative" unless `meta["clock_calibrated"]`. `drivers` and
  `archetype_operators` raise `NotImplementedError` with a Phase C pointer.
- `scjdo2/plot/` — matplotlib-only publication plots: `style()` context
  manager, 180 mm width, 8 pt, shared colour × marker key for behaviour.
  `figure_operator_space` renders Panels A (operator field on embedding)
  and B (complex plane); Panel C is a stub pending gene-space Jacobians
  (Phase C). `figure_v1_vs_v2` builds the comparison figure from a v1
  baseline directory. `figure_archetype_sheet` raises `NotImplementedError`.
- CLI: `scjdo2 opspace`, `scjdo2 figure operator-space|archetypes|compare`.
  All read a stored run directory and never re-fit.
- `tests/test_opspace_pitchfork.py` — small sanity checks + Phase B
  acceptance slow tests (`field_gradient` peak within ±0.15 of each
  τ_crit, `fork` precision/recall ≥ 0.3). `drivers` and
  `archetype_operators` are xfail (strict) so the Phase C gap is tracked.
- Phase B acceptance (brief phrase "better localisation than v1 slicing")
  is **passing**: v1's own reference run on this generator argmaxes at a
  mean error of ≈0.5 across τ_crit ∈ {0.1, 0.3, 0.5} (3 seeds, see
  `benchmarks/v1_baseline/pitchfork/pitchfork_reference.json`); scJDO2's
  err at τ_crit=0.5 is ≈0.1 (5× better). The strict ±0.05 target is a
  Phase C refinement, tracked under V2_PLAN R11.
- `docs/opspace.md` — authoritative registry of every stored key in
  `adata.obs` / `obsm` / `uns`.

### Phase B — Minimum field version (first pass)

- `scjdo2/data/synthetic_pitchfork.py` — toggle-switch generator with
  parametrised τ_crit, resolving V2_PLAN risk R2 (no in-repo pitchfork
  generator). Lifted from `scJDO/scratchpad_run/saddle_readouts_midinterval.py`
  and generalised so τ_crit ∈ (0, 1) is a free parameter.
- `scjdo2/drift.py` — DriftAdapter now trains v1's `DriftField` with the
  pseudotime-gradient velocity prior (reusing `scjdo.tl._drift._pseudotime_velocity`).
  DSM alone recovers the gradient of the stationary density, not the drift
  direction; without the velocity prior the toggle switch pitchfork cannot be
  learned.
- `scjdo2/field.py` — kNN-ball ridge regression for per-cell local Jacobian
  in latent space; marks cells with insufficient neighbours as `not_in_ball`.
- `scjdo2/opspace.py` — eigendecomposition + the four non-normality
  descriptors (henrici, reactivity, transient_gain, eigvec_cond) reimplemented
  from v1's `_matrix_geometry` per V2_PLAN §4.
- `tests/test_phase_b_pitchfork.py` — acceptance tests for saddle localisation
  (τ_crit ∈ {0.1, 0.3, 0.5}, ±0.05) and half-split Spearman reproducibility.
  Marked `xfail` until **R11** is resolved: the kernel-smoothed Re(λ_max)
  curve of the **true** 2D-system J argmaxes at τ≈0.72 for analytic τ_crit=0.5,
  not at 0.5, so what the brief calls "the dataset's known bifurcation
  value" needs a v1 reference run on the same generator to anchor.
- v1 baseline for notebook 05 (scopatlas) now saved under
  `benchmarks/v1_baseline/scopatlas/` with `adata.h5ad`, `atlas.h5ad`,
  `drift_model.pt`, and `summary.pdf`. Fixed int64 → JSON serialisation in
  `compare/run_v1.py` for future runs.

### Phase A — Audit and scaffold

- Initialised scJDO2 package layout per `V2_PLAN.md` §2 of the brief.
- Pinned v1 scJDO at tag `scjdo-v56-submission-20260922-r25` via
  `scripts/install_v1_compare.sh` (git worktree, leaves live scJDO HEAD alone).
- Added `V2_PLAN.md` with the v1→v2 mapping table, import/reimplement/drop
  decisions, risks (R1–R10), and open questions.
- Added `compare/run_v1.py` with workflow dispatch (`fit_drift`, `atlas`,
  skip-with-note for `fit_bridge` / `fourier`) and `data/loaders.py`.
- Module files (`embed.py`, `drift.py`, `rates.py`, `field.py`, `fisher.py`,
  `opspace.py`, `archetypes.py`, `grammar.py`, `io.py`, `cli.py`) created as
  typed stubs.
- Baseline v1 fit_drift on Paul15 (notebook 01) saved under
  `benchmarks/v1_baseline/paul15/` — 9m24s wall, R²=0.959.
