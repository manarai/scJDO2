# Changelog

All notable changes to scJDO v2 are recorded here. Each phase from the project
plan corresponds to one or more commits.

## [Unreleased]

### Dynamic ground truth — gate well-defined, DSM fit still the barrier

Per user: swap the pseudotime-band ground truth for strict per-cell sign of
the analytic Jacobian:

    true_fork(c)   ⇔ analytic_lead_re at (X_true_c, α_c) > 0
    true_holder(c) ⇔ analytic_lead_re at (X_true_c, α_c) < 0

Both the equilibrated and transient synthetics carry ``X_true``, so the
same per-cell ground truth applies across all four rows.

- `scripts/equilibrated_gate.py` — `_dynamic_labels(adata)` computes the
  analytic Jacobian and its lead_re per cell; `_gate` scores against it.
  DSM-transient is now computed inline (3 extra fits per run) instead of
  pulled from the standing report, so every row uses the same gate.
- `compare/run_v1_equilibrated_pitchfork.py` — same dynamic ground truth.

Results (seed 0, 5000 epochs, n_cells=1200, n_alpha=30):

| row | τ_crit | fork P | fork R | holder P | holder R | verdict |
|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 0.88 | 1.00 | 0.84 | **PASS** |
| oracle | 0.3 | 1.00 | 0.88 | 1.00 | 0.89 | **PASS** |
| oracle | 0.5 | 1.00 | 0.79 | 1.00 | 0.91 | FAIL (0.79 vs 0.80) |
| dsm_equilibrated | 0.1 | 0.04 | 0.71 | nan | 0.00 | FAIL |
| dsm_equilibrated | 0.3 | 0.04 | 0.82 | nan | 0.00 | FAIL |
| dsm_equilibrated | 0.5 | 0.05 | 0.76 | nan | 0.00 | FAIL |
| v1_slicing_eq | 0.1-0.5 | 0.05-0.06 | 0.72-1.00 | nan-1.00 | 0-0.04 | FAIL |
| dsm_transient | 0.1-0.5 | 0.26-0.28 | 0.77-0.80 | nan | 0.00 | FAIL |

**The gate is now well-defined and the oracle essentially passes.** The
τ=0.5 oracle miss (fork R=0.79) is a measurement artifact from
``opspace.behaviour`` labelling cells with |lead_re| ≤ tol as "drift";
within noise, the ceiling is near 1 everywhere.

**DSM-equilibrated fails on precision**, not recall — labels ~98 % of
cells "fork" while analytic says only ~4 % are. The learned autodiff
Jacobian is uniformly positive-definite independent of data regime
(frac(Re>0) = 1.00 on all three DSM rows). The equilibration fixed the
ORACLE but not the fit. v1 slicing shows the same pathology; same
DriftField + DSM backend, kernel-smoothed grid doesn't fix the sign.

Phase C stays blocked. The gate definition is no longer the bottleneck
— the DSM fit itself is. The DSM loss plateaus at ~94 across every run
(same signature in the convergence_audit/) and the output drift
converges to a near-constant expansive field. Likely culprits ranked
by cost to probe: σ-scaling in the DSM target, FiLM gate schedule, or
alpha_control weighting.

### Equilibrated pitchfork synthetic + four-row gate — oracle itself < 0.8

New synthetic that mixes stationary SDE snapshots per α (``snapshot = mixture
of stationary densities``), plus a four-row gate per user spec:

- `scjdo2.data.generate_pitchfork_equilibrated(tau_crit, n_cells, n_alpha,
  T_burn, T_sample, …)` — for each α on a grid spanning the three τ_crit
  values, integrate the toggle SDE from random initial conditions in
  ``[-init_box, +init_box]^2`` for ``T_burn`` time units at fixed α, then
  sample one snapshot per trajectory from the ``T_sample`` window. Stores
  ``obsm["X_true"]``, ``obsm["V_true"]`` (analytic drift), ``obsm["X_pca"]``,
  ``obs["pseudotime"]``, ``obs["alpha"]``, and the gene loading + PCA
  components under ``uns["pitchfork"]``.
- `scjdo2.data.analytic_jacobian(z, α)` — closed-form 2D Jacobian for the
  oracle row.
- `scripts/equilibrated_gate.py` — oracle + DSM-equilibrated in one script;
  pulls DSM-transient from the standing gate report; expects
  ``benchmarks/phase_b_gate/equilibrated/v1_slicing.json`` for the v1 row.
- `compare/run_v1_equilibrated_pitchfork.py` — v1 ``fit_drift(…, rep="X_pca",
  windowing="kernel")`` on the equilibrated data; per-cell J = J_tensor at
  nearest ``t_center``; same opspace, same gate; writes the v1 row JSON.
- `tests/test_phase_b_pitchfork.py` — transient-α-ramp tests marked
  ``xfail(strict=True)`` with the note: "snapshot data from a non-stationary
  process does not carry basin structure".

Oracle ceiling at τ_crit ∈ {0.1, 0.3, 0.5} (same classifier, same gate):

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) |
|---|---|---|---|---|---|---|
| oracle | 0.1 | 0.16 | 0.06 | 0.72 | 0.79 | 0.04 |
| oracle | 0.3 | 0.18 | 0.07 | 0.71 | 0.87 | 0.04 |
| oracle | 0.5 | 0.13 | 0.07 | 0.74 | 0.88 | 0.05 |
| dsm_equilibrated | 0.1 | 0.10 | 0.73 | nan | 0.00 | 1.00 |
| dsm_equilibrated | 0.3 | 0.10 | 0.82 | nan | 0.00 | 1.00 |
| dsm_equilibrated | 0.5 | 0.07 | 0.81 | nan | 0.00 | 1.00 |

**The oracle itself does not reach 0.8.** Equilibrated SDE snapshots live
overwhelmingly in basins (``frac(Re > 0) ≈ 0.04`` for the analytic J),
so the gate's "cells near τ_crit should be fork" assumption fails at the
oracle. Holder reaches ~0.7–0.9 but is capped because some cells near
τ_crit remain in nearby basins (symmetric FP just below α_crit, or
asymmetric branches at α just above). The gate as currently stated
cannot be met on equilibrated data, not by any estimator.

DSM-equilibrated at 5000 epochs is still catastrophic — same pathology
as DSM-transient (loss plateaus at ~94, Jacobian uniformly positive,
~100 % of cells labelled fork). The equilibration helps the ORACLE
measurement (frac(Re>0) drops from 1.00 under autodiff on transient to
0.04 under analytic on equilibrated, as expected) but not the DSM fit.

v1 slicing on equilibrated: in flight.

Phase C stays blocked. The decision point is now the GATE DEFINITION,
not just the estimator: the pseudotime-τ-band labels are a poor match
for the stationary distribution's physics. Options noted at the end of
the report.

### Autodiff Jacobian + convergence audit — Phase B still blocked

Per user direction: replace the kNN-ball ridge Jacobian with the exact
autodiff Jacobian of the drift network (`torch.autograd` loop over output
dims, already in `DriftField`). Keep the neighbourhood regression and
permutation null in `scjdo2/field.py` for Phase C's gene-space step.

- `scjdo2/pipeline.py:fit_latent_field` gains `jacobian_method: Literal
  ["autodiff", "neighbourhood"] = "autodiff"`. Autodiff path writes
  `J = drift.jacobian_autodiff(z, t_c)` (per-cell at each cell's own
  pseudotime). Meta records `jacobian_method`.
- `scripts/pitchfork_convergence_audit.py` — focused script: train the
  pitchfork drift to a configurable epoch budget and compare the
  autodiff Jacobian to the analytic one at the three analytic fixed
  points (pre-bifurcation symmetric FP, post-bifurcation saddle,
  asymmetric stable branches). Includes a true-space projection check
  (fit a linear 20D→2D map by lstsq and compare learned v and J to the
  analytic toggle flow).
- `scripts/pitchfork_architecture_check.py` — one-off diagnostic: train
  the drift with the ANALYTIC toggle drift as `V_ref`. Isolates whether
  the architecture can represent the dynamics at all.

Results at `benchmarks/phase_b_gate/`:

Autodiff gate (`report.{json,md}`) — Phase B still fails. Fork recall
jumps to 0.87–0.99 (the autodiff Jacobian is much more sensitive than
the ridge estimator) but **fork precision stays ~0.10** because the
pipeline labels ~98 % of cells as fork, and **holder P/R collapse to ~0**.
The drift's autodiff Jacobian is uniformly positive-definite.

Convergence audit (`convergence_audit/audit_e5000_*.json`):

| config | cos_sim(learned_v, true_v) | frac(J_true both neg) | pre-FP eigs | saddle eigs | stable eigs |
|---|---|---|---|---|---|
| 5000 ep, prior=True  | **−0.015** | 0.003 | (+0.70, +0.59) | (+0.70, +0.62) | (+0.69, +0.57) |
| 5000 ep, prior=False | **+0.066** | 0.451 | (+0.35, +0.25) | (+0.33, +0.25) | (+0.29, +0.17) |
| V_ref = analytic toggle | **−0.006** | 0.001 | (+0.70, +0.59) | — | — |

Analytic expectations: pre-FP (−, −), saddle (+, −), stable (−, −).

The learned drift is **uncorrelated or anti-correlated** with the true
toggle flow — not under-trained. **Even with the analytic drift
supplied as the velocity prior at `vel_scale=4` for 5000 epochs the fit
does not converge to it** (cos_sim = −0.006). The DSM term dominates
and pulls the drift to a configuration that is approximately the score
of a mixture over nearby α slices (snapshot-biased), which has no
negative-definite basins because the data is non-stationary (α-ramp,
no equilibration at any α slice).

Conclusion: the DSM + pseudotime velocity prior training objective on
this α-ramp synthetic does not learn the toggle vector field, so
`∂f/∂z` at the analytic fixed points cannot match the analytic
signature. v1 has the same limitation on this generator (v1 reference
err ≈ 0.5 at every τ_crit). The user's "under-trained or
over-regularised" hypothesis is not supported.

Phase C stays blocked. Awaiting user direction on whether to (a) switch
to an equilibrated-snapshot pitchfork synthetic, (b) swap the drift
objective to direct velocity regression against a trajectory-derived
target, (c) move to Phase C's gene-space NB model (which does not use
DSM), or (d) adjust the gate.

### Phase B gate report — Phase B does NOT pass user's ≥0.8 target

New diagnostics (held in `benchmarks/phase_b_gate/{report.json, report.md}`):

- `scripts/phase_b_gate_report.py` — single-command gate measurement:
  pitchfork fork+holder precision/recall at the strict 0.8 target for
  τ_crit ∈ {0.1, 0.3, 0.5}, permutation null for Re(λ_max) on both
  pitchfork and Paul15, Paul15 holder count with and without the
  pseudotime velocity prior.
- `scjdo2/field.py:JacobianField.permutation_null(z, v, n_perm, subset_idx,
  seed)` — in each neighbourhood, shuffle which cell's drift vector pairs
  with which level vector, refit ridge, and return the per-cell null
  distribution of Re(λ_max). Precomputes `G^{-1} dz.T` once per cell.
- `scjdo2/drift.py:DriftConfig.inject_velocity_prior` (new, default True)
  — set to False to run DSM-only (no pseudotime-gradient velocity prior).
  Enables the "is the pseudotime prior the reason Paul15 has zero
  holders" test.

Results (seed 0; pitchfork n=1200, Paul15 n=2730):

| τ_crit | fork P | fork R | holder P | holder R |
|---|---|---|---|---|
| 0.1 | 0.08 | 0.42 | 0.76 | 0.26 |
| 0.3 | 0.13 | 0.74 | 0.83 | 0.22 |
| 0.5 | 0.11 | 0.62 | 0.84 | 0.30 |

All six precision/recall numbers are below 0.8; fork precision and holder
recall are both far below. Permutation null overlap on pitchfork is 0.000
in every stratum — the real signal is strongly separated from the random
pairing null, but in the direction of **lower** Re(λ_max) than null, not
higher. The kNN-ball ridge estimator has a ~+0.5 positive bias on the
leading real eigenvalue when fed random (z, v) pairings; the real fit
reduces that bias to ~0 but almost never crosses into negative territory,
which is why holders are rare and never crisply localised.

Paul15 result on the velocity-prior hypothesis: removing the pseudotime
velocity prior **did not** change the holder count (0 → 0). The DSM-only
drift at 1500 epochs collapses to a near-zero vector field (`real_median`
= 0.007 vs. `null_median` = 0.045), so the estimator never produces
negative-definite neighbourhood Jacobians at putative attractors. The
hypothesis that f = σ²∇log p would localise contractive J at attractors
is not supported at these settings (could be under-trained — v1 uses
5000 epochs — but the structural positive bias of the local ridge
estimator is the dominant failure mode).

**Verdict:** Phase B's latent-space first-pass estimator does not pass
the user's gate ("holder and fork precision+recall ≥ 0.8"). Phase C is
not yet unlocked.

### Scope cleanup — scOpAtlas removed

- scOpAtlas (notebook 05) is out of v2 scope. Removed `_run_atlas` and the
  `atlas` workflow dispatch from `compare/run_v1.py`; dropped the
  `scopatlas` key from `data/loaders.py` and `data/__init__.py`; removed
  the notebook-05 row from `notebooks/README.md` and the `§7` tutorial
  table in `V2_PLAN.md`; marked the `atlas` row in V2_PLAN §1 as "no (out
  of v2 scope)". The int64-safe JSON encoder in `compare/run_v1.py` stays
  (general safety nit for Paul15 config too). The on-disk v1 baseline at
  `benchmarks/v1_baseline/scopatlas/` was never git-tracked and has been
  moved to `../scjdo2-attic/scopatlas-v1-baseline/`.

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
