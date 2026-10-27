# scJDO2 orientation layer — revised spec (draft for approval)

## 0. Scope and invariants

* Post-hoc layer. It never enters the training of `E_θ` and never modifies the
  operator `J = adata.uns["scjdo2"]["J"]` (or `"J_latent"`): a byte hash of `J`
  before and after is stored and must match in every run and test.
* No scJDO v1 drift code is imported, called or ported. scJDO2 has one drift,
  `f = −∇E_θ`, fitted by score matching (the locked default is exact score
  matching; DSM remains an option).
* Why the layer exists (methods text): with `E = −log p`, `f = ∇log p` points
  toward higher density — toward cell-type centres — and is zero at a basin's
  centre. It says nothing about which way development runs. Direction, speed and
  rotation need separate evidence; this layer supplies it and records where it
  came from.
* `J` is symmetric (minus the Hessian of `E`), so it has real eigenvalues,
  orthogonal eigenvectors and no rotation. Rotation (`omega`) is **not** a
  property of `J`: it is the antisymmetric part of a separately estimated
  Jacobian of *measured* velocity.
* Embedding-agnostic. The layer works in whatever `embedding_key` it is given.
  scJDO2's embedding layer is out of scope here.

## 1. API

`scjdo2.orient.orient(adata, sources=[...], embedding_key, n_axes=5, config=...)`,
`sources` ⊆ {"lineage", "labelling", "velocity", "pseudotime"}, any subset.
Optional CLI: `scjdo2 run --orientation lineage,pseudotime`.

Each source implements
`orient(adata, J, embedding_key, axes) -> SourceResult` with
`sign (n, k) ∈ {+1, −1, 0}`, `magnitude (n, k)` or None, `omega (n, d, d)` or
None, `support (n, k) bool`, `meta` (dict).

Modules: `scjdo2/orient/{base, axes, lineage, labelling, velocity, pseudotime, merge, flags}.py`.

## 2. Axes

* Per cell, `eigh` of the symmetrised `J[i]`; axes ordered by eigenvalue,
  largest first (softest axis = axis 1); `k = n_axes` (default 5, ≤ d).
* **Sign convention:** each eigenvector is flipped so that its largest-magnitude
  component is positive (deterministic, reproducible).
* **Gap rule:** axis `j` at cell `i` is usable only if its relative eigen-gap
  `min(|λ_j − λ_{j−1}|, |λ_j − λ_{j+1}|) / median|λ|` ≥ `gap_min` (default 0.1).
  Otherwise every source gets `support = False` there and the ledger says
  `degenerate`.

## 3. Sign definition

For a direction vector `v_i` supplied by a source at cell `i`:
`sign_ij = sign(v_i · u_ij)`. Support requires both
* `|v_i · u_ij| / ‖v_i‖ ≥ align_min` (default 0.2), and
* `‖v_i‖ ≥` the source's noise floor (default: its 10th-percentile norm), which
  removes basin-centre cells where velocity ≈ 0.

## 4. Sources (each contributes only what it measures)

| source | direction vector `v_i` | sign | magnitude | omega | support |
|---|---|---|---|---|---|
| lineage | later-day minus earlier-day centroid of the same clone (`clone.clone_cross_day_pairs`) | yes | no | no | cells in clones with cross-day siblings |
| labelling | measured latent velocity from the labelled-rate path (`labelled_rates.py` → `measured_drift.py`), per-hour clock | yes | `|v·u_j|` per hour (`clock_identified = True`) | antisymmetric part of a local ridge regression of `v` on `z` (`field.py` neighbourhood) | cells with labelled counts and pulse time |
| velocity | the user's **scVelo** run, read as `obsm["velocity_<embedding_key>"]` (e.g. from `scvelo.tl.velocity_embedding(basis=…)`), or gene-space velocity through a **user-supplied linear map** (with the per-cell chain rule for log-normalisation, `d log1p(x)/dt = v/(1+x)`); refused for a non-linear embedding without one of these; **never computed by scJDO2** | yes | no (`clock_identified = False`) | as labelling, no clock | cells with velocity |
| pseudotime | the user's **Palantir** run, read as an `obs` pseudotime column plus its start cell; sign from the kNN weighted least-squares gradient of that column in the embedding (k = 30); **never computed by scJDO2** | yes | no | 0 (`rotation_identified = False`) | all cells it reaches |

**External sources are inputs, not dependencies.** scVelo and Palantir are run
by the user, separately, with their own settings; scJDO2 reads their stored
outputs, records the key names and any version information present in `uns`,
and never re-runs or re-tunes them. Neither package is a scJDO2 dependency.

## 5. Merge

* **Sign, per (cell, axis):** the highest-ranked supporting source wins
  (`lineage > labelling > velocity > pseudotime`). If any lower-ranked
  supporting source disagrees, `conflict = True` for that (cell, axis) and the
  disagreeing sources are recorded. Observed fate beats inferred direction.
  Velocity is a consistency check wherever a higher source has support, and
  primary only where none does.
* **No blending.** Sources are never averaged or weighted together. Each
  (cell, axis, component) takes one source's value, and the ledger names it.
  Prior work composes direction sources by weighted kernel sums (CellRank's
  kernel composition: Lange et al., *Nat Methods* 2022; Weiler et al.,
  *Nat Methods* 2024). scJDO2 instead ranks the sources, takes the highest one
  with support, and flags every disagreement, so a measured direction is never
  diluted by an inferred one and every conflict stays visible.
* **Magnitude:** labelling only.
* **Omega, per cell:** labelling, else velocity (clock unidentified), else
  pseudotime (0, rotation unidentified), else none.

## 6. Outputs

* `obsm["orient_sign"]` (n, k, int8: +1, −1, 0 = unsupported);
  `obsm["orient_magnitude"]` (n, k, NaN where absent);
  `obsm["orient_sign_source"]` (n, k, code; codes in `uns`);
  `obsm["orient_conflict"]` (n, k, bool);
  `uns["scjdo2"]["omega"]` (n, d, d; NaN where none).
* Per-cell ledger in `obs`: `sign_source` (source for axis 1, or `"mixed"`),
  `magnitude_source`, `omega_source`, `orientation_class` (`"measured"` if axis-1
  sign came from lineage, labelling or velocity; `"imputed"` if from pseudotime;
  `"none"`), `n_sources_supporting`, `conflict_flag` (any axis),
  `conflict_sources`, `equilibrium_flag`, `stationarity_flag`.
* `uns["scjdo2"]["orient_meta"]`: sources run, ranking, thresholds, axis count,
  `J` hash before/after (`hash_unchanged`).

## 7. Flags (replace the undefined `stationarity_flag`)

* `equilibrium_flag` ∈ {`equilibrium`, `non_equilibrium`, `not_identified`}:
  where measured velocity exists (labelling or velocity), compare `v` with the
  scaled score `D·f` (global least-squares `D`): equilibrium if the cosine is
  ≥ 0.9 and the relative residual (the probability-flux term) is ≤ 0.3.
* `stationarity_flag` ∈ {`stationary`, `non_stationary`, `not_identified`}:
  only with labelling (the clock fixes `D`): local balance
  `|∇·(p v) − D Δp|`, with `p ∝ exp(−E)`, by kNN finite differences, relative to
  its scale; threshold fixed in the test below. Otherwise `not_identified`.
* A rotating steady state is stationary but non-equilibrium; the two flags keep
  that distinction.

## 8. Tests (gates computed by code only)

Every rate is over (cell, axis) pairs that pass the gap rule and have support;
each test also reports the support fraction, which must be ≥ 0.5 so a gate
cannot pass on a handful of cells.

1. **Pitchfork with known velocity and simulated clones** (new generator):
   `["lineage","labelling"]` sign agreement with truth ≥ 0.97; `["labelling"]`
   ≥ 0.95; `["pseudotime"]` with the correct root ≥ 0.90.
2. **Rotating two-well** (new generator; tests the velocity regression, not the
   energy model): `["labelling"]` omega median relative Frobenius error ≤ 0.3;
   `["pseudotime"]` omega = 0 and `rotation_identified = False`.
3. **Conflict:** flip velocity signs on 10 % of cells; `["labelling","velocity"]`:
   labelling wins on every injected cell; `conflict` true on ≥ 0.95 of them.
4. **Coverage:** clones cover 40 % of cells; `["lineage","pseudotime"]`:
   `orientation_class == "measured"` exactly on the covered cells, `"imputed"` on
   the rest; ledger coverage = 1.0.
5. **Flags:** pure gradient system with velocity → `equilibrium` on ≥ 0.9 of
   supported cells; rotating two-well → `non_equilibrium` on ≥ 0.9; no velocity
   source → `not_identified` on 100 %.
6. **Hash:** `J` byte-identical before and after, in every test.
7. **Paul15, `["pseudotime"]`** (report only): agreement of the axis-1 sign with
   an **independent** ordering — the annotated progenitor → mature stage order —
   not with the pseudotime that is the source (that would be 1.0 by
   construction); `n_cells_oriented`.
8. **LARRY, `["lineage","pseudotime"]`** (report only): `fraction_measured`;
   sign agreement lineage vs pseudotime on the overlap.

## 9. v1

* scJDO v1 is already not a pip dependency (`compare = []`); the report states
  it with the dependency diff.
* `compare/` stays as a separate v1-vs-v2 benchmark harness, outside the
  library and its tests.
* `tests/test_opspace_pitchfork.py`: tests that check scJDO2's own pitchfork
  behaviour are rewritten to run without v1; tests that only compare against v1
  move to the `compare/` harness or are deleted.

## 10. Work plan

* Branch `orientation` from `main` (`cb9298f`).
* Claude: the library, both synthetic generators, tests 1–6, the v1 test
  cleanup, `report.json` for tests 1–6.
* The user: scVelo and Palantir runs on the datasets used for tests 7–8,
  stored in the AnnData (velocity in `obsm`, pseudotime + start cell in
  `obs`/`uns`).
* Manus (task files under the coordination protocol): tests 7–8 on Paul15 and
  LARRY, reading those stored outputs.
* Deliverables: `scjdo2/orient/*.py`, `tests/test_orient.py`,
  `scjdo2/data/synthetic_oriented.py`, `report.json`, dependency diff.

## 11. As implemented (2026-10-16) — differences from the draft above

* **Noise floor per source** (§3): `labelling` and `velocity` use the 10th
  percentile of their velocity norms; `lineage` (a measured clone
  displacement) and `pseudotime` (a gradient, not a velocity) use 0.
* **`orientation_class`** (§6): `"measured"` if any axis's sign came from
  lineage, labelling or velocity; `"imputed"` if every signed axis came
  from pseudotime; `"none"` if no axis is signed. `sign_source` = source of
  the first signed axis. `ledger_coverage` = fraction of cells not `"none"`.
* **Test 3 (conflict)** is scored on injected cells where both sources sign
  some axis (§8: rates only where there is support); a conflict cannot
  exist elsewhere. Also checked: false conflicts on clean cells ≤ 1 %.
* **Test 4 (coverage)**: `"measured"` equals exactly the covered cells on
  which lineage signs a usable axis, and never appears on uncovered cells.
  "Exactly the covered cells" is unattainable with the gap rule: near the
  wells the two softest axes are near-degenerate, and lineage cannot sign
  an axis the gap rule rejects (72–75 % of covered cells get a sign).
* **Synthetic pitchfork**: cells are sampled at continuous times within
  ± half the day spacing of their collection day (three exact snapshots
  left gaps in state space that no local pseudotime gradient can bridge).
* **Test 1, pseudotime — gate not met** (marked `xfail(strict=True)`):
  agreement 0.69–0.71 at `align_min = 0.2`, 0.84–0.86 at 0.7. A pseudotime
  gradient is the direction of one scalar, not the velocity; off the main
  direction of motion its signs are noise, and 55–58 % of the signs it
  assigns are on axes along which cells barely move. Decision pending.
* **Omega** is fitted by a compact local ridge regression in
  `orient/base.py` rather than `field.py`'s API; same estimator
  (v_n − v_c ≈ J_v (z_n − z_c) over k = 30 neighbours).

## 12. Two orientation outputs (decided by the user, 2026-10-16; supersedes §5's single ranking)

S02 (`coord/orient/reviews/S02.md`) showed that the sources measure two
different things. Lineage measures **developmental direction** — where a
clone went between its earliest and latest sampling day — and matched that
truth at 0.98–0.99. Labelling and velocity measure **instantaneous flow**
— how a cell moves now. One merged sign ranked lineage above labelling, so
where trajectories bend or slow, lineage overrode the better flow source
(truth sided with labelling in 70–81 % of their conflicts).

* **Flow sign** — sources `labelling > velocity`; magnitude (labelling
  clock) and rotation (`omega`) belong here; flags use it. Scored against
  the instantaneous velocity truth.
* **Developmental sign** — sources `lineage > pseudotime`. Scored against
  the developmental truth (latest-minus-earliest-day clone displacement).
* Each group is merged as in §5 within itself. **The groups are never
  merged with each other.** Where both sign an axis and disagree,
  `obsm["orient_flow_vs_dev"]` records it; `obs["flow_vs_dev_disagree"]`
  summarises it per cell.
* Outputs: `obsm["orient_{flow,dev}_{sign,source,conflict}"]`,
  `obsm["orient_flow_magnitude"]`, `uns["scjdo2"]["omega"]`; per-cell
  `obs["{flow,dev}_{sign_source,orientation_class,n_sources_supporting,
  conflict_flag,conflict_sources}"]`. `orientation_class`: flow
  `measured` (labelling, velocity); dev `measured` (lineage) or `imputed`
  (pseudotime).
* **Pseudotime remains below its gate** (0.69–0.71 against the
  developmental truth too): its weakness is local gradient noise off the
  main direction of motion, not the choice of truth. Gate kept as
  `xfail(strict=True)`; decision still open.
* `scripts/orient_end_to_end.py` (S02) used the single-sign API and is
  superseded; S03 updates it.

## 13. Pseudotime: accepted limitation (decided by the user, 2026-10-16)

The 0.90 gate for pseudotime is retired. On the synthetic pitchfork,
pseudotime-derived developmental signs agree with the truth 0.69–0.71
(against both the developmental and the instantaneous truth; 0.84–0.86 only
at `align_min = 0.7`, with half the coverage). Reason: the gradient of one
scalar is a single direction; on axes along which cells do not move, its
sign is noise. Consequences:

* pseudotime stays the lowest-ranked developmental source and its signs are
  always `orientation_class = "imputed"`;
* a regression test guards the documented level (agreement ≥ 0.65);
* anyone reading developmental signs should treat imputed ones as
  ≈ 70 %-reliable on axes off the main direction of motion.

## 14. Pseudotime start-cell check (2026-10-17)

The start cell must lie in the lowest `root_quantile_max = 0.10` of
pseudotime (was: lowest 1 %, an arbitrary choice). The check exists to
catch a wrong or reversed root; Palantir places its start cell near, not
at, the minimum (orient/S03: 0.6–3.7th percentile on all 9 datasets). A
root at the end of the trajectory is still refused (tested); the root's
percentile is recorded in the source's metadata.

## 15. Dataset-level reliability check (decided by the user, 2026-10-17)

Motivation: in orient/S03, scVelo's velocity (on a generator later found to
be unfair to splicing methods: 2 % unspliced, no switching genes) agreed
with labelling on only 0.06–0.24 of the (cell, axis) pairs both signed
(systematically reversed). Ranking alone kept labelling's own cells intact,
but velocity then filled the remaining cells with mostly wrong signs (flow
0.896 → 0.665).

Rule. Per group, each source listed in `OrientConfig.reliability_min`
(default `{"velocity": 0.6}`) is compared, on the (cell, axis) pairs it
and the higher-ranked sources of its group both sign, with the sign those
higher sources decide. If there are at least `reliability_min_pairs`
(200) such pairs and the agreement is below the threshold, the source is
**demoted to a cross-check for this dataset**: it decides no sign, gives no
omega and does not count as supporting; its disagreements are still
recorded as conflicts. With no higher-ranked source present, or too few
shared pairs, the source is `not_checked` and used as before.
Output: `orient_meta["reliability"][group][source]` (status, agreement,
n_pairs, threshold) and `orient_meta["demoted"]`. Switch off with
`reliability_min={}`.

On the S03 data the check demotes scVelo in all 9 datasets and the flow
output equals labelling's exactly. Pseudotime is not checked by default
(lineage and Palantir agreed well in S03); retest in orient/S04.

## 16. Pseudotime real-time guard (2026-10-17)

Palantir replaces the given start cell by the nearest diffusion-map
boundary cell unless `use_early_cell_as_start=True`; the start cell to
record is the one it used (the pseudotime minimum). The §14 start-cell
check then cannot catch a reversed run. Where `obs[cfg.day_key]` has more
than one value, the pseudotime source requires Spearman(pseudotime, day)
> `pseudotime_day_min_spearman` (default 0) and refuses otherwise; the
correlation is recorded in the source's metadata. Sampling day is a
measured readout, not truth.

**Amendment (decided by the user, 2026-10-18; orient/S06).** Correlation
with day alone let 104 of 180 runs started from day-4 cells through
(Spearman −0.75…+0.54 for day-4 roots, +0.33…+0.84 for day-1 roots: no
threshold separates them). The source now also requires the **pseudotime
minimum to lie on the earliest sampling day**
(`pseudotime_start_on_earliest_day=True`). On the S06 runs this refuses
all 180 day-4-root runs and 8 of 180 day-1-root runs (conservative); it
was derived from those runs and is confirmed on fresh runs in orient/S07.

## 17. Lineage needs well-sampled clones (decided by the user, 2026-10-18)

Under realistic sparse sampling (orient/S06, v2 data) most clones have one
sampled cell on their earliest or latest day; their direction is one
cell's state minus another's, and lineage agreed with the developmental
truth at 0.72–0.74 against pseudotime's 0.96, so `{lineage, pseudotime}`
was worse than pseudotime alone. A clone now gives a direction only if it
has at least `lineage_min_cells_per_end = 2` sampled cells on both its
earliest and latest day; other cells fall to the lower-ranked source. The
counts used and skipped are in the source's metadata.
