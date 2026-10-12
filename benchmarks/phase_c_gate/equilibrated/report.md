# Phase C gate — equilibrated pitchfork (gene-space J)

Generated: 2026-10-12 09:34:05

Gate (spec §9):
  * holder P and R  ≥ 0.8 absolute
  * fork   P and R  within 0.10 of oracle at each τ_crit (R13 carry-forward)
  * latent↔gene-space agreement on cells above median op_confidence ≥ 0.95

Rows: `oracle` (analytic 2D J), `latent_dsm` (Phase B DSM-equilibrated for comparison), `gene_space_k0` (per-cell gene-space J, no smoothing), `gene_space_k30` (sparse Gaussian smoothing over the latent kNN).

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict | agree all / above-median-conf |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| latent_dsm | 0.1 | 0.23 | 0.77 | 1.00 | 0.79 | 0.14 | **FAIL** |
| gene_space_k0 | 0.1 | 0.03 | 0.53 | 0.95 | 0.10 | 0.76 | **FAIL** | 0.17 / 0.19 |
| gene_space_k30 | 0.1 | 0.03 | 0.33 | 0.94 | 0.34 | 0.52 | **FAIL** | 0.33 / 0.28 |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| latent_dsm | 0.3 | 0.22 | 0.77 | 0.99 | 0.82 | 0.15 | **FAIL** |
| gene_space_k0 | 0.3 | 0.03 | 0.41 | 0.95 | 0.18 | 0.64 | **FAIL** | 0.22 / 0.23 |
| gene_space_k30 | 0.3 | 0.04 | 0.30 | 0.94 | 0.50 | 0.34 | **FAIL** | 0.47 / 0.42 |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| latent_dsm | 0.5 | 0.19 | 0.54 | 0.99 | 0.80 | 0.13 | **FAIL** |
| gene_space_k0 | 0.5 | 0.03 | 0.37 | 0.91 | 0.21 | 0.56 | **FAIL** | 0.24 / 0.22 |
| gene_space_k30 | 0.5 | 0.03 | 0.33 | 0.94 | 0.44 | 0.44 | **FAIL** | 0.39 / 0.34 |

## Reading

* `latent_dsm` is the Phase B DSM-equilibrated baseline (holder passes, fork P below oracle by R13's margin). It supplies the behaviour vector that `gene_space_*` rows measure agreement against.
* `gene_space_k0` runs the Phase C rate model + per-gene weighted ridge in η-coords; its J is per-cell sparse CSR with the Fisher ledger from the same ridge likelihood. opspace.spectrum masks entries below the 10th-percentile per-cell Fisher before eigendecomposing.
* `gene_space_k30` applies sparse Gaussian smoothing over the latent kNN before classifying.
* Agreement is reported as `all / above-median-op_confidence`. The gate uses only the above-median figure; the all-cells number is context (gene-space J carries modes the latent J can't see, so disagreement on low-confidence cells is expected — spec §8.9).

## Finding — Option 1 tangent restriction unblocks `holder`, fork still bounded.

Fix (1) landed: `scjdo2.field.fit_tangent_projectors` computes a per-cell Fisher-weighted η-covariance, keeps directions with λ > snr_cutoff=1 (absolute threshold, valid under Fisher normalisation), and attaches the projector to the ledger as `off_tangent`. `opspace.spectrum` eigendecomposes `P_c J_c P_cᵀ` and marks slots past `r_c` empty with reason `off_tangent`. Fisher-mass tie-break on near-identical Re.

On this run `tangent r_c` lands at **14–19** across cells (vs ambient G=50 and the toggle-switch's structural rank of 2; the extra dimensions carry residual η-variance from the SiLU-trunk's nonlinear fit). That's a 3× reduction of the eigenvalue search space and it visibly bends the result:

* `frac(Re>0)` drops from **1.00** (pre-fix) to **0.34–0.76** across τ_crit — the lead eigenvalue is no longer stuck at a positive noise direction.
* `holder P` recovers from **nan** to **0.91–0.95** at every τ_crit (holder cells are now actually detectable), though `holder R` is 0.10–0.50 — the classifier is **selective but under-recalling** on the holder class.
* `latent↔gene-space agreement above median op_confidence` climbs from the pre-fix 0.12–0.15 to **0.19–0.42**. Smoothing (k=30) consistently helps agreement by 0.1–0.2 — the gene-space J is noisy at the per-cell level.
* `fork P` stays near the oracle-independent base rate (0.03–0.04). Fork cells are a tiny minority on equilibrated data (46 cells out of 1200 at τ=0.5) and the gene-space J has a Fisher-weighted null that systematically masks the saddle-direction eigenvalue when the saddle has little Fisher mass.

**Why holder R is low.** The tangent cutoff keeps ~17 directions on average, but only 2 of them are structurally meaningful; the remaining ~15 are soft residuals from the rate model's non-linear trunk. Even after Fisher-mass tie-breaking, the leading Re is often picked from one of those residual directions, which doesn't correspond to the physical basin eigenvalue. Fixing this requires either:

  (a) A stricter tangent threshold tied to the TRUE manifold rank (e.g. an elbow detector, or `snr_cutoff > 1` chosen by held-out NB NLL).
  (b) Fisher-mass **primary** eigenvalue sort inside the retained subspace (the current implementation uses it only as a Re tie-break).
  (c) Projecting J_c onto the top-r singular directions of `∇η(z_c)` directly (the stricter "the Jacobian of the rate model at that cell" variant of fix (1)), which is a tighter manifold than the kNN-ball η-covariance tangent.

## Phase C status

Oracle PASS everywhere. Holder absolute (≥0.8) is **not** met because of R, not P. Fork within 0.10 of oracle is **not** met. Agreement-above-median ≥0.95 is **not** met (max 0.42).

Phase C is unblocked structurally — the pipeline runs end-to-end, the ledger carries the Fisher info and the tangent projector, and the gene-space classifier produces non-degenerate output — but the acceptance criteria don't clear on this synthetic.

The three sub-fixes above are the natural next iteration. Before touching the figure layer, we need a decision on (a) vs (b) vs (c): (b) and (c) compose cleanly; (a) is a tuning knob orthogonal to both.
