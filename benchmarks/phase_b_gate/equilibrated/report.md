# Equilibrated pitchfork gate — four-row table (final)

Generated: 2026-10-12 05:58:45

Gate (Phase B, v2):
  * holder P and R  ≥ 0.8 (absolute)
  * fork P and R    within 0.10 of the oracle on the same ground truth

Ground truth: fork = analytic_lead_re > tol, holder = analytic_lead_re < -tol,
where tol = 0.1 · std(analytic_lead_re) — same scheme as opspace.behaviour.

σ sweep: σ_min ∈ {0.02, 0.05}, σ_max=1.0. σ_min=0.05 won on held-out DSM
loss (0.738 vs 0.800 mean). Final gate uses σ_min=0.05.

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| dsm_equilibrated | 0.1 | 0.31 | 0.77 | 0.99 | 0.91 | 0.10 | **FAIL** |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| dsm_equilibrated | 0.3 | 0.38 | 0.75 | 0.99 | 0.91 | 0.09 | **FAIL** |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| dsm_equilibrated | 0.5 | 0.42 | 0.61 | 0.98 | 0.92 | 0.07 | **FAIL** |
| v1_slicing_equilibrated | 0.1 | 0.04 | 1.00 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.3 | 0.05 | 0.73 | 0.72 | 0.03 | 0.93 | **FAIL** |
| v1_slicing_equilibrated | 0.5 | 0.05 | 1.00 | 1.00 | 0.04 | 0.97 | **FAIL** |
| dsm_transient | 0.1 | 0.88 | 0.21 | 0.75 | 0.98 | 0.07 | **FAIL** |
| dsm_transient | 0.3 | 0.78 | 0.20 | 0.77 | 0.97 | 0.08 | **FAIL** |
| dsm_transient | 0.5 | 0.21 | 0.04 | 0.74 | 0.85 | 0.07 | **FAIL** |

## Reading

Ground truth: analytic 2D J of f_toggle at (X_true, α), quantized by
the same 0.1·std tolerance scheme opspace.behaviour applies. Oracle is
self-consistent and hits P/R ~ 1.0 by construction.

* **Oracle** PASSES at every τ_crit — the ceiling snapshot data allows.
* **DSM-equilibrated** PASSES on holder absolute (P ≥ 0.98, R ≥ 0.91).
  Fork is CLOSE to oracle on recall (0.61-0.77 vs 1.00) but precision
  is still well short (0.31-0.42 vs 1.00). Within-0.10-of-oracle NOT met.
* **v1 slicing on equilibrated** still shows the uniformly expansive
  Jacobian pathology (frac(Re>0) ≈ 0.93-1.00); v1 cannot disable its
  residual MLP from the public API (V2_PLAN R12).
* **DSM-transient** is the α-ramp negative control — same pathology
  signature as expected.
