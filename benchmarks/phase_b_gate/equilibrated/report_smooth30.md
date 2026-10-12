# Equilibrated pitchfork gate — four-row table

Generated: 2026-10-12 07:08:09

Gate (Phase B, v2):
  * holder P and R  ≥ 0.8 (absolute)
  * fork P and R    within 0.10 of the oracle on the same ground truth

Ground truth: fork = analytic_lead_re > tol, holder = analytic_lead_re < -tol,
where tol = 0.1 · std(analytic_lead_re) — same scheme as opspace.behaviour.

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| dsm_equilibrated | 0.1 | 0.31 | 0.77 | 0.99 | 0.91 | 0.10 | **FAIL** |
| dsm_equilibrated_smoothed(k=30) | 0.1 | 0.40 | 0.70 | 0.99 | 0.95 | 0.08 | **FAIL** |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| dsm_equilibrated | 0.3 | 0.38 | 0.75 | 0.99 | 0.91 | 0.09 | **FAIL** |
| dsm_equilibrated_smoothed(k=30) | 0.3 | 0.54 | 0.57 | 0.98 | 0.96 | 0.06 | **FAIL** |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| dsm_equilibrated | 0.5 | 0.42 | 0.61 | 0.98 | 0.92 | 0.07 | **FAIL** |
| dsm_equilibrated_smoothed(k=30) | 0.5 | 0.64 | 0.35 | 0.97 | 0.97 | 0.03 | **FAIL** |
| v1_slicing_equilibrated | 0.1 | 0.04 | 1.00 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.3 | 0.05 | 0.73 | 0.72 | 0.03 | 0.93 | **FAIL** |
| v1_slicing_equilibrated | 0.5 | 0.05 | 1.00 | 1.00 | 0.04 | 0.97 | **FAIL** |
| dsm_transient | 0.1 | 0.88 | 0.21 | 0.75 | 0.98 | 0.07 | **FAIL** |
| dsm_transient_smoothed(k=30) | 0.1 | 1.00 | 0.05 | 0.72 | 1.00 | 0.02 | **FAIL** |
| dsm_transient | 0.3 | 0.78 | 0.20 | 0.77 | 0.97 | 0.08 | **FAIL** |
| dsm_transient_smoothed(k=30) | 0.3 | 1.00 | 0.10 | 0.75 | 1.00 | 0.04 | **FAIL** |
| dsm_transient | 0.5 | 0.21 | 0.04 | 0.74 | 0.85 | 0.07 | **FAIL** |
| dsm_transient_smoothed(k=30) | 0.5 | nan | 0.00 | 0.72 | 1.00 | 0.00 | **FAIL** |

## Reading

Ground truth: fork/holder labels derived from the analytic 2D J of
``f_toggle(X_true_c, α_c)``, quantized by the same 0.1·std tolerance
scheme ``opspace.behaviour`` applies. The oracle becomes self-
consistent and P/R ~ 1.0 by construction.

* **Oracle** is the ceiling — classifier applied to analytic J.
* **DSM-equilibrated** measures how well a learned drift's autodiff
  Jacobian (``u = β·score`` with the residual MLP frozen zero,
  ``β = σ_sde²``, multi-σ DSM) matches the oracle labels.
* **v1_slicing_equilibrated** uses v1's unmodified DriftField (with
  the residual MLP it cannot disable from the public API — V2_PLAN
  R12), so it reproduces the uniformly-expansive-Jacobian pathology
  as designed.
* **DSM-transient** is the α-ramp negative control: snapshot score
  on non-stationary data has no basin structure to recover.

Phase C unblocks when DSM-equilibrated meets:
  * holder P and R  ≥ 0.8 absolute
  * fork P and R    within 0.10 of the oracle at the same τ_crit.
