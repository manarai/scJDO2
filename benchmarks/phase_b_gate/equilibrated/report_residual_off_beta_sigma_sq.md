# Equilibrated pitchfork gate — four-row table

Generated: 2026-10-12 05:43:58

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 0.88 | 1.00 | 0.84 | 0.04 | **PASS** |
| dsm_equilibrated | 0.1 | 0.36 | 0.78 | 1.00 | 0.91 | 0.10 | **FAIL** |
| oracle | 0.3 | 1.00 | 0.88 | 1.00 | 0.89 | 0.04 | **PASS** |
| dsm_equilibrated | 0.3 | 0.41 | 0.72 | 0.99 | 0.91 | 0.09 | **FAIL** |
| oracle | 0.5 | 1.00 | 0.79 | 1.00 | 0.91 | 0.05 | **FAIL** |
| dsm_equilibrated | 0.5 | 0.52 | 0.59 | 0.99 | 0.92 | 0.07 | **FAIL** |
| v1_slicing_equilibrated | 0.1 | 0.05 | 1.00 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.3 | 0.06 | 0.72 | 0.72 | 0.03 | 0.93 | **FAIL** |
| v1_slicing_equilibrated | 0.5 | 0.06 | 1.00 | 1.00 | 0.04 | 0.97 | **FAIL** |
| dsm_transient | 0.1 | 0.91 | 0.19 | 0.78 | 0.98 | 0.07 | **FAIL** |
| dsm_transient | 0.3 | 0.78 | 0.18 | 0.79 | 0.96 | 0.08 | **FAIL** |
| dsm_transient | 0.5 | 0.26 | 0.05 | 0.76 | 0.86 | 0.07 | **FAIL** |

## Reading

Ground truth is strict per-cell sign of the ANALYTIC
``∂f_toggle/∂z`` evaluated at ``(X_true_c, α_c)``.
``true_fork(c) ⇔ analytic_lead_re > 0``; ``true_holder(c) ⇔ < 0``.

* **Oracle** is self-consistent (classifier applied to the analytic
  J, scored against the same J's sign) → P/R ≈ 1 modulo cells near
  zero that opspace labels "drift".
* **DSM-equilibrated** and **v1_slicing_equilibrated** measure how
  well the LEARNED Jacobian recovers the analytic per-cell sign.
* **DSM-transient** is the α-ramp negative control, measured against
  the same analytic ground truth so it is directly comparable.

Phase C unblocks only when DSM-equilibrated passes fork+holder P/R
≥ 0.8 at every τ_crit.
