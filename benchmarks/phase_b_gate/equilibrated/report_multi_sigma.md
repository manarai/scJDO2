# Equilibrated pitchfork gate — four-row table

Generated: 2026-10-12 05:36:26

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.5 | 1.00 | 0.89 | 1.00 | 0.88 | 0.04 | **PASS** |
| dsm_equilibrated | 0.5 | 0.05 | 0.67 | 0.94 | 0.08 | 0.81 | **FAIL** |
| dsm_transient | 0.5 | 0.27 | 0.56 | 0.79 | 0.19 | 0.65 | **FAIL** |

> v1_slicing_on_equilibrated: PENDING — produce via compare/run_v1_equilibrated_pitchfork.py

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
