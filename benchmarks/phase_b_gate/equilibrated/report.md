# Equilibrated pitchfork gate — four-row table

Generated: 2026-10-12 04:48:22

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 0.16 | 0.06 | 0.72 | 0.79 | 0.04 | **FAIL** |
| dsm_equilibrated | 0.1 | 0.10 | 0.73 | nan | 0.00 | 1.00 | **FAIL** |
| oracle | 0.3 | 0.18 | 0.07 | 0.71 | 0.87 | 0.04 | **FAIL** |
| dsm_equilibrated | 0.3 | 0.10 | 0.82 | nan | 0.00 | 1.00 | **FAIL** |
| oracle | 0.5 | 0.13 | 0.07 | 0.74 | 0.88 | 0.05 | **FAIL** |
| dsm_equilibrated | 0.5 | 0.07 | 0.81 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.1 | 0.12 | 1.00 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.3 | 0.19 | 1.00 | 0.00 | 0.00 | 0.93 | **FAIL** |
| v1_slicing_equilibrated | 0.5 | 0.08 | 1.00 | 1.00 | 0.05 | 0.97 | **FAIL** |
| dsm_transient | 0.1 | 0.09 | 0.89 | 0.00 | 0.00 | nan | **FAIL** |
| dsm_transient | 0.3 | 0.10 | 0.99 | nan | 0.00 | nan | **FAIL** |
| dsm_transient | 0.5 | 0.10 | 0.87 | 0.86 | 0.01 | nan | **FAIL** |

## Reading

* **Oracle** is the ceiling — analytic ∂f_toggle/∂z at each cell’s (X_true, α). frac(Re>0) ~ 0.04: equilibrated SDE snapshots live almost entirely in basins, so the gate’s pseudotime-band ground truth ("cells near τ_crit are fork") is structurally capped — most near-τ_crit cells are still in nearby stable FPs, not at the saddle. Holder caps at ~0.7-0.9 for the same reason.
* **DSM-equilibrated** fits a drift whose autodiff Jacobian is uniformly expansive (frac(Re>0) = 1.00) — same pathology as on the transient synthetic. The equilibration fixes the ORACLE measurement but not the DSM fit.
* **v1 slicing on equilibrated** shows the same expansive-Jacobian pathology (frac(Re>0) ~ 0.93-1.00); same DriftField + DSM backend as v2.
* **DSM-transient** is the standing negative control.

Phase C stays blocked. The oracle’s < 0.8 says the gate definition needs revisiting (pseudotime bands don’t match stationary physics), or the synthetic needs more saddle mass.
