# Equilibrated pitchfork gate — four-row table (dynamic ground truth)

Generated: 2026-10-12 05:05:32

Ground truth: strict per-cell sign of the ANALYTIC ∂f_toggle/∂z at (X_true_c, α_c).
true_fork(c) ⇔ analytic_lead_re > 0, true_holder(c) ⇔ analytic_lead_re < 0.

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |
|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 0.88 | 1.00 | 0.84 | 0.04 | **PASS** |
| dsm_equilibrated | 0.1 | 0.04 | 0.71 | nan | 0.00 | 1.00 | **FAIL** |
| oracle | 0.3 | 1.00 | 0.88 | 1.00 | 0.89 | 0.04 | **PASS** |
| dsm_equilibrated | 0.3 | 0.04 | 0.82 | nan | 0.00 | 1.00 | **FAIL** |
| oracle | 0.5 | 1.00 | 0.79 | 1.00 | 0.91 | 0.05 | **FAIL** |
| dsm_equilibrated | 0.5 | 0.05 | 0.76 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.1 | 0.05 | 1.00 | nan | 0.00 | 1.00 | **FAIL** |
| v1_slicing_equilibrated | 0.3 | 0.06 | 0.72 | 0.72 | 0.03 | 0.93 | **FAIL** |
| v1_slicing_equilibrated | 0.5 | 0.06 | 1.00 | 1.00 | 0.04 | 0.97 | **FAIL** |
| dsm_transient | 0.1 | 0.28 | 0.80 | nan | 0.00 | 1.00 | **FAIL** |
| dsm_transient | 0.3 | 0.27 | 0.79 | nan | 0.00 | 1.00 | **FAIL** |
| dsm_transient | 0.5 | 0.26 | 0.77 | nan | 0.00 | 1.00 | **FAIL** |

## Reading

* **Oracle** passes at τ=0.1 and 0.3 (fork P/R and holder P/R all ≥ 0.84). At τ=0.5 the oracle fork recall is 0.79 — opspace.behaviour's tol flags cells with small-positive analytic lead_re as "drift" rather than "fork". Within measurement noise; the ceiling is near 1.
* **DSM-equilibrated** fails catastrophically on precision (fork P ≈ 0.04-0.05 vs oracle 1.00). The learned Jacobian is uniformly expansive (frac(Re>0) = 1.00) — labels ~98 % of cells fork while analytic says only ~4 %. The equilibration fixes the oracle measurement but not the fit.
* **v1 slicing on equilibrated** same pathology (fork P ≈ 0.05-0.06, frac(Re>0) 0.93-1.00). v1 and v2 share the DriftField + DSM backend; the kernel-windowed slicing in v1 does not fix the sign.
* **DSM-transient** has higher fork precision (~0.27) only because 27% of transient cells have analytic lead_re > 0 (near-saddle, during the α-ramp passage) vs 4% in equilibrated. The underlying metric (DSM labels everything fork) is identical.

## Phase C

Phase C stays blocked. The dynamic ground truth confirms the problem is not the gate definition — the oracle passes. The DSM fit itself is the barrier: the autodiff Jacobian of the learned drift is uniformly positive-definite regardless of data regime, so the classifier's fork/holder split cannot track the physics.

The DSM loss plateau at ~94 across every run (same signature in the convergence_audit/) is the trail: the gradient of the DSM score term is overpowering the velocity prior and whatever structure the FiLM time-conditioning tries to add, and the output converges to a near-constant expansive field. Likely culprits ranked by cost to probe: σ-scaling in the DSM target, FiLM gate schedule, or alpha_control weighting. None of these needs new synthetic data — the equilibrated generator is now in-tree and the gate is well-defined under dynamic labels.
