# K562 Perturb-seq prediction — v0.5-basin-lock

Tag: `v0.5-basin-lock` (commit on tagged branch, Replogle 2022 Essential K562).

## Setup
* n_kd evaluated: 1661
* n_kd in linear regime (|δx_obs| ≤ median): 831
* median |δx_obs| cutoff: 4.132
* symmetry median (control-cell Jacobian): 1.42e-07
* rate-model held-out NB NLL: -4.488

## Baseline comparison

| arm              |   n_kd |   scjdo2_cos_median |   scjdo2_cos_mean |   scjdo2_cos_gt_0 |   de_only_cos_median |   random_cos_median |   random_cos_std_median |   spearman_mag_pred_vs_obs |
|:-----------------|-------:|--------------------:|------------------:|------------------:|---------------------:|--------------------:|------------------------:|---------------------------:|
| all              |   1661 |               0.089 |             0.084 |             0.63  |                0.003 |                  -0 |                   0.022 |                      0.193 |
| linear_regime    |    831 |               0.063 |             0.065 |             0.603 |                0.001 |                  -0 |                   0.022 |                      0.161 |
| nonlinear_regime |    830 |               0.106 |             0.104 |             0.658 |                0.004 |                   0 |                   0.022 |                      0.058 |

## Driver check

Spearman(left-eigvec sensitivity, |δx_obs| per gene) = **-0.310**
(n genes evaluated = 1661).

Negative sign — genes that the operator ranks as high-sensitivity
drivers have SMALLER observed self-knockdown shifts. Possible
explanation: high-sensitivity genes have strong basin-direction
coupling, so their own knockdown gets buffered along the basin
floor. First-result observation, no post-hoc tuning.

## Figures

*  — per-KD cosine vs mean Fisher
  confidence of the ok-mask entries (log10 n_cells colouring).
*  — |δx_pred| vs |δx_obs| per KD,
  linear-regime cells highlighted.

## Pre-stated decisions (V2_PLAN §9e)

* HVG 2000 on controls + mandatory inclusion of perturbation
  targets (3 559 genes kept after filter_genes min_cells=3 in
  controls).
* KDs with ≥ 50 cells AND target present in var; 1 661 pass.
* Linear-regime split = per-KD |δx_obs| ≤ median across KDs.
* Baselines: random unit vectors (10/KD) + DE-only; GEARS skipped.
* Control-basin operator = mean per-cell latent J across controls;
  grad_eta evaluated at the control centroid.
* Prediction = grad_eta @ (−J_mean⁻¹) @ pinv(grad_eta) @ b_g with
  Tikhonov ε = 1e-6·‖J‖_F / d for numerical stability.
* No archetype decomposition, no pseudotime, no prior.

## Primary readout

* scjdo2 cos_median = **0.089** (all 1661 KDs); mean 0.084; frac>0 = **63%**.
* DE-only cos_median = **0.003**; random null cos_median = **−5e-5** with
  std ~0.022.
* scjdo2 beats baselines by ~4σ in cosine but absolute signal is weak.
* Linear regime cos_median 0.063; nonlinear 0.106 — nonlinear KDs
  (stronger shifts) are better predicted on cosine, consistent with
  the direction being easier to recover when |δx_obs| is above
  measurement noise.
* Driver-sensitivity Spearman = −0.31 — anti-correlation, written up
  as a result; no post-hoc change to the sensitivity formula.
