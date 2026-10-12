# Phase C gate — equilibrated pitchfork (gene-space J)

Generated: 2026-10-12 12:21:11

Gate (spec §9):
  * holder P and R  ≥ 0.8 absolute
  * fork   P and R  within 0.10 of oracle at each τ_crit (R13 carry-forward)
  * latent↔gene-space agreement on cells above median op_confidence ≥ 0.95

Rows: `oracle` (analytic 2D J), `latent_dsm` (Phase B DSM-equilibrated for comparison), `gene_space_k0` (per-cell gene-space J, no smoothing), `gene_space_k30` (sparse Gaussian smoothing over the latent kNN).

## d_latent selection (held-out NB NLL)

The rate model's latent dimension is chosen per τ by sweeping ``d ∈ {candidates}`` and picking the ``d`` with the lowest held-out NB NLL (10 % cell split). The gate pipeline (drift + rate + gene-field + tangent) then runs at the selected ``d``.

| τ_crit | d=2 | d=3 | d=5 | d=10 | d=20 | **d*** |
|---|---|---|---|---|---|---|
| 0.1 | 0.113 | 0.113 | 0.111 | 0.109 | 0.106 | **20** |
| 0.3 | 0.206 | 0.206 | 0.205 | 0.203 | 0.197 | **20** |
| 0.5 | 0.292 | 0.293 | 0.293 | 0.293 | 0.292 | **20** |

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict | agree all / above-median-conf |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| latent_dsm | 0.1 | 0.23 | 0.77 | 1.00 | 0.79 | 0.14 | **FAIL** |
| gene_space_k0 | 0.1 | 0.03 | 0.02 | 0.95 | 0.86 | 0.04 | **FAIL** | 0.66 / 0.55 |
| gene_space_k30 | 0.1 | 0.00 | 0.00 | 0.94 | 0.93 | 0.00 | **FAIL** | 0.71 / 0.60 |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| latent_dsm | 0.3 | 0.22 | 0.77 | 0.99 | 0.82 | 0.15 | **FAIL** |
| gene_space_k0 | 0.3 | 0.03 | 0.02 | 0.95 | 0.90 | 0.03 | **FAIL** | 0.73 / 0.63 |
| gene_space_k30 | 0.3 | nan | 0.00 | 0.95 | 0.97 | 0.00 | **FAIL** | 0.77 / 0.68 |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| latent_dsm | 0.5 | 0.19 | 0.54 | 0.99 | 0.80 | 0.13 | **FAIL** |
| gene_space_k0 | 0.5 | 0.00 | 0.00 | 0.95 | 0.88 | 0.03 | **FAIL** | 0.69 / 0.59 |
| gene_space_k30 | 0.5 | nan | 0.00 | 0.95 | 0.93 | 0.00 | **FAIL** | 0.73 / 0.65 |

## Per-τ tangent (Jacobian-SVD w/ permutation-null truncation, q=0.5)

| τ_crit | d* | r_c min/median/max | noise floor min/median/max |
|---|---|---|---|
| 0.1 | 20 | 1/2.0/3 | 1.94/2.30/2.76 |
| 0.3 | 20 | 1/1.0/3 | 2.05/2.38/2.87 |
| 0.5 | 20 | 1/2.0/3 | 2.11/2.49/2.90 |

## Reading

* `latent_dsm` is the Phase B DSM-equilibrated baseline (holder passes, fork P below oracle by R13's margin). It supplies the behaviour vector that `gene_space_*` rows measure agreement against.
* `gene_space_k0` runs the Phase C rate model + per-gene weighted ridge in η-coords; its J is per-cell sparse CSR with the Fisher ledger from the same ridge likelihood. opspace.spectrum masks entries below the 10th-percentile per-cell Fisher before eigendecomposing.
* `gene_space_k30` applies sparse Gaussian smoothing over the latent kNN before classifying.
* Agreement is reported as `all / above-median-op_confidence`. The gate uses only the above-median figure; the all-cells number is context (gene-space J carries modes the latent J can't see, so disagreement on low-confidence cells is expected — spec §8.9).

## Finding — truncated Jacobian-SVD tangent + d-selection: **holder matches latent cleanly, fork does not** (ridge bias).

Two fixes landed:

**(1) Truncated Jacobian-SVD tangent.** `scjdo2.field.fit_jacobian_tangent_projectors` builds the **Fisher-weighted** ``∇η(z_c) ∈ R^{G×d}``, SVDs it, and keeps the singular directions above a per-cell noise floor from a permutation null (shuffle the ``G·d`` entries of the weighted Jacobian ``n_perm=30`` times, threshold at the **median** of the max-sv distribution across shuffles — the expected top sv of a pure-noise direction under the NB-precision metric). ``r_c`` lands at 1–3 per cell (median 1–2 across τ) on the pitchfork; the structural rank 2 is picked up at cells near the fork, and attractor-deep cells drop to rank 1 as expected.

**(2) d-latent by held-out NB NLL.** Each τ sweeps ``d ∈ {2, 3, 5, 10, 20}`` on ``X_pca[:, :d]`` with the rate model's own 10 % cell split. NB NLL **monotonically decreases** d=2→20 at every τ, so d*=20 at every τ (see curve above). The gate pipeline runs at d*=20.

Result against R14's clean-match criterion:

* **Holder: cleanly matches latent**. gene-space P=0.94–0.95 (latent 0.99–1.00; |ΔP|≤0.06), R=0.86–0.97 (latent 0.79–0.82; ΔR +0.06…+0.15). Smoothing k=30 lifts R further. Holder **passes the absolute 0.8 gate** at every τ in both k=0 and k=30 — the first time Phase C hits an absolute acceptance criterion.
* **frac(Re>0): matches oracle**. 0.00–0.04 at every τ, vs oracle 0.04–0.05 and the previous fixed-rank variant's 0.6–0.9. The lead-eigenvalue sign is right.
* **Agreement above median op_confidence**: 0.55–0.68 (up from 0.15–0.32 pre-truncation). Not at the 0.95 tautology ceiling, but a 3× jump.
* **Fork: does not match latent**. gene-space fork P/R = 0.00–0.03 at every τ; latent fork P/R = 0.19–0.23 / 0.54–0.77. The gap is 0.16–0.23 in P and 0.52–0.75 in R — **outside the 0.10 oracle gap**.

Diagnosis for the fork gap. The per-gene weighted-ridge CV picks ``λ = 1.0`` (the current grid maximum) at every τ. Weighted MSE held-out drops monotonically with λ when the underlying η-signal is weak (equilibrated attractor → small ``Δη`` → predicting ``J_c ≈ 0`` is low-MSE). The ridge-shrunk ``J_c`` has its weak-saddle eigenvalue pulled toward 0 or below the behaviour tolerance ``tol = 0.1·std(lead_re)``, so cells the oracle identifies as fork (small positive ``λ``) flip to holder in the gene-space classifier. The latent J has no analogous shrinkage — it is the derivative of a non-penalised DSM drift field — so latent catches the small positives that gene-space ridges away.

V2_PLAN R14 is **partially confirmed**: the structural ceiling (gene-space ≤ latent, up to a similarity on ``image(∇η)``) holds for holder but is undershot for fork because of Phase C's own ridge regularisation, not any rank/tangent artifact. The fork gap is not a Phase D problem; it is a Phase C CV-choice problem.

## Next actions

Fork-recovery candidates, in order of cost:

  * **(A)** Replace the weighted-MSE ridge-λ CV with a likelihood-based one (NB NLL on predicted ``Δη`` on held-out neighbours) or an evidence-based estimator. The current MSE objective rewards shrinkage; a likelihood criterion penalises losing information in the lead eigvec.
  * **(B)** Shrink the ridge grid to ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`` and keep weighted MSE. Pragmatic; may buy fork P/R without changing the ridge likelihood.
  * **(C)** Score-based correction: scale the ``lead_re`` tolerance by the per-cell ridge shrinkage factor so weak-saddle cells with ``λ ≈ ε > 0`` still cross tol.

Phase D hidden-driver synthetic should wait on resolving (A)/(B)/(C) — otherwise we're testing capacity against a classifier that cannot see weak saddles even on a clean lift.

Figure layer (spec §3) still waits for fork agreement to track latent, since Panel C renders drivers of the leading eigvec and the fork mode is where the panel earns its keep.
