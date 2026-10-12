# Phase C gate — equilibrated pitchfork (gene-space J)

Generated: 2026-10-12 11:05:24

Gate (spec §9):
  * holder P and R  ≥ 0.8 absolute
  * fork   P and R  within 0.10 of oracle at each τ_crit (R13 carry-forward)
  * latent↔gene-space agreement on cells above median op_confidence ≥ 0.95

Rows: `oracle` (analytic 2D J), `latent_dsm` (Phase B DSM-equilibrated for comparison), `gene_space_k0` (per-cell gene-space J, no smoothing), `gene_space_k30` (sparse Gaussian smoothing over the latent kNN).

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict | agree all / above-median-conf |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| latent_dsm | 0.1 | 0.23 | 0.77 | 1.00 | 0.79 | 0.14 | **FAIL** |
| gene_space_k0 | 0.1 | 0.03 | 0.60 | 0.86 | 0.02 | 0.89 | **FAIL** | 0.11 / 0.15 |
| gene_space_k30 | 0.1 | 0.02 | 0.51 | 0.93 | 0.04 | 0.87 | **FAIL** | 0.13 / 0.17 |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| latent_dsm | 0.3 | 0.22 | 0.77 | 0.99 | 0.82 | 0.15 | **FAIL** |
| gene_space_k0 | 0.3 | 0.04 | 0.68 | 0.97 | 0.07 | 0.76 | **FAIL** | 0.15 / 0.17 |
| gene_space_k30 | 0.3 | 0.05 | 0.73 | 0.96 | 0.26 | 0.59 | **FAIL** | 0.30 / 0.32 |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| latent_dsm | 0.5 | 0.19 | 0.54 | 0.99 | 0.80 | 0.13 | **FAIL** |
| gene_space_k0 | 0.5 | 0.04 | 0.59 | 1.00 | 0.08 | 0.76 | **FAIL** | 0.15 / 0.18 |
| gene_space_k30 | 0.5 | 0.04 | 0.67 | 0.97 | 0.22 | 0.67 | **FAIL** | 0.23 / 0.25 |

## Reading

* `latent_dsm` is the Phase B DSM-equilibrated baseline (holder passes, fork P below oracle by R13's margin). It supplies the behaviour vector that `gene_space_*` rows measure agreement against.
* `gene_space_k0` runs the Phase C rate model + per-gene weighted ridge in η-coords; its J is per-cell sparse CSR with the Fisher ledger from the same ridge likelihood. opspace.spectrum masks entries below the 10th-percentile per-cell Fisher before eigendecomposing.
* `gene_space_k30` applies sparse Gaussian smoothing over the latent kNN before classifying.
* Agreement is reported as `all / above-median-op_confidence`. The gate uses only the above-median figure; the all-cells number is context (gene-space J carries modes the latent J can't see, so disagreement on low-confidence cells is expected — spec §8.9).

## Finding — Option (c) Jacobian-SVD tangent (r_c = d) **underperforms** the earlier ball-covariance variant on this synthetic; the rank-ceiling observation still holds and now sits in V2_PLAN R14.

What landed. `scjdo2.field.fit_jacobian_tangent_projectors` replaces the ball-covariance tangent (dropped). For each cell it computes ``U, Σ, V^T = svd(∇η(z_c))`` and keeps **all** d left singular directions as rows of ``P_c``, so ``r_c = d`` by construction. Fisher-mass stays a tie-breaker inside the kept subspace (unchanged from Option 1).

Structural claim (V2_PLAN R14, confirmed). With a ``d``-dim decoder the gene-space J is a lift of the latent J: ``J_gene = ∇η · J_latent · (∇η)^+`` has rank at most ``d``, and its spectrum on ``image(∇η(z_c))`` equals the latent spectrum up to a similarity. The ceiling for `gene_space_*` on this synthetic is therefore `latent_dsm`'s row (R13's inherited fork P gap). Phase C can match Phase B here; it cannot exceed it.

Operational surprise. At ``r_c = d = 20`` on PCA(20) of the pitchfork, the gene-space row is **worse** than the earlier ball-covariance run at ``r_c ≈ 17``: holder R collapsed from 0.10–0.50 (ball, both k variants) to 0.02–0.26 (jacobian_svd), and ``frac(Re>0)`` climbed back to 0.6–0.9 (vs ball 0.3–0.8). The two tangent choices should produce the same subspace when ``∇η`` is full-rank d — which it is here — so the gap comes from WHAT the extra directions carry:

  * The pitchfork is intrinsically 2-D; its PCA(20) has 18 near-null modes. ``∇η`` learned on counts puts 18 small (but non-zero) singular values alongside the top 2.
  * The per-gene weighted ridge from Step 5 is numerically weakest exactly along those small-σ directions — low Fisher mass, dominated by the ``λ·I`` regulariser, so ``J_c`` entries there are near-random.
  * **No σ-threshold on the SVD** means the full d directions feed `np.linalg.eig(block)`. The random small-σ entries numerically dominate the leading Re of the ``(d, d)`` block and pull the behaviour classifier off-signal. The ball-covariance variant implicitly truncated at ``σ² > 1`` under Fisher normalisation (r_c ≈ 17), which happened to clip the worst noise directions.

What this says. Option (c) as the user specified it (r_c = d, no σ-threshold) is the right *structural* lens — tangent is ``image(∇η)`` exactly — but on PCA(20) of a 2-D attractor it includes 18 near-null directions where the Phase C ridge fit is numerically dominated by its regulariser. A follow-up would either (i) lower the latent dim to the true intrinsic rank (d_latent = 2 or 3), (ii) add a σ-threshold (keep ``σ_j > rel·σ_1``), or (iii) use a proper hidden-driver synthetic where the extra ``d − 2`` directions are informative. All three are Phase D work; recorded in R14.

## Next actions

* R14 note stands in V2_PLAN.md: gene-space J is a rank-d lift of latent J, so Phase C's ceiling on this synthetic is `latent_dsm`'s row. Deferred follow-ups: σ-threshold, intrinsic-rank PCA, or hidden-driver synthetic — all Phase D.
* Figure layer (spec §3) still waits for a run where `holder` cells on the above-median-conf subset agree with the latent classifier; current agree-above max is 0.32 at τ=0.3 k=30. Panel C's leading-eigvec drivers are not meaningful below ~0.6 agreement.
