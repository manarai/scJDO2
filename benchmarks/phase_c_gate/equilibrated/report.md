# Phase C gate — equilibrated pitchfork (gene-space J)

Generated: 2026-10-12 08:12:28

Gate (spec §9):
  * holder P and R  ≥ 0.8 absolute
  * fork   P and R  within 0.10 of oracle at each τ_crit (R13 carry-forward)
  * latent↔gene-space agreement on cells above median op_confidence ≥ 0.95

Rows: `oracle` (analytic 2D J), `latent_dsm` (Phase B DSM-equilibrated for comparison), `gene_space_k0` (per-cell gene-space J, no smoothing), `gene_space_k30` (sparse Gaussian smoothing over the latent kNN).

| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict | agree all / above-median-conf |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.1 | 1.00 | 1.00 | 1.00 | 0.85 | 0.04 | **PASS** |
| latent_dsm | 0.1 | 0.23 | 0.77 | 1.00 | 0.79 | 0.14 | **FAIL** |
| gene_space_k0 | 0.1 | 0.03 | 0.63 | nan | 0.00 | 1.00 | **FAIL** | 0.08 / 0.13 |
| gene_space_k30 | 0.1 | 0.04 | 0.79 | nan | 0.00 | 1.00 | **FAIL** | 0.09 / 0.12 |
| oracle | 0.3 | 1.00 | 1.00 | 1.00 | 0.90 | 0.04 | **PASS** |
| latent_dsm | 0.3 | 0.22 | 0.77 | 0.99 | 0.82 | 0.15 | **FAIL** |
| gene_space_k0 | 0.3 | 0.03 | 0.66 | nan | 0.00 | 1.00 | **FAIL** | 0.09 / 0.15 |
| gene_space_k30 | 0.3 | 0.03 | 0.66 | nan | 0.00 | 0.98 | **FAIL** | 0.09 / 0.15 |
| oracle | 0.5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.05 | **PASS** |
| latent_dsm | 0.5 | 0.19 | 0.54 | 0.99 | 0.80 | 0.13 | **FAIL** |
| gene_space_k0 | 0.5 | 0.04 | 0.70 | nan | 0.00 | 1.00 | **FAIL** | 0.08 / 0.14 |
| gene_space_k30 | 0.5 | 0.03 | 0.67 | nan | 0.00 | 1.00 | **FAIL** | 0.09 / 0.15 |

## Reading

* `latent_dsm` is the Phase B DSM-equilibrated baseline (holder passes, fork P below oracle by R13's margin). It supplies the behaviour vector that `gene_space_*` rows measure agreement against.
* `gene_space_k0` runs the Phase C rate model + per-gene weighted ridge in η-coords; its J is per-cell sparse CSR with the Fisher ledger from the same ridge likelihood. opspace.spectrum masks entries below the 10th-percentile per-cell Fisher before eigendecomposing.
* `gene_space_k30` applies sparse Gaussian smoothing over the latent kNN before classifying.
* Agreement is reported as `all / above-median-op_confidence`. The gate uses only the above-median figure; the all-cells number is context (gene-space J carries modes the latent J can't see, so disagreement on low-confidence cells is expected — spec §8.9).

## Finding — gene-space lead eigenvalue is dominated by the ambient G-dim noise floor on this synthetic.

Every `gene_space_*` row reports **frac(Re>0) ≈ 1.00** and **holder P = nan, R = 0** at every τ_crit, while the ORACLE on the exact same cells reports frac(Re>0) ≈ 0.04–0.05 and holder P/R ≈ 1.00 / 0.85–0.92. The pipeline is wired right (η recovers to 0.85 cos-sim on toy fits, J↔F share the same sparse layout, Fisher mask fires at the 10th-percentile cutoff) — the gene-space classifier is nevertheless unusable on this synthetic.

**Root cause.** The pitchfork gene matrix is a dim-2 projection `X_obs = z_true @ W` of a 2D process, so the per-cell η-manifold has effective **rank 2** out of G=50 live columns. Step 5's ridge fits a dense `J_c` on the full L=G=50 subspace: the 2 informative eigenvalues (which the oracle Jacobian carries) sit alongside ~48 near-zero / noise eigenvalues from the ridge regulariser. `spectrum()` takes the top-`n_eig` eigenvalues by Re, which almost always picks a noise direction's small positive real part over the informative basin eigenvalue. **frac(Re>0) = 1.0 is a sampling artifact** of taking max(Re) over 48 noise + 2 signal directions.

**Why the current `fisher_mask` default doesn't rescue this.** The 10th-percentile cutoff removes the lowest-Fisher 10 % of entries, but on this synthetic every gene is uniformly Fisher-significant (dense signal, no sparse expression), so the mask is a near no-op. On real scRNA data most genes drop out in any given neighbourhood and the mask recovers rank — this is the regime the spec was written for.

**What this changes.** The gate falls through the fork criterion for a DIFFERENT reason than Phase B's R13 (where DSM score noise flipped borderline cells). Here the pathology is **effective-rank vs ambient-dim mismatch in the eigenvalue selection**. Two separate fixes are candidate for a follow-up iteration:

  1. **Rank-aware spectrum** — truncate `J_c` to the top-r singular directions of `∇η(z_c)` (the Jacobian of the rate model at that cell) before eigendecomposing. The G-dim ambient space is replaced by the effective r-dim manifold; the eigvals carry only informative modes. Natural fit with the Fisher-weighted invariant.
  2. **Fisher-mass eigenvalue sort** — sort eigenvalues by `|eigvec|^T · F_diag · |eigvec|` instead of by Re, so the leading eigenvalue is the one with the most Fisher support rather than the numerically-largest Re on a near-degenerate ridge solution.

Either or both can be prototyped against this same gate without re-fitting rates or gene-space J; the gate harness is designed to swap the opspace.spectrum call. Reporting the table unmasked surfaces the signal rather than hides it.

## Next actions

* Decide between fix (1) rank-aware spectrum vs (2) Fisher-mass eigenvalue sort (or both; they compose).
* Re-run the gate with the chosen fix before the figure layer (spec §3) ships — Panel C renders drivers of the LEADING eigvec, and the current leading eigvec is a noise direction, which would make every per-cell driver plot meaningless.
