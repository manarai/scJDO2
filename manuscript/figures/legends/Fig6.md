**Figure 6 | The score operator: a density-curvature descriptor, not validated against perturbation or lineage (S10, S18, S18b, S22, S18d).**
**a**, S10: fork AUC of score J (−β∇²E, energy network fitted by exact-Laplacian score matching) vs OT flow J on four non-stationary v3
presets (PCA-20, 3 seeds each). Score-J AUC on default 0.16–0.31, stress
0.15–0.50 and batch 0.17–0.29 (AUC < 0.45, i.e. committed cells ranked above saddle cells, on
8 of 9 seeds), 0.43–0.52 on transition;
OT flow J 0.97–1.00 on default.
**b**, S18 (pre-registered; P1 failed): Replogle K562 CRISPRi; all fits on 10,691 non-targeting controls; 865 of 1299
targets with ≥ 100 cells passed the BH-adjusted permutation test. Fraction of each knockdown's centroid shift captured by m = 3 directions
in control PCA-20: local score-J soft axes 0.045, global score J 0.038, local covariance
0.292, global covariance 0.589 (medians); random-subspace expectation 0.15.
P1 (fraction of targets above their random-null 95th percentile ≥ 0.70): 0.053. k = 10/15/30 (report only): local score J
0.099/0.056/0.051.
**c**, S18b (pre-registered; both gates failed). Left, H1: fraction of each knockdown's energy distance explained by re-weighting control states
(k = 20): median 0.31 (required ≥ 0.5); 235 of 865 targets (0.27) above the size-matched noise 95th percentile.
Right, H2: score-J basin stiffness vs occupancy sensitivity across 59 basins; partial Spearman ρ (ranks residualised on basin size and
covariance trace) 0.05, one-sided p (negative) 0.65 (10,000 permutations; recomputed ρ matches h2.json);
ordinary ρ 0.06; operator-free Leiden spread ρ 0.03 (p⁻ 0.60).
**d**, Left, S22 (pre-registered; L1 failed, L2 failed, L3 failed): Spearman ρ between a LARRY clone's median
score-J λ₁ on day *t* and the entropy H of its sister cells' types on day *t* + 2 (filled: seed 0, gated; open: seeds 1–2, report only), next to the
local-density and distance-to-Undifferentiated-centroid baselines. Pooled (1,818 clone/time pairs, 67 with tied modal types): λ₁ ρ
−0.003 (one-sided permutation p 1); Undifferentiated (1,401 pairs): λ₁ ρ −0.10
(p 1), centroid distance 0.22, density 0.08; partial ρ of λ₁ given density
−0.08. λ₁ ρ over seeds 0–2 and both cohorts: −0.130 to −0.003.
Right, S18d: fraction of the weak-knockdown-tertile centroid shift of each transcription-machinery (T) target captured by three control score-J soft directions
(vermillion) or three control-covariance directions (gray); bars, medians over T targets; dashed, 95th percentile of the across-T median under 10,000
covariance-matched random subspaces (shared basis). N1, depth-projected log-normalization; N2, analytic Pearson residuals. RPE1 (pre-registered; underpowered:
1 qualifying T target, ≥ 10 required): score J 0.26/0.27 (N1/N2), covariance
0.13/0.62, null95 0.44/0.64. K562 (report only; 14 T, 38 M targets): score J
0.42/0.34, covariance 0.35/0.20, null95 0.44/0.39; score J above
covariance, one-sided paired Wilcoxon p 0.27/0.09; T above mitochondrial/translation (M) targets, one-sided Mann–Whitney
p 0.002/0.012. Supplementary Tables S2 and S3.
