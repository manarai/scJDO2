**Figure 4 | An independent simulator (dyngen; S21b, pre-registered).** Nine dyngen datasets (bifurcating seeds 10/16/18,
cycle s0–s2, linear s0–s2; 5,000 cells, 1,000 genes), re-sampled from the saved S21 simulations at eight synchronised time points;
same untuned settings as S14's reference flow J (PCA-15, entropic OT, MeasuredDrift drift seeds 0–2) and S10 score J; truth read
only after all fits. **a**, Fork AUC (λ₁, fork vs committed cells): flow J 1.00,
0.92–0.99 and 0.31–0.38 (inverted) on
s10, s16, s18; score J 0.98, 0.58, 0.85.
Gate F1 (≥ 0.85) holds on 2 of 3 datasets (gate failed).
**b**, Median rotational index max|Im λ|/max|λ| of flow J, cycle (filled) vs linear (open) per dataset seed and drift seed (lines pair
the same drift seed): cycle s1 0.19–0.32 and s2 0.36–0.40 vs linear
0.07–0.09; cycle s0
0.02 vs linear s0 0.08–0.10 (gate F2 failed: 2 of 3
cycles pass on every seed). Score J is symmetric (rotational index 0.00).
**c**, Fraction of cells with Re λ₁ > 0 (flow J) by truth region (report only): on linear backbones the fraction is 0 in the early and
committed regions for every seed, and 0 over all 5,000 cells (maximum 0.0000 over 9 dataset × seed
rows, RESULT_S21b.md). **d**, Cells in the fork region at three time points (S21) vs eight (S21b):
s10 22 vs 329, s16 0 vs
746, s18 0 vs 776.
Diagnosis of the inverted dataset (s18): pending.
