**Figure 3 | The flow operator from a time course alone ranks branching cells, untuned (S10, S12–S15, S14b).**
Flow J = Jacobian of MeasuredDrift fitted to entropic-OT barycentric velocities between sampling days (days 2/4/6) in PCA
of log-normalised counts. Fork AUC = AUC of the leading eigenvalue λ₁ for true saddle vs committed cells (truth from generator
state, used for scoring only).
**a**, S14 held-out test (pre-registered; test seeds never used for selection): 11 test datasets
(v3_transition s2–s4, v3_default s2–s3, v3_batch s0–s2, v3_stress s0–s2) at k = 10, 15, 20. Reference (a-priori) configuration:
AUC 0.89–0.97 on v3_transition and 0.97–1.00 on the other presets.
Selected configuration (tuned one factor at a time on two development seeds): 0.42–0.95 on
v3_transition (gate G1, ≥ 0.85 on every transition seed at k* = 15 and neighbours, failed; minimum 0.42).
**b**, S10 (pre-registered gate on labelling flow J; other operators report only): fork AUC per operator on v3 presets (PCA-20; 3 seeds
each; filled = v3_default). v3_default: OT flow J 0.97–1.00, labelling flow J
0.53–0.89 (raw and smoothed variants), scVelo flow J
0.27–0.51, score J 0.16–0.31.
**c**, Left, S12 Step 0 (6 datasets: v3_default and v3_transition s0–s2): OT flow J fork AUC by embedding — PCA-10
0.88–1.00, PCA-20 0.82–1.00, PCA-50 0.56–0.97,
scVI-10 0.31–0.96, FA-20 0.04–0.53, FA-20 with noise-whitened cost
0.21–0.97. Right, S13 (pre-registered; gate failed) on v3_transition s0–s2: whitening the embedding
(PCA-20 0.37–0.58), whitening only the drift fit
(PCA-10 0.79–0.96), whitening only the OT
cost (PCA-10 0.53–0.79, PCA-20
0.28–0.43).
**d**, Flow J vs the Jacobian of the generator's true expected velocity in the same PCA-15 coordinates (J_ref; S15 Step 0;
4 test datasets × J_ref neighbourhood 15/30): norm ratio 0.02–0.07,
median per-cell cosine −0.002 to 0.008, per-cell λ₁ Spearman
−0.11 to 0.00, relative error after optimal scalar c* 1.00.
The reference was verified by finite differences of the true expected velocity (median per-cell cosine ≥ 1.000).
Across S14 test rows the median relative Frobenius error is 1.000–1.005.
**e**, Left, *post hoc*: peak-to-median ratio R of the 12-bin day-weighted strict-pseudotime λ₁ profile (reference, k = 15) on the
three transition test seeds (1.91–2.69; |detected − true bin|, in parentheses in the panel, 1, 0, 5)
and three no-fork seeds (0.58–1.44). Right, S14b and S14c construction (pre-registered): bootstrap stability of the true critical peak bin per reserve seed s5–s40
(dashed: 0.90 rule; blue: eligible; hatched: fails only the ≥ 50-cells-per-bin rule, seeds 8, 29, 36 with smallest bins of
42, 46, 46 cells; ×: generation stopped by the memory guard (14 seeds); ▽: other generation failures (seeds
13, 14, 15)). Of 19 assessed seeds, 2 were eligible (seeds
12, 16; none of the 10 assessed S14c seeds) against the 4 required, so the confirmatory transition
test could not be constructed and no model was fitted (S14b and S14c construction failures; gate values null).
