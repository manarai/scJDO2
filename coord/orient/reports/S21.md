# S21

## Summary

Accepted regenerated bifurcating seeds: 10, 16, 18.

## What ran

Three accepted bifurcating_v2 datasets plus cycle_s0–2 and linear_s0–2; PCA-15 observed-only preparation; flow-J drift seeds 0, 1, 2; cached S10 score-J per dataset; truth scoring after all fits.

## Gate numbers

F1 fork AUC gate: FAIL; minimum accepted-bifurcating flow-J AUC = 1.0.

F2 rotation gate: FAIL.

## Deviations

F1 is not evaluable for accepted seeds 16 and 18 because the contract-defined fork region is empty (n_fork=0); F2 fails for cycle_s1 at all drift seeds and cycle_s2 at drift seed 2. Score-J and J_ref remain report-only.

## Observations

See scores.csv and jref_diagnosis.csv for operator and region-level values.

## Suggestions

Review report-only score-J and J_ref diagnostics before using them for a gate.

## Questions

None.

## Artifacts

- benchmarks/dyngen/s21/part_a_seeds.csv
- benchmarks/dyngen/s21/scores.csv
- benchmarks/dyngen/s21/jref_diagnosis.csv
- benchmarks/figures/dyngen/s21_lambda1_pca.png
- benchmarks/figures/dyngen/s21_auc_bars.png
