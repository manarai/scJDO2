# Phase B gate report

Generated: 2026-10-12 03:42:25

## Pitchfork — fork and holder precision/recall at the ≥ 0.8 target

| τ_crit | fork P | fork R | holder P | holder R | verdict |
|---|---|---|---|---|---|
| 0.1 | 0.08 | 0.42 | 0.76 | 0.26 | **FAIL** |
| 0.3 | 0.13 | 0.74 | 0.83 | 0.22 | **FAIL** |
| 0.5 | 0.11 | 0.62 | 0.84 | 0.30 | **FAIL** |

### Pitchfork — permutation null (sample 15+15)

| τ_crit | real med (near) | null med (near) | overlap (near) | real med (far) | null med (far) | overlap (far) |
|---|---|---|---|---|---|---|
| 0.1 | -0.005 | 0.539 | 0.000 | 0.029 | 0.571 | 0.000 |
| 0.3 | 0.031 | 0.574 | 0.000 | 0.072 | 0.613 | 0.000 |
| 0.5 | 0.022 | 0.543 | 0.000 | -0.010 | 0.512 | 0.000 |

## Paul15 — behaviour counts and null overlap

| velocity prior | holder | drift | fork | damped_osc | growing_osc | frac(Re>0) | null med | real med | overlap mean |
|---|---|---|---|---|---|---|---|---|---|
| True | 0 | 0 | 1612 | 0 | 1118 | 1.000 | 0.121 | 0.055 | 0.028 |
| False | 0 | 0 | 1607 | 0 | 1123 | 1.000 | 0.045 | 0.006 | 0.001 |
