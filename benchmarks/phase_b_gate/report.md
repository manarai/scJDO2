# Phase B gate report

Generated: 2026-10-12 03:56:42

## Pitchfork — fork and holder precision/recall at the ≥ 0.8 target

| τ_crit | fork P | fork R | holder P | holder R | verdict |
|---|---|---|---|---|---|
| 0.1 | 0.09 | 0.89 | 0.00 | 0.00 | **FAIL** |
| 0.3 | 0.10 | 0.99 | nan | 0.00 | **FAIL** |
| 0.5 | 0.10 | 0.87 | 0.86 | 0.01 | **FAIL** |

### Pitchfork — permutation null (sample 15+15)

| τ_crit | real med (near) | null med (near) | overlap (near) | real med (far) | null med (far) | overlap (far) |
|---|---|---|---|---|---|---|
| 0.1 | 0.017 | 0.539 | 0.000 | 0.039 | 0.571 | 0.000 |
| 0.3 | 0.051 | 0.574 | 0.000 | 0.053 | 0.613 | 0.000 |
| 0.5 | 0.022 | 0.543 | 0.000 | 0.025 | 0.512 | 0.000 |

## Paul15 — behaviour counts and null overlap

| velocity prior | holder | drift | fork | damped_osc | growing_osc | frac(Re>0) | null med | real med | overlap mean |
|---|---|---|---|---|---|---|---|---|---|
| True | 0 | 0 | 2004 | 0 | 726 | 1.000 | 0.121 | 0.124 | 0.215 |
| False | 0 | 0 | 1599 | 0 | 1131 | 1.000 | 0.045 | 0.003 | 0.000 |
