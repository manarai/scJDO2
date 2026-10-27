**Supplementary Table S3 | S18d: dose response of score-J capture of knockdown displacements (Replogle CRISPRi).** For each target, cells were split into knockdown tertiles; capture = fraction of the tertile's centroid shift from all controls (PCA-20 of each normalisation) lying in the three control score-J soft directions (S_J) or the three top control-covariance directions (S_C). T: transcription-machinery targets; M: mitochondrial/translation targets (medians over targets). null95: 95th percentile of the across-T median capture under 10,000 covariance-matched random 3-dimensional subspaces (one shared basis per draw). Criteria (fixed in the task): D1 median S_J(T) > null95; D2 S_J > S_C over T, one-sided paired Wilcoxon p < 0.05; D3 S_J(T) > S_J(M), one-sided Mann–Whitney p < 0.05. N1: depth-projected log-normalisation; N2: analytic Pearson residuals. **Pre-registered:** RPE1, weak tertile, D1–D3 each required in both N1 and N2, with fewer than 10 qualifying T targets reported as underpowered. RPE1 had n_T = 1 (POLR2I) and n_M = 5: outcome **underpowered**, no gate claimed. K562 (n_T = 14, n_M = 38) and the medium and strong tertiles are **report only**. Source: `benchmarks/perturb_dose/gates.json` (medians re-derived from `capture_{screen}_{norm}.csv`; p-values recomputed).

| screen | norm. | tertile | status | n T / M | median S_J(T) | median S_C(T) | null95 | D1 | D2 | D3 | p Wilcoxon (S_J > S_C) | p Mann–Whitney (T > M) |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| RPE1 | N1 | weak | pre-registered (underpowered) | 1 / 5 | 0.259 | 0.132 | 0.444 | no | no | no | 0.5 | 0.5 |
| RPE1 | N1 | medium | report only | 1 / 5 | 0.191 | 0.281 | 0.503 | no | no | no | 1 | 0.667 |
| RPE1 | N1 | strong | report only | 1 / 5 | 0.200 | 0.307 | 0.518 | no | no | no | 1 | 0.667 |
| RPE1 | N2 | weak | pre-registered (underpowered) | 1 / 5 | 0.268 | 0.617 | 0.642 | no | no | no | 1 | 0.833 |
| RPE1 | N2 | medium | report only | 1 / 5 | 0.286 | 0.767 | 0.707 | no | no | no | 1 | 1 |
| RPE1 | N2 | strong | report only | 1 / 5 | 0.300 | 0.697 | 0.672 | no | no | no | 1 | 1 |
| K562 | N1 | weak | report only | 14 / 38 | 0.422 | 0.347 | 0.440 | no | no | yes | 0.271 | 0.00214 |
| K562 | N1 | medium | report only | 14 / 38 | 0.410 | 0.362 | 0.482 | no | no | yes | 0.357 | 0.000172 |
| K562 | N1 | strong | report only | 14 / 38 | 0.452 | 0.338 | 0.484 | no | yes | yes | 0.0123 | 5.04e-05 |
| K562 | N2 | weak | report only | 14 / 38 | 0.340 | 0.203 | 0.387 | no | no | yes | 0.0863 | 0.0119 |
| K562 | N2 | medium | report only | 14 / 38 | 0.363 | 0.158 | 0.416 | no | yes | yes | 0.0392 | 0.00243 |
| K562 | N2 | strong | report only | 14 / 38 | 0.367 | 0.167 | 0.419 | no | yes | yes | 0.0176 | 0.000579 |

Overall outcome (pre-registered RPE1): **underpowered** (D1–D3 status: D1 underpowered, D2 underpowered, D3 underpowered).
