# S05 — synthetic generator v2: completed results

## What ran

* Generated or deterministically reused all four v2 presets × seeds 0–2; retained artifacts were re-generated into isolated directories and had identical `obs` plus all layers.
* Ran frozen O1–O3 on the 12 v2 H5ADs and the three v1 `default` references.
* Ran O4 once on `v2_default_s0` by importing the unchanged S03 scVelo workflow.
* Ran `python -m pytest -q tests/test_orient.py tests/test_smoke.py tests/test_synthetic_multimodal.py`: 63 passed, 0 failed; one non-fatal `FactorAnalysis` convergence warning.

## Fixed gates

| dataset | O1 true-rate cosine | O2 true-γ cosine | unspliced share | detected multi-day clones | on genes qualifying | status |
|---|---:|---:|---:|---:|---:|---|
| v2_default_s0 | 0.631690 / ≥0.40 | 0.501203 / ≥0.30 | 0.1764 | 367 | 1.000 | pass |
| v2_default_s1 | 0.640256 / ≥0.40 | 0.494447 / ≥0.30 | 0.1476 | 355 | 1.000 | pass |
| v2_default_s2 | 0.471460 / ≥0.40 | 0.308470 / ≥0.30 | 0.1842 | 378 | 1.000 | pass |

## V1 and v2 realism ledger

| dataset | UMI/cell median | genes/cell median | unspliced share | unspliced zeros | new/total | barcode fraction | multi-day clones (fraction) |
|---|---:|---:|---:|---:|---:|---:|---:|
| default_s0 | 4614 | 481 | 0.0205 | 0.9470 | 0.2779 | 0.6002 | 209 (0.7712) |
| default_s1 | 4325 | 489 | 0.0172 | 0.9478 | 0.2848 | 0.6008 | 191 (0.7127) |
| default_s2 | 2971 | 489 | 0.0250 | 0.9392 | 0.2942 | 0.5965 | 208 (0.7820) |
| v2_default_s0 | 4945 | 482 | 0.1764 | 0.7237 | 0.2480 | 0.6035 | 367 (0.1991) |
| v2_default_s1 | 5214 | 488 | 0.1476 | 0.7144 | 0.2577 | 0.6090 | 355 (0.1897) |
| v2_default_s2 | 3322 | 486 | 0.1842 | 0.7063 | 0.2393 | 0.5950 | 378 (0.2022) |
| v2_stress_s0 | 5080 | 481 | 0.1698 | 0.7403 | 0.1238 | 0.3073 | 136 (0.1212) |
| v2_stress_s1 | 5080 | 489 | 0.1374 | 0.7319 | 0.1284 | 0.2996 | 112 (0.0993) |
| v2_stress_s2 | 3268 | 484 | 0.1713 | 0.7257 | 0.1282 | 0.2931 | 110 (0.0973) |
| v2_batch_s0 | 5457 | 479 | 0.1764 | 0.7219 | 0.2521 | 0.6101 | 353 (0.1929) |
| v2_batch_s1 | 5166 | 487 | 0.1468 | 0.7130 | 0.2555 | 0.6071 | 348 (0.1923) |
| v2_batch_s2 | 3565 | 481 | 0.1849 | 0.7070 | 0.2470 | 0.5903 | 356 (0.1941) |
| v2_stationary_cycle_s0 | 1367 | 428 | 0.1475 | 0.8104 | 0.2102 | 0.5760 | 0 (0.0000) |
| v2_stationary_cycle_s1 | 1482 | 421 | 0.1513 | 0.8077 | 0.2062 | 0.5920 | 0 (0.0000) |
| v2_stationary_cycle_s2 | 1401 | 419 | 0.1548 | 0.8035 | 0.2124 | 0.5961 | 0 (0.0000) |

## Oracle cosines

| dataset | O1 true rates | O2 true γ | O3 fitted γ (report-only) |
|---|---:|---:|---:|
| default_s0 | 0.284504 | 0.491637 | -0.154088 |
| default_s1 | 0.258408 | 0.463103 | -0.159198 |
| default_s2 | 0.176445 | 0.374961 | -0.097518 |
| v2_default_s0 | 0.631690 | 0.501203 | 0.015577 |
| v2_default_s1 | 0.640256 | 0.494447 | 0.007796 |
| v2_default_s2 | 0.471460 | 0.308470 | -0.015912 |
| v2_stress_s0 | 0.706547 | 0.368750 | 0.218242 |
| v2_stress_s1 | 0.708546 | 0.378024 | 0.235822 |
| v2_stress_s2 | 0.564437 | 0.221709 | 0.150693 |
| v2_batch_s0 | 0.663355 | 0.527852 | -0.010552 |
| v2_batch_s1 | 0.650569 | 0.509127 | 0.069252 |
| v2_batch_s2 | 0.513725 | 0.334037 | -0.009394 |
| v2_stationary_cycle_s0 | 0.947274 | 0.807848 | 0.912928 |
| v2_stationary_cycle_s1 | 0.956120 | 0.831283 | 0.926529 |
| v2_stationary_cycle_s2 | 0.943243 | 0.758947 | 0.882299 |

## O4 scVelo dynamical (report-only)

* `v2_default_s0` overall PCA cosine: **-0.031067**.
* By day: day 1 = 0.287882; day 2 = -0.373273; day 3 = 0.028783; day 4 = -0.033230.

## Resource record

| dataset | generation wall minutes | peak RSS GiB |
|---|---:|---:|
| v2_default_s0 | 4.405 | 3.217 |
| v2_default_s1 | 4.431 | 3.304 |
| v2_default_s2 | 4.378 | 3.561 |
| v2_stress_s0 | 4.184 | 3.125 |
| v2_stress_s1 | 4.425 | 3.212 |
| v2_stress_s2 | 4.752 | 3.678 |
| v2_batch_s0 | 4.224 | 3.456 |
| v2_batch_s1 | 4.270 | 3.424 |
| v2_batch_s2 | 4.504 | 3.457 |
| v2_stationary_cycle_s0 | 0.556 | 1.377 |
| v2_stationary_cycle_s1 | 0.469 | 1.563 |
| v2_stationary_cycle_s2 | 0.475 | 1.308 |

Measured Revision 1 generator wall time: 0.6845 h. Measured Revision 1 CPU time including generation, audits, O4 and tests: 0.7763 h.

## Deviations

The previous run was stopped after an environment-stalled `v2_stress_s2` process. Revision 1 followed the committed amended one-at-a-time procedure; `v2_stress_s2` completed in 4.752 minutes and 3.678 GiB RSS. No task parameters were changed.

## Artifacts

* `benchmarks/synthetic_multimodal/MANIFEST_v2.csv` — 12 H5AD records and SHA-256 hashes.
* `benchmarks/synthetic_multimodal/v2_audit/audit.csv` — 61 rows (15 realism rows, 45 O1–O3 rows, and one O4 row).
* `benchmarks/synthetic_multimodal/v2_audit/revision1_timing.jsonl` and `audit_timing.jsonl` — generation/reuse and audit timing provenance.
* The local H5ADs under `benchmarks/synthetic_multimodal/data/v2/` are gitignored by project policy.
