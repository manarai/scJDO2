# S14b — test construction failure

## Outcome

Step 1 exhausted the frozen transition reserve (seeds 5–16) with **2 eligible seeds** (s12 and s16), below the required 4. The task explicitly says to report construction failure and stop when this happens. No null calibration, transition or held-out flow fits, or truth scoring were run. Therefore `T_null` and gates X1–X4 are unavailable (`null`), and no profile figure or S14 comparison values can be produced.

## Eligibility

| Seed | Status | Peak bin | Bootstrap stability | Result / reason |
|---:|---|---:|---:|---|
| 5 | assessed | 5 | 0.580 | ineligible: stability < 0.90 |
| 6 | assessed | 6 | 0.560 | ineligible: stability < 0.90 |
| 7 | assessed | 5 | 0.700 | ineligible: stability < 0.90 |
| 8 | assessed | 4 | 0.955 | ineligible: at least one bin has <50 cells |
| 9 | assessed | 1 | 0.355 | ineligible: peak bin outside 2–9; stability <0.90 |
| 10 | assessed | 0 | 0.310 | ineligible: peak bin outside 2–9; stability <0.90 |
| 11 | assessed | 5 | 0.565 | ineligible: stability <0.90 |
| 12 | assessed | 5 | 0.900 | **eligible** |
| 13 | generation failed after retry | — | — | both attempts failed with `ValueError: n < 0`; no dataset or manifest row |
| 14 | stopped at wall limit | — | — | 17,293.73 s wall; no dataset or manifest row |
| 15 | stopped at memory cap | — | — | stopped as RSS approached 14 GB; measured peak 14,727,905,280 bytes; no dataset or manifest row |
| 16 | assessed | 5 | 0.915 | **eligible** |

Every assessed seed had complete strict-root provenance. Full bin counts, profiles, checksums, and criterion booleans are in `g2_transition_eligibility.csv`.

## Compute and versions

Recorded S14b checkpoint CPU time totals 31,469.43 seconds (**8.74 CPU-hours**), including generation attempts and truth-only eligibility work. Generation s16 completed at 13,996,392,448 bytes peak RSS; preparation s16 used 6,007,963,648 bytes. Runtime: Python 3.13.2, NumPy 2.5.4, pandas 2.3.3, AnnData 0.11.4, Scanpy 1.12.4, Palantir 1.4.5, SciPy 1.18.1, scikit-learn 1.9.1; sandbox thread limit 3.

## Outputs

- `g2_transition_eligibility.csv` is committed before any flow fit.
- `null_calibration.csv` records that Step 2 was not run.
- `test.csv` records that Step 3 was not run.
- `gates.json` leaves thresholds and gate results null; this is not a gate pass or failure.
