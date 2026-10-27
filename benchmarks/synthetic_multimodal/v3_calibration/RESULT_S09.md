# orient/S09 — blocked before calibration completion

## Status

Blocked by the fixed 16 CPU-hour task budget. No LHS batch after batch 3, no coordinate refinement, no fixed v3 knobs, no final v3 datasets, S08 comparison, or S05 oracle measurements were run.

## Completed calibration ledger

- Seeded LHS vectors ranked: **32 / 64**
- Calibration attempts: **34** (32 ranked completions; 2 resolved first-attempt retries; 0 terminal exclusions)
- Measured calibration CPU: **2.343783 h**
- Mean successful evaluation CPU: **257.543 s**
- Peak observed resident memory: **8.779 GB**
- Best currently observed candidate: `lhs_000` with 8 unmet target statistics and normalized distance 51.373291; it is not a calibrated final vector.

## Budget stop

Using the observed successful-evaluation mean for the remaining 162 required evaluations and retaining the documented final-measurement reserve, the no-additional-retry projection is **16.019812 CPU-hours**, exceeding the fixed 16-hour cap by **0.019812 CPU-hours**. The task requires stopping before further compute in this condition.

## Retry record

- `lhs_007`: first attempt had scVelo `EOFError`; the one allowed same-vector retry succeeded.
- `lhs_012`: first attempt exposed a v3 whole-gene rounding boundary (4,001 discrete allocations for 4,000 genes). The additive allocation repair was tested; the one allowed same-vector retry succeeded.
- No candidate is terminally excluded.

## Required work not run

The remaining 32 LHS candidates, 130 coordinate-refinement evaluations, v3 preset fixation, 12 final datasets, S08 comparison/figures, and S05 O1–O4 are not run. Their gate values are therefore not available.

## Required schema sentinels

`MANIFEST_v3.csv`, `compare_v3.csv`, and `oracles_v3.csv` have only their production schemas and zero data rows. `knobs.json` records `status: blocked_not_selected`. These are not measurements or selections; they exist to make the blocked state auditable without fabricating values.

## Revision 2 (2026-10-19) — LHS complete, stopped before refinement

- LHS: **64 / 64** ranked (lhs_032–063 in Revision 2: 0 failures, 0 retries, 0 excluded).
- Best LHS vector: `lhs_058`, 6 / 13 targets unmet (C2, C4, G2, L1, L2, S1), normalised distance 67.394. Not selected (refinement not run).
- Projected total 24.195 CPU-h > 24 CPU-h budget (`admission_plan_r2.json`); coordinate refinement, v3_default fixation, final datasets, S08 comparison and S05 oracles not run.
- Revision 2 CPU is whole-process-tree (`/usr/bin/time -l`, `process_cpu.csv`); the prior 2.344 h ledger excluded loky worker CPU (estimated 1.091 h).

## Revision 3 (2026-10-19) — refinement complete, blocked on the size rule

- Refinement 130 / 130 (0 failures; 10 identical-vector reuses). Selected `refine_p2_n_founders_3`: 5 / 13 unmet (C2, C5, G2, L2, S1), distance 33.350.
- Size rule: f_sample 0.151 → n_founders = 200 (lower bound) gives 12 164 collected cells (seed 0), outside 4 000–6 000 (`rescale_n_founders.json`). Final measurement not run.
- G2 unmet in all 184 unique vectors (1–45); its knobs were not at bounds, so it is not classified unreachable.
- CPU ≈ 16.61 h of 40 (whole process trees, incl. estimated 1.091 h prior unmetered).

## Revision 4 (2026-10-19) — block refinement, v3_default fixed, final measurement stopped on memory

- Blocks: 1 not applied (S1-meeting points have 7-8 unmet); 2 base kept (block2_04 excluded after 2 ARPACK failures); 3 accepted n_founders 300, f_sample 0.0519 (8 576 cells); 4 base kept. v3_default = block3_confirm (5 / 13 unmet at reduced size).
- v3_default full size, seeds 0/1/2: 5 / 6 / 6 of 13 met (realism gate fail); O1 0.157 / 0.121 / 0.151 (< 0.30, fail); O2 0.785 / 0.806 / 0.774 (pass).
- Stopped: v3_default_s0 peak RSS 17.18 GB > 16 GB. v3_stress, v3_batch, v3_stationary_cycle and the S08 comparison not run.
- CPU ≈ 25.63 h of 40.

## Revision 5 (2026-10-20) — re-ranked selection, full final measurement

- Bands: single-reference targets x [1/1.25, 1.25] (L1 0.155–0.254, L2 4.05–6.34 h, G1 0.301–0.471, G2 3.2–5.0, G3 0.531–0.830); S1 required.
- Re-rank of 257 unique vectors: 6 meet S1; winner `block1_09` (7 unmet). Lineage step: `r5block3_27` (n_founders 800, f_sample 0.0289, 12 701 cells).
- v3_default full size, seeds 0/1/2: 6 / 6 / 6 of 13 met (realism gate fail); O1 0.283 / 0.212 / 0.283 (fail, < 0.30); O2 0.432 / 0.454 / 0.536 (pass).
- S08 match rate: v3_default 0.385, v3_stress 0.337, v3_batch 0.389, v3_stationary_cycle 0.489 (v2_default 0.216, v1 0.352).
- CPU 35.08 h measured (about 36.7 h with two unrecorded killed units) of 40.
