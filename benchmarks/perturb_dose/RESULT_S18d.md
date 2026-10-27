# orient/S18d — done

## Summary

Pre-registered RPE1 confirmation: **underpowered**; D1–D3 must hold separately in both N1 and N2.
K562 and medium/strong tertiles are report-only. Every shift is against all controls. Predictor hashes precede perturbed access.

Recorded CPU: 1.021392 h / 16 h. No pushes.

## What ran

Fresh reruns and cache restoration: `benchmarks/perturb_dose/REPRODUCE.md`.

- `.venv/bin/python scripts/s18d_dose.py preflight` — exit 0; wall 131.521s; CPU 0.005273h; max RSS 46804 KiB.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0; wall 27.114s; CPU 0.004136h; max RSS 895552 KiB.
- `.venv/bin/python scripts/s18d_dose.py preflight` — exit 0; wall 137.545s; CPU 0.005466h; max RSS 46684 KiB.
- `.venv/bin/python scripts/s18d_dose.py controls --screen rpe1` — exit 0; wall 23.120s; CPU 0.013023h; max RSS 1936736 KiB.
- `.venv/bin/python scripts/s18d_dose.py subspaces --screen rpe1 --norm N1` — exit 0; wall 183.778s; CPU 0.218621h; max RSS 778712 KiB.
- `.venv/bin/python scripts/s18d_dose.py subspaces --screen rpe1 --norm N2` — exit 0; wall 181.773s; CPU 0.216970h; max RSS 797332 KiB.
- `.venv/bin/python scripts/s18d_dose.py controls --screen k562` — exit 0; wall 26.129s; CPU 0.013203h; max RSS 1831412 KiB.
- `.venv/bin/python scripts/s18d_dose.py subspaces --screen k562 --norm N1` — exit 0; wall 177.751s; CPU 0.215607h; max RSS 798908 KiB.
- `.venv/bin/python scripts/s18d_dose.py subspaces --screen k562 --norm N2` — exit 0; wall 177.741s; CPU 0.216313h; max RSS 781480 KiB.
- `.venv/bin/python scripts/s18d_dose.py freeze-controls` — exit 0; wall 1.004s; CPU 0.000045h; max RSS 31676 KiB.
- `.venv/bin/python scripts/s18d_dose.py select-targets --screen rpe1` — exit 0; wall 1.004s; CPU 0.000288h; max RSS 86296 KiB.
- `.venv/bin/python scripts/s18d_dose.py repair-gene-mapping` — exit 1; wall 5.019s; CPU 0.001177h; max RSS 728272 KiB.
- `.venv/bin/python scripts/s18d_dose.py repair-gene-mapping` — exit 0; wall 5.020s; CPU 0.001419h; max RSS 781228 KiB.
- `.venv/bin/python scripts/s18d_dose.py select-targets --screen k562` — exit 0; wall 1.005s; CPU 0.000287h; max RSS 87664 KiB.
- `.venv/bin/python scripts/s18d_dose.py dose --screen rpe1` — exit 0; wall 52.258s; CPU 0.035958h; max RSS 1080940 KiB.
- `.venv/bin/python scripts/s18d_dose.py capture --screen rpe1 --norm N1` — exit 0; wall 3.012s; CPU 0.001139h; max RSS 262184 KiB.
- `.venv/bin/python scripts/s18d_dose.py capture --screen rpe1 --norm N2` — exit 0; wall 3.012s; CPU 0.001083h; max RSS 257932 KiB.
- `.venv/bin/python scripts/s18d_dose.py dose --screen k562` — exit 0; wall 62.326s; CPU 0.044946h; max RSS 1073544 KiB.
- `.venv/bin/python scripts/s18d_dose.py capture --screen k562 --norm N1` — exit 0; wall 6.033s; CPU 0.002265h; max RSS 391036 KiB.
- `.venv/bin/python scripts/s18d_dose.py capture --screen k562 --norm N2` — exit 0; wall 5.019s; CPU 0.002077h; max RSS 390928 KiB.
- `.venv/bin/python scripts/s18d_dose.py aggregate` — exit 0; wall 6.023s; CPU 0.001491h; max RSS 162212 KiB.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0; wall 16.061s; CPU 0.004109h; max RSS 864492 KiB.
- `.venv/bin/python scripts/s18d_dose.py verify-final` — exit 0; wall 32.123s; CPU 0.008356h; max RSS 318108 KiB.
- `.venv/bin/python scripts/s18d_dose.py report --status done` — exit 0; wall 2.008s; CPU 0.000408h; max RSS 94012 KiB.
- `.venv/bin/python coord/coord_check.py check orient S18d --base ac3f502 --run-reviews` — exit 0; wall 15.061s; CPU 0.004104h; max RSS 890872 KiB.
- `.venv/bin/python scripts/s18d_dose.py prepare-delivery` — exit 0; wall 13.051s; CPU 0.003595h; max RSS 93768 KiB.
- `.venv/bin/python coord/coord_check.py check orient S18d --base ac3f502` — exit 0; wall 1.004s; CPU 0.000033h; max RSS 25732 KiB.

## Gate numbers

| Screen | Norm | n T / M | Median SJ(T) | Median SC(T) | Null95 of T medians | D1 numeric | D2 p (SJ > SC) | D3 p (T > M) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| RPE1 | N1 | 1 / 5 | 0.25876 | 0.131628 | 0.444391 | False | 0.5 | 0.5 |
| RPE1 | N2 | 1 / 5 | 0.268036 | 0.617288 | 0.641648 | False | 1 | 0.833333 |
| K562 (report) | N1 | 14 / 38 | 0.42245 | 0.346701 | 0.439834 | False | 0.270813 | 0.00213672 |
| K562 (report) | N2 | 14 / 38 | 0.339895 | 0.202935 | 0.38679 | False | 0.0863037 | 0.0119425 |

Primary gates: `{"D1": {"by_normalisation": {"N1": false, "N2": false}, "both_normalisations_criterion_met": false, "status": "underpowered", "passed": null}, "D2": {"by_normalisation": {"N1": false, "N2": false}, "both_normalisations_criterion_met": false, "status": "underpowered", "passed": null}, "D3": {"by_normalisation": {"N1": false, "N2": false}, "both_normalisations_criterion_met": false, "status": "underpowered", "passed": null}}`

RPE1 outcome: **underpowered**. Underpowered threshold is fixed at fewer than 10 T targets; no discovery pass is asserted in that case.

Medium/strong report-only comparisons: see `gates.json` and per-target `capture.csv`.

## Deviations

- Authorized planner amendment ac3f502 fixes median null aggregation, array_split tertiles and common accepted original-log-PCA20 significance cohort before any S18d result.
- S18c's accepted controls-only log transform/HVGs are reused from verified retained caches; N1 is fitted in its depth-projected coordinates and N2 residual PCA is fitted controls-only. No score fit is reused across different normalisations.
- Cloud execution reuses .venv and both original files; no Mac, merge, push, protected-source edit or test modification.
- Fewer than 10 qualifying RPE1 T targets: fixed underpowered rule. Defined raw descriptive medians/tests are still reported, but no confirmatory gate pass is claimed.
- An initial gene-ID/symbol implementation mismatch was corrected before perturbed X access: raw var.index is Ensembl IDs, while var.gene_name contains target symbols. Initial empty qualification diagnostic preserved; frozen subspaces, numerical control arrays and hashes unchanged; no refit or significance retest.
- A final-audit indentation SyntaxError was fixed before the freeze command could execute; no scientific unit was repeated.

## Observations

- All controls are retained in every displacement reference (RPE1 11,485; K562 10,691).
- Both screens' control-only predictors and 10,000 covariance-matched QR bases were saved and hashed before any S18d perturbed expression access.
- RPE1 qualifying T=1, M=5; fixed outcome: underpowered.
- RPE1 N1 weak T median score-J 0.25876, covariance 0.131628, shared-basis median-null95 0.444391.
- RPE1 N2 weak T median score-J 0.268036, covariance 0.617288, shared-basis median-null95 0.641648.
- K562 N1 weak T median score-J 0.42245, covariance 0.346701, shared-basis median-null95 0.439834.
- K562 N2 weak T median score-J 0.339895, covariance 0.202935, shared-basis median-null95 0.38679.
- RPE1 N1 depth-plane removal leaves rank 18 in the 20-dimensional coordinates; the literal fixed stiffness-eigenvalue positivity rule was not changed.
- K562 N1 depth-plane removal leaves rank 18 in the 20-dimensional coordinates; the literal fixed stiffness-eigenvalue positivity rule was not changed.
- K562 report-only weak T>M score-J comparisons have one-sided p=0.002136716264471399 (N1) and 0.011942540127144782 (N2); its D1 and D2 numerical criteria do not hold in either normalisation.

## Suggestions

- The planner could consider a separately pre-registered higher-cell-count RPE1 screen to address the one-qualifying-T-target power limitation; no follow-up, parameter change or cohort relaxation was run.

## Questions

None. Approved conventions are committed in the task.

## Artifacts

`benchmarks/perturb_dose/` contains subspaces and SHA-256 file, dose/capture tables, gates, control metadata, immutable target lists and exclusions, exact cache files and timed receipts. Figures are under `benchmarks/figures/perturb_dose/`. Raw H5ADs are excluded from the delivery ZIP; original URLs, sizes and SHA-256 are recorded in `REPRODUCE.md` and integrity metadata. Complete artifact sizes, SHA-256 and regeneration/restoration instructions are in the JSON report.
