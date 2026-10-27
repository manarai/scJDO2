# orient/S22 — done

## Summary

All fixed scientific units completed; gates are reported unchanged. Input SHA-256 is `74f01c2a9121531ac4a0af156c2b96e5be63d1af1957075377277f57249286c8`. Gated results use seed 0 only; seeds 1–2 and the untied-only pooled check are report only. No tuning was performed.

## What ran

- `S22_download_and_checksums`: exit 0; 5.324 seconds wall; 0.00033610 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `S22_bundle_import_and_input_placement`: exit 0; 2.965 seconds wall; 0.00071539 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `tests_before`: exit 0; 24.092 seconds wall; 0.00408886 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `input_preflight`: exit 0; 1.004 seconds wall; 0.00027680 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `index_metadata`: exit 0; 1.004 seconds wall; 0.00008880 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `tests_before_amendment`: exit 0; 70.262 seconds wall; 0.00414512 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `score_seed0`: exit 0; 284.043 seconds wall; 0.18413845 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `score_seed1`: exit 0; 278.993 seconds wall; 0.18390317 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `score_seed2`: exit 0; 280.999 seconds wall; 0.18523987 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `clones`: exit 0; 2.008 seconds wall; 0.00040058 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `baselines`: exit 0; 6.022 seconds wall; 0.00260632 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `statistics_seed0`: exit 0; 3.012 seconds wall; 0.00060860 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `statistics_seed1`: exit 0; 3.012 seconds wall; 0.00061457 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `statistics_seed2`: exit 0; 2.008 seconds wall; 0.00060990 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `summarize`: exit 0; 8.030 seconds wall; 0.00207726 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `tests_after`: exit 0; 19.069 seconds wall; 0.00410516 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `audit`: exit 1; 1.004 seconds wall; 0.00018417 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `audit_import_repaired`: exit 0; 7.026 seconds wall; 0.00182923 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `report_final`: exit 0; 1.004 seconds wall; 0.00005018 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `coord_check`: exit 0; 20.081 seconds wall; 0.00406745 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.
- `coord_check_final`: exit 0; 16.064 seconds wall; 0.00412083 CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.

Completed measured unit CPU use: 0.58420680 / 8 CPU-hours.

### Fixed methods

Fresh S11-exact EnergyDriftAdapter fits on the delivered 12,000-row sample, seeds 0, 1 and 2. Autodiff Jacobians use the protected adapter, internal chunk 128; eigenvalues use `eigvalsh` directly. The S11 symmetry check is on 100 training rows with check seed 0, threshold max_asym ≤ 1e-4. Clone medians are computed over all its day-t cells; H uses natural logarithms over next-time type frequencies. Tied pairs receive no majority label; their permutation type is `tied`, separately by t, and switch counts next-time types outside all tied modal types. Baselines use all same-day cells, exclude the focal cell from 30 Euclidean neighbours, and use the all-day-2 Undifferentiated centroid. Partial Spearman uses average ranks and OLS residualization on ranked density plus an intercept. Each reported cohort resets `default_rng(0)` for 1,000 restricted permutations and uses the positive one-sided +1 tail; undefined correlations are null, never imputed.

## Gate numbers

| Gate | Fixed condition | Observed seed-0 result | Pass |
|---|---|---|---|
| L1 | pooled ρ > 0; p < 0.01 | ρ=-0.00264272; p=1; n=1818 | False |
| L2 | Undifferentiated ρ > 0; p < 0.01 | ρ=-0.101161; p=1; n=1401 | False |
| L3 | abs(ρλ) > both abs(ρbaseline); partial ρ > 0 | λ=-0.101161; density=0.0763236; centroid=0.221025; partial=-0.0777412 | False |

### Seeds 1–2 and removal of ties — report only

| Seed | Pooled ρ / p | Undifferentiated ρ / p | Partial ρ | Pooled without ties ρ / p |
|---|---|---|---|---|
| 0 | -0.00264272 / 1 | -0.101161 / 1 | -0.0777412 | 0.0014121 / 0.999001 |
| 1 | -0.0184252 / 0.999001 | -0.110547 / 1 | -0.0910604 | -0.00926023 / 1 |
| 2 | -0.0347663 / 1 | -0.129688 / 1 | -0.110943 | -0.0272647 / 1 |

## Deviations

- Planner Amendment 1 was supplied on 2026-10-24 and appended/committed before all score fits and fate statistics; no majority label was chosen for tied pairs.
- The unchanged delivered tests have 48 passing tests and no xfails; the workstream text mentions three strict xfails.
- The cloud bundle imported a newer issue snapshot than the local orientation ref. Mechanical review uses the authorized planner amendment commit as --base; the full execution diff is also audited against e90440c, allowing only the exact authorized task amendment. No merge, push or Mac access occurred.
- The auxiliary final audit initially failed before numerical checks because coord is not an importable package from the script directory. Only audit module loading was repaired to use the unchanged checker by absolute repository path; the failed receipt is preserved. Scientific function bodies and all saved scientific files were verified unchanged, and no model was refitted.
- A pre-commit artifact assertion caught autogenerated, ignored Python bytecode incorrectly included in the report manifest. The bytecode cache was excluded from the manifest only; saved scientific files and function bodies were verified unchanged. The completed checker/test receipt was retained and final review rerun after committing the corrected reports.

## Observations

- The fit sample is exactly 12,000 delivered rows: 4,000 each on days 2, 4 and 6.
- Every one of the 19,623 barcoded day-2/day-4 cells was included in each completed score-J query; no lineage/type filtering was used for fitting.
- Seed 0: max symmetry asymmetry 2.57674e-07; median cell lambda1 0.589464; positive-eigenvalue count distribution (count: cells) {'0': 4141, '1': 12380, '2': 3095, '3': 7}.
- Seed 1: max symmetry asymmetry 2.17993e-07; median cell lambda1 0.58655; positive-eigenvalue count distribution (count: cells) {'0': 3751, '1': 13240, '2': 2618, '3': 14}.
- Seed 2: max symmetry asymmetry 2.58655e-07; median cell lambda1 0.526145; positive-eigenvalue count distribution (count: cells) {'0': 4674, '1': 13287, '2': 1662}.
- The eligible cohort has 1818 clone/time pairs from 1694 distinct clones; 67 tied pairs were retained in L1 and report-only per-t pooled tests in their own per-t permutation strata, and excluded from Undifferentiated and per-type tests.
- Tied pairs per t: {'2': 3, '4': 64}; untied Undifferentiated pairs per t: {'2': 227, '4': 1174}.
- 798 pairs have H=0; observed entropy range is [-0.0, 1.5047882836811908].

## Suggestions

None. Only the issued task and planner amendment were executed.

## Questions

None.

## Artifacts

The result bundle contains the S22 script, exact saved model weights, per-cell eigenspectra and λ₁/eigen-count tables, clone truth and baseline tables, restricted permutation null arrays, summary statistics, figures and unit receipts. Input NPZ and raw H5ADs are not included. Artifact sizes/SHA-256 values and regeneration commands are in the JSON report. To reproduce, restore the verified input to the task data path, install the project into its venv, and run `fit --seed 0/1/2`, `clones`, `baselines`, `scores --seed 0/1/2`, and `summarize` as separately timed units in a fresh tree; never refit over existing saved weights.
