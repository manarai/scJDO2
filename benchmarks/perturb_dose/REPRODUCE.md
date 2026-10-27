# orient/S18d — restoration and execution

This is a pre-registered confirmation with planner amendment revision 1, authorized on 2026-10-22 before any S18d result. Do not change cohorts, definitions, thresholds, model settings, or numerical conventions after examining results. K562 and medium/strong tertiles are report-only. D1–D3 must hold separately in BOTH normalisations, with at least 10 qualifying RPE1 T targets.

## Preferred: restore verified completed outputs

Clone `S18d_result.bundle` and check out `manus/orient-S18d`. Extract `S18d_benchmarks.zip` into the repository root. It contains `benchmarks/perturb_dose/` including saved control transformations, model weights, per-control H, frozen subspaces and 10,000 null bases, raw-dose-derived tables, shifts, median-null distributions, qualification/exclusion lists, figures and exact accepted source snapshots. It also contains the final reports and script. Verify artifact SHA-256 values against `coord/orient/reports/S18d.json` and subspaces against `benchmarks/perturb_dose/subspaces.sha256`. Do not repeat completed units; the script intentionally refuses existing scientific artifacts/checkpoints.

The returned bundle has a complete history for the task branch; it is not the small incremental input bundle. The authorized planner task amendment is the only change outside ordinary S18d-owned paths. No merges, pushes or Mac operations were performed.

## Original read-only inputs (not included in the ZIP)

Replogle 2022 Figshare article 20029387:

| Screen | File/download URL | Bytes | SHA-256 |
|---|---|---:|---|
| K562 | `https://ndownloader.figshare.com/files/35773219` — `K562_essential_raw_singlecell_01.h5ad` | 10661879995 | `3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee` |
| RPE1 | `https://ndownloader.figshare.com/files/35775606` — `rpe1_raw_singlecell_01.h5ad` | 8700873216 | `9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1` |

Expected repository locations are `benchmarks/perturb_k562/data/K562_essential_raw_singlecell_01.h5ad` and `benchmarks/perturb_valleys/data/rpe1_raw_singlecell_01.h5ad`. Never modify these original files. Stop if either size or SHA-256 differs. Integrity hashing reads bytes only; it is not scientific access to perturbed expression.

## Environment

Use a repository-local `.venv`, Python ≥3.10; the recorded run used Python 3.12.3. Install `pip install -e ".[dev]"` into that environment only. Exact packages are recorded in `pip-freeze.txt`; hardware and threads are in `environment.json` and unit receipts. Cloud rules supersede Mac launchers/resource splits. The runner uses at most six threads, conservatively monitors an 8 GiB process-tree RSS ceiling, and limits total compute to 16 CPU hours.

## Fresh-tree scientific regeneration (only if deliberately requested)

Use a separate fresh output tree/repository with originals and reference inputs restored, not the completed output directory. Source snapshots of the accepted S18c control transforms and original significance tables are part of this archive. Controls-only reference input caches are in `reference_inputs/`. Accepted energy tests are reused exactly, not rerun in the new normalisations. The T/M symbol patterns are in the task and script; target symbols are resolved using the dataset-provided var.gene_name column (var.index contains Ensembl IDs); no alias or imputed target-gene mapping is used.

Every command below is one detached unit through the same timed runner. Check back and commit after each unit; never run a second scientific unit concurrently. Prefix with `.venv/bin/python scripts/s18d_dose.py unit <unique-unit-name> --` to produce a resource/time/checkpoint receipt. The first input preflight and mandatory tests must pass before scientific commands.

1. `preflight`, then `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py`. Existing test/input receipts belong to the recorded execution, not a fresh run.
2. `controls --screen rpe1` and `controls --screen k562`: select only original non-targeting rows; reuse verified accepted log-HVG/PCA20 transforms; fit the N1 control depth regression and N2 analytic residual PCA. No perturbed X or coordinates are opened.
3. For each screen and each N1/N2, `subspaces --screen <screen> --norm <norm>`. S02 configuration is unchanged: hidden96, depth3, epochs1500, seed0, batch256, exact ESM, lr0.0005, beta1.0. The operator is mean H across all controls. Use literal positive stiffness eigenvalues for the soft selection; do not threshold tiny eigenvalues or reduce N1 from 20 to 18 coordinates.
4. `freeze-controls`: verify every control-only score/covariance basis and 10,000 random QR bases, save the global predictor barrier with SHA-256 and git checkpoint. This MUST precede ANY perturbed scientific access in either screen.
5. `select-targets --screen <screen>` for both screens: accepted original log-PCA20 BH<0.01, ≥150 cells, target gene detected in ≥50% controls. Save T/M lists and every exclusion before capture. One common cohort in N1/N2.
6. `dose --screen <screen>`: frozen projections, exact raw target/full-library knockdown, ties by original row, `numpy.array_split` tertiles. This is the first S18d raw perturbed expression access.
7. `capture --screen <screen> --norm <norm>` for four screen/normalisation combinations: all-control centroids, score/covariance captures and shared-basis random-null medians. Do not resample bases or qualify targets after seeing capture.
8. `aggregate`: fixed gate tables, per-cell/per-target aggregates and deterministic figures. Underpowered is not a gate pass; raw defined numbers are retained descriptively.
9. Required post-analysis tests, then `verify-final`: recheck original hashes, fixed tertiles, frozen hashes, all-control shifts, published CSV round-trip reproduction of raw captures/tests and predictor ordering.
10. `report --status done`, commit, then `.venv/bin/python coord/coord_check.py check orient S18d --base <planner-amendment-commit> --run-reviews`. The approved amendment is used as checker baseline; `verify-final` separately audits all changes against imported commit f3e166e, allowing only that explicit task amendment in addition to ordinary task paths.

`report` and `figures` are metadata/presentation regenerations from immutable computed results, not refits or new tests. `prepare-delivery` and the saved `finalize_driver.py` are bounded packaging steps; the latter saves the final external timed delivery receipt without dirtying the branch after bundle creation. That external receipt is included in the archive and its final CPU total is recorded in the delivery manifest.
