# S22 restoration and execution

This is the issued S22 protocol plus planner Amendment 1 dated 2026-10-24. Gate definitions and fit settings are unchanged. Only seed 0 supplies the gates.

## Restore the delivered result

The result is an incremental Git bundle requiring `e90440cfa3d87aa9c76ae0a0c6be697712ac6094`. Verify it in a repository that has that prerequisite, then fetch `manus/orient-S22` from the bundle. No remote push is needed.

The input is not in the result bundle. Restore `larry_s22_inputs.npz` from the user-provided `scJDO2_orient-S22.zip`, then verify SHA-256:

`74f01c2a9121531ac4a0af156c2b96e5be63d1af1957075377277f57249286c8`

Place it at `benchmarks/real_flow/data/larry_s22_inputs.npz`. Its source ZIP and standalone input bundle IDs, sizes and checksums are in `source_archive.json`. Do not overwrite the original input. No raw H5AD is needed for this task or included in the result.

The saved result includes exact fitted weights, per-cell eigenvalues, clone-level tables and the 1,000-permutation null arrays. Restoring results does not require any model refit. `scripts/s22_score_lineage.py audit` verifies the original input, saved spectrum/clone values, amended tie handling, observed correlations and permutation p-values without fitting any model.

## Fresh execution only

Do not run `fit` in a delivered result tree: it intentionally refuses to overwrite existing weights. For a fresh computation, start from the issued prerequisite in a separate clean workspace containing the amended task, S22 script, `preregistration.json`, `amendment_record.json` and the preflight metadata, but no fitted model or scientific output. Preserve the same protected S11 and energy-adapter sources. Create the project venv in that workspace and use the recorded dependency versions (`pip-freeze.txt` and report `versions`). Never delete committed result files to force a rerun.

Every unit below is run separately, detached and timed, with a checkpoint reviewed before the next unit. The script sets three native threads and enforces an 8 GiB per-process RSS limit and the fixed 8 CPU-hour task budget.

```bash
.venv/bin/python scripts/s22_score_lineage.py unit tests_before -- .venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py
.venv/bin/python scripts/s22_score_lineage.py unit score_seed0 -- .venv/bin/python scripts/s22_score_lineage.py fit --seed 0
.venv/bin/python scripts/s22_score_lineage.py unit score_seed1 -- .venv/bin/python scripts/s22_score_lineage.py fit --seed 1
.venv/bin/python scripts/s22_score_lineage.py unit score_seed2 -- .venv/bin/python scripts/s22_score_lineage.py fit --seed 2
.venv/bin/python scripts/s22_score_lineage.py unit clones -- .venv/bin/python scripts/s22_score_lineage.py clones
.venv/bin/python scripts/s22_score_lineage.py unit baselines -- .venv/bin/python scripts/s22_score_lineage.py baselines
.venv/bin/python scripts/s22_score_lineage.py unit statistics_seed0 -- .venv/bin/python scripts/s22_score_lineage.py scores --seed 0
.venv/bin/python scripts/s22_score_lineage.py unit statistics_seed1 -- .venv/bin/python scripts/s22_score_lineage.py scores --seed 1
.venv/bin/python scripts/s22_score_lineage.py unit statistics_seed2 -- .venv/bin/python scripts/s22_score_lineage.py scores --seed 2
.venv/bin/python scripts/s22_score_lineage.py unit summarize -- .venv/bin/python scripts/s22_score_lineage.py summarize
.venv/bin/python scripts/s22_score_lineage.py unit tests_after -- .venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py
.venv/bin/python scripts/s22_score_lineage.py unit audit -- .venv/bin/python scripts/s22_score_lineage.py audit
```

Commit completed steps, write the final report, and run the required mechanical checker/review command with the authorized planner-amendment commit as the explicit baseline. The script also audits the full execution diff against the original `e90440c` snapshot, allowing only the exact authorized task amendment outside the task's ordinary output paths. This avoids counting already-imported coordinator changes from an older local `orientation` ref as S22 execution changes; it does not merge or modify any protected source.

## Frozen technical conventions

Fit on the full delivered 12,000-row sample (4,000 per day), in input order, float32, with all untouched `EnergyDriftConfig` defaults. Evaluate the protected autodiff Jacobian with internal chunk 128 and use `eigvalsh` directly. Float64 arithmetic is used for clone medians and distances; CSVs use 17 significant digits and round-trip parsing. Average ranks are used for Spearman ties. Partial Spearman is the Pearson correlation of separately residualized ranks given ranked density and an intercept.

Each reported cohort's null resets `default_rng(0)`, performs 1,000 permutations within `(t, permutation_type)`, and uses the positive one-sided +1 tail. Tied modal types receive no chosen majority; their pooled permutation label is `tied`, separately by t. They are excluded from Undifferentiated/per-type strata, and their switch fraction is measured outside the full set of modal types. The pooled test without ties is report only. These conventions were recorded before any fit or fate statistic.
