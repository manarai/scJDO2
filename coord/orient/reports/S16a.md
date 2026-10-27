# S16a baseline runs

This main section records the original round-0 status; the appended **Revision 1** section below contains the current aggregate status and updates.

## Scope and inputs

The five full v3 datasets were checked against the read-only agent A manifest before use. All SHA-256 values matched. No source dataset was copied or modified, and no truth fields were used for fitting or scoring.

Strict Palantir pseudotime and the S10 root rule were saved for all five datasets in `benchmarks/baselines/scjdo_v1/pseudotime/`. The two s2 arrays came from the authorized S10 `day2_strict_r0` outputs, aligned by their `names` key. Transition s3/s4 and default s3 were computed with S03 settings (`n_components=10`, `knn=30`, `num_waypoints=500`, `seed=0`, strict mode). The selected roots were:

| Dataset | Root cell | Pseudotime source |
| --- | --- | --- |
| v3_transition_s2 | cell_0004186 | S10 `day2_strict_r0` |
| v3_transition_s3 | cell_0001681 | S03 settings, strict |
| v3_transition_s4 | cell_0000711 | S03 settings, strict |
| v3_default_s2 | cell_0004186 | S10 `day2_strict_r0` |
| v3_default_s3 | cell_0001681 | S03 settings, strict |

The v1 wrapper passes the same root to `prepare_trajectory`, because `_run_fit_drift` has no external pseudotime argument. The pinned v56 worktree and `.venv-v1` were installed. The first transition s2 attempt accidentally resolved dependencies from the task's primary venv due wrapper path precedence; it was stopped at 4,679/5,000 epochs after 61:17 (peak RSS 8,278,179,840 bytes), and no output was retained. I fixed the wrapper's path ordering and verified AnnData, Torch, and scJDO resolve from `.venv-v1`/the v56 worktree. The corrected retry failed before fitting because Scanpy import raised a Numba cache locator error for `scanpy/get/_kernels.py`. Following the instruction to stop v1 after another environment failure, I did not change package versions or try more v1 datasets.

## Baseline results

Two CellRank runs produced operator outputs. `v3_transition_s2` completed the moscot RealTimeKernel plus ConnectivityKernel weighted combination (`0.8/0.2`) and GPCCA with five macrostates, terminal states, and fate probabilities. Its measured peak RSS was 8,528,297,984 bytes. `v3_transition_s3` completed with the observed PCA-30 representation and the same settings, but its measured peak RSS was 8,799,666,176 bytes, above the 8 GiB per-process cap. I stopped further CellRank runs after that measurement. The S3 output is retained and marked `complete_over_cap` in the run manifest. The saved transition operators use CSR storage with float32 values.

The Dynamo one-shot run on transition s2 stopped in `recipe_one_shot_data`/`velocity_N` with `AssertionError: #existing gene is 0`; its measured peak RSS was 12,771,672,064 bytes, above the cap. I did not continue to additional Dynamo or scVelo dynamical runs.

The spliceJAC default estimator on transition s2 stopped during its default preprocessing with `TypeError: normalize_per_cell() got an unexpected keyword argument 'n_top_genes'` from the installed scVelo version. I left package versions unchanged and did not continue the same failing call on the other seeds.

The v1 import failure is recorded in `runs.csv`; no v1 fit completed. The wrapper uses the strict Palantir root cell as `prepare_trajectory`'s root because the v1 core has no external pseudotime parameter. The live `/Users/terooatt/Documents/scJDO` checkout remained at `3ffa1d0aec66cd9087bf6b006e2355b1cc3c2064`; its user files were left untouched.

No baseline results were scored against truth. Per-dataset statuses, settings, versions, errors, runtime, and memory measurements are in [`runs.csv`](../../../benchmarks/baselines/runs.csv). The `.venv-baselines` freeze is in [`pip-freeze.txt`](../../../benchmarks/baselines/pip-freeze.txt).

## Metrics

- `runs_completed`: 2 baseline operator runs (CellRank transition s2 and s3; s3 exceeded the memory cap).
- `runs_failed`: 3 attempted baselines (Dynamo one-shot, spliceJAC, and the corrected v1 import attempt). The incomplete v1 run using the wrong dependency path is documented separately and is not counted as a completed baseline.
- `support_runs_completed`: 5 strict Palantir root/pseudotime outputs.
- `versions`: CellRank 2.3.3, moscot 0.5.2, spliceJAC 0.0.1, scVelo 0.3.4, Scanpy 1.12.4, AnnData 0.13.4, Dynamo 1.5.3, Palantir 1.4.5, scJDO v1 0.3.0, Numba 0.68.0 in `.venv-v1`.
- `cpu_hours`: 1.734 including timed CellRank and Dynamo runs, the incomplete v1 attempt, and the corrected v1 import failure; excludes the untimed spliceJAC failure, Palantir helper runs, and environment setup.


## Verification

The required coordination check passes. The requested test suites pass: 48 tests passed across `tests/test_orient.py` and `tests/test_smoke.py`. All five read-only source files were SHA-256 verified again after the runs and match agent A's manifest. The live v1 checkout remains at `3ffa1d0aec66cd9087bf6b006e2355b1cc3c2064` with the same 11 untracked user paths recorded before installation.

## Revision 1

### CellRank at the 14 GiB process limit

The three remaining datasets completed with 3 threads. Each was run as a separate timed process. Saved operators and metadata are under `benchmarks/baselines/cellrank/`.

| Dataset | Wall time | Peak RSS | Result |
| --- | ---: | ---: | --- |
| v3_transition_s4 | 411.49 s | 7,998,849,024 bytes | Complete |
| v3_default_s2 | 416.39 s | 7,002,423,296 bytes | Complete; GPCCA reported non-convergence, used GMRES fallback, and predicted a 17-cell terminal state where 30 cells were requested |
| v3_default_s3 | 937.23 s | 8,235,335,680 bytes | Complete |

The earlier `v3_transition_s3` CellRank output peaked at 8,799,666,176 bytes. It exceeded the former 8 GiB limit but is below the revised 14 GiB limit, so it is accepted as complete under this revision.

### Dynamo one-shot diagnosis and retry

The prior `#existing gene is 0` assertion is raised in Dynamo's CSC velocity estimator, in `dynamo/estimation/csc/velocity.py::get_n_genes`, during one-shot `dynamics` estimation when the estimator's internal data mapping contains no non-`None` matrices. This is downstream of preprocessing; it does not establish that the basic gene-count filter removed every gene. In the read-only transition s2 input, 2,456 genes meet the shared-count threshold of 30, so that threshold alone does not explain the empty estimator input.

I retried the documented `recipe_one_shot_data` path with `n_top_genes=4000`, `keep_filtered_genes=True`, and `keep_raw_layers=True`. Preprocessing logged 1,562 genes flagged as outliers and zero outlier cells, then reached the spliced/unspliced moment calculation. The process was stopped at 90.90 s after peak RSS reached 15,471,050,752 bytes, above the 14 GiB limit; the recipe did not return and no operator file was produced. Dynamo was not patched, and scVelo dynamical was not run.

### spliceJAC environment and runs

Created `.venv-splicejac` with Python 3.11, Scanpy 1.9.8 (the newest 1.9.x release), AnnData 0.9.2, scVelo 0.2.5, SciPy 1.13.1, NumPy 1.26.4, pandas 2.2.3, and scikit-learn 1.5.2. The freeze is recorded at `benchmarks/baselines/splicejac/pip-freeze.txt`.

The first compatibility trial exposed SciPy 1.17's removal of sparse `.A`; pinning SciPy 1.13.1 restored the API used by scVelo 0.2.5. The next trial exposed spliceJAC's documented prerequisite for `transition_genes`; the runner now calls `scanpy.tl.rank_genes_groups(groupby="clusters")` before transition-gene extraction. No package source was patched. All five runs then completed:

| Dataset | Wall time | Peak RSS | Leiden clusters | Transition pairs |
| --- | ---: | ---: | ---: | ---: |
| v3_transition_s2 | 44.53 s | 8,745,058,304 bytes | 8 | 56 |
| v3_transition_s3 | 41.01 s | 8,707,637,248 bytes | 8 | 56 |
| v3_transition_s4 | 42.17 s | 8,756,166,656 bytes | 8 | 56 |
| v3_default_s2 | 51.81 s | 8,748,711,936 bytes | 12 | 132 |
| v3_default_s3 | 53.03 s | 8,589,344,768 bytes | 14 | 182 |

### scJDO v1 final retry

The single final retry used the corrected v1 `sys.path` ordering and `NUMBA_CACHE_DIR=.sandbox/cache/numba`. It trained all 5,000 epochs and completed 101 per-cell Jacobian batches, but exited with status 1 (`time: signal: Invalid argument`) before saving an operator NPZ/JSON. Runtime was 3,149.75 s and peak RSS was 9,837,363,200 bytes. The output is recorded as a failed final retry; v1 was stopped without another attempt.

No outputs were scored against truth. The updated per-run settings, statuses, versions, wall times, and peak RSS values are in `benchmarks/baselines/runs.csv`.


### Revision metrics and verification

Revision totals are 10 completed baseline outputs (five CellRank and five spliceJAC), two failed outputs (the Dynamo one-shot stop and final v1 retry), and 13 baseline combinations not run (remaining Dynamo, all scVelo dynamical, and remaining v1 combinations). The aggregate `cpu_hours` metric is 3.216. `coord/coord_check.py check orient S16a` passed; `tests/test_orient.py` and `tests/test_smoke.py` passed (48 tests). All five source dataset hashes still match agent A's read-only manifest. The live `/Users/terooatt/Documents/scJDO` checkout remains at `3ffa1d0` with the same 11 untracked user paths. No truth scoring was performed.
