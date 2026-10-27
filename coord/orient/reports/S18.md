# orient/S18 — K562 perturbation response

## Summary

Status: done. Original K562 SHA-256 verified; all fits used only 10691 non-targeting controls.
Significant targets: 865/1299.
P1 FAIL: 46/865 = 0.053179; required >=0.70.
The highest median capture among the three comparison methods is **global_covariance**.
Measured completed-unit CPU hours: 1.100443 (budget 12).

## What ran

- `initial timing utility failed` — exit 127, CPU-h 0.000000.
- `.venv/bin/pip install --no-cache-dir -e .[dev]` — exit 0, CPU-h 0.043234.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0, CPU-h 0.004742.
- `.venv/bin/python scripts/s18_perturb_response.py download` — exit 0, CPU-h 0.014316.
- `.venv/bin/python scripts/s18_perturb_response.py embed` — exit 0, CPU-h 0.029168.
- `.venv/bin/python scripts/s18_perturb_response.py effects` — exit 1, CPU-h 0.000635.
- `.venv/bin/python scripts/s18_perturb_response.py repair-labels` — exit 0, CPU-h 0.001950.
- `.venv/bin/python scripts/s18_perturb_response.py effects` — exit 0, CPU-h 0.195980.
- `.venv/bin/python scripts/s18_perturb_response.py fit-capture --k 10` — exit 0, CPU-h 0.111901.
- `.venv/bin/python scripts/s18_perturb_response.py fit-capture --k 15` — exit 0, CPU-h 0.161249.
- `.venv/bin/python scripts/s18_perturb_response.py fit-capture --k 20` — exit 0, CPU-h 0.213703.
- `.venv/bin/python scripts/s18_perturb_response.py fit-capture --k 30` — exit 0, CPU-h 0.314896.
- `.venv/bin/python scripts/s18_perturb_response.py summarize` — exit 0, CPU-h 0.001726.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0, CPU-h 0.004009.
- `.venv/bin/python scripts/s18_perturb_response.py verify-final` — exit 0, CPU-h 0.002865.
- `.venv/bin/python scripts/s18_perturb_response.py report` — exit 0, CPU-h 0.000068.

Environment: fresh `.venv`, Python 3.12.3, 6 CPUs; complete `pip freeze` is committed as `benchmarks/perturb_k562/pip-freeze.txt`. RAM seen in timed units (bytes): [8419680256, 18754445312, 25599549440]; RAM at report (bytes): 25599549440. The sandbox allocation autoscaled during setup/download. Peak timed process RSS: 3105136 KiB. CPU hours are measured child user+system time for the recorded units, not wall time multiplied by available cores.

The embedding uses control-median all-gene library normalization, log1p, 2,000 seurat_v3 control-count HVGs and exact control-only PCA. No gene scaling or whitening was added. EnergyDriftAdapter uses the literal S02 configuration for each k. Local operators, local covariance and observed shifts use exactly the same 200 nearest controls. Random nulls use 1,000 actual QR-generated subspaces per k (nested m=1,3,5).

## Gate numbers

| Metric | Value | Requirement / result |
|---|---:|---|
| P1 fraction above target random-null 95th percentile | 0.053179 | >=0.70; FAIL |
| Median capture: local_score_J | 0.044680 | Report only |
| Median capture: global_score_J | 0.037833 | Report only |
| Median capture: local_covariance | 0.292426 | Report only |
| Median capture: global_covariance | 0.588926 | Report only |
| Median paired difference: local_score_J_minus_local_covariance | -0.215488 | Wilcoxon two-sided p=5.2484395e-121 |
| Median paired difference: local_score_J_minus_global_covariance | -0.518645 | Wilcoxon two-sided p=1.3423983e-136 |
| Median paired difference: local_covariance_minus_global_covariance | -0.276779 | Wilcoxon two-sided p=6.3347459e-124 |

All k/m sensitivities and per-target counts/effects are in `summary.json` and `capture.csv`. These sensitivity values are report-only, not additional gates.

## Deviations

- Mac folder, launchers and fixed split replaced by the task's cloud amendment; fresh .venv and six available threads were used.
- The cloud image lacked /usr/bin/time; the initial setup failed before pip ran. A Python resource/monotonic timer was used for every subsequent unit.
- The provided branch contains no xfail annotations in tests/test_orient.py; the exact mandated command passed all 48 tests before analysis. Protected tests were not changed.
- Exact full control-only PCA SVD computed once and truncated to k=10,15,20,30; this yields the same nested PCA-k axes without repeating the identical full decomposition.
- The branch-only input bundle omitted orientation. For the mechanical checker only, a local orientation ref was created at the exact imported pre-run snapshot, 4c1aad9c627ac4371ced46525baac956529260d9; no merge or push occurred.
- The first effect unit failed before any statistic/RNG draw because our cached labels had object dtype. Labels were reserialized from the verified raw obs as Unicode, with every fitted numeric array verified bitwise unchanged. No fit or completed statistic was repeated.

## Observations

- Original SHA-256 matches exactly; 10691 controls among 310385 cells.
- 865 of 1299 targets with >=100 cells passed BH-adjusted permutation p<0.01.
- With 200 permutations and the add-one p-value, the smallest attainable p is 1/201; selection resolution is recorded, not changed.
- P1 local score-J null exceedance fraction is 0.053179.
- Among local score J, local covariance and global covariance, global_covariance has the highest median capture at k=20,m=3.
- GATA1 is the largest energy-distance target (2.93, 108 cells); its local score-J capture is 0.788 versus 0.607 for local covariance. The example targets were chosen by effect size, not by score-J performance.

## Suggestions

- In a separate approved task, evaluate held-out control score-matching fit quality and control-only axis stability to distinguish score-estimation limitations from failure of the soft-axis response hypothesis. No such fit or analysis was run here.

## Questions

- None.

## Artifacts

Committed results: `targets.csv`, `capture.csv`, per-k captures, provenance, unit receipts/logs, environment freeze, this report and the two figures. Large derived caches remain uncommitted and accompany the results download. The public 10.66 GB raw input is excluded from the results archive; its URL and exact checksum are recorded.

Original data: https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387
Download: https://ndownloader.figshare.com/files/35773219
SHA-256: `3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee`
