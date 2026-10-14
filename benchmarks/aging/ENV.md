# A01 environment

## Runtime

| Item | Value |
|---|---|
| Python | 3.13.2 (conda-forge build; isolated `.venv`) |
| scjdo2 | 0.1.0.dev1 (editable install) |
| PyTorch | 2.14.1 |
| scvi-tools | 1.5.1 |
| Scanpy | 1.12.4 |
| cellxgene-census | 1.18.0 |
| scArches | 0.6.1 |
| AnnData | 0.10.9 |
| NumPy | 2.5.4 |
| pandas | 2.3.3 |
| pytest | 9.1.1 |

## Hardware and storage

| Item | Value |
|---|---|
| Host | MacBook Pro (Apple M3 Max) |
| CPU | 14 cores (10 performance, 4 efficiency) |
| GPU | Apple M3 Max, 30 GPU cores; Metal 4; PyTorch MPS available |
| RAM | 36 GB (38,654,705,664 bytes) |
| Disk available at task start | 387 GiB |
| OS | macOS 26.7.1 (arm64) |
| CUDA | Not available |
| Persistence | Yes — this is a local git worktree; `.venv` and gitignored large outputs remain in `~/Documents/scJDO2-manus` across this task's sessions. |

## Installation

Created `~/Documents/scJDO2-manus/.venv`, installed the editable project with
`pip install -e ".[compare]"`, then installed `cellxgene-census`,
`scvi-tools`, `scanpy`, `scarches`, and `pytest` only into that environment.
`pytest` was added because it is required to run the task's mandated test
commands and is not a project runtime dependency.

## Test baseline

| Command | Result |
|---|---|
| `python -m pytest tests/test_compare_aging.py tests/test_measured_flow_figures.py -q` | **10 passed** in 25.50 s |
| `python -m pytest tests -q` | **24 passed, 10 xfailed, 6 failed** in 6.28 s |

All six full-suite failures were in `tests/test_opspace_pitchfork.py` and each
was caused by `ModuleNotFoundError: No module named 'scjdo'`. This is the
predeclared v1-package absence in `coord/PROJECT.md` and A01; no other test
failed. `tests/test_compare_aging.py` passed, satisfying the required
pre-data test gate.
