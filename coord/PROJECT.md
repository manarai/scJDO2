# Project: scJDO2

| | path | branch |
|---|---|---|
| Claude's folder | `~/Documents/scJDO2` | each workstream's base branch |
| Manus's folder | `~/Documents/scJDO2-manus` | `manus/<workstream>-<ID>` |

**Python:** ≥ 3.10, `.venv` in Manus's folder. Install the project with
`pip install -e ".[compare]"` plus what each task lists.

**Concurrency:** one open task per workstream; Manus runs one task at a
time.

**Project-wide rules**

* `main` is never changed by Manus, and nothing is merged into `main`
  from a workstream without the user.
* `tests/test_opspace_pitchfork.py` needs the v1 `scjdo` package; its
  failures from `ModuleNotFoundError: scjdo` are known and not a stop.

## Workstreams

| workstream | base branch | goal |
|---|---|---|
| `aging` | `aging-basins` | Does aging loosen cell-type basins along specific directions (TMS lung + marrow), per PREREG_aging_basins.md |
| `hlca` | `aging-basins` | Same aging-basin tests on human lung: HLCA core (Sikkema, Theis et al., Nat Med 2023) with scArches reference mapping; compare with mouse |
