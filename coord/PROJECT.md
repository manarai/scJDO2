# Project: scJDO2

| | path | branch |
|---|---|---|
| Claude's folder | `~/Documents/scJDO2` | each workstream's base branch |
| Manus's folder | `~/Documents/scJDO2-manus` | `manus/<workstream>-<ID>` |

**Python:** ≥ 3.10, `.venv` in Manus's folder. Install the project with
`pip install -e ".[compare]"` plus what each task lists.

**Concurrency:** one open task per workstream. Tasks of different
workstreams may run at the same time, each in its own Manus cloud
sandbox (see below); a sandbox runs one task.

## Cloud sandbox runs (`perturb`, `reprog`, `routes`; added 2026-11-08)

Tasks in these workstreams run in **Manus cloud sandboxes**, not on the
user's Mac, so the Mac folder rules in `MANUS.md` (folder
`~/Documents/scJDO2-manus`, reading Claude's folder) do not apply. Same
procedure as orient/S18 (2026-10-21):

* **Code in:** the user uploads `scJDO2_<ws>-<ID>.bundle` (git bundle of
  the task branch). `git clone scJDO2_<ws>-<ID>.bundle scJDO2 && cd scJDO2
  && git checkout manus/<ws>-<ID>`. Never push anywhere.
* **Reviews:** a revision arrives as a new bundle of the same branch with
  the review merged in; read `coord/<ws>/reviews/<ID>.md` from it.
* **Environment:** fresh venv in the sandbox (`python -m venv .venv`,
  `pip install -e ".[dev]"` plus what the task lists). Record
  `pip freeze`, CPU count, RAM and architecture in the report.
* **Data:** download only from the public deposit the task names, record
  the URL, and check the checksum the task gives (or record sha256 when
  the task says so). A mismatch or a failed download is a stop.
* **Long jobs:** the platform kills jobs after about 2 hours. Run every
  step longer than ~10 min as a detached process (`nohup … &`) that
  writes checkpoints and a log, and poll it. Record wall time and peak
  RSS per step.
* **Memory:** sandboxes have had ~18 GB RAM. Read large `.h5ad` files
  backed (`anndata.read_h5ad(..., backed="r")`) and in row chunks; if a
  step would still exceed memory, stop and ask.
* **Return:** commit on the task branch as usual, then
  `git bundle create <ws>-<ID>_result.bundle manus/<ws>-<ID>` and give
  the user that file and the task's gitignored artifacts listed in the
  report as downloads.

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
| `atlas` | `aging-basins` | Common operator-descriptor reference space per biological theme (first: hematopoiesis — Paul15, dynamo, Setty); recurrent motifs across datasets |
| `perturb` | `routes-api` | Does scJDO2's gene coupling, fitted on unperturbed cells only, predict held-out Perturb-seq knockdown responses better than co-expression and trained linear baselines (Replogle K562); then routes to new states (TF Atlas) only if it does |
| `reprog` | `routes-api` | Route search and timing-based drivers on lineage-traced reprogramming (CellTag MEF→iEP; Schiebinger iPSC): do routes predict held-out clone fate and do drivers rank known factors above baselines |
| `routes` | `routes-api` | Check `scjdo2 simulate` walker routes against measured OT transitions between sampling days (LARRY first) |
