# Claude ⇄ Manus coordination protocol (aging-basins arm)

Shared contract for both agents. Claude plans, specifies and verifies;
Manus executes the large-scale compute. Both work on the user's
computer, in two folders of the **same git repository**:

| folder | who | branch |
|---|---|---|
| `~/Documents/scJDO2` | Claude | `aging-basins` |
| `~/Documents/scJDO2-manus` (a git worktree) | Manus | `manus/<task-id>` |

Separate folders mean neither agent's branch switches, installs or
half-written files can break the other's. Same repository means no
pushing, pulling or pull requests: each side reads the other's commits
directly. GitHub is only a backup, pushed when the user asks.

The user relays one-line messages; their content is always "read file
X". Everything an agent needs is in a committed file.

## Roles

| | Claude | Manus |
|---|---|---|
| Writes | `coord/tasks/`, `coord/reviews/`, `coord/BOARD.md`, PREREG amendments, library code unless a task grants it | `coord/reports/`, the scripts and outputs a task grants |
| Works in | `~/Documents/scJDO2` | `~/Documents/scJDO2-manus` only |
| Never | runs the large fits | writes anything in `~/Documents/scJDO2`; edits PREREG, tasks, reviews or BOARD; merges into `aging-basins`; pushes to GitHub |

`MANUS_INSTRUCTIONS_aging_basins.md` is the full method spec. A task
names the spec sections it executes; if a task and the spec disagree,
the task wins and Manus says so in the report.

## Cycle

1. **Issue.** Claude commits `coord/tasks/<id>.md` (`status: issued`)
   on `aging-basins` and gives the user a one-line message for Manus.
2. **Execute.** In its folder, Manus starts the task branch from the
   current `aging-basins` (`git switch -c manus/<id> aging-basins`),
   runs the steps, commits per spec section, writes
   `coord/reports/<id>.md` and `.json`, commits them, and tells the
   user it is done.
3. **Verify.** Claude runs `python scripts/coord_check.py check <id>`,
   re-derives the gate numbers from the committed outputs, runs the
   tests in Manus's folder, and writes `coord/reviews/<id>.md`:
   * `ACCEPT` — Claude merges `manus/<id>` into `aging-basins`
     (`--no-ff`, keeping Manus's per-section commits) and issues the
     next task;
   * `REVISE` — numbered fixes; Manus commits them on the same branch
     and appends `## Revision n` to the report;
   * `STOP` — a prereg gate failed or a rule cannot execute; the user
     decides.
4. **Answer.** If Manus is stuck, it commits the report with
   `status: blocked` and the question under `## Questions`. Claude
   answers in `coord/reviews/<id>.md` under `## Answers`; the answer is
   binding as if it were in the task.

Reviews are written after the task branch was cut, so Manus reads
them from Claude's folder: `~/Documents/scJDO2/coord/reviews/<id>.md`
(read-only).

One task is open at a time; the next is issued only after `ACCEPT`.

## Task file (`coord/tasks/<id>.md`, Claude writes)

Front matter `id, title, owner, status (draft|issued|accepted|stopped),
base_commit, spec_sections, depends_on`, then: Goal, Steps, Gate,
Stop-and-ask conditions, and one fenced `json` contract block that
`coord_check.py` reads:

```json
{
  "allowed_paths": ["glob", "..."],
  "required_files": ["committed path that must exist on the branch"],
  "required_metrics": ["key that must appear in the report json"],
  "compute_budget_cpu_hours": 24
}
```

## Report (Manus writes)

`coord/reports/<id>.json`, machine-read:

```json
{
  "task": "A01",
  "status": "done | blocked | partial",
  "branch": "manus/A01",
  "started_utc": "...", "finished_utc": "...",
  "compute": {"cpu_hours": 0.0, "gpu_hours": 0.0, "hardware": "..."},
  "metrics": {"<required_metric>": "<value>"},
  "artifacts": [
    {"path": "...", "bytes": 0, "sha256": "...", "committed": false,
     "regenerate": "python scripts/... --tissue lung"}
  ],
  "deviations": ["what differed from the task and why"],
  "questions": []
}
```

`coord/reports/<id>.md`, human-read, sections in order: Summary
(≤ 10 lines), What ran (commands, in order), Gate numbers (table),
Deviations, Questions, Artifacts.

## Rules that bind both sides

* Pre-registered rules in `PREREG_aging_basins.md` are frozen. Only
  Claude writes amendments, only before the run they govern, dated,
  with "made before any data run governed by it".
* Report what happened. A failed gate, a crash or a negative result is
  a report, not a reason to change a parameter. Never fabricate,
  simulate or impute data.
* Large files (`*.h5ad`, models, `*.pt`) are gitignored and stay on
  disk in `~/Documents/scJDO2-manus`; later tasks reuse them there.
  List them in `artifacts` with size, sha256 and the command that
  regenerates them. Every script is seeded.
* Manus's Python environment is `~/Documents/scJDO2-manus/.venv`
  (gitignored). Nobody installs packages into the user's base or
  conda environments.
* Manus never changes files outside the task's `allowed_paths`;
  `coord_check.py` fails the review if it does.
* Commit subjects carry the spec section (`aging: §3 scArches shared
  space`). Manus commits end with
  `Co-Authored-By: Manus <noreply@manus.im>`.
* Compute beyond the task's budget needs the user's approval first.
