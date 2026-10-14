# Claude ⇄ Manus coordination protocol (aging-basins arm)

Shared contract for both agents. Claude (local, in this repo) plans,
specifies and verifies; Manus (remote sandbox) executes large-scale
compute. The repo on GitHub is the only channel: everything either
agent needs to know is in a committed file. The user relays one-line
messages between the two; the content of those messages is always
"go read file X on branch Y".

## Roles

| | Claude | Manus |
|---|---|---|
| Owns | `coord/tasks/`, `coord/reviews/`, `coord/BOARD.md`, prereg amendments, library code under `scjdo2/` unless a task grants it | `coord/reports/`, the scripts and outputs a task grants |
| Branch | `aging-basins` | `manus/<task-id>` cut from `aging-basins` at the commit the task names |
| Never | runs the large fits | edits `PREREG_aging_basins.md`, `coord/tasks/`, `coord/reviews/`, pushes to `main` or `aging-basins`, force-pushes |

`MANUS_INSTRUCTIONS_aging_basins.md` is the full method spec. A task
file names the spec sections it executes; if a task and the spec
disagree, the task wins and Manus says so in the report.

## Cycle

1. **Issue.** Claude commits `coord/tasks/<id>.md` with `status: issued`
   on `aging-basins`, updates `BOARD.md`, pushes, and gives the user a
   one-line message for Manus.
2. **Execute.** Manus checks out `aging-basins` at the task's
   `base_commit`, creates `manus/<id>`, does the work, commits per
   spec section, writes `coord/reports/<id>.md` and
   `coord/reports/<id>.json`, pushes, opens a PR
   `manus/<id> → aging-basins` titled `[<id>] <task title>`.
3. **Verify.** Claude fetches the branch, runs
   `python scripts/coord_check.py check <id>`, re-derives the task's
   gate numbers from committed outputs, and writes
   `coord/reviews/<id>.md` with a verdict:
   * `ACCEPT` — Claude merges the PR (with the user's approval) and
     issues the next task;
   * `REVISE` — numbered fixes; Manus pushes to the same branch and
     appends a `## Revision n` section to the report;
   * `STOP` — a prereg gate failed or a rule cannot execute; the user
     decides.
4. **Answer.** If Manus stops on a question, it pushes the report with
   `status: blocked` and the question under `## Questions`. Claude
   answers in `coord/reviews/<id>.md` under `## Answers`; the answer is
   binding as if it were in the task.

One task is open at a time. Tasks are gated: the next is issued only
after the previous is `ACCEPT`.

## Task file (`coord/tasks/<id>.md`, Claude writes)

Front matter `id, title, status (draft|issued|accepted|stopped),
base_commit, spec_sections, depends_on`, then: Goal, Steps, Gate
(numeric pass conditions), Stop-and-ask conditions, and one fenced
`json` contract block that `coord_check.py` reads:

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
  "head_commit": "<sha>",
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
  with the sentence "made before any data run governed by it".
* Report what happened. A failed gate, a crash or a negative result is
  a report, not a reason to change a parameter. Never fabricate,
  simulate or impute data.
* Large files (`*.h5ad`, models, `*.pt`) are gitignored. They are
  listed in `artifacts` with size, sha256 and the exact command that
  regenerates them. Every script is seeded, so a lost sandbox costs
  compute, not results. If the sandbox does not persist between
  tasks, Manus says so in the first report.
* Manus never changes files outside the task's `allowed_paths`.
  `coord_check.py` fails the review if it does.
* Commit subjects carry the spec section (`aging: §3 scArches shared
  space`). Manus commits end with
  `Co-Authored-By: Manus <noreply@manus.im>`.
* Compute beyond the task's budget needs the user's approval first.
