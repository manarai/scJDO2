# Claude ⇄ Manus coordination protocol

Generic contract, identical in every project. Project specifics live in
`coord/PROJECT.md`; analysis specifics in `coord/<workstream>/WORKSTREAM.md`.

Claude plans, specifies and verifies. Manus executes the large-scale
analysis. Both work on the user's computer in two folders of the
**same git repository** (named in `PROJECT.md`):

* **Claude's folder** — the repository itself, on the workstream's base
  branch.
* **Manus's folder** — a git worktree next to it (`<repo>-manus`), on
  the task branch `manus/<workstream>-<task>`.

Separate folders: neither agent's branch switches, installs or
half-written files can break the other's. Same repository: no pushing,
pulling or pull requests; each side reads the other's commits directly.

The user relays one-line messages. Their content is always a pointer
to committed files; everything an agent needs is in a file.

## Layout

```
coord/
  PROTOCOL.md          this file (generic)
  MANUS.md             Manus's standing instructions (generic)
  PROJECT.md           folders, environment, project-wide rules, workstreams
  BOARD.md             every workstream's tasks and status
  coord_check.py       mechanical review checker (generic)
  <workstream>/        one per analysis (e.g. aging, atlas, k562)
    WORKSTREAM.md      goal, base branch, spec docs, frozen rules, protected paths
    tasks/<ID>.md      Claude writes
    reports/<ID>.md    Manus writes (+ <ID>.json)
    reviews/<ID>.md    Claude writes
```

Task IDs are per workstream (`A01`, `A02`, …). The full name of a
task is `<workstream>/<ID>`; its branch is `manus/<workstream>-<ID>`.

## Roles

| | Claude | Manus |
|---|---|---|
| Writes | everything under `coord/` except reports; amendments to frozen rules; library code unless a task grants it | `coord/<ws>/reports/`, the scripts and outputs a task grants |
| Works in | Claude's folder | Manus's folder only |
| Never | runs the large compute | writes in Claude's folder; edits anything under `coord/` except its own reports; edits a workstream's protected paths; merges; pushes |

## Cycle

1. **Issue.** Claude commits `coord/<ws>/tasks/<ID>.md`
   (`status: issued`) on the base branch, creates the task branch
   `manus/<ws>-<ID>` at that commit, updates `BOARD.md`, and gives the
   user the one-line message for Manus.
2. **Execute.** Manus, in its folder, runs `git switch manus/<ws>-<ID>`,
   does the steps, commits per step or spec section, writes and commits
   `coord/<ws>/reports/<ID>.md` and `.json`, runs the checker, and tells
   the user it is done.
3. **Verify.** Claude runs `python coord/coord_check.py check <ws> <ID>
   --run-reviews`, re-derives the gate numbers from committed outputs,
   runs the tests in Manus's folder, and writes
   `coord/<ws>/reviews/<ID>.md` with a verdict:
   * `ACCEPT` — Claude merges the task branch into the base branch
     (`--no-ff`, keeping Manus's commits) and issues the next task;
   * `REVISE` — numbered fixes; Manus commits them on the same branch
     and appends `## Revision n` to its report;
   * `STOP` — a frozen gate failed or a rule cannot execute; the user
     decides.
4. **Answer.** If Manus is stuck, it commits its report with
   `status: blocked` and the question under `## Questions`. Claude
   answers under `## Answers` in the review; the answer is binding as
   if it were in the task.

Reviews are written after the task branch was cut, so Manus reads them
from Claude's folder, read-only: `<Claude's folder>/coord/<ws>/reviews/<ID>.md`.

One task is open per workstream. Different workstreams can run at the
same time only if `PROJECT.md` says Manus has capacity for it; each
uses its own task branch, and Manus switches branches only between
tasks, never mid-task.

## Task file (`coord/<ws>/tasks/<ID>.md`)

Front matter `id, workstream, title, owner (claude|manus), status
(draft|issued|accepted|stopped), base_commit, spec_sections,
depends_on`. Body: Goal, Steps, Gate (numeric conditions), Stop and
ask, and one fenced `json` contract block:

```json
{
  "allowed_paths": ["glob", "..."],
  "required_files": ["committed path that must exist on the branch"],
  "required_metrics": ["key that must appear in the report json"],
  "compute_budget_cpu_hours": 24,
  "protected_symbols": {"path.py": ["def name"]},
  "review_commands": ["python scripts/x.py --ref {head}"]
}
```

`review_commands` run during review with `{head}` replaced by the task
branch; a non-zero exit is a failure.

## Report (Manus writes)

`coord/<ws>/reports/<ID>.json`:

```json
{
  "task": "<ws>/<ID>",
  "status": "done | blocked | partial",
  "started_utc": "...", "finished_utc": "...",
  "compute": {"cpu_hours": 0.0, "gpu_hours": 0.0, "hardware": "..."},
  "metrics": {"<required_metric>": "<value>"},
  "artifacts": [
    {"path": "...", "bytes": 0, "sha256": "...", "committed": false,
     "regenerate": "python scripts/... --seed 0"}
  ],
  "deviations": ["what differed from the task and why"],
  "questions": []
}
```

`coord/<ws>/reports/<ID>.md`, sections in order: Summary (≤ 10 lines),
What ran (commands, in order), Gate numbers (table), Deviations,
Questions, Artifacts.

## Rules that bind both sides

* Frozen rules (pre-registrations, locked parameters) named in
  `WORKSTREAM.md` change only by a dated amendment Claude writes
  before the run it governs.
* Report what happened. A failed gate, a crash or a negative result is
  a report, not a reason to change a parameter. Never fabricate,
  simulate or impute data.
* Large outputs stay on disk in Manus's folder (gitignored); later
  tasks reuse them there. List them in `artifacts` with size, sha256
  and the command that regenerates them. Every script is seeded.
* Manus's Python environment is `.venv` inside its folder. Nobody
  installs into the user's base or conda environments.
* Manus changes only the task's `allowed_paths`; the checker fails the
  review otherwise.
* Manus commits end with `Co-Authored-By: Manus <noreply@manus.im>`.
* Compute beyond a task's budget needs the user's approval first.
