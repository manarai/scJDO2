# Changelog

All notable changes to scJDO v2 are recorded here. Each phase from the project
plan corresponds to one or more commits.

## [Unreleased]

### Phase A — Audit and scaffold

- Initialised scJDO2 package layout per `V2_PLAN.md` §2 of the brief.
- Pinned v1 scJDO at tag `scjdo-v56-submission-20260922-r25` via the `[compare]`
  optional extra.
- Added `V2_PLAN.md` with the v1→v2 mapping table, import/reimplement/drop
  decisions, risks, and open questions.
- Added `compare/run_v1.py` skeleton and `data/` loader stubs pointing at the
  five tutorial datasets already present under `../scJDO/data/` and
  `../scJDO/examples/data/`.
- Module files (`embed.py`, `drift.py`, `rates.py`, `field.py`, `fisher.py`,
  `opspace.py`, `archetypes.py`, `grammar.py`, `io.py`, `cli.py`) created as
  typed stubs. No scientific code yet.
