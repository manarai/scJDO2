# Board — scJDO2

Live status: `python coord/coord_check.py board`.

| task | owner | status | title | verdict |
|---|---|---|---|---|
| aging/A00 | claude | accepted | Prereg amendments §2a, §4b + tests | ACCEPT |
| aging/A01 | manus | accepted | Environment, TMS fetch, preprocessing | ACCEPT |
| aging/A02 | manus | accepted | scArches shared space, drift, operators | ACCEPT |
| aging/A03 | manus | accepted | Per-type compare, artifacts, modes, outcome | ACCEPT |
| aging/A04 | manus | accepted | Figures, QC set, notebook, RESULT.md | ACCEPT |
| aging/X04 | manus | accepted | EXPLORATORY: mapping-shrinkage check | ACCEPT |

## Log

* 2026-10-14 — protocol set up; aging/A00 issued to Claude; A01–A04 drafted.
* 2026-10-14 — aging/A00 accepted (20f1371); A01 issued.
* 2026-10-14 — switched to a local shared repo; Manus works in the worktree `~/Documents/scJDO2-manus`.
* 2026-10-14 — generic multi-project layout: `coord/PROJECT.md`, workstream `coord/aging/`; branch `manus/aging-A01`.
* 2026-10-14 — aging/A01 blocked (4-week labels); PREREG §2b; REVISE issued.
* 2026-10-14 — aging/A01 rev 1 blocked at lung HVG; PREREG §2c (old = 18–24 m, HVG span 1.0); REVISE 2 issued.
* 2026-10-14 — aging/A01 accepted (d4eb2fe), merged; A02 issued.
* 2026-10-14 — aging/A02 accepted (6699362), merged; A03 issued with shared-space-only tests clarified.
* 2026-10-14 — aging/A03 accepted (79aeb37): frozen outcome "isotropic noise increase; no aging claim beyond stiffness"; A04 issued.
* 2026-10-14 — aging/A04 REVISE: saddle marker placement, aging_vectors axis ranges, cosmetics.
* 2026-10-14 — aging/A04 accepted (40cf3a0), merged. Mouse arm A00–A04 complete.
* 2026-10-14 — user approved X04 (exploratory); issued. Reading rule fixed before the run.
* 2026-10-14 — aging/X04 blocked: contract fence missing (Claude) and device="cpu" vs A02's "auto"; REVISE.
* 2026-10-14 — aging/X04 accepted (dc648cf): no evidence mapping hides change; A03 isotropy passes not seed-robust; granulocytopoietic stiffening robust.

## hlca

| task | owner | status | title | verdict |
|---|---|---|---|---|
| hlca/H01 | manus | issued | HLCA data, reference model, age inventory | — |
| hlca/H02 | claude | draft | Human pre-registration (from H01 inventory) | — |
| hlca/H03 | manus | draft | scArches shared space + drift on HLCA | — |
| hlca/H04 | manus | draft | Per-type tests and outcome, human lung | — |
| hlca/H05 | manus | draft | Figures, mouse–human comparison | — |

* 2026-10-14 — hlca workstream created (HLCA core + scArches); H01 issued. Manus runs aging/A01 first, then hlca/H01.
