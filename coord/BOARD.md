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
| aging/X05 | manus | accepted | EXPLORATORY: per-mouse uncertainty | ACCEPT |
| aging/X06 | manus | issued | EXPLORATORY: fixed-axis statistic on mouse basins | — |

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
* 2026-10-14 — user approved X05 then hlca/H01; X05 issued.
* 2026-10-14 — aging/X05 REVISE: per-mouse operators must use equal cell counts (task design flaw, Claude).
* 2026-10-14 — aging/X05 accepted (f1b9538): equal-count verdict "not distinguishable from mouse-to-mouse variation". Mouse arm closed.

## hlca

| task | owner | status | title | verdict |
|---|---|---|---|---|
| hlca/H01 | manus | accepted | HLCA data, reference model, age inventory | ACCEPT |
| hlca/H02 | claude | accepted | Human pre-registration (from H01 inventory) | ACCEPT |
| hlca/H03 | manus | accepted | scArches shared space + drift on HLCA | ACCEPT |
| hlca/H04 | manus | accepted | Per-type tests and outcome, human lung | ACCEPT |
| hlca/H05 | manus | accepted | Figures, mouse–human comparison | ACCEPT |
| hlca/HX02 | manus | accepted | EXPLORATORY: power analysis (MDE per type) | ACCEPT |
| hlca/HX01 | manus | accepted | EXPLORATORY: T cell / monocyte follow-up | ACCEPT |
| hlca/HX03 | manus | accepted | EXPLORATORY: fixed-axis curvature statistic (grid + real data) | ACCEPT |

* 2026-10-14 — hlca workstream created (HLCA core + scArches); H01 drafted.
* 2026-10-14 — H01 re-issued on current aging-basins after the mouse arm closed.
* 2026-10-15 — hlca/H01 accepted (c036c4c): 107 donors, <45: 48, ≥60: 17; study/assay/site lean old. H02 next (Claude).
* 2026-10-15 — PREREG_aging_hlca.md approved by the user and frozen (9c29466); H03 issued.
* 2026-10-15 — hlca/H03 blocked by the platform's 2-h job limit; REVISE: detached execution, full-data HVG equivalence check.
* 2026-10-15 — hlca/H03 rev 1: HVG equivalence 2000/2000 (accepted); fit killed by its own logging hook; REVISE 2: non-raising hook, dry run first.
* 2026-10-15 — hlca/H03 accepted (0ba8886), merged; strata + outcome checker added (Claude); H04 issued.
* 2026-10-15 — hlca/H04 accepted (66ea582): "isotropic noise increase; no aging claim beyond stiffness — no majority stiffness change"; reproduced to 1e-16.
* 2026-10-15 — hlca/H05 issued (figures, mouse–human comparison, notebook).
* 2026-10-15 — hlca/H05 accepted (db09352), merged. Human arm H01–H05 complete.
* 2026-10-15 — user approved HX02 (exploratory power analysis); issued after a one-type design probe.
* 2026-10-15 — hlca/HX02 accepted (aec59d9): uniform MDE x1.5, drift 2 sigma, single-axis >x8 in 12/13 (statistic insensitive near-zero soft eigenvalues).
* 2026-10-15 — Claude designed softaxis_curvature_trend (validated: 6 types, x1.25–x5 MDE, 0 uniform false passes); user approved; HX03 issued.
* 2026-10-15 — hlca/HX03 accepted (13c43a9): no soft-axis change (0/13 primary), median MDE x3; reproduced to 1e-16.
* 2026-10-15 — user approved aging/X06, hlca/HX01, write-up, GitHub backup; X06 and HX01 issued (run X06 first).
* 2026-10-15 — hlca/HX01 accepted (74f4209): T cell and monocyte hints study-driven. aging/X06 still queued.
