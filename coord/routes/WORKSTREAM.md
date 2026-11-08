---
workstream: routes
base_branch: routes-api
---

# Workstream: routes

## Goal

Check `scjdo2 simulate` against data it did not use: for time-course
datasets, compare walker steps between states with the measured
optimal-transport transitions between consecutive sampling days
(`scjdo2.flow_transitions`, shuffle null), and report where walkers take
steps the data says are rarer than chance. LARRY first, where clones give
a third, independent reference.

## Plan

| task | owner | what |
|---|---|---|
| T00 | claude | merge `flow-motifs` (OT plan, `flow_transitions`) with `routes-sim`; add the comparison and its caveat to `simulate`; tests |
| T01 | manus (cloud) | LARRY run: walker vs OT vs clonal transitions; gate fixed in T01 |

Archetype motifs (enriched archetype→archetype transitions) wait for
atlas/F01's replicate verdict.

## Required tests

`tests/test_routes.py`, `tests/test_flow_motifs.py`, `tests/test_smoke.py`.

## Contract

```json
{
  "protected_paths": ["coord/routes/tasks/**", "coord/routes/WORKSTREAM.md", "scjdo2/routes/**",
                      "scjdo2/flow_motifs.py"]
}
```
