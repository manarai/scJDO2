---
workstream: reprog
base_branch: routes-api
---

# Workstream: reprog

## Goal

On lineage-traced reprogramming, test two things scJDO2's route
simulation could add: (1) **route search** — the most probable chain of
measured cells between states, including routes against the drift — and
whether routes predict held-out clone fate better than proximity; (2)
**drivers by timing** — genes that change before commitment along a route
and predict clone fate — and whether they rank known factors above simple
baselines (genes that differ between start and end states; genes that
track pseudotime). Data: CellTag MEF→iEP (Biddy et al. 2018) and the
Schiebinger et al. 2019 iPSC time course.

## Plan

| task | owner | what |
|---|---|---|
| C01 | manus (cloud) | data inventory: download, verify, build h5ad, clones × days, annotations, factor detectability; no fits |
| C02 | claude | route-search and timing-driver code in `scjdo2.routes`, with tests; pre-registration written from C01's inventory |
| C03 | manus (cloud) | pre-registered run |

## Known design issue (C01 must report on it)

The classic factors are transgenes: Hnf4a and Foxa1 are transduced in
MEF→iEP, and OKSM is dox-induced in Schiebinger. Their transcripts may be
high in every cell from day 0, which would make them trivially "early".
C02's known-factor list depends on whether transgene and endogenous reads
can be told apart in the counts.

## Documents

* Method: `scjdo2/routes/` (walk, direction, fate); spec for C02 to follow.
* Frozen rules: each task's Gate section once issued.

## Required tests

`tests/test_routes.py`, `tests/test_smoke.py`.

## Contract

```json
{
  "protected_paths": ["coord/reprog/tasks/**", "coord/reprog/WORKSTREAM.md", "scjdo2/routes/**"]
}
```
