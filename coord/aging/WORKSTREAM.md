---
workstream: aging
base_branch: aging-basins
---

# Workstream: aging

## Goal

Test whether aging loosens cell-type basins direction-specifically
(PREREG §4b isotropy) and along the young soft axis (PREREG §5
anticipation), on Tabula Muris Senis lung and bone marrow, with old
cells mapped into a young scArches reference.

## Documents

* Method spec: `MANUS_INSTRUCTIONS_aging_basins.md` (tasks name its
  sections). Where a task and the spec differ, the task wins.
* Frozen rules: `PREREG_aging_basins.md` (§2a and §4b amendments in
  force since commit 20f1371).

## Required tests

`tests/test_compare_aging.py` must pass before any data run; a failure
is a stop.

## Contract

```json
{
  "protected_paths": ["PREREG_aging_basins.md",
                       "MANUS_INSTRUCTIONS_aging_basins.md"]
}
```
