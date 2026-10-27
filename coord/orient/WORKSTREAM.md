---
workstream: orient
base_branch: orientation
---

# Workstream: orient — orientation layer ground truth and end-to-end tests

## Goal

Test the orientation layer (`scjdo2/orient/`, `docs/ORIENTATION_SPEC.md`)
end to end: every source from realistic readouts (lineage barcodes,
metabolic labelling, spliced/unspliced counts, sampling time, pseudotime),
against a synthetic system with known truth.

## Documents

* `docs/ORIENTATION_SPEC.md` — the layer's design and gates.
* `docs/V1_REMOVAL.md` — v1 is gone; do not reintroduce it.

## Required tests

`tests/test_orient.py` and `tests/test_smoke.py` must pass (the 3 strict
xfails in `test_orient.py` stay xfail) before and after every task.

## Contract

```json
{
  "protected_paths": ["docs/ORIENTATION_SPEC.md", "docs/V1_REMOVAL.md",
                       "scjdo2/orient/**", "scjdo2/energy_drift.py",
                       "scjdo2/labelled_rates.py", "scjdo2/measured_drift.py",
                       "tests/test_orient.py"]
}
```
