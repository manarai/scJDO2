---
workstream: hlca
base_branch: aging-basins
---

# Workstream: hlca — human lung aging on the Human Lung Cell Atlas

## Goal

Run the aging-basins tests (isotropy §4b, anticipation §5, outcome
rule §6 of `PREREG_aging_basins.md`) on human lung, using the Human
Lung Cell Atlas core (Sikkema, …, Theis et al., *Nat Med* 2023; 107
healthy donors, ~10–76 years, scANVI-integrated) and scArches
reference mapping. Then compare each cell type's aging change with
the mouse result from the `aging` workstream (shared versus
species-specific aging).

Human data differs from TMS in ways that decide the design: age is
continuous, and age may be confounded with the source study, smoking,
sampling site (airway versus parenchyma) and assay. So the order is:
inventory (H01, metadata only), then a human pre-registration written
from that inventory (H02, Claude), then fits (H03), tests (H04),
figures and the mouse comparison (H05).

## Documents

* Method spec for the pipeline: `MANUS_INSTRUCTIONS_aging_basins.md`
  §§3–7 (the mouse arm); human-specific changes are in the hlca tasks
  and, from H02, in `PREREG_aging_hlca.md`. Where they differ, the
  task wins.
* Frozen rules: `PREREG_aging_basins.md` §§4b, 5, 6 (statistics and
  outcome rule, reused unchanged); `PREREG_aging_hlca.md` (written in
  H02, before any fit on human data).
* References: scArches HLCA mapping tutorial
  <https://docs.scarches.org/en/stable/hlca_map_classify.html>;
  HLCA reference model on Zenodo (record 7599104, current; 6337966,
  v1).

## Required tests

`tests/test_compare_aging.py` must pass before any data run; a failure
is a stop.

## Contract

```json
{
  "protected_paths": ["PREREG_aging_basins.md", "PREREG_aging_hlca.md",
                       "MANUS_INSTRUCTIONS_aging_basins.md",
                       "scjdo2/compare.py"]
}
```
