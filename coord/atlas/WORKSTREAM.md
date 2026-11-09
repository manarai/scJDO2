---
workstream: atlas
base_branch: aging-basins
---

# Workstream: atlas

## Goal

Build a common reference space for one biological theme, so that any
new dataset from that theme can be placed in it and its operator motifs
compared with the reference. First theme: hematopoiesis (Paul15 mouse
myeloid progenitors, dynamo scNT-seq human hematopoiesis, Setty human
CD34+ marrow). The space is built from **orientation-free, scale-free
operator descriptors** (sorted real spectrum, Henrici index, well
stiffness), not from gene expression or the energy landscape. These
descriptors don't depend on the latent axes, so separately fitted
datasets land in the same coordinates without aligning expression
first. The question: do motifs recur across independent datasets of
one theme more than a dataset-shuffle null predicts, and do they sit on
the same lineage steps?

Order of tasks: **R01** checks that the descriptors are stable within a
dataset (seeds, training length) and not driven by batch. **R02**
(cross-dataset recurrence and lineage concordance) is issued only after
R01 passes. A failed R01 gate is a result; the space isn't built on
unstable coordinates.

## Documents

* Method spec: the task files (no separate spec). Where they differ
  from earlier aging tasks, the task wins.
* Frozen rules: the Gate section of each task, fixed before its data
  run. Never moved after a number is seen.

## Standing cautions

* Operators pooled across datasets carry batch unless shown otherwise
  (`scjdo2/motifs.py` docstring). R01 is that check.
* A recurring motif is a shared gene-program pattern, not evidence of
  shared TF→target regulation.
* One drift only: the energy drift (`scjdo2/energy_drift.py`). Never install
  or import v1 `scjdo` (`docs/V1_REMOVAL.md` on the `orientation` branch).
* Score-J energy *landscapes* (E surfaces) are not used as the reference
  (unreliable above 2-d); only operator descriptors are.

## Required tests

`tests/test_motifs.py`, `tests/test_behaviour_landscape.py` and `tests/test_landscape_api.py`
must pass before any data run; a failure is a stop.

## Contract

```json
{
  "protected_paths": ["scjdo2/motifs.py", "scjdo2/opspace.py", "scjdo2/energy_drift.py",
                       "scjdo2/drift.py", "scjdo2/pipeline.py",
                       "coord/atlas/WORKSTREAM.md"]
}
```
