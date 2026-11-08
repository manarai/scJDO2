---
workstream: perturb
base_branch: routes-api
---

# Workstream: perturb

## Goal

Does scJDO2's gene coupling carry causal information? The coupling of an
archetype is M(t*) = Pᵀ(e^{Bt*} − I)P: the predicted total change of every
gene after a small change in one gene. A CRISPRi knockdown measured days
later is that kind of total effect. Fit the operator on **unperturbed
control cells only**, then ask whether column j of M predicts the measured
response to knocking down gene j better than co-expression and than linear
baselines trained on other perturbations. Data: Replogle et al. 2022 K562
essential-gene screen. If and only if the pre-registered test (P02)
passes, a later task tests perturbed-drift routes to new states on the TF
Atlas (Joung et al. 2023).

## Disclosed priors (planner knowledge, not results of this workstream)

* orient/S18–S18d (latent score J, not the gene-level operator): score-J
  soft axes did not predict K562 knockdown shifts (P1 fail; median capture
  0.045 vs 0.589 for global covariance); basins did not predict occupancy
  change; RPE1 had 1 qualifying target. Conclusion then: "Perturb-seq is
  anchor-op's question". This workstream is that question.
* Gene-level operators on hematopoiesis, iPSC, LARRY and CD34+ marrow did
  not recover TRRUST targets beyond co-expression (orient/S31 and route
  runs). The gene operator is a lift of a ~10-d latent operator (rank ≤
  n_pcs), so its columns may mostly reflect shared loadings.
* K562 is proliferating and not at equilibrium; the only direction
  available on controls is the snapshot score drift, whose Jacobian is
  symmetric by construction.

## Plan

| task | owner | what | gate |
|---|---|---|---|
| P01 | manus (cloud) | data, qualifying knockdowns, one control-only operator fit, coverage; **no prediction scores** | feasibility (F1–F3) |
| P02 | manus (cloud) | pre-registered prediction test, 3 fit seeds, held-out folds, 4 baselines | G1–G3 (thresholds confirmed by the user before issue) |
| P03 | — | TF Atlas routes to new states | only if P02 passes; written after P02 |

## Documents

* Method: `scjdo2/routes/run.py` (fit settings, steps 1–3),
  `scjdo2/routes/dynamics.py::operator_tables`, `scjdo2/influence.py`.
* Frozen rules: the Gate and rule sections of each task file once issued.

## Required tests

`tests/test_routes.py`, `tests/test_influence.py`, `tests/test_smoke.py`
must pass in the sandbox venv before any data run; a failure is a stop.

## Contract

```json
{
  "protected_paths": ["coord/perturb/tasks/**", "coord/perturb/WORKSTREAM.md",
                      "scjdo2/routes/**", "scjdo2/influence.py", "scjdo2/drivers.py",
                      "scjdo2/pipeline.py", "scjdo2/energy_drift.py"]
}
```
