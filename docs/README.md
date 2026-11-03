# Documentation index

Start with the [README](../README.md), then run the scripts in
[examples/](../examples/).

## For users

| page | contents |
|---|---|
| [math.md](math.md) | the mathematics of both routes: optimal-transport velocity, drift Jacobian, descriptors, score matching, density curvature, the holder/fork/drift rule, orientation, and what each route can identify |
| [flow.md](flow.md) | `sj.flow`: parameters, outputs, descriptor invariance, how to use the result and how not to |
| [../API.md](../API.md) | every top-level function (`flow`, `flow_groups`, `flow_stability`, `fit`, `analyse`, `orient`, `figures`, `run`), every figure, recipes, the module behind each call |
| [opspace.md](opspace.md) | registry of the keys the analysis writes into `AnnData` |
| [SYNTHETIC_MULTIMODAL.md](SYNTHETIC_MULTIMODAL.md) | simulated datasets with lineage, labelling and splicing readouts and exact truth; which keys are inputs and which are truth |
| [ORIENTATION_SPEC.md](ORIENTATION_SPEC.md) | the full rules of the orientation layer (`sj.orient`) |

## For developers (design records)

| page | contents |
|---|---|
| [opspace_figure_spec.md](opspace_figure_spec.md) | original specification of the operator-space analysis and figure API |
| [scJDO2_figure_layer_build_instructions.md](scJDO2_figure_layer_build_instructions.md) | build order and function contracts for the figure layer |
| [V1_REMOVAL.md](V1_REMOVAL.md) | what was removed from scJDO v1 and how to recover it |
| [../V2_PLAN.md](../V2_PLAN.md), [../SUMMARY.md](../SUMMARY.md) | development plan and history of the v2 rewrite |
| [../CHANGELOG.md](../CHANGELOG.md) | changes by version |

These records describe decisions as they were made. Where they disagree with
the user pages above, the user pages describe the current package.

## Manuscript

| page | contents |
|---|---|
| [../manuscript/REPRODUCE_v57.md](../manuscript/REPRODUCE_v57.md) | how to regenerate every figure and number of the preprint |
| [../notebooks/README.md](../notebooks/README.md) | what each figure notebook draws and from which result files |
