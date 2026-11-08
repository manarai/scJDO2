# scJDO2 route simulation report

## Result

- Direction source: **clone** (lineage barcodes); drift fit median cosine 0.84 on 25,624 cells.
- 300 walkers from **Undifferentiated** end in: Undifferentiated 12, Neutrophil 134, Monocyte 109, Baso 41, Mast 3, Meg 1.
- Route Undifferentiated → Neutrophil: 134 walkers; top archetypes [1, 7, 0].
- Route Undifferentiated → Monocyte: 109 walkers; top archetypes [1, 2, 7].
- Route Undifferentiated → Baso: 41 walkers; top archetypes [6, 0, 2].

## Held-out clone fate test

| method | macro AUROC | 95% CI | top-fate acc. |
|---|---|---|---|
| walk_drift | 0.897 | 0.871–0.922 | 0.635 |
| supervised_ceiling | 0.890 | 0.864–0.912 | 0.628 |
| walk_undirected | 0.867 | 0.836–0.893 | 0.452 |
| proximity | 0.828 | 0.803–0.854 | 0.090 |

## Caveats (generated from the numbers)

- About 95% of genes carry no operator weight in a typical archetype; absent genes (including known regulators) say nothing about their role.
- Most influential genes are hubs (52 hubs against 22 drivers and responders): driver/responder labels are thin here.
- Operators are mostly symmetric (median directional/symmetric ratio 0.38), although a clone drift can be directional; direction between genes is weak.
- Reference network: operator edges beat random targets in 8 of 8 archetypes but plain co-expression in only 7.

## Archetypes

| id | behaviour | cells | drivers | hubs | responders | directional/symmetric | unmeasured genes |
|---|---|---|---|---|---|---|---|
| 0 | holder | 488 | 4 | 5 | 1 | 0.26 | 95% |
| 1 | fork | 391 | 0 | 9 | 0 | 0.51 | 96% |
| 2 | fork | 333 | 5 | 4 | 1 | 0.29 | 95% |
| 3 | fork | 155 | 1 | 8 | 0 | 0.42 | 96% |
| 4 | fork | 216 | 0 | 4 | 4 | 0.28 | 96% |
| 5 | fork | 168 | 1 | 5 | 2 | 0.64 | 96% |
| 6 | fork | 384 | 1 | 7 | 2 | 0.56 | 95% |
| 7 | fork | 466 | 0 | 10 | 0 | 0.34 | 95% |

## How to read this

- Walkers hop between real cells along the drift; where they end is the model's fate tendency from the start state, not a measured fate unless the fate test above supports it.
- Archetypes are groups of cells sharing a local linear operator; their genes are coupled programs.
- **Do not** present gene coupling edges as a gene regulatory network or TF → target regulation.
- **Do not** read playback hops as real time.
- States never reached may be disconnected in the data, not impossible fates.

## Files

- `route_page.html` — interactive page (open in a browser).
- `diagnostics.json` — every number above.
- `routes.json` — route panels (archetype curves, genes, coupling edges).
- `routes.csv`, `archetypes.csv`, `fate_test.csv` — tables.
