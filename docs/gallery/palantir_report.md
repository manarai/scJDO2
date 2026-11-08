# scJDO2 route simulation report

## Result

- Direction source: **pseudotime** (pseudotime); drift fit median cosine 0.62 on 5,780 cells.
- 300 walkers from **HSC** end in: Mono 81, Ery 33, CLP 2, DCs 184.
- Route HSC → Precursors → DCs: 184 walkers; top archetypes [0, 7, 4].
- Route HSC → Precursors → Mono: 81 walkers; top archetypes [0, 7, 4].
- Route HSC → Ery: 33 walkers; top archetypes [6, 2, 5].
- Drift vs pseudotime: median cosine 0.59 (1 = same direction, 0 = unrelated).

## Caveats (generated from the numbers)

- About 96% of genes carry no operator weight in a typical archetype; absent genes (including known regulators) say nothing about their role.
- Most influential genes are hubs (59 hubs against 15 drivers and responders): driver/responder labels are thin here.
- Operators are mostly symmetric (median directional/symmetric ratio 0.19), as expected from a pseudotime drift, which is close to a gradient field; direction between genes is weak.
- Reference network: operator edges beat random targets in 8 of 8 archetypes but plain co-expression in only 7.
- No lineage tracing or time course: where walkers end is not checked against true fates. With pseudotime as the only direction, walkers drift toward the states with the highest pseudotime, and the end shares can change with the cell sample and number of hops.

## Archetypes

| id | behaviour | cells | drivers | hubs | responders | directional/symmetric | unmeasured genes |
|---|---|---|---|---|---|---|---|
| 0 | fork | 437 | 2 | 6 | 1 | 0.10 | 96% |
| 1 | fork | 178 | 0 | 10 | 1 | 0.18 | 95% |
| 2 | fork | 76 | 1 | 8 | 1 | 0.19 | 95% |
| 3 | fork | 228 | 3 | 6 | 0 | 0.10 | 96% |
| 4 | growing_oscillator | 702 | 1 | 6 | 2 | 0.34 | 96% |
| 5 | fork | 35 | 1 | 6 | 0 | 0.04 | 97% |
| 6 | fork | 64 | 0 | 9 | 1 | 0.24 | 95% |
| 7 | fork | 498 | 1 | 8 | 0 | 0.44 | 96% |

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
