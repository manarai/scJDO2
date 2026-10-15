# EXPLORATORY — aging/X05 per-mouse uncertainty

## Question
Does the granulocytopoietic stiffness difference exceed mouse-to-mouse variation, and how stable are A03 calls when one old mouse is omitted? This exploratory task does not alter the frozen outcome.

## What ran
Stored A02 shared seed-0 and X04 shared seeds 1–2 embeddings/Jacobians only: 30-cell per-mouse centroid operators, exact mouse-label permutations, and primary-seed leave-one-old-mouse-out A03 comparisons. No model was fitted.

## Reading rule (fixed before the run)
* Granulocytopoietic stiffening **exceeds mouse-to-mouse variation** if all three hold: `p_stiffer ≤ 0.05` in seed 0; `p_stiffer ≤ 0.05` in at least one of seeds 1, 2; and the leave-one-out `stiff_ratio` stays ≥ 1.5 for every dropped old mouse. Otherwise **"not distinguishable from mouse-to-mouse variation"**.
* For every type, report the fraction of leave-one-out drops in which `iso_pass` and `ant_pass` equal the A03 call ("call stability").

## Per-type results
| tissue | cell_type | p_stiffer_seed0 | p_stiffer_seed1 | p_stiffer_seed2 | loo_min_stiff_ratio | iso_call_stability | ant_call_stability | mice_excluded_by_30_cell_floor |
|---|---|---|---|---|---|---|---|---|
| lung | fibroblast of lung | 0.04762 | 0.04762 | 0.04762 | 1.009 | 1 | 1 | 18-F-50;3-F-56;3-F-57 |
| lung | natural killer cell | 0.5317 | 0.5317 | 0.5238 | 0.9918 | 1 | 1 | 18-F-50 |
| marrow | granulocyte | 0.303 | 0.3788 | 0.1818 | 1.033 | 1 | 0.7 |  |
| marrow | granulocytopoietic cell | 0.0303 | 0.04545 | 0.01515 | 3.021 | 1 | 1 |  |
| marrow | hematopoietic precursor cell | 1 | 0.1667 | 1 | 1.018 | 0.8 | 1 | 18-F-50;18-F-51;18-M-52;18-M-53;21-F-54;21-F-55;24-M-60;24-M-61 |
| marrow | monocyte | 0.2 | 0.3091 | 0.01818 | 1.099 | 1 | 1 | 18-M-53 |
| marrow | precursor B cell | 0.04762 | 0.3333 | 0.9524 | 1.061 | 0.9 | 1 | 18-M-53;21-F-55;24-M-59;24-M-60;24-M-61 |
| marrow | promonocyte | nan | nan | nan | 1.041 | 1 | 0.8 | 18-F-50;18-F-51;18-M-52;18-M-53;21-F-54;21-F-55;24-M-58;24-M-59;24-M-60;24-M-61 |

## Mechanical verdict
**granulocytopoietic stiffening exceeds mouse-to-mouse variation**

No interpretation beyond the fixed rule is made here.
