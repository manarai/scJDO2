# EXPLORATORY — aging/X05 per-mouse uncertainty

## Question
Does granulocytopoietic stiffness exceed mouse-to-mouse variation, and how stable are A03 calls when one old mouse is omitted? This exploratory task does not alter the frozen outcome.

## What ran
Stored A02 seed-0 and X04 shared seeds 1–2 embeddings/Jacobians only. Revision 1 uses 200 deterministic equal-count subsamples per donor/type, where `n_common` is the smallest count among mice meeting the 30-cell floor; full-count values remain provenance. No model was fitted.

## Reading rule (fixed before the run)
* Granulocytopoietic stiffening **exceeds mouse-to-mouse variation** if all three hold: `p_stiffer ≤ 0.05` in seed 0; `p_stiffer ≤ 0.05` in at least one of seeds 1, 2; and the leave-one-out `stiff_ratio` stays ≥ 1.5 for every dropped old mouse. Otherwise **"not distinguishable from mouse-to-mouse variation"**.
* For every type, report the fraction of leave-one-out drops in which `iso_pass` and `ant_pass` equal the A03 call ("call stability").

## Per-type results
| tissue | cell_type | n_common | p_stiffer_seed0 | p_stiffer_seed1 | p_stiffer_seed2 | p_stiffer_fullcount_seed0 | loo_min_stiff_ratio | iso_call_stability | ant_call_stability |
|---|---|---|---|---|---|---|---|---|---|
| lung | fibroblast of lung | 36 | 0.04762 | 0.04762 | 0.04762 | 0.04762 | 1.009 | 1 | 1 |
| lung | natural killer cell | 35 | 0.2302 | 0.3413 | 0.3175 | 0.5317 | 0.9918 | 1 | 1 |
| marrow | granulocyte | 70 | 0.1818 | 0.4091 | 0.1818 | 0.303 | 1.033 | 1 | 0.7 |
| marrow | granulocytopoietic cell | 72 | 0.04545 | 0.1667 | 0.2576 | 0.0303 | 3.021 | 1 | 1 |
| marrow | hematopoietic precursor cell | 30 | 0.5 | 0.1667 | 1 | 1 | 1.018 | 0.8 | 1 |
| marrow | monocyte | 54 | 0.1091 | 0.2727 | 0.03636 | 0.2 | 1.099 | 1 | 1 |
| marrow | precursor B cell | 37 | 0.04762 | 0.09524 | 0.4762 | 0.04762 | 1.061 | 0.9 | 1 |
| marrow | promonocyte | 111 | nan | nan | nan | nan | 1.041 | 1 | 0.8 |

## Mechanical verdict (equal-count)
**not distinguishable from mouse-to-mouse variation**

## Full-count provenance verdict
**granulocytopoietic stiffening exceeds mouse-to-mouse variation**

The equal-count correction was added after review because the original design compared unequal cell counts. No interpretation beyond the fixed rule is made here.
