EXPLORATORY — statistic designed after the frozen result

## What ran
The verified A02 seed-0 and X04 seeds-1/2 shared spaces were used without model fitting. Per type and seed, mice with at least 30 cells contributed deterministic 100-draw equal-count donor operators. The pooled young 30-cell centroid operator defined the fixed soft axis. Seed 0 also ran the stated old-cell single-axis factors and uniform ×2 specificity injection.

## Reading rule (verbatim)
a type shows **soft-axis change with age** if `p ≤ 0.05` in seed 0 **and** in at least one of seeds 1, 2, with the same sign of slope.

## Real data
| seed | tissue | cell_type | n_young | n_old | n_common | n_labellings | min_attainable_p | slope | p | passes | ref_soft_eigenvalue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | lung | fibroblast of lung | 2 | 5 | 36 | 21 | 0.047619 | -0.000497452 | 0.183816 | false | -6.88507 |
| 1 | lung | fibroblast of lung | 2 | 5 | 36 | 21 | 0.047619 | -0.00368266 | 0.045954 | true | -4.61522 |
| 2 | lung | fibroblast of lung | 2 | 5 | 36 | 21 | 0.047619 | -0.00502627 | 0.045954 | true | -3.35983 |
| 0 | lung | natural killer cell | 4 | 5 | 35 | 126 | 0.00793651 | -0.00178786 | 0.012987 | true | -12.9827 |
| 1 | lung | natural killer cell | 4 | 5 | 35 | 126 | 0.00793651 | 9.63403e-05 | 0.69031 | false | -4.70904 |
| 2 | lung | natural killer cell | 4 | 5 | 35 | 126 | 0.00793651 | 0.000227597 | 0.400599 | false | -10.3927 |
| 0 | marrow | granulocyte | 2 | 10 | 70 | 66 | 0.0151515 | -0.0241431 | 0.137862 | false | -1.49228 |
| 1 | marrow | granulocyte | 2 | 10 | 70 | 66 | 0.0151515 | -0.0261989 | 0.321678 | false | -1.76801 |
| 2 | marrow | granulocyte | 2 | 10 | 70 | 66 | 0.0151515 | -0.0259119 | 0.477522 | false | -0.724963 |

## Power and specificity (seed 0)
| tissue | cell_type | condition | F | slope | p | passes |
|---|---|---|---|---|---|---|
| lung | fibroblast of lung | single_axis | 1.25 | 0.00374297 | 0.045954 | true |
| lung | fibroblast of lung | single_axis | 1.5 | 0.00656992 | 0.045954 | true |
| lung | fibroblast of lung | single_axis | 2 | 0.0101036 | 0.045954 | true |
| lung | fibroblast of lung | single_axis | 3 | 0.0136373 | 0.045954 | true |
| lung | fibroblast of lung | single_axis | 5 | 0.0164642 | 0.045954 | true |
| lung | fibroblast of lung | uniform | 2 | -0.000497452 | 0.183816 | false |
| lung | natural killer cell | single_axis | 1.25 | 0.0053165 | 0.012987 | true |
| lung | natural killer cell | single_axis | 1.5 | 0.0100527 | 0.012987 | true |
| lung | natural killer cell | single_axis | 2 | 0.015973 | 0.012987 | true |
| lung | natural killer cell | single_axis | 3 | 0.0218933 | 0.012987 | true |
| lung | natural killer cell | single_axis | 5 | 0.0266296 | 0.012987 | true |
| lung | natural killer cell | uniform | 2 | -0.00178786 | 0.012987 | true |
| marrow | granulocyte | single_axis | 1.25 | 0.0360917 | 0.028971 | true |
| marrow | granulocyte | single_axis | 1.5 | 0.0762505 | 0.010989 | true |
| marrow | granulocyte | single_axis | 2 | 0.126451 | 0.010989 | true |
| marrow | granulocyte | single_axis | 3 | 0.176653 | 0.010989 | true |
| marrow | granulocyte | single_axis | 5 | 0.216816 | 0.010989 | true |
| marrow | granulocyte | uniform | 2 | -0.0241431 | 0.137862 | false |

## Mechanical summary
| label | n_young | n_old | n_common | n_labellings | min_attainable_p | mde_single_axis_F | uniform_x2_passes | supporting_seeds | x06_verdict |
|---|---|---|---|---|---|---|---|---|---|
| lung fibroblast | 2 | 5 | 36 | 21 | 0.047619 | 1.25 | false | 1;2 | no soft-axis change with age |
| lung NK | 4 | 5 | 35 | 126 | 0.00793651 | 1.25 | true | none | no soft-axis change with age |
| marrow granulocyte | 2 | 10 | 70 | 66 | 0.0151515 | 1.25 | false | none | no soft-axis change with age |
