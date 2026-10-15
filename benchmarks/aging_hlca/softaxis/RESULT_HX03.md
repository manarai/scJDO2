EXPLORATORY — statistic designed after the frozen result

## What ran
The exact HX02 real-cell donor operator and injection protocol was repeated for all 13 primary types: no injection, single-axis F ∈ {1.25, 1.5, 2, 3, 5, 8}, and uniform F ∈ {1.5, 2, 3}. The fixed-axis `softaxis_curvature_trend` used 50 fixed 80%-donor replicates and 200 age permutations per replicate. Types with max uniform false-pass power > 0.20 were excluded before three full-donor real-data analyses at 1,000 permutations.

## Reading rule (verbatim)
* a type shows **soft-axis change with age** in an analysis if `p ≤ 0.05`; it shows it **after correction** if `p ≤ 0.05 / n` (`n` = number of types in step 3);
* a type is **robust** if it shows the change after correction in `primary` **and** at p ≤ 0.05 in both `a_within_study` and `e_secondary_space`, with the same sign of slope;
* slope > 0 means `κ` moves toward zero or positive (the soft axis flattens relative to the rest), slope < 0 the opposite.

## Power-grid MDE and specificity gate
| ann_level_3 | mde_single_axis_F | ref_soft_eigenvalue | max_uniform_pass_rate |
|---|---|---|---|
| AT2 | 1.25 | -0.5756 | 0 |
| Basal | > max | 2.723 | 0 |
| Dendritic cells | 8 | 0.4078 | 0 |
| EC arterial | 2 | -1.065 | 0 |
| EC capillary | > max | -0.0192 | 0 |
| EC venous | > max | -0.5291 | 0 |
| Fibroblasts | 2 | -0.2121 | 0 |
| Innate lymphoid cell NK | 5 | -1.091 | 0 |
| Macrophages | 3 | -0.01073 | 0 |
| Monocytes | 8 | -0.05331 | 0.18 |
| Multiciliated lineage | 5 | -0.4968 | 0 |
| Secretory | > max | 0.3544 | 0 |
| T cell lineage | 3 | -0.2424 | 0 |

## Real-data statistics
| analysis | ann_level_3 | n_donors | slope | p | passes | ref_soft_eigenvalue | mde_single_axis_F |
|---|---|---|---|---|---|---|---|
| primary | AT2 | 51 | -0.000203 | 0.4775 | False | -0.5756 | 1.25 |
| a_within_study | AT2 | 51 | -0.000203 | 0.7003 | False | -0.5756 | 1.25 |
| e_secondary_space | AT2 | 51 | -0.0007817 | 0.5624 | False | 1.029 | 1.25 |
| primary | Basal | 43 | -0.00808 | 0.3247 | False | 2.723 | > max |
| a_within_study | Basal | 43 | -0.00808 | 0.07692 | False | 2.723 | > max |
| e_secondary_space | Basal | 43 | -0.001086 | 0.09391 | False | 0.2154 | > max |
| primary | Dendritic cells | 53 | -0.003117 | 0.2208 | False | 0.4078 | 8 |
| a_within_study | Dendritic cells | 53 | -0.003117 | 0.1059 | False | 0.4078 | 8 |
| e_secondary_space | Dendritic cells | 53 | 0.0002084 | 0.8342 | False | 1.165 | 8 |
| primary | EC arterial | 25 | 0.0003429 | 0.7383 | False | -1.065 | 2 |
| a_within_study | EC arterial | 25 | 0.0003429 | 0.6394 | False | -1.065 | 2 |
| e_secondary_space | EC arterial | 25 | -5.511e-06 | 0.951 | False | -0.8109 | 2 |
| primary | EC capillary | 24 | -0.003091 | 0.1878 | False | -0.0192 | > max |
| a_within_study | EC capillary | 24 | -0.003091 | 0.1628 | False | -0.0192 | > max |
| e_secondary_space | EC capillary | 24 | 3.985e-05 | 0.5035 | False | -0.8337 | > max |
| primary | EC venous | 42 | -0.00309 | 0.3596 | False | -0.5291 | > max |
| a_within_study | EC venous | 42 | -0.00309 | 0.1079 | False | -0.5291 | > max |
| e_secondary_space | EC venous | 42 | -7.588e-06 | 0.7303 | False | -0.8133 | > max |
| primary | Fibroblasts | 32 | 0.001439 | 0.4925 | False | -0.2121 | 2 |
| a_within_study | Fibroblasts | 32 | 0.001439 | 0.3237 | False | -0.2121 | 2 |
| e_secondary_space | Fibroblasts | 32 | 0.0001268 | 0.1239 | False | -0.7731 | 2 |
| primary | Innate lymphoid cell NK | 29 | -0.0009914 | 0.5974 | False | -1.091 | 5 |
| a_within_study | Innate lymphoid cell NK | 29 | -0.0009914 | 0.6803 | False | -1.091 | 5 |
| e_secondary_space | Innate lymphoid cell NK | 29 | -4.492e-05 | 0.4975 | False | -0.9078 | 5 |
| primary | Macrophages | 65 | -0.0006275 | 0.7872 | False | -0.01073 | 3 |
| a_within_study | Macrophages | 65 | -0.0006275 | 0.7123 | False | -0.01073 | 3 |
| e_secondary_space | Macrophages | 65 | -0.002372 | 0.02697 | True | 1.244 | 3 |
| primary | Monocytes | 60 | -0.003756 | 0.08791 | False | -0.05331 | 8 |
| a_within_study | Monocytes | 60 | -0.003756 | 0.2068 | False | -0.05331 | 8 |
| e_secondary_space | Monocytes | 60 | -0.0004813 | 0.5055 | False | -0.02149 | 8 |
| primary | Multiciliated lineage | 69 | -0.0008631 | 0.5155 | False | -0.4968 | 5 |
| a_within_study | Multiciliated lineage | 69 | -0.0008631 | 0.5005 | False | -0.4968 | 5 |
| e_secondary_space | Multiciliated lineage | 69 | 7.426e-05 | 0.9241 | False | 1.592 | 5 |
| primary | Secretory | 62 | -0.008275 | 0.3397 | False | 0.3544 | > max |
| a_within_study | Secretory | 62 | -0.008275 | 0.1828 | False | 0.3544 | > max |
| e_secondary_space | Secretory | 62 | 2.23e-06 | 0.997 | False | 1.721 | > max |
| primary | T cell lineage | 58 | -0.0001416 | 0.953 | False | -0.2424 | 3 |
| a_within_study | T cell lineage | 58 | -0.0001416 | 0.9201 | False | -0.2424 | 3 |
| e_secondary_space | T cell lineage | 58 | 1.811e-05 | 0.6513 | False | -0.9142 | 3 |

## Mechanical counts
| types_in_application | alpha | corrected_alpha | n_pass_primary | n_pass_primary_corrected | n_pass_within_study | n_pass_secondary | n_robust | robust_types | n_types_failing_uniform_gate | types_failing_uniform_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 0.05 | 0.003846 | 0 | 0 | 0 | 1 | 0 |  | 0 |  |
