EXPLORATORY

## What ran
H03 primary drift cells were joined to H01 full-metadata `ann_level_4` labels by cell ID. Composition uses all full-metadata cells of each H04 contributing donor. Subtype stiffness uses only the H03 drift subsample at the specified 25-cell floor, 100 deterministic equal-count draws, and 1,000 seed-zero age permutations. Leave-one-study-out analyses reuse H04 donor log-stiffness values without rebuilding operators.

## Reading rule (verbatim)
* a hint is **composition-explained** if the subtype slopes (step 3) are both |slope| < half the H04 slope **and** the composition fraction correlates with age at p ≤ 0.05;
* a hint is **study-driven** if leaving out any single study moves p above 0.05 **or** the study-fixed-effects slope is < half the H04 slope in magnitude;
* otherwise **unexplained**.

## Composition correlations
| ann_level_3 | fraction_name | n_h04_donors | n_valid_fraction | rho_age | p_age | rho_h04_log_stiffness | p_h04_log_stiffness |
|---|---|---|---|---|---|---|---|
| T cell lineage | CD8 / (CD4 + CD8) | 58 | 58 | 0.0292094 | 0.827697 | -0.0239011 | 0.858654 |
| Monocytes | Non-classical / (Classical + Non-classical) | 60 | 60 | -0.190173 | 0.145557 | -0.248347 | 0.0557081 |

## Subtype stiffness
| ann_level_3 | ann_level_4 | n_donors | stiff_slope | stiff_p | passes | n_common | n_draws | n_perm |
|---|---|---|---|---|---|---|---|---|
| T cell lineage | CD4 T cells | 27 | -0.00536248 | 0.043956 | true | 25 | 100 | 1000 |
| T cell lineage | CD8 T cells | 44 | -0.00303675 | 0.0949051 | false | 25 | 100 | 1000 |
| Monocytes | Classical monocytes | 57 | -0.00491181 | 0.003996 | true | 25 | 100 | 1000 |
| Monocytes | Non-classical monocytes | 19 | 0.00254319 | 0.347652 | false | 25 | 100 | 1000 |

## Study dependence
| ann_level_3 | n_donors | n_studies | h04_stiff_slope | h04_stiff_p | fe_age_slope |
|---|---|---|---|---|---|
| T cell lineage | 58 | 12 | -0.00564399 | 0.004995 | -0.00186459 |
| Monocytes | 60 | 10 | -0.00292763 | 0.0769231 | -0.00139293 |

### Leave-one-study-out
| ann_level_3 | left_out_study | n_donors | stiff_slope | stiff_p | ci95_low | ci95_high |
|---|---|---|---|---|---|---|
| T cell lineage | Banovich_Kropski_2020 | 38 | -0.00838157 | 0.000999001 | -0.0117113 | -0.00505188 |
| T cell lineage | Barbry_Leroy_2020 | 51 | -0.00400131 | 0.02997 | -0.00768885 | -0.000313768 |
| T cell lineage | Jain_Misharin_2021_10Xv1 | 56 | -0.00589374 | 0.002997 | -0.00920343 | -0.00258405 |
| T cell lineage | Jain_Misharin_2021_10Xv2 | 55 | -0.0053711 | 0.003996 | -0.0089091 | -0.0018331 |
| T cell lineage | Krasnow_2020 | 55 | -0.00645509 | 0.000999001 | -0.00996727 | -0.00294291 |
| T cell lineage | Lafyatis_Rojas_2019_10Xv1 | 56 | -0.00539668 | 0.002997 | -0.00893733 | -0.00185603 |
| T cell lineage | Lafyatis_Rojas_2019_10Xv2 | 54 | -0.00596431 | 0.000999001 | -0.00966126 | -0.00226736 |
| T cell lineage | Meyer_2019 | 55 | -0.00538289 | 0.001998 | -0.00893479 | -0.00183098 |
| T cell lineage | Misharin_2021 | 57 | -0.00563436 | 0.001998 | -0.00906522 | -0.00220349 |
| T cell lineage | Misharin_Budinger_2018 | 57 | -0.00519127 | 0.003996 | -0.00863907 | -0.00174346 |
| T cell lineage | Nawijn_2021 | 50 | -0.00577483 | 0.00599401 | -0.00960858 | -0.00194108 |
| T cell lineage | Teichmann_Meyer_2019 | 54 | -0.00487586 | 0.00699301 | -0.00831775 | -0.00143398 |
| Monocytes | Banovich_Kropski_2020 | 31 | -0.00546626 | 0.010989 | -0.00930493 | -0.00162758 |
| Monocytes | Jain_Misharin_2021_10Xv2 | 59 | -0.00281635 | 0.0779221 | -0.00608402 | 0.000451315 |
| Monocytes | Krasnow_2020 | 57 | -0.00345293 | 0.041958 | -0.00677149 | -0.000134365 |
| Monocytes | Lafyatis_Rojas_2019_10Xv1 | 58 | -0.00227554 | 0.188811 | -0.0055642 | 0.00101312 |
| Monocytes | Lafyatis_Rojas_2019_10Xv2 | 56 | -0.00252281 | 0.140859 | -0.00600336 | 0.000957741 |
| Monocytes | Meyer_2019 | 57 | -0.00231931 | 0.16983 | -0.00556508 | 0.000926454 |
| Monocytes | Misharin_2021 | 58 | -0.00283854 | 0.0829171 | -0.00616693 | 0.000489852 |
| Monocytes | Misharin_Budinger_2018 | 53 | -0.00261478 | 0.105894 | -0.00590113 | 0.000671564 |
| Monocytes | Nawijn_2021 | 54 | -0.00306535 | 0.0779221 | -0.00649742 | 0.000366728 |
| Monocytes | Teichmann_Meyer_2019 | 57 | -0.00266929 | 0.121878 | -0.00593799 | 0.000599409 |

## Mechanical verdicts
| ann_level_3 | verdict | composition_explained | study_driven | composition_p_age | max_loso_p | h04_slope | fe_slope |
|---|---|---|---|---|---|---|---|
| T cell lineage | study-driven | false | true | 0.827697 | 0.02997 | -0.00564399 | -0.00186459 |
| Monocytes | study-driven | false | true | 0.145557 | 0.188811 | -0.00292763 | -0.00139293 |
