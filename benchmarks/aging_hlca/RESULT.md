# HLCA aging basins — H04 frozen tests

## §4 Eligible types and operators
As executed: `n_common=50`, 100 deterministic equal-count donor draws, 30-cell centroid operators, and seed `crc32(type|donor)` were used. The primary eligibility set is: AT2; Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage.

| Analysis | Eligible types | n eligible |
|---|---|---:|
| primary | AT2; Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 13 |
| a_within_study | AT2; Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 13 |
| b_banovich | AT2; Monocytes | 2 |
| c_parenchyma | AT2; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 12 |
| d_never_smokers | AT2; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 6 |
| e_secondary_space | AT2; Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 13 |
| f_two_group | AT2; Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Monocytes; Multiciliated lineage; Secretory; T cell lineage | 13 |

`donor_operators.csv` records every contributing donor/type/operator under each analysis.

## §5 Outcome
**isotropic noise increase; no aging claim beyond stiffness** — no majority stiffness change

The header and qualifier above are the fixed §6 rule applied to `tests_primary.csv`; no sensitivity result changes them.

## §6 Per-type table
| ann_level_3 | n_donors | n_lt45 | n_ge60 | iso_S | iso_q95 | iso_p | iso_pass | median_beta | ant_stat | ant_q95 | ant_p | ant_pass | stiff_slope | stiff_p | stiff_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AT2 | 51 | 20 | 9 | 0.0042282 | 0.00946452 | 0.652348 | false | -0.000285479 | 0.365281 | 0.786442 | 0.543457 | false | -0.000617718 | 0.394605 | none |
| Basal | 43 | 21 | 8 | 0.0155997 | 0.0224333 | 0.24975 | false | -0.00148519 | 0.2735 | 0.330343 | 0.121878 | false | 1.01439e-05 | 0.989011 | none |
| Dendritic cells | 53 | 22 | 6 | 0.0179846 | 0.0235513 | 0.17982 | false | 0.000858995 | 0.380889 | 0.45146 | 0.111888 | false | -0.00104853 | 0.464535 | none |
| EC arterial | 25 | 7 | 5 | 0.0104691 | 0.0288692 | 0.735265 | false | 0.000304465 | 0.046727 | 0.550725 | 0.885115 | false | 0.000888947 | 0.542458 | none |
| EC capillary | 24 | 10 | 5 | 0.0189284 | 0.0313865 | 0.336663 | false | 0.00378904 | 0.447289 | 0.443418 | 0.046953 | true | 0.002934 | 0.234765 | none |
| EC venous | 42 | 18 | 6 | 0.014382 | 0.0281076 | 0.325674 | false | -0.00227869 | 0.644375 | 0.820416 | 0.301698 | false | -0.00234979 | 0.0879121 | none |
| Fibroblasts | 32 | 13 | 5 | 0.010419 | 0.028944 | 0.65035 | false | 0.00301664 | 0.314081 | 0.625418 | 0.458541 | false | 0.00246945 | 0.411588 | none |
| Innate lymphoid cell NK | 29 | 9 | 8 | 0.0121958 | 0.018678 | 0.366633 | false | -0.00230237 | 0.500125 | 0.617162 | 0.180819 | false | -0.00236671 | 0.436563 | none |
| Macrophages | 65 | 31 | 8 | 0.00771859 | 0.0195639 | 0.646354 | false | -0.00136249 | 0.41776 | 0.579201 | 0.202797 | false | 0.000332181 | 0.803197 | none |
| Monocytes | 60 | 21 | 11 | 0.00388468 | 0.0153548 | 0.922078 | false | -0.00248696 | 0.219814 | 0.552799 | 0.476523 | false | -0.00292763 | 0.0769231 | none |
| Multiciliated lineage | 69 | 32 | 11 | 0.0139231 | 0.0171018 | 0.128871 | false | 0.000876343 | 0.00652675 | 0.414338 | 0.973027 | false | 0.000809809 | 0.483516 | none |
| Secretory | 62 | 31 | 11 | 0.00766215 | 0.0223694 | 0.734266 | false | 0.000379474 | 0.152406 | 0.684422 | 0.78022 | false | 0.000198945 | 0.885115 | none |
| T cell lineage | 58 | 26 | 8 | 0.0087435 | 0.020907 | 0.582418 | false | -0.0051018 | 0.29361 | 0.568401 | 0.375624 | false | -0.00564399 | 0.004995 | loss |

## §7 Sensitivity table
### Eligibility after subset restriction
| analysis | n_types | primary_types_not_tested |
|---|---|---|
| a_within_study | 13 | none |
| b_banovich | 2 | Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK; Macrophages; Multiciliated lineage; Secretory; T cell lineage |
| c_parenchyma | 12 | Basal |
| d_never_smokers | 6 | Basal; Dendritic cells; EC arterial; EC capillary; EC venous; Fibroblasts; Innate lymphoid cell NK |
| e_secondary_space | 13 | none |
| f_two_group | 13 | none |

### Tests as executed
| analysis | ann_level_3 | n_donors | n_lt45 | n_ge60 | iso_S | iso_q95 | iso_p | iso_pass | median_beta | ant_stat | ant_q95 | ant_p | ant_pass | stiff_slope | stiff_p | stiff_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_within_study | AT2 | 51 | 20 | 9 | 0.0042282 | 0.00666984 | 0.554446 | false | -0.000285479 | 0.365281 | 0.710132 | 0.446553 | false | -0.000617718 | 0.418581 | none |
| a_within_study | Basal | 43 | 21 | 8 | 0.0155997 | 0.0266424 | 0.402597 | false | -0.00148519 | 0.2735 | 0.335437 | 0.503497 | false | 1.01439e-05 | 0.99001 | none |
| a_within_study | Dendritic cells | 53 | 22 | 6 | 0.0179846 | 0.0200409 | 0.157842 | false | 0.000858995 | 0.380889 | 0.527573 | 0.701299 | false | -0.00104853 | 0.265734 | none |
| a_within_study | EC arterial | 25 | 7 | 5 | 0.0104691 | 0.0236636 | 0.592408 | false | 0.000304465 | 0.046727 | 0.336321 | 0.785215 | false | 0.000888947 | 0.507493 | none |
| a_within_study | EC capillary | 24 | 10 | 5 | 0.0189284 | 0.0257484 | 0.214785 | false | 0.00378904 | 0.447289 | 0.431527 | 0.03996 | true | 0.002934 | 0.311688 | none |
| a_within_study | EC venous | 42 | 18 | 6 | 0.014382 | 0.023214 | 0.723277 | false | -0.00227869 | 0.644375 | 0.71294 | 0.164835 | false | -0.00234979 | 0.421578 | none |
| a_within_study | Fibroblasts | 32 | 13 | 5 | 0.010419 | 0.0200913 | 0.446553 | false | 0.00301664 | 0.314081 | 0.710426 | 0.582418 | false | 0.00246945 | 0.43956 | none |
| a_within_study | Innate lymphoid cell NK | 29 | 9 | 8 | 0.0121958 | 0.0214211 | 0.272727 | false | -0.00230237 | 0.500125 | 0.580165 | 0.125874 | false | -0.00236671 | 0.455544 | none |
| a_within_study | Macrophages | 65 | 31 | 8 | 0.00771859 | 0.0174225 | 0.562438 | false | -0.00136249 | 0.41776 | 0.48976 | 0.135864 | false | 0.000332181 | 0.702298 | none |
| a_within_study | Monocytes | 60 | 21 | 11 | 0.00388468 | 0.0188076 | 0.903097 | false | -0.00248696 | 0.219814 | 0.539823 | 0.515485 | false | -0.00292763 | 0.214785 | none |
| a_within_study | Multiciliated lineage | 69 | 32 | 11 | 0.0139231 | 0.0125148 | 0.030969 | true | 0.000876343 | 0.00652675 | 0.21571 | 0.958042 | false | 0.000809809 | 0.340659 | none |
| a_within_study | Secretory | 62 | 31 | 11 | 0.00766215 | 0.0176303 | 0.587413 | false | 0.000379474 | 0.152406 | 0.287678 | 0.497502 | false | 0.000198945 | 0.764236 | none |
| a_within_study | T cell lineage | 58 | 26 | 8 | 0.0087435 | 0.0153 | 0.32967 | false | -0.0051018 | 0.29361 | 0.444443 | 0.483516 | false | -0.00564399 | 0.194805 | none |
| b_banovich | AT2 | 25 | 11 | 5 | 0.00708128 | 0.00946531 | 0.217782 | false | 0.00108888 | 0.260491 | 0.792289 | 0.726274 | false | -0.000211634 | 0.856144 | none |
| b_banovich | Monocytes | 29 | 12 | 5 | 0.0143076 | 0.0224929 | 0.274725 | false | 0.000166293 | 0.040705 | 0.520528 | 0.896104 | false | 0.000772717 | 0.781219 | none |
| c_parenchyma | AT2 | 49 | 20 | 9 | 0.00423159 | 0.00963052 | 0.691309 | false | -0.000377155 | 0.352879 | 0.771923 | 0.55045 | false | -0.000656848 | 0.335664 | none |
| c_parenchyma | Dendritic cells | 32 | 13 | 5 | 0.0109607 | 0.0286244 | 0.71029 | false | 0.000190451 | 0.176576 | 0.419526 | 0.422577 | false | -0.000916182 | 0.615385 | none |
| c_parenchyma | EC arterial | 22 | 7 | 5 | 0.0109658 | 0.0290279 | 0.716284 | false | 0.000266066 | 0.0525505 | 0.447155 | 0.856144 | false | 0.00107228 | 0.483516 | none |
| c_parenchyma | EC capillary | 23 | 10 | 5 | 0.0189555 | 0.0304815 | 0.35964 | false | 0.00380461 | 0.449201 | 0.461604 | 0.0599401 | false | 0.00296015 | 0.235764 | none |
| c_parenchyma | EC venous | 23 | 9 | 5 | 0.00181138 | 0.0207897 | 0.997003 | false | -9.69968e-05 | 0.248835 | 0.382882 | 0.23976 | false | -0.000848351 | 0.542458 | none |
| c_parenchyma | Fibroblasts | 25 | 12 | 5 | 0.00884662 | 0.0232248 | 0.62038 | false | 0.00124309 | 0.209277 | 0.516777 | 0.525475 | false | 0.00128422 | 0.661339 | none |
| c_parenchyma | Innate lymphoid cell NK | 24 | 9 | 7 | 0.0132472 | 0.0185878 | 0.185814 | false | -0.00217505 | 0.265675 | 0.562227 | 0.544456 | false | -0.0021328 | 0.4995 | none |
| c_parenchyma | Macrophages | 55 | 24 | 8 | 0.011476 | 0.0223541 | 0.462537 | false | -0.000588524 | 0.108478 | 0.526735 | 0.716284 | false | 0.000277771 | 0.84016 | none |
| c_parenchyma | Monocytes | 50 | 20 | 10 | 0.00354166 | 0.0158489 | 0.942058 | false | -0.00270998 | 0.315793 | 0.594062 | 0.351648 | false | -0.00296957 | 0.100899 | none |
| c_parenchyma | Multiciliated lineage | 28 | 11 | 5 | 0.0077129 | 0.0221866 | 0.635365 | false | -0.000197925 | 0.0376725 | 0.552603 | 0.921079 | false | -0.00132094 | 0.266733 | none |
| c_parenchyma | Secretory | 20 | 9 | 5 | 0.0188038 | 0.0305912 | 0.36963 | false | 0.000643096 | 0.0665752 | 0.779991 | 0.917083 | false | 0.000552679 | 0.646354 | none |
| c_parenchyma | T cell lineage | 36 | 15 | 7 | 0.0101144 | 0.0256776 | 0.604396 | false | -0.00357477 | 0.254595 | 0.65523 | 0.516484 | false | -0.0037393 | 0.0799201 | none |
| d_never_smokers | AT2 | 22 | 7 | 6 | 0.0050935 | 0.00724612 | 0.23976 | false | -0.000315784 | 0.252781 | 0.745537 | 0.658342 | false | -0.00117934 | 0.118881 | none |
| d_never_smokers | Macrophages | 27 | 12 | 7 | 0.0228065 | 0.0250755 | 0.0899101 | false | -0.000619921 | 0.075668 | 0.607216 | 0.853147 | false | 0.000434313 | 0.799201 | none |
| d_never_smokers | Monocytes | 28 | 9 | 7 | 0.00316978 | 0.0182193 | 0.957043 | false | -0.00405147 | 0.144877 | 0.646865 | 0.728272 | false | -0.00509068 | 0.011988 | loss |
| d_never_smokers | Multiciliated lineage | 36 | 16 | 8 | 0.00546105 | 0.0210021 | 0.629371 | false | 0.0010527 | 0.122001 | 0.394177 | 0.541459 | false | 0.00185043 | 0.18981 | none |
| d_never_smokers | Secretory | 35 | 16 | 7 | 0.0219574 | 0.0223975 | 0.0599401 | false | 0.00138672 | 0.306673 | 0.667315 | 0.4995 | false | 0.00218169 | 0.157842 | none |
| d_never_smokers | T cell lineage | 30 | 12 | 5 | 0.0184163 | 0.0238998 | 0.193806 | false | -0.00502784 | 0.251917 | 0.618403 | 0.472527 | false | -0.00592056 | 0.00599401 | loss |
| e_secondary_space | AT2 | 51 | 20 | 9 | 0.00586272 | 0.0235176 | 0.834166 | false | 0.000367199 | 0.0473747 | 0.255527 | 0.727273 | false | 0.000315614 | 0.467532 | none |
| e_secondary_space | Basal | 43 | 21 | 8 | 0.0221603 | 0.0263324 | 0.107892 | false | -0.000627093 | 0.262454 | 0.465644 | 0.325674 | false | -0.00064292 | 0.277722 | none |
| e_secondary_space | Dendritic cells | 53 | 22 | 6 | 0.00592738 | 0.0148931 | 0.611389 | false | -9.29713e-07 | 0.148166 | 0.318963 | 0.387612 | false | -5.54297e-05 | 0.863137 | none |
| e_secondary_space | EC arterial | 25 | 7 | 5 | 0.00414555 | 0.00763685 | 0.380619 | false | -0.000170129 | 0.0319312 | 0.378014 | 0.911089 | false | -0.000100609 | 0.338661 | none |
| e_secondary_space | EC capillary | 24 | 10 | 5 | 0.00160019 | 0.00624926 | 0.879121 | false | 8.72802e-05 | 0.323001 | 0.337487 | 0.0679321 | false | -6.07084e-05 | 0.821179 | none |
| e_secondary_space | EC venous | 42 | 18 | 6 | 0.00175002 | 0.00342361 | 0.347652 | false | -6.53026e-05 | 0.212755 | 0.37251 | 0.381618 | false | -0.000101539 | 0.268731 | none |
| e_secondary_space | Fibroblasts | 32 | 13 | 5 | 0.0219216 | 0.0229119 | 0.0659341 | false | 8.5236e-05 | 0.0390266 | 0.282617 | 0.836164 | false | 7.1402e-05 | 0.893107 | none |
| e_secondary_space | Innate lymphoid cell NK | 29 | 9 | 8 | 0.00182702 | 0.0117108 | 0.859141 | false | -0.000253424 | 0.0637606 | 0.291196 | 0.669331 | false | -0.000204471 | 0.622378 | none |
| e_secondary_space | Macrophages | 65 | 31 | 8 | 0.0184517 | 0.0181291 | 0.043956 | true | 3.45538e-05 | 0.333463 | 0.342742 | 0.0599401 | false | 0.000114174 | 0.845155 | none |
| e_secondary_space | Monocytes | 60 | 21 | 11 | 0.00454531 | 0.0266582 | 0.831169 | false | -0.000192893 | 0.0679939 | 0.34636 | 0.712288 | false | -0.000582004 | 0.038961 | loss |
| e_secondary_space | Multiciliated lineage | 69 | 32 | 11 | 0.00145893 | 0.0161075 | 0.866134 | false | -7.9032e-05 | 0.0702719 | 0.483046 | 0.79021 | false | -4.11982e-05 | 0.812188 | none |
| e_secondary_space | Secretory | 62 | 31 | 11 | 0.0156473 | 0.0230494 | 0.25974 | false | 0.00044925 | 0.0968443 | 0.372541 | 0.643357 | false | 0.000614497 | 0.37962 | none |
| e_secondary_space | T cell lineage | 58 | 26 | 8 | 0.000796184 | 0.00246128 | 0.913087 | false | -1.0586e-05 | 0.0222622 | 0.338488 | 0.922078 | false | 3.5977e-05 | 0.729271 | none |
| f_two_group | AT2 | 29 | 20 | 9 | 0.181804 | 0.288187 | 0.382617 | false | -0.018988 | 0.0469753 | 0.770023 | 0.947053 | false | -0.0276479 | 0.403596 | none |
| f_two_group | Basal | 29 | 21 | 8 | 0.375917 | 0.945042 | 0.797203 | false | -0.0304133 | 0.320277 | 0.328632 | 0.0589411 | false | -0.0122475 | 0.825175 | none |
| f_two_group | Dendritic cells | 28 | 22 | 6 | 0.847907 | 1.09584 | 0.171828 | false | 0.0311064 | 0.270982 | 0.464547 | 0.285714 | false | -0.0753943 | 0.324675 | none |
| f_two_group | EC arterial | 12 | 7 | 5 | 0.403479 | 0.725029 | 0.893107 | false | 0.0406942 | 0.0973333 | 0.428786 | 0.775225 | false | 0.0663896 | 0.366633 | none |
| f_two_group | EC capillary | 15 | 10 | 5 | 0.664466 | 1.21071 | 0.408591 | false | 0.263899 | 0.422238 | 0.449999 | 0.0729271 | false | 0.212972 | 0.04995 | gain |
| f_two_group | EC venous | 24 | 18 | 6 | 0.811739 | 1.31307 | 0.240759 | false | -0.0766675 | 0.741399 | 0.849873 | 0.241758 | false | -0.0674356 | 0.313686 | none |
| f_two_group | Fibroblasts | 18 | 13 | 5 | 0.217578 | 1.02269 | 0.934066 | false | 0.0700374 | 0.0209053 | 0.521434 | 0.923077 | false | 0.0705678 | 0.562438 | none |
| f_two_group | Innate lymphoid cell NK | 17 | 9 | 8 | 0.790733 | 0.845699 | 0.0759241 | false | -0.00903951 | 0.31641 | 0.521908 | 0.324675 | false | -0.0184404 | 0.796204 | none |
| f_two_group | Macrophages | 39 | 31 | 8 | 0.568716 | 0.950034 | 0.297702 | false | -0.0616594 | 0.220747 | 0.593014 | 0.552448 | false | -0.0186701 | 0.76024 | none |
| f_two_group | Monocytes | 32 | 21 | 11 | 0.319217 | 0.644768 | 0.405594 | false | -0.0644025 | 0.37869 | 0.593703 | 0.267732 | false | -0.0975639 | 0.180819 | none |
| f_two_group | Multiciliated lineage | 43 | 32 | 11 | 0.382402 | 0.759436 | 0.338661 | false | 0.0626255 | 0.0266156 | 0.402092 | 0.913087 | false | 0.0492627 | 0.296703 | none |
| f_two_group | Secretory | 42 | 31 | 11 | 0.391407 | 0.8112 | 0.588412 | false | 0.0624113 | 0.194008 | 0.668324 | 0.725275 | false | 0.05765 | 0.241758 | none |
| f_two_group | T cell lineage | 34 | 26 | 8 | 0.694331 | 0.929946 | 0.187812 | false | -0.161063 | 0.265498 | 0.637639 | 0.445554 | false | -0.170667 | 0.0559441 | none |
