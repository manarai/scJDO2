# S03 two-output end-to-end results

## What ran
- Joined each validated S02 fitted unit to its matched S03 user file using only dynamical `velocity_<embedding>`, `palantir_pseudotime`, and `palantir_start_cell`; S02 models and `J` were not refit.
- Ran the fixed two-output layer: flow uses labelling > scVelo velocity, developmental uses lineage > Palantir pseudotime. Raw labelling is primary; smoothed labelling is the sensitivity.
- The supplied Palantir root is the generator-selected earliest-day progenitor-nearest cell (the task's declared best-case exception); it is recorded, not used as a truth score.
- Exact simulator truth was read only after observed-data orientation calls, for scoring instantaneous flow and expected-count clone chord direction.

## Gates (verbatim)
- Flow: velocity (dynamical) sign agreement ≥ 0.75 and oriented fraction ≥ 0.3; combined coverage is non-decreasing and labelling decisions are exact; conflicts side with labelling ≥ 0.60.
- Developmental: lineage agreement ≥ 0.95; lineage+pseudotime oriented fraction ≥ 0.90; conflicts side with lineage ≥ 0.80.
- Flow/dev independence and J hashes hold exactly in every run. Stochastic scVelo is report-only.

## Gate results
| gate | minimum | per_seed | minimum_observed | pass |
|---|---|---|---|---|
| Flow velocity agreement (dynamical; moving axes) | 0.75 | 0.1338, 0.1770, 0.2389 | 0.13384274094158244 | False |
| Flow velocity oriented fraction | 0.3 | 0.7656, 0.7502, 0.7119 | 0.7119176907687911 | True |
| Flow coverage non-decreasing and labelling decision unchanged | 1.0 | True, True, True | 1.0 | True |
| Flow conflict truth sides with labelling (moving axes) | 0.6 | 0.9786, 0.9723, 0.9691 | 0.9691313711414213 | True |
| Developmental lineage agreement (moving axes) | 0.95 | 0.9953, 0.9879, 0.9883 | 0.9879245283018868 | True |
| Developmental lineage+pseudotime oriented fraction | 0.9 | 0.8401, 0.8456, 0.7979 | 0.7979422692197771 | False |
| Developmental conflict truth sides with lineage (moving axes) | 0.8 | 0.9057, 0.6552, 0.7164 | 0.6551724137931034 | False |
| Flow/dev independence exact in every run | 1.0 | True | 1.0 | True |
| J hash unchanged in every call | 1.0 | True | 1.0 | True |

## Primary PCA/raw summary (mean and minimum over seeds)
| preset | group | combination | n_seeds | sign_agreement_moving_mean | sign_agreement_moving_min | oriented_fraction_mean | oriented_fraction_min |
|---|---|---|---|---|---|---|---|
| batch | dev | all_four | 3 | 0.8694 | 0.7847 | 0.7639 | 0.6775 |
| batch | dev | lineage | 3 | 0.994 | 0.9897 | 0.3949 | 0.3325 |
| batch | dev | lineage+pseudotime | 3 | 0.8694 | 0.7847 | 0.7639 | 0.6775 |
| batch | dev | pseudotime | 3 | 0.7047 | 0.4962 | 0.7058 | 0.6144 |
| batch | flow | all_four | 3 | 0.6852 | 0.663 | 0.7816 | 0.7009 |
| batch | flow | labelling | 3 | 0.9314 | 0.9197 | 0.578 | 0.5098 |
| batch | flow | labelling+velocity | 3 | 0.6852 | 0.663 | 0.7816 | 0.7009 |
| batch | flow | velocity | 3 | 0.1946 | 0.1369 | 0.6813 | 0.5685 |
| default | dev | all_four | 3 | 0.9816 | 0.9791 | 0.8279 | 0.7979 |
| default | dev | lineage | 3 | 0.9905 | 0.9879 | 0.4466 | 0.4298 |
| default | dev | lineage+pseudotime | 3 | 0.9816 | 0.9791 | 0.8279 | 0.7979 |
| default | dev | pseudotime | 3 | 0.9732 | 0.9703 | 0.8002 | 0.7614 |
| default | flow | all_four | 3 | 0.7059 | 0.6794 | 0.8303 | 0.8185 |
| default | flow | labelling | 3 | 0.9438 | 0.9276 | 0.6278 | 0.5873 |
| default | flow | labelling+velocity | 3 | 0.7059 | 0.6794 | 0.8303 | 0.8185 |
| default | flow | velocity | 3 | 0.1832 | 0.1338 | 0.7426 | 0.7119 |
| stress | dev | all_four | 3 | 0.9651 | 0.9545 | 0.8197 | 0.8075 |
| stress | dev | lineage | 3 | 0.9597 | 0.947 | 0.1994 | 0.1971 |
| stress | dev | lineage+pseudotime | 3 | 0.9651 | 0.9545 | 0.8197 | 0.8075 |
| stress | dev | pseudotime | 3 | 0.9687 | 0.9584 | 0.8052 | 0.7918 |
| stress | flow | all_four | 3 | 0.5983 | 0.5154 | 0.8361 | 0.8174 |
| stress | flow | labelling | 3 | 0.8876 | 0.8506 | 0.6167 | 0.5848 |
| stress | flow | labelling+velocity | 3 | 0.5983 | 0.5154 | 0.8361 | 0.8174 |
| stress | flow | velocity | 3 | 0.1415 | 0.1344 | 0.7427 | 0.725 |

## Source-addition and conflict summary (PCA/raw)
| preset | group | coverage_added_fraction | coverage_added_agreement | higher_decides_agreement | higher_truth_sided_fraction | coverage_non_decreasing | higher_decision_unchanged_exact |
|---|---|---|---|---|---|---|---|
| batch | dev | 0.3609 | 0.4877 | 0.9897 | 0.9869 | True | True |
| batch | flow | 0.2108 | 0.2156 | 0.9456 | 0.9722 | True | True |
| batch | dev | 0.345 | 0.6828 | 0.9955 | 0.9873 | True | True |
| batch | flow | 0.1912 | 0.3068 | 0.929 | 0.9641 | True | True |
| batch | dev | 0.401 | 0.9235 | 0.9968 | 0.9695 | True | True |
| batch | flow | 0.2089 | 0.3333 | 0.9197 | 0.9633 | True | True |
| default | dev | 0.3832 | 0.9749 | 0.9953 | 0.9057 | True | True |
| default | flow | 0.1858 | 0.3835 | 0.9529 | 0.9786 | True | True |
| default | dev | 0.3925 | 0.9708 | 0.9879 | 0.6552 | True | True |
| default | flow | 0.1904 | 0.3771 | 0.9508 | 0.9723 | True | True |
| default | dev | 0.3681 | 0.9739 | 0.9883 | 0.7164 | True | True |
| default | flow | 0.2312 | 0.3261 | 0.9276 | 0.9691 | True | True |
| stress | dev | 0.6058 | 0.9861 | 0.97 | 0.2432 | True | True |
| stress | flow | 0.192 | 0.2393 | 0.9162 | 0.9396 | True | True |
| stress | dev | 0.6303 | 0.9596 | 0.9622 | 0.4082 | True | True |
| stress | flow | 0.2022 | 0.2554 | 0.8961 | 0.94 | True | True |
| stress | dev | 0.6249 | 0.9578 | 0.947 | 0.4308 | True | True |
| stress | flow | 0.264 | 0.1683 | 0.8506 | 0.9548 | True | True |

## Complete outputs
- `results.csv`: per-unit combination scores.
- `summary.csv`: per-preset, embedding, variant, group, and combination aggregate scores.
- `by_day.csv`: the corresponding day-stratified scores.
- `RUN_METADATA.json`: user-file manifest, call audit, and gate records.
- Figures: `benchmarks/figures/orient/s03/`.
- S03 user manifest recorded 9 primary user files; stochastic status is report-only per Revision 2.
