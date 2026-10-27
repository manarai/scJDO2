# S11 real-flow results

## What ran
- Read-only scNT-seq and LARRY sources in their fixed common PCA-30 embeddings.
- Measured flow (seeds 0–2), local ridge, SparseVFC analytic, and score operators; 5 deterministic permutation nulls per comparison block.

## Criteria (verbatim)
- Reproducible: independent-source measured-flow λ1 Spearman and median |cos| exceed the largest of five nulls.
- Estimator-robust: measured vs local and measured vs Dynamo agree above their matching nulls likewise.

## Criterion table

| comparison_kind | left | right | right_estimator | lambda1_spearman | null_lambda1_spearman_max | eigenvector_abs_cosine | null_eigenvector_abs_cosine_max | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_sources | ot | lineage | measured_s0 | 0.268476 | 0.270825 | 0.287919 | 0.321324 | False |
| estimators | ot | ot | local | 0.714633 | -0.0294981 | 0.486774 | 0.212311 | True |
| estimators | ot | ot | dynamo | 0.934063 | 0.155545 | 0.946295 | 0.299724 | True |
| estimators | lineage | lineage | local | 0.340651 | -0.0194866 | 0.373703 | 0.315721 | True |
| estimators | lineage | lineage | dynamo | 0.570695 | -0.136732 | 0.469048 | 0.454154 | True |
| independent_sources | labelling | scvelo | measured_s0 | 0.224651 | 0.0928836 | 0.146078 | 0.133079 | True |
| independent_sources | labelling | dynamo_gene | measured_s0 | -0.325645 | 0.308529 | 0.175001 | 0.203993 | False |
| independent_sources | scvelo | dynamo_gene | measured_s0 | -0.0042141 | 0.133692 | 0.130328 | 0.14551 | False |
| estimators | labelling | labelling | local | -0.344086 | 0.0335254 | 0.173864 | 0.187096 | False |
| estimators | labelling | labelling | dynamo | -0.178374 | -0.134722 | 0.236189 | 0.281711 | False |
| estimators | scvelo | scvelo | local | 0.0527043 | 0.118 | 0.371154 | 0.475756 | False |
| estimators | scvelo | scvelo | dynamo | -0.0369631 | 0.0578658 | 0.515158 | 0.503926 | False |
| estimators | dynamo_gene | dynamo_gene | local | 0.310561 | 0.135758 | 0.215677 | 0.260187 | False |
| estimators | dynamo_gene | dynamo_gene | dynamo | 0.676322 | 0.189238 | 0.582421 | 0.390725 | True |

## Score-vs-flow summary

| dataset | source | lambda1_spearman_all_cells | leading_eigenvector_abs_cosine_median | n_cells |
| --- | --- | --- | --- | --- |
| larry | lineage | 0.105234 | 0.257303 | 1321 |
| larry | ot | 0.261506 | 0.183281 | 8000 |
| scnt_hema | dynamo_gene | 0.241428 | 0.171075 | 1947 |
| scnt_hema | labelling | -0.461079 | 0.166832 | 1947 |
| scnt_hema | scvelo | -0.143284 | 0.151354 | 1947 |

## Notes
- Stored processed Dynamo VecFld_pca is report-only in its own PCA basis and is never cross-basis compared.
