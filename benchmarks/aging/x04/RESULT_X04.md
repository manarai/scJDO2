# EXPLORATORY — aging/X04 mapping-shrinkage check

## Question
Does a joint young+old scVI space expose age change that young-reference scArches mapping suppresses? This exploratory analysis does not modify the frozen A03 outcome.

## What ran
Three CPU joint scVI seeds (0–2), shared-mapping seed replicates (1–2; A02 seed 0 reused), and the locked 1,500-epoch seed-0 exact-ESM drift fit for every new latent. Per-type quantities used 30-cell centroid operators and `compare_types(..., n_null=1000, seed=0)`.

## Reading rule (fixed before the run)
* the type **shows hidden softening** if joint `stiff_ratio` ≤ 0.90 **and** shared `stiff_ratio` ≥ 0.95 **and** the joint and shared seed ranges do not overlap;
* **shows hidden displacement** if joint `offset_z` exceeds the shared `offset_z` by more than the larger of the two seed ranges;
* verdict for the arm: **"mapping hides change"** if ≥ 2 of the 3 basin types (lung fibroblast, lung NK, marrow granulocyte) or ≥ 4 of all 8 types show hidden softening or hidden displacement; otherwise **"no evidence that mapping hides change"**.

## Per-type seed means and flags
| tissue | cell_type | stiff_ratio_mean_joint | stiff_ratio_mean_shared | offset_z_mean_joint | offset_z_mean_shared | hidden_softening | hidden_displacement | shows_hidden_change |
|---|---|---|---|---|---|---|---|---|
| lung | fibroblast of lung | 1.064 | 1.08 | 0.1551 | 0.4853 | False | False | False |
| lung | natural killer cell | 0.9257 | 0.9983 | 0.1135 | 0.2024 | False | False | False |
| marrow | granulocyte | 1.031 | 1.035 | 0.1443 | 0.1985 | False | False | False |
| marrow | granulocytopoietic cell | 2.315 | 4.481 | 0.3008 | 0.3823 | False | False | False |
| marrow | hematopoietic precursor cell | 0.9841 | 1.084 | 0.1081 | 0.1335 | False | False | False |
| marrow | monocyte | 1.017 | 1.112 | 0.1242 | 0.2679 | False | False | False |
| marrow | precursor B cell | 1.01 | 1.006 | 0.1424 | 0.2025 | False | False | False |
| marrow | promonocyte | 0.9972 | 1.153 | 0.1393 | 0.2243 | False | False | False |

## Mechanical verdict
**no evidence that mapping hides change**

No interpretation beyond the fixed reading rule is made here.
