**Figure 2 | A benchmark calibrated to six real datasets (S08, S09, S19).**
**a**, Six of the 20 S08 statistics (IDs C1–C8, G1–G5, L1–L2, S1–S4, T2; 32 sub-statistics).
Black dots: value (or median) in each real dataset (scNT-seq haematopoiesis and neurons, LARRY, Setty CD34⁺,
pancreas, dentate gyrus; not every statistic applies to every dataset); grey band: real min–max range. Coloured dots:
one per synthetic dataset (v1: 3 seeds of the default preset; v2 and v3: 3 seeds × each preset). Zero fraction:
real 0.84–0.96 vs v1 median 0.36,
v2 0.37, v3 0.87.
**b**, Whether the synthetic median lies in the real range, per statistic and preset (fraction of three seeds); grey, not
applicable. **c**, Fraction of statistics matched under the S08 rule (distance ≤ max(real spread, fixed tolerance) for every
applicable real reference): v1 0.35, v2_default 0.22,
v3_default 0.39, v3_batch 0.39, v3_stress 0.34,
v3_stationary 0.49; dyngen (S19, nine datasets, bifurcating/cycle/linear × 3 seeds, 25 applicable
statistics) 0.51 (range 0.44–0.60).
Against the 13 pre-fixed S09 calibration targets, v3_default met 6 of 13 on each seed (realism gate failed);
targets unmet on all three seeds: C2_genes_detected, G3_multi_time_clone_fraction, S1_unspliced_fraction, S3_steady_share, S4_recovered_fraction. The user accepted v3 with these misses documented (S09 decision).
**d**, Empirical CDFs of three saved S08 distributions for five real datasets and v1/v2 default seed 0 (v3 distributions
were not saved). Descriptive comparison; the S09 13-target count was pre-registered, the S08/S19 matrices are report only.
