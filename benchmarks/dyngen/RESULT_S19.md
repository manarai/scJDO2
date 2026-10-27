# orient/S19 — dyngen realism

S08 reference statistics were imported from `scripts/s08_reference_stats.py`; the comparison helpers were imported from `scripts/s08_compare.py`. Neither source file was edited.

v3 reference source: cached local computation from shared v3 h5ad read-only.

## Match summary

| dataset | statistics | match rate | within-real-range rate |
|---|---:|---:|---:|
| bifurcating_s0 | 25 | 0.560 | 0.640 |
| bifurcating_s1 | 25 | 0.480 | 0.520 |
| bifurcating_s2 | 25 | 0.520 | 0.680 |
| cycle_s0 | 25 | 0.520 | 0.720 |
| cycle_s1 | 25 | 0.560 | 0.640 |
| cycle_s2 | 25 | 0.600 | 0.640 |
| linear_s0 | 25 | 0.480 | 0.640 |
| linear_s1 | 25 | 0.440 | 0.560 |
| linear_s2 | 25 | 0.440 | 0.640 |

## Notes

- A statistic matches when its KS distance is no larger than the maximum of the real-reference spread and the imported S08 fixed tolerance.
- dyngen has no new/total labelling readout or clone barcode input; S08 labelling and lineage rows are recorded as not applicable rather than derived from truth.
- `realism.csv` contains the per-statistic comparison and within-real-range table.

## Within-real-range table by statistic

| statistic | datasets | match rate | within-real-range rate | real range min | real range max |
|---|---:|---:|---:|---:|---:|
| C1_log10_library | 9 | 0.000 | 0.000 | 3.34294 | 3.87731 |
| C2_genes_detected | 9 | 0.000 | 0.000 | 1350.71 | 3140 |
| C3_hvg_log_mean | 9 | 0.333 | 1.000 | 0.0556541 | 0.233005 |
| C4_dispersion | 9 | 0.000 | 0.000 | 2.60829 | 4.58896 |
| C5_zero_fraction | 9 | 0.222 | 0.556 | 0.799848 | 0.936632 |
| C6_hvg_pair_correlation | 9 | 0.778 | 0.778 | -0.00337463 | 0.0209033 |
| C7_participation_ratio | 9 | 1.000 | 0.444 | 3.94035 | 8.38121 |
| C7_pca_variance_ratio | 9 | 0.444 | 0.556 | 0.0045109 | 0.0070786 |
| C8_leiden_share | 9 | 0.000 | 0.000 | 0.0259386 | 0.0924499 |
| S1_unspliced_fraction | 9 | 1.000 | 1.000 | 0.0778457 | 0.499185 |
| S2_unspliced_spliced_ratio | 9 | 1.000 | 1.000 | 0.0921888 | 359529 |
| S3_f_down | 9 | 0.000 | 0.667 | 0.10989 | 0.360501 |
| S3_f_up | 9 | 0.222 | 1.000 | 0.0753116 | 0.358965 |
| S3_gamma_prime | 9 | 0.000 | 1.000 | 0.100382 | 2.8408 |
| S3_n_fit_cells | 9 | 1.000 | 1.000 | 98 | 289 |
| S3_n_valid_cells | 9 | 1.000 | 1.000 | 1947 | 5780 |
| S3_phase_portrait_class | 9 | 0.000 | 0.000 | 0.179375 | 0.25625 |
| S3_zero_slope_fraction | 9 | 1.000 | 1.000 | 0 | 0.2875 |
| S3b_scvelo_switch_class | 9 | 0.778 | 1.000 | 0.1675 | 0.44125 |
| S4_dynamics_likelihood | 9 | 0.000 | 0.556 | 0.121549 | 0.258334 |
| S4_likelihood_ge_0_3 | 9 | 1.000 | 0.778 | 0.00625 | 0.2325 |
| S4_recovered_fraction | 9 | 1.000 | 0.444 | 0.17375 | 0.6275 |
| T2_palantir_entropy | 9 | 1.000 | 1.000 | 0 | 0.986104 |
| T2_root_time | 9 | 0.000 | 0.000 | 12 | 12 |
| T2_terminal_states | 9 | 1.000 | 1.000 | 0 | 3 |
