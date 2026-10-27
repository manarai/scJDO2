# orient/S15 result

## Status

Complete: all 130 expected analysis checkpoints are present; frozen gates are reported below without retuning.

## Frozen configuration

S14 raw-coordinate reference flow J, PCA-15, epsilon factor 0.05, extrapolated last day, 96x96x96 drift, weight decay 1e-4, one member.

## Step 0 J_ref diagnosis

| c_star_dataset | dataset | jref_neighbours | k_flow | lambda1_sign_agreement | lambda1_spearman | median_norm_J | median_norm_J_over_norm_Jref | median_norm_Jref | median_per_cell_cosine | median_relative_error_J | median_relative_error_cstar_J | n_cells | basis_coordinates_confirmed | basis_observed_pca15_reconstruction_max_abs | basis_fd_dt | basis_fd_velocity_median_cosine | basis_fd_velocity_fraction_cosine_ge_0_99 | basis_jref_fd_neighbours | basis_jref_fd_median_per_cell_cosine | basis_jref_fd_fraction_per_cell_cosine_ge_0_99 | basis_jref_fd_median_relative_frobenius_difference | basis_jref_fd_lambda1_spearman | basis_jref_fd_lambda1_sign_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.0411895981009549 | v3_default_s2 | 15 | 15 | 0.4786556969885129 | -0.1074441731870673 | 0.050462097745676 | 0.0221204891329504 | 2.27223774779408 | -0.0007090253068626 | 1.0002416322119625 | 0.9999998002193132 | 12884 | True | 1.0213966632477425e-06 | 1e-05 | 0.9999999991257552 | 1.0 | 30 | 0.9999999999996928 | 1.0 | 1.0978848186714706e-06 | 0.9999999995904076 | 1.0 |
| -0.0351219540018196 | v3_default_s2 | 30 | 15 | 0.4786556969885129 | -0.086124506504575 | 0.050462097745676 | 0.0555747281294526 | 0.906198244592111 | -0.0004667936936187 | 1.001530505934192 | 1.0000008097364572 | 12884 | True | 1.0213966632477425e-06 | 1e-05 | 0.9999999991257552 | 1.0 | 30 | 0.9999999999996928 | 1.0 | 1.0978848186714706e-06 | 0.9999999995904076 | 1.0 |
| -0.035931139007609 | v3_transition_s2 | 15 | 15 | 0.912992859360447 | -0.014418615235494 | 0.0555229536873137 | 0.0242878360023773 | 2.2939952712572818 | 4.168597222284146e-06 | 1.000296216312548 | 1.0000004030746272 | 12884 | True | 1.104484780967141e-06 | 1e-05 | 0.9999999992807256 | 1.0 | 30 | 0.9999999999997128 | 1.0 | 1.061565289244492e-06 | 0.9999999996689596 | 1.0 |
| -0.0405533842047521 | v3_transition_s2 | 30 | 15 | 0.912992859360447 | -0.0096120403791615 | 0.0555229536873137 | 0.0630010317368953 | 0.8818270704459442 | -9.376836383406104e-05 | 1.002045456329467 | 1.0000025179200491 | 12884 | True | 1.104484780967141e-06 | 1e-05 | 0.9999999992807256 | 1.0 | 30 | 0.9999999999997128 | 1.0 | 1.061565289244492e-06 | 0.9999999996689596 | 1.0 |
| -0.1022089112176467 | v3_transition_s3 | 15 | 15 | 0.9687108886107636 | -0.0070739032071013 | 0.0586686697580618 | 0.0279295448554929 | 2.085472975675685 | -0.001164855655337 | 1.0004458236301312 | 1.0000009479386918 | 12784 | True | 1.066773130276033e-06 | 1e-05 | 1.000000000359036 | 1.0 | 30 | 0.9999999999998336 | 1.0 | 7.807779648299386e-07 | 0.999999999764514 | 1.0 |
| -0.0651076227190867 | v3_transition_s3 | 30 | 15 | 0.9687108886107636 | -0.0048844929358559 | 0.0586686697580618 | 0.0688705796250423 | 0.8427982209969436 | -0.0023314908994363 | 1.0026050320137423 | 1.0000001195831292 | 12784 | True | 1.066773130276033e-06 | 1e-05 | 1.000000000359036 | 1.0 | 30 | 0.9999999999998336 | 1.0 | 7.807779648299386e-07 | 0.999999999764514 | 1.0 |
| 0.0483764576160507 | v3_transition_s4 | 15 | 15 | 0.6204430625728722 | -0.0140845377474975 | 0.0553457610053048 | 0.0305462063697868 | 1.820043889576732 | 0.0026105692146006 | 1.0003734167585805 | 0.9999973270885062 | 12865 | True | 1.0180492324707302e-06 | 1e-05 | 0.9999999998243976 | 1.0 | 30 | 0.9999999999998416 | 1.0 | 7.62036608505089e-07 | 0.9999999997463912 | 1.0 |
| 0.0843429165514484 | v3_transition_s4 | 30 | 15 | 0.6204430625728722 | -0.0180795516346804 | 0.0553457610053048 | 0.0734698796448561 | 0.7534157227070052 | 0.0083407143288164 | 1.0020853273531474 | 0.9999684476958652 | 12865 | True | 1.0180492324707302e-06 | 1e-05 | 0.9999999998243976 | 1.0 | 30 | 0.9999999999998416 | 1.0 | 7.62036608505089e-07 | 0.9999999997463912 | 1.0 |

The scale ratios are far below one and the post-scaling errors remain approximately one with near-zero cosine/Spearman values; this supports no local directional relation, not a pure scale or time-unit mismatch.

## Amendment S15-A1 — infeasible matched-K Leiden

For every affected analysis, the fixed resolution bisection is repeated over Leiden seeds 0–9; the first exact-K seed is used. If none yields exact K, the smallest attempted resolution with at least K clusters is selected and closest PCA-15 centroids are merged pairwise to K; if that is impossible the comparator is marked infeasible with no gate-rule change.

| dataset | operator | k | drift_seed | view | requested_k | amendment | branch | leiden_seed | resolution | n_clusters_before_merge | n_seed_search_attempts | n_merges | merges | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_stationary_cycle_s1 | flow | 15 | 0 | operator_archetype | 8 | S15-A1 | exact_seed_search | 1 | 0.4375 |  | 68 | 0 | [] |  |
| v3_stationary_cycle_s1 | flow | 15 | 0 | descriptor_kmeans | 2 | S15-A1 | exact_seed_search | 0 | 0.125 |  | 4 | 0 | [] |  |

## Gates

- **gate_A1_saddle_archetype**: PASS
- **gate_A2_nmi**: FAIL
- **gate_A3_stability**: FAIL

Overall (one view must satisfy all A1–A3): FAIL.

## Full stability tables

All 130 expected analysis checkpoints are present. The complete, per-comparison table below covers k=10/15/20, all synthetic presets, and fixed-sample LARRY; it is report-only and did not alter A1–A3.

| dataset | view | comparison | k | left | right | ari |
| --- | --- | --- | --- | --- | --- | --- |
| v3_transition_s0 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.3218908640770161 |
| v3_transition_s0 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.3356107719392736 |
| v3_transition_s0 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.35636571430812 |
| v3_transition_s0 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.3565976542046167 |
| v3_transition_s0 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.45676519582083336 |
| v3_transition_s0 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.38452441702937973 |
| v3_transition_s0 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.3918769987024017 |
| v3_transition_s0 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.35685730740862626 |
| v3_transition_s0 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.3912851124427748 |
| v3_transition_s0 | operator_archetype | pca_k |  | 10 | 15 | 0.45809969380256005 |
| v3_transition_s0 | operator_archetype | pca_k |  | 10 | 20 | 0.3521426756493548 |
| v3_transition_s0 | operator_archetype | pca_k |  | 15 | 20 | 0.3417353441568767 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.14024984291382542 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.2098229966326856 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.20526623889211934 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.21884783740884875 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.27020167754277663 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.25273755402143483 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.13553998325755603 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.15433467387178335 |
| v3_transition_s0 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.33757275736732456 |
| v3_transition_s0 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.2506711116612898 |
| v3_transition_s0 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.17252271827734936 |
| v3_transition_s0 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.18792163385000646 |
| v3_transition_s1 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.3480500734877178 |
| v3_transition_s1 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.4300738465594153 |
| v3_transition_s1 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.38242511998661366 |
| v3_transition_s1 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.3137124271541144 |
| v3_transition_s1 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.2193135633941192 |
| v3_transition_s1 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.19031512835831355 |
| v3_transition_s1 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.4115089116948674 |
| v3_transition_s1 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.2515185968812767 |
| v3_transition_s1 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.2875860504192704 |
| v3_transition_s1 | operator_archetype | pca_k |  | 10 | 15 | 0.3421823494689504 |
| v3_transition_s1 | operator_archetype | pca_k |  | 10 | 20 | 0.31300277361288065 |
| v3_transition_s1 | operator_archetype | pca_k |  | 15 | 20 | 0.36559811603717923 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.33372624357393627 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.06361010975521263 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.018670407792804 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.03627631250906499 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.21787370152615954 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.07173567305120661 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.09207762681873481 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.19964641597853663 |
| v3_transition_s1 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.0909391273630203 |
| v3_transition_s1 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.3406898920352748 |
| v3_transition_s1 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.1678532426580577 |
| v3_transition_s1 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.1857256771965775 |
| v3_transition_s2 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.32038505211910073 |
| v3_transition_s2 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.2836932545019006 |
| v3_transition_s2 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.3027971685340454 |
| v3_transition_s2 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.32211028293579197 |
| v3_transition_s2 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.2691722109438515 |
| v3_transition_s2 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.33634811496913547 |
| v3_transition_s2 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.3391909087549355 |
| v3_transition_s2 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.3012881891984012 |
| v3_transition_s2 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.3890554109709317 |
| v3_transition_s2 | operator_archetype | pca_k |  | 10 | 15 | 0.309315579852492 |
| v3_transition_s2 | operator_archetype | pca_k |  | 10 | 20 | 0.3128405437442683 |
| v3_transition_s2 | operator_archetype | pca_k |  | 15 | 20 | 0.30003884633236505 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.4184401542193899 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.4660618681971871 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.551720936887245 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.5272736021254368 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.34135894006147466 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.37317705886718816 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.21108525152888255 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.40761708956573733 |
| v3_transition_s2 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.20336370799938594 |
| v3_transition_s2 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.45963671835308134 |
| v3_transition_s2 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.3639837421105937 |
| v3_transition_s2 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.36902426330598564 |
| v3_transition_s3 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.4395177072466367 |
| v3_transition_s3 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.48578277151398686 |
| v3_transition_s3 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.45413655514260337 |
| v3_transition_s3 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.2157287824709361 |
| v3_transition_s3 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.38541267583010125 |
| v3_transition_s3 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.2574985353702679 |
| v3_transition_s3 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.24715928633047257 |
| v3_transition_s3 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.28267000681078086 |
| v3_transition_s3 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.2829087190552246 |
| v3_transition_s3 | operator_archetype | pca_k |  | 10 | 15 | 0.36430658577926023 |
| v3_transition_s3 | operator_archetype | pca_k |  | 10 | 20 | 0.33330308159250527 |
| v3_transition_s3 | operator_archetype | pca_k |  | 15 | 20 | 0.3605885498772804 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.5239584315374842 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.40999446126734196 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.42520564606297423 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.20644157270130126 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.47812814789841546 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.20865021446430992 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.5145258115906884 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.5087435364817481 |
| v3_transition_s3 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.5360328300665024 |
| v3_transition_s3 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.4450238175307625 |
| v3_transition_s3 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.42332139761710913 |
| v3_transition_s3 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.4331126488109181 |
| v3_transition_s4 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.33506368370478173 |
| v3_transition_s4 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.40467886674645454 |
| v3_transition_s4 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.40979072322300075 |
| v3_transition_s4 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.36195520297556333 |
| v3_transition_s4 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.4391916775628189 |
| v3_transition_s4 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.4086717393254357 |
| v3_transition_s4 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.44757672444923974 |
| v3_transition_s4 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.42728021653726567 |
| v3_transition_s4 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.45785546886512085 |
| v3_transition_s4 | operator_archetype | pca_k |  | 10 | 15 | 0.3744274642539546 |
| v3_transition_s4 | operator_archetype | pca_k |  | 10 | 20 | 0.3240481030284833 |
| v3_transition_s4 | operator_archetype | pca_k |  | 15 | 20 | 0.36296549770153236 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.12869613733124272 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.04893717082590148 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.06441823916630725 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.010321939826240927 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.037746491308922665 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | -0.02409140730831449 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.1163857804932849 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.1940371765687821 |
| v3_transition_s4 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.14675157414894127 |
| v3_transition_s4 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.09793764015115682 |
| v3_transition_s4 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.01752746629531151 |
| v3_transition_s4 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.008384404745030605 |
| v3_default_s0 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.19373548517304606 |
| v3_default_s0 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.2181458463132421 |
| v3_default_s0 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.2111311056029142 |
| v3_default_s0 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.2466934238909655 |
| v3_default_s0 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.3327948578365374 |
| v3_default_s0 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.30185667632083923 |
| v3_default_s0 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.32223495181495015 |
| v3_default_s0 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.2749708309814751 |
| v3_default_s0 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.2598745617903963 |
| v3_default_s0 | operator_archetype | pca_k |  | 10 | 15 | 0.18933468787107166 |
| v3_default_s0 | operator_archetype | pca_k |  | 10 | 20 | 0.22902192281124523 |
| v3_default_s0 | operator_archetype | pca_k |  | 15 | 20 | 0.18468649824471228 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.2690697446312413 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.26883284230234716 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.2528655400961297 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.2939358364227335 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.315264254087983 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.33282644602211997 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.41592094408113794 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.38290354670406646 |
| v3_default_s0 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.38421454355738394 |
| v3_default_s0 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.31820848036910393 |
| v3_default_s0 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.3817946465279956 |
| v3_default_s0 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.3889622806581323 |
| v3_default_s1 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.15601728993047564 |
| v3_default_s1 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.14770719332942198 |
| v3_default_s1 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.12963924373803326 |
| v3_default_s1 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.229087181528874 |
| v3_default_s1 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.20142026832659743 |
| v3_default_s1 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.2137232244735512 |
| v3_default_s1 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.36056750300501844 |
| v3_default_s1 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.3765098978386952 |
| v3_default_s1 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.49615406568542 |
| v3_default_s1 | operator_archetype | pca_k |  | 10 | 15 | 0.10729259640921049 |
| v3_default_s1 | operator_archetype | pca_k |  | 10 | 20 | 0.12793064991118253 |
| v3_default_s1 | operator_archetype | pca_k |  | 15 | 20 | 0.24704418146862434 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.16380209897335077 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.11433942482417037 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.12879668131003566 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.08935816917691132 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.10076476445064825 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.1453447516404923 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.15572617600169802 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.1495225179810675 |
| v3_default_s1 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.14595093853121927 |
| v3_default_s1 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.06533249585756545 |
| v3_default_s1 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.10344760669552533 |
| v3_default_s1 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.05624538112933347 |
| v3_default_s2 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.24689898095982613 |
| v3_default_s2 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.16684494037936307 |
| v3_default_s2 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.140973837598998 |
| v3_default_s2 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.25425072374525187 |
| v3_default_s2 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.22351078423159884 |
| v3_default_s2 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.18875628813439072 |
| v3_default_s2 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.3017275293505781 |
| v3_default_s2 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.19084340300119776 |
| v3_default_s2 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.2308526093814959 |
| v3_default_s2 | operator_archetype | pca_k |  | 10 | 15 | 0.24942278471079185 |
| v3_default_s2 | operator_archetype | pca_k |  | 10 | 20 | 0.19304054093254247 |
| v3_default_s2 | operator_archetype | pca_k |  | 15 | 20 | 0.2859807785971628 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.34288255076054375 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.31411756212783304 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.4606160426445623 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.17660230320025774 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.11935537055528381 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.1765750623690693 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.229092907468706 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.19319528096124813 |
| v3_default_s2 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.21374711061283036 |
| v3_default_s2 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.14568074088269167 |
| v3_default_s2 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.11639271419365363 |
| v3_default_s2 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.14909547080723534 |
| v3_default_s3 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.255522392320909 |
| v3_default_s3 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.1761708421212013 |
| v3_default_s3 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.24549328787850233 |
| v3_default_s3 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.22797357546713576 |
| v3_default_s3 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.22996718105312336 |
| v3_default_s3 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.311160089432987 |
| v3_default_s3 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.19831608002575 |
| v3_default_s3 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.2693683330002431 |
| v3_default_s3 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.21882222668055568 |
| v3_default_s3 | operator_archetype | pca_k |  | 10 | 15 | 0.2189220663279851 |
| v3_default_s3 | operator_archetype | pca_k |  | 10 | 20 | 0.2056868779711419 |
| v3_default_s3 | operator_archetype | pca_k |  | 15 | 20 | 0.27034209971565903 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.283056378851575 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.23995612994392182 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.271162419675682 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.2748995373363891 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.308565088261044 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.3212095955405308 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.21206984657069594 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.338464361004071 |
| v3_default_s3 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.18157054007180126 |
| v3_default_s3 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.1188693252747349 |
| v3_default_s3 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.16285262683946833 |
| v3_default_s3 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.3785462517996266 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.07782136077115132 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.07651209705655686 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.07032543795097577 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.21186321261289368 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.18936587030421068 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.1694266786646145 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.22132695934671118 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.2331498696063312 |
| v3_stationary_cycle_s0 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.20745288542315632 |
| v3_stationary_cycle_s0 | operator_archetype | pca_k |  | 10 | 15 | 0.024243630743909155 |
| v3_stationary_cycle_s0 | operator_archetype | pca_k |  | 10 | 20 | 0.03255863114003949 |
| v3_stationary_cycle_s0 | operator_archetype | pca_k |  | 15 | 20 | 0.1455316887722883 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.01856422537170216 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.029621864969143718 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.019726909470337377 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.028918452406536073 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.030563849377026518 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.05970267231427562 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.1121184100633354 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.059578902170676594 |
| v3_stationary_cycle_s0 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.11580275130377042 |
| v3_stationary_cycle_s0 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.0022214190248651803 |
| v3_stationary_cycle_s0 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.0027085467287421444 |
| v3_stationary_cycle_s0 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.011301320553069457 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.0728166374917226 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.07749122433634993 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.05280914641521424 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.14554647306195123 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.11299329802126684 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.10893470468080613 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.16941919231831082 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.16691720935797844 |
| v3_stationary_cycle_s1 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.18338912032842097 |
| v3_stationary_cycle_s1 | operator_archetype | pca_k |  | 10 | 15 | 0.04864180501495469 |
| v3_stationary_cycle_s1 | operator_archetype | pca_k |  | 10 | 20 | 0.050867975500159146 |
| v3_stationary_cycle_s1 | operator_archetype | pca_k |  | 15 | 20 | 0.12288127052543821 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.04693931920047066 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.04154459161897724 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.03191081554693467 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.22752068211347484 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.22756441646477427 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.22620242682058048 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.126278088724347 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.28006445799741564 |
| v3_stationary_cycle_s1 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.15197940785535718 |
| v3_stationary_cycle_s1 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.004354835024413246 |
| v3_stationary_cycle_s1 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.002256405459474028 |
| v3_stationary_cycle_s1 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.15759486936335287 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.07956849616778228 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.0729946689112097 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.07481490907764886 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.1360122752911562 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.12803752154161635 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.1540557924134673 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.20990069280501508 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.19498239573803644 |
| v3_stationary_cycle_s2 | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.15436248152639057 |
| v3_stationary_cycle_s2 | operator_archetype | pca_k |  | 10 | 15 | 0.03787487883976627 |
| v3_stationary_cycle_s2 | operator_archetype | pca_k |  | 10 | 20 | 0.048231461905285586 |
| v3_stationary_cycle_s2 | operator_archetype | pca_k |  | 15 | 20 | 0.1289354899446356 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.015388313560940621 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.016258815497648606 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.010776149478061563 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.07491436834023679 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.07414162740154762 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.12566308490978464 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.08795477818979315 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.1689758706962844 |
| v3_stationary_cycle_s2 | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.11042580875324287 |
| v3_stationary_cycle_s2 | descriptor_kmeans | pca_k |  | 10 | 15 | 0.009628152146596963 |
| v3_stationary_cycle_s2 | descriptor_kmeans | pca_k |  | 10 | 20 | 0.000890901368576563 |
| v3_stationary_cycle_s2 | descriptor_kmeans | pca_k |  | 15 | 20 | 0.05386899817684117 |
| larry | operator_archetype | drift_seed | 10.0 | 0 | 1 | 0.4430040575606216 |
| larry | operator_archetype | drift_seed | 10.0 | 0 | 2 | 0.48403645250803323 |
| larry | operator_archetype | drift_seed | 10.0 | 1 | 2 | 0.5107245533471442 |
| larry | operator_archetype | drift_seed | 15.0 | 0 | 1 | 0.6451271251744878 |
| larry | operator_archetype | drift_seed | 15.0 | 0 | 2 | 0.7917157900152573 |
| larry | operator_archetype | drift_seed | 15.0 | 1 | 2 | 0.6022052012936779 |
| larry | operator_archetype | drift_seed | 20.0 | 0 | 1 | 0.5708929009089838 |
| larry | operator_archetype | drift_seed | 20.0 | 0 | 2 | 0.6018131793847352 |
| larry | operator_archetype | drift_seed | 20.0 | 1 | 2 | 0.5588497981738334 |
| larry | operator_archetype | pca_k |  | 10 | 15 | 0.42884600773448406 |
| larry | operator_archetype | pca_k |  | 10 | 20 | 0.48415356714138247 |
| larry | operator_archetype | pca_k |  | 15 | 20 | 0.6373924192639767 |
| larry | descriptor_kmeans | drift_seed | 10.0 | 0 | 1 | 0.7010556461152987 |
| larry | descriptor_kmeans | drift_seed | 10.0 | 0 | 2 | 0.7103338940393659 |
| larry | descriptor_kmeans | drift_seed | 10.0 | 1 | 2 | 0.6830674846027671 |
| larry | descriptor_kmeans | drift_seed | 15.0 | 0 | 1 | 0.7072513265752158 |
| larry | descriptor_kmeans | drift_seed | 15.0 | 0 | 2 | 0.7647232754046454 |
| larry | descriptor_kmeans | drift_seed | 15.0 | 1 | 2 | 0.7641285708452924 |
| larry | descriptor_kmeans | drift_seed | 20.0 | 0 | 1 | 0.7304324631434995 |
| larry | descriptor_kmeans | drift_seed | 20.0 | 0 | 2 | 0.7344268802650613 |
| larry | descriptor_kmeans | drift_seed | 20.0 | 1 | 2 | 0.8190099245165947 |
| larry | descriptor_kmeans | pca_k |  | 10 | 15 | 0.6000600521952311 |
| larry | descriptor_kmeans | pca_k |  | 10 | 20 | 0.5500263741322048 |
| larry | descriptor_kmeans | pca_k |  | 15 | 20 | 0.7298623148109946 |

## S15-A2 completion

The detached remainder runner completed 158 flow/analysis stages: 158 completed, 0 skipped, 0 failed, and 0 retries. `pending_units.csv` is therefore a zero-row completion record.

## Report-only cycle and real data

| dataset | operator | k | drift_seed | high_rotation_archetypes | pass_definition | per_archetype_median | threshold | truth_rotation_ratio_median | view |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_stationary_cycle_s0 | flow | 10 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6275915388004528, "1": 0.6008898036012085, "2": 0.6990420946358619, "3": 0.6053633511866576, "4": 0.6261967225575532, "5": 0.6351032922010014, "6": 0.6746847348596472, "7": 0.638776259527724} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 10 | 0 | ["0", "1"] | True | {"0": 0.5301006594514837, "1": 0.7397718847203534} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 10 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6751250996580006, "1": 0.5613213883607844, "2": 0.6468455040827874, "3": 0.7124582210467295, "4": 0.6837312775859075, "5": 0.5615585678415353, "6": 0.7374198746707159, "7": 0.5758697379381283} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 10 | 1 | ["0", "1"] | True | {"0": 0.7736266412185542, "1": 0.5194394544194556} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 10 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6705502217099396, "1": 0.635668667332234, "2": 0.6029871681964566, "3": 0.5253030745078495, "4": 0.5655160541163178, "5": 0.7440647065540222, "6": 0.6717250361945187, "7": 0.6723803078866692} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 10 | 2 | ["0", "1"] | True | {"0": 0.7557891053718961, "1": 0.49049935023301006} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 15 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9779827269841845, "1": 0.9700705704636914, "2": 0.9653314073262754, "3": 0.9817760844888439, "4": 0.9825310996055093, "5": 0.983078196723963, "6": 0.9804721431708167, "7": 0.9801344150823418} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 15 | 0 | ["0", "1"] | True | {"0": 0.9656525856026741, "1": 0.9837555581460486} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 15 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9807781422630897, "1": 0.9747613352478448, "2": 0.9817352172913109, "3": 0.9900929937258127, "4": 0.9860210866246721, "5": 0.9885758065357144, "6": 0.9827539061820807, "7": 0.9779188561719703} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 15 | 1 | ["0", "1"] | True | {"0": 0.9675818159784764, "1": 0.9906913419592066} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 15 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9706720126002445, "1": 0.9788818546213516, "2": 0.9704769362201602, "3": 0.9804004524495245, "4": 0.9937879555180322, "5": 0.9659223999754396, "6": 0.9753871375052938, "7": 0.9729960637674322} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 15 | 2 | ["0", "1"] | True | {"0": 0.9473487273542323, "1": 0.9885459411264872} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 20 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9842096065978388, "1": 0.9790543610379362, "2": 0.9637417131150074, "3": 0.9866560559483626, "4": 0.9914525607361964, "5": 0.9826291799630237, "6": 0.9787549683825403, "7": 0.9897724516909666} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 20 | 0 | ["0", "1"] | True | {"0": 0.9627724941544882, "1": 0.9903573273075593} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 20 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9878918834731459, "1": 0.9343317370325543, "2": 0.9783556518186886, "3": 0.9914505606698133, "4": 0.9882962030563984, "5": 0.9887290608723165, "6": 0.9850134217099007, "7": 0.9854909470566953} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 20 | 1 | ["0", "1"] | True | {"0": 0.9614640130701422, "1": 0.9909349649154836} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | flow | 20 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9781478435425783, "1": 0.9778927399430128, "2": 0.978071983254901, "3": 0.984202764678644, "4": 0.9845440399508486, "5": 0.943190068130237, "6": 0.9798428643573925, "7": 0.990526408010513} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | flow | 20 | 2 | ["0", "1"] | True | {"0": 0.9883728413228239, "1": 0.9518283551946998} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s0 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0} | 0.29398520574764014 | 0.5879704114952803 | operator_archetype |
| v3_stationary_cycle_s0 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0} | 0.29398520574764014 | 0.5879704114952803 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 10 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6139789767033683, "1": 0.6635849328217104, "2": 0.7060396907174632, "3": 0.6724377817331672, "4": 0.5266613821211208, "5": 0.6922307487147078, "6": 0.6292623218588528, "7": 0.5697326889148844} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 10 | 0 | ["0", "1"] | True | {"0": 0.7182970601798355, "1": 0.541729655559877} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 10 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6435454033936232, "1": 0.6186220321605477, "2": 0.6501834028804565, "3": 0.6092082595809373, "4": 0.6487772910332377, "5": 0.6025160963742353, "6": 0.6183537377792868, "7": 0.6205219408174546} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 10 | 1 | ["0", "1"] | True | {"0": 0.5348225013318838, "1": 0.6895610937392371} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 10 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6797841847066326, "1": 0.6026093960149745, "2": 0.6281555148020842, "3": 0.5908682309594389, "4": 0.6418191714156051, "5": 0.5680712814216164, "6": 0.645029198625025, "7": 0.6653640943786716} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 10 | 2 | ["0", "1"] | True | {"0": 0.712049497754407, "1": 0.5453623989812235} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 15 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9797016807643647, "1": 0.9776343705505339, "2": 0.9780275838402251, "3": 0.9788022967569383, "4": 0.9726012305641182, "5": 0.9946031205226699, "6": 0.9837333341197514, "7": 0.9756054303624295} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 15 | 0 | ["0", "1"] | True | {"0": 0.9796373968168948, "1": 0.9785898476624681} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 15 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9888271105853388, "1": 0.9821050874393034, "2": 0.9825971438259886, "3": 0.9853917648287699, "4": 0.9751016306691003, "5": 0.9824398968156545, "6": 0.9831741252863793, "7": 0.9805784325562434} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 15 | 1 | ["0", "1"] | True | {"0": 0.9837083670457988, "1": 0.9813674354731245} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 15 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9892399437444556, "1": 0.9705229170250737, "2": 0.9812926031218798, "3": 0.9772619419256584, "4": 0.9852770837740127, "5": 0.9872554621566678, "6": 0.9710651143563346, "7": 0.9815861458841595} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 15 | 2 | ["0", "1"] | True | {"0": 0.9829843242337497, "1": 0.9785319880101625} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 20 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9809187959920744, "1": 0.9897525506798794, "2": 0.9821904143863547, "3": 0.9738077286266553, "4": 0.9821783536397594, "5": 0.9623727372223934, "6": 0.9761691051607999, "7": 0.9750913369833594} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 20 | 0 | ["0", "1"] | True | {"0": 0.972661292748791, "1": 0.9831595588297417} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 20 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9743810254904547, "1": 0.9699075034757333, "2": 0.9775847986121571, "3": 0.9771426289824028, "4": 0.9830414173552318, "5": 0.9755562719368285, "6": 0.9880379057535937, "7": 0.9781885498469853} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 20 | 1 | ["0", "1", "2"] | True | {"0": 0.9810237422494394, "1": 0.9784045858225575, "2": 0.977984634720618} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | flow | 20 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9783600811307995, "1": 0.9691493322221602, "2": 0.9835554041466528, "3": 0.9773701628417892, "4": 0.9834870193415134, "5": 0.9767899273201087, "6": 0.9811657123598325, "7": 0.9774329758212345} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | flow | 20 | 2 | ["0", "1"] | True | {"0": 0.9804208633213947, "1": 0.976952683155148} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s1 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0} | 0.29428433803626114 | 0.5885686760725223 | operator_archetype |
| v3_stationary_cycle_s1 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0} | 0.29428433803626114 | 0.5885686760725223 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 10 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6747382152022688, "1": 0.6328817591973757, "2": 0.644287295705089, "3": 0.7127221555108083, "4": 0.609850296811788, "5": 0.6333237876345936, "6": 0.7121165743133588, "7": 0.6555296900096843} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 10 | 0 | ["0", "1"] | True | {"0": 0.5370426093724571, "1": 0.7196186498891034} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 10 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.6794338353412095, "1": 0.6449058862370098, "2": 0.6621248392100079, "3": 0.532340923838598, "4": 0.6422728603286414, "5": 0.6043762037833224, "6": 0.5845146364693973, "7": 0.6228233220135312} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 10 | 1 | ["0", "1"] | True | {"0": 0.5089099511474717, "1": 0.7112975559748036} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 10 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.7123181398202453, "1": 0.5721449594571137, "2": 0.6598417830288259, "3": 0.6826662964377439, "4": 0.7279711540381332, "5": 0.7033095681107204, "6": 0.6197420442277004, "7": 0.6718376703040403} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 10 | 2 | ["0", "1"] | True | {"0": 0.7408064806240857, "1": 0.5814981110686017} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 15 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.985707931095026, "1": 0.9880278192461659, "2": 0.9843220664010565, "3": 0.9866951105407133, "4": 0.982568971827194, "5": 0.9926409927957329, "6": 0.9849242879800285, "7": 0.9849994375954361} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 15 | 0 | ["0", "1"] | True | {"0": 0.9903993750422411, "1": 0.9775884029812112} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 15 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9888374814293455, "1": 0.980270419394502, "2": 0.991053847063313, "3": 0.9766213583346335, "4": 0.979122011082221, "5": 0.9779121019155765, "6": 0.9900418642761996, "7": 0.9861372394444671} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 15 | 1 | ["0", "1"] | True | {"0": 0.9912166382339379, "1": 0.9718807581087034} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 15 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9903697080864291, "1": 0.9840248931241513, "2": 0.9893944065443052, "3": 0.9919169115921298, "4": 0.9844912482140217, "5": 0.9912806483937023, "6": 0.9905211119532614, "7": 0.9807435363259844} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 15 | 2 | ["0", "1"] | True | {"0": 0.9742030777635139, "1": 0.9923071853384953} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 20 | 0 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9872390192033234, "1": 0.9864742143888576, "2": 0.9890003253200106, "3": 0.9835434685848541, "4": 0.9903863479837502, "5": 0.9806654187248989, "6": 0.9843847174664778, "7": 0.9753226875434059} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 20 | 0 | ["0", "1"] | True | {"0": 0.9802856102516868, "1": 0.9884160627001379} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 20 | 1 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9874752001370433, "1": 0.9807088100087193, "2": 0.992053427126706, "3": 0.9798417394597125, "4": 0.9859805517860809, "5": 0.9816854875690764, "6": 0.9851332722325669, "7": 0.9792391986349251} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 20 | 1 | ["0", "1"] | True | {"0": 0.9711681007580943, "1": 0.9909582229378802} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | flow | 20 | 2 | ["0", "1", "2", "3", "4", "5", "6", "7"] | True | {"0": 0.9912782149162764, "1": 0.984180012158209, "2": 0.9876860894858528, "3": 0.9874149019283243, "4": 0.9844977930440231, "5": 0.9926606513274019, "6": 0.9873945407814623, "7": 0.98812623521431} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | flow | 20 | 2 | ["0", "1"] | True | {"0": 0.9826196605899779, "1": 0.9910331530959231} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |
| v3_stationary_cycle_s2 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0} | 0.2939558438301313 | 0.5879116876602626 | operator_archetype |
| v3_stationary_cycle_s2 | score | 15 | 0 | [] | False | {"0": 0.0, "1": 0.0} | 0.2939558438301313 | 0.5879116876602626 | descriptor_kmeans |

| dataset | operator | k | drift_seed | view | n_cells | ari | k_selected | leiden | leiden_ari | leiden_nmi | nmi | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| larry | flow | 10 | 0 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.296875, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "bracketed": true, "leiden_seed": 0, "resolution": 0.296875}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d0.json |
| larry | flow | 10 | 0 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d0.json |
| larry | flow | 10 | 1 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.296875, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "bracketed": true, "leiden_seed": 0, "resolution": 0.296875}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d1.json |
| larry | flow | 10 | 1 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d1.json |
| larry | flow | 10 | 2 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.296875, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 10, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.28125}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.296875}], "bracketed": true, "leiden_seed": 0, "resolution": 0.296875}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d2.json |
| larry | flow | 10 | 2 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k10_d2.json |
| larry | flow | 15 | 0 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.359375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.359375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d0.json |
| larry | flow | 15 | 0 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.037109375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.037109375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d0.json |
| larry | flow | 15 | 1 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.359375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.359375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d1.json |
| larry | flow | 15 | 1 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.037109375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.037109375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d1.json |
| larry | flow | 15 | 2 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.359375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.359375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d2.json |
| larry | flow | 15 | 2 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.037109375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.037109375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k15_d2.json |
| larry | flow | 20 | 0 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.3125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.3125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d0.json |
| larry | flow | 20 | 0 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d0.json |
| larry | flow | 20 | 1 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.3125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.3125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d1.json |
| larry | flow | 20 | 1 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d1.json |
| larry | flow | 20 | 2 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.3125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.3125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.3125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d2.json |
| larry | flow | 20 | 2 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.03125, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 16, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 11, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.03125}], "bracketed": true, "leiden_seed": 0, "resolution": 0.03125}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_flow_k20_d2.json |
| larry | score | 15 | 0 | operator_archetype | 12000 |  | 8 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.359375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 9, "resolution": 0.375}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.3125}, {"leiden_seed": 0, "n_clusters": 7, "resolution": 0.34375}, {"leiden_seed": 0, "n_clusters": 8, "resolution": 0.359375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.359375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_score_k15_d0.json |
| larry | score | 15 | 0 | descriptor_kmeans | 12000 |  | 2 | {"amendment": "S15-A1", "attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "branch": "exact_seed_search", "leiden_seed": 0, "merges": [], "resolution": 0.037109375, "seed_search": [{"attempts": [{"leiden_seed": 0, "n_clusters": 19, "resolution": 1.0}, {"leiden_seed": 0, "n_clusters": 13, "resolution": 0.5}, {"leiden_seed": 0, "n_clusters": 6, "resolution": 0.25}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.125}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03125}, {"leiden_seed": 0, "n_clusters": 4, "resolution": 0.046875}, {"leiden_seed": 0, "n_clusters": 3, "resolution": 0.0390625}, {"leiden_seed": 0, "n_clusters": 1, "resolution": 0.03515625}, {"leiden_seed": 0, "n_clusters": 2, "resolution": 0.037109375}], "bracketed": true, "leiden_seed": 0, "resolution": 0.037109375}]} |  |  |  | benchmarks/synthetic_multimodal/s15/checkpoints/analysis_larry_score_k15_d0.json |
