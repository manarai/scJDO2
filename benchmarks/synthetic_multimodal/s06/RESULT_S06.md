# orient/S06 executed result

## What ran
Observed-data fitted units, fixed scVelo/Palantir runs, and truth-separated scoring calls completed. Simulator truth was absent from every `orient()` input and used only in a separate scoring copy after orientation.

## Gates
- **gate_flow_labelling**: FAIL — `{"agreement_min": 0.8231498285944747, "oriented_min": 0.6682712431979908, "pass": false}`
- **gate_flow_check_guard**: PASS — `{"agreement_margin_min": 0.0, "oriented_margin_min": 0.0, "pass": true}`
- **gate_dev_lineage**: FAIL — `{"agreement_min": 0.7206502107, "pass": false}`
- **gate_dev_pseudotime**: PASS — `{"median": 0.9620140216698534, "minimum": 0.9601621471, "pass": true}`
- **gate_dev_coverage**: PASS — `{"oriented_min": 0.8153124334, "pass": true}`
- **gate_independence_hash**: PASS — `{"hash_all": true, "independence_all": true, "pass": true}`
- **gate_flags_stationary**: PASS — `{"minimum_non_equilibrium": 1.0, "pass": true}`
- **gate_j_auc**: FAIL — `{"minimum": 0.1814485891, "pass": false}`
- **gate_negative_control**: FAIL — `{"accepted_day4_n": 104, "day1_default_refused_n": 0, "day4_accepted_bad_start_n": 104, "day4_accepted_low_forward_n": 9, "n_day1_default": 90, "n_day4": 180, "pass": false}`

## Counts
- scored rows: 10872
- Palantir guard records: 360
- J class records: 54
- flags rows: 6
