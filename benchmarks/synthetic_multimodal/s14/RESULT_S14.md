# orient/S14 — held-out Part C

## Locked choices
* Truth-free Revision 1 k*: **15**.
* Selected development configuration: `{"ensemble": 5, "eps": 0.025, "hidden": [128, 128, 128], "last": "extrapolate", "space": "whitened", "wd": 0.001}`.
* Synthetic truth was accessed only after each observed OT/drift fit completed; test seeds were not used to select k or configuration.

## Gates
* **gate_G1_auc_band: FAIL** — selected fork AUC >= 0.85 on every transition test seed at k* and immediate grid neighbours
* **gate_G2_transition: FAIL** — informative transition bin absolute offset <= 1; nofork peak/median below every transition-test ratio
* **gate_G3_larry: FAIL** — selected LARRY lineage agreement exceeds all five nulls and held-out flow energy beats both baselines
* **gate_G4_k_sanity: FAIL** — selected development fork AUC >= 0.85 on both transition development seeds at k*

## Test output
* `test.csv` has 86 rows; `gates.json` records all component values.
* Strict Palantir r0 provenance is in `palantir_roots_test.csv` (14 synthetic test datasets).
* Timed CPU ledger: **9.664 CPU-h** / 24.000 CPU-h (closed logs available at finalization).
