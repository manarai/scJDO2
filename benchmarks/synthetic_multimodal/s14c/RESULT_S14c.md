# S14c — test construction failure

## Outcome

After traversing the complete transition reserve s17–s40, the total eligible set remains **s12 and s16** (2 of the required 4). The task directs a stop at this point. The final truth-only eligibility table was committed before any flow fit. No null calibration, transition flow fit, held-out fit, model-based truth scoring, gate score, or profile figure was produced. `T_null` and X1–X4 are unavailable (`null`), not passes or failures.

## Extended reserve

| Seed | Generation | Eligible | True peak bin | Minimum bin cells | Stability | Reason |
|---:|---|---:|---:|---:|---:|---|
| 17 | guard_stop | no | — | — | — | RSS reached guard threshold (12776652800 bytes >= 12000000000) |
| 18 | guard_stop | no | — | — | — | RSS reached guard threshold (12020203520 bytes >= 12000000000) |
| 19 | guard_stop | no | — | — | — | RSS reached guard threshold (12497469440 bytes >= 12000000000) |
| 20 | guard_stop | no | — | — | — | RSS reached guard threshold (12054757376 bytes >= 12000000000) |
| 21 | completed | no | 4 | 47 | 0.855 | one or more frozen criteria failed |
| 22 | guard_stop | no | — | — | — | RSS reached guard threshold (12363333632 bytes >= 12000000000) |
| 23 | guard_stop | no | — | — | — | RSS reached guard threshold (12157157376 bytes >= 12000000000) |
| 24 | guard_stop | no | — | — | — | RSS reached guard threshold (12279087104 bytes >= 12000000000) |
| 25 | guard_stop | no | — | — | — | RSS reached guard threshold (12078415872 bytes >= 12000000000) |
| 26 | guard_stop | no | — | — | — | RSS reached guard threshold (12031934464 bytes >= 12000000000) |
| 27 | guard_stop | no | — | — | — | RSS reached guard threshold (12061605888 bytes >= 12000000000) |
| 28 | guard_stop | no | — | — | — | RSS reached guard threshold (12349784064 bytes >= 12000000000) |
| 29 | completed | no | 3 | 46 | 0.995 | one or more frozen criteria failed |
| 30 | completed | no | 5 | 46 | 0.595 | one or more frozen criteria failed |
| 31 | completed | no | 3 | 130 | 0.465 | one or more frozen criteria failed |
| 32 | completed | no | 4 | 48 | 0.335 | one or more frozen criteria failed |
| 33 | completed | no | 6 | 43 | 0.890 | one or more frozen criteria failed |
| 34 | completed | no | 5 | 46 | 0.650 | one or more frozen criteria failed |
| 35 | completed | no | 4 | 44 | 0.645 | one or more frozen criteria failed |
| 36 | completed | no | 5 | 46 | 1.000 | one or more frozen criteria failed |
| 37 | completed | no | 0 | 148 | 0.620 | one or more frozen criteria failed |
| 38 | guard_stop | no | — | — | — | RSS reached guard threshold (12019417088 bytes >= 12000000000) |
| 39 | guard_stop | no | — | — | — | RSS reached guard threshold (12036210688 bytes >= 12000000000) |
| 40 | guard_stop | no | — | — | — | RSS reached guard threshold (12006047744 bytes >= 12000000000) |

Eligible seeds s12 and s16 were reused from S14b without regeneration. The table includes all frozen criteria, roots, profiles, and provenance. Guard-stopped seeds remain recorded as ineligible and have no manifest row or prepared array.

## Generator stops

- s17: `guard_stop`; peak RSS 12,792,348,672 bytes; wall 907.2 s.
- s18: `guard_stop`; peak RSS 12,088,508,416 bytes; wall 168.4 s.
- s19: `guard_stop`; peak RSS 12,497,469,440 bytes; wall 1187.0 s.
- s20: `guard_stop`; peak RSS 12,116,738,048 bytes; wall 1058.1 s.
- s22: `guard_stop`; peak RSS 12,384,174,080 bytes; wall 1192.2 s.
- s23: `guard_stop`; peak RSS 12,167,200,768 bytes; wall 1367.7 s.
- s24: `guard_stop`; peak RSS 12,279,087,104 bytes; wall 1311.6 s.
- s25: `guard_stop`; peak RSS 12,127,207,424 bytes; wall 979.0 s.
- s26: `guard_stop`; peak RSS 12,091,981,824 bytes; wall 1148.0 s.
- s27: `guard_stop`; peak RSS 12,063,817,728 bytes; wall 1586.3 s.
- s28: `guard_stop`; peak RSS 12,432,244,736 bytes; wall 1579.5 s.
- s38: `guard_stop`; peak RSS 12,073,304,064 bytes; wall 2352.9 s.
- s39: `guard_stop`; peak RSS 12,072,566,784 bytes; wall 2091.3 s.
- s40: `guard_stop`; peak RSS 12,045,615,104 bytes; wall 1048.1 s.

## Compute and versions

The measured outer logs cover 42 timed units and record 13.3983 CPU-hours. The first s17 outer time log is missing; including its three-thread wall-time upper bound gives **14.1543 CPU-hours**, under the 24 CPU-hour task limit. Thread limit was 3; per-process memory cap was 14 GB, with the stricter 12 GB generation guard.

Runtime: Python 3.13.2; NumPy 2.5.4; pandas 2.3.3; AnnData 0.11.4; Scanpy 1.12.4; Palantir 1.4.5; SciPy 1.18.1; scikit-learn 1.9.1; scvelo 0.3.4; PyTorch 2.14.1; Matplotlib 3.11.2.

## Outputs

- `g2_transition_eligibility.csv` is committed in Step 1 before any flow fit.
- `null_calibration.csv` and `test.csv` explicitly record `not_run`; `gates.json` records null thresholds and gate values.
- No truth-scored model result or figure is reported.
