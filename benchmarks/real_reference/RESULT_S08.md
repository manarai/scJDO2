# S08 real-data reference comparison

This record is descriptive: common preprocessing/statistic functions were applied to saved real and synthetic inputs. No generator or orientation model was changed.

## What ran

- 6 real references and 12 synthetic datasets contributed to `352` available synthetic-statistic comparisons.
- Distance: KS for saved distributions, total variation for class-share statistics, and median absolute log ratio for scalar feature vectors.
- A match requires distance ≤ max(real spread, fixed tolerance) for every applicable real reference; `within_real_range` independently records whether the scalar or distribution median falls in the real min–max range.

## Match summary

| preset | statistics | match_rate | matched |
|---|---:|---:|---:|
| v1 | 88 | 0.352 | 31 |
| v2_batch | 89 | 0.202 | 18 |
| v2_default | 88 | 0.216 | 19 |
| v2_stress | 87 | 0.230 | 20 |

## Matrix

See `compare.csv`; one deterministic panel per statistic is in `benchmarks/figures/real_reference/`.
