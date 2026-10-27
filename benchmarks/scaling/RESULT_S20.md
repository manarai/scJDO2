# S20 result — scalable LARRY transport

The full S20 report is in [`coord/orient/reports/S20.md`](../../coord/orient/reports/S20.md), with machine-readable metrics in [`coord/orient/reports/S20.json`](../../coord/orient/reports/S20.json).

| Gate | Result | Value |
| --- | --- | --- |
| G1 agreement at 4,000/day | PASS | minibatch median velocity cosine 0.999962; flow-J lambda1 Spearman 0.999963 |
| G2 full LARRY | PASS | minibatch full OT 290.72 s / 2,544,615,424 bytes; full fit/J 1,278.34 s / 1,305,067,520 bytes |

Best scalable solver: `minibatch`. The complete agreement and scaling tables are [`agreement.csv`](agreement.csv) and [`scaling.csv`](scaling.csv). Dense full-LARRY OT was skipped because its larger float64 pairwise cost matrix is approximately 21.0 GB before solver workspace.
