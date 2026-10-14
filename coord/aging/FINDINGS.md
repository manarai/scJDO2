# Findings — aging

Claude's running interpretation, one dated entry per accepted (or
stopped) task, newest last. Confirmatory and exploratory results are
labelled; only confirmatory ones speak to the frozen rules.

<!-- entry format
## YYYY-MM-DD — <ws>/<ID> (<kind>, <verdict>)
**Result.** the numbers that matter, with the rule they were judged by.
**Meaning.** what this says about the workstream's question; what it does not.
**Caveats.** sample size, QC concerns, assumptions.
**Next.** tasks proposed, and which the user approved.
-->

## 2026-10-14 — aging/A00 (planned, ACCEPT)

**Result.** §4b isotropy statistic on synthetic data, 40 seeds per
case: false-pass 0.075 for identical groups and for uniform 1.6×
widening (nominal 0.05), power 1.00 for single-axis 3× widening
(median S 2.21 vs null q95 0.11).

**Meaning.** The test now separates the two hypotheses it exists for:
uniform softening fails, single-direction loss passes. The §4a
Spearman test would have passed uniform softening (ρ > 0.95). No
biological claim yet; no real data has been run.

**Caveats.** Synthetic per-cell Jacobians are Gaussian-basin Hessians
with 15 % multiplicative noise; real operators are noisier and
centroid estimates use 30 cells. Calibration on real data comes from
the split-half null itself.

**Next.** aging/A01 (planned) issued.

## 2026-10-14 — aging/A01 (planned, REVISE — blocked at age-label audit)

**Result.** Fetch matches PREREG §1. 4-week cells (excluded by design)
are 10.2 % of lung, 7.5 % of marrow. Young/old cells: lung 3 612 /
18 420 (4 / 10 donors), marrow 3 490 / 33 703 (2 / 13 donors).
**Meaning.** Young cell counts will decide how many types pass the
≥ 200 gate. Two rule defects surfaced before any preprocessing and were
fixed by PREREG §2b: the 4-week exemption, and §2a's per-donor
downsampling (each mouse has one age).
**Caveats.** Few young donors, especially marrow (2). "Old" includes
"20 months and over" mice.
**Next.** A01 revision issued. Exploratory old-age split proposed (X01),
awaiting the user.
