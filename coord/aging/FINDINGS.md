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

## 2026-10-14 — aging/A01 revision 1 (planned, REVISE — blocked at lung HVG)

**Result.** Eligible types: lung 2 (fibroblast, NK), marrow 6. Lung
`seurat_v3` HVG fails per donor for donors with < ~100 cells. Donor IDs
show 30-month mice inside "20 months and over".
**Meaning.** PREREG §2c restores old = 18–24 months (30-month excluded;
eligibility unchanged) and sets HVG `span=1.0` for both tissues. The
pooled outcome will rest on 8 types, 6 from marrow.
**Caveats.** Lung latent trained on 776 young cells of 2 types.
**Next.** Revision 2 issued. X01 (18–21 vs 24 months) and X02
(30-month as very old) proposed, exploratory, awaiting the user.

## 2026-10-14 — aging/A01 (planned, ACCEPT, revision 2)

**Result.** Eligible types: lung 2 (fibroblast 263, NK 385 cells per
group), marrow 6 (231–727 per group). Donors young/old: lung 4/6,
marrow 2/10, after excluding 4-week and 30-month mice.
**Meaning.** Ready for A02. The §6 pooled majority is over 8 types.
**Caveats.** Lung reference trained on 648 young cells of 2 types;
marrow on 2 694 of 6; young-donor replication thin.
**Next.** A02 issued. X01/X02 exploratory, awaiting the user.
