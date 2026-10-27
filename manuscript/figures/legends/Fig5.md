**Figure 5 | Real data: lineage agreement, held-out prediction and scale (S14, S11, S20).** LARRY in vitro haematopoiesis
(GSE140802; days 2/4/6); dense analyses use a fixed sample of 4,000 cells per day (`default_rng(0)`); barcodes used for scoring only.
**a**, S14 held-out test (pre-registered; gate G3 applies to the selected configuration and failed). Left, Spearman between flow-J λ₁ and
lineage displacement (◆) vs five permutation nulls (●): reference 0.56 (null max 0.15),
selected 0.55 (null max 0.14). Right, energy distance between the day-6 distribution
predicted by integrating the fitted drift from day 4 and the held-out day 6: reference flow 0.69, selected
flow 0.94; mean-displacement baseline 0.87, no-movement
baseline 0.88.
**b**, S11 (pre-registered criteria; 6 of 14 met). Left, per-cell cosine between velocity sources: LARRY OT vs clonal displacement
(n = 1321 cells with a lineage velocity; median 0.69); scNT-seq haematopoiesis
labelling vs dynamo -0.05, labelling vs scVelo 0.03,
scVelo vs dynamo 0.15 (n = 1947). Right, LARRY cell-type
median λ₁ of the OT and lineage flow J (PCA-30); Undifferentiated > Neutrophil and Monocyte for both. Per-cell OT vs lineage λ₁ Spearman
0.27 (null level); OT flow J across fit seeds 1–2 and dynamo SparseVFC 0.92–0.95.
**c**, S20 (pre-registered gates G1, G2 passed): wall time (OT solid, drift fit + Jacobians dashed) and OT-stage peak memory vs total cells
(2,000–16,000 per day and all 130,887 cells) for dense, mini-batch (4,000-cell batches) and low-rank (moscot rank 50/200) OT;
dense OT on the full data was not run (cost matrix ≈ 21 GB). Full LARRY with mini-batch OT: OT 291 s (2.54 GB),
drift fit + Jacobians 1278 s (1.31 GB). Right, agreement with dense OT: mini-batch median velocity cosine
0.9998 and λ₁ Spearman 0.9999 at 8,000 cells/day (the 4,000/day gate row is trivial: one batch = whole day);
low-rank cosine 0.79–0.80, |λ₁ Spearman| ≤ 0.07.
**d**, Cell-type λ₁ on all cells (cell-count-weighted mean of per-day medians, 11 cell types) vs the 4,000-per-day S11 result: Spearman 0.82.
