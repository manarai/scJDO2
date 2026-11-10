# Findings — atlas

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

## 2026-11-08 — atlas/F01 (planned, ACCEPT; result: not stable)
**Result.** LARRY, replicate A (well 1) vs B (well 2), 3 drift seeds each, K* = 12 (auto, at the candidate boundary). G0 pass; F1 median replicate ARI 0.514 < 0.70 FAIL; F2 min matched cosine (A0, B0) −0.44 < 0.80 FAIL; F3 ratio 0.801 ≥ 0.80 PASS. Frozen verdict: **not stable**. All gate numbers re-derived by Claude from the J files.
**Meaning.** The flow operator is reproducible (per-cell λ₁ ρ 0.97 across wells, same as across seeds); the K = 12 archetype labelling is not, mostly because of the decomposition itself (seed refits: ARI 0.59–0.65, worst matched shapes −0.42 to 0.53). Held-out error keeps falling to K = 12: J_flow shapes look continuous on LARRY, not a few discrete motifs. Large archetypes (A1, cosines 0.82–0.94) recur; small ones do not. Premise 2 is not established at K = 12; premise-3 work waits.
**Caveats.** One in-vitro dataset; "replicate" = well; one K rule frozen; Sinkhorn non-convergence warnings in mini-batch OT.
**Next.** Proposed: F02 (exploratory, agreement vs K on the existing J files, ~0.3 CPU-h); doc caveat on archetype reproducibility in flow-motifs. Awaiting user.

## 2026-11-08 — atlas/R01 (planned, withdrawn; STOP, closed)
**Result.** Ran after its withdrawal (2026-11-07); stopped at step 1 with 0 of 20 fits. No gate evaluated.
**Meaning.** None for the atlas question. By-product inventory: the dynamo hematopoiesis file has 1,956 genes and no raw counts, so it cannot enter count-based operator fits as is; Paul15 and Setty have counts.
**Caveats.** Inventory only; not re-run by Claude beyond reading its tables.
**Next.** Closed. Branch kept, not merged.

## 2026-11-10 — atlas/F02 (exploratory, ACCEPT; reading: K = 2 only)
**Result.** F01's gate applied at K ∈ {2, 3, 4, 5, 6, 8} on the six saved F01 fits: only K = 2 reproduces (replicate ARI 0.82, min matched cosine 0.98, F3 1.08; both archetypes stable). K = 3: ARI 0.65 fails; K ≥ 5 shapes fail too. All numbers re-derived by Claude; K = 2 and 3 labels recomputed from J identically.
**Meaning.** Flow J on LARRY has one reproducible split: an expanding, uncommitted operator (78 % of day-2 cells, high λ₁) vs a contracting, committing one (22 %, nearly all day-2 Monocyte/Neutrophil/Baso cells). λ₁ alone separates them (AUROC 0.90–0.91). Finer motifs do not reproduce; J_flow varies continuously. Premise-3 motif work is not built on flow-J archetypes; λ₁ is the reproducible readout.
**Caveats.** Exploratory (rule written after F01); one dataset; replicate = well; day-2 cells only; one decomposition method.
**Next.** Proposed: flow λ₁ layer in `scjdo2 simulate` (Claude, part of T00); routes/T02 replicate test of the gene-operator archetypes (exploratory, ~4 CPU-h); flow-motifs doc caveat. Awaiting user.
