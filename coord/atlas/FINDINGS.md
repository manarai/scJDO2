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
