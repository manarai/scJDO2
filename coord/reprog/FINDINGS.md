# Findings — reprog

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

## 2026-11-08 — reprog/C01 (planned, ACCEPT)
**Result.** I1 PASS (CellTag 85,010 cells, Schiebinger 236,285, day on every cell, raw counts); I2 PASS (661 CellTag clones on ≥ 2 days). Only 8 clones have published outcomes. Re-derived from the committed tables.
**Meaning.** A held-out clone-fate test must define fate from each clone's late-day cells, not from published outcomes; clones are small (median 5 cells) and author states cover 38% of CellTag cells. The iEP transgene is a separate joint feature (FoxA1.HNF4a); endogenous Hnf4a/Foxa1 stay ~0, so a "known factor is early" check is not meaningful for iEP. Schiebinger OKSM transgene and endogenous reads cannot be separated.
**Caveats.** Annotation coverage; Schiebinger cell sets partly OT-derived; DiPSC controls excluded.
**Next.** C02 pre-registration (Claude) with clone fate from late-day cells and no transgene known-factor gate.
