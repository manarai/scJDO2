# Findings — hlca

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

## 2026-10-15 — hlca/H01 (planned, ACCEPT) — inventory

**Result.** HLCA core: 584 944 cells, 107 donors (106 aged 10–76);
<45: 48 donors, ≥60: 17. scArches surgery on the HLCA reference works
with the reference frozen exactly.
**Meaning.** Enough donors for a donor-level test, unlike mouse. But
old donors cluster in two studies, and study, assay (3' v3) and
airway sampling lean old; one study (Banovich_Kropski_2020, 17 vs 5)
supports a within-study contrast.
**Caveats.** Metadata only; no fit, no statistic.
**Next.** H02: human pre-registration (Claude), for user approval.

## 2026-10-15 — hlca/H02 (planned, ACCEPT) — human pre-registration frozen

**Result.** `PREREG_aging_hlca.md` approved and frozen before any human
fit: continuous age, donor-level operators at equal counts, age
permutations across donors, young-only scArches reference primary,
stiffness qualifier on the header. 13 types expected eligible.
**Next.** H03 (preprocessing, shared spaces, drift).

## 2026-10-15 — hlca/H03 (planned, ACCEPT, revision 2)

**Result.** Young-only HLCA reference (266 145 cells, 400 epochs) with
315 372 older-donor cells mapped by scArches; latent encoder frozen
exactly; drift on 56 904 subsample cells in both spaces, symmetric to
~1e-7. Eligibility 13 / 12 / 6 / 2 as predicted.
**Meaning.** Ready for H04. Older donors' cells fit the young model less
well (ELBO 655 vs 550).
**Caveats.** scVI weights not saved; frozen check rests on the audit
code. Banovich-only sensitivity has 2 types.
**Next.** H04: frozen tests §5–§7.

## 2026-10-15 — hlca/H04 (planned, ACCEPT) — the frozen human result

**Result.** **isotropic noise increase; no aging claim beyond stiffness
— no majority stiffness change.** Isotropy 0/13, anticipation 1/13
(EC capillary, p = 0.047), stiffness loss 1/13 (T cell lineage,
p = 0.005). Reproduced independently to 1e-16.
**Meaning.** No direction-specific or anticipatory aging change in the
operators of any human lung cell type; the two single hits are fragile
(T-cell softening disappears within studies and in the all-ages space;
EC capillary is borderline everywhere). Mouse and human agree: no
robust aging change in operator shape.
**Caveats.** 17 old donors clustered in two studies; young-reference
fit worse for older donors (all-ages space agrees); one scVI seed;
modest effects not excluded.
**Next.** H05 issued. Proposed exploratory: HX01 (T cell / monocyte
follow-up), HX02 (power).
