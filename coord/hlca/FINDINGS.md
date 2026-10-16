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

## 2026-10-15 — hlca/H05 (planned, ACCEPT) — human arm complete

**Result.** Figures, executed notebook and RESULT.md complete; the
pre-registered mouse–human comparison: fibroblasts mouse ×1.01 (seed
range 1.01–1.18) vs human ×1.07; NK mouse ×1.00 vs human ×0.98; all
isotropy and anticipation calls fail on both sides.
**Meaning.** Mouse and human agree on the two matched types: no aging
change in operator shape or stiffness beyond a few percent.
**Next.** HX01, HX02 proposed (exploratory), awaiting the user.

## 2026-10-15 — hlca/HX02 (EXPLORATORY, ACCEPT) — power

**Result.** Minimum detectable effects at 80 % power, ages 10–76:
uniform softening median ×1.5 (all 13 types detectable); drift along the
young soft axis median 2 σ (all 13); single-axis softening > ×8 in 12 of
13 types (AT2 ×3).
**Meaning.** The human null is informative for **uniform stiffness loss
(≥ 1.5-fold excluded in most types)** and moderately for drift, but
**not for single-axis loosening**: the log-eigenvalue isotropy statistic
loses power when the soft eigenvalue is near zero or positive, which it
is in most human types. The headline must say so.
**Caveats.** Baseline rows contain H04's borderline real signals (T cell
0.96), so those types' MDEs are optimistic.
**Next.** Proposed: a fixed-axis curvature statistic, validated on the
HX02 injections (exploratory); HX01.

## 2026-10-15 — fixed-axis curvature statistic designed (Claude, exploratory)

**Result.** `softaxis_curvature_trend`: synthetic false pass 0.06 under
null, uniform and drift; power 1.00 single-axis. On HX02's real-data
injections (6 types) 80 % power at ×1.25 (AT2), ×2 (Fibroblasts),
×3 (Macrophages), ×5 (NK, Multiciliated); Basal (unstable soft axis)
none; 0 passes under uniform ×2.
**Meaning.** A usable test of single-direction loosening for human lung.
Designed after H04, so anything it finds is exploratory.
**Next.** HX03: all 13 types, then the real data (reading rule fixed).

## 2026-10-15 — hlca/HX03 (EXPLORATORY, ACCEPT) — fixed-axis statistic on real data

**Result.** No type shows soft-axis change with age (primary 0/13,
within-study 0/13, all-ages space 1/13 at chance level; 0 robust).
Power: median MDE ×3; 9/13 types testable by ×8 (AT2 ×1.25; EC
arterial, Fibroblasts ×2). Reproduced independently to 1e-16.
**Meaning.** The project's original hypothesis — that aging loosens a
cell type's basin along one specific direction — is not supported in
human lung, now with real power: a ≥ 2–3-fold flattening of the soft
axis across ages 10–76 is excluded in most types.

## Human arm — consolidated conclusion (frozen H04 + exploratory HX02/HX03)

In the HLCA core (106 donors, 13 types), aging does not detectably
change the operators of lung cell types:
* uniform stiffness loss ≥ ~1.5-fold is excluded in most types (HX02);
* single-direction loosening of the soft axis ≥ ~2–3-fold is excluded
  in 9 of 13 types (HX03; exploratory statistic);
* drift along the young soft axis ≥ ~2 σ is excluded (HX02);
* the two isolated hits (T cell stiffness, EC capillary anticipation)
  do not survive sensitivity checks.
Limits: 17 donors ≥ 60 clustered in two studies; one scVI seed; four
types untestable for single-direction change; the single-direction
statistic was designed after the frozen result.

## 2026-10-15 — hlca/HX01 (EXPLORATORY, ACCEPT) — T cell and monocyte hints

**Result.** Both study-driven, neither composition-explained. Study
fixed effects cut the T cell slope from −0.0056 to −0.0019 per year and
the monocyte slope from −0.0029 to −0.0014; CD8 fraction is unrelated to
age (p = 0.83). Recomputed independently.
**Meaning.** The two softening hints from H04 reflect differences
between studies, not aging within them. The consolidated human
conclusion is unchanged.

## 2026-10-15 — paused

Paused by the user pending scOpAlign; the underlying goal is an
operator-based reference atlas. Reusable for it: HLCA donor-level
operators and drift fits (`benchmarks/aging_hlca/`), the fixed-axis
statistic, and the lesson that per-batch-adaptive encoders can absorb
the differences an operator reference must keep. Open threats to the
aging null (batch absorption, density-shape interpretation) are listed
in the write-up's limits and are untested.
