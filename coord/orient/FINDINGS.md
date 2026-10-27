# Findings — orient

Claude's running interpretation, one dated entry per accepted (or
stopped) task, newest last.

## 2026-10-16 — layer built (Claude), before this workstream

Orientation layer on analytic-J synthetic data: lineage + labelling sign
agreement 0.978–0.981, labelling 1.000, omega median relative error ≤ 0.3,
conflict/coverage/flags pass; pseudotime 0.69–0.71 vs gate 0.90 (xfail,
decision pending). Sources were given latent velocities directly; this
workstream replaces that with count-level readouts.

## 2026-10-16 — orient/S01 (planned, ACCEPT, revision 2) — generator

**Result.** Multimodal generator with lineage (60 % barcode detection,
77 % multi-day clones), labelling (new/total = ε × truth), spliced/
unspliced with 238/800 genes carrying a spliced lag, four sampling days,
exact truth in PCA/FA. 12 datasets, ≈ 3 400 cells each (default).
**Caveats for S02.** Truth should be evaluated at expected counts (3–7 %
sign ambiguity otherwise); `stationary_cycle` needs regenerating larger;
dynamics mostly in days 1–2.
**Next.** The user runs scVelo and Palantir; then S02 (end-to-end tests).

## 2026-10-16 — orient/S02 (planned, ACCEPT; fixed gates failed)

**Result.** Default, PCA: lineage 0.774, labelling 0.841 (raw 0.896),
combined 0.803 vs gates 0.90/0.85/0.90; coverage and flags pass. Truth
leak-free; implementation correct.
**Meaning.** The orientation layer merges two different quantities.
Lineage tracks developmental direction (0.98–0.99 against a chord truth);
labelling tracks instantaneous flow. Ranked into one sign, lineage
overrides the better flow source where they diverge (truth sides with
labelling in 70–81 % of conflicts). Recommended: separate flow and
developmental signs.
**Caveats.** All sources weaken in slow, settled cells; smoothing lowers
sign accuracy; PCA > FA.

## 2026-10-17 — orient/S03 (planned, ACCEPT; 3 of 7 gates failed)

**Result.** Default, PCA, raw labelling: flow — labelling 0.896, scVelo
velocity 0.222, both 0.665 (coverage 0.63 → 0.83); developmental —
lineage 0.981, Palantir pseudotime 0.913 (0.973 moving axes), both 0.926
(coverage 0.83). Independence exact; J unchanged.
**Meaning.** (Corrected 2026-10-17: the failure is due to the generator — 2 % unspliced, no switching genes — see the S03 review's correction.) scVelo fails on this system (gene-level sign at
chance, embedding reversed after day 1); labelling is the reliable flow
source, and velocity as a filler lowers flow accuracy. Palantir pseudotime
is good on realistic data with a good root, unlike on the analytic
generator. Both orientation outputs are informative; the flow output needs
a guard against an unreliable velocity source.
**Caveats.** Palantir's root was truth-chosen (best case). One generator.

## 2026-10-17 — Synthetic data audit (Claude; correction)

**Result.** The S01 data is computed correctly, but three readouts are
unrealistic: unspliced is 2 % of counts (real ~15–30 %) and no gene switches
or saturates, so splicing-based rate estimation reverses direction even
though the direction is recoverable with the true rates; counts are pure
Poisson (real data overdispersed); 77 % of detected clones span ≥ 2 days
because 20 % of all cells are sampled per day (real lineage experiments
sample far less). Labelling (new/total 0.28, ε 0.9) is plausible to
optimistic; embedding and pseudotime are sound.
**Meaning.** scVelo's S03 failure is a property of the generator; lineage
results (0.98) are probably inflated; labelling and pseudotime results
likely hold. The generator is fixed before any further result (orient/S05).

## 2026-10-17 — orient/S05 (planned, ACCEPT): generator v2

**Result.** v2 presets: unspliced 15–18 % of counts (v1 2 %); 20 % of
detected clones span ≥ 2 days (v1 77 %), median clone 1 cell; Gamma-Poisson
noise; 30 % switching genes. Signal is recoverable before any method: true
splicing rates on kNN moments 0.47–0.64, true labelling γ 0.31–0.50
(default, 3 seeds). Phase-portrait rate fitting: +0.52 on switching genes,
−0.20 on ramping genes, ≈ 0 overall; scVelo −0.03.
**Meaning.** The data is now fair to every readout. Splicing velocity
works where its switching model holds and fails where transcription ramps;
the reliability check (§15) is the guard. Lineage will cover far fewer
cells than in S02–S04.
**Caveats.** One gene mix (30 % switching) — a choice, not a measured
property of real tissue. FA coordinates depend on the scikit-learn version.

## 2026-10-17 — Palantir start cell (Claude, from S04's unreviewed output)

**Result.** Five random day-1 start cells gave pseudotimes with Spearman
0.9999–1.0 against the truth-chosen one (42 of 45 runs). Palantir's
`run_palantir` replaces `early_cell` by the nearest diffusion-map
boundary cell unless `use_early_cell_as_start=True`.
**Meaning.** S03's "oracle root" caveat was largely moot; the start cell
our check read (`uns["palantir_start_cell"]`) was the given cell, not the
one Palantir used. S06 records the actual start and tests strict mode.

## 2026-10-18 — orient/S06 (planned, ACCEPT; 4 of 9 gates failed): full pipeline on v2

**Result.** v2_default PCA: flow labelling 0.82–0.87 (velocity demoted;
scVelo alone 0.19–0.53); dev pseudotime (realistic roots) 0.96, lineage
0.72–0.74 on 18 % of cells, both 0.91–0.93 on 82–86 %. Score-matching J:
AUC 0.18 for saddle vs committed (one soft positive axis in nearly every
cell, also on v1 and stationary data); flow J from labelling (one dataset)
AUC 0.996. Real-time guard accepted 104/180 reversed-root runs.
**Meaning.** (1) Under realistic sparse lineage sampling, pseudotime is the
better developmental source; the ranking lineage > pseudotime needs a
data-quality condition. (2) The score-matching J encodes density geometry,
not the dynamics, on a developmental time series; dynamical forks and basins
need a flow-based operator. (3) The pseudotime guard must check the start
cell's sampling day, not only the correlation with day.
**Caveats.** Flow-J result on one dataset; the start-day rule was derived
from these runs. One generator.

## 2026-10-18 — orient/S08 (planned, ACCEPT): real-data reference statistics

**Result.** Against six real datasets (scNT-seq haematopoiesis and neurons,
LARRY, Setty CD34, pancreas, dentate gyrus), generators v1 and v2 are
unrealistic in expression level and sparsity (HVG mean 0.69 vs 0.03–0.16;
zeros 37 % vs 84–96 %; 800 genes), dimensionality (participation ratio 2.6
vs 3.9–8.4), noise (φ 1.5 vs 2.1–3.6), gene programmes (no steady genes vs
12–46 %), scVelo fit ease (97.5 % genes recovered vs 17–63 %) and turnover
spread. Lineage: LARRY has median clone size 4 and 66 % of clones on ≥ 2
time points — generator **v1** matches, v2's sparse sampling does not.
**Meaning.** Every orientation result so far was obtained on data easier
than real data in counts and dimensionality; the S06 lineage weakness and
the §17 rule rest on unrealistically sparse lineage. The generator is to be
calibrated to these statistics (S09) before any further method result.
**Caveats.** Real datasets differ among themselves (unspliced share 8–50 %);
LARRY is the only lineage reference and scNT-seq haematopoiesis the only raw
labelling one.

## 2026-10-18 — Critical-transition pilot (Claude; generator v2, feasibility)

**Result.** On v2_default seeds 0–2, the flow operator (Jacobian of the drift
fitted on labelling velocity) peaks exactly at the true pitchfork transition
(μ ≈ 0.12) on every seed, and its curvature along the fate axis rises toward
it (Spearman 0.58–0.73). The score-matching J peaks late (μ ≈ 0.9, committed
cells); the early-warning variance peaks just after the transition (the
split); the critical-state index peaks late.
**Meaning.** Critical-transition detection is feasible with the flow
operator, not with the score operator — consistent with S06. Worth a
formal test with gates on generator v3, with null presets and real time
courses (LARRY, scNT-seq) and perturbations (K562 Perturb-seq).
**Caveats.** Pilot on unrealistic data; fate axis from the truth; few
pre-transition cells. `coord/orient/reviews/ct_pilot_2026-10-18/`.

**Addendum (same day).** A flow operator from the **time course alone**
(entropic optimal transport between consecutive days → velocity →
MeasuredDrift → Jacobian) peaks one bin before the true transition on all
three seeds (μ −0.09 to −0.11 vs 0.12–0.13) and rises toward it (Spearman
0.26–0.40), with no labelling. This would make transition detection
applicable to any time course (e.g. LARRY). The user ranks critical
transitions above the orientation evaluation (2026-10-18).

## 2026-10-19 — orient/S11 (planned, ACCEPT; 6 of 14 criteria): flow J on real data

**Result.** LARRY: OT and lineage velocities agree (median cos 0.70) but
per-cell flow Jacobians agree only at null level (λ₁ Spearman 0.27); the OT
operator is seed- and estimator-stable (0.93–0.95, incl. dynamo SparseVFC);
both flows rank undifferentiated cells above neutrophils/monocytes.
scNT-seq: labelling, scVelo and dynamo velocities disagree before any
operator (ours vs dynamo's labelling −0.05).
**Meaning.** Flow J is not yet reproducible per cell on real data. Two
separate weak points: velocity definition on labelling data (scNT-seq), and
the velocity-to-Jacobian step (LARRY). Coarse (cell-type) flow structure is
consistent across independent sources.
**Caveats.** One labelling and one lineage dataset; null partly reflects
cell geometry; dynamo's own fit was below its quality threshold.

## 2026-10-19 — Labelling velocity on real data depends on the γ assumption (Claude)

**Result.** On scNT-seq haematopoiesis, dynamo's labelling velocity is our
formula on kNN-smoothed counts with one γ per labelling time; ours uses raw
counts and one γ per gene. Using dynamo's γ raises agreement in the
embedding from ≈ 0 to 0.41 (dynamo's own reconstruction 0.56); smoothing
alone does not. The velocity is a small difference of large terms (0.37).
**Meaning.** On real labelling data the velocity direction is not robustly
identified: it depends on whether γ is shared across pulses (batches).
Labelling-based flow J must be shown robust to this choice; our labelling
velocity should be computed on smoothed counts for real data. LARRY's OT and
lineage velocities do not depend on γ and agree (0.70).
**Caveats.** One dataset; pulses confounded with batches.
`coord/orient/reviews/labelling_diag_2026-10-19/`.

## 2026-10-20 — orient/S09 (planned, executed; v3 fails the pre-fixed stopping rule)

**Result.** Calibrated generator v3 (local agent, 257 vectors, 12 datasets):
expression level, zeros, noise, barcode rate, clone size and labelling
signal match real data; unspliced share 0.40–0.52 (band 0.25–0.35, real
0.08–0.50), splicing oracle 0.21–0.28 (< 0.30), dimensionality slightly
high, multi-time clones 0.44–0.49 (≥ 0.53), steady genes ≈ 1 %, scVelo
still too easy. Match rate with real data 0.39 (v2 0.22, v1 0.35).
**Meaning.** v3 is more realistic than v2 in splicing, sparsity and lineage,
but not uniformly better than v1; by the rule fixed in advance it does not
pass, and the choice of how to proceed is the user's.

**User decision (2026-10-20):** accept v3 with documented misses; S10 proceeds on it with the caveats stated next to every splicing- or lineage-based number.

## 2026-10-20 — orient/S10 (planned, executed; 5 of 6 gates fail, 3 by design)

**Result.** On calibrated v3: the flow operator from the **time course alone
(OT)** separates true saddle from committed cells with AUC 0.97–1.00 (PCA)
and aligns with the fate axis (0.97–0.99) — but inverts on FA (0.04–0.25).
Labelling flow J 0.53–0.89, scVelo flow J 0.27–0.51, score J 0.17–0.31.
Palantir default mode relocated every day-2 start to a late cell (guard
refused all). Count noise in the embedding exceeds 48-h diffusion, so
diffusion estimates fail. The transition test was untestable (fork in the
first pseudotime bins). Real LARRY: OT flow J peaks in undifferentiated
cells, stable across fit seeds.
**Meaning.** Time-course flow J is the most promising operator — label-free
and fork-sensitive — pending the FA dependence; diffusion is not
identifiable from these data; the transition test needs day-balanced bins.

## 2026-10-20 — orient/S12 (stopped at Step 0): OT flow J is embedding-dependent

**Result.** Fork AUC of OT flow J holds in PCA-10 (≥ 0.875) and nearly in
PCA-20, but not in PCA-50 (0.56–0.97), FA-20 (0.04–0.53, inverted) or
scVI-10 (0.31–0.96). OT velocities agree across PCA and FA in gene space
(0.52–0.56); FA compresses the fate axis.
**Meaning.** The S10 headline is specific to PCA at 10–20 dims: the
velocity→Jacobian pipeline is not invariant to the embedding's metric
(Euclidean OT cost, fixed drift regularisation). This is the
"representation" problem, now concrete.

## 2026-10-20 — orient/S13 (gate FAIL): whitening rejected; the OT metric carries the signal

**Result.** Whitening each embedding destroyed fork detection (PCA-20
0.82–0.97 → 0.37–0.58); whitening only the drift fit kept it (PCA-10
0.80–0.96); whitening only the OT cost destroyed it.
**Meaning.** OT flow J needs a variance-weighted transport metric (PCA of
log-normalised data, ~10–20 PCs); equalising directions (whitening, FA) lets
noise dominate. The embedding dependence is a choice of metric to make on
principle, not to remove.

## 2026-10-22 — orient/S18 (Manus cloud; P1 FAIL): score-J soft axes do not predict K562 knockdown shifts

**Result.** Median fraction of each knockdown's shift captured by 3 directions
(PCA-20): local score J 0.045, global covariance 0.589 (chance 0.15). In an
independent check, score J reports strong curvature along the top PCs where
the controls' variance is largest.
**Meaning.** K562 controls are several locally stiff states; knockdowns
mostly redistribute cells among them (composition), so a within-basin linear
response test does not apply. Score J's local curvature is consistent; the
stability pillar's real-data test should be state occupancy, not linear
response.

## 2026-10-22 — orient/S18b (Manus cloud; both gates fail): score-J basins do not predict K562 occupancy change

**Result.** Re-weighting control states explains a median 31 % of
knockdown shifts (27 % of targets above noise); basin stiffness vs
occupancy sensitivity partial ρ +0.05 (p 0.65); Leiden spread equally null.
**Meaning.** Corrects S18's reading: knockdowns are not mainly
redistribution. Score J's stability descriptors predict neither linear
response nor occupancy change in K562 — a proliferating, non-equilibrium
line outside score J's domain. Negative result; no further K562 variants.

## 2026-10-22 — EXPLORATORY (post hoc, not a gated result): strong transcription-machinery knockdowns move along score-J soft valleys

**Result (Claude, `reviews/s18_strong_knockdowns_exploratory.py`, 60 strongest
S18 targets, shift vs all controls).** Local score-J soft-axis capture:
transcription machinery (Mediator, SUPT5H/6H, INTS2, GATA1, PAF1) median 0.91
vs local / global covariance 0.29 / 0.30; mitochondria / translation 0.22 vs
0.73 / 0.72. Operator taken from random controls instead of near the target:
0.06 — the effect is local to where cells end up.
**Meaning.** Not predictive from controls alone and found after two failed
pre-registered tests on the same data — hypothesis-generating only. Needs a
pre-registered predictive test on an independent screen (e.g. Replogle RPE1).

## 2026-10-22 — orient/S19 (ACCEPT): dyngen independent datasets ready

Nine dyngen datasets with separated truth (bifurcation, cycle, linear null;
three time points). S08 realism match 0.51 (our v3 0.39). Ready for the
Part-2 replication of the headline results after S14/S15.

## 2026-10-22 — orient/S14 (all gates fail): untuned flow J robust; tuning overfits; values not recovered

**Result.** Held-out fork AUC of the a-priori reference flow J: 0.89–0.97
(v3_transition test seeds, k 10–20) and 0.97–1.00 (other presets); the tuned
configuration 0.42–0.95. Relative error to the reference Jacobian ≈ 1.0
everywhere. Transition detection not specific (null peaks as high). LARRY:
lineage agreement 0.55 (null 0.14); held-out-day prediction below baselines.
**Meaning.** Label-free flow J reliably ranks fork vs committed cells on
unseen data without tuning, but does not recover the Jacobian's values;
critical-transition detection is dropped as a claim.

## 2026-10-22 — orient/S20: minibatch OT scales flow J to full LARRY

Minibatch POT matches dense OT (8 000/day: velocity cosine 0.9998, lambda1 rho 0.9999;
the 4 000/day gate row is trivial because one batch = whole day). Full LARRY (131 k cells)
in ~26 min and < 3 GB; cell-type lambda1 agrees with the 4 000/day result (rho 0.82).
Low-rank OT (moscot) fails (cosine 0.79, lambda1 rho ~0).
**Correction (same day).** On LARRY the reference configuration *does* beat
both held-out-day baselines (energy 0.687 vs 0.867 / 0.882; lineage ρ 0.56,
null 0.15); only the tuned configuration fails.

## 2026-10-23 — orient/S21 (dyngen): sampling design skips the fork; no rotation from 3 snapshots
dyngen 3-time-point sampling put no cells at the branch point (planner spec
error); F1 not evaluable. Rotation not detected on dyngen cycles (F2 fail).
Pre-branch cells show Re λ₁ > 0 (100 %) vs 0 % on linear backbones (report only).

## 2026-10-23 — orient/S18d (bundle imported 2026-10-26): underpowered; S18 series closed
RPE1 confirmation: 1 qualifying target (< 10) → underpowered by the fixed
rule; no gate claimed. K562 (report only): weak-knockdown T > M one-sided
p 0.002 (N1) / 0.012 (N2), but D1 (above shared-basis null) and D2 (score J
> covariance) fail in both normalisations. Conclusion: score J does not
predict knockdown responses beyond covariance; Perturb-seq is anchor-op's
question. Series S18–S18d closed.

## 2026-10-24 — orient/S21b (dyngen, 8 time points): flow J works on 2/3 forks and 2/3 cycles
Fork AUC 1.00 / 0.92–0.99 / **0.31–0.38 (inverted)**; score J 0.98 / 0.58 / 0.85.
Rotation detected on cycle s1, s2, not s0. Linear: no positive λ₁ anywhere.
S21's "pre-branch λ₁ > 0" reading withdrawn (sampling artefact).

## 2026-10-24 — orient/S22 (bundle imported 2026-10-26): score-J stiffness does not predict sister-cell fate coherence (LARRY)
L1 pooled ρ(λ₁_clone, fate entropy) −0.003 (p = 1); L2 Undifferentiated
ρ −0.10 (p = 1, wrong sign); L3 distance to the progenitor centroid predicts
better (|ρ| 0.22) than λ₁ (0.10); partial ρ given density −0.08. All gates fail.
With S18–S18d, score J has no positive result against an independent
biological truth so far.

## 2026-10-25 — orient/S14b: test could not be built (2 of 4 eligible transitions)
Only seeds 12 and 16 (of 9 assessed) had a stable interior true peak; v3's
critical peak is usually flat on the 12-bin strict-pseudotime scale. Three
seeds hit generator defects (`n < 0`; two runaways). No model was fitted:
transition detection remains unconfirmed.

## 2026-10-25 — orient/S15 (interim; gates final, stability grid running): flow archetype isolates the saddle; not better than Leiden overall; seed-unstable
A1 PASS (operator-archetype view): saddle recall 0.72–0.79, precision 3.4–4.1× base.
A2 FAIL: NMI flow 0.35–0.40 vs Leiden 0.35–0.37 vs score 0.05–0.31.
A3 FAIL: drift-seed ARI 0.22–0.44. J_ref verified by an independent
finite-difference route (cosine 1.000): flow J genuinely ≠ the true Jacobian
(per-cell cosine ≈ 0, norm 2–7 %).

## 2026-10-26 — orient/S14c: construction failure again (2/4); the ≥ 50-cell bin rule excludes well-ordered datasets
No eligible seed in s17–s40 (14 guard stops; 10 assessed). Seeds with stable interior peaks (s8 0.955, s29 0.995, s36 1.000) fail only the occupancy rule (min bin 42–46, set by day 2 ≈ 180 cells). No model fitted; transition detection still unconfirmed.
