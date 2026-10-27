# Label-free flow operators and stability operators from single-cell time courses: what local dynamical operators can and cannot identify

David Redd, Sam Green, Tommy W. Terooatea*

Genomics & Bioinformatics Center, Department of Biology, Brigham Young University, Provo, UT, USA
*Correspondence: tommy.terooatea@byu.edu

> **DRAFT v57 (2026-10-25)** — new version of the scJDO preprint (v56). Sections marked **[PENDING]** wait for running tests (S15 archetypes, S14c transitions, S16 head-to-head); sections marked **[v56 →]** are carried over from v56 with the edits stated. Every number is from a pre-registered, reviewed task in `coord/orient/` unless labelled *post hoc* or *report only*.

---

## Changes from v56

1. **Two operators, matched to the time information measured.** v56 used one drift field learned by diffusion score matching (DSM) on pseudotime. v57 (scJDO2) distinguishes the **score operator** *J_score = β∇² log p* (snapshot; describes the curvature of the cell-state density) from the **flow operator** *J_flow = ∂f/∂z* of a drift *f* fitted to a **measured** velocity (time-course optimal transport, lineage, metabolic labelling or RNA velocity). Both are placed in one operator space with archetypes.
2. **New: label-free flow operator from a time course alone** (entropic OT between sampling days → barycentric velocity → drift → Jacobian), validated against lineage on real data and scaled to full datasets.
3. **Correction:** v56 recommended Factor Analysis (FA) as the default representation. This **does not hold for flow operators**: OT flow J inverts on FA (fork AUC 0.04–0.53) and needs a variance-weighted transport space (PCA of log-normalised counts, 10–20 components).
4. **New: an identifiability analysis** (representation, clock, rotation, stationarity, diffusion, composition) and an explicit list of what was tested and not supported.
5. **New benchmarks:** a generator calibrated to six real datasets, an independent simulator (dyngen), held-out seeds, and pre-registered gates throughout.

---

## Abstract

Local Jacobians of single-cell dynamics promise to show where along a trajectory cells are unstable, but what such operators identify from measured data has rarely been tested against independent truth. We present scJDO2, a framework that separates two operators by the time information they require: a **score operator**, the Hessian of the log density of cell states, available from any snapshot; and a **flow operator**, the Jacobian of a drift fitted to a measured velocity, which we estimate label-free from a sampled time course by entropic optimal transport. Both are mapped into a shared, basis-invariant operator space.

On synthetic data calibrated to six real datasets, the flow operator from the time course alone separates cells at a fate bifurcation from committed cells on held-out seeds (AUC 0.89–1.00), with settings fixed in advance; tuning on development seeds overfits. On LARRY lineage-traced haematopoiesis, it agrees with clonal displacement (Spearman 0.56 vs a permutation maximum of 0.15) and predicts a held-out sampling day better than mean-displacement and no-movement baselines (energy distance 0.69 vs 0.87 and 0.88). Mini-batch transport reproduces dense transport (velocity cosine 0.9998) and scales to all 130,887 cells in under 30 min and 3 GB. On an independent simulator it detects two of three bifurcations and two of three cycles, and never reports instability on linear trajectories.

We also report what is not supported. Neither operator recovers the values of the true Jacobian; the flow operator's leading eigenvalue is an empirical marker of branching, not a measurement. The score operator's curvature does not predict CRISPRi knockdown responses or the fate coherence of sister cells. The flow operator depends on the transport metric, inverting on Factor Analysis. Diffusion is not identifiable at typical sampling intervals. We release the software with these diagnostics built in.

---

## Introduction

[Revise v56's introduction; keep its first paragraph (motivation: where along a trajectory is the system sensitive) and its review of Dynamo [3] / SpliceJAC [4] / CellRank. Replace the framing paragraphs with the three points below.]

*Draft points.*
1. A Jacobian is only as meaningful as the velocity it differentiates. Snapshot methods differentiate a density-derived field (a score); kinetic methods differentiate a splicing- or labelling-derived velocity. These are **different operators** with different invariances: the score Hessian is symmetric and only its eigenvalue signs are coordinate-invariant (Sylvester's law of inertia), while a flow Jacobian transforms by similarity and its eigenvalues are invariant.
2. Many datasets have **sampling times** but no reliable kinetics. Optimal transport between days already provides cell-level velocities (Waddington-OT, moscot); we ask whether the Jacobian of a drift fitted to them carries usable local information.
3. Claims about local operators need **independent truth**: held-out seeds, an independent simulator, lineage barcodes. We pre-registered every test reported here and report failures alongside successes.

---

## Results

### 1. Two operators, one operator space

[Figure 1.]

*Score operator.* A neural energy *E(z)* is fitted by score matching on cells in a low-dimensional embedding *z*; *J_score(z) = −β∇²E(z)* is symmetric. It describes the local curvature of the cell-state density, which reflects stability only for a system near steady state.

*Flow operator.* A velocity *v(z)* is measured (OT between sampling days, lineage displacement, metabolic labelling or RNA velocity). A smooth drift *f* is fitted to *v* (`MeasuredDrift`), and *J_flow(z) = ∂f/∂z* is computed by automatic differentiation at every cell.

*Operator space.* Per cell, basis-invariant descriptors of each operator (spectrum, sign of Re λ₁, rotational index max|Im λ|/max|λ|, non-normality, transient gain), standardised and rank-transformed, define an operator space; archetypes are its recurrent patterns. Score and flow operators transform differently under a change of embedding coordinates *A* (score: *A⁻ᵀHA⁻¹*; flow: *AJA⁻¹*); only descriptors invariant under both are compared directly.

*Identifiability.* Six components limit what can be identified from data: representation (the embedding metric), clock (time units), rotation (antisymmetric flow invisible to a density), stationarity (the score describes stability only at steady state), diffusion, and composition (proliferation and death). Each is tested below.

### 2. A benchmark calibrated to real data

We measured 20 statistics on six real datasets (scNT-seq haematopoiesis and neurons, LARRY, Setty CD34⁺, pancreas, dentate gyrus): expression level and sparsity, dimensionality, count noise, gene programmes, splicing-fit difficulty, turnover and lineage structure. Earlier generator versions were unrealistic (e.g. zeros 37 % vs 84–96 % in real data). The calibrated generator (v3) matches expression level, zeros, noise, barcode rate, clone size and labelling signal; it misses on unspliced share, multi-time-point clones and scVelo fit difficulty (match rate 0.39 vs 0.22 for v2), and we accepted it with these misses documented. Presets include a pitchfork bifurcation with a controlled critical point, a no-fork null, batch effects, stress, a stationary cycle and time-course sampling (days 2/4/6) with lineage barcodes and labelling. [Figure 2.]

### 3. The flow operator from a time course alone ranks branching cells, untuned

*Label-free estimate.* Entropic OT (POT Sinkhorn, ε = 0.05 × median squared distance) between consecutive days in PCA-15 of log-normalised counts gives each cell a barycentric velocity; the drift is fitted to these velocities and differentiated.

*Held-out test.* On held-out seeds of the transition preset, the leading eigenvalue λ₁ of *J_flow* separates cells at the true saddle from committed cells with AUC 0.89–0.97 across k = 10–20 principal components, and 0.97–1.00 on the default, batch and stress presets. These are the a-priori default settings. A configuration tuned one factor at a time on two development seeds did worse on the same held-out seeds (0.42–0.95): **tuning overfits, and the defaults are robust.** Labelling-based flow J (0.53–0.89), scVelo-based flow J (0.27–0.51) and the score operator (0.16–0.31; the system is far from steady state) were weaker. [Figure 3a–c.]

*Representation.* The result depends on the transport metric. It holds in PCA-10 to PCA-20 and degrades in PCA-50 (0.56–0.97), scVI-10 (0.31–0.96) and Factor Analysis (0.04–0.53, inverted), although OT velocities agree between PCA and FA in gene space. Whitening the embedding destroys it (0.37–0.58); whitening only the drift fit preserves it (0.79–0.96); whitening only the OT cost destroys it. **The signal is carried by the variance-weighted OT metric**, which equal-variance representations (FA, whitening) remove. This corrects v56's recommendation of FA. [Figure 3c.]

*What the operator is not.* Against the Jacobian of the generator's true expected velocity in the same coordinates, the flow operator's per-cell relative error is ≈ 1.0. Its norm is 2–7 % of the reference's, the optimal rescaling does not reduce the error, the per-cell cosine is ≈ 0, and the per-cell λ₁ Spearman correlation is at most 0.11 in magnitude (−0.02 to −0.005 on the transition test seeds). **The flow operator does not recover the Jacobian's values;** its λ₁ ranks branching cells empirically. The reference was verified by an independent route (finite differences of the true expected velocity in the same coordinates; per-cell cosine with the reference 1.000), so the discrepancy is in the estimate, not the reference. All operator-space descriptors in this paper are therefore sign- or rank-based.

*Transition detection.* [PENDING S14c.] Post hoc, on three held-out transition and three no-fork seeds, the peak-to-median ratio of binned λ₁ along strict pseudotime separated all transitions (1.91–2.69) from all no-forks (0.58–1.44), with timing offsets of 0, 1 and 5 bins. A confirmatory test with pre-model eligibility and an independently calibrated no-fork threshold is running. The first attempt could not be built: the generator's true critical peak was resolvable in only 2 of 9 seeds.

### 4. An independent simulator

On dyngen [ref] (5,000 cells, 1,000 genes, eight synchronised sampling times; truth used for scoring only), with the same untuned settings:
- **Bifurcations:** fork AUC 1.00, 0.92–0.99 and 0.31–0.38 (inverted) on the three datasets, consistent across three drift-fit seeds. Score operator 0.98, 0.58 and 0.85.
- **Cycles:** rotational index higher on cycles than on linear trajectories for two of three cycles (0.19–0.40 vs 0.07–0.09); not for the third (0.02).
- **Linear trajectories:** no cell had Re λ₁ > 0, under any seed.

Three sampling times were too coarse: they sampled 22, 0 and 0 cells at the branch point, and OT could not resolve rotation. [Figure 4. Diagnosis of the inverted dataset: PENDING.]

### 5. Real data: lineage agreement, held-out prediction and scale

*LARRY* (in vitro haematopoiesis, days 2/4/6, 4,000 cells per day; barcodes used for scoring only):
- **Velocity:** OT and clonal-displacement velocities agree (median cosine 0.69).
- **Lineage:** the flow operator's λ₁ ranking agrees with lineage displacement (Spearman 0.56; permutation maximum 0.15).
- **Prediction:** integrating the fitted drift from day 4 predicts the held-out day-6 distribution better than mean-displacement or no-movement baselines (energy distance 0.69 vs 0.87 and 0.88).
- **Stability:** fits are seed- and estimator-stable (λ₁ Spearman 0.92–0.95, including dynamo's SparseVFC on the same velocity), and both flows rank undifferentiated cells above neutrophils and monocytes.
- **Per cell:** the flow Jacobians from OT and from lineage velocities agree only at null level (λ₁ Spearman 0.27). Coarse, cell-type-level structure is reproducible; per-cell operators are not. [Figure 5a–b.]

*Scale.* Mini-batch OT (4,000-cell batches, ten appearances per cell) reproduces dense OT at 8,000 cells per day (velocity cosine 0.9998; λ₁ Spearman 0.9999). It runs on all 130,887 LARRY cells in 291 s (2.5 GB), plus drift fit and Jacobians in 21 min (1.3 GB). Cell-type λ₁ on the full data agrees with the 4,000-per-day result (Spearman 0.82). Low-rank OT (moscot, rank 50/200) does not preserve the operator (cosine 0.79–0.80, λ₁ Spearman ≈ 0). [Figure 5c–d.]

*Labelling data.* On scNT-seq haematopoiesis, labelling, scVelo and dynamo velocities disagree before any operator is fitted. The labelling velocity depends on whether degradation rates are shared across pulses (agreement with dynamo rises from ≈ 0 to 0.41 with dynamo's convention; diagnostic in `coord/orient/reviews/labelling_diag_2026-10-19/`, to be committed as a result table). Labelling-based flow operators must be shown robust to this choice before interpretation. [Supplement.]

### 6. The score operator: a steady-state descriptor, not validated against perturbation or lineage

On steady-state data the score operator yields stable, interpretable stability archetypes ([v56 → PBMC basin and landscape figures]). As expected from its definition, it fails out of equilibrium: on the developmental benchmark it ranks committed cells above saddle cells (AUC 0.16–0.31). We tested it against two independent biological truths, both pre-registered:
- **CRISPRi knockdowns (Replogle K562 and RPE1).**
  - The score operator's soft axes capture a median 4.5 % of knockdown shifts, against 58.9 % for the control covariance.
  - Basin stiffness does not predict knockdown-induced occupancy change (partial ρ +0.05, p 0.65).
  - A dose-response confirmation on RPE1 was underpowered (1 qualifying target); in K562 (report only) score-J capture did not exceed a shared-basis null or the covariance baseline.
  - A post hoc observation (transcription-machinery knockdowns move along local soft valleys near their destination) is hypothesis-generating only.
- **Clonal fate coherence (LARRY).** Clones whose cells sit in stiffer regions do not keep their sisters in fewer fates. Pooled ρ(λ₁, fate entropy) is −0.003; within undifferentiated cells ρ is −0.10, wrong sign. Distance to the progenitor centroid predicts fate spread better (|ρ| 0.22).

**The score operator therefore describes density curvature, and we do not claim that it measures biological stability.** [Supplement: full tables.]

### 7. Archetypes of dynamics [S15 interim — gates final, report-only stability grid running]

Archetypes were fitted to unit-norm flow operators (semi-NMF, K by held-out reconstruction), with K-means on rank-transformed descriptors as a second view. They were compared with score-operator archetypes and with Leiden clustering of expression at matched K, on the three held-out transition seeds.
- **A flow archetype isolates the saddle.** One archetype captures saddle cells with recall 0.72–0.79 at 3.4–4.1× their base rate on every test seed (gate A1 passes, operator-archetype view; the descriptor view fails on one seed, 1.5×).
- **It does not beat expression clustering overall.** Against all true regimes, NMI is 0.35–0.39 for flow archetypes vs 0.35–0.37 for Leiden at matched K and 0.05–0.31 for score archetypes. The flow operator beats the score operator on every seed but beats Leiden on only one (gate A2 fails).
- **Archetypes are not stable across drift-fit seeds.** Assignment ARI is 0.22–0.44 (gate A3, ≥ 0.5, fails).

Reading: flow archetypes identify the branching regime specifically, but as a whole partition they carry about as much regime information as expression clusters, and they depend on the fit seed. We report archetypes as a descriptive layer, not a validated regime classifier.

### 8. Comparison with existing Jacobian methods [PENDING S16]

[CellRank, spliceJAC, dynamo SparseVFC on the common OT velocity; outputs exist (S16a), scoring pending.]

### 9. Results carried over from v56 [v56 →]

[Keep, with the edits noted:]
- *Dense reprogramming time course: branch-specific operator handoffs.* Keep; state that operators are DSM-drift (score-type) Jacobians on pseudotime and are **descriptive**; add that the S14/S15 tests do not validate them.
- *Hematopoiesis: branch-specific local sensitivity* and *partial concordance with Dynamo / difference from SpliceJAC.* Keep.
- *Synthetic benchmarks …motivate Factor Analysis.* **Revise:** FA remains acceptable for the score drift, not for flow operators (Section 3).
- *Constructed obstruction (identifiability)* and *bifurcation-saddle limit.* Keep; link to the identifiability components in Section 1.
- *K562 Schrödinger-bridge feasibility.* Keep as feasibility; cross-reference Section 6.

---

## Discussion

*Draft points.*
- The useful object is the flow operator from a **measured** velocity, and a time course alone is enough to obtain one. Its value is empirical and comparative (ranking branching cells, agreeing with lineage, predicting held-out days), not a recovered Jacobian.
- Representation is a modelling choice with consequences. The transport metric carries the signal, so equal-variance representations are unsuitable for OT flow operators.
- Density curvature is not stability out of equilibrium. Even near steady state, our tests on knockdowns and clones did not support reading it as biological stability.
- Negative results are part of the method's specification. We provide them as built-in diagnostics: the representation check, the transport-agreement check, the scale check and the null presets.
- Relation to the landscape–flux decomposition (Wang et al.): aligning the two operators through a fitted mobility *D* with *f ≈ D∇log p* separates gradient from flux. Testing this is future work.

### Not shown / tested and not supported
- Recovery of Jacobian values (flow or score).
- Critical-transition timing [PENDING S14c].
- Per-cell reproducibility of flow operators across velocity sources on real data.
- Diffusion estimation (count noise in the embedding exceeds 48-h diffusion).
- Score-operator prediction of knockdown responses or clonal fate.
- Fork detection on every independent-simulator bifurcation (2 of 3).
- Uniqueness relative to all prior methods (no systematic review yet).

---

## Methods

[Keep v56 Methods for the DSM drift, temporal tensor, archetypes, reprogramming, hematopoiesis and SB sections. Add:]

**Embedding for flow operators.** Library normalisation to the median, log1p, 2,000 highly variable genes (seurat_v3 on counts), PCA with k = 15 (k = 10–20 tested), raw (unwhitened) coordinates.

**Time-course velocity.** Entropic OT (POT `ot.sinkhorn`, ε = 0.05 × median squared distance, balanced) between consecutive sampling times; barycentric projection; velocity = (barycentre − position)/Δt; last time point by extrapolation. Mini-batch variant for large data: 4,000-cell source batches, every cell in exactly ten batches, targets sampled without replacement; barycentres averaged.

**Drift and Jacobian.** `MeasuredDrift`: MLP 96×96×96, weight decay 1e-4, single member, seed 0 (default; seeds 1–2 for variability); Jacobian by autodiff at every cell.

**Score operator.** Energy network (96 hidden units, depth 3), exact-Laplacian score matching, 1,500 epochs, lr 5e-4; symmetry check max asymmetry ≤ 1e-4; Hessian by autodiff.

**Descriptors and scoring.** λ₁ = eigenvalue with largest real part; fork AUC = λ₁ of saddle vs committed cells (truth regions defined from generator state); rotational index max|Im λ|/max|λ|; transition profile = per-bin median λ₁ over 12 day-weighted strict-pseudotime bins; peak-to-median ratio R.

**Pre-registration.** Every task fixed data, settings, gates and stopping rules before execution (`coord/orient/tasks/`), with reviews and amendments recorded before results (`coord/orient/reviews/`).

**Independent simulator.** dyngen 1.1.1, backbones bifurcating / cycle / linear, 5,000 cells, 1,000 genes, 64 simulations (bifurcating), synchronised experiment with 8 time points re-sampled from saved simulations.

**LARRY.** GEO GSE140802; 130,887 cells; fixed sample of 4,000 cells per time point (`default_rng(0)`) for dense analyses; barcodes for scoring only.

## Data Availability
[Update v56 list: add LARRY (GSE140802), Replogle 2022 K562/RPE1 (figshare), scNT-seq haematopoiesis, dyngen settings and manifests.]

## Code Availability
scJDO2: [GitHub URL, Zenodo DOI — to be created at release]. All analyses: `scripts/`, task specifications and reviews: `coord/orient/`.

## References
[v56 list + Waddington-OT (Schiebinger 2019), moscot (Klein 2025), POT (Flamary 2021), dyngen (Cannoodt 2021), LARRY (Weinreb 2020), Replogle 2022, Wang landscape–flux (Wang 2008 / 2015), CellRank 2 (Weiler 2024).]
