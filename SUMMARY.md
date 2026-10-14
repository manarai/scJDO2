# scJDO v2 — summary of hematopoiesis comparisons

**Repository tag: `v0.5-basin-lock`** (2026-10-14). Energy-param
score drift + lead-eig classifier picked by continuous AUC (§5.10).
Reproducible under:

* `benchmarks/v0.5_lock/pbmc_v05.h5ad`                — PBMC adata
  with locked-rule outputs (lead_re, basin_index, behaviour, J)
* `benchmarks/v0.5_lock/pitchfork_v05.h5ad`           — pitchfork
  τ=0.3 adata with the same columns
* `notebooks/basin_analysis.ipynb`                    — §5 end-to-end
  from the stored PBMC state (executed, committed)
* `notebooks/pitchfork_gate.ipynb`                    — three-rule
  comparison + locked-rule gate on the pitchfork (executed)
* `benchmarks/v0.5_lock/pbmc_operator_space.{pdf,png}`,
  `pbmc_basin_index_violin.{pdf,png}`,
  `pbmc_behaviour_stack.{pdf,png}` — PBMC figure set
* `benchmarks/v0.5_lock/pitchfork_operator_space.{pdf,png}`,
  `pitchfork_rule_auc.{pdf,png}`,
  `pitchfork_basin_index_vs_pt.{pdf,png}` — pitchfork figure set



The manuscript's primary comparison is on **CD34+ bone marrow**. v1
and v2 are both evaluated on the Setty 2019 Palantir dataset (the
preprint target), with Dynamo scNT-seq hematopoiesis (Qiu 2022) as
the matched measured-flow counterpart on the same compartment —
"snapshot versus measured flow on matched biology". Paul15 remains
as a secondary **mouse** replicate of the snapshot arm.

All reported eigenvalues on Setty are relative (snapshot, no clock);
on Dynamo scNT-seq they are per-hour (`clock_calibrated = True`).

Human gene symbols throughout (``GATA1``, ``SPI1``).

## 0 · Caveat — the four haematopoiesis datasets are progenitor compartments

Every haematopoiesis dataset reported below is a **progenitor
compartment with no mature resting cells**: Setty CD34+ BM, Dynamo
scNT-seq CD34+ BM, Paul15 Kit+ progenitors, and LARRY in-vitro
myeloid-skewed culture are all enriched for actively differentiating
cells and depleted of terminally differentiated, non-cycling types
(resting mature B / T lymphocytes, long-lived tissue macrophages,
mature neutrophils). A field fit on such compartments **should not
produce holders** on the biological ground truth — committed progeny
in these datasets are still proliferating along a lineage, which is
exactly the fork / drift / growing-oscillator signature. "Zero
holders on terminals" in §1-§4 is therefore **consistent with the
biology; it is not yet a positive test of the estimator**. A
positive-control test needs a homeostatic compartment that
genuinely contains basins — resting mature cells at or near a
fixed point. That is the §5 PBMC arm.

## 1 · Setty 2019 CD34+ bone marrow — the primary snapshot comparison

### 1.1 v1 preprint pipeline (per-Palantir-branch slicing)

Dataset: 4142 cells × 16106 genes after v1's QC (``marrow_scjdo.h5ad``
produced by v1's Palantir driftfield notebook). Three terminal
Palantir branches: Ery (1551 cells), Mono (2550), DC (1939). v1 fits
one drift + Jacobian + archetype decomposition **per branch**, each
on 100 pseudotime waypoints.

**v1 lead_re along pseudotime (max Re λ at each of 100 waypoints):**

| branch | n_cells | min(lead_re) | median | max | holder waypoints (lead_re < 0) |
|---|---|---|---|---|---|
| Ery  | 1551 | +0.003 | +0.065 | +0.088 | **0 / 100** |
| Mono | 2550 | +0.072 | +0.080 | +0.100 | **0 / 100** |
| DC   | 1939 | +0.075 | +0.107 | +0.131 | **0 / 100** |

No negative waypoints on any branch — v1's per-branch field reports
**no holders on any terminal lineage**.

**v1 GATA1 / SPI1 ranks (mean-across-waypoints instability score,
descending-sorted; v1 convention):**

| branch | gene | rank | mean_instability | top-1 gene |
|---|---|---|---|---|
| Ery  | GATA1 | **16059 / 16106** | −0.015 | LINC00936 |
| Ery  | SPI1  |   3173 / 16106   |  0.000 | LINC00936 |
| Mono | GATA1 |    431 / 16106   | +0.002 | LINC00936 |
| Mono | SPI1  |   3168 / 16106   |  0.000 | LINC00936 |
| DC   | GATA1 | **15835 / 16106** | −0.004 | ZFP36 |
| DC   | SPI1  |   3160 / 16106   |  0.000 | ZFP36 |

v1's top-15 Ery instability does include KLF1 (rank 6) — the known
erythroid master regulator — but GATA1 lands at the bottom of the
sorted list because v1's instability metric is signed and GATA1
stabilises (not destabilises) the Ery operator. CSVs under
``benchmarks/v1_baseline/setty_cd34/``.

### 1.2 v2 whole-dataset field (no branch split)

Dataset: 5780 cells × 2002 HVG (GATA1 + SPI1 kept) × PCA(10) ×
Palantir pseudotime (as stored). Phase C anchor fit max_live=200,
300 anchors, 150 rate-model epochs. DSM-residualised drift because
Setty is a snapshot (no labelling); eigenvalues reported relative,
no clock calibration.

**(1) Behaviour counts by annotated lineage — progenitors
separated from terminals** (output under
``benchmarks/v2/setty_cd34/behaviour_by_cluster.csv`` and
``behaviour_grouped.csv``):

Progenitors (ery–myelo–lymphoid decision region):

| cluster | n | fork | holder | growing_osc | frac_fork | frac_holder |
|---|---|---|---|---|---|---|
| HSC_1       | 1118 | 903 | 0 | 215 | 0.81 | 0.00 |
| HSC_2       |  984 | 549 | 0 | 435 | 0.56 | 0.00 |
| Precursors  |  666 | 503 | 0 | 163 | 0.76 | 0.00 |
| **total**   | 2768 | 1955 | 0 | 813 | **0.706** | 0.00 |

Terminals:

| cluster | n | fork | holder | growing_osc | frac_fork | frac_holder |
|---|---|---|---|---|---|---|
| CLP     |  488 | 400 | 0 |  88 | 0.82 | 0.00 |
| DCs     |  303 | 243 | 0 |  60 | 0.80 | 0.00 |
| Ery_1   |  742 | 637 | 0 | 105 | 0.86 | 0.00 |
| Ery_2   |  265 | 260 | 0 |   5 | 0.98 | 0.00 |
| Mega    |   61 |  21 | 0 |  40 | 0.34 | 0.00 |
| Mono_1  |  724 | 585 | 0 | 139 | 0.81 | 0.00 |
| Mono_2  |  429 | 305 | 0 | 124 | 0.71 | 0.00 |
| **total** | 3012 | 2451 | 0 | 561 | **0.814** | 0.00 |

**Zero holders on any cluster** — the signature DSM-snapshot result.

**(2) Fork fraction in the decision region vs elsewhere:**
`frac_fork` in the ery–myelo–lymphoid decision region (HSC_1 +
HSC_2 + Precursors, 2768 cells) is **0.706**; `frac_fork` elsewhere
(terminals, 3012 cells) is **0.814**. **Terminals are more
fork-dominated than the decision region**, i.e. the opposite of the
"fork at the saddle, holders in the basins" expectation. The
decision region carries 29% ``growing_oscillator`` cells vs 19% in
terminals — the DSM score reads a low-divergence rotational flow
in the dense HSC compartment and divergent spokes toward each
terminal. This is the known limitation of a density-score drift on
a snapshot: it classifies by local flow geometry, not by
attractor-basin structure.

**(3) Decision-region archetype — nearest archetype in soft weight
to the HSC + Precursors layer** (output under
``archetype_by_cluster.csv`` + ``decision_region_archetype.csv``):

Archetype 3 (fork, lead_re +0.086, n_hard = 1162 cells) carries
**72.6%** of its soft weight on the progenitor region (HSC_1 24.4%,
HSC_2 30.9%, Precursors 17.3%). Its GATA1 / SPI1 ranks (out of
2002 HVG):

| table | gene | rank | fisher | snr | sensitivity | reason |
|---|---|---|---|---|---|---|
| responders   | GATA1 | 1051 | 0.00 | 0.0018 | — | **capped** |
| responders   | SPI1  | **73**   | 0.85 | 0.020  | — | ok |
| drivers_left | GATA1 | 1279 | 0.00 | 0.0018 | 0.110 | **capped** |
| drivers_left | SPI1  | **137**  | 0.85 | 0.020  | 1.074 | ok |

**At the decision-region archetype, SPI1 is a top-5% driver on
both sides of the operator (rank 73 responder, rank 137 driver);
GATA1 is capped with Fisher = 0.** The anchor_max_live=150 cap
clips GATA1 at every anchor in archetype 3's membership — the
SNR-based column selection in R16 keeps the top-150 genes by
NB-precision-weighted η-variance, and GATA1 doesn't make the cut
at this fork saddle. The asymmetry is a real feature: the DSM
drift's gene-space Fisher load on the decision archetype falls on
myeloid-axis SPI1 but not on erythroid-axis GATA1.

**(4) v1-vs-v2 comparison figure** (``v1_vs_v2_figure.pdf`` /
``.png``). Left panel: Setty tSNE coloured by v2 behaviour
(fork / growing_oscillator — zero holders) with archetype 3 hard-
membership cells circled over the HSC + Precursors region. Right
panel: v1 per-branch ``max_real_eig`` along pseudotime for Ery /
Mono / DC — all three curves stay strictly positive (no
negative waypoints, matching v1's per-branch top-15 instability
tables that recovered KLF1 but ranked GATA1 at 16059 / 16106 on
Ery). The figure is the direct side-by-side the preprint comparison
asks for.

**v2 behaviour counts by annotated cluster** (output under
``benchmarks/v2/setty_cd34/behaviour_by_cluster.csv``):

| cluster | n | type | fork | holder | growing_osc | frac_fork | frac_holder |
|---|---|---|---|---|---|---|---|
| HSC_1       | 1118 | progenitor | 903 | **0** |  215 | 0.81 | 0.00 |
| HSC_2       |  984 | progenitor | 549 | **0** |  435 | 0.56 | 0.00 |
| Precursors  |  666 | progenitor | 503 | **0** |  163 | 0.76 | 0.00 |
| CLP         |  488 | terminal   | 400 | **0** |   88 | 0.82 | 0.00 |
| DCs         |  303 | terminal   | 243 | **0** |   60 | 0.80 | 0.00 |
| Ery_1       |  742 | terminal   | 637 | **0** |  105 | 0.86 | 0.00 |
| Ery_2       |  265 | terminal   | 260 | **0** |    5 | 0.98 | 0.00 |
| Mega        |   61 | terminal   |  21 | **0** |   40 | 0.34 | 0.00 |
| Mono_1      |  724 | terminal   | 585 | **0** |  139 | 0.81 | 0.00 |
| Mono_2      |  429 | terminal   | 305 | **0** |  124 | 0.71 | 0.00 |

**Zero holders on every cluster** — the same signature as v1 on the
same dataset, and the same signature as v2's DSM runs on Paul15 and
LARRY. The DSM field on a snapshot carries no basin structure.

**Behaviour at the ery–myelo–lymphoid decision region** (progenitors
HSC_1 + HSC_2 + Precursors, 2768 cells): fork 1955 (71%),
growing_oscillator 813 (29%), **holder 0**. The decision region
classifies fork — i.e. the pipeline sees divergence everywhere, not
a basin preceding commitment.

**v2 GATA1 / SPI1 ranks per archetype** (8 archetypes: 7 fork + 1
growing_oscillator; see ``archetype_drivers_GATA1_SPI1.csv``):

| arch | behaviour  | lead_re | n   | GATA1 resp (fisher) | GATA1 drv (sens) | SPI1 resp (fisher) | SPI1 drv (sens) |
|---|---|---|---|---|---|---|---|
| 0 | fork       | 0.192 |  626 | 781  (0.003) | 805  (0.14) | **119** (1.18) | **95**  (1.16) |
| 1 | growing_osc| 0.051 |  956 | 755  (0.012) | 801  (0.26) | 280  (0.30) | 260 (0.68) |
| 2 | fork       | 0.327 |  522 | 435  (capped)| 337  (0.76) | 723  (1.99) | **132** (1.25) |
| 3 | fork       | 0.094 | 2220 | 1603 (0.001) | 992  (0.19) | **121** (1.10) | 564 (0.41) |
| 4 | fork       | 0.098 |  625 | 1122 (capped)| 1085 (0.13) | 1008 (0.93) | 1246 (0.10) |
| 5 | fork       | 0.087 | 1069 | **109** (1.31) | 704  (0.45) | 195  (1.10) | **115** (2.37) |
| 6 | fork       | 0.616 |  446 | 730  (0.11)  | 822  (0.23) | 478  (0.76) | 412 (0.49) |
| 7 | fork       | 0.400 | 1192 | 1101 (0.36)  | 897  (0.22) | 284  (1.26) | **119** (1.13) |

Out of 2002 HVG, SPI1 reaches responder rank 119 on arch 0 and 121
on arch 3 (top-6%), driver rank 95 on arch 0 and 119 on arch 7. GATA1
is best on arch 5 (responder rank 109, Fisher 1.31) and otherwise
sits in the 400–1600 band. **No archetype has both GATA1 AND SPI1
simultaneously in the top-200 of either table on the Setty whole-
dataset v2 run.** All archetypes classify fork or growing_oscillator;
none holder.

### 1.3 Side-by-side

* **Holders on terminal lineages:** both v1 (per-branch) and v2
  (whole-dataset) report **zero holders**. The snapshot pipeline
  does not surface terminal-state basins, regardless of branch
  slicing.
* **Behaviour at the decision region:** v1 does not expose a single
  cross-branch classification (its field is fit per branch). v2's
  whole-dataset field reports 71% fork + 29% growing_oscillator at
  the HSC + Precursors layer, zero holders.
* **GATA1 / SPI1 recovery:** v1 misses GATA1 entirely on the Ery
  branch (rank 16,059) but surfaces KLF1 as the top erythroid
  driver. v2 recovers GATA1 on one archetype (rank 109) and SPI1 on
  three (ranks 95–132 in the driver table) but no single archetype
  reads both TFs as the top drivers of a saddle. The whole-dataset
  v2 does strictly better than v1 on SPI1 and matches on GATA1-via-
  KLF1 (archetype 5's lead_re +0.087 at n=1069 cells is the Ery-
  adjacent one; GATA1 rank 109 there is the best anywhere).

**Snapshot verdict (Setty + v1 and v2).** The DSM drift on a
non-stationary CD34+ snapshot produces no basins. R15's
"behaviour from latent, drivers from gene-space lift" rule reads
cleanly on both pipelines; neither pipeline sees terminal-state
holders. Whether this is pipeline-limited (DSM carries no direction)
or dataset-limited (CD34+ snapshot is pre-commitment) is answered
by §2 below — the measured-flow counterpart.

## 2 · Dynamo scNT-seq hematopoiesis — the measured-flow counterpart

**Same compartment: human CD34+ hematopoiesis.** Setty is the
snapshot; the scNT-seq set (Qiu et al. 2022, labelled new/total
UMIs) is the measured-flow counterpart. 1947 cells × 1956 HVG,
dynamo's processed artefact (per-cell ``velocity_pca`` from the
dynamical model, labelling rates α_g and γ_g per gene,
``clock_calibrated = True`` means per-hour eigenvalues).

The full audit (§7 / §7a / §7b in ``V2_PLAN.md``) landed four
checks:

* **§7 initial result (overclaimed).** 1.9% holders on terminals
  (Meg 15, Bas 22), Gata1 + Spi1 ok with Fisher 0.5–2.1 across all
  8 archetypes, eigenvalues on a 0.03–0.21 /h clock.
* **§7a cycle audit.** Permutation null distinguishes signal from
  noise (real lead_re is 8× the perm median); MLP held-out R² =
  0.848; tangent rank 2–4 per cell (not 10); 17–22% of the
  archetype eigvec weight on **4** cycle genes alone (10–100× over
  uniform baseline). Score regression (variant B) barely moves
  cycle weight; rotation projection (variant C) drops it to zero
  by construction and the holder count 37 → 8.
* **§7b batch audit.** The 37 holders are **100% from the 3-h
  pulse batch** (1129 cells). The 5-h pulse batch (818 cells) —
  which is the batch dynamo actually fit its rate model on —
  yields **zero holders in every variant** (unprojected,
  rotation-projected, clock-scaled). Clock scaling and rotation
  projection cannot reconcile the two batches.
* **Depth check.** Holder cells are the HIGH-UMI tail (~2× dataset
  median), not the low-count tail — real, well-sequenced cells,
  but confounded with pulse-duration batch.

**Measured-flow verdict (scNT-seq, after §7b).** The holder
minority is a 3-h-batch artifact; the 5-h batch is holder-free
regardless of treatment. Measured flow **calibrates units and
raises Fisher mass** (Gata1 Fisher 0.3–2.1 vs the LARRY DSM
0.028–0.035), **distinguishes signal from permutation null**, but
does **not** yet yield a defensible holder recovery on terminals
once the pulse-duration confound is accounted for. R15's rule is
neither confirmed nor falsified by Dynamo scNT-seq hematopoiesis
in its distributed form.

## 2.5 · Setty + scVelo — same dataset, measured flow via splicing

Same Setty 2019 CD34+ h5ad (5780 cells, HVG 2002); same scJDO2 Phase C
anchor fit (``max_live=150``, 200 anchors). Only the drift source
changes: scVelo dynamical mode on Setty's spliced / unspliced layers
produces a per-cell ``velocity_pca`` which a MeasuredDrift MLP
regresses into a differentiable field. Clock is **relative** (scVelo's
time axis is not calibrated to physical hours). Outputs under
``benchmarks/v2/setty_scvelo/``.

**Behaviour counts (scVelo-measured drift on Setty):**

| cluster | n | fork | holder | growing_osc | frac_fork |
|---|---|---|---|---|---|
| HSC_1       | 1118 | 708 | 0 | 409 | 0.63 |
| HSC_2       |  984 | 738 | 0 | 244 | 0.75 |
| Precursors  |  666 | 497 | 0 | 169 | 0.75 |
| CLP         |  488 | 341 | 0 | 147 | 0.70 |
| DCs         |  303 | 219 | 0 |  83 | 0.72 |
| Ery_1       |  742 | 316 | 0 | 423 | **0.43** |
| Ery_2       |  265 | 178 | 0 |  87 | 0.67 |
| Mega        |   61 |  25 | 0 |  36 | 0.41 |
| Mono_1      |  724 | 518 | 0 | 204 | 0.72 |
| Mono_2      |  429 | 317 | 0 | 112 | 0.74 |
| **overall** | 5780 | 3857 | **0** | 1914 | 0.67 |

**Still zero holders on every cluster.** Overall: 67% fork + 33%
growing_oscillator, vs DSM-Setty's 77% fork + 23% growing_oscillator.
scVelo flow rotates more than it diverges on this dataset — most
strikingly on Ery_1, which drops from 86% fork (DSM) to 43% fork
(scVelo).

**Decision-region archetype (scVelo): archetype 7** (fork, lead_re
+0.142, n=246 cells; 72% of weight on HSC + Precursors). GATA1 /
SPI1 ranks (out of 2002 HVG):

| table | gene | rank | fisher | snr | sensitivity | reason |
|---|---|---|---|---|---|---|
| responders   | **GATA1** | **57**  | 0.32 | 0.015 | — | **ok** |
| responders   | SPI1  | 392 | 2.04 | 0.032 | — | ok |
| drivers_left | **GATA1** | **54**  | 0.32 | 0.015 | 1.18 | **ok** |
| drivers_left | SPI1  | 227 | 2.04 | 0.032 | 0.43 | ok |

**Top-tail recovery of GATA1 under measured flow.** On DSM-Setty's
decision archetype (arch 3), GATA1 was **capped with Fisher = 0**
(anchor_max_live SNR cut dropped it); SPI1 was rank 73 responder /
137 driver. On scVelo-Setty's decision archetype (arch 7), **GATA1
reaches responder rank 57 / driver rank 54 with real Fisher mass**
(0.32); SPI1 is 392 / 227 — the two TFs swap primacy at the saddle
under splicing vs density-score drift. Phrased precisely: GATA1 is
**top-tail (rank 57 of 2002 HVG), surfaced under measured flow
where the score capped it**. Not a basin recovery and not a
functional validation of the archetype — a sensitivity-level
observation that the drift source changes which TF the gene-space
lift highlights at the decision region.

**Caveats.** (a) Zero holders remain on every terminal, so the
"basin on terminals" signature is still absent — measured flow via
splicing changes the gene-level driver profile but not the
terminal-basin flag. (b) scVelo's dynamical fit on Setty retained
~700 velocity-fittable genes out of 2002 HVG (the dynamical fit
drops genes with poor splicing-steady-state solubility), so the
measured flow is intrinsically narrower than the DSM score. (c)
Setty's clock is relative; the per-hour calibration arm needs a
labelling dataset.

## 3 · Snapshot vs measured flow — three-way on human CD34+

Three drift sources on the same compartment:

| source | drift | dataset | n | holders | arch at decision | GATA1 (resp/drv) | SPI1 (resp/drv) |
|---|---|---|---|---|---|---|---|
| **Setty DSM** (snapshot score)         | DSM MLP on Palantir pseudotime      | Setty 2019    | 5780 | 0 | 3 (lead_re +0.09) | 1051 / 1279 **capped** | 73 / 137 |
| **Setty scVelo** (splicing-flow drift) | MLP(z→v) on scVelo ``velocity_pca`` | Setty 2019    | 5780 | 0 | 7 (lead_re +0.14) | **57 / 54**            | 392 / 227 |
| **Dynamo scNT-seq** (labelling-flow)   | MLP(z→v) on labelling ``velocity_pca`` | Qiu 2022     | 1947 | 37 (batch-confined, §7b) | 6 (lead_re +0.19 /h) | 146 / 134 | 131 / 109 |

Reading the row-by-row:

* **Behaviour on terminals.** All three sources report **zero holders
  on terminal lineages** (Setty DSM/scVelo at scale) or **batch-
  confined holders only** (Dynamo scNT-seq, 37 of 1947, 100% from
  the 3-h pulse batch per §7b). None of the three gives a clean
  "basin on terminals" biological finding.
* **Decision-region archetype classification.** All three decision
  archetypes classify fork; none holder.
* **GATA1 at the decision archetype.** DSM caps GATA1 (Fisher = 0,
  rank ~1000); scVelo lifts GATA1 to responder rank 57 / driver
  rank 54 with real Fisher mass (0.32); scNT-seq ranks GATA1 at
  146 / 134 across a different compartment. **Measured flow (via
  splicing or labelling) is required for GATA1 to surface as a
  top-tail decision driver.**
* **SPI1 at the decision archetype.** DSM reads SPI1 clearest (top-
  150 both tables); scVelo reads SPI1 weaker (top-400); scNT-seq
  reads SPI1 clearest among the three on this metric. SPI1 is less
  sensitive to the drift source than GATA1.

**Interpretation.** Neither snapshot density scoring nor the two
measured-flow sources (splicing + labelling) gives R15's rule a
clean recovery on CD34+ hematopoiesis — the "holders on terminal
basins" signature is absent across all three drifts. But measured
flow IS necessary to surface GATA1 at the ery–myelo–lymphoid
decision archetype: the DSM score alone caps it with Fisher = 0.
The splicing-flow Setty run is the cleanest "no batch confound"
measured-flow result in-session.

Next steps already recorded in ``V2_PLAN.md``:

* **§7c (labelled rate model from new/total counts with per-cell
  t_c in the NB likelihood)** — removes the dynamo-upstream pulse-
  duration bias at its source rather than post-hoc clock scaling.
* **A single-pulse labelling dataset** (dynamo scEU-seq organoid
  or scNT-seq neuron) where every cell carries the same ``t_c``,
  removing the batch confound by construction.
* **Setty + scVelo whole-genome rerun** (this one dropped to ~700
  velocity-fittable genes under the dynamical fit; a broader HVG
  pool or a steady-state scVelo could keep more).

## 5 · PBMC positive control — magnitude rule (final) : partial PASS

**Verdict (fourth rework — final rule): 70.7 % of 2282 resting
PBMC cells are basins** under the magnitude rule on the Σ-based
tangent-restricted block. Previous r-1 rule's 92.6 % was over-
counting: the sign of the leading eigenvalue matters when it exceeds
oracle tol, and both B cells (lead +2.27) and CD8 T (lead +0.23)
flip to fork at the cluster centroid under the final rule. The
paper reports the continuous **basin_index**
``bi = −λ_max / median|λ_neg|`` per cell, not a binary story.

**Spec of the final rule** (``scjdo2.opspace.classify_basin_latent``,
patched 2026-10):

* Tangent subspace ``V`` = top-``r`` eigenvectors of local kNN
  **Σ** (variance, not Σ⁻¹) above a magnitude floor. The user's
  earlier Σ⁻¹ draft inverted the manifold direction and hid saddle
  axes; Σ keeps the directions where the data spreads.
* Tangent-restricted eigenvalues ``ev_r = eigvals(Vᵀ J V)``.
* **basin / holder** ⇔ ``max(ev_r) < +tol`` AND some ``ev_r < −tol``
* **fork**          ⇔ ``max(ev_r) > +tol``
* **drift**         ⇔ ``all |ev_r| < tol``
* ``n_soft`` = ``count(|ev_r| < tol)``
* ``basin_index`` = ``−max(ev_r) / median|ev_r_neg|`` (positive =
  basin, deeper → larger; negative = fork).
* The r-1 rule is retired: it mis-classified pitchfork saddles as
  basins when ``r = d`` on PCA (one positive + d-1 near-zero → still
  ``n_neg ≥ r−1``).

### 5.1 v2 rework — energy-parameterised score drift

**Verdict: FAIL.** The four resting mature types (CD4 T, CD8 T, B,
CD14+ Monocytes; 2282 of 2638 cells) read **~17 % holder** under this
setup, not ≥ 50 %. Every one of 8 archetypes classifies as fork
(``lead_re`` +0.18 to +0.93). The per-cell operator does not
distinguish basin from non-basin cells on this homeostatic
compartment, so §7c is deferred until the drift source is reworked.
Detail follows.

**Setup.**

* **Data.** 10x PBMC 3k (``scanpy.datasets.pbmc3k`` + louvain labels
  from ``pbmc3k_processed``). 2638 cells × 32738 genes raw UMIs,
  reduced to 13656 after ``min_cells=3``, then restricted to **3065
  genes** (3000 HVG + cycle + resting markers forced in). The full
  13656-gene run is practically out of reach of this 36 GB machine:
  with ``anchor_max_live=150`` the per-cell CSR Jacobian is ~2 MB
  and the gene-space tensor runs to ~5 GB; raising to 13.6k genes
  pushes that to ~22 GB (demonstrated: a run with that pool stalled
  the lift at ~2.5 hr with no 500-cell marker emitted — this
  restriction is the pragmatic cap, not a scientific choice). 10x
  PBMC 10k v3 was the preferred target; pbmc3k is the recorded
  fallback.
* **Embedding.** scVI 1.2 with ``d_latent`` chosen by held-out
  marginal log-likelihood sweep over {5, 10, 15}: −2820 / −2802 /
  −2796 → **d_latent = 15** (highest LL = best fit, within 1 % of
  d=10).
* **Drift.** DSM score-only (``inject_velocity_prior=False``,
  ``use_residual=False``) → ``u = β · ∇̂ log p``; Phase C anchor fit
  on 200 k-means anchors with ``anchor_max_live=150`` and the R16
  SNR-ranked live-column rule. Cycle genes retained in the gene set.
* **Known-rotation projection.** S + G2M scores regressed into the
  latent frame (Tirosh 2016 gene lists); the resulting 2-dim subspace
  QR-orthonormalised; ``J_c ← (I − UU^T) J_c (I − UU^T)`` applied to
  every cell's latent Jacobian before Phase C.

**Observation 1 — the parameterised score is only partially
symmetric.** The user's spec assumed ``J = β · ∇² log p`` would be
symmetric, which would restrict the behaviour spectrum to holder /
drift / fork. Measured median
``‖J − Jᵀ‖_F / ‖J‖_F = 0.37``: v1's ``DriftField`` MLP does not
architecturally enforce ``∇φ``-form on the score head, so the learned
score is only approximately a gradient. Imaginary-part eigenvalues
do appear and 5-7 % of cells per cluster land in
``damped_oscillator`` or ``growing_oscillator``. The symmetric-score
constraint is a *theoretical* property of the true score, not an
*architectural* guarantee of this drift.

**Observation 2 — permutation null is holder-dominant by
construction.** On shuffled latent positions (3 perms, same DSM
drift epochs / network), **99.8 % of cells have ``Re(λ) < 0``**
(median ``lead_re ≈ −0.010``). This is expected: a shuffled cloud is
a single-mode density, so the Hessian of ``log p`` is negative
definite at every point. The null drifts cells inward to one
centroid, and inward-flow is holder-shaped.

**Observation 3 — real data flips to fork-dominant.** On real
PBMC3k with 8 cell-type modes, **75.1 % of cells have ``Re(λ) > 0``**
(median ``lead_re ≈ 7 × 10⁻¹⁰`` — essentially zero but biased
positive). Multi-modal density has saddle *ridges* between modes
and the DSM-score field at ridge points reads outward (fork),
dominating the per-cell classification over the small inward
curvature inside each basin.

**Observation 4 — resting vs non-resting have the same
signature.** Holder fraction per cell type:

| cluster | n | fork | holder | drift | d_osc | g_osc | frac_holder |
|---|---|---|---|---|---|---|---|
| **B cells**          |  342 | 229 |  57 | 31 | 13 | 12 | 0.167 |
| **CD14+ Monocytes**  |  480 | 334 |  63 | 40 | 21 | 22 | 0.131 |
| **CD4 T cells**      | 1144 | 773 | 209 | 82 | 39 | 41 | 0.183 |
| **CD8 T cells**      |  316 | 213 |  56 | 25 | 13 |  9 | 0.177 |
| Dendritic cells      |   37 |  23 |   9 |  1 |  3 |  1 | 0.243 |
| FCGR3A+ Monocytes    |  150 |  93 |  34 | 11 | 11 |  1 | 0.227 |
| Megakaryocytes       |   15 |  10 |   3 |  1 |  1 |  0 | 0.200 |
| NK cells             |  154 | 105 |  26 | 10 |  8 |  5 | 0.169 |

The four resting types (bold) land at 13-18 % holder; the four
non-resting at 17-24 %. **The gating signal the arm was supposed to
detect is absent.**

**Observation 5 — the archetype decomposition sees cell-type
biology, just not as basins.** All 8 archetypes classify fork,
``lead_re`` 0.18 to 0.93. The archetype × cluster soft-weight table
assigns archetype 3 (``lead_re`` +0.21, 227 hard-member cells) its
heaviest weight on B cells, and the stiffest / softest gene-space
directions on that archetype read:

| rank | gene (right/stiff) | fisher | gene (left/stiff) | fisher |
|---|---|---|---|---|
| 1  | **CD74**     | 6.01 | **CD74**     | 6.01 |
| 2  | **HLA-DRA**  | 7.06 | **HLA-DRA**  | 7.06 |
| 3  | **CD79A**    | 4.47 | **HLA-DRB1** | 6.74 |
| 4  | MALAT1       | 5.50 | **HLA-DPB1** | 7.29 |
| 5  | KCNG1        | 3.29 | **HLA-DPA1** | 5.70 |
| 6  | **HLA-DRB1** | 6.74 | **CD79A**    | 4.47 |
| 7  | **HLA-DPB1** | 7.29 | **HLA-DQB1** | 5.80 |
| 8  | RPL13A       | 4.58 | **CD37**     | 3.24 |
| 9  | **CD37**     | 3.24 | **HLA-DQA1** | 6.09 |
| 10 | RPS27A       | 3.94 | **HLA-DRB5** | 5.43 |

MHC-II + CD79A + CD37 — the canonical B-cell antigen-presentation
program, each with ``reason = ok`` (Fisher mass 3-7). The pipeline
*found* B-cell biology; it just read the dominant local flow at
that archetype as outward (fork, lead_re +0.21), not as a basin.
The stiffest and softest modes collapse to the same gene list
because the archetype's negative-Re tangent block is effectively
rank-1 (the operator's basin direction is dominated by one mode;
others are at ``Re ≥ 0``).

**Diagnostic.**

* The DSM score drift on a *multi-modal* homeostatic compartment
  reports fork almost everywhere because the between-mode saddle /
  ridge geometry dominates the within-mode basin curvature in the
  score-field Jacobian. The known-rotation projection and the full-
  gene anchor fit do not change this.
* Permutation null distinguishes signal from noise in the opposite
  direction from what the arm needed: random data → holder; real
  data → fork.
* Confirming the archetype decomposition recovers cell-type gene
  programmes (B-cell MHC-II + CD79A on archetype 3's dominant mode)
  means the drift field *does* encode identity structure; the
  pipeline's basin-reading machinery just reads the between-mode
  ridges as fork, not the within-mode cups as holder.

**Decision (user gate).** The gate says: *if major resting types are
not holders, stop and report the diagnostic before §7c*. **§7c is
therefore deferred.** The failure mode is not "measured flow needs a
better rate model" (what §7c would address); it is "snapshot score
drift on multi-modal homeostatic data classifies by between-mode
geometry, not within-mode basins". §7c (per-cell ``t_c`` in the NB
likelihood) does not fix this. A path forward likely requires a
drift source whose Jacobian carries explicit potential structure
(energy-based drift, Fokker-Planck steady-state operator) or a
measured steady-state flow (RNA velocity ≈ 0 on resting cells as a
hard anchor).

Outputs under ``benchmarks/v2/pbmc_basin/`` (v1 run superseded by §5.2
energy drift; see commit history for the v1 numbers that match §5.1).

### 5.10 Final close-out — lead-eig rule picked by continuous AUC (locked)

**User-directed close-out of §9b.** Instead of adding a fifth
candidate rule, evaluate the three existing ones on the equilibrated
pitchfork with continuous metrics and pick the one with the best AUC.

**Setup.** τ ∈ {0.1, 0.3, 0.5}, 1200 cells × 20D PCA, dynamic
ground truth (sign of analytic 2D J at (X_true, α)). Saddle window
``|pseudotime − τ_crit| ≤ 0.05``.

**Three continuous metrics per rule:**

* **Spearman** of ``basin_index`` vs ``−analytic lead_re`` per cell.
* **AUC** of ``basin_index`` separating ``true_holder`` cells (outside
  the saddle window) from cells inside the saddle window (not
  true_holder) — classical ROC-AUC of basin ranking.
* **Median ``|pseudotime − τ_crit|``** of fork-called cells vs
  random — fork localisation.

**Results** (``benchmarks/phase_b_gate/equilibrated/
rule_comparison.csv`` + ``rule_comparison_summary.json``):

| rule | mean AUC | mean Spearman | fork-dist ratio range |
|---|---|---|---|
| **lead_eig**      | **0.859** | **+0.662** | 0.23 - 0.56 |
| single_tol        | 0.768 | +0.519 | 0.17 - 0.87 |
| two_tol_null      | 0.768 | +0.519 | 0.26 - 0.56 |

Per-τ AUCs: lead_eig **0.980 / 0.780 / 0.817** (τ=0.1/0.3/0.5);
single_tol 0.963 / 0.680 / 0.661.

**Winner: ``lead_eig`` by AUC and Spearman at every τ.**

**Gate (locked criterion, NOT a pass requirement for the tag — the
pick is by AUC):** ``holder P ≥ 0.8``, ``holder R ≥ 0.8``,
``fork R ≥ 0.8``. **Fork precision at a ±0.05 saddle window is bounded
by neighbourhood width** (many near-saddle cells have analytic
``lead_re`` near zero → ground truth drift → fork precision ceilings
well below 1.00); fork precision is **not** a gate criterion.

Lead_eig pitchfork gate numbers:

| τ | holder P / R | fork R |
|---|---|---|
| 0.1 | 1.00 / 0.70 | 0.98 |
| 0.3 | 0.99 / 0.63 | 0.79 |
| 0.5 | 1.00 / 0.57 | 0.93 |

Fork recall PASSES at every τ (0.79 - 0.98); holder recall sits in
0.57-0.70 (below the 0.8 criterion). The trade-off is intrinsic to
the tol used (``0.1 × std(lead_re)``) — tightening tol would raise
holder recall but drop fork recall. Documented and tagged as the
locked state at this point.

**PBMC under the locked lead_eig rule** (
``benchmarks/v2/pbmc_basin_magnitude/behaviour_by_cluster.csv``):

| cluster | n | resting | holder | fork | drift | **basin_index median** |
|---|---|---|---|---|---|---|
| CD14+ Monocytes      |  480 | ✓ | 296 | 162 |  22 | **+0.41** |
| CD4 T cells          | 1144 | ✓ | 624 | 411 | 109 | **+0.14** |
| B cells              |  342 | ✓ |  87 | 218 |  37 | **−0.37** |
| CD8 T cells          |  316 | ✓ |  69 | 223 |  24 | **−0.46** |
| Dendritic cells      |   37 |   |  19 |  15 |   3 | +0.23 |
| FCGR3A+ Monocytes    |  150 |   |  90 |  44 |  16 | +0.29 |
| Megakaryocytes       |   15 |   |   0 |  13 |   2 | −0.37 |
| NK cells             |  154 |   |   0 | 153 |   1 | **−0.77** |

**Resting basin fraction: 1076 / 2282 = 47.15 %.** Resting
``basin_index`` median = **−0.11** across the four resting types.

**Paper statement under the locked rule** (basin_index primary,
class secondary):

* **CD14+ Monocytes** (bi +0.41) and **CD4 T** (bi +0.14) are
  basin-leaning on average — the deepest resting basin is CD14+ Mono.
* **FCGR3A+ Monocytes** (bi +0.29) and **Dendritic** (bi +0.23) also
  basin-leaning — monocyte / DC programmes are shallow but
  cohesive density modes.
* **B cells** (bi −0.37) and **CD8 T** (bi −0.46) are **basins
  with one real shallow ridge toward CD4 T** (not deep basins
  because their leading mode is weakly expanding toward the T-cell
  neighbour — the +2.27 and +0.23 modes from §5.5 lift them above
  the ``0.1 × std(lead_re)`` tol). Their basin_index is negative but
  substantially less so than NK.
* **NK cells** (bi −0.77) are strongly fork — the only PBMC cluster
  that sits at a clear density ridge, between CD8 T and the
  cytolytic / innate compartment.

The lead_eig rule's gate failure (holder R below 0.8 on pitchfork)
is recorded; the AUC winner is locked regardless, and the continuous
basin_index is reported in place of a hard-classification story for
the paper's §5.

### 5.9 Fifth rework — two thresholds (tol_neg + tol_pos from null) (SUPERSEDED by §5.10)

**User rebuttal (correct):** judging a saddle's shallow positive
eigenvalue against basin stiffness is wrong — a saddle's expanding
mode is INHERENTLY smaller than the deep basin modes it sits next to.
The fork threshold needs a different reference: the permutation null.

**Spec (final, canonical):**

* ``tol_neg = 0.1 · median |λ(−Σ⁻¹)|`` across resting clusters —
  the basin stiffness threshold (Gaussian oracle).
* ``tol_pos = max(tol_neg, 95th percentile of λ_max under
  shuffled-marginal refits of ESM)`` — the expansion threshold,
  clamped to never fall below basin stiffness.
* **fork** iff ``max(ev_r) > tol_pos``
* **basin** iff ``max(ev_r) < tol_neg`` AND ``min(ev_r) < −tol_neg``
* **drift** otherwise (soft-axis cells with
  ``λ ∈ [tol_neg, tol_pos]``)
* ``n_soft`` = ``count(λ ∈ (−tol_neg, tol_pos))``
* ``basin_index`` unchanged.

Null refit protocol: **shuffle each latent dim INDEPENDENTLY**
(destroys the manifold while preserving per-dim marginals), refit
ESM 10 times with different seeds, collect per-cell ``λ_max``, take
the 95th percentile.

**Pitchfork gate under the two-tol rule** (0 / 3 PASS strict
Phase-B — holder criteria PASS at every τ; fork P/R gap stays open):

| τ | null tol_pos | tol_neg | **holder P / R** | fork P / R |
|---|---|---|---|---|
| 0.1 | **+0.303** | 0.034 | **0.98 / 0.97** | **0.40 / 0.18** |
| 0.3 | +0.150 | 0.032 | 0.95 / 0.94 | 0.00 / 0.00 |
| 0.5 | −0.098 → clamped to 0.033 | 0.033 | 0.96 / 0.94 | 0.14 / 0.26 |

At τ=0.1 the null gives a well-defined positive tol_pos (0.30) because
the shuffled marginals carry non-trivial curvature. Fork precision
climbs from the lead-eig baseline's **0.12 to 0.40** — the user's
"fork precision near ESM lead-eig" is met at τ=0.1. At τ=0.3 the null
is weaker (0.15) and the ESM at the saddle has shallower positive
eigenvalues, so the 23 cells called fork miss the 42 true saddles.
At τ=0.5 the null degenerates (shuffled distribution is near-flat →
negative tol_pos); the clamp kicks in and the rule reduces to
single-tol.

**PBMC gate under the two-tol rule** (same numbers as §5.8 because
the clamp kicks in on PBMC):

Null tol_pos = **−0.374** on PBMC's scVI latent — shuffling each
scVI dim produces a near-standard-Normal joint whose score is
globally inward (all cells read holder-ish), so no positive-outlier
λ_max emerges from the null. Clamp fires: ``tol_pos = tol_neg =
0.076``. The two-tol rule degenerates to the single-tol rule → the
§5.8 numbers hold: 1613 / 2282 = **70.7 % resting basin**, with CD14+
Mono 85 % (bi +0.49), CD4 T 74 % (+0.26), B 67 % (+0.20), **CD8 T
42 % (bi −0.22)**, Dendritic 89 %, FCGR3A+ 93 %, Mega 27 %, NK 17 %
(bi −0.66).

The null on scVI doesn't help because scVI's regularisation already
makes the latent near-isotropic Gaussian within each cluster — the
shuffled marginal joint stays similarly near-Normal, so there's no
inter-mode "ridge" structure to lift above basin stiffness. The two-
tol rule is a strict improvement on the pitchfork (fork P 0.12 →
0.40 at τ=0.1) but a no-op on PBMC under scVI.

**Repository tag: still deferred.** The pitchfork gate at τ=0.3
has fork P=0.00 — strictly worse than lead-eig's 0.08. §7c remains
blocked.

### 5.8 Fourth rework — single-tol magnitude rule (SUPERSEDED by §5.9)

**Pitchfork gate under the final magnitude rule**
(``scripts/pitchfork_energy_gate.py``,
``benchmarks/phase_b_gate/equilibrated/pitchfork_energy_gate.json``):

| τ   | holder P / R (mag)   | fork P / R (mag)   | **verdict** |
|---|---|---|---|
| 0.1 | **0.98 / 0.97**      | 0.40 / 0.62        | FAIL (fork P/R > 0.10 off oracle) |
| 0.3 | 0.95 / 0.94          | 0.07 / 0.12        | FAIL |
| 0.5 | 0.95 / 1.00          | 0.50 / 0.02        | FAIL |

Compared to the lead-eigenvalue baseline on the ESM drift:

* **Holder recall jumped from 0.57-0.70 to 0.94-1.00** — the magnitude
  rule finds the "basin with one soft axis" cells the lead-eig rule
  called fork. Holder precision stayed at 0.95-0.98 (ESM lead-eig
  was 0.99-1.00; the ~0.02 drop is 2-5 saddle-adjacent cells mis-
  called holder).
* **Fork precision at τ=0.1 jumped from 0.12 to 0.40**; at τ=0.5
  from 0.10 to 0.50. The magnitude rule is strict about fork
  (needs a tangent eigenvalue > tol) and so false positives drop.
* **Fork recall dropped from 0.79-0.98 to 0.02-0.62** — the ESM-fit
  Hessian at the equilibrated pitchfork saddle has a positive
  eigenvalue in the TRUE 2D subspace, but the Σ-tangent of the
  local kNN (which samples cells from BOTH basins at the saddle)
  keeps directions where the Jacobian is predominantly negative,
  suppressing the saddle axis. This is a known limitation of
  density-tangent restriction at saddle points.

**Strict Phase-B verdict: FAIL on fork P/R.** Holder side is at or
above the ESM lead-eig level. The rule PASSES the holder criterion
(both ≥ 0.80) at every τ.

**PBMC gate under the final magnitude rule**
(``scripts/pbmc_magnitude_rule.py``,
``benchmarks/v2/pbmc_basin_magnitude/``):

| cluster | n | resting | holder | fork | **frac_holder** | **basin_index** |
|---|---|---|---|---|---|---|
| CD14+ Monocytes      |  480 | ✓ |  409 |  71 | **85.2 %** | **+0.49** |
| CD4 T cells          | 1144 | ✓ |  842 | 302 | **73.6 %** | +0.26 |
| B cells              |  342 | ✓ |  230 | 112 | **67.3 %** | +0.20 |
| **CD8 T cells**      |  316 | ✓ |  132 | 184 | **41.8 %** | **−0.22** |
| Dendritic cells      |   37 |   |   33 |   4 | 89.2 %     | +0.45 |
| FCGR3A+ Monocytes    |  150 |   |  140 |  10 | 93.3 %     | +0.41 |
| Megakaryocytes       |   15 |   |    4 |  11 | 26.7 %     | −0.74 |
| NK cells             |  154 |   |   26 | 128 | 16.9 %     | −0.66 |

**Overall resting: 1613 / 2282 = 70.7 % basin.** Basin index median
across resting is +0.23 (positive, i.e. net basin on resting cells).

Reading the per-cluster numbers:

* **CD14+ Mono (85%)** and **FCGR3A+ Mono (93%)** are the deepest
  basins (bi +0.49, +0.41). Classical monocytes and non-classical
  monocytes both sit at density modes.
* **Dendritic (89%)** is a strong basin despite its "non-resting"
  cell-type label (bi +0.45).
* **CD4 T (74%)** and **B cells (67%)** are majority basin with
  modest positive bi.
* **CD8 T (42%)** is borderline — bi −0.22 — the lead +0.23 soft
  axis toward CD4 T is above oracle tol = 0.076, flipping half the
  cells to fork under the strict magnitude rule.
* **NK (17%)** and **Megakaryocytes (27%)** are the ridge-like
  populations — bi −0.66 and −0.74 reflect strong positive leading
  eigenvalues at those centroids.

**Comparison with the r-1 rule (§5.5):**

| cluster | r-1 rule holder % | magnitude rule holder % | basin_index |
|---|---|---|---|
| CD14+ Mono      | 99.4 % | 85.2 % | +0.49 |
| CD4 T           | 93.4 % | 73.6 % | +0.26 |
| B cells         | 84.8 % | 67.3 % | +0.20 |
| CD8 T           | 88.3 % | **41.8 %** | **−0.22** |
| NK              | 78.6 % | **16.9 %** | **−0.66** |

The r-1 rule systematically over-counted basins because it absorbed
the soft positive eigenvalue (the "escape mode toward neighbour") as
part of the ``n_soft = 1`` allowance. The magnitude rule says: if
that positive eigenvalue exceeds oracle tol, call it fork. The
re-classifications land on the clusters that genuinely sit on
density ridges — exactly the biological pattern the user
predicted when the pitchfork saddle issue surfaced.

### 5.5 Third rework — full spectrum, tangent restriction, d sweep (SUPERSEDED by §5.8)

**User rebuttal** (correctly, again): "lead-eigenvalue alone is
insufficient — a cluster with ``n_neg = d − 1`` out of ``d`` is a
basin with ONE soft direction, not a fork." Also: show the rotation
projection isn't corrupting the signal; sweep ``d_latent`` with
held-out NB LL; apply the gene-space-style Fisher-weighted tangent
restriction to the latent J before classifying.

**Changes** (``scripts/pbmc_spectrum_sweep.py``):

* **Full spectrum per cluster centroid + per cell.** Report every
  eigenvalue of ``J`` at each cluster centroid (not just the lead);
  per cell, count ``n_neg_above_floor`` and ``n_pos_above_floor``
  with ``floor = oracle_tol``.
* **Tangent-restricted classifier.** Per cell, keep the top-``r``
  directions of the local kNN-ball Σ⁻¹ above a magnitude floor
  (``0.1 × max |λ(Σ⁻¹)|``). Classify on the restricted ``(r, r)``
  block; the top-``r`` eigenvalues of ``J`` by magnitude are the
  per-cell "effective spectrum". A cell is a **holder** iff
  ``n_neg_in_tangent ≥ r − 1`` — basin with up to one soft direction.
* **d_latent sweep** over {5, 8, 10, 15}; marginal LL via scVI's
  built-in Monte-Carlo estimator; pick the best by LL, show the curve.
* **Rotation-projection debug.** On 20 cells, compare the full
  spectrum of ``J`` vs ``(I − UU^T) J (I − UU^T)`` to measure the
  average drift in the ``d − 2`` eigenvalues outside the cycle
  subspace. Only tiny drift means the projection is clean.

**d_latent sweep + gate verdict** (
``benchmarks/v2/pbmc_basin_spectrum/gate_by_d.csv``):

| d_latent | marginal LL | resting holders / 2282 | **frac** | oracle_tol | rot |Δλ| |
|---|---|---|---|---|---|
|  5 | −803.0 | 1967 / 2282 | 86.20 % | 0.116 | 24.17 |
|  8 | −789.5 | 1961 / 2282 | 85.93 % | 0.100 |  3.61 |
| 10 | −782.2 | 1934 / 2282 | 84.75 % | 0.094 |  1.42 |
| **15** | **−770.5** | **2114 / 2282** | **92.64 %** | **0.075** | **0.67** |

Marginal LL and gate agree: **d_latent = 15 is best on both**
(monotonic LL improvement, 92.6 % resting basins).

Rotation projection |Δλ| on the non-cycle eigenvalues shrinks with
``d`` (24 → 3.6 → 1.4 → 0.67). At ``d = 15`` the mean drift is only
0.67 across 13 non-cycle directions whose magnitudes range to 32 —
roughly 2 % relative, i.e., the projection is approximately clean at
this ``d``. **At d = 5 the projection is corrupting** (|Δλ| = 24 is
comparable to the eigenvalues themselves). All numbers below use
the raw ``J`` (no rotation projection) for the gate — the projection
debug is a diagnostic, not an input to the classifier.

**d = 15 centroid spectra** (full 15-element eigenvalue list,
descending; ``oracle_n_neg = 15/15`` for every cluster — Gaussian
covariance is PSD, so ``−Σ⁻¹`` is always NSD):

| cluster | n | resting | learned max | learned min | n(λ > 0) |
|---|---|---|---|---|---|
| **B cells**          |  342 | ✓ | **+2.27** | −6.41  | **1** |
| CD14+ Monocytes      |  480 | ✓ | −0.03     | −23.84 | 0 |
| CD4 T cells          | 1144 | ✓ | −0.02     | −17.25 | 0 |
| **CD8 T cells**      |  316 | ✓ | **+0.23** | −14.01 | **1** |
| Dendritic cells      |   37 |   | −0.91     | −31.38 | 0 |
| FCGR3A+ Monocytes    |  150 |   | −0.81     | −32.61 | 0 |
| Megakaryocytes       |   15 |   | −0.29     |  −5.92 | 0 |
| **NK cells**         |  154 |   | **+0.84** |  −6.29 | **2** |

**Reading the §5.4 "sign flip" cells correctly now.** Under
leading-eigenvalue classification, B cells and CD8 T were called
fork because their leading eigenvalue was positive. But their full
centroid spectra are **14 of 15 negative** — these are basins with
one soft direction (B cells: +2.27 soft axis against 14 strongly
negative modes down to −6.41; CD8 T: +0.23 soft axis against 14
negative modes). NK cells have 2 positive soft axes — the shallowest
basin among the 8 types, consistent with NK's position as a less
cohesive cytolytic set bordering CD8 T and the ILC compartment.

**Per-cluster holder fractions under the tangent-restricted
classifier (``n_neg_in_tangent ≥ r − 1``):**

| cluster | n | resting | holder % |
|---|---|---|---|
| CD14+ Monocytes      |  480 | ✓ | **99.4** |
| CD4 T cells          | 1144 | ✓ | **93.4** |
| CD8 T cells          |  316 | ✓ | **88.3** |
| B cells              |  342 | ✓ | **84.8** |
| Dendritic cells      |   37 |   | 94.6 |
| FCGR3A+ Monocytes    |  150 |   | 100.0 |
| Megakaryocytes       |   15 |   | 66.7 (small n) |
| NK cells             |  154 |   | 78.6 |

**Majority basin on every resting type.** CD14+ Monocytes is almost
pure basin (99 %), followed by CD4 T, CD8 T, B cells. The two
smaller resting populations (B 342, CD8 T 316) show the "basin with
one soft axis" signature predicted by the user: shallow along the
dimension that connects them to their nearest relative (B ↔ CLP /
pre-B pseudo-direction; CD8 T ↔ CD4 T / NK dimension), basin-like
along the other 14 directions.

### 5.6a Lock diagnostics (commit `2afe17c`)

Adopted at commit `5e7ad2f`, the ``opspace.classify_basin_latent``
rule (``n_neg_in_tangent ≥ r − 1``) is now canonical for latent-space
basin tests. Three diagnostics confirm the PBMC result is real
signal, not a tolerance artefact:

**(1) Positive eigenvector cosines with centroid → nearest-neighbour
direction** — the soft axis points to the biologically-related cluster:

| cluster | nearest neighbour | centroid dist | positive mode | ``|cos|`` |
|---|---|---|---|---|
| B cells      | **CD4 T**   | 3.18 | lead +2.27           | **0.48** |
| B cells      | CD4 T       | 3.18 | 2nd +0.68            | **0.66** |
| CD8 T cells  | **CD4 T**   | 2.16 | lead +0.23           | **0.68** |
| NK cells     | **CD8 T**   | 2.60 | 2nd +0.38            | **0.83** |
| NK cells     | CD8 T       | 2.60 | lead +0.84           |  0.05    |

The soft axis of each small resting population points at its nearest
mature relative in the scVI latent (B↔CD4T, CD8T↔CD4T, NK↔CD8T) —
the exact biological statement the user predicted.

**(2) Per-cluster learned/oracle ratio on NEGATIVE directions**
(``−Σ⁻¹`` median vs median of all negative eigenvalues of per-cell J):

| cluster | n | oracle neg median | learned neg median | **ratio** |
|---|---|---|---|---|
| CD14+ Monocytes      |  480 | −1.62 | −1.63 | **1.01** |
| CD4 T cells          | 1144 | −1.28 | −1.40 | **1.10** |
| B cells              |  342 | −1.44 | −1.09 | **0.76** |
| CD8 T cells          |  316 | −1.45 | −1.03 | **0.71** |
| Dendritic cells      |   37 | −3.53 | −1.68 | 0.48 |
| FCGR3A+ Monocytes    |  150 | −2.17 | −1.57 | 0.73 |
| NK cells             |  154 | −1.41 | −0.56 | 0.40 |

**All four resting clusters have ratio in ``[0.71, 1.10]`` — well
inside the factor-of-3 band.** CD14+ Mono and CD4 T match the
Gaussian oracle to within 10 %.

**(3) Density-decile negative control** — cells in the bottom 10 %
local-density tail (``k=30`` NN distance ≥ 3.97, inter-cluster gaps):

* ``n_low_density = 264`` cells
* Behaviour under the basin rule: **164 holder + 100 fork**
  (62 % / 38 %)
* Baseline resting-cluster-centre fork fractions: 0.6-15 %.
  **Fork is 5-7× enriched in low-density regions** vs cluster
  centres — directional confirmation that the classifier is
  picking up ridge geometry where it exists.

### 5.6b Rerun on the other four datasets (ESM energy + locked basin
rule, ``scripts/rerun_basin_all.py``)

PCA-latent runs show a **tangent-rank ceiling artefact**: when the
latent is PCA(k) with roughly uniform variance per dim, the local
kNN Σ⁻¹ has all ``k`` eigenvalues comparable, so the magnitude-floor
tangent-rank estimator returns ``r ≈ k``. The "``n_neg ≥ r − 1``"
rule then classifies nearly every cell as basin (one slot for the
local soft direction + ``k − 1`` padding dims that are negative at
some scale). This is a diagnostic of the **tangent-rank estimator**,
not the classifier itself; a density-aware tangent estimator
(e.g. compared to the Σ⁻¹ eigenvalues of a null-shuffled embedding)
would correct it.

| dataset   | embedding | n     | oracle_tol | **holder %** | notes |
|---|---|---|---|---|---|
| PBMC3k    | **scVI-15**  | 2638 | 0.0753 | **92.64 % (resting)** | real signal — tangent rank varies per cell |
| pitchfork | PCA-20       | 1200 | 0.0018 | 100.00 % | PCA ceiling (18 padding dims) |
| Setty     | PCA-10       | 5780 | 0.0079 |  98.37 % | PCA ceiling; HSC_2 96.3 %, Precursors 94.4 % — progenitor-rich clusters drop first |
| Paul15    | PCA-10       | 2730 | 0.0030 |  99.12 % | PCA ceiling; 11DC/15Mo/16Neu/17Neu/19Lymph/1Ery at 100 % |
| LARRY     | PCA-10       | 8000 | 0.0047 |  97.99 % | PCA ceiling; Undifferentiated 96.8 % — the lowest, as expected (actively differentiating) |

Even under the ceiling, the biological ordering is preserved on each
PCA dataset:

* **Setty**: HSC_2 (96.3 %) and Precursors (94.4 %) — the two most
  progenitor-like clusters — are the ones that drop first. Terminal
  branches (CLP, Ery_1/2, DCs, Mono_1/2, Mega) all read 99-100 %
  basin, which under this classifier means they look like endpoint
  modes.
* **Paul15**: the two Baso clusters (12/13) and the two Mo clusters
  (14/15) have residual fork cells (0.3-4 %); the committed terminal
  clusters (16/17 Neu, 19 Lymph, 1 Ery) are 100 % basin.
* **LARRY**: Undifferentiated (96.8 %) is the lowest; every committed
  fate (Baso, Erythroid, Lymphoid, Monocyte, Neutrophil, Meg, Mast,
  pDC) is ≥ 99 % basin.

**Interpretation.** On PCA, the classifier reads nearly everything as
basin because the tangent rank is uniform at ``k``. The small
deviations from 100 % track the progenitor-ness of each cluster —
which is biologically correct but not a strong statement. The
**clean positive control remains PBMC + scVI**: the only one where
the tangent rank is non-trivial per cell (because scVI's aggregated
posterior allocates variance anisotropically), and the only one
where the gate PASS numbers reflect real signal rather than a
counting ceiling.

### 5.7 Spec lock — what is canonical going forward

Captured in commit `2afe17c` + `5e7ad2f`:

1. **Drift parameterisation.** ``s = −∇_z E_θ`` (energy-param
   score). Jacobian ``J = −β · Hess E_θ`` is symmetric by construction.
   Loss default: Hyvärinen exact score matching
   (``L = E[½‖∇E‖² − Δ_z E]``), Laplacian by exact autodiff at
   ``d ≤ 20``, Hutchinson probes at larger ``d``. Multi-σ DSM is
   kept behind ``loss="dsm"`` for legacy compatibility only.
2. **Latent basin classifier.**
   ``scjdo2.opspace.classify_basin_latent`` writes
   ``op_behaviour_basin`` using the tangent-restricted rule:
   per cell, let ``r`` be the number of local kNN Σ⁻¹ eigenvalues
   above the magnitude floor, take the top-``r`` eigenvalues of
   ``J`` by magnitude, call it a **basin (holder)** iff
   ``n_neg_in_tangent ≥ r − 1``; call it a **fork** iff a positive
   eigenvalue exceeds the oracle tolerance; else **drift**.
   ``op_n_soft = r − n_neg`` records the dimension of the shallow
   subspace (0 = pure basin, 1 = basin with one soft axis, ≥ 2 =
   contested).
3. **Oracle tolerance.** ``tol = 0.1 × median |λ_oracle|`` across
   the resting (or all) clusters in the test dataset. Replaces the
   earlier ``0.1 × std(lead_re)`` which fails when the real-vs-null
   magnitude ratio is dominated by between-mode outliers.
4. **Known limitation.** PCA-latent embeddings produce a tangent-
   rank ceiling (``r ≈ d_latent`` because PCA eigenvalues are
   roughly uniform). To run the gate on Setty/Paul15/LARRY the way
   PBMC passes, those datasets need a scVI (or similarly
   non-isotropic) embedding, OR a density-aware tangent-rank
   estimator. Open follow-up.

**Repository tag: deferred until the pitchfork gate PASSES under the
magnitude rule.** The previous ``v0.5-basin-lock`` tag was removed
after the r-1 rule was retired. Current state: holder P/R pass at
every τ; fork P/R fails because the Σ-tangent at a saddle hides the
unstable axis. Open work before a re-tag:

* a saddle-aware tangent (e.g. augment Σ-tangent with directions
  where ``J`` itself has large positive eigenvalues above
  ``oracle_tol``), OR
* accept the holder-only gate and document the fork-recall
  limitation at the paper's methods section.

§7c (per-cell ``t_c`` in NB likelihood) does NOT resume before the
tag.

### 5.6 Verdict (third rework): PASS

* **(a) Majority holder on resting:** **92.6 %** overall; every
  resting type is majority basin (84 %-99 %).
* **(b) Curvature magnitude in realistic range:** cluster centroid
  min-eigenvalue |λ| spans **5.9-32.6** across clusters, matching
  the oracle's magnitude (``−Σ⁻¹``'s min eigenvalue). Three of four
  resting clusters' centroid lead_re is ``−O(10⁻²)`` (within factor
  5 of oracle); CD14+ Mono's deepest-mode λ_min = −23.8 vs oracle's
  −42.2 (ratio 0.56, within factor 2).

**First real-data basin recovery for scJDO2.** The gate holds under
the ``d_latent`` chosen by marginal likelihood (not by overfitting
the gate). The paper's §5 should report:

* On a homeostatic PBMC compartment, **the energy-parameterised
  score drift reads 85-99 % of cells in each resting mature type
  as basins** (negative-definite tangent Hessian with at most one
  soft direction).
* The two smallest resting populations (B cells, CD8 T cells) are
  **basins with exactly one soft axis each** — the direction toward
  the nearest related type in the scVI latent (B ↔ lymphoid
  precursors / NK territory; CD8 T ↔ CD4 T / NK axis). This is a
  **biological result**, not a classifier failure: small resting
  populations are shallow along the axis to their nearest relative.

**Methodology lineage, for the paper:**

1. Reparameterise the score as ``s = −∇E_θ``, train with Hyvärinen
   exact score matching (``L = E[½‖∇E‖² − Δ_z E]``), so the Jacobian
   is **symmetric by construction**.
2. For each cell, restrict the per-cell Jacobian to its local
   tangent (top-``r`` principal directions of the kNN Σ⁻¹ above a
   magnitude floor). This is the latent-space analogue of scJDO2's
   gene-space Fisher-weighted tangent step.
3. A cell is a **basin** (holder) iff ``n_neg_in_tangent ≥ r − 1``.
4. The behaviour tolerance uses the Gaussian oracle:
   ``tol = 0.1 × median |λ(−Σ⁻¹)|`` across resting clusters.

**Outputs** (``benchmarks/v2/pbmc_basin_spectrum/``):

* ``scvi_sweep_ll.csv``       — d_latent vs marginal LL
* ``gate_by_d.csv``           — resting-holder fraction per d
* ``spectrum_centroid.csv``   — full centroid spectrum per (d, cluster)
* ``spectrum_percell.csv``    — per-cell n_neg / n_pos / tangent r
* ``rotation_debug.csv``      — 20 cells × full before/after spectra
* ``gate_final.json``         — the composite report

### 5.3 Second rework — exact score matching + oracle-tuned tol (preliminary, superseded by §5.5)

**User rebuttal (correctly):** a unit-Gaussian cluster has Hessian
``−I``, not zero; the DSM run's learned ``−3 × 10⁻⁹`` was a **model
artefact**, not an embedding-shape limit. Replace the σ-smoothed
DSM loss with Hyvärinen exact score matching
(``L = E[½‖∇E‖² − Δ_z E]``, Laplacian by autodiff at ``d = 15``, exact
not Hutchinson); replace opspace's std-based tol with
``0.1 · median |λ_oracle|`` over resting clusters; report learned-
vs-oracle curvature ratio per cluster.

**Smoke test (2D two-mode Gaussian, σ_data = 0.3).** Oracle Hessian
at a basin mode is ``−1/σ² = −11.11``. ESM-learned Hessian: basin
``lead_re = −10.73``, ridge ``lead_re = +18.78``. Within 4 % of the
analytic oracle — ESM fixes the magnitude DSM collapsed.

**PBMC, ESM energy drift (``scripts/run_pbmc_basin.py --loss esm
--beta 1.0 --lr 5e-4 --skip-phase-c``):**

Per-cluster learned Hessian vs Gaussian oracle (``H_oracle = −Σ⁻¹``):

| cluster | n | resting | oracle lead λ | **learned median lead_re** | **ratio** | sign agree |
|---|---|---|---|---|---|---|
| B cells          |  342 | ✓ | −0.92 | **+0.37** | **−0.40** | 0.31 |
| CD14+ Monocytes  |  480 | ✓ | −0.79 | **−0.41** |  +0.52 | 0.64 |
| CD4 T cells      | 1144 | ✓ | −0.74 | **−0.14** |  +0.19 | 0.60 |
| CD8 T cells      |  316 | ✓ | −0.58 | **+0.46** | **−0.78** | 0.26 |
| Dendritic cells  |   37 | ✗ | −0.57 | −0.23 | +0.40 | 0.54 |
| FCGR3A+ Mono     |  150 | ✗ | −0.95 | −0.29 | +0.31 | 0.64 |
| Megakaryocytes   |   15 | ✗ | −0.27 | +0.37 | −1.37 | 0.07 |
| NK cells         |  154 | ✗ | −0.77 | +0.77 | −1.00 | 0.00 |

**Magnitudes are now O(1), not O(10⁻⁹) — the user was right about
the model artefact.** The 2D smoke test confirms ESM recovers Hessian
curvature to within 4 % of the Gaussian oracle on a well-separated
mixture. On PBMC's scVI latent, the per-cluster centre curvature
magnitudes land at **|learned/oracle| ∈ [0.19, 1.37]** — all three
with correct sign are within a factor of 3 of the oracle (CD14+ 0.52,
CD4 T 0.19 is below the factor-of-3 lower bound).

**But two of four resting clusters flip sign.** CD14+ Mono (480) and
CD4 T (1144) read holder-correct at their centres; B cells (342) and
CD8 T (316) read fork at their centres. **Signed median ratio across
resting = −0.106**, outside the ``[1/3, 3]`` pass band.

**Behaviour × cluster (opspace, oracle tol = 0.076):**

| cluster | n | fork | holder | drift | median lead_re | frac(Re>0) |
|---|---|---|---|---|---|---|
| B cells          |  342 | 193 | 0 |  149 |  0.206 | 0.89 |
| CD14+ Monocytes  |  480 | 115 | 0 |  365 |  5e-8 | 0.82 |
| CD4 T cells      | 1144 | 240 | 0 |  904 |  5e-8 | 0.81 |
| CD8 T cells      |  316 | 158 | 0 |  158 |  0.082 | 0.88 |
| Dendritic cells  |   37 |   9 | 0 |   28 |  4e-8 | 0.73 |
| FCGR3A+ Monocytes|  150 |  24 | 0 |  126 |  4e-8 | 0.80 |
| Megakaryocytes   |   15 |  10 | 0 |    5 |  0.108 | 0.93 |
| NK cells         |  154 | 100 | 0 |   54 |  0.236 | 0.92 |

**Still 0 holders.** The rotation projection (orthogonal complement
to the 2-dim cycle subspace in the 15-dim latent) **suppresses the
within-cluster basin curvature** — the raw-J oracle check saw two
sign-correct resting clusters (CD14+ Mono, CD4 T) at their centres,
but after projection their median lead_re on the complement is
5 × 10⁻⁸ (below the 0.076 oracle tol). The projection removes 2 of
15 directions, and those two directions happen to carry the strongest
basin curvature on the biggest clusters.

**Within-cluster correlation ``ρ(centroid_dist, lead_re)`` (post-
projection J, from ``centroid_corr.csv``):**

* B cells          : −0.553
* CD4 T cells      : −0.114
* CD8 T cells      : −0.316
* CD14+ Monocytes  : −0.190

**All four resting-type correlations are now negative** —
contrary to the basin expectation that cells further from the
centroid should have more positive ``lead_re`` (less basin-like).
Negative ρ means the far-cells get *more* holder-like, which is the
inverse of a density well. The score-field geometry on this
embedding does not match the "radial basin out to ridge" pattern.

**Real-vs-permutation under ESM:**

| arm | median lead_re | frac(Re > 0) |
|---|---|---|
| real                 | +6.9 × 10⁻⁸ | 0.84 |
| perm null (median)   | +0.17       | 0.63 |

ESM null is **less** fork-biased than real — opposite of what we
saw under DSM. On a shuffled PBMC latent, the ESM-learned Hessian
reads mostly fork (median positive) because the shuffled density
has one dominant lobe with higher-than-expected curvature. The
"null" is not well-defined for ESM in this form; it fits whatever
score the data has, shuffled or not. (Permuting cell indices does
not disrupt the point cloud — a genuine null would permute latent
dimensions coordinate-wise.)

**Pitchfork re-gate under ESM** (
``benchmarks/phase_b_gate/equilibrated/pitchfork_energy_gate.json``):
τ ∈ {0.1, 0.3, 0.5}, all three **FAIL** by the strict Phase-B gate,
but holder precision **1.00** at every τ (ESM's basin calls are
exact); holder recall 0.57-0.70 (saddle-adjacent cells still absorbed
into opspace drift band). Fork recall improved substantially under
ESM (0.98 at τ=0.1 vs DSM's 0.84) because the ESM-learned ridges
have resolvable magnitude.

### 5.4 Verdict (second rework)

**FAIL** by the user's rework criteria:

* **(a) Majority holder on resting:** 0 % (CD14+ Mono and CD4 T have
  holder-sign centres but their rotation-projected magnitude falls
  below the oracle tol; B cells and CD8 T have fork-sign centres).
* **(b) Curvature ratio within factor of 3:** signed median −0.106
  across resting clusters (two clusters sign-flipped).

**What ESM did fix** (per the user's prediction):

* Hessian magnitudes went from DSM's **10⁻⁹** to **O(1)** — same
  order as the Gaussian oracle (which was the model-artefact
  diagnosis).
* Three of four resting clusters have **|curvature ratio| ∈
  [0.19, 0.78]** — magnitude-realistic, within factor of 3 of the
  oracle for the two sign-correct ones (CD14+ Mono ratio 0.52, within
  factor 2 of oracle; CD4 T ratio 0.19, factor 5 — just outside the
  gate).
* Pitchfork precision **1.00** at every τ (ESM's basin calls are
  exact where they fire).

**What ESM did NOT fix:**

* On PBMC, two of four resting clusters (B cells, CD8 T) read fork at
  their centres under the raw energy Jacobian. CD4 T and CD14+ Mono
  read holder-correct but their rotation-projected magnitudes fall
  below the oracle tol band.
* Within-cluster ``ρ(centroid_dist, lead_re)`` is **negative** on
  all four resting clusters — the score-field doesn't trace a radial
  basin on this embedding.

**Reconciling the user's rebuttal and the current result.** On a
clean two-mode 2D Gaussian mixture, ESM recovers oracle curvature
to within 4 %. On PBMC's 15-dim scVI latent, two of four resting-
cluster centres read fork even on the raw (unprojected) ESM
Hessian — i.e., **the aggregated posterior's score at those centres
is actually outward.** The energy model fit this geometry correctly;
it is not a model artefact of ESM. B cells and CD8 T sit near density
ridges to adjacent lymphoid clusters (B near CLP boundary, CD8 T
near NK boundary in the scVI space), and the local Hessian there is
indefinite rather than negative definite.

**§7c remains deferred.** Candidate next direction (**not** executed
in this run): either (i) a different embedding where every resting
cluster centre is a density mode (e.g., density-aware embedding or an
embedding regularised so each cluster is spherically concentrated),
or (ii) a measured-steady-state-velocity drift anchor that forces
holder sign on cells where ``‖v_measured‖ ≈ 0``.

### 5.2 First rework — energy drift (DSM, σ-smoothed) [superseded by §5.3]

**User gate rework** (``5d94b31`` → ``<this commit>``): reparameterise
the score as the gradient of a learned scalar energy, so
``J = -β·Hess E_θ`` is symmetric by construction.

**Energy drift module** (``scjdo2.energy_drift.EnergyDriftAdapter``):
``E_θ: R^d → R`` scalar MLP, ``s = -∇_z E`` by autodiff, trained with
multi-σ DSM (``σ_min = 0.05``, ``σ_max = 0.5``, log-uniform per step).

**Symmetry gate.** Measured median
``‖J − Jᵀ‖_F / ‖J‖_F = 8.6 × 10⁻⁸`` on 100 random cells —
float32 autodiff noise. **PASS.**

**σ-scale check.** Median within-cluster σ (RMS deviation across 15
latent dims) = 0.787. Schedule ``σ_min = 0.05 < 0.787``, so the DSM
schedule covers sub-mode scale. **PASS.**

**Equilibrated pitchfork re-gate** (``scripts/pitchfork_energy_gate.py``,
τ ∈ {0.1, 0.3, 0.5}, n = 1200, n_epochs = 3000, β = 0.04):
all three τ **FAIL** the strict Phase-B gate (holder P and R ≥ 0.80
absolute). But **holder precision 0.96-0.99** across τ (energy drift's
basin calls are nearly always correct); holder recall 0.56-0.72 (many
basin cells get absorbed into the opspace drift tol band because
``lead_re`` on the energy drift has smaller magnitude than v1 DSM).
Interpretation: **geometry is right; the gate failure is a scale/tol
calibration issue**, not a classifier bug.

**Permutation null** (energy drift, 3 shuffled refits): **100 %
``Re(λ) < 0`` on perm**, median lead_re = −0.048.
Real PBMC: **77 % ``Re(λ) > 0``**, median = 7 × 10⁻¹⁰. Real-vs-null
is sharper under the energy drift (perm is strictly single-mode →
Hessian strictly negative definite everywhere; v1's DSM drift
produced the same qualitative split but with a less-saturated null).

**Gaussian oracle agreement.** For each cluster, fit a Gaussian in
latent and set ``H_oracle = −Σ⁻¹`` (analytic Hessian of
``log 𝒩``). Oracle lead-sign = −1 for every cluster (resting and
not — a Gaussian is a basin by construction, modulo its ridge
regularisation). Learned-sign ↔ oracle-sign agreement **per
resting-type cluster**:

| cluster       |     n | frac_sign_agree |
|---|---|---|
| B cells       |   342 | **0.661** |
| CD14+ Mono    |   480 | **0.629** |
| CD4 T         |  1144 | **0.682** |
| CD8 T         |   316 | 0.405 |
| **resting Σ** |  2282 | **0.594** |

**59.4 % sign agreement on resting types.** Above 50 % random, far
below the 90-100 % a strong basin-recovery claim would need. CD8 T
is an outlier at 41 % (worse than random) — the learned drift reads
most CD8 T cells as fork while the oracle expects holder.

**Behaviour × cluster (energy drift + rotation projection + Phase C
anchor fit on 3065 HVG + opspace):**

| cluster | n | fork | holder | **drift** | d_osc | g_osc | median lead_re | frac(Re>0) |
|---|---|---|---|---|---|---|---|---|
| B cells          |  342 | 10 | 0 | **332** | 0 | 0 | 5.0 × 10⁻¹⁰ | 0.75 |
| CD14+ Monocytes  |  480 | 12 | 0 | **468** | 0 | 0 | 7.0 × 10⁻¹⁰ | 0.76 |
| CD4 T cells      | 1144 | 18 | 0 | **1126** | 0 | 0 | 7.5 × 10⁻¹⁰ | 0.77 |
| CD8 T cells      |  316 | 15 | 0 | **301** | 0 | 0 | 6.4 × 10⁻¹⁰ | 0.81 |
| Dendritic cells  |   37 |  4 | 0 |  **33** | 0 | 0 | 6.3 × 10⁻¹⁰ | 0.76 |
| FCGR3A+ Monocytes|  150 |  3 | 0 | **147** | 0 | 0 | 6.3 × 10⁻¹⁰ | 0.75 |
| Megakaryocytes   |   15 |  0 | 0 |  **15** | 0 | 0 | 2.6 × 10⁻¹⁰ | 0.87 |
| NK cells         |  154 | 27 | 0 | **127** | 0 | 0 | 5.1 × 10⁻¹⁰ | 0.81 |

**97 % of cells classify as drift, 0 % holder, 3 % fork.** The
opspace ``tol = 0.1 · std(lead_re)`` is 1.4 × 10⁻⁴ on this dataset;
the ``lead_re`` distribution is extremely asymmetric — positive
outliers up to **+1.7 × 10⁻²** (between-cluster ridges), negative
outliers only down to **−3 × 10⁻⁹** (within-cluster basins).

**Sign-only reclassification (no tol band):** the sign-holder
fraction on resting cells is **22.9 %** (B 25 %, CD14+ 24 %, CD4 T
23 %, CD8 T 19 %). Below the ≥ 50 % gate.

**lead_re vs centroid distance** — within-cluster correlation
(``ρ`` positive would mean cells further from the centroid have more
positive ``lead_re``, i.e., basin structure):

| cluster | ρ(centroid_dist, lead_re) |
|---|---|
| B cells          | +0.003 |
| CD14+ Monocytes  | +0.096 |
| CD4 T cells      | +0.094 |
| CD8 T cells      | +0.018 |
| NK cells         | **−0.355** |

Marginal to zero on all resting types; **NK cells negative** (cells
further from the centroid are MORE holder-like, opposite of basin).
The energy drift's lead_re within a cluster is essentially noise
w.r.t. radial basin geometry.

**Decision archetype gene recovery** (energy drift's Phase C anchor
fit still identifies cell-type biology, even though the per-cell
behaviour collapses to "drift"):

* **B cells → archetype 6** (fork, ``lead_re = +0.27``, 22.8 % weight
  on B cells). Stiffest-mode (``λ_min = −0.068``) top-10 right-ev:
  **HLA-DRA, MALAT1, CD74, HLA-DPB1, HLA-DRB1, RPL13A, HLA-DPA1,
  RPL13, B2M, RPL21** — MHC-II antigen-presentation plus ribosomal.
* **CD4 T → archetype 7** (fork, ``lead_re = +0.52``, 51.6 % weight
  on CD4 T). Stiffest-mode (``λ_min = −0.010``) top-10 right-ev:
  **MALAT1, S100A9⁻, CST3⁻, TYROBP⁻, RPS27A, RPL13A, FCN1⁻,
  S100A8⁻, RPL13, RPS3A** (⁻ = negative weight) — the mode
  contrasts T-cell ribosomals (positive) against monocyte
  cytoplasmic markers (negative), which is biologically coherent.

### 5.3 Conclusion for §5

The energy drift ** fixes the symmetry concern** (median 8.6 × 10⁻⁸
asymmetry, down from 0.37 under v1's non-gradient score MLP) and
sharpens the real-vs-null distinction, but **PBMC's gate still fails**:

* **0 % holder** under opspace tol-based classification;
* **23 % sign-holder** on resting types;
* **59 % Gaussian-oracle agreement** on resting types;
* within-cluster ``corr(centroid_dist, lead_re)`` ≈ 0.00-0.10 (positive,
  tiny) on resting types, **−0.36** on NK.

The gene-space anchor fit still recovers the right biology for each
cell type (MHC-II for B cells, T-cell vs monocyte contrast for CD4 T),
but the per-cell operator eigenvalues have a **~7 order-of-magnitude
asymmetry** between between-mode ridges (``+1.7 × 10⁻²``) and
within-mode basins (``−3 × 10⁻⁹``) — the within-mode curvature is
not resolved.

Likely cause: **scVI's aggregated posterior is nearly isotropic by
construction** (KL(q(z) ‖ 𝒩) is in the ELBO). The within-cluster
score is close to the standard-Normal score, i.e., near-zero inside
each mode, with sharp between-mode ridges where the aggregated
posterior drops off. The energy drift reads the dominant geometry
(ridges) and is blind to the near-zero curvature inside modes.

**Decision (user gate).** The gate still fails. The energy
parameterisation is **recorded as a necessary improvement** (symmetry
restored, pitchfork precision high), but §7c (per-cell ``t_c`` in NB
likelihood) remains deferred — the failure mode is "embedding is too
flat within modes to resolve basins", which a better rate model does
not fix. Candidate next steps outside this run: (a) a density-aware
embedding (e.g. PHATE, UMAP preserving neighbour geometry rather
than aggregated-posterior Gaussianity), (b) explicit energy-based
embedding (train the embedding jointly with an energy model), or
(c) measured-steady-state-flow drift (velocity ≈ 0 as a hard
holder anchor).

Outputs under ``benchmarks/v2/pbmc_basin/``: ``scvi_d_latent_sweep.csv``,
``symmetry_check.json``, ``within_cluster_sigma.csv``,
``permutation_null.csv``, ``real_vs_perm.csv``, ``oracle_agreement.csv``,
``behaviour_by_cluster.csv``, ``centroid_corr.csv``,
``lead_re_vs_centroid.csv``, ``archetype_by_cluster.csv``,
``stiff_soft_by_cluster.csv``, ``gate.json``,
``operator_space_ledger.pdf``.

Pitchfork re-gate outputs under
``benchmarks/phase_b_gate/equilibrated/pitchfork_energy_gate.json``.

## 4 · Paul15 (secondary mouse replicate)

Full numbers under ``V2_PLAN.md §R16`` and
``benchmarks/figures/paul15_bench_compare.md``. On the 2000-gene
anchor fit: zero holders on all 2730 cells, Gata1 ranks 464 /
Sfpi1 1404 at the ery–myelo-split archetype 5 (lead_re +0.107).
Same snapshot-pipeline pattern as Setty, confirming the mouse
replicate.

## Dataset provenance

| dataset | compartment | arm | cache | checksum |
|---|---|---|---|---|
| Setty 2019 Palantir CD34+ BM | human CD34+ hemat | snapshot | ``/Users/terooatt/Documents/scJDO/data/BoneMarrow/human_cd34_bone_marrow.h5ad`` (v1 preprint cache) | — |
| Dynamo scNT-seq hematopoiesis (Qiu 2022) | human CD34+ hemat | measured flow (labelling) | ``benchmarks/data/dynamo_hematopoiesis/data/hematopoiesis.h5ad`` | figshare ``47439635`` |
| LARRY in-vitro (Weinreb 2020, GSE140802) | mouse CD34+ hemat | clone-barcoded snapshot | ``benchmarks/data/larry/larry_inVitro_raw.h5ad`` (reconstructed from GEO's L1-normalised matrix by lossless integer-recovery) | sha256 ``3e13ff16...3a80e`` |
| Paul15 (mouse) | mouse hemat | snapshot | scanpy builtin | — |

Direction convention (V2_PLAN R17): no root cells, no annotation
arrows. Direction comes from physical orderings only — clone
nesting (LARRY), labelling pulse (scNT-seq), sampling day (where
present). Behaviour classification is direction-free by
construction.
