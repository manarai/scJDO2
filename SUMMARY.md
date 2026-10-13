# scJDO v2 — summary of hematopoiesis comparisons

The manuscript's primary comparison is on **CD34+ bone marrow**. v1
and v2 are both evaluated on the Setty 2019 Palantir dataset (the
preprint target), with Dynamo scNT-seq hematopoiesis (Qiu 2022) as
the matched measured-flow counterpart on the same compartment —
"snapshot versus measured flow on matched biology". Paul15 remains
as a secondary **mouse** replicate of the snapshot arm.

All reported eigenvalues on Setty are relative (snapshot, no clock);
on Dynamo scNT-seq they are per-hour (`clock_calibrated = True`).

Human gene symbols throughout (``GATA1``, ``SPI1``).

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

**Recovery vs DSM-Setty on GATA1.** On DSM-Setty's decision archetype
(arch 3), GATA1 was **capped with Fisher = 0** (anchor_max_live SNR
cut dropped it); SPI1 was rank 73 responder / 137 driver. On scVelo-
Setty's decision archetype (arch 7), **GATA1 is rank 57 / 54 with
real Fisher mass**, SPI1 is 392 / 227 — the two TFs swap primacy at
the saddle under splicing vs density-score drift. Measured flow via
splicing surfaces erythroid-axis GATA1 at the decision region
where the DSM drift did not. **This is the first clean recovery of
GATA1 as a top-tail driver of a scJDO2 archetype under a drift
other than the DSM score.**

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
