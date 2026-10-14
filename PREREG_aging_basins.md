# Pre-registration — aging basins arm (`aging-basins` branch)

**Branch:** `aging-basins`  (parent: `main @ cb9298f`,
tag `v0.5-basin-lock`)
**Pre-reg timestamp:** 2026-10-14, before any run on real data.
**Classification rule:** `scjdo2.opspace.classify_basin_latent(rule="lead_eig")`,
tol = 0.1 × std(lead_re), basin_index = −lead_re. **No changes to
the lock.** No archetype decomposition, no pseudotime, no prior.

---

## 1 — Datasets

| source | tissue | n_cells | age groups | donors | notes |
|---|---|---|---|---|---|
| Tabula Muris Senis (TMS), **cellxgene_census 2024-07-01**, Lung 10x | lung  | 24 540 | 3 m (young), 18-24 m (old) | multiple | dataset_id `0fb7916e-7a68-4a4c-a441-3ab3989f29a7` |
| TMS, cellxgene_census 2024-07-01, Bone marrow 10x | marrow | 40 220 | 3 m (young), 18-24 m (old) | multiple | dataset_id `0bd1a1de-3aee-40e0-b2ec-86c7a30c7149` |
| PBMC aging human cohort | *n/a* | *n/a* | *n/a* | *n/a* | **attempted via cellxgene_census + sfaira**; if no cohort with raw UMIs + donor IDs ships, the arm runs with the two TMS tissues only. Status recorded in RESULT.md. |

Pre-stated choice of tissues: lung (user-required) + bone marrow
(largest 10x TMS dataset after lung, broad cell-type diversity).
PBMC human aging deferred unless a raw-count + donor-ID cohort is
accessible in a few CPU-hours; the prereg does not promise it.

## 2 — Preprocessing & embedding

* **Per-tissue** raw UMI matrix pulled from the census; cell-type
  annotation = `cell_type_ontology_term_id`'s human-readable label
  (TMS `free_annotation` where available).
* Age groups:
  * **young** = 3 months
  * **old**   = 18-24 months (TMS "aged" labels)
* **Downsampling** (BEFORE any model fit): for each type, downsample
  young cells to the number of old cells per donor; types with
  < 200 cells in **either** group after downsampling are dropped.
  (Inclusion rule, pre-stated: ``≥ 200 cells in BOTH groups``.)
* **Shared scArches/scVI space.** scVI trained on young ONLY
  (ensures the shared basis reflects young geometry); old mapped
  in via scvi-tools' `online_update` / `query_mode`. Fallback: a
  single scVI fit on young+old with age as a batch covariate (this
  matches the shared space's statistical intent). Which path is
  used is recorded in RESULT.md.
* **Standalone** scVI fits per age group — independent latent per
  age, same 2 000 HVG from pooled data, same `d_latent = 15` as the
  PBMC lock run.
* **Donors as replicates:** per-donor shift estimates are reported
  wherever donor information is available in `obs`.

## 3 — Operators + per-type measurements

Per `(tissue, cell_type, age_group)`:

| field | source |
|---|---|
| `n` | number of cells |
| spectrum | eigenvalues of mean per-cell latent `J` across the 30 cells nearest the type centroid |
| `u` (soft axis) | most-positive eigenvector of that centroid `J` |
| `oracle_curvature_ratio` | learned median negative-direction eigenvalue / `−Σ⁻¹` median negative eigenvalue |
| `basin_index` | `−lead_re` per cell |
| `op_behaviour_basin` | `classify_basin_latent(rule="lead_eig")` |

Both the shared-space and the standalone fit are reported side by
side. Types whose basin exists in ONLY ONE representation are
flagged `correction_artifact` and NOT counted toward the test
pass fractions.

## 4 — Test 1 — isotropy (per type)

**Hypothesis:** aging increases operator stiffness anisotropically
(some directions soften more than others). Null = isotropic
loosening (every direction's ratio ~ equal).

**Statistic:** per matched direction `i`, ratio
`r_i = λ_i^(old) / λ_i^(young)` where directions are matched by
eigenvector cosine (Hungarian / greedy, highest-cosine pairs).
**CV(ratio across directions)** is the per-type statistic.

**Null:** 1 000 shuffles of the ratios across directions. Pass =
observed CV above the 95th percentile of the shuffled distribution.

Reporting:
* `CV_r` per type
* `null_q95` per type
* `isotropy_pass` boolean per type
* **fraction of eligible types passing**

## 5 — Test 2 — anticipation (per type)

**Hypothesis:** the direction of aging displacement `d = old
centroid − young centroid` lies along the young operator's soft
axis `u_young` (anticipation of future state along the slow mode).

**Statistic:** `|cos(u_young, d)|`.

**Null:** 1 000 random half-splits of the young cells of that type
at the old sample size; for each split, compute the displacement
between the two halves and its `|cos|` with `u_young`.
Pass = observed `|cos|` above the 95th percentile of the null.

Additional (reported but not a gate):
* if annotation sub-states enriched in old along `u_young` exist
  per type, report their mean position along `u_young` with error.

Reporting:
* `|cos(u_young, d)|` per type
* `null_q95` per type
* `anticipation_pass` boolean per type
* **fraction of eligible types passing**

## 6 — Interpretation rule (fixed BEFORE any run)

| Test 1 (isotropy) | Test 2 (anticipation) | Outcome |
|---|---|---|
| pass in majority | pass in majority | **directional, anticipatory loss** |
| pass in majority | fails in majority | **specific loosening without anticipation** |
| fails in majority | fails in majority | **isotropic noise increase**, no aging claim beyond stiffness |
| fails in majority | pass in majority | **anticipation without specific loosening** — report, but a 4th case under the rule table was not pre-stated — the exact wording of the outcome will be the row header above with that gloss, recorded under a §4-corner note |

"Majority" = > 50 % of eligible types across both tissues pooled.
Per-tissue majorities also reported.

The outcome text is written into `benchmarks/aging/RESULT.md`
under exactly the row that matches.

## 7 — Figures

Via the new v0.5 API (`scjdo2.plot.figures_v05`):

1. `spectrum_ladder(adata, by="cell_type", groups=("young", "old"))`
   — per-type centroid spectrum side-by-side.
2. `soft_axis_map(adata)` of the young representation with
   per-type **old-centroid displacement arrows** overlaid.
3. `basin_index_by_type(adata, by="cell_type")` — per-type
   distribution faceted by age.
4. `ledger_map(adata)` per age — basin / saddle / drift class per
   cell.

## 8 — Results table stub (filled during the run)

| tissue | cell_type | n_young | n_old | shared or standalone | CV_r | iso_q95 | iso_pass | \|cos(u_y,d)\| | ant_q95 | ant_pass | basin_fraction_young | basin_fraction_old | basin_index_median_young | basin_index_median_old | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lung | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| marrow | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

Totals row at the bottom with pass fractions.

## 9 — Autonomy rules in effect

* No criterion changes after seeing results — the gates in §4 and
  §5 are frozen above.
* Pre-stated numeric rules only — the 200-cell inclusion, the
  1 000-shuffle null, the 95th-percentile pass line are all above.
* Negative outcomes are written up as results — no re-tuning after
  seeing numbers.
* Stop only for irreversible actions or large compute (> few
  CPU-hours).

**No merge to main until the full run + interpretation are
written.** The `aging-basins` branch holds everything.
