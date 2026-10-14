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

### 2a — Prereg amendment: downsampling direction (2026-10-14, made before any data run governed by it)

§2 says "downsample young cells to the number of old cells per
donor", but young is the smaller group in both TMS tissues, so the
rule cannot execute as written. **Rule as executed:** per
`(donor, cell_type)`, downsample the **larger** age group to the
size of the smaller one, seed 0; drop `(donor, cell_type)` cells
with fewer than 20 cells on either side (bookkeeping floor, not a
gate); then keep only cell types with `≥ 200` cells in **both** age
groups (the §2 inclusion gate, unchanged).

### 2b — Prereg amendment: age labels and downsampling unit (2026-10-14, made before any preprocessing, model fit or statistic on these data)

Context: the TMS fetch (aging/A01, census 2024-07-01, dataset IDs as
§1) has run; only `obs` metadata has been read (age labels, donor
counts). Nothing governed by this amendment has run.

1. **4-week cells are excluded by design** (spec §1; young = 3 months
   per §2). They are 2 508 / 24 540 lung cells (2 donors) and
   3 027 / 40 220 marrow cells (2 donors). They are not "unmapped"
   labels; they are dropped before §2a.
2. **"Old" as executed** = census `development_stage` labels
   `18 month-old stage` and `20 month-old stage and over`. The census
   does not separate 21, 24 and older mice inside the second label, so
   "old" may include mice older than the 18–24 months stated in §2.
   This is recorded, not corrected.
3. **Downsampling unit.** Every TMS donor (mouse) has exactly one age,
   so §2a's per-`(donor, cell_type)` balancing cannot execute.
   **Rule as executed:** per `cell_type`, downsample the larger age
   group to the size of the smaller, seed 0, sampling as evenly across
   that group's donors as their cell counts allow (each donor
   contributes `min(n_donor, quota)` cells, with the quota raised until
   the target is met). Before this, drop `(donor, cell_type)` sets with
   fewer than 20 cells (bookkeeping floor, not a gate). The §2
   inclusion gate is unchanged: `≥ 200` cells in **both** age groups
   after downsampling.
4. **Donor counts** (young / old, 4-week excluded): lung 4 / 10,
   marrow 2 / 13. Donor-level replication of shifts is reported where
   possible (§2), and marrow's two young donors are a stated limit.

### 2c — Prereg amendment: old = 18–24 months; HVG span (2026-10-14, made before any model fit or statistic)

Context: aging/A01 revision 1 ran preprocessing (eligibility counts and
QC only). No model has been fitted and no statistic computed. Neither
change below alters which cell types are eligible (checked: lung 2,
marrow 6, the same types with or without the change).

1. **Old = 18–24 months, as §2 states.** §2b.2 assumed the census label
   `20 month-old stage and over` could not be resolved. It can: TMS
   donor IDs begin with the age in months (`18-…`, `21-…`, `24-…`,
   `30-…`). The label contains 30-month mice (lung 4 of 10 old donors,
   marrow 3 of 13). **Rule as executed:** old = donors whose ID prefix
   is 18, 21 or 24; donors with prefix 30 are excluded before the
   §2b donor floor and downsampling. Young = prefix 3 (3 months); 4-week
   donors (prefix 1) excluded as §2b.1. §2b.2 is superseded.
2. **HVG smoothing span.** `seurat_v3` with `batch_key="donor_id"` fits a
   LOESS per donor; at the default `span=0.3` (and 0.5) it fails with a
   near-singularity for every lung donor contributing fewer than ~100
   cells (7 of 13 donors in revision 1). **Rule as executed, both
   tissues:** `flavor="seurat_v3", n_top_genes=2000,
   batch_key="donor_id", span=1.0`. If this fails, stop; no other
   fallback is pre-specified.

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

**Hypothesis:** aging widens the operator in a direction-coherent
way — directions soft in young remain the softening axis in old,
not isotropically, not random.

### 4a — Prereg amendment (2026-10-14, before any run on real data)

> **Superseded by §4b** (same day, before any data run). Kept for the record.

The original §4 statistic "CV of ratios across directions" with a
"shuffle-ratios null" is mathematically degenerate: any permutation
of a vector preserves its CV, so the null distribution collapses to
a single point and no observed CV ever beats its own 95th percentile.
Confirmed by two synthetic cases (`tests/test_compare_aging.py`):
isotropic widening → CV=0 vs null_q95=0 (undefined pass); gradient
widening (log-spaced 1×→10×) → CV=1.239 vs null_q95=1.239 (equal
by invariance).

**Replacement (frozen BEFORE any aging data run):**

**Statistic:** Spearman correlation
`ρ_λ = ρ(|λ_a_i|, |λ_b_j(i)|)` across cosine-matched pairs
`(i, j(i))`, where matching is greedy by eigenvector cosine. A
measure of whether the eigenvalue ordering is preserved across age
within the SAME direction family.

**Null:** 1 000 random pairings of a-eigenvalues to b-eigenvalues
(eigenvectors ignored). Compute `ρ_λ` per draw.

**Pass:** observed `ρ_λ > null_q95` → eigenvalue orderings coherent
with the matched-direction structure.

**Interpretation (frozen):**
* `ρ_λ > null_q95`  → **direction-coherent widening** (specific
  axes widen more than others, and the same axes remain soft in
  old as in young).
* `ρ_λ ≤ null_q95`  → either isotropic (uniform widening, ρ_λ ill-
  defined on tied spectra) or incoherent across age (no stable
  axis identity) — both cases count as **no direction-specific
  loosening signal**.

Reporting:
* `rho_lambda` per type
* `null_q95` per type
* `isotropy_pass` boolean per type (True iff `ρ_λ > null_q95`)
* **fraction of eligible types passing**

### 4b — Prereg amendment: isotropy statistic (2026-10-14, made before any data run governed by it; replaces §4a)

**Why.** §4a tests the wrong thing. Under uniform (isotropic)
softening every eigenvalue scales by the same factor, the rank order
is preserved and `ρ_λ ≈ 1`, so "pass = ρ_λ > null" passes the very
outcome §4 is meant to reject. Confirmed in
`tests/test_compare_aging.py::test_isotropic_widening_is_what_spearman_passed`
(uniform 1.6× widening: `ρ_λ > 0.95`).

**Operators.** Per `(type, age group)`, the centroid operator is the
mean per-cell `J` over the 30 cells **of that set** nearest the
set's centroid in the shared space (`compare.centroid_operator`;
this makes §3's "30 cells nearest the type centroid" explicit).

**Statistic.** For cosine-matched direction pairs `k`
(greedy, as §4a), `r_k = log(|λ_old,k| / |λ_young,k|)`;
`S = max_k |r_k − median(r)|`, how far the most-changed direction
departs from the typical change. Isotropic softening gives `S ≈ 0`;
single-axis loss gives large `S`.

**Null.** Split-half within young: 1 000 random splits of the young
cells of the type into two disjoint halves of
`min(n_old, ⌊n_young / 2⌋)` cells; `S` between the two halves'
centroid operators, computed from the same fixed shared-space drift.
Sampling noise under no aging.

**Pass.** `S > null_q95`.

**Provenance, not gates.** `rho` (§4a Spearman) and `cv_ratio`
(original §4) are reported per type.

**Seeds.** Per-type seed = `seed + crc32(cell_type) mod 2³¹`
(the previous `hash(cell_type)` is salted per Python process and
was not reproducible).

**Synthetic checks (must pass before any data run).** Isotropic
1.6× widening fails; single-axis 3× widening passes; identical
Gaussians fail. Calibration over 40 seeds each (`n = 400` per group,
`d = 10`, 200 null draws): false-pass rate 0.075 for identical and for
uniform 1.6×, power 1.00 for single-axis 3× (median `S = 2.21` vs
`null_q95 = 0.11`).

**Interpretation (replaces §4a's).**
* `S > null_q95` → **direction-specific loosening**: at least one
  direction changed beyond the typical change by more than sampling
  noise allows.
* `S ≤ null_q95` → no direction-specific signal: isotropic change
  or no change.

Reporting per type: `S`, `iso_null_q95`, `iso_pass`, `rho`,
`cv_ratio`; fraction of eligible types passing.

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
