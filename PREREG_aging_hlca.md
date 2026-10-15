# Pre-registration — human lung aging arm (HLCA)

**Workstream:** `hlca` (branch `aging-basins`).
**Status:** DRAFT for the user's approval. Frozen when committed with
the line "Approved by the user on <date>" below. Written 2026-10-15
from the hlca/H01 **metadata inventory only** (donor ages, studies,
covariates, cell counts). No model has been fitted to human data and
no statistic has been computed on it.

**Design choices made by the user (2026-10-15):** age as a continuous
variable; a young-only reference as the primary shared space; all
donors in the primary test, with study as the batch variable.

**Carried over from the mouse arm (`PREREG_aging_basins.md`, frozen):**
the energy-drift operator and locked basin rule
(`classify_basin_latent(rule="lead_eig")`), the idea of the §4b
isotropy and §5 anticipation tests, and the §6 outcome table. What
changes, and why:

* **The donor is the unit** (lesson of aging/X05): the mouse tests'
  cell-level split-half null does not see donor-to-donor variation.
  Every human test uses one operator per donor and permutes **age
  across donors**.
* **Equal cell counts per donor** (X05): every donor operator is built
  from the same number of cells.
* **Age is continuous**: each test is a slope against age rather than
  an old-versus-young difference.

---

## 1 — Data

* HLCA core (Sikkema, …, Theis et al., *Nat Med* 2023), CZI H5AD with
  full metadata and raw counts, dataset `066943a2-fdac-4b29-b348-40cede398e4e`
  (as fetched in hlca/H01; sha256 in `coord/hlca/reports/H01.json`).
* Age: `age_or_mean_of_age_range` (years). The one donor without a
  numeric age is excluded from everything (106 donors remain, 10–76
  years).
* Cell types: `ann_level_3`. The heterogeneous bucket **"Rare"** is
  excluded.
* All 14 core studies are used.

## 2 — Preprocessing

* Raw counts in `.layers["counts"]`.
* HVGs: `seurat_v3`, `n_top_genes=2000`, `batch_key="study_or_dataset"`,
  `span=1.0` (as mouse §2c), on all cells of the 106 donors.
* **Drift subsample** (tractability): per `(donor, ann_level_3 type)`,
  at most **60 cells**, drawn with seed `crc32(donor|type)`. The scVI
  models use all cells; the drift and all operators use this
  subsample.

## 3 — Shared space

* **Primary — young-only reference + scArches.** Reference = all cells
  of donors aged **< 45** (48 donors). scVI `n_latent=15`, `n_layers=2`,
  NB, `batch_key="study_or_dataset"`, CPU, `max_epochs=400`, early
  stopping patience 30, seed 0. Donors aged ≥ 45 (58) are mapped in by
  `SCVI.load_query_data(..., accelerator="cpu", device="auto")`,
  `weight_decay=0`, exactly as aging/A02. Frozen check: latent-encoder
  parameters **and** BatchNorm buffers change by exactly 0, else stop.
  Studies without donors < 45 enter only as new query batches.
* **Secondary — published HLCA reference.** The HLCA scANVI embedding
  from Zenodo record 7599104 (all ages). The same drift and tests are
  run in it and reported beside the primary result; it never decides
  the outcome.
* Drift: `fit_energy_drift` on the drift subsample of the shared latent,
  `n_epochs=1500, seed=0, beta=1.0, lr=5e-4`, ESM exact, locked defaults;
  `classify_basin_latent(rule="lead_eig")`, tol = 0.1·std(lead_re).
  Symmetry `sym_max ≤ 1e-4`, else stop.

## 4 — Eligibility and operators

* `n_common = 50`. A donor contributes to a type if it has ≥ 50 cells
  of that type in the drift subsample.
* A type is **eligible** if ≥ 20 donors contribute, including ≥ 5 aged
  < 45 and ≥ 5 aged ≥ 60.
* **Donor operator:** `compare.donor_operator(z, J, n_common=50,
  n_draws=100, n_cells_centroid=30, seed=crc32(type|donor))`: the mean
  over 100 equal-count draws of the 30-nearest-cell centroid operator,
  and the mean centroid.
* **Feasibility (metadata only, Claude 2026-10-15):** under this rule
  13 types are eligible: AT2, Basal, Dendritic cells, EC arterial,
  EC capillary, EC venous, Fibroblasts, Innate lymphoid cell NK,
  Macrophages, Monocytes, Multiciliated lineage, Secretory, T cell
  lineage (24–69 donors each; 5–11 aged ≥ 60). Parenchyma only: 12;
  never-smokers only: 6. Both mouse-matched types are eligible.
* **Reference operator:** `compare.centroid_operator` (30 cells) on the
  pooled drift-subsample cells of the type from donors aged < 45.

## 5 — Tests (per eligible type; all 1 000 age permutations across contributing donors, seed 0)

1. **Isotropy (continuous §4b)** — `compare.isotropy_trend`:
   `r_dk = log(|λ_dk| / |λ_ref,k|)` over cosine-matched directions;
   `β_k` = slope of `r_dk` on age; `S = max_k |β_k − median β|`.
   **Pass:** `S > q95` of the null.
2. **Anticipation (continuous §5)** — `compare.anticipation_trend`:
   `|cos(u_ref, b)|`, where `u_ref` is the soft axis of the reference
   operator and `b` the slope of donor centroids on age.
   **Pass:** statistic `> q95`.
3. **Stiffness (reported, not a header gate)** — `compare.stiffness_trend`:
   slope of `log median |λ|` on age, two-sided permutation p.
   "Stiffness loss" = negative slope with p ≤ 0.05.

Permutations are **not** stratified by study in the primary test (the
user's choice); study enters through the batch variable in §3. §7(a)
reports the stratified version.

## 6 — Outcome rule (fixed before any run)

"Majority" = > 50 % of eligible types.

| Isotropy | Anticipation | Outcome header |
|---|---|---|
| pass in majority | pass in majority | **directional, anticipatory loss** |
| pass in majority | fail in majority | **specific loosening without anticipation** |
| fail in majority | fail in majority | **isotropic noise increase; no aging claim beyond stiffness** |
| fail in majority | pass in majority | **anticipation without specific loosening (corner case; report as such)** |

**Stiffness qualifier (new, learned from the mouse arm).** The header is
always followed by exactly one of:
* "— with stiffness loss in the majority of types", if §5.3 finds
  stiffness loss in > 50 % of eligible types;
* "— with stiffness gain in the majority of types", if a positive slope
  with p ≤ 0.05 holds in > 50 %;
* "— no majority stiffness change".

The header and qualifier are written verbatim into
`benchmarks/aging_hlca/RESULT.md`, with no reinterpretation.

## 7 — Sensitivity analyses (secondary; reported beside; never change §6)

a. Age permutations **stratified within study** (only studies with
   donors on both sides of the age range carry information).
b. **Banovich_Kropski_2020 only** (17 donors < 45, 5 ≥ 60).
c. **Parenchyma only**: cells with `anatomical_region_ccf_score == 0.97`.
d. **Never-smokers only.**
e. The **secondary shared space** (published HLCA reference), §3.
f. Age as two groups, **< 45 vs ≥ 60**, with the same donor-level
   operators, for direct comparison with the mouse arm.

For a–d, eligibility is re-applied; types that fall out are listed, not
rescued.

## 8 — Mouse–human comparison (fixed before either result is read for it)

| mouse (TMS lung) | human (HLCA `ann_level_3`) |
|---|---|
| fibroblast of lung | Fibroblasts |
| natural killer cell | Innate lymphoid cell NK |

For each matched pair, report side by side: mouse A03/X04 calls and
stiffness ratio; human §5 results and §7(f). Agreement is descriptive;
with two pairs, no statistic is computed.

## 9 — Statistical checks (before any data run)

`tests/test_compare_hlca.py`: 40 synthetic donors aged 10–76, with
donor-level jitter in scale and centroid. Calibration (Claude,
2026-10-15):

| scenario | isotropy pass | anticipation pass | stiffness p ≤ 0.05 |
|---|---|---|---|
| no aging (40 seeds; 200 for isotropy/stiffness) | 0.050 / 0.045 | 0.000 | 0.100 / **0.070** |
| uniform widening with age | 0.050 | 0.050 | 1.000 (slope < 0) |
| single-axis widening with age | 1.000 | 0.000 | 0.125 |
| centroid drift along soft axis | 0.050 | 1.000 | 0.100 |

The stiffness test runs slightly above nominal on null data (0.07 over
200 seeds), one reason it qualifies the header rather than deciding it.

## 10 — Limits stated in advance

* Old donors are few (17 aged ≥ 60) and clustered: 10 come from two
  studies (Banovich_Kropski_2020, Seibold_2020_10Xv3).
* Study, assay (10x 3′ v3) and airway sampling lean old. The primary
  test relies on batch correction for study; §7(a–c) test whether the
  result depends on that.
* HLCA "healthy" donors come from varied sources (transplant donors,
  tumour-adjacent tissue, volunteers); this is not modelled.
* Negative results are results.

## 11 — Autonomy

As `PREREG_aging_basins.md` §10: no criterion changes after results;
any change before the run it governs is a dated amendment written by
Claude.
