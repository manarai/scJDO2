# scJDO2 — Operator-space analysis API and figure spec

**Repo:** `/Users/terooatt/Documents/scJDO2`
**Scope:** add a public analysis and plotting API, `scjdo2.opspace` and `scjdo2.plot`, that turns a fitted operator field (Step 5 output plus Step 6 ledger) into operator-space coordinates, archetype summaries, and a three-panel publication figure. This is the deliverable for Step 7 and the figure half of Step 8. Follow the existing `V2_PLAN.md` conventions and the package layout already scaffolded.

Read this whole file first. Build in the order given. Stop and ask if the stored Jacobian format in `scjdo2/io.py` differs from what §1 assumes.

**Merger note (2026-10-12):** this file is the authoritative merged spec. It is the body of the originally-attached `scJDO2_figure_api_instructions.md` plus §§7–10 (Phase B / Phase C forms, Phase C design lock Q1–Q16, R13 carry-forward and acceptance, handoff checklist). The duplicate `docs/scJDO2_figure_api_instructions.md` has been deleted per the user's merger directive. Where the two originals disagreed, the authoritative spec wins (see §10.1 for the single disagreement surfaced).

**Addendum (2026-10-12, Task 41 build instructions §0).** The merged spec is amended with the four additions below. Where this addendum and the earlier body disagree, this addendum wins and `docs/scJDO2_figure_layer_build_instructions.md` + `V2_PLAN.md` (R14, R15) record the deviations.

1. **Tangent restriction everywhere.** All spectral quantities (eigenvalues, eigenvectors, Henrici, transient gain, drivers) are computed on `P_c J_c P_cᵀ` (per `scjdo2.field.fit_jacobian_tangent_projectors`, Fisher-weighted ∇η SVD with a permutation-null rank — see V2_PLAN R14) and lifted back to gene space through `P_c`. Off-tangent directions never appear as eigenvalues; genes whose weight in the lifted eigenvector comes entirely from off-tangent directions are reported with reason `off_tangent`. This replaces the un-restricted treatment implicit in §§2.1–2.2 of the original spec.
2. **Sensitive-holder mark.** A cell is `sensitive` when `henrici` is above the dataset's 90th percentile. Stored as a boolean in `obs["op_sensitive"]`. Rendered as a dashed outer ring (radius 2× the marker, 0.75 pt, dash 3-2, same colour as the behaviour class) in Panels A and B. Eigenvalue colouring and behaviour class are unchanged.
3. **Complex eigenvector rendering.** For a complex leading mode, Panel C and the archetype sheet draw one-sided bars: length = |v_g|, colour = arg(v_g) on a cyclic map (matplotlib `twilight`), with a small horizontal cyclic colour bar labelled "early in cycle → late in cycle". No sign axis. Phase is reported relative to the gene with the largest magnitude (fixed to phase 0 by the sign convention in §2.1).
4. **Relative-units label.** Every axis or column that carries an eigenvalue says "relative" unless `uns["scjdo2"]["clock_calibrated"]` is `True`. Tested in §5 test 7.

**Addendum (2026-10-12, V2_PLAN R15 — classifier source).** `obs["op_behaviour"]`, `obs["op_sensitive"]`, and the scalar invariants used by Panels A and B (`lead_re`, `lead_im`, `henrici`, `transient_gain`, `spectral_gap`, `op_confidence`, `n_*`) are computed from the **latent per-cell Jacobian** ``J_latent(c)``. The gene-space J supplies drivers (Panel C) and the Fisher ledger. Panel C drivers come from the chain-rule lift ``w_gene = ∇η(z_c) · v_latent`` of the latent leading eigenvector, scored by |w_gene,g| weighted by Fisher precision — not from an independent gene-space eigendecomposition. See V2_PLAN R15 for the rationale.

---

## 0. Ground rules

1. All inputs come from the on-disk formats already defined in `scjdo2/io.py` (AnnData with per-cell sparse Jacobians and the Fisher ledger). Do not invent a second storage format.
2. Every quantity in the figure must be computable from the stored objects alone, with no re-fitting. The plotting layer reads; it never estimates.
3. Eigenvalues are in **relative time units** unless `adata.uns["scjdo2"]["clock_calibrated"]` is `True`. Every axis label and every exported table must say "relative" in that case. Never print hours.
4. Complex eigenvalues come in conjugate pairs. Store and plot only the member with non-negative imaginary part, and document that.
5. Fisher confidence is carried through every table and every mark: as a column in tables, as opacity and an error bar in figures. A function that drops it is wrong.
6. Matplotlib only for the static figure (journal-ready vector PDF, 180 mm wide, 8 pt text, colour-blind-safe palette). No seaborn, no plotly for the static output. An optional interactive HTML export is allowed later, not now.
7. Write tests on the synthetic pitchfork dataset so that every function has a known-answer check.
8. **Every step must be re-runnable from its stored inputs.** `scjdo2 fit-rate` on an existing adata (Q16 lock) is the user-facing form of this invariant; the opspace functions obey the same rule.

---

## 1. Inputs (what the API assumes)

Per cell `c`, from Step 5 and Step 6:
- `J_c`: sparse genes × genes Jacobian (rows = all genes, columns = live genes in that cell's neighbourhood).
- `F_c`: diagonal Fisher information, same shape as `J_c`, with a reason-code array for empty entries (`no_counts`, `no_variation`, `not_in_ball`).
- `z_c`: latent position (Step 1) and `v_c`: velocity (Step 2).
- Archetype weights from Step 8 if already run; if not, the API must still work and the archetype panels are skipped with a warning.

---

## 2. Analysis API — `scjdo2/opspace.py`

Implement these functions. Signatures are indicative; keep them simple and typed.

### 2.1 `spectrum(adata, n_eig=5, fisher_mask=True, min_fisher=None)`
For each cell, compute the top `n_eig` eigenvalues of `J_c` by real part, with eigenvectors. If `fisher_mask`, zero out entries of `J_c` whose Fisher information is below `min_fisher` (default: 10th percentile over the cell's live entries) before decomposing, so unreliable entries cannot create spurious modes. Store in `adata.obsm["op_eigvals"]` (complex, cells × n_eig), `adata.uns["op_eigvecs"]` (sparse or memmapped, cells × n_eig × genes, complex), and record the mask threshold used.

### 2.2 `invariants(adata)`
Per cell, from the stored spectrum, compute and store in `adata.obs`:
- `lead_re`, `lead_im`: real and imaginary part of the leading eigenvalue (non-negative Im member).
- `n_stable`, `n_unstable`, `n_near_zero`, `n_complex_pairs` (thresholds configurable; defaults relative to the spread of eigenvalues across the dataset, not absolute numbers).
- `spectral_gap`: difference in real part between first and second eigenvalue.
- `henrici`: Henrici departure from normality of `J_c` (restricted to live columns). `transient_gain`: max over t of the 2-norm of exp(J_c t), computed on the live block only, capped at a configurable horizon. Reimplement these two formulas; do not import the v1 Koopman module.
- `op_confidence`: a scalar per cell summarising the ledger — median Fisher information over the entries that contribute to the leading eigenvector (weights above 5 % of the max). This is the opacity used in every plot.

### 2.3 `behaviour(adata, tol=None)`
Classify each cell's leading mode into one of: `holder` (Re < −tol, Im = 0), `fork` (Re > tol, Im = 0), `drift` (|Re| ≤ tol, Im = 0), `damped_oscillator` (Re < −tol, Im > 0), `growing_oscillator` (Re > tol, Im > 0), plus a boolean `sensitive` flag when `henrici` is above the dataset's 90th percentile. Default `tol` = 0.1 × the standard deviation of `lead_re` across cells. Store as `adata.obs["op_behaviour"]` (categorical) and `adata.obs["op_sensitive"]`.

### 2.4 `drivers(adata, which="lead", top=15, min_fisher=None)`
For each cell, or for each archetype if archetype operators exist, return a tidy DataFrame with columns `gene`, `weight` (signed real part of the eigenvector entry, after fixing the eigenvector's arbitrary overall sign so that **the largest-|magnitude| entry is real and positive**; this convention works identically for real and complex vectors and is independent of how many genes go up), `magnitude`, `phase` (radians, only meaningful for complex modes; NaN otherwise), `fisher`, `fisher_se` (1/sqrt(F) as the standard error of the underlying Jacobian entries that feed this gene), `reason` (the ledger reason code where the entry is empty). Sort by `magnitude` descending, filter by `min_fisher`, keep `top`. Document the sign convention in the docstring.

### 2.5 `archetype_operators(adata)`
If Step 8 weights exist, compute each archetype's operator as the Fisher-weighted mean of `J_c` over cells, weighted by archetype membership, and run 2.1 to 2.4 on those operators. Store in `adata.uns["archetypes"]` with the automatic behaviour name from 2.3 and the driver table from 2.4.

### 2.6 `field_gradient(adata, k=15)`
Per cell, the mean Frobenius distance between `J_c` and the Jacobians of its `k` nearest latent neighbours, normalised by the Frobenius norm of `J_c`. High values mark where the operator changes fastest. Store as `adata.obs["op_field_gradient"]`.

### 2.7 `lineage_path(adata, start_cells, n_steps=200, dt=None)`
Integrate the drift field from a set of starting cells, record the nearest cell at each step, and return the sequence of (`lead_re`, `lead_im`, `op_behaviour`, archetype weights) along the path. This is the curve in Panel B and the archetype sequence for Step 9.

### 2.8 `export_tables(adata, out_dir)`
Write CSVs: per-cell invariants, per-archetype summary (behaviour name, eigenvalues, Henrici, transient gain, n cells, mean confidence), per-archetype drivers, lineage paths. Every eigenvalue column header carries `_relative` unless clock-calibrated.

---

## 3. Figure API — `scjdo2/plot.py`

### 3.1 `figure_operator_space(adata, lineages=None, archetype=None, out=None)`
Produces the three-panel figure as one matplotlib `Figure` and, if `out` is given, saves PDF + PNG at 300 dpi. Layout: A and B side by side on the top row, C full width below. Shared categorical colour key for behaviour classes across all panels, shown once in the top-right of Panel A:

| behaviour | colour (hex) | marker |
|---|---|---|
| holder | #378ADD (blue) | filled circle |
| drift | #888780 (grey) | filled circle |
| fork | #D85A30 (coral) | filled circle |
| damped_oscillator | #7F77DD (purple) | ringed circle |
| growing_oscillator | #7F77DD (purple) | ringed circle with dot |

Marker shape is required so the figure survives greyscale printing; colour alone is not allowed.

**Panel A — operator field on the embedding.** Scatter of cells at `z_c` (first two dims, or a UMAP of z if `adata.obsm["X_umap"]` exists). Two layers: a diverging colour map of `lead_re` (blue–white–red, centred at 0, symmetric limits at the 98th percentile of |lead_re|) with a thin colour bar labelled "leading eigenvalue, Re (relative)", and on top, outline rings on cells with `lead_im > 0`. Point alpha = `op_confidence` rescaled to [0.15, 1]. Draw the lineage paths from 3.3 as thin grey curves. Annotate the cell with the highest `lead_re` within the fork class as "fork" with a leader line.

**Panel B — leading eigenvalues in the complex plane.** Scatter of (`lead_re`, `lead_im`) per cell, one point per cell, coloured and shaped by behaviour, alpha = confidence. Horizontal line at Im = 0, vertical dashed line at Re = 0 labelled "Re = 0". Axis labels: "Re (growth rate, relative)" and "Im (rotation rate, relative)". A footnote under the panel: "conjugate partner at −Im omitted". Overlay each lineage path from 3.3 as a line with an arrowhead at its end. Region labels placed once, in quiet space, not on points: "stable" at left, "unstable" at right, "oscillating" above the axis.

**Panel C — eigenvector of one archetype.** For `archetype` (default: the archetype whose behaviour is `fork`; if none, the cell with maximum `lead_re`), horizontal bars of the top 15 driver genes from 2.4. Bars extend right for positive weight, left for negative. Rows ordered by sign then magnitude, so the two gene groups stack as two blocks, with a thin horizontal rule between them. Bar alpha = per-gene Fisher confidence rescaled to [0.2, 1]; a horizontal error bar at the bar end = `fisher_se`. Bar colour = the archetype's behaviour colour. For a complex-mode archetype, **bar length = `magnitude` and bar colour = `phase` on a cyclic colour map (`twilight`), with a small cyclic colour bar labelled "phase (early → late)". No inset.** Gene names on the left; no gene with an empty-ledger reason code appears without a hollow bar and the reason in grey text.

Title lines are panel letters only. All explanatory text lives in the figure legend written by the user, not inside the figure.

### 3.2 `figure_archetype_sheet(adata, out=None)`
One row per archetype: a tiny complex-plane inset with that archetype's eigenvalues, a bar chart of its top 10 drivers (same rules as Panel C), and a text block with behaviour name, Henrici, transient gain, number of cells, mean confidence. This is the supplementary figure that lets a reader check every archetype.

### 3.3 `figure_v1_vs_v2(adata_v2, v1_results, out=None)`
Side-by-side: left, v1 pseudotime-slice archetype sequence along each branch as it appears in the v1 outputs; right, Panel A from 3.1. Mark the fork on both. This is the comparison figure for the journal version and consumes the Phase A v1 baseline outputs.

### 3.4 Style
A shared `style.py` with the colour table above, 8 pt base font, 0.5 pt axes, no top/right spines, and a `with scjdo2.plot.style():` context manager. Export every figure as PDF (vector) and PNG.

---

## 4. CLI

Add to `scjdo2/cli.py`:

```
scjdo2 fit --with-rate-model --input run_dir/ --out run_dir/            # Steps 1–6 bundled
scjdo2 fit --skip-rate-model --input run_dir/ --out run_dir/            # Phase B subset
scjdo2 fit-rate --input run_dir/ --out run_dir/                         # Steps 3–6 on an existing adata
scjdo2 opspace --input run_dir/ --out run_dir/opspace/                  # runs 2.1–2.8
scjdo2 figure operator-space --input run_dir/ --out figs/               # 3.1
scjdo2 figure archetypes     --input run_dir/ --out figs/               # 3.2
scjdo2 figure compare        --v1 v1_dir/ --v2 run_dir/ --out figs/     # 3.3
```

Each command is idempotent and reads only stored results. `fit-rate` exists so Step 4/5/6 are re-runnable from stored Step-1 outputs without redoing drift (Q16 lock).

---

## 5. Tests and acceptance

On the synthetic pitchfork dataset (known τ_crit ∈ {0.1, 0.3, 0.5}):
- `behaviour` labels cells within ±0.05 of each τ_crit as `fork` and cells far from it as `holder`; report precision and recall, require both ≥ 0.8.
- `drivers` on the fork archetype ranks the planted driver genes in the top 10 with precision ≥ 0.8 after Fisher filtering.
- `spectrum` returns conjugate-free output: no two stored eigenvalues per cell are conjugates of each other.
- `field_gradient` peaks within ±0.05 of each τ_crit.
- Running `figure_operator_space` and `figure_archetype_sheet` produces a PDF with no overlapping text (check with a bounding-box test on the matplotlib artists) and no empty panels.
- Removing the clock flag changes every axis label to say "relative"; setting it removes the word.

On the haematopoiesis tutorial: Gata1 and Spi1 appear with opposite signs in the fork archetype's driver table, and the fork class is concentrated where the erythroid and myeloid branches separate (report the fraction of fork-class cells within the top-decile `op_field_gradient` region).

---

## 6. Definition of done

- `scjdo2.opspace` and `scjdo2.plot` importable, documented with docstrings and a short `docs/opspace.md` that lists every stored key in `adata.obs`, `obsm`, `uns`.
- All four CLI commands run on all five tutorial datasets.
- Tests in §5 pass and are in `tests/`.
- `benchmarks/figures/<dataset>/` contains the three figures per dataset.
- `V2_PLAN.md` updated with what was built and any deviation, each with a one-line reason.

---

## 7. Phase B vs Phase C forms

Every figure and every opspace function has a Phase-B-safe fallback: if gene-space information is not present (Phase B saved runs), the panel degrades to a labelled stub, never a wrong plot. The authoritative shapes are in §§2–3; this section lists which pieces come online when.

### 7.1 Figure 1 — three panels

* **Panel A** — operates in Phase B form on the latent-space Jacobian (works today).
* **Panel B** — ditto.
* **Panel C** — *Phase B form:* stub text with a 1-sentence reference to `V2_PLAN R13` explaining why gene-space drivers are not yet available. *Phase C form:* §3.1 Panel C as written (magnitude × twilight phase).

### 7.2 Figure 2 — archetype sheet

* *Phase B form:* `NotImplementedError`. The adapter around v1 semi-NMF (Phase B Step 8) lands with the Phase C gene-space J so archetype operators are definable.
* *Phase C form:* §3.2 as written.

### 7.3 Figure 3 — v1 vs v2

* Already implemented in Phase B. §3.3 documents the shape and the invariant that Panel A is reused on the right.

### 7.4 Opspace functions

* `spectrum`, `invariants`, `behaviour`, `field_gradient`, `lineage_path`, `export_tables` — operational since Phase B on latent J.
* `drivers`, `archetype_operators` — require Phase C gene-space J. Return a documented warning and empty frame on a Phase B adata.
* `spectrum(fisher_mask=True)` — on Phase B adata (no Fisher ledger) this falls back to `fisher_mask=False` with a warning; Phase C flips the default back to the spec's `True`.

---

## 8. Phase C design lock (Q1–Q16, 2026-10-12)

Decisions already captured by this spec in §§1–6 are *not* repeated here. The following are the explicit answers to Phase C's build-time questions, locked ahead of code.

### 8.1 Rate model (`scjdo2.rates`) — Q1, Q2

* **NB mean:** `μ_cg = s_c · exp(η_g(z_c))`, i.e. `log μ_cg = log s_c + η_g(z_c)`.
* **Size factor:** `s_c = total_counts_c / median(total_counts)`, entered as a **log-offset** inside the exponent (never a division).
* **Dispersion:** gene-specific inverse dispersion `θ_g`, learned as a free parameter.
* **Architecture:** shared smooth trunk (SiLU activations, weight decay), gene-specific **linear heads** on top of the trunk. Hidden 128 or similar; the trunk is the regularised piece.
* **Latent positions:** **fixed** from Step 1. The rate model never updates `z`.
* **Objective:** NB log-likelihood on raw counts. 10 % held-out cells for early stopping and model selection. Same split used for the Step 5 ridge-λ held-out choice.
* **Velocity prior (real data):** must be a **differentiable field** inside the autodiff drift (small MLP or GP), never a kNN buffer / gather. The plotted drift and the classified Jacobian must be the same function.

### 8.2 Chain rule & J estimation coordinates — Q3

* `dη_g/dt = ∇_z η_g(z_c) · f(z_c)` by autodiff.
* `dμ_g/dt = μ_g · dη_g/dt` (chain rule).
* **J estimated in `η` (log-rate) coordinates**, not rate coordinates. η is NB's natural linear predictor and Fisher is well-conditioned there. Eigenvalues of the η-J and rate-J agree at fixed points; eigvectors (loadings) differ by a diagonal similarity. Documented in `scjdo2.opspace.spectrum` and `scjdo2.rates`.

### 8.3 Step 5 regression & Fisher weighting — Q4, Q5, Q6

* **No Fisher-weighted pseudoinverse.** Step 5 is a **per-gene-row weighted ridge regression**, closed form.
* Per-cell per-gene **weight:** `w_cg = μ_cg · θ_g / (μ_cg + θ_g)` (NB precision of η).
* **Ridge strength λ:** chosen by held-out NB likelihood on the 10 % split from §8.1.
* **Storage:** per-cell CSR block in a zarr group keyed by cell. Columns restricted to the cell's **live genes** with an index array. Latent J stored in `obsm`. Never materialise a dense `cells × G × G` tensor. (See §1 for the sparse-genes×genes assumption the figure code reads against.)

### 8.4 Fisher ledger (Step 6) — Q7

* Diagonal curvature of the same ridge likelihood per J entry.
* Same sparse layout as J (CSR per cell).
* Reason codes: `no_counts`, `no_variation`, `not_in_ball` (plus `ok`).
* Thresholds stored alongside: the fraction cutoffs and the ridge λ that produced the ledger.

### 8.5 Confidence → α — Q8

* **Rank-based rescaling.** `op_confidence(c)` is rank-mapped to `[0.15, 1]` for cell alpha. Per-gene Fisher is rank-mapped to `[0.2, 1]` for driver bar alpha. Outliers cannot flatten the rest of the field.
* The scalar `op_confidence` definition is §2.2 (median Fisher over leading-eigvec entries above 5 % of max). The plot layer only applies the rank rescaling on top.

### 8.6 Figure rendering — Q9, Q10

* **Driver sign convention:** fix the eigenvector phase so the **largest-|magnitude| entry is real and positive**. Same rule for real and complex vectors. Documented in `opspace.drivers` (§2.4).
* **Complex eigvec rendering:** magnitude → bar length; phase → bar colour on `twilight` (cyclic); small cyclic colour bar labelled "phase (early → late)". No inset.

### 8.7 Archetypes — Q11

* **K** chosen by **held-out NB likelihood** across `K ∈ {2..6}`. Cap at K=6.
* **Decomposition:** v1 semi-NMF adapter, seeded with k-means on the stacked `J_gene` tensor.

### 8.8 Smoothing — Q12

* `smooth_k=0` default in `opspace.compute`. The **Phase C gate reports both `k=0` and `k=30`**, same shape as the Phase B gate. The default flips to `k=30` only with the reason recorded in CHANGELOG.

### 8.9 Latent ↔ gene-space consistency — Q13

* `op_behaviour` agreement between latent-space and gene-space classifications, computed on **cells above the median `op_confidence`**, must be ≥ 0.95 on the synthetic pitchfork gate.
* The all-cells agreement is reported alongside for context, **not gated**: gene-space J carries modes the latent J can't see, so disagreement on low-confidence cells is expected.

### 8.10 Fork P/R — Q14, Q15, R13 carry-forward

* **Fork P and R within 0.10 of oracle** at every `τ_crit` on the equilibrated pitchfork is the Phase C gate (carried from the un-met Phase B gate; see §9).
* First gate run: equilibrated pitchfork only. Paul15 and the other tutorial datasets are **sanity runs**, not gate inputs.

### 8.11 CLI — Q16

See §4. `fit --with-rate-model` is bundled with `--skip-rate-model` escape; `fit-rate` runs Step 3+4+5+6 on an existing adata so each step is re-runnable from its stored inputs.

---

## 9. Phase C acceptance criteria — carrying R13 forward

The fork P/R-within-0.10-of-oracle criterion carried forward from `V2_PLAN R13` lives here. Phase C is acceptable iff:

1. **Fork gate** (R13 carry-forward). On the equilibrated pitchfork at `τ_crit ∈ {0.1, 0.3, 0.5}`, the Phase C gene-space pipeline (or the smoothed `k=30` variant, with the reason recorded) satisfies:
   * holder P and R ≥ 0.8 absolute, **and**
   * fork P and R within 0.10 of the oracle at each `τ_crit`.
2. **Latent↔gene-space agreement** (§8.9). ≥ 0.95 cell-level `op_behaviour` agreement on cells above the median `op_confidence`; the all-cells agreement is reported but not gated.
3. **Spec §5 passes.** Behaviour P/R ≥ 0.8 absolute on the ±0.05 pseudotime band; drivers precision ≥ 0.8 for planted genes; field_gradient peaks within ±0.05 of each τ_crit; figures render without overlapping text and without empty panels.
4. **Figure 1 Panel C** renders a non-stub on the Phase C fit of the pitchfork — gene-space drivers wired to the figure.

Acceptance evidence: `benchmarks/phase_c_gate/equilibrated/report.md`, `benchmarks/figures/pitchfork/fig1_opspace.{pdf,png}`, and the `tests/` suite in §5.

---

## 10. Spec reconciliation and handoff checklist

### 10.1 Points where the two originals disagreed

Only one surfaced: **driver sign convention.** The draft at `docs/opspace_figure_spec.md` proposed "positive-sum phase"; the authoritative `scJDO2_figure_api_instructions.md` says "largest-|magnitude| entry real and positive". The authoritative version wins (reflected in §2.4 and §8.6). No other disagreements.

### 10.2 Phase B → Phase C handoff checklist

* [x] `scjdo2.field.smooth_jacobians` + `JacobianField.smooth_field`.
* [x] Equilibrated gate reports (raw + smoothed) with closed Phase B verdict.
* [x] V2_PLAN R13 recording the gap + Phase C criterion carry-forward.
* [x] CHANGELOG entry announcing Phase B closure.
* [x] Spec merger: this file; `scJDO2_figure_api_instructions.md` deleted.
* [ ] `scjdo2.rates` NB rate model (§8.1).
* [ ] Step 4 chain rule in η-coords (§8.2) wired to `scjdo2.pipeline`.
* [ ] Step 5 per-gene weighted ridge in η (§8.3) with zarr-per-cell CSR storage.
* [ ] Step 6 Fisher ledger (§8.4) with reason codes.
* [ ] CLI: `scjdo2 fit --with-rate-model`, `scjdo2 fit --skip-rate-model`, `scjdo2 fit-rate`.
* [ ] Figure 1 Phase C form wired to real drivers; Figure 2 archetype sheet; §3.1 Panel C complex-mode rendering.
* [ ] Tests in §5 under `tests/`.
* [ ] `benchmarks/phase_c_gate/equilibrated/report.md` with §9 fork + agreement criteria met (or re-escalated).
