# opspace / figure spec — draft

Status: **draft, started 2026-10-12** immediately after Phase B closure.
This spec defines what `scjdo2.opspace` and `scjdo2.plot` have to deliver
to make the operator-space story readable from a plot and reproducible
from stored keys alone. It is the bridge between Phase B (which produced
the per-cell Jacobian field and classifier) and Phase C (which will
re-ground the field in a gene-space NB rate model).

The spec is written against the keys registry in [`opspace.md`](opspace.md).
A reader of a saved AnnData must be able to render every figure here
without re-running the pipeline.

---

## 1. Goals, non-goals, invariants

**Goal.** For each cell `c`, communicate three things:

1. **Where** `c` sits in the operator space (`lead_re`, `lead_im`,
   behaviour class).
2. **How confident** we are in that assignment (`op_confidence`, Fisher
   ledger, `op_sensitive`), i.e. is the cell near a decision boundary.
3. **Why** — which modes (eigvecs → genes in Phase C) carry the
   behaviour.

**Non-goals.** No pseudotime axis in the primary figures (that is the
v1 story we moved away from); no v1 archetype activations as the main
narrative — those live only in the comparison figure.

**Invariants.**

* The plotting layer **reads** keys; it never recomputes `J`, eigvals,
  invariants, or behaviour. If a key is missing, raise a clear error
  pointing at the opspace function that writes it.
* Every figure has a Phase-B-safe fallback: if gene-space information is
  not present (Phase B saved runs), the panel degrades to a labelled
  stub, never a wrong plot.
* Colour mapping for behaviour classes is a single source of truth in
  `scjdo2.plot.style`; opspace tables and figures use the same palette.
* Confidence modulates opacity (not colour) so the behaviour class is
  always readable.

---

## 2. Figure catalogue

Three figures, each with a Phase B form and a Phase C form.

### Figure 1 — `figure_operator_space(adata)` — three panels

**A — Embedding, behaviour-coloured.** Scatter in `use_rep`
(`X_umap` if present else `X_pca`), coloured by `op_behaviour`,
opacity by `op_confidence`. Overlay lineage paths from
`opspace.lineage_path` as thin lines when provided. Fork cells are
highlighted with an edge stroke. The axis tag reads "relative time"
when `uns["scjdo2"]["meta"]["clock_calibrated"]` is False, "clock time"
when True.

**B — Spectrum plot.** Scatter in the complex plane, `Re(λ₁)` on x,
`Im(λ₁)` on y, coloured by `op_behaviour`. Vertical bands at
`±behaviour_tol` from `uns["scjdo2"]["behaviour_tol"]` show the drift
zone. Points above the Henrici 90th percentile (`op_sensitive=True`)
get a halo stroke — these are the non-normal cells where the lead
eigenvector is unstable and the classification is one adversarial
perturbation away.

**C — Drivers.**
* *Phase B form:* stub text explaining that gene-space Jacobians are
  Phase C, with a 1-sentence reference to `V2_PLAN R13` so a reader
  understands why fork precision is bounded.
* *Phase C form:* horizontal bar chart of top-K genes for the leading
  eigenvector of the picked archetype (or the fork archetype by
  default), signed by the eigvec component. Driver weights come from
  `opspace.drivers(adata, which="archetype")` once gene-space is wired.

**File output.** `fig1_opspace.pdf` + `fig1_opspace.png` (300 dpi).

### Figure 2 — `figure_archetype_sheet(adata)` — Phase C

One row per archetype (produced by the Phase B Step 8 semi-NMF adapter
+ Phase C gene-space J). Each row has three cells:

| cell | content |
|---|---|
| left | eigenvalue inset in the complex plane (archetype operator eigvals) |
| middle | top-K drivers bar (signed), using gene-space eigvecs |
| right | stat block: behaviour class, Henrici, reactivity, transient gain, support size |

**Phase B form:** `NotImplementedError` (current state). Spec stays here
for Phase C to implement.

### Figure 3 — `figure_v1_vs_v2(adata_v2, v1_results)` — comparison

Already implemented (Phase B); spec documents the invariants:

* Left panel: v1 slicing archetype activations vs. pseudotime (read from
  `v1_adata.uns["scjdo"]["activations"]` and `t_centers`).
* Right panel: Panel A of Figure 1 (same embedding, same colour scheme).
* Purpose: show that v2 preserves the behavioural content v1 extracted
  but does not depend on a pseudotime axis. Not a quantitative gate.

---

## 3. Required opspace API additions for Phase C

Everything below writes into AnnData keys already named in
`docs/opspace.md` under "Not yet set (Phase C)". The figure code then
reads them; no plot touches raw matrices.

### 3.1 `opspace.gene_space_jacobian(adata, rate_model, ...) -> None`

Writes `adata.uns["scjdo2"]["J_gene"]` — per-cell gene-space Jacobian,
either `(n_cells, G, G)` sparse or a `(n_cells,)` list of CSR matrices
depending on sparsity. Signature:

* Input: a fitted NB rate model (Phase C), the already-populated latent
  `J` (so the chain-rule factor is available).
* Mechanism: `J_gene(c) = ∂μ/∂z |_c @ J(c) @ (∂μ/∂z |_c)^+`, with the
  Fisher-weighted pseudoinverse supplied by the rate model's ledger.
* Writes: `uns["scjdo2"]["gene_names"]`, `uns["scjdo2"]["J_gene"]`,
  updates `uns["scjdo2"]["meta"]["jacobian_method"] = "gene_space"`.

### 3.2 `opspace.fisher_ledger(adata, rate_model, ...) -> None`

Writes `adata.uns["scjdo2"]["fisher_ledger"]` and updates
`adata.obs["op_confidence"]`. The ledger is a dict with keys
`per_cell` (Fisher info trace), `per_gene` (NB variance), and
`reasons` (why a cell is low-confidence: low counts, high noise,
boundary cell).

### 3.3 `opspace.archetype_decomposition(adata, K, ...) -> None`

Semi-NMF of the stacked `J_gene` across the manifold into `K`
archetype operators. Writes `adata.uns["scjdo2"]["archetypes"]` with
keys: `W` (soft assignments), `H` (archetype operators), `behaviour`
(classified by applying `opspace.behaviour` to each archetype's
operator), `drivers` (top-K signed drivers per archetype).

### 3.4 `opspace.compute(adata, ..., smooth_k=30) -> None`

A thin wrapper that chains `spectrum → invariants → behaviour →
field_gradient` (optionally with `scjdo2.field.smooth_jacobians`
pre-applied) in one call. Default `smooth_k=0` (no smoothing); users
opt in. This mirrors the Phase B gate script's `_gate_smoothed`
helper so the production pipeline exposes the same choice.

---

## 4. Phase C acceptance criteria (bringing R13 forward)

The fork P/R-within-0.10-of-oracle criterion carried forward from Phase
B R13 lives here. For Phase C to be acceptable:

1. On the equilibrated pitchfork gate, **DSM-equilibrated** (or its
   Phase C gene-space equivalent) must satisfy:
   * holder P and R ≥ 0.8 absolute, **and**
   * fork P and R within 0.10 of the oracle at each `τ_crit`.
2. The gene-space Jacobian must reproduce the latent-space behaviour
   classification at ≥ 0.95 agreement (cell-level) on simulated data
   where the ground truth is known, so Phase C does not regress Phase
   B's holder performance.
3. Figure 1 Panel C must render a non-stub for at least one archetype
   on a Phase C fit of the pitchfork — i.e. gene-space drivers are
   wired to the figure.

Acceptance evidence: `benchmarks/phase_c_gate/equilibrated/report.md`
(to be produced), plus `figures/fig1_opspace_pitchfork.{pdf,png}`
regenerated from a Phase C adata.

---

## 5. Open design questions (to resolve in Phase C kickoff)

* **Driver sign convention.** The eigvec's phase is free; we pick the
  phase that maximises the sum of positive driver weights so bars read
  as "genes that go up". Document this in `opspace.drivers`.
* **Panel C and complex eigenvectors.** For `damped_oscillator` /
  `growing_oscillator` the leading eigvec is complex. Options:
  (i) show real part only; (ii) show two bar panels (real, imag);
  (iii) show the magnitude |v| with a dot indicating the phase. Pick
  one at Phase C kickoff.
* **Smoothing default in `opspace.compute`.** Current default `k=0`
  (off). If Phase C's gene-space J inherits the per-cell precision
  gap, we may flip the default to `k=30` once we check the gene-space
  gate.
* **Confidence source after Phase C.** `op_confidence` currently reads
  `1.0` everywhere. Fisher ledger lands it in a defined scale; the
  question is whether the colour alpha mapping stays linear or moves
  to `sqrt(c)` so low-confidence cells aren't invisible.

---

## 6. Phase B → Phase C handoff checklist

* [x] `scjdo2.field.smooth_jacobians` + `JacobianField.smooth_field`.
* [x] Equilibrated gate reports (raw + smoothed) with closed Phase B
      verdict.
* [x] V2_PLAN R13 recording the gap + Phase C criterion carry-forward.
* [x] CHANGELOG entry announcing Phase B closure.
* [ ] Phase C skeleton: `scjdo2.rates` (NB model) and the three Phase C
      opspace additions above.
* [ ] Figure 1 Phase C form wired to real drivers.
* [ ] `benchmarks/phase_c_gate/equilibrated/report.md` with the fork
      criterion met (or re-escalated if gene-space does not close it).
