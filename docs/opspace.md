# `scjdo2.opspace` — stored keys reference

The analysis API writes every quantity into `AnnData`. The plotting layer
reads these keys and never re-estimates. This page is the authoritative
registry.

## Set by `scjdo2.pipeline.fit_latent_field`

| Container | Key | Shape / type | Description |
|---|---|---|---|
| `adata.obsm` | `X_drift_v` | `(n_cells, d)` float | Drift velocity `v_c = f(z_c)` (Step 2 output). |
| `adata.uns["scjdo2"]` | `J` | `(n_cells, d, d)` float | Per-cell Jacobian field. Phase B stores a dense latent-space tensor; Phase C switches to a sparse per-cell genes × genes layout. |
| `adata.uns["scjdo2"]` | `meta` | dict | Pipeline config: `clock_calibrated` (bool), `use_rep`, `time_key`, `k`, nested `drift` / `field` sub-dicts. |
| `adata.uns["scjdo2"]` | `field_reason_summary` | dict | Fraction of cells by reason code (`ok`, `no_counts`, `no_variation`, `not_in_ball`). |

## Set by `scjdo2.opspace.spectrum`

| Container | Key | Shape / type | Description |
|---|---|---|---|
| `adata.obsm` | `op_eigvals` | `(n_cells, n_eig)` complex | Conjugate-free leading eigenvalues of `J_c`, sorted by `Re` desc. Only the `Im ≥ 0` member of each conjugate pair is kept; `Im < 0` partners are discarded by construction. Padded with NaN+NaNj when fewer than `n_eig` kept. |
| `adata.uns["scjdo2"]` | `op_eigvecs` | `(n_cells, n_eig, d)` complex | Matching eigenvectors. In Phase C `d = G` (gene space). |
| `adata.uns["scjdo2"]` | `op_valid` | `(n_cells,)` bool | False where `J_c` had NaN entries. |
| `adata.uns["scjdo2"]` | `op_eig_reason` | `(n_cells, n_eig)` object | Per-cell per-eigenvalue reason code (`ok`, `no_variation`, `off_tangent`). Phase C only. |
| `adata.uns["scjdo2"]` | `op_tangent_rank` | `(n_cells,)` int32 | Per-cell truncated-SVD tangent rank from `scjdo2.field.fit_jacobian_tangent_projectors`. Phase C only. |
| `adata.obs` | `op_rank` | int | Phase C only — mirror of `op_tangent_rank` for easy filtering/plotting. |
| `adata.uns["scjdo2"]` | `smooth_k` | int | Set when `spectrum(smooth_k > 0)` was used; records the kNN pre-smoothing radius applied to `J` for the duration of the call. |
| `adata.uns["scjdo2"]` | `spectrum_meta` | dict | `{"n_eig", "fisher_mask", "min_fisher", "J_kind", "tangent_restricted", "smooth_k"}`. |

## Set by `scjdo2.opspace.invariants`

Columns written to `adata.obs`:

| Column | Type | Description |
|---|---|---|
| `lead_re` | float | `Re(λ₁)` of the leading eigenvalue (conjugate-free; `Im ≥ 0`). |
| `lead_im` | float | `Im(λ₁)`. |
| `n_stable` | int | Count of eigenvalues with `Re < −tol`. |
| `n_unstable` | int | Count of eigenvalues with `Re > tol`. |
| `n_near_zero` | int | Count of eigenvalues with `|Re| ≤ tol`. |
| `n_complex_pairs` | int | Count of eigenvalues with `Im > 0` (each pair represented once). |
| `spectral_gap` | float | `Re(λ₁) − Re(λ₂)`. |
| `henrici` | float | Normalised Henrici departure from normality (`[0, 1]`). |
| `transient_gain` | float | `max_{n ∈ [1, N]} ||J^n||_2`, N configurable. |
| `reactivity` | float | `σ_max(J) / ρ(J)` (`≥ 1`; `= 1` iff `J` is normal). |
| `eigvec_cond` | float | `κ(V)` of the eigenvector matrix. |
| `op_confidence` | float | Fisher-weighted leading-eigvec confidence; `1.0` everywhere in Phase B until the ledger is persisted. |

## Set by `scjdo2.opspace.behaviour`

| Container | Key | Type | Description |
|---|---|---|---|
| `adata.obs` | `op_behaviour` | categorical | One of `holder`, `drift`, `fork`, `damped_oscillator`, `growing_oscillator`, or `""` (invalid cell). |
| `adata.obs` | `op_sensitive` | bool | `True` when `henrici` is at or above the `sensitive_pct`'th percentile (default 90). |
| `adata.uns["scjdo2"]` | `behaviour_tol` | float | The threshold used for `near_zero`; default `0.1 × σ(lead_re)`. |
| `adata.uns["scjdo2"]` | `sensitive_threshold` | float \| None | Henrici cutoff for `op_sensitive`. |
| `adata.uns["scjdo2"]` | `invariants_meta` | dict | `{"tol", "horizon"}`. Written by `invariants`; `horizon` default `5 / median\|lead_re\|` clamped to `[5, 200]`. |

## Set by `scjdo2.opspace.field_gradient`

| Container | Key | Type | Description |
|---|---|---|---|
| `adata.obs` | `op_field_gradient` | float | Mean `||J_j − J_c||_F / ||J_c||_F` over `j` in the `k` latent neighbours of `c`. |

## Returned by `scjdo2.opspace.lineage_path`

A list of dicts (one per start cell), each with keys `start`, `cells`,
`z`, `lead_re`, `lead_im`, `op_behaviour`.

## Written by `scjdo2.opspace.export_tables`

CSV files under `out_dir/`:

- `per_cell_invariants.csv` — one row per cell with every column above
  that applies. Eigenvalue columns carry `_relative` unless
  `adata.uns["scjdo2"]["meta"]["clock_calibrated"]` is `True`.
- `per_archetype_summary.csv` — written when
  `adata.uns["scjdo2"]["archetypes"]` is populated. Phase C.

## Set by the Phase C pipeline (prerequisite for `scjdo2.opspace.compute` R15 orchestration)

| Container | Key | Shape / type | Description |
|---|---|---|---|
| `adata.uns["scjdo2"]` | `J` | list of `(G, G)` CSR | Per-cell gene-space Jacobian from `scjdo2.field.fit_gene_space_jacobians`. Each cell's columns are restricted to its live-gene index. |
| `adata.uns["scjdo2"]` | `J_latent` | `(n_cells, d, d)` float | Dense latent-space J from the Phase B pipeline. Required for the R15 behavioural pass of `scjdo2.opspace.compute` (behaviour from latent, drivers from gene-space + ∇η lift). |
| `adata.uns["scjdo2"]` | `fisher_ledger` | `FisherLedger` or dict | Per-cell Fisher info with reason codes and the Fisher-weighted tangent projector + rank (`off_tangent` sub-dict). |
| `adata.uns["scjdo2"]` | `grad_eta` | `(n_cells, G, d)` float | `∇η(z_c)` from the NB rate model; consumed by `compute` to produce the lifted driver eigenvector. |

## Set by `scjdo2.opspace.compute` (§1 wrapper, V2_PLAN R15)

| Container | Key | Shape / type | Description |
|---|---|---|---|
| `adata.uns["scjdo2"]` | `op_eigvec_latent_lead` | `(n_cells, d)` complex | The latent leading eigenvector cached from the behavioural pass. |
| `adata.uns["scjdo2"]` | `op_eigvec_lifted` | `(n_cells, G)` complex | `w_c = ∇η(z_c) · v_latent_lead(c)` — the chain-rule lift read by `drivers(use_lift=auto)` for Panel C. Set only when `grad_eta` is present. |
| `adata.uns["scjdo2"]` | `compute_meta` | dict | `{"smooth_k", "n_eig", "min_fisher", "did_latent_pass", "has_lifted_eigvec"}`. |

## Not yet set

- `adata.uns["scjdo2"]["archetypes"]` — archetype weights, operators,
  behaviour name, drivers. Phase B Step 8 adapter is still pending;
  `compute` skips `archetype_operators` with a `RuntimeWarning` until
  it lands.
- `adata.uns["scjdo2"]["gene_names"]` — explicit gene index for
  gene-space Jacobians. Phase D io work.

See `V2_PLAN.md` R11 (Phase C refinement), R14 (truncated-SVD tangent
+ d-selection + Option A NB-NLL ridge CV), and R15 (behaviour from
latent, drivers from gene-space ∇η-lift) for the design rules.
