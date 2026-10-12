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
| `adata.uns["scjdo2"]` | `op_eigvecs` | `(n_cells, n_eig, d)` complex | Matching eigenvectors. |
| `adata.uns["scjdo2"]` | `op_valid` | `(n_cells,)` bool | False where `J_c` had NaN entries. |
| `adata.uns["scjdo2"]` | `spectrum_meta` | dict | `{"n_eig", "fisher_mask", "min_fisher"}`. |

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
| `adata.obs` | `op_sensitive` | bool | `True` when `henrici` is at or above the dataset's 90th percentile. |
| `adata.uns["scjdo2"]` | `behaviour_tol` | float | The threshold used for `near_zero`; default `0.1 × σ(lead_re)`. |
| `adata.uns["scjdo2"]` | `sensitive_threshold` | float \| None | Henrici cutoff for `op_sensitive`. |

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

## Not yet set (Phase C)

- `adata.uns["scjdo2"]["fisher_ledger"]` — per-cell Fisher info with
  reason codes. Phase C work.
- `adata.uns["scjdo2"]["archetypes"]` — archetype weights, operators,
  behaviour name, drivers. Phase B Step 8 adapter + Phase C.
- `adata.uns["scjdo2"]["gene_names"]` — gene index for gene-space
  Jacobians. Phase C.

See `V2_PLAN.md` R11 for the Phase C refinement plan.
