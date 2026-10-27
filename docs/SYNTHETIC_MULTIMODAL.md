# Synthetic multimodal trajectory datasets

`scjdo2.data.synthetic_multimodal.make_multimodal(preset, seed, **overrides)`
creates seeded `AnnData` datasets that pair four destructive real-time
collections with lineage, metabolic-label, and RNA-count readouts. They are
intended to be **input-compatible examples**, not biological reference data.

## Generate the fixed benchmark set

```bash
python scripts/make_synthetic_multimodal.py --all
# Or a single dataset:
python scripts/make_synthetic_multimodal.py --preset default --seed 0
```

The writer saves large `.h5ad` files locally under
`benchmarks/synthetic_multimodal/data/` (gitignored) and commits only the
small `benchmarks/synthetic_multimodal/MANIFEST.csv` with file hashes and
provenance. Presets are `default`, `stress`, `batch`, and `stationary_cycle`;
each fixed benchmark uses seeds 0, 1, and 2.

## Readouts for downstream tools

| Consumer | Key(s) to use | Meaning |
|---|---|---|
| scVelo | `layers["spliced"]`, `layers["unspliced"]` | Captured count-level spliced and unspliced RNA. `X` is a copy of spliced counts. |
| Palantir | `obsm["X_pca"]` or `obsm["X_fa"]`; `uns["suggested_start_cell"]` | PCA/FA fitted to normalized, `log1p`-transformed spliced counts. The suggested root is the earliest-day cell nearest true `z1 = 0`. |
| LabelledRateModel | `layers["new"]`, `layers["total"]`, `obsm["X_pca"]` or `obsm["X_fa"]`, `obs["pulse_hours"]` | Inputs in the order required by `LabelledRateModel.fit`. `total` is exactly `spliced + unspliced`. **Use the analysis embedding, never `obsm["z_true"]`:** truth must not be an input. |
| Lineage methods | `obs["clone"]`, `obsm["clone_matrix"]` | Stochastically detected clone IDs and a sparse cells-by-clones LARRY-style matrix. Missing `clone` means barcode not detected. |
| Time-aware methods | `obs["day"]`, `obs["time_hours"]` | Four collection times: 24, 48, 72, and 96 hours (`stationary_cycle` has one day). |

The `stress` preset changes barcode and label efficiency, assigns 1/2/4-hour
pulses by day, and doubles splicing for a recorded 20% gene subset midway
through the simulation. The `batch` preset records per-day, per-gene capture
multipliers in `uns["batch_gene_effects"]`. The `stationary_cycle` preset
holds progression at zero while retaining the rotating progenitor pool.

## Exact truth — never a downstream-method input

The following keys are retained solely to score a method after it has been run:

| Key | Truth content |
|---|---|
| `obsm["z_true"]`, `obsm["v_state"]` | Latent state and analytic state drift at collection. |
| `obsm["jacobian_state"]`, `obsm["omega_state"]` | Analytic state Jacobian and antisymmetric rotational component. |
| `obs["clone_true"]`, `obs["fate_true"]` | Complete founder identity and latent fate labels. |
| `layers["u_true"]`, `layers["s_true"]` | Pre-capture unspliced and spliced RNA pools. |
| `layers["u_new_true"]`, `layers["s_new_true"]`, `layers["new_fraction_true"]` | Parallel label-pulse truth. |
| `layers["ds_dt_true"]` | Exact spliced expression velocity, in gene space. |
| `obsm["velocity_true_pca"]`, `obsm["velocity_true_fa"]` | Chain-rule truth velocity mapped into the corresponding linear embedding. |
| `varm["W"]`, `var["transcription_intercept"]`, `var["beta_initial"]`, `var["beta_final"]`, `var["gamma"]` | Kinetic/transcription parameters used to synthesize the data. |

`embed_linear(adata, method="pca"|"fa", n_comp=20)` stores embedding
parameters and normalization details in `uns["embedding_pca"]` or
`uns["embedding_fa"]`. `true_velocity_in_embedding` maps the retained truth
through the same normalization, `log1p`, and fitted linear map; use it only for
post hoc validation.


## Truth for scoring (added after review orient/S01)

* `obsm["velocity_true_expected_<pca|fa>"]` — the true expression velocity
  in the embedding, evaluated at the noise-free expected counts. **Score
  orientation against this.**
* `obsm["velocity_true_<pca|fa>"]` — the same, evaluated at the observed
  Poisson counts; its sign differs from the expected truth on a few percent
  of (cell, axis) pairs. Kept for comparison.
* Every key ending in `_true` (and `velocity_true_*`, `omega_state`,
  `v_state`, `jacobian_state`, `fate_true`, `clone_true`) is truth: never an
  input to a source, an embedding, scVelo or Palantir.
* `stationary_cycle` uses 5 000 founders by default (≈ 2 800 cells).

## V2 realistic-splicing benchmarks

`v2_default`, `v2_stress`, `v2_batch`, and `v2_stationary_cycle` are isolated
from the original S01 presets. Generate them with:

```bash
python scripts/make_synthetic_multimodal.py --v2-all
```

The 12 v2 H5AD files are local, gitignored artifacts under
`benchmarks/synthetic_multimodal/data/v2/`; their independent committed record
is `benchmarks/synthetic_multimodal/MANIFEST_v2.csv`. V2 keeps the S01 latent
state dynamics, destructive collections, stress/batch knobs, and truth schema,
but changes the RNA/readout model to make rate recovery tests realistic.

* V2 has 5,000 founders and a 1% destructive collection fraction (the stationary
  control stays at 20% for its one-day collection). RNA pools are replayed only
  along collected cells' ancestral lineages, never for the full ~300,000-cell
  living population.
* `beta_initial` has median 0.6 h⁻¹ and `capture_eta_unspliced` is 0.6. The
  stress preset doubles beta for its recorded 20% gene subset at the fixed
  halfway point.
* `var["gene_kind"]` records 560 log-linear genes and 80 each `on`, `on_off`, and
  `fate` switching genes. The switching parameters are in `var["theta"]`,
  `var["delta"]`, and `var["fate_sign"]`; unused entries are NaN.
* `var["nb_theta"]` holds the gene-level Gamma–Poisson overdispersion shape.
  The per-cell measurement multiplier is intentionally **not stored**. It is
  shared by the observed spliced/unspliced draws but excluded from `s_true`,
  `ds_dt_true`, and the expected-count embedding truth.

All keys labelled `*_true` (and `velocity_true_expected_*`) remain post-hoc
truth only. They are never inputs to scVelo, Palantir, embeddings, or a lineage
method. The v2 audit is run by `scripts/audit_synthetic.py`; O1/O2 are fixed
recoverability gates only for v2-default seeds, whereas O3/O4 and realism
summaries are reported without tuning.

## V3 calibrated presets (orient/S09)

`make_multimodal("v3_default" | "v3_stress" | "v3_batch" | "v3_stationary_cycle", seed)`
adds a generator calibrated to the S08 real-data reference statistics. v1 and v2
are unchanged.

* 4 000 genes; transcription intercepts `b_g ~ N(μ_b, σ_b)`; gene programmes
  `constant`, `linear`, `on`, `on_off`, `fate` (`var["gene_kind"]`).
* `n_prog` extra Ornstein–Uhlenbeck latent dimensions (τ ~ U(6, 48) h, sd 1).
  These are nuisance dimensions: `obsm["z_true"]` holds all of them, while
  `uns["synthetic_multimodal"]["core_dims"] = [0..5]` and the dynamics truth
  (`v_state`, `jacobian_state`, fate, forward truth) uses the six core
  dimensions only.
* Gamma–Poisson noise with `θ_g ~ LogNormal(log θ₀, 0.5)`; `γ_g ~ LogNormal(log γ₀, σ_γ)`;
  `β_g ~ LogNormal(log β₀, 0.5)`; unspliced capture `η_u`.
* scNT-seq-like labelling (3 h and 5 h pulses, half of each sampling day) and
  LARRY-like lineage design (sampling at days 2, 4 and 6; `p_barcode`,
  `n_founders`, `f_sample`).
* `V3_DEFAULT_KNOBS` (in `scjdo2/data/synthetic_multimodal.py`) holds the
  calibrated vector. It was fitted only to S08 statistic tables
  (`benchmarks/synthetic_multimodal/v3_calibration/`: `calibration_log.csv`,
  `rerank_r5.json`, `r5_lineage_select.json`). `v3_stress` and `v3_batch` add
  v2's stress and batch modifications; `v3_stationary_cycle` uses v2's
  stationary one-day design with v3 genes and noise.
* Full-size datasets: about 12–13 k cells (v3_default, v3_stress, v3_batch) and
  2.8 k cells (v3_stationary_cycle). Generation takes 25–55 min and 12–18 GB
  peak RSS per dataset.

```bash
python scripts/s09_measure.py --dataset v3_default --seed 0 --part generate   # writes data/v3/, MANIFEST_v3.csv
python scripts/s09_measure.py --dataset v3_default --seed 0 --part measure    # S08 statistics (+ S05 O1–O4 for v3_default)
python scripts/s09_measure.py --compare && python scripts/s09_measure.py --gates
```

The measured realism against the real references is in
`v3_calibration/compare_v3.csv`, `targets_met_by_seed.csv` and `gates_v3.json`,
with figures in `benchmarks/figures/real_reference/v3/`.
