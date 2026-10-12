# V2_PLAN — scJDO v1 → scJDO2

This is scJDO2's living plan. The v1 repo at `/Users/terooatt/Documents/scJDO`
is read-only; v1 is pinned here at tag **`scjdo-v56-submission-20260922-r25`**
(highest `-r*` on the submission series, `__version__ = "0.3.0"`, Python ≥ 3.9,
PyTorch ≥ 2.2).

Phase A snapshot:
- Package layout (§2 of the brief) scaffolded; all modules are typed stubs that
  raise `NotImplementedError`.
- Tutorials 01 and 05 map 1-to-1 to v2's pipeline. Tutorials 02 (Fourier), 03
  (bridge synthetic), 04 (Paul15 bridge) do not — v2 does not implement those
  tracks. We will still run them through v1 for the baseline, and the
  comparison report will state the scope mismatch plainly for those three.
- Open blocker for Phase B: no in-repo pitchfork generator in `scjdo/data/`.
  The τ_crit ∈ {0.1, 0.3, 0.5} dataset lives under `scratchpad_run/` and
  `reproducibility/code/`. We will copy the α-ramp generator into
  `scjdo2/data/synthetic_pitchfork.py` at the start of Phase B (risk R2).

---

## 1. v1 package map

Walk of `scjdo/` with one-line role, plus whether each module is actually used
by the five example notebooks. Modules in **bold** are v2-relevant (imported
or replaced); others are drop candidates.

| Module | File(s) | Role | Used by 01–05? |
|---|---|---|---|
| **core API** | `scjdo/__init__.py` | Re-exports `pp`, `tl`, `pl`, `DriftField`, `SchrodingerBridge`, `tensors_from_anndata`. | yes |
| **pp** | `scjdo/pp/_trajectory.py` | `prepare_trajectory`: normalize → HVG → PCA/scVI/Harmony → kNN → DPT/Palantir pseudotime. | 01, 02, 04 (direct); 05 (indirect) |
| **tl.drift** | `scjdo/tl/_drift.py` | `fit_drift`, `fit_drift_branches`, `get_instability_genes`, `branch_drift_analysis`, `peak_interior`. Train DriftField → per-cell/windowed Jacobians → semi-NMF/Koopman → instability genes. | 01 |
| **tl.bridge** | `scjdo/tl/_bridge.py` | `fit_bridge`, `fit_bridge_branches`, `get_bridge_instability_genes`. | 03, 04 |
| **tl.regulators** | `scjdo/tl/_regulators.py` | `infer_regulators`, `infer_regulators_branches`, `load_network`, `filter_regulators`. | 01, 04 |
| **models.drift** | `scjdo/models/drift.py` | `DriftField(cfg, X_ref, V_ref)` — FiLM-conditioned score+residual MLP with optional kNN velocity prior; `.jacobian(x, t)` returns `(B, D, D)` autograd Jacobian. | 01 |
| **models.schrodinger_bridge** | `scjdo/models/schrodinger_bridge.py` | `SchrodingerBridge(cfg)` — forward/backward neural ODE with Sinkhorn OT coupling. | 03, 04 |
| **models.fourier_score_network** | `scjdo/models/fourier_score_network.py` | `MultiBandScoreNet` — multi-band score network over DFT bands. | 02 |
| **models.(representation / hybrid_grn / scjdo_comm)** | `scjdo/models/*.py` | Latent-space, hybrid GRN, cell-communication models — manuscript side. | no |
| **archetypes.decompose** | `scjdo/archetypes/decompose.py` | `jacobian_modes`, `jacobian_modes_svd`, `_seminmf`, `_nnls_rows`, `_rescale`, `_tv_smoothed_W_update`. Semi-NMF with column-wise NNLS. | 01 (via `fit_drift`) |
| **archetypes.koopman** | `scjdo/archetypes/koopman.py` | `koopman_modes`, `_matrix_geometry` (henrici, reactivity, transient_gain, eigvec_cond), `_split_complex_pairs`, EDMD. | optional in `fit_drift` |
| **archetypes.windowing** | `scjdo/archetypes/windowing.py` | Adaptive Gaussian kernel windowing: `kernel_jacobian_tensor`, `knn_adaptive_h`, `score_bandwidth`, `select_bandwidth`, `build_temporal_operator`. S = R · C · L bandwidth selection. | 01 (via `fit_drift`) |
| archetypes.cellrank | `scjdo/archetypes/cellrank.py` | CellRank-style driver computation. | no |
| **nn** | `scjdo/nn/score_net.py`, `scjdo/nn/time_embed.py` | Shared time embedding & score net building blocks. | yes (indirect) |
| **losses** | `scjdo/losses.py` | `denoising_score_matching`, `control_energy`, `fp_residual_loss`, `local_sigma`. | 01 (via `fit_drift`) |
| io | `scjdo/io/anndata.py`, `scjdo/io/anndata_hybrid.py` | `tensors_from_anndata` + hybrid variant. | 01–05 (indirect) |
| **pl.drift** | `scjdo/pl/_drift.py` | `drift_field`, `sensitivity`, `archetypes`, `coordination`, `instability_genes`, `summary_figure`. | 01 |
| pl.bridge | `scjdo/pl/_bridge.py` | 6 bridge plots. | 03, 04 |
| pl.regulators | `scjdo/pl/_regulators.py` | 6 regulator plots. | 01, 04 |
| cli | `scjdo/cli/main.py` | `scjdo drift|bridge` CLI. | no (invoked from shell) |
| pipeline | `scjdo/pipeline/*.py` | Standalone training scripts (hybrid GRN etc.). | no |
| atlas | `scjdo/atlas/*.py` (10 files) | scOpAtlas regime classification. | no (out of v2 scope) |
| fourier | `scjdo/fourier/*.py` | Fourier score diffusion. | 02 |
| grn | `scjdo/grn/*.py` | Hybrid GRN inference. | no |
| transport | `scjdo/transport/*.py` | Sinkhorn / spectral OT. | 03 |
| comm | `scjdo/comm/*.py` | Cell-cell communication. | no |
| simulate | `scjdo/simulate/trajectories.py` | `euler_integrate`, `apply_archetype_modulation`. | no |
| training | `scjdo/training/fourier_trainer.py` | `SpectralDiffusionTrainer`. | 02 |
| validation | `scjdo/validation/*.py` | archetype_cosine_similarity, temporal_shuffle_null, prior_sensitivity_report. | no (used in reproducibility) |
| utils / viz / analysis / data | small helpers | `data/synthetic.py` has `make_spiral`, `rna_velocity_field` only — **no pitchfork generator** (see R2). | partial |

**Takeaway:** 7 of 27 subpackages are hit by the in-scope examples: `pp`,
`tl` (drift / bridge / regulators), `pl`, `models.drift`,
`models.schrodinger_bridge`, `models.fourier_score_network`, `archetypes`.
scOpAtlas (notebook 05) is out of v2 scope — its regime classifier is
replaced by v2's operator space.

---

## 2. Mapping table — v1 → v2 step

| v2 Step | Concept | v1 implementation | v2 action |
|---|---|---|---|
| 1 — positional embedding | scVI / FA / PCA, frozen after fit | `scjdo/pp/_trajectory.py:_run_scvi`, `_run_pca`. FA is not implemented in v1. | **Reimplement** in `scjdo2/embed.py`. Do not import `pp._trajectory` (entangled with DPT/Palantir pseudotime). |
| 2 — drift field | Neural-ODE drift, autodiff Jacobian | `scjdo/models/drift.py:DriftField, DriftConfig` + training loop in `scjdo/tl/_drift.py:fit_drift` (lines 395–416). | **Import** DriftField; write v2's own ~50-line training loop that reuses `scjdo/losses.py:denoising_score_matching` and `control_energy`. |
| 3 — gene rate model | NB decoder, size factor as offset | **None in v1** (v1 derives gene scores by linear projection through PCA loadings, `scjdo/tl/_drift.py:_gene_scores`, no count likelihood). | **New.** scvi-tools module in `scjdo2/rates.py`. |
| 4 — rates along drift | Autodiff chain rule | **None in v1** (v1's sensitivity comes from Jacobian eigenvalues of the drift, not from `∇_z μ_g · f`). | **New.** `scjdo2/rates.py`. |
| 5 — neighbourhood Jacobian field | kNN-ball regression | v1 estimates **per-cell** Jacobians by autodiff of DriftField (`scjdo/models/drift.py:DriftField.jacobian`, lines 263–281; batched in `scjdo/tl/_drift.py:_compute_jacobians_per_cell`), then collapses to a `(T, D, D)` temporal tensor via Gaussian-kernel pseudotime windowing (`scjdo/archetypes/windowing.py:build_temporal_operator`, `kernel_jacobian_tensor` lines 42–83). | **Replace.** `scjdo2/field.py` regresses rate on level vectors in a kNN ball in latent space, likelihood-weighted. |
| 6 — Fisher ledger | Per-entry curvature + reason codes | **None in v1.** | **New.** `scjdo2/fisher.py`. |
| 7 — operator space | Invariants + leading eigvecs | `archetypes/windowing.py:max_real_eig_curve`, and in Koopman mode `archetypes/koopman.py:_matrix_geometry` (lines 143–208): `henrici`, `reactivity`, `transient_gain`, `eigvec_cond`. | **Reimplement 4 formulas** (not the Koopman backend) in `scjdo2/opspace.py`. See §4. |
| 8 — archetypes | Semi-NMF, column-wise NNLS | `archetypes/decompose.py:_seminmf`, `_nnls_rows`, `_rescale`. | **Import** core routine; adapter reshapes operator-space `(N, f)` coords instead of calling `jacobian_modes`. Fisher weighting layered on top (row-scale `M` by `sqrt(w_i)`). |
| 9 — grammar scaffold | Sequence + commutator norms | **None in v1.** | **New minimal.** `scjdo2/grammar.py`. |

---

## 3. The two v1 imports (per brief rule 3)

### 3a. DriftField — thin adapter in `scjdo2/drift.py`

- File: `scjdo/models/drift.py` (298 lines, self-contained).
- Public API:
  ```python
  @dataclass
  class DriftConfig:
      dim: int = 64; hidden: int = 256; depth: int = 4; beta: float = 0.1
      use_spectral_norm: bool = True
      use_velocity_prior: bool = False
      vel_scale: float = 2.0; vel_k: int = 15; vel_tau: float = 1.0
      vel_time_mode: str = "flat"       # flat | mid | root | rise
      vel_conf_power: float = 1.0; alpha_control: float = 0.001
      jacobian_dim_warn: int = 500

  class DriftField(nn.Module):
      def __init__(self, cfg: DriftConfig,
                   X_ref: Optional[torch.Tensor] = None,
                   V_ref: Optional[torch.Tensor] = None): ...
      def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
      def jacobian(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor
      def jacobian_approx(self, x, t, n_proj=64) -> torch.Tensor
  ```
- Fitting inputs: `X` (N × D latent), `t` (per-cell scalar — v1 uses pseudotime
  but any monotone surrogate works), optional `V_ref` for the kNN velocity
  prior.
- Standalone: **yes.** No AnnData/pseudotime inside the class. The
  entanglement is in `tl/_drift.py:fit_drift` (the training loop). v2 can skip
  it and write its own loop (DSM + control energy from `scjdo/losses.py` are
  ~50 lines).
- Clock calibration: DriftField treats `t` as opaque; swap pseudotime for a
  per-cell clock scale at the input, no model change needed. The brief's
  `clock_calibrated` flag lives in v2's wrapper, not in DriftField.

### 3b. Semi-NMF — thin adapter in `scjdo2/archetypes.py`

- File: `scjdo/archetypes/decompose.py` (211 lines, pure numpy + scipy).
- Import `_seminmf`, `_nnls_rows`, `_rescale`. **Do NOT** call `jacobian_modes`
  — it reshapes `H` back to `(K, D, D)`, which v2 does not want.
- `_seminmf(M, rank, max_iter, tol, seed, tv_lambda) -> (W, H, err)` — ALS;
  `H` by ridge solve, `W` by column-wise NNLS (`scipy.optimize.nnls`),
  monotone-non-increasing objective; `_tv_smoothed_W_update` is optional ISTA
  with TV penalty (keep at `tv_lambda=0` in v2 — v2's neighbour-smoothness
  lives in Step 5, not Step 8).
- v2 feeds operator-space `M ∈ ℝ^{N × f}` (cells × operator features) directly,
  bypassing the `(T, D, D)` reshape. Fisher weighting: row-scale `M` by
  `sqrt(w_i)` before `_seminmf` (both H- and W-updates are quadratic, so a
  row-weight absorbs cleanly). Confirm with a unit test in Phase C.

---

## 4. Reimplement list — Koopman non-normality descriptors

Reimplemented in `scjdo2/opspace.py` from `scjdo/archetypes/koopman.py:_matrix_geometry`
(lines 143–208). We cite v1 for provenance but **do not import** it — avoids
pulling EDMD, trajectory reduction, and complex-pair splitting into Step 7.

Exact formulas:

```python
# Input: J ∈ ℝ^{D×D} (per-cell Jacobian from Step 5).

eigs   = np.linalg.eigvals(J)
frob2  = float(np.sum(J * J))
frob   = float(np.sqrt(frob2))

# (1) Henrici departure-from-normality  ν ∈ [0, 1]
sum_lam2    = float(np.sum(np.abs(eigs) ** 2))
henrici_abs = float(np.sqrt(max(0.0, frob2 - sum_lam2)))
henrici     = henrici_abs / (frob + 1e-12)

# (2) Reactivity = σ_max(J) / ρ(J)   (≥ 1; equality iff J is normal)
sigma_max  = float(np.linalg.norm(J, ord=2))
rho        = float(np.max(np.abs(eigs))) if eigs.size else 0.0
reactivity = sigma_max / (rho + 1e-12)

# (3) Transient gain = max_{n ∈ [1, N]} ||J^n||_2
P    = np.eye(D, dtype=np.float64)
gain = 0.0
for _ in range(int(n_max)):        # v1 default n_max = 30
    P  = J @ P
    gn = float(np.linalg.norm(P, ord=2))
    if gn > gain: gain = gn
    if rho < 1.0 and gn < 1e-12: break
transient_gain = gain

# (4) Eigvec conditioning κ(V)
_, V = np.linalg.eig(J)
eigvec_cond = float(np.linalg.cond(V))
```

v2's descriptors act on **operator-space `J_c` per cell**, not on a windowed
Koopman `K`. The v1 caveats about non-neutral metric under `whiten=False` do
not apply to v2 (v2 estimates `J` in gene space, no EDMD, no SVD reduction).
Document that in `opspace.py`.

---

## 5. Drop list — v1 modules v2 does NOT port

| v1 module(s) | Reason |
|---|---|
| `scjdo/models/schrodinger_bridge.py`, `scjdo/tl/_bridge.py`, `scjdo/pl/_bridge.py`, `scjdo/transport/*` | Bridge workflow out of scope. |
| `scjdo/tl/_regulators.py`, `scjdo/pl/_regulators.py` | Regulatory inference out of scope. v2 drivers come from the leading eigenvector of the local operator, filtered by Fisher confidence. |
| `scjdo/archetypes/koopman.py` + EDMD hooks | Koopman backend out of scope; v2 reimplements only the 4 non-normality formulas (§4). |
| `scjdo/archetypes/windowing.py` | **Replaced** by latent-space neighbourhood field in `scjdo2/field.py`. |
| `scjdo/archetypes/cellrank.py` | Not used by 01–05; CellRank-style drivers not part of v2 design. |
| `scjdo/atlas/*` | scOpAtlas regime classifier; v2's operator space is the direct replacement. |
| `scjdo/fourier/*`, `scjdo/models/fourier_score_network.py`, `scjdo/training/fourier_trainer.py` | Fourier track; parallel modeling line, not in the 9-step pipeline. |
| `scjdo/grn/*`, `scjdo/models/hybrid_grn.py`, `scjdo/pipeline/train_hybrid_grn.py` | Hybrid GRN track. |
| `scjdo/comm/*`, `scjdo/models/scjdo_comm.py` | Cell-cell communication. |
| `scjdo/cli/main.py`, `scjdo/pipeline/{train_from_anndata, train_bridge}.py` | v1 CLI out of scope; v2 writes `scjdo2/cli.py`. |
| `scjdo/simulate/trajectories.py` | ODE simulation / archetype modulation — manuscript figures only. |
| `scjdo/validation/*` | Tied to v1 semi-NMF outputs and bootstrap-on-windowing; v2 writes new acceptance tests. |
| `scjdo/io/anndata_hybrid.py` | Hybrid-GRN IO; not used in 01–05. |
| `scjdo/pp/_trajectory.py` | Entangled with DPT/Palantir and HVG prep; v2 writes a lean `embed.py`. |

**Kept (not imported into `scjdo2`, but used by `compare/run_v1.py`):**
`scjdo/tl/_drift.py:fit_drift` and `scjdo/pp/_trajectory.py:prepare_trajectory`
are invoked by the comparison harness to produce the baseline.

---

## 6. Pseudotime windowing (what v2 replaces)

v1 locations v2 replaces with `scjdo2/field.py`:

- **Adaptive Gaussian kernel windowing** — `scjdo/archetypes/windowing.py`:
  - `kernel_jacobian_tensor(J_per_cell, t_per_cell, grid, h, cell_weights)`
    lines 42–83 — kernel aggregation.
  - `knn_adaptive_h(t_per_cell, grid, k=80, floor=0.005)` lines 85–93 — kNN-adaptive `h(τ)`.
  - `score_bandwidth` lines 114–148 — R (bootstrap reproducibility), C (peak
    contrast), L (peak localisation), S = R · C · L.
  - `select_bandwidth` lines 151–194 — argmax S subject to `n_eff_interior_min ≥ 30`.
  - `build_temporal_operator` lines 201–283 — end-to-end builder; called from
    `scjdo/tl/_drift.py:445–476`.
- **Legacy fixed windowing** (`windowing='fixed'`) — `scjdo/tl/_drift.py:_build_windows`
  lines 166–185 + body lines 480–501 (100 windows at 80% overlap, Gaussian
  smoothing with `smooth_sigma=1.5`).

v2 does not transfer the bandwidth-sweep machinery (R · C · L); its
model-selection target is **held-out NB log-likelihood** on `k` (brief,
Phase C).

---

## 7. The four tutorial notebooks

Notebook 05 (`05_scopatlas_complete_workflow.ipynb`) is **out of v2 scope**:
its regime classifier is replaced by v2's operator space, and scOpAtlas as
a track is not carried into this repo (no `scjdo.atlas` import, no
baseline, no notebook 05 counterpart).

| # | Notebook | Dataset | Key v1 calls | Primary outputs | v2 coverage |
|---|---|---|---|---|---|
| 01 | `01_paul15_hybrid_drift_tutorial.ipynb` | **Paul15** (`sc.datasets.paul15()`, HVG=2000, PCA=50; root `7MEP`) | `pp.prepare_trajectory` → `tl.fit_drift(n_archetypes≈5, n_epochs=5000, windowing='kernel')` → `pl.summary_figure` etc. → `tl.infer_regulators` → regulator plots | `figure3.pdf`, drift / sensitivity / archetypes / coordination / instability_genes PDFs, regulator table + 5 regulator PDFs | **Full.** Headline tutorial; the drift+archetype loop maps directly to v2 Steps 2, 5, 7, 8. |
| 02 | `02_paul15_fourier_tutorial.ipynb` | Paul15 | `prepare_trajectory`; Fourier track only (`scjdo.fourier.*`, `MultiBandScoreNet`, `SpectralDiffusionTrainer`). No `fit_drift`. | spectrum / training / real-vs-generated / band_genes PDFs | **Out of scope.** v2 does not implement Fourier diffusion. |
| 03 | `03_schrodinger_bridge_tutorial.ipynb` | 2D synthetic source/target gaussians | `SchrodingerBridge`, `sinkhorn_divergence`, `compute_ot_plan`. No `fit_drift`. | 6 bridge PDFs | **Out of scope.** |
| 04 | `04_paul15_schrodinger_tutorial.ipynb` | Paul15 | `prepare_trajectory` → `tl.fit_bridge` (source=bottom-20% pt, target=top-20%); forward/backward instability genes + regulators | 6-panel `bridge_summary.pdf`, 2× regulator sets | **Out of scope.** |

**Comparison report consequence (brief §4):** only 01 produces a full
side-by-side comparison. For 02, 03, 04 the report states "v2 does not
implement this track; v2 operator field on the same dataset is at
`<path>`, with the following properties …". The v1 outputs are still
generated and saved as baselines so §4's runtime/memory table has rows
for each in-scope notebook.

**Datasets locally available:**
- Paul15 is `sc.datasets.paul15()` (scanpy cache; no local file required).
- `examples/data/paul15/` has auxiliary Paul15 artifacts.
- `data/BoneMarrow/`, `data/moignard15/`, `data/replogle_2022_k562_essential.h5ad`
  exist but are not touched by notebooks 01–05.
- **No pitchfork synthetic dataset generator** is public under `scjdo/data/` —
  see risk R2.

---

## 8. Dependencies and install footguns

v1 `pyproject.toml`:

```
torch>=2.2  numpy>=1.23  scipy>=1.10  matplotlib>=3.7  seaborn>=0.12
tqdm>=4.66  scikit-learn>=1.3  anndata>=0.10  scanpy>=1.9
scvelo>=0.3  cellrank>=2.0  pandas>=2.0  h5py>=3.10  numba>=0.58
```

Footguns v2 inherits via the `[compare]` extra:

1. `cellrank>=2.0` pulls `jax`, `pygam`, `pygpcca`. `import scjdo` likely
   imports it transitively via `tl/__init__.py` → `_bridge.py`; a bare
   `import scjdo` may fail without it. Verified during Phase A install.
2. `scvelo>=0.3` compiles against numpy; on Apple Silicon prefer the
   prebuilt wheel.
3. `torch>=2.2` on macOS/Darwin 25.6 ARM — PyPI wheel + MPS is fine.
4. `numba>=0.58` requires LLVM-matched `llvmlite`.
5. `faiss-cpu` optional — v1's KNN has a numpy fallback.
6. v1 installs console script `scjdo` — v2 must register `scjdo2`.
7. v1's import name `scjdo` collides with any `pip install scjdo` from PyPI
   — pin via local path at tag (we use `scripts/install_v1_compare.sh` with
   `git worktree`).

Our strategy: a **conda env `scJDO2`**, cloned from the pre-existing `scJDO`
env (which already has torch 2.12, scvi-tools 1.3.3, scanpy 1.11.5, anndata
0.11.4). After cloning, `pip install -e .` for scJDO2 and
`bash scripts/install_v1_compare.sh` for v1 at the preprint tag via a git
worktree at `/tmp/scJDO-v1-<tag>/`.

---

## 9. Risks and open questions

**R1. DriftField time axis.** DriftField's `forward(x, t)` takes a per-cell
scalar `t`. In v1 that's pseudotime; in v2 we pass pseudotime to the drift
model only. **Step 5 does not re-couple pseudotime** — it uses latent-space
neighbourhoods. Phase B must keep these decoupled.

**R2. (blocks Phase B start) No in-repo pitchfork generator.** Brief's Phase B
acceptance requires τ_crit ∈ {0.1, 0.3, 0.5}. The generator lives in
`scJDO/scratchpad_run/saddle_readouts_midinterval.py` and
`scJDO/reproducibility/code/T*.py`. Phase B's first commit copies the α-ramp
generator into `scjdo2/data/synthetic_pitchfork.py` so v2 is not path-coupled
to v1's layout.

**R3. Semi-NMF tensor layout.** `_seminmf` is axis-agnostic, but
`jacobian_modes` reshapes `H` to `(K, D, D)`. v2 **must** call `_seminmf`
directly on operator-space `(N, f)` and keep patterns as `(K, f)` without
reshape. Fisher row-weighting: pre-scale `M` by `sqrt(w_i)`; the ridge solve
and NNLS both absorb a quadratic weight cleanly. Verify in Phase C with a
synthetic unit test.

**R4. `use_spectral_norm=True` suppresses eigenvalues.** v1 default. Bounds
`σ_max(J)` and may hide saddles on the pitchfork test. **Phase B default:
`False`.** Toggle on only if training is unstable without it.

**R5. Palantir / MAGIC for Paul15 reproducibility.** v1 README says Fig 3
needs `palantir==1.4.4` + MAGIC. For the comparison report, plain `fit_drift`
(tutorial 01's call, no `fit_drift_branches`) is enough — DPT suffices.
Palantir+MAGIC only enters if we try to reproduce the manuscript's
branch-separated analysis, which is not required by the brief.

**R6. Rotational non-identifiability.** Brief says v2 does **not** resolve
this. We keep an unused hook in `scjdo2/drift.py` for an optional known
rotation (cell-cycle phase) to subtract before Step 5, and document it as
"not calibrated" when absent. Do not implement in Phase B.

**R7. Clock calibration.** Expose a `clock_scale: Optional[np.ndarray]`
kwarg on `DriftAdapter.fit` that rescales `t` per cell before `forward`.
Simple to implement; defer to Phase B.

**R8. Fisher ledger storage.** Row-wise per gene (each gene's regression is
independent) → ledger is `(N_cells, N_genes)`. Two options: (a) sparse CSR
with reason-code tags per entry, (b) dense NaN-mask with a parallel
categorical reason array. **Default to (b)** — simpler, 1 byte/entry extra.
Revisit in Phase C if HVG sparsity makes (a) worthwhile.

**R9. CLI name collision.** v1 installs `scjdo`; v2 installs `scjdo2`. No
collision.

**R10. Reuse Koopman descriptors "by reimplementing".** The 4 formulas in §4
are copied literally from `scjdo/archetypes/koopman.py:_matrix_geometry`. v2
cites v1 for provenance in `opspace.py` but does not import — keeps `opspace`
free of EDMD baggage.

**R11 (discovered Phase B first pass; resolved).** The brief's "±0.05 against
the dataset's known bifurcation value" was ambiguous. The in-repo v1
reference (`benchmarks/v1_baseline/pitchfork/pitchfork_reference.json`) now
shows v1 is a non-localizer on this generator — mean `tau_R1_lambda_max`
error vs. analytic τ_crit is ≈0.5 across {0.1, 0.3, 0.5} × 3 seeds, and
most seeds argmax at the τ=0.95 edge. Phase B is accepted on the brief's
"better localisation than v1 slicing" wording: scJDO2's error at τ=0.5 is
≈0.1 (5× better than v1). The strict ±0.05 target against analytic τ_crit
is a Phase C refinement — likely requires the NB rate model (so the
velocity prior is replaced by `∇_z μ_g · f`, breaking the "argmax where
motion is largest" bias our Phase B drift exhibits), and may also need
sweeping `use_spectral_norm`, `vel_scale`, and `width`.

**R12 (identified Phase B gate iteration).** v1's `DriftField.residual` MLP
is **unidentified** under `control_energy(u) = mean(u²)` supervision on the
full forward sum `u = β·score + residual + vel_scale·gate·v_hat`. Nothing
pins the residual's output; it converges to approximately cancel the
velocity-prior buffer (same magnitude, opposite sign), inheriting the
velocity prior's spatial structure in `∂residual/∂z`. On the pitchfork,
the pseudotime-gradient velocity prior has a locally expansive Jacobian,
so the compensation gives `∂residual/∂z` uniformly positive-definite.
This explains why **every drift-based estimator on this benchmark —
v2 autodiff on transient, v2 autodiff on equilibrated, v1 slicing on
equilibrated — produces `frac(Re(λ_max) > 0) ≈ 1.00`**, independent of
the data regime, the σ schedule, or the training length. The DSM module
itself trains correctly (unit-Gaussian sanity in
`benchmarks/phase_b_gate/dsm_sanity/report.json` returns learned
`∂score/∂z ≈ -I`); the pathology is downstream.

Fix adopted in v2:
* ``DriftConfig.use_residual: bool = True`` (new). When set to ``False``
  (``scjdo2/drift.py``) the residual module is replaced by a frozen zero
  module. Drift becomes ``u = β·score (+ vel_prior if any)``, whose
  autodiff Jacobian is exactly ``β · ∂²log p / ∂z²`` plus the velocity
  contribution. On the equilibrated pitchfork gate we set
  ``use_residual=False``, ``inject_velocity_prior=False``, and
  ``β = σ_sde²`` so the drift is ``σ²·∇log p`` in the SDE's units.
* **Velocity priors on real data must be differentiable smooth fields**
  — fit a small MLP or GP to the pseudotime-gradient targets so the
  prior is part of the autodiff drift. v1's `KNNVelocity` buffer is a
  non-differentiable gather and must not be used in v2's gene-space
  phase: the plotted drift and the classified Jacobian must be the
  **same function**.

---

## 10. Phase A deliverables status

| Deliverable | Status | Where |
|---|---|---|
| scJDO2 skeleton (§2 layout) | **done** | `/Users/terooatt/Documents/scJDO2/` |
| v1 pinned at preprint tag | **done** | `scjdo-v56-submission-20260922-r25`, see `scripts/install_v1_compare.sh` |
| v1 installs from scJDO2 without edits | **in progress** | conda env `scJDO2` (clone of `scJDO`); install_v1_compare.sh runs after env is ready |
| v1 baseline outputs saved on 5 datasets | **not started** | compute-heavy; defer to user go-ahead per Phase A ask |
| V2_PLAN.md | **done** (this file) | — |
| compare/run_v1.py + data loaders | **done (skeleton)** | `compare/run_v1.py`, `data/loaders.py` |

Deviations from the brief:
- Phase A acceptance point 3 ("v1 baseline outputs saved on all five tutorial
  datasets") is **deferred to a user go-ahead**; the brief implies hours of
  compute per dataset, and running without confirmation seemed off-piste. The
  harness is ready; one command (`python -m compare.run_v1 --dataset <key>
  --out benchmarks/v1_baseline/<key>`) kicks each off.
- The user environment already had a working `scJDO` conda env; we clone
  from it rather than build torch/scvi-tools from scratch. The brief said
  "venv" but conda is the pragmatic match for this stack on this machine.
