# V2_PLAN — scJDO v1 → scJDO2

This is scJDO2's living plan. The v1 repo at `/Users/terooatt/Documents/scJDO`
is read-only; v1 is pinned here at tag **`scjdo-v56-submission-20260922-r25`**
(highest `-r*` on the submission series, `__version__ = "0.3.0"`, Python ≥ 3.9,
PyTorch ≥ 2.2).

**Primary hematopoiesis comparison is the Setty 2019 Palantir CD34+ bone
marrow dataset** (the preprint target). v1 reports per-Palantir-branch
slicing; v2 reports a whole-dataset field. The matched measured-flow
counterpart (same compartment, human CD34+ hematopoiesis) is Qiu 2022
scNT-seq hematopoiesis, loaded via dynamo. **Paul15 is a secondary mouse
replicate of the snapshot arm**; the LARRY in-vitro set (Weinreb 2020)
adds clone-barcoded nesting on mouse. All side-by-side results live in
``SUMMARY.md``.

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

**R13 (Phase B closing note; carried forward to Phase C acceptance).**
Under the Phase B v2 gate (holder P and R ≥ 0.8 absolute; fork P and R
within 0.10 of oracle), the DSM-equilibrated row **passes holder** at
every τ_crit (P ≥ 0.97, R ≥ 0.92) but **fork precision does not come
within 0.10 of oracle** even after the Step 5 post-hoc Gaussian-kernel
smoothing (bandwidth = median neighbour distance, k=30 fixed): best
raw/smoothed fork P across τ ∈ {0.1, 0.3, 0.5} is 0.31→0.40, 0.38→0.54,
0.42→0.64 against oracle 1.00. The ``frac(Re>0)`` aggregates match
analytic (0.03–0.08 vs 0.04–0.05) — the structural sign is right; the
remaining gap is **per-cell score noise near zero**: cells with
|analytic_lead_re| only slightly above the 0.1·std tolerance get flipped
by DSM-fit noise, so precision suffers even where recall and the global
expansion fraction are correct. Smoothing confirms this: averaging over
neighbours shrinks per-cell variance (fork P rises) at the cost of
borderline-positive cells migrating into drift (fork R falls).

Phase B is **closed** with this gap recorded: *fork precision is bounded
by per-cell score noise; the oracle shows the ceiling; to be revisited
with Fisher-weighted gene-space Jacobians (Phase C)*. The fork P/R
"within 0.10 of oracle" criterion moves from a Phase B blocker to a
**Phase C acceptance criterion** (listed alongside the Fisher-weighted
gene-space Jacobian deliverables). Phase B's effective acceptance
reduces to: holder P and R ≥ 0.8 absolute (met), ``frac(Re>0)``
within 2× of oracle (met: ~2× at worst), and the oracle-vs-DSM gap
documented row-by-row in
``benchmarks/phase_b_gate/equilibrated/report.md``.

Rationale for shifting fork criterion to Phase C: the gap is not a
structural/sign/scaling pathology but a signal-to-noise issue at the
decision boundary. Fisher-weighted gene-space Jacobians (Phase C) add
(a) likelihood-weighted neighbourhoods that down-weight noisy cells
near the boundary, (b) gene-space J whose spectral radius is set by
rate-model structure rather than latent score curvature alone, and
(c) rate-model-informed per-cell variance for the lead_re statistic
— all of which should tighten precision without sacrificing recall.

**R14 (Phase C rank ceiling; recorded 2026-10-12).** With a
``d``-dimensional decoder ``η: R^d → R^G`` the per-cell gene-space
Jacobian is a lift of the latent Jacobian: ``J_gene(c) = ∇η(z_c) ·
J_latent(c) · (∇η(z_c))^+`` has rank at most ``d``, and its spectrum
on ``image(∇η(z_c))`` equals the latent spectrum up to a similarity.
The ``G − d`` ambient directions carry no decoder-visible dynamics.

Operational implementation. Three paired choices anchor Phase C on
this synthetic:

* **d_latent by held-out NB NLL.** The rate model sweeps ``d ∈
  {2, 3, 5, 10, 20}`` on ``X_pca[:, :d]`` using its own 10 % cell
  split and picks the ``d`` minimising held-out NB NLL. The chosen
  ``d`` seeds drift, rate, gene-field and tangent.
* **Truncated Jacobian-SVD tangent.** The projector at each cell is
  built from the top-``r_c`` left singular vectors of the Fisher-
  weighted ``∇η(z_c)``, with ``r_c`` set per cell by a permutation
  null (shuffle the ``G·d`` entries, threshold at the median of the
  shuffled max-sv distribution — the expected top sv of a pure-
  noise direction under the NB-precision metric). On the equilibrated
  pitchfork this lands ``r_c`` at 2–3 per cell, matching the
  intrinsic rank of the 2-D attractor.
* **Ridge λ by held-out NB NLL (Option A).** The per-cell weighted
  ridge in ``scjdo2.field.fit_gene_space_jacobians`` picks λ by
  held-out NB NLL on neighbour counts, not weighted MSE. For each
  held-out neighbour ``i`` of cell ``c``, the ridge predicts a
  velocity shift ``v̂_i = a · (η_i − η_c)``; that shift is used as a
  log-rate offset from ``c``, giving a predicted log-mean
  ``log μ̂_i,g = η_c,g + v̂_i,g + log s_i`` and the standard NB
  log-pmf against ``counts[i, g]``. Weighted MSE is kept as a
  secondary report. The Gaussian MSE approximant rewards shrinkage
  (``yhat → 0`` is low-MSE on the weak equilibrated signal and picks
  the grid-max λ at every τ), so the NB NLL on real counts is the
  faithful selection criterion for ``λ``.

Consequence on this synthetic. Gene-space ``J`` restricted to the
truncated tangent is a lift of ``J_latent`` on the decoder-visible
subspace: in the clean limit the two spectra agree up to a
similarity, so the gene-space row of the Phase C gate **can match
``latent_dsm`` but cannot exceed it** — the above-median-conf
agreement ceiling is set by the latent↔latent self-identity (so
``>0.95`` is a tautology, not a test), and the fork P/R ceiling is
R13's `latent_dsm` ceiling, inherited.

Phase C synthetic — **closed on 2026-10-12** (benchmarks/phase_c_gate/
equilibrated/report.md).

* Holder reaches the latent ceiling: gene-space P/R ≈ 0.94–0.95 /
  0.86–0.97 at every τ, passing the absolute 0.8 gate for both
  k=0 and k=30.
* frac(Re>0) matches oracle (0.00–0.04 vs 0.04–0.05) at every τ.
* Fork gap persists (gene-space fork P/R ≈ 0.00–0.03 vs latent
  0.19–0.23 / 0.54–0.77) and is **not** a ridge-CV artifact — both
  NB-NLL and weighted-MSE selectors pick the same λ=1.0 on this
  synthetic. The gap is R13's `latent_dsm` fork gap, amplified
  through the lift ``∇η · J_latent · (∇η)⁺`` where the weak-saddle
  bias and the ``(∇η)⁺`` noise outside the r_c-truncated tangent
  pull the lead eigenvalue below ``tol = 0.1·std(lead_re)``.
* Recorded. Follow-up work on fork recovery belongs to a Phase B
  revisit (sharper DSM for weak saddles) and to the Phase D
  hidden-driver synthetic — not to another Phase C ridge-CV sweep.

Testing new modes requires both a synthetic with a hidden driver
(some basin direction orthogonal to the top-``d`` PCs — e.g. a
3-attractor process embedded with a correlated nuisance axis that
captures one PCA slot) **and** rate-model capacity above the latent
dim so the decoder can represent that direction. Both are deferred
to Phase D. Until then the Phase C gate on this synthetic serves as
a structural-pipeline check (ledger, truncated tangent, Fisher-mass
tie-break, sparse smoothing, d-selection) rather than a classifier-
exceeds-latent gate.

**R15 (figure-layer design rule; recorded 2026-10-12).** *Behaviour
classification uses the latent operator; gene-space J supplies
drivers and the Fisher ledger.* Operationally:

* ``obs["op_behaviour"]`` (and ``obs["op_sensitive"]``, ``obs``
  invariants used by Panels A and B — ``lead_re``, ``lead_im``,
  ``henrici``, ``transient_gain``, ``spectral_gap``, ``n_*``, ``op_
  confidence``) are computed from the latent per-cell Jacobian
  ``J_latent(c)``.
* The gene-space per-cell Jacobian ``J_gene(c)`` and its Fisher
  ledger are only used downstream for driver tables (``scjdo2.
  opspace.drivers``, ``archetype_operators``) and for Panel C.
* Panel C drivers are computed from the **gene-space eigenvector
  lifted on the latent leading mode**: take the latent lead eigvec
  ``v_latent``, lift via chain rule to ``w_gene = ∇η(z_c) · v_
  latent`` (a (G,) vector), score each gene by |w_gene,g| weighted
  by Fisher precision ``w_cg``, filter by the gene-space ledger
  reason codes, and render as 3.4's bars with Fisher-SE whiskers.

Rationale. Phase C R14 established that gene-space ``J`` is a lift
``J_gene = ∇η · J_latent · (∇η)⁺``; the lift preserves the spectrum
on ``image(∇η)`` up to similarity *in the clean limit* but, in
practice, (a) the ``(∇η)⁺`` factor amplifies ridge-shrinkage bias
on weak eigenvalues (the fork saddle) and (b) off-tangent noise
leaks sign-flipping components into the gene-space lead eigvec. On
the equilibrated pitchfork the gene-space row misses fork cells
even though holder is at the ceiling (see §9 R14 above). The
latent operator is where the behaviour signal is clean; gene-space
is where the per-gene *loadings* and the Fisher *uncertainty* live.
Panels A and B therefore render latent classifications; Panel C's
driver table renders the gene-space view of the latent leading
mode.

Operational deviation from ``docs/scJDO2_figure_layer_build_
instructions.md``: that document's §1.3 ``behaviour`` and §1.4
``drivers`` are specified against the gene-space operator by
default. R15 overrides the default source for behaviour (latent)
and keeps the gene-space source for drivers (now via the explicit
∇η-lift on the latent lead mode, not the independent gene-space
eigvec). The ``scjdo2.opspace.compute`` wrapper (§1 of the
instructions) applies R15.

Task 41 — **closed on 2026-10-12.** One commit per section as laid
out in ``docs/scJDO2_figure_layer_build_instructions.md``. The §5
suite passes (8 tests + the Phase B / opspace pitchfork regression
suites). ``scripts/make_figures_pitchfork.py`` produces the pitchfork
figure set under ``benchmarks/figures/pitchfork/``; the Paul15 wire-
up stays deferred until ``scjdo2 run`` is implemented in Phase D.
Known residuals: archetype operators + `figure_archetype_sheet` for
real data wait on the Phase B Step 8 semi-NMF adapter; the per-cell
CSR gene-space J is not yet round-trippable through h5ad (Phase D io
work), so the figure script re-fits on each invocation.

**R16 (anchor-cap book-keeping, recorded 2026-10-13).** *A gene
excluded from an anchor's live set by the ``anchor_max_live`` cap
reports reason ``capped``, not ``no_variation``.* The former is a
storage choice; the latter is a measurement that the gene's η-
variance across the ball is below the ``variation_threshold`` floor.
Collapsing them produced false ``no_variation`` entries for real
transcription factors (e.g. Gata1 / Sfpi1 on the Paul15 2000-gene
run) and hid the fact that the pipeline *could* see them if the cap
were higher.

Operationally:

* ``scjdo2.field.REASON_CAPPED = 4`` and the Fisher ledger's
  ``REASON_CODES`` tuple extends to ``("ok", "no_counts",
  "no_variation", "not_in_ball", "capped")``. The plot layer's
  stacked-bar reason summary now carries the extra slice.
* The anchor path ranks live columns by **NB-precision-weighted η
  variance** on the log-rate scale (``var_i(η_i,g) · <w_i,g>_i``, a
  chi-square-like SNR statistic), not raw ``var_i(η_i,g)``, so the
  cap keeps genes whose signal survives NB noise, not just the loudest
  movers.
* The default ``anchor_max_live`` is **300**; 150 clipped
  biologically-interesting TFs on real data. ``uns["scjdo2"]
  ["anchor_snr"]`` ``(n_anchors, G)`` and ``uns["scjdo2"]
  ["anchor_capped_cols"]`` (ragged list) round-trip through the
  sidecar.
* ``archetype_operators`` reads per-cell reasons from the ledger and
  aggregates: ``fisher_diag > 0`` → ``ok``; else, if *any* member
  cell had the gene capped at its dominant anchor → ``capped``; else
  modal non-ok code (almost always ``no_variation``). The driver
  DataFrame gains an ``snr`` column — mean SNR across the archetype's
  members' dominant anchors — so a ``capped`` or ``no_variation``
  verdict carries a measurement, not just a label.
* **No must-keep gene lists.** The ledger now distinguishes "cap
  wouldn't fit it" from "no SNR here"; forcing a priori gene lists
  into every anchor's live set would reintroduce the lie by other
  means.

**Zero-holder diagnosis on Paul15 (R16, 2000-gene, 2026-10-13).**
The run reports zero ``holder`` cells across all 2730 cells by the
latent-operator classification (R15), with ``frac(Re > 0) = 1.00``
and median ``lead_re`` of +0.067 / +0.071 / +0.099 / +0.090 on
erythrocytes / monocytes / megakaryocytes / progenitors. This is
the **transient-pitchfork pattern**, not a drift-fit failure.
Evidence distinguishing the two: (a) magnitudes track biology — the
terminal megakaryocyte cluster scores the highest median
``lead_re`` and the terminal erythrocyte cluster the lowest, with
progenitors in between, rather than being noise-flat or
implausibly aligned across cell types; (b) the gene-space per-cell
Jacobian restricted to the Fisher tangent *does* peel off a holder
minority exactly where equilibrating terminals should sit —
``frac(Re > 0) = 0.63`` on erythrocytes and ``0.90`` on monocytes,
while megakaryocytes and progenitors remain uniformly positive
(``1.00``). A drift-fit failure would not produce this biology-
aligned ery > mono > mk/prog gradient, and would smear noise
uniformly across clusters. The latent-everywhere-fork /
gene-space-partial-holder split is the ∇η-lift signature R14 §9
flagged (ridge bias on weak contractions pushed through
``(∇η)⁺``), consistent with Paul15 being a differentiation
snapshot with ongoing basin approach but no fully attained steady
states.

**R17 (direction and the ``no root cell`` rule, recorded 2026-10-13).**
*scJDO2 never consumes a root cell, a lineage prior, or an
annotation-based arrow.* Behaviour classification depends only on
the per-cell operator spectrum (``lead_re``, ``lead_im`` and the
tangent-restricted eigvecs); it is **direction-free** by
construction and must stay that way. Any need for a time direction
must come from a *physical* ordering, not a user-supplied arrow:

* **Clone nesting** (shared barcodes across sampling days) — pairs
  (c_early, c_late) give a per-pair direction without a global
  root.
* **Splicing kinetics** (La Manno 2018 / scVelo) — nascent/mature
  ratios are a chemistry-level direction when the equilibration
  assumption is defensible.
* **Metabolic labelling** (scEU-seq, dynamo) — isotope ratios give
  per-cell nascent-fraction direction.
* **Sampling-day labels** — a coarse time axis along a lineage
  path; see R18 below for the clock-check rule.

Operational implications:

* ``opspace.compute``/``behaviour``/``spectrum`` never consume a
  root cell. ``obs["pseudotime"]`` enters the drift fit as a
  smoothness prior (not a direction), and the drift is velocity-
  free in the sense that its sign is set only by Step 2's velocity
  prior, which must itself be derivable from one of the physical
  orderings above.
* Latent-space arrows on figures come from the drift ``f(z)``, not
  from annotation. Overlaying cluster labels is fine (R15 Panel
  A); drawing arrows from "progenitor" to "terminal" is not.
* The R15 driver-table split into ``responders`` (right-ev) and
  ``drivers`` (left-ev) is likewise direction-free: both tables
  describe the local Jacobian's leading mode, and the biology of a
  terminal state vs a progenitor is read off from the operator's
  behaviour, not injected.

**Paul15 zero-holder is the non-stationary-snapshot finding.** The
2000-gene R16 run reports zero ``holder`` cells across all 2730
cells by the latent operator (``frac(Re > 0) = 1.00`` on every
lineage group), consistent with Paul15 being a snapshot of ongoing
differentiation rather than a set of attained steady states. The
∇η-tangent reduction of the gene-space ``J`` peels off a holder
minority exactly on the terminal clusters (37% of erythrocytes,
10% of monocytes; megakaryocytes and progenitors stay fork at
100%) — the ridge-bias signature from R14 §9. Full numbers are in
``benchmarks/figures/paul15_bench_compare.md``.

**R18 (clock is a check, never a signal, recorded 2026-10-13).**
Sampling-day labels (and any other physical time axis) are used
*only* to check that the lineage-path ordering of ``obs
["pseudotime"]`` (or the equivalent latent flow coordinate)
increases with the day label. They never enter the drift fit, the
eigenvalue computation, or the behaviour classifier. All reported
eigenvalues stay in ``clock_calibrated = False`` units
(``(relative)`` label on the figures); if a dataset has a tight
clock and the check passes, we may *also* report calibrated
eigenvalues alongside, but the relative pair stays the default.

**Phase D (redefined, 2026-10-13).** The hidden-driver synthetic
moves after Phase D. Phase D adds:

1. ``opspace.drivers``: left-ev drivers + sensitivity (see this
   file's commit log). Done.
2. R17/R18 recorded here. Done.
3. ``scjdo2.data.larry`` loader for Weinreb 2020 (GSE140802;
   raw counts, clone barcodes, day labels).

   **Data provenance (recorded 2026-10-13).** GEO GSE140802
   distributes the in-vitro counts matrix as
   ``GSM4185642_stateFate_inVitro_normed_counts.mtx.gz`` (2.07 GB
   compressed). The MatrixMarket header is ``coordinate real
   general`` with floating-point entries; the AllonKleinLab
   paper-data README confirms the normalisation is
   ``norm_ij = (umi_ij / total_i) · S`` with ``S = median_total``.
   ``cospar.datasets.hematopoiesis()`` downloads from
   ``wangshouwen.lab.westlake.edu.cn/.../LARRY_adata_preprocessed.
   h5ad``, which returns ``HTTP 403 Forbidden`` as of 2026-10-13.
   ``GSE140802_RAW.tar`` (4.67 GB) is a repackaging of the same
   per-GSM supplementary files — no per-sample raw 10x matrices
   are present.

   **However, Weinreb's normalisation is losslessly invertible.**
   The row-sum of the normed matrix is exactly ``S`` for every
   cell (verified: row-sums 2878.7490 within float noise across
   all 130,887 cells). For each cell the minimum positive normed
   value is achieved at ``umi_ij = 1``, so ``norm_min_i = S /
   total_i`` and ``umi_ij = round(norm_ij / norm_min_i)`` recovers
   the integer UMI exactly. ``scripts/assemble_larry_raw.py``
   streams the ``.mtx.gz``, applies the per-row inversion, and
   writes ``benchmarks/data/larry/larry_inVitro_raw.h5ad``.
   Verified on the full dataset: max rounding error over all 177M
   nonzero entries = ``3.41e-13`` (pure float64 roundoff); recovered
   per-cell UMI totals min/median/max = 300/2249/29043 (physically
   plausible for inDrops). SHA-256 of the assembled file is
   recorded in ``benchmarks/data/larry/CHECKSUMS.txt``.
   ``scjdo2.data.load_larry()`` preferentially reads this h5ad; the
   normed_counts.mtx.gz and the auxiliary supplementary files
   (``gene_names.txt.gz``, ``metadata.txt.gz``,
   ``clone_matrix.mtx.gz``) stay in the cache dir as the raw
   inputs. ``cospar`` is in the env but not called at load time.
4. Clone arm in Step 2: a clone-split map over Panel A plus a
   clone-consistency score for the drift (fraction of clone pairs
   where integrating ``f`` forward from the earlier-day member
   reaches a cell within ``ε`` of the later-day member).
5. Velocity arm: fit scVelo dynamical on LARRY, wrap the per-cell
   velocity in a small MLP(latent → velocity), use that MLP as
   the Step 2 drift prior, and report clone-consistency with and
   without the velocity prior; where the two drifts disagree,
   name the cell types.
6. Clock check: day labels only check lineage-path pseudotime
   monotonicity; eigenvalues stay relative.

**Paul15 archetype-5 negative finding (recorded 2026-10-13).** On
the 2000-gene R16 run, the ery–myeloid-split archetype (5; fork,
``lead_re = +0.107``, 50% ery · 15% myelo · 11% MEP) ranks Gata1
and Sfpi1 modestly in both views:

| gene  | responders rank | drivers_left rank | fisher | snr   | reason |
|-------|-----------------|-------------------|--------|-------|--------|
| Gata1 | 464             | 493               | 5.52   | 0.236 | ok     |
| Sfpi1 | 1404            | 1615              | 1.01   | 0.151 | ok     |

Neither TF makes the top-N of either the right- or the left-
eigenvector on either side of the operator. This is a **negative
finding**: the Paul15 fork archetype does not encode the Gata1 /
Sfpi1 switch on either side of the local operator — the switch
isn't a saddle *inside this snapshot*. Consistent with the zero-
holder result on the same dataset: in a non-stationary mid-
differentiation snapshot there is no basin structure, so the
ery–myelo bifurcation does not appear as a saddle whose
leading-mode left-ev weighs Gata1 or Sfpi1. LARRY — which carries
clone-barcoded cross-day lineage history and includes
pre-commitment progenitors — is the proper test: a saddle
encoding the Gata1 / Spi1 switch should show up at the
annotated clone-split positions, driven by Spi1 on one axis and
Gata1 on the other, with holder behaviour on the downstream
terminal states. If that structure appears on LARRY, Phase C's
operator decomposition reads the ery/myelo switch correctly on a
dataset that actually has the switch; if it doesn't appear, the
scJDO2 pipeline cannot see the switch on either dataset and R15's
"behaviour from latent, drivers from gene-space lift" rule needs a
rethink. Full gate below.

Full gate on LARRY: behaviour counts by annotated fate, holders
on terminal states, fork at the clone-split positions, driver
tables with Gata1 / Spi1 (note the symbol: LARRY uses Spi1, Paul15
uses the Sfpi1 synonym).

**LARRY §4–§6 gate result (recorded 2026-10-13, 10k-cell clone-
preserving subsample, 2000 HVG, max_live=150, see
``benchmarks/v2/larry_10k/``). The headline is §4; §6 and the
archetype tables are context, not recovery claims.**

* **§4 clone arm — THE finding (negative).** 369 clone-split
  events detected (clones spanning ≥2 annotated fates on later
  days), 47,946 cross-day clone pairs on the subsample. Clone-
  consistency of the residual-free DSM drift (Step 2 as currently
  implemented, no lineage prior) = **15 / 5000 = 0.3%** at the
  median-kNN latent radius. Integrating the fitted drift forward
  from an earlier-day clone member almost never reaches the
  clone's later-day member. **The DSM drift on a differentiating
  snapshot carries no lineage information.** This replicates the
  transient-pitchfork pattern and the Paul15 non-stationary-
  snapshot pattern (zero holders) on a third dataset — the
  density-score drift carries neither basins nor direction. A
  measured flow is required.
* **§6 clock check — not a recovery claim.** 6/11 annotated
  fates have monotone latent-flow pseudotime with sampling day;
  the 5 non-monotone fates (Baso, Monocyte, Erythroid, Eos, pDC)
  mostly carry very few day-2 cells so the median collapses to
  the d4/d6 centroid. R18 fixes this as a check reporting data,
  not a drift-fit signal.
* **§5 velocity arm — skipped.** LARRY GSE140802 publishes total
  UMIs only (no spliced / unspliced layers). scVelo dynamical
  mode cannot run without a velocyto re-alignment of the raw
  FASTQs; the re-alignment is scheduled as a background job.
* **Behaviour counts by fate — base rate, not a result.** Zero
  holder cells on every fate (same transient-pitchfork pattern
  as Paul15); fork fractions span 0.21–0.98 depending on fate.
  Terminal fates (Erythroid, Meg, Mast) hit 0.9+ fork; the
  "fork at terminal states" observation is the base-rate finding
  that *every* cell on this dataset classifies as fork or
  growing_oscillator — the pipeline reports no basins anywhere.
* **"Fork at clone-split positions" — base rate.** 368 of 369
  nearest-cell classifications at clone-split positions are
  fork. Because LARRY under R15's latent-operator classifier
  reports fork on nearly every cell (base rate ≈ 100% fork +
  growing_oscillator, zero holders), the clone-split-position
  check is non-informative on this run: a random set of
  positions would score the same way. This is **not** evidence
  that the pipeline recovers the clone-split saddle.
* **Gata1 / Spi1 archetype drivers — tiny Fisher mass, not
  recovery.** On archetype 6 (fork, lead_re +0.122, 1534 cells)
  both Gata1 and Spi1 report ``reason=ok`` with Fisher mass of
  **0.035** and **0.028** respectively — three orders of
  magnitude below the Paul15 archetype-5 Fisher masses of 5.52
  and 1.01 that were themselves not top drivers there. SNRs of
  0.015 and 0.011 are at the noise floor of the Fisher-weighted
  tangent reduction. Reporting that both TFs appear with
  ``ok`` reason overstates the signal: the measurement is
  compatible with "cap-adjacent entries happened to survive on
  that archetype", not with a saddle encoding the Gata1 / Spi1
  switch. The result is **neither a positive nor a negative**
  on the saddle question; it is below the resolution of the
  density-score pipeline, as predicted by §4.

**Conclusion for V2_PLAN (recorded 2026-10-13).** *A density-
score drift on a differentiating snapshot carries neither basins
nor lineage direction.* This conclusion is now supported on three
datasets:

1. The transient pitchfork synthetic (Phase C gate; zero holders,
   lead_re everywhere positive during the sweep).
2. Paul15 2000-gene (zero holders across all 2730 cells by the
   latent operator; archetype-5 does not encode the Gata1 /
   Sfpi1 switch — the archetype-5 negative finding).
3. LARRY in-vitro 10k (zero holders across all fates;
   clone-consistency 0.3%; the archetype-6 Gata1 / Spi1 ``ok``
   verdict carries Fisher mass at the noise floor and does
   not constitute recovery of the switch).

**A measured flow is required.** The next arm — Phase D §7
below — uses metabolic labelling (scNT-seq) to put ``dη/dt`` on
a per-cell per-hour footing, making Step 2's drift a *measurement*
instead of a density score. If that drift is lineage-consistent
on scNT-seq hematopoiesis, holder behaviour appears on
terminals, and the Gata1 / Spi1 fork sits at the erythroid–
myeloid split with Fisher mass above the noise floor, then the
V2_PLAN main hypothesis holds subject to a measured-flow source.
If not, R15's rule needs a rethink.

**Phase D §7 (labelling arm, recorded 2026-10-13).** Dynamo
hematopoiesis scNT-seq (Qiu et al., 2022). Dataset carries ``new``
and ``total`` UMI count layers plus the labelling pulse duration
``t_label``; the labelling pulse makes the rate equation closed-
form:

    dμ_g/dt = α_g(z) - γ_g · μ_g

with per-gene synthesis α_g(z) and degradation γ_g recovered from
the ratio of new to pre-existing UMIs over the pulse. ``dη/dt``
is then in per-hour units, and ``clock_calibrated = True``.

Step changes under §7:

* ``scjdo2.rates`` gains a ``LabelledRateModel`` (NB on new + NB
  on pre-existing = total − new, with synthesis α_g(z) via a
  small MLP(z → α_g) and degradation γ_g per-gene; MAP with the
  pulse-duration prior from the dataset).
* ``scjdo2.pipeline`` adds ``fit_labelled_rate_field`` that
  replaces the DSM-score drift of Step 2 with ``f(z) = MLP(z, θ)``
  trained on the measured per-cell ``dη/dt`` (the measured flow).
  The MLP is differentiable so Phase C's autodiff Jacobian path
  works unchanged.
* ``scjdo2.data`` gains a dynamo-hematopoiesis loader
  (``load_dynamo_hematopoiesis``). Primary path:
  ``dynamo.sample_data.hematopoiesis()``; fallback: the GEO deposit
  linked from Qiu 2022 Supplementary.
* Full gate on dynamo-hematopoiesis: behaviour counts by annotated
  fate, holders on terminal states, fork near the erythroid–
  myeloid split, Gata1 / Spi1 ranks, eigenvalues in per-hour
  units with ``clock_calibrated: True`` reported on the figures.
* **Velocyto re-alignment of LARRY** runs as a background job in
  parallel. If the splicing-aware LARRY artefact lands before the
  Phase D §7 gate finishes, we also report scVelo-prior drift
  clone-consistency on LARRY (the §5 remedy) as a secondary
  measured-flow source for comparison.

**§7 gate on Dynamo hematopoiesis — AUDIT REQUIRED BEFORE
RECOVERY CLAIM (restated 2026-10-13).** The initial §7 writeup
overstated the result: 0.2% holders on 1947 cells and 95% fork at
the annotated split are the *same pattern* as Paul15 and LARRY in
different units. **Measured flow calibrates units and raises
Fisher mass; whether it yields holders is undetermined until the
five audit checks are in.** Running them now (outputs under
``benchmarks/v2/dynamo_hematopoiesis/audit_*.csv``):

**(1) Permutation null on the measured-flow drift** (10 refits
with the velocity-vector-cell assignment shuffled, same MLP
settings and seeds offset).

    quantity           real fit     perm null (median)   perm q05–q95
    frac_fork          0.981        1.000                [1.000, 1.000]
    frac_holder        0.019 (37)   0.000                [0.000, 0.000]
    median_lead_re     0.0512 /h    0.0069 /h            [0.0057, 0.0107]
    q95(lead_re)       0.0832 /h    ~0.020 /h            [0.013, 0.032]

  Real lead_re distribution is ~8× the permutation-null median
  with no overlap. The ~2% holder minority NEVER appears in the
  null — zero holders in 10/10 permutations. The signal is above
  noise. The null's 100% fork is base rate; the real fit's small
  holder minority is a non-zero but weak deviation from base rate.

**(2) MLP held-out velocity R² + regularisation.** 80 / 20 split,
same architecture (SiLU MLP hidden=(96, 96, 96)), weight-decay
grid ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}, 1500 epochs:

    weight_decay   held-out R²
    0              0.848
    1e-5           0.848
    1e-4           0.848
    1e-3           0.848
    1e-2           0.848

  R² ≈ 0.85 across all regs — the MLP captures the measured
  velocity field well and weight decay is not load-bearing here.
  Jacobian estimates are driven by the data, not noise.

**(3) d_latent + tangent-rank distribution.** d_latent = 10.
Per-cell tangent rank from the permutation-null-cut SVD on the
Fisher-weighted ∇η decoder Jacobian:

    rank 2  508 cells (26%)
    rank 3  1396 cells (72%)
    rank 4  43 cells (2%)
    min=2, median=3, max=4

  **The operator per cell lives in a 2–4-dim subspace, not 10.**
  The behaviour verdict (fork vs holder) is on a 3×3 restricted
  block for most cells. Archetype blocks are QR'd to the max
  member rank — at most 4 dimensions. This is a *major* qualifier
  on any "the lead eigenvalue says X" statement.

**(4) GATA1 / SPI1 ranks per archetype** (both tables, out of
1956 genes). All 8 archetypes classify fork; both TFs report
``reason=ok`` with ``fisher`` ≥ 0.15 on every archetype:

    arch  lead_re(1/h)  GATA1 resp / drv rank  (fisher)   SPI1 resp / drv rank  (fisher)
    0     0.198         51   / 79              1.57       103 / 172             0.68
    1     0.153         256  / 154             1.48       104 / 222             0.80
    2     0.122         172  / 135             1.29       135 / 185             0.98
    3     0.152         316  / 467             1.07       86  / 131             1.08
    4     0.024         139  / 70              0.87       582 / 309             0.90
    5     0.113         334  / 1207            1.23       198 / 996             0.91
    6     0.173         121  / 96              1.74       257 / 307             0.50
    7     0.156         436  / 450             0.15       184 / 138             1.88

  Ranks mostly in [51, 582] out of 1956 (top 2.5–30%). The driver
  (left-ev) ranking differs substantially from the responder
  ranking on arch 5 (GATA1 334→1207, SPI1 198→996) — the archetype
  that lost its signal under left-ev. **GATA1 and SPI1 are
  consistent top-tail drivers, not top-10 drivers.** Fisher masses
  (0.15–1.88) are indeed three orders of magnitude above the
  LARRY DSM run's 0.028–0.035, confirming the "measured flow
  raises Fisher mass" claim.

**(5) Cell-cycle projection on archetype eigvecs.** Seurat S + G2M
panel (Tirosh 2016, 97 genes) and Dynamo's own
``cell_phase_genes`` curated panel (61 genes) — of these, **only 4
genes overlap with the processed HVG set** (dynamo's HVG filter
excluded most cycle genes). Projecting ``|v_g|²`` of each
archetype's leading eigvec onto those 4 cycle genes:

    archetype  table         frac_seurat_cycle  frac_dynamo_cycle
    0          responders    0.207              0.217
    0          drivers_left  0.180              0.188
    1          responders    0.158              0.165
    1          drivers_left  0.158              0.164
    2          responders    0.090              0.097
    2          drivers_left  0.101              0.107
    3          responders    0.112              0.117
    3          drivers_left  0.094              0.098
    4          responders    0.058              0.062
    4          drivers_left  0.057              0.061
    5          responders    0.199              0.210
    5          drivers_left  0.104              0.110
    6          responders    0.166              0.180
    6          drivers_left  0.189              0.203
    7          responders    0.021              0.022
    7          drivers_left  0.020              0.021

  **Baseline expectation under uniform |v_g|² weight: 4 / 1956 =
  0.2%.** Measured: 2–22% of eigvec squared weight falls on just
  4 cycle genes — 10–100× over the uniform baseline. On
  archetypes 0, 5, 6 (the ones with the largest lead_re) the
  cycle contamination is 17–22%. **The archetype operator is
  substantially cell-cycle-aligned.** The 2–22% figure is a lower
  bound: of the 61 cell-cycle genes in dynamo's own panel, only
  4 survived the HVG filter, so the true cycle load on the eigvec
  is almost certainly higher once the full cycle panel is in the
  gene set.

**Verdict after (1)–(5), restated.** The measured flow does:

  (i) *calibrate units* — lead_re is in per-hour, so the clock
      caveat (R18) resolves to ``clock_calibrated = True`` on this
      run;
  (ii) *raise Fisher mass* — GATA1 and SPI1 reach Fisher 0.15–1.88
       vs the LARRY DSM run's 0.028–0.035; and
  (iii) *distinguish signal from permutation null* — the real
        lead_re distribution and the small holder minority are
        both above the null band.

The measured flow does **not** yet:

  (iv) *recover holders on terminal states as a biological
       finding* — the 0.2% overall holder fraction is the same
       pattern as Paul15 and LARRY in different units; the perm
       null confirms the real fit's holders are not pure noise
       but it does not confirm they correspond to terminal-state
       equilibration vs cell-cycle-driven damping;
  (v) *recover a Gata1/Spi1 saddle at the ery/myelo split* — all
      the "fork at MEP-like / GMP-like" and the fork labels on
      every archetype are compatible with cell-cycle-driven
      divergence in a low-dim tangent (3 dims per cell on
      median) with 17–22% of the eigvec weight on cycle genes on
      the strongest archetypes.

**§7a (recorded 2026-10-13, outputs under
``benchmarks/v2/dynamo_hematopoiesis/s7a_*.csv``). Cycle-genes
back in the embedding; cycle subspace projected out of the operator.**

Per-cell S + G2M scores from the Seurat S / G2M panels (only 4 of
97 genes survive the dynamo HVG filter: 1 S + 3 G2M). Latent-frame
cycle directions by OLS regression of each score onto ``X_pca``;
orthonormalise to ``U_cycle ∈ R^(10 × 2)``. Phase stored as
``obs["_phase"] = atan2(G2M, S)``. Three variants compared:

  A. **Unprojected** — the §7 fit.
  B. **Score-regressed** — residualise ``velocity_pca`` on
     ``[S, G2M]`` per cell (OLS), refit MeasuredDrift, run full
     Phase C refit.
  C. **Rotation-projected** — keep A's full Phase C fit; replace
     the latent operator by ``J_c ← (I - UU^T) J_c (I - UU^T)``
     per cell BEFORE the eigendecomp. Projected directions are
     tagged ``reason=known_rotation`` in the ledger. The gene-space
     ``J`` is unchanged.

Side-by-side on the latent operator:

    variant               n   fork  holder  frac_h  cycle_w (med / q95 of |U^T v|²/|v|²)
    A unprojected        1947 1910    37    1.9%    0.079 / 0.362
    B score-regressed    1947 1909    38    1.95%   0.068 / 0.343
    C rotation-projected 1947 1939     8    0.4%    ~0    / ~0

  - **Holders by annotated fate (all three variants):** 100% on
    terminals (Bas + Meg). A: 22 Bas + 15 Meg; B: 27 Bas + 11
    Meg; C: 7 Bas + 1 Meg. HSC, MEP-like, GMP-like, Ery, Mon,
    Neu: zero holders under every variant.
  - **Cycle weight in the leading latent eigvec**: variant A's
    median 7.9% (q95 36%) is high — the top eigvec of the raw
    operator is substantially cycle-aligned on at least 5% of cells.
    Variant B's score regression barely moves this (6.8% / 34%),
    because OLS on ``velocity_pca`` only removes the linear part
    of the cycle drift. Variant C's projection drives it to zero
    by construction and drops the holder count 37 → 8 (22%
    survival).

**37-holder deep dive (variant A).**
  - **100% on terminal fates** (Bas + Meg) and **100% from the
    3 h pulse batch**. The 5 h pulse batch (818 cells) contributes
    zero holders.
  - 0 G0-like under ``|S|<0.2 ∧ |G2M|<0.2 ∧ cycle_logexp < median``
    — the panel is too small to tell (only 1 S gene survives HVG).
  - Mean cycle log-expression = 0 for 32 of the 37 cells (no
    cycle-gene UMIs at all); the 5 remaining holders have
    ``mean_cycle_logexp`` between 0.17 and 1.21.
  - The pulse-duration batch confound (day 3 vs day 5) is a direct
    threat: the drift MLP sees velocity vectors in two chemically
    different units stacked into one latent frame, and the entire
    holder minority sits on the slow-pulse side.

**GATA1 / SPI1 under cycle variants.**
  - Variant A: GATA1 responder ranks ∈ [51, 436], SPI1 ∈ [86, 582].
  - Variant B (score-regressed): GATA1 ranks improve on 4 of 8
    archetypes (median goes to ~130), but SPI1 degrades sharply —
    reports ``reason=capped`` with Fisher = 0 on archetypes 5 and 6
    (the strongest lead_re archetypes). Score regression *reduces*
    the SPI1 signal on the two archetypes where it was clearest.
  - Variant C: GATA1 / SPI1 ranks unchanged (gene-space ``J`` is
    identical to A); only the archetype's behaviour label depends
    on the latent projection, and all 8 archetypes stay fork.

**Recovery verdict (refined).** The recovery criterion was
"holders survive on terminals AFTER the projection AND cycle
weight drops under 5%". Variant C hits both — 8 holders survive,
all on terminals; cycle weight ~0 by construction. **But** every
one of the 8 surviving holders is from the 3 h pulse batch and
the cycle panel is only 4 genes deep. The 5 h pulse batch (818
cells) contributes zero holders in every variant. The holder
signal is confounded with pulse duration, not clearly with
terminal biology. Score regression (B) does not move the needle
(cycle weight drops <1 pt, SPI1 recovery degrades on strong
archetypes), confirming that OLS-on-scores is the wrong lever —
the operator rotation projection (C) is cleaner but still leaves
the batch confound unresolved. **The §7a result is: the measured-
flow pipeline produces a terminal-biased holder minority that is
dominated by cell-cycle rotation (29 of 37 holders are inside the
cycle subspace), with the residual 8 holders carrying a pulse-
duration batch artifact.** No recovery claim.

**Next.** Either (a) subset the dataset to a single pulse batch
and re-run, which halves the sample but removes the clock
confound; or (b) treat ``pulse_duration`` as a per-cell clock
scale in the drift fit (``clock_scale`` input in DriftConfig
already exists for the DSM path; needs the equivalent in
MeasuredDrift). Both are § 7b candidates.

**§7b (recorded 2026-10-13, outputs under ``s7b_*.csv``).**
Pulse duration as a per-cell clock scale, then per-batch splits.

**(1) Clock scale.** Dynamo's ``uns['dynamics']['t']`` is a
constant 5 h for every cell, so the stored ``velocity_pca`` is
reported on a 5-h-equivalent clock for both the 3-h batch
(JL12_0 + JL12_1, 1129 cells) and the 5-h batch (JL_10, 818
cells). We rescale ``velocity_pca_c * (t_ref / t_c)`` with
``t_ref = 5`` so 3-h cells carry ``× 5/3 = 1.667``, 5-h cells
``× 1``. ``obs['clock_scale']`` records the scale. On pre-scaling
velocity norms, median-3h / median-5h = 1.954 (the 3-h batch
already dominates magnitude); after scaling, the ratio widens to
3.257 — multiplicative clock scaling moves the 3-h batch further
from the 5-h batch, consistent with the stored dynamo velocity
being more biased than a single scalar can repair.

**(2) + (3) Variant A/C under clock-scaled velocity AND per-
batch splits** (all MeasuredDrift refit from scratch, same
MLP settings, 1500 epochs):

    fit                  variant  n     n_hold  frac   cycle_w_med  per-fate holders (Bas / Meg / Mon / Ery / Neu)
    all_unscaled (§7a)   A        1947  37      1.9%   0.079        22 / 15 / 0 / 0 / 0
    all_unscaled (§7a)   C        1947   8      0.4%   ~0            7 /  1 / 0 / 0 / 0
    all_clock_scaled     A        1947  46      2.4%   0.068        16 / 29 / 1 / 0 / 0
    all_clock_scaled     C        1947  11      0.6%   ~0            4 /  7 / 0 / 0 / 0
    batch_3h             A        1129  68      6.0%   0.123        24 / 34 / 10 / 0 / 0
    batch_3h             C        1129  17      1.5%   ~0            5 /  8 /  4 / 0 / 0
    **batch_5h           A         818   0      0.0%   0.024         0 /  0 /  0 / 0 / 0**
    **batch_5h           C         818   0      0.0%   ~0            0 /  0 /  0 / 0 / 0**

  **The 5-h batch alone produces ZERO holders in every variant.**
  The 3-h batch alone produces 68 (6.0%) under A — the largest
  holder fraction in any §7 experiment — and 17 (1.5%) under C.
  Clock-scaling the pooled data (46 / 11) sits between the two
  batches but closer to the 3-h-batch result because that batch
  is larger. Rotation projection C survives at ~1.5% on the 3-h
  batch and ~0% on the 5-h batch; same pattern as A at smaller
  magnitude.

**(4) Holder-cell depth check.** For every holder cell in every
fit above:

    fit                     n_holder   UMI_ratio_vs_median        n_genes_ratio_vs_median
    all_clock_scaled / A    46         2.12 (q05 1.40, q95 3.32)  1.41 (median)
    all_clock_scaled / C    11         1.69 (q05 1.37, q95 2.81)  1.22
    batch_3h / A            68         1.96 (q05 1.35, q95 2.99)  —
    batch_3h / C            17         ~1.7                        —

  **Holders are the HIGH-UMI tail, not the low-count tail.**
  Median holder UMI is ~2× the dataset median; q05 is 1.35× (so
  no holder is a low-depth call). These are well-sequenced,
  gene-rich cells. Not noise; a feature of the drift fit on the
  3-h batch specifically.

**Verdict.** The holder minority is a **3-h-batch artifact**:

  * The 5-h batch (818 cells, dynamo's actual rate-fit reference)
    is holder-free under every treatment: unprojected, cycle-
    rotation-projected, clock-rescaled, and even by itself.
  * The 3-h batch (1129 cells) carries every single holder in
    every variant; the holders are HIGH-UMI, HIGH-gene-count
    cells confined to Bas / Meg / Mon.
  * Multiplicative clock scaling by ``t_ref / t_c`` moves holders
    around but does not reconcile the two batches — dynamo's
    ``velocity_pca`` is more deeply tied to the 5-h fit than a
    scalar scale can repair.

**No recovery claim.** The §7 "holders on terminal Meg + Bas"
observation does not replicate in the 5-h batch, which is the
batch dynamo actually fit the rate model on. Holders survive the
cycle-rotation projection on the 3-h batch only, exactly where
dynamo's rate parameters are mis-matched. R15's rule is neither
confirmed nor falsified by Dynamo scNT-seq hematopoiesis — the
dataset in its distributed form cannot cleanly separate measured-
flow dynamics from pulse-duration bias.

**Next.** Three remaining options for a defensible measured-flow
gate:

  * Refit the labelled rate model from new + total counts
    per-cell, with ``t_c`` entering the NB likelihood as
    ``E[new_cg] / E[total_cg] = 1 - exp(-γ_g · t_c)`` so γ is
    estimated on a per-hour footing consistent across batches
    (a proper §7b (1), replacing dynamo's upstream fit; needs a
    LabelledRateModel). Then redo variants A/C.
  * Use dynamo's per-batch ``time_3_gamma`` / ``time_5_gamma``
    columns (already in ``var``) to rebuild per-batch velocity
    and compare, which bypasses the global ``velocity_pca``.
  * Switch to a dataset where the labelling clock is uniform
    (e.g. scEU-seq organoid or the single-pulse scNT-seq
    neuron dataset, both already surfaced by ``dynamo.sample_data``).

**Velocyto re-alignment of LARRY** is still open; the gate on
Dynamo hemat.scNT-seq is independent of it and the two
measured-flow sources (labelling + splicing) will cross-check
each other when the LARRY re-alignment lands.

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
