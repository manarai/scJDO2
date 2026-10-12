"""Toggle-switch pitchfork system for Phase B acceptance.

Source: `scJDO/scratchpad_run/saddle_readouts_midinterval.py` (α-ramp over a
symmetric 2D toggle switch). We generalise the mid-interval case so τ_crit is
a parameter rather than fixed at 0.5. The α window is centred around α_crit =
(4/3) · 3^(-1/4) and sized by ``width``; the location of α_crit within the
window is then set by τ_crit:

    α_min = α_crit − τ_crit · width
    α_max = α_crit + (1 − τ_crit) · width
    τ_crit = (α_crit − α_min) / (α_max − α_min)   (invariant)

Default ``width = 1.98`` matches the original mid-interval control (Δ = 0.99,
total width 1.98) so τ_crit = 0.5 reproduces that script's α range.

Phase B's acceptance test instantiates this generator at τ_crit ∈ {0.1, 0.3,
0.5}, fits scJDO2 (Steps 1–2–5–7–8), and checks that the leading real
eigenvalue curve argmaxes within ±0.05 of each τ_crit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from anndata import AnnData
from scipy.optimize import brentq
from sklearn.decomposition import PCA

# Analytic bifurcation point of the symmetric toggle switch
# dz_x/dt = α / (1 + z_y^4) − z_x,   dz_y/dt = α / (1 + z_x^4) − z_y
ALPHA_CRIT: float = (4.0 / 3.0) * (3.0 ** (-0.25))
DEFAULT_WIDTH: float = 1.98  # matches scJDO/scratchpad_run/saddle_readouts_midinterval.py


def analytic_jacobian(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of the toggle flow at ``(z, α)``.

    ``z`` is ``(n, 2)``, ``alpha`` is ``(n,)``. Returns ``(n, 2, 2)``.
    """
    z = np.asarray(z, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    x, y = z[:, 0], z[:, 1]
    off_fx = -4.0 * alpha * y ** 3 / (1.0 + y ** 4) ** 2
    off_fy = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    J = np.zeros((z.shape[0], 2, 2), dtype=np.float64)
    J[:, 0, 0] = -1.0
    J[:, 0, 1] = off_fx
    J[:, 1, 0] = off_fy
    J[:, 1, 1] = -1.0
    return J


def _toggle_sym_fp_robust(alpha: float) -> float:
    """Symmetric FP of the toggle: solve ``x^5 + x = α``. Reliable for all
    ``α >= 0`` (the iterative map ``x ← α/(1+x⁴)`` only converges for
    ``α < α_crit``, so we bracket a root instead).
    """
    if alpha <= 0:
        return 0.0
    return float(brentq(lambda x: x ** 5 + x - alpha, 0.0, max(alpha, 1e-6) + 1.0))


@dataclass
class PitchforkMeta:
    tau_crit: float
    alpha_crit: float
    alpha_min: float
    alpha_max: float
    width: float
    n_cells: int
    n_genes: int
    n_latent: int
    noise_sigma: float
    T_total: float
    dt: float
    sigma_sde: float
    seed: int


def _toggle_sym_fp(alpha: float, tol: float = 1e-10, max_iter: int = 400) -> float:
    """Symmetric fixed point of the toggle: ``x = α / (1 + x^4)``."""
    x = max(alpha, 1e-3)
    for _ in range(max_iter):
        xn = alpha / (1.0 + x**4)
        if abs(xn - x) < tol:
            break
        x = xn
    return float(x)


def _toggle_drift(z: np.ndarray, alpha: float) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    fx = alpha / (1.0 + y**4) - x
    fy = alpha / (1.0 + x**4) - y
    return np.stack([fx, fy], axis=-1)


def _simulate_trajectories(
    n_cells: int,
    alpha_min: float,
    alpha_max: float,
    T_total: float,
    dt: float,
    sigma_sde: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Snapshot sampler: run an α-ramp SDE and record a snapshot per cell at a
    random time point. Returns ``(z, t)`` of shapes ``(n_cells, 2)`` and
    ``(n_cells,)`` with ``t ∈ [0, 1]`` (τ = step·dt / T_total)."""
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T_total / dt))
    target_step = np.sort(rng.integers(1, n_steps + 1, size=n_cells))
    t_per_cell = (target_step * dt) / T_total

    z = np.full((n_cells, 2), _toggle_sym_fp(alpha_min), dtype=np.float32)
    z += 0.02 * rng.standard_normal(z.shape).astype(np.float32)

    z_out = np.zeros_like(z)
    sqrt_dt = float(np.sqrt(dt))
    idx = 0
    for step in range(1, n_steps + 1):
        a_t = alpha_min + (alpha_max - alpha_min) * (step * dt) / T_total
        z = (
            z
            + _toggle_drift(z, a_t) * dt
            + sigma_sde * sqrt_dt * rng.standard_normal(z.shape).astype(np.float32)
        )
        while idx < n_cells and target_step[idx] == step:
            z_out[idx] = z[idx]
            idx += 1
    return z_out.astype(np.float32), t_per_cell.astype(np.float32)


def generate_pitchfork(
    tau_crit: float = 0.5,
    width: float = DEFAULT_WIDTH,
    n_cells: int = 1500,
    n_genes: int = 200,
    n_latent: int = 20,
    noise_sigma: float = 0.3,
    T_total: float = 10.0,
    dt: float = 0.02,
    sigma_sde: float = 0.20,
    seed: int = 42,
    counts: bool = False,
    counts_depth: Optional[float] = None,
) -> AnnData:
    """Generate a toggle-switch pitchfork AnnData.

    The 2D latent state ``z`` is linearly projected to ``n_genes`` observed
    features (Gaussian loadings, unit-norm columns), then observed with
    additive Gaussian noise. ``X_pca`` on the AnnData holds a
    ``n_latent``-dim PCA of the observed matrix.

    Parameters
    ----------
    tau_crit : float
        Where the bifurcation falls in the (normalised) time axis ``[0, 1]``.
        Phase B acceptance uses ``{0.1, 0.3, 0.5}`` and expects the field to
        localise it within ``±0.05``.
    counts : bool
        If ``True``, store a Poisson sample of ``exp(X)`` in ``adata.X``
        instead of the Gaussian-observed matrix. Phase C's NB likelihood
        model uses this; Phase B does not.
    counts_depth : float or None
        Per-cell depth multiplier when ``counts=True``. ``None`` → use
        ``exp(X) * 10`` roughly.

    Returns
    -------
    AnnData
        ``X``                — ``(n_cells, n_genes)`` observed matrix (float32)
                               or Poisson counts if ``counts=True``.
        ``obsm["X_true"]``   — ``(n_cells, 2)`` true latent coordinates.
        ``obsm["X_pca"]``    — ``(n_cells, n_latent)`` PCA of ``X``.
        ``obs["pseudotime"]`` — ``(n_cells,)`` normalised time along the ramp.
        ``uns["pitchfork"]`` — :class:`PitchforkMeta` as a plain dict.
    """
    if not (0.0 < tau_crit < 1.0):
        raise ValueError(f"tau_crit must be in (0, 1); got {tau_crit}")
    if width <= 0:
        raise ValueError(f"width must be positive; got {width}")

    alpha_min = ALPHA_CRIT - tau_crit * width
    alpha_max = ALPHA_CRIT + (1.0 - tau_crit) * width

    z_true, t_pt = _simulate_trajectories(
        n_cells=n_cells,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        T_total=T_total,
        dt=dt,
        sigma_sde=sigma_sde,
        seed=seed,
    )

    rng_proj = np.random.default_rng(seed + 1)
    W = rng_proj.standard_normal((2, n_genes)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    X_obs = z_true @ W

    rng_g = np.random.default_rng(seed + 2)
    X_obs = X_obs + noise_sigma * rng_g.standard_normal(X_obs.shape).astype(np.float32)

    if counts:
        depth = float(counts_depth) if counts_depth is not None else 10.0
        rate = np.exp(X_obs - X_obs.max(axis=0, keepdims=True)) * depth
        X_mat = rng_g.poisson(rate).astype(np.float32)
    else:
        X_mat = X_obs

    pca = PCA(n_components=min(n_latent, n_genes), random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)

    adata = AnnData(X=X_mat)
    adata.obs_names = [f"c{i:05d}" for i in range(n_cells)]
    adata.var_names = [f"g{j:04d}" for j in range(n_genes)]
    adata.obsm["X_true"] = z_true
    adata.obsm["X_pca"] = Z_lat
    adata.obs["pseudotime"] = t_pt
    adata.uns["pitchfork"] = PitchforkMeta(
        tau_crit=float(tau_crit),
        alpha_crit=float(ALPHA_CRIT),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        width=float(width),
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        n_latent=int(n_latent),
        noise_sigma=float(noise_sigma),
        T_total=float(T_total),
        dt=float(dt),
        sigma_sde=float(sigma_sde),
        seed=int(seed),
    ).__dict__
    return adata


def _equilibrate_sde(
    alpha: float,
    n_cells: int,
    T_burn: float,
    T_sample: float,
    dt: float,
    sigma_sde: float,
    init_box: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integrate the toggle SDE at **fixed** ``α`` from random initial
    conditions until burn-in; then sample one snapshot per trajectory from
    the stationary window. Returns ``(n_cells, 2)`` snapshot positions.
    """
    z = rng.uniform(-init_box, init_box, size=(n_cells, 2)).astype(np.float64)
    sqrt_dt = float(np.sqrt(dt))
    n_burn = int(np.ceil(T_burn / dt))
    n_sample = max(1, int(np.ceil(T_sample / dt)))
    for _ in range(n_burn):
        z = z + _toggle_drift(z, alpha) * dt + sigma_sde * sqrt_dt * rng.standard_normal(z.shape)
    # pick a random step within the sample window per cell
    pick = rng.integers(0, n_sample, size=n_cells)
    out = np.zeros_like(z)
    for step in range(n_sample):
        z = z + _toggle_drift(z, alpha) * dt + sigma_sde * sqrt_dt * rng.standard_normal(z.shape)
        mask = pick == step
        if mask.any():
            out[mask] = z[mask]
    return out.astype(np.float32)


def generate_pitchfork_equilibrated(
    tau_crit: float = 0.5,
    width: float = DEFAULT_WIDTH,
    n_cells: int = 1200,
    n_genes: int = 200,
    n_latent: int = 20,
    noise_sigma: float = 0.3,
    T_burn: float = 20.0,
    T_sample: float = 5.0,
    dt: float = 0.02,
    sigma_sde: float = 0.20,
    n_alpha: int = 20,
    init_box: float = 2.5,
    seed: int = 42,
    counts: bool = False,
    counts_depth: Optional[float] = None,
) -> AnnData:
    """Equilibrated pitchfork — snapshot is a **mixture of stationary densities**.

    For each ``α`` on a grid spanning ``[α_min, α_max]`` we burn in the toggle
    SDE from random initial conditions in ``[-init_box, +init_box]^2`` for
    ``T_burn`` time units at fixed ``α`` (so each trajectory thermalises to
    its basin) and then draw one snapshot from the subsequent ``T_sample``
    window. Trajectories above the diagonal reach the upper stable branch,
    those below reach the lower branch; the SDE noise mixes near the saddle.

    ``obsm`` keys:
        ``X_true``   — ``(n_cells, 2)`` 2D state at the snapshot
        ``V_true``   — ``(n_cells, 2)`` deterministic drift ``f_toggle(z, α)``
        ``X_pca``    — ``(n_cells, n_latent)`` PCA of the Gaussian-observed matrix

    ``obs``:
        ``pseudotime`` ∈ [0, 1] with ``τ_crit`` marking ``α = α_crit``.

    ``uns["pitchfork"]`` records all generator parameters plus the gene
    loading matrix ``W`` and the PCA components, so an analytic 2D Jacobian
    can be projected to the 20D PCA basis later.
    """
    if not (0.0 < tau_crit < 1.0):
        raise ValueError(f"tau_crit must be in (0, 1); got {tau_crit}")
    if width <= 0:
        raise ValueError(f"width must be positive; got {width}")

    alpha_min = ALPHA_CRIT - tau_crit * width
    alpha_max = ALPHA_CRIT + (1.0 - tau_crit) * width
    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha, dtype=np.float64)

    # Deal cells across the α grid, as evenly as possible.
    per_alpha = np.full(n_alpha, n_cells // n_alpha, dtype=np.int64)
    per_alpha[: n_cells - per_alpha.sum()] += 1
    assert per_alpha.sum() == n_cells

    rng = np.random.default_rng(seed)
    z_true = np.zeros((n_cells, 2), dtype=np.float32)
    alpha_per_cell = np.zeros(n_cells, dtype=np.float64)
    idx = 0
    for a, k in zip(alpha_grid, per_alpha):
        if k == 0:
            continue
        z_true[idx:idx + k] = _equilibrate_sde(
            alpha=float(a), n_cells=int(k),
            T_burn=T_burn, T_sample=T_sample, dt=dt,
            sigma_sde=sigma_sde, init_box=init_box, rng=rng,
        )
        alpha_per_cell[idx:idx + k] = float(a)
        idx += k

    # Shuffle cells so pseudotime isn't sorted by position in the array
    perm = rng.permutation(n_cells)
    z_true = z_true[perm]
    alpha_per_cell = alpha_per_cell[perm]
    pseudotime = ((alpha_per_cell - alpha_min) / (alpha_max - alpha_min)).astype(np.float32)

    v_true = _toggle_drift(z_true.astype(np.float64), alpha_per_cell).astype(np.float32)

    rng_proj = np.random.default_rng(seed + 1)
    W = rng_proj.standard_normal((2, n_genes)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    X_obs = z_true @ W
    rng_g = np.random.default_rng(seed + 2)
    X_obs = X_obs + noise_sigma * rng_g.standard_normal(X_obs.shape).astype(np.float32)

    if counts:
        depth = float(counts_depth) if counts_depth is not None else 10.0
        rate = np.exp(X_obs - X_obs.max(axis=0, keepdims=True)) * depth
        X_mat = rng_g.poisson(rate).astype(np.float32)
    else:
        X_mat = X_obs

    pca = PCA(n_components=min(n_latent, n_genes), random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)

    adata = AnnData(X=X_mat)
    adata.obs_names = [f"c{i:05d}" for i in range(n_cells)]
    adata.var_names = [f"g{j:04d}" for j in range(n_genes)]
    adata.obsm["X_true"] = z_true
    adata.obsm["V_true"] = v_true
    adata.obsm["X_pca"] = Z_lat
    adata.obs["pseudotime"] = pseudotime
    adata.obs["alpha"] = alpha_per_cell.astype(np.float32)
    pitch_meta = PitchforkMeta(
        tau_crit=float(tau_crit),
        alpha_crit=float(ALPHA_CRIT),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        width=float(width),
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        n_latent=int(n_latent),
        noise_sigma=float(noise_sigma),
        T_total=float(T_burn + T_sample),
        dt=float(dt),
        sigma_sde=float(sigma_sde),
        seed=int(seed),
    ).__dict__
    pitch_meta["variant"] = "equilibrated"
    pitch_meta["T_burn"] = float(T_burn)
    pitch_meta["T_sample"] = float(T_sample)
    pitch_meta["n_alpha"] = int(n_alpha)
    pitch_meta["init_box"] = float(init_box)
    pitch_meta["gene_loading"] = np.asarray(W)
    pitch_meta["pca_components"] = np.asarray(pca.components_, dtype=np.float32)
    pitch_meta["pca_mean"] = np.asarray(pca.mean_, dtype=np.float32)
    adata.uns["pitchfork"] = pitch_meta
    return adata
