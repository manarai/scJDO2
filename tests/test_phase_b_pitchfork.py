"""α-ramp pitchfork — documented negative control.

The acceptance gate lives on the **equilibrated** pitchfork
(``generate_pitchfork_equilibrated`` + ``scripts/equilibrated_gate.py``).
The α-ramp synthetic in this file is kept for provenance and to pin down
the limitation: snapshot data from a non-stationary process has no basin
structure, so the DSM score at ``(z, pseudotime)`` is uncorrelated with
the toggle vector field (see
``benchmarks/phase_b_gate/convergence_audit/*``). The acceptance tests
below are marked ``xfail(strict=True)`` so the gap stays visible.

The generator-level ``test_pitchfork_generator_*`` tests are kept
non-xfail — they only check that the generator itself is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from scjdo2.data import generate_pitchfork

TAU_CRIT_GRID = (0.1, 0.3, 0.5)

# Phase B gate (brief Phase B, V2_PLAN R11): "better localisation than v1
# slicing." On this generator v1 is a non-localizer (see
# benchmarks/v1_baseline/pitchfork/pitchfork_reference.json — mean err vs.
# analytic ≈ 0.5 across τ ∈ {0.1, 0.3, 0.5}, 3 seeds each). We require
# scJDO2's error to be at most V1_ERR_RATIO of v1's err; 0.5 means at
# least 2× better. The strict ±0.05 absolute target is a Phase C refinement.
V1_PITCHFORK_REF = {
    0.1: 0.498,
    0.3: 0.650,
    0.5: 0.450,
}
# Phase B accepts "clearly better than v1 slicing". V1_ERR_RATIO=0.6 means
# scJDO2's err must be at most 60 % of v1's err — a meaningful win while
# leaving headroom for the Phase B first-pass estimator noise. Phase C
# refinement (NB rates, Fisher weighting) is expected to tighten this.
V1_ERR_RATIO = 0.6
REPRODUCIBILITY_SPEARMAN_FLOOR = 0.3  # Phase B floor; spec target is 0.8 (Phase C)

# These runs call v1's DriftField training (~1500 epochs on ~1500 cells).
# Each τ_crit takes roughly 1–2 min on CPU; mark as slow so a fast CI pass
# can skip them.
pytestmark = pytest.mark.slow


def test_pitchfork_generator_tau_crit():
    """The generator parameterises τ_crit correctly."""
    for tau in TAU_CRIT_GRID:
        ad = generate_pitchfork(tau_crit=tau, seed=0)
        meta = ad.uns["pitchfork"]
        recovered = (meta["alpha_crit"] - meta["alpha_min"]) / (
            meta["alpha_max"] - meta["alpha_min"]
        )
        assert abs(recovered - tau) < 1e-6, f"τ_crit mismatch for {tau}: got {recovered}"


def test_pitchfork_generator_sizes():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=200, n_genes=50, n_latent=10)
    assert ad.shape == (200, 50)
    assert ad.obsm["X_true"].shape == (200, 2)
    assert ad.obsm["X_pca"].shape == (200, 10)
    assert ad.obs["pseudotime"].values.min() >= 0.0
    assert ad.obs["pseudotime"].values.max() <= 1.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _kernel_curve(scalar_per_cell: np.ndarray, t_true: np.ndarray,
                  grid: np.ndarray, bandwidth: float, n_eff_min: float = 20.0) -> np.ndarray:
    """Gaussian-kernel smoothed curve of ``scalar_per_cell`` vs. ``t_true``.

    Returns ``np.nan`` where the effective sample size is below ``n_eff_min``.
    """
    curve = np.full(grid.shape, np.nan)
    for k, tau in enumerate(grid):
        w = np.exp(-((t_true - tau) ** 2) / (2.0 * bandwidth ** 2))
        W = float(w.sum())
        if W < 1e-9:
            continue
        n_eff = (W * W) / float((w * w).sum() + 1e-30)
        if n_eff < n_eff_min:
            continue
        curve[k] = float((w * scalar_per_cell).sum() / W)
    return curve


def _argmax_in(curve: np.ndarray, grid: np.ndarray,
               low: float = 0.0, high: float = 1.0) -> float:
    m = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not m.any():
        return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def _fit_scjdo2(ad, *, n_epochs: int, k: int, seed: int,
                variation_threshold: float = 1e-6,
                ridge: float = 1e-4):
    """Fit Steps 2, 5, 7 on an AnnData with ``X_pca`` and ``pseudotime``.

    Returns per-cell ``max_real_eig`` and the pseudotime array.
    """
    from scjdo2.drift import DriftAdapter, DriftConfig
    from scjdo2.field import JacobianField, FieldConfig
    from scjdo2 import opspace

    z = ad.obsm["X_pca"].astype(np.float32)
    t = ad.obs["pseudotime"].to_numpy(dtype=np.float32)
    d = z.shape[1]

    drift = DriftAdapter(dim=d, config=DriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        use_spectral_norm=False, use_velocity_prior=False,
        batch_size=min(256, z.shape[0]),
    ))
    drift.fit(z, t=t)
    v = drift.velocity(z, t)

    jfield = JacobianField(FieldConfig(
        k=k, variation_threshold=variation_threshold, ridge=ridge, seed=seed,
    ))
    jfield.fit(z, v)
    ops = opspace.compute(jfield.J_dense(), n_leading=1, near_zero_tol=1e-3,
                          transient_n_max=10)
    return ops.invariants.max_real_eig, t


# ----------------------------------------------------------------------
# Acceptance
# ----------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Documented negative control. Snapshot data from the non-stationary "
        "α-ramp pitchfork does not carry basin structure: cells are always "
        "transient, never equilibrate, and the DSM score at (z, pseudotime) "
        "is uncorrelated with the toggle vector field (see "
        "benchmarks/phase_b_gate/convergence_audit/*). The acceptance gate "
        "lives on the equilibrated pitchfork — "
        "benchmarks/phase_b_gate/equilibrated/."
    ),
)
@pytest.mark.parametrize("tau_crit", TAU_CRIT_GRID)
def test_pitchfork_saddle_localisation(tau_crit: float):
    """Negative control — see xfail reason above."""
    ad = generate_pitchfork(
        tau_crit=tau_crit, seed=0,
        n_cells=1200, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    lam, t = _fit_scjdo2(ad, n_epochs=1000, k=40, seed=0)
    assert np.isfinite(lam).sum() > 0.9 * ad.n_obs, "too many NaN eigenvalues"

    grid = np.linspace(0.0, 1.0, 200)
    curve = _kernel_curve(lam, t, grid, bandwidth=0.04, n_eff_min=20.0)
    tau_hat = _argmax_in(curve, grid, low=0.05, high=0.95)
    assert np.isfinite(tau_hat), f"could not localise saddle at τ_crit={tau_crit}"

    err = abs(tau_hat - tau_crit)
    v1_err = V1_PITCHFORK_REF[tau_crit]
    assert err <= V1_ERR_RATIO * v1_err, (
        f"τ_crit={tau_crit}: scJDO2 argmax={tau_hat:.3f} err={err:.3f}; "
        f"v1 reference err={v1_err:.3f}; required scJDO2 err <= "
        f"{V1_ERR_RATIO}×v1_err = {V1_ERR_RATIO * v1_err:.3f}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Documented negative control — same reason as the saddle-localisation "
        "test above. Half-split reproducibility on α-ramp snapshot data is "
        "not expected to pass; the acceptance gate uses the equilibrated "
        "pitchfork."
    ),
)
def test_pitchfork_half_split_reproducibility():
    """Negative control — see xfail reason above."""
    from scipy.stats import spearmanr

    ad = generate_pitchfork(
        tau_crit=0.5, seed=0,
        n_cells=2000, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    rng = np.random.default_rng(0)
    idx = rng.permutation(ad.n_obs)
    half = ad.n_obs // 2
    a_half = ad[idx[:half]].copy()
    b_half = ad[idx[half:]].copy()

    lam_a, t_a = _fit_scjdo2(a_half, n_epochs=1000, k=40, seed=1)
    lam_b, t_b = _fit_scjdo2(b_half, n_epochs=1000, k=40, seed=2)

    # Evaluate both on a shared pseudotime grid via kernel smoothing.
    grid = np.linspace(0.05, 0.95, 50)
    curve_a = _kernel_curve(lam_a, t_a, grid, bandwidth=0.04)
    curve_b = _kernel_curve(lam_b, t_b, grid, bandwidth=0.04)
    m = np.isfinite(curve_a) & np.isfinite(curve_b)
    assert m.sum() >= 10, "too few grid points with valid curve on both halves"
    rho, _ = spearmanr(curve_a[m], curve_b[m])
    assert rho >= REPRODUCIBILITY_SPEARMAN_FLOOR, (
        f"half-split Spearman {rho:.3f} < {REPRODUCIBILITY_SPEARMAN_FLOOR}"
    )
