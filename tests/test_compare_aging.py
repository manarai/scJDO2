"""Unit tests for scjdo2.compare — isotropy_test + anticipation_test.

PREREG §4b amendment (2026-10-14, before any aging data run):
isotropy_test uses ``S = max_k |r_k − median(r)|`` with
``r_k = log(|λ_old,k| / |λ_young,k|)`` over cosine-matched
directions, against a split-half-within-young null. It replaces the
§4a Spearman statistic, which preserves rank under uniform softening
(ρ ≈ 1) and so passed the outcome §4 is meant to reject.

Synthetic cells: positions ``z ~ N(μ, Σ)`` and per-cell Jacobians
``J_i = −Σ^{-1/2} (I + E_i) Σ^{-1/2}`` with small symmetric noise
``E_i``, so the centroid operator is ``≈ −Σ⁻¹`` (the basin Hessian
of a Gaussian, what the lead-eig rule reads) with realistic
cell-to-cell scatter.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scjdo2.compare import (
    anticipation_test,
    compare_types,
    isotropy_statistic,
    isotropy_test,
    match_directions,
)

D = 10
N = 400
SIGMA_YOUNG = np.logspace(0, 1, D)            # std per axis; softest = last


def _cells(sigma: np.ndarray, n: int, rng, mu=None, noise: float = 0.15):
    d = sigma.size
    mu = np.zeros(d) if mu is None else mu
    z = mu + rng.standard_normal((n, d)) * sigma
    E = rng.standard_normal((n, d, d)) * noise
    E = 0.5 * (E + np.transpose(E, (0, 2, 1)))
    s_inv = 1.0 / sigma
    J = -(s_inv[None, :, None] * (np.eye(d)[None] + E) * s_inv[None, None, :])
    return z, J


def _iso(sig_b, seed=0, n_null=300):
    rng = np.random.default_rng(seed)
    z_a, J_a = _cells(SIGMA_YOUNG, N, rng)
    z_b, J_b = _cells(sig_b, N, rng)
    return isotropy_test(z_a, J_a, z_b, J_b, n_null=n_null, seed=seed)


def test_isotropic_widening_fails():
    """Every axis 1.6× wider: all r_k equal, S at the null → fail."""
    for seed in range(3):
        iso = _iso(SIGMA_YOUNG * 1.6, seed=seed)
        assert not iso.passes, (
            f"seed {seed}: uniform 1.6× widening must FAIL — "
            f"S={iso.S:.3f}, null_q95={iso.null_q95:.3f}")
        assert np.allclose(np.nanmedian(iso.log_ratios),
                           np.log(1 / 1.6 ** 2), atol=0.1)


def test_isotropic_widening_is_what_spearman_passed():
    """Documents why §4a was replaced: uniform softening keeps the
    eigenvalue order, so the provenance ρ is ≈ 1 while §4b fails."""
    iso = _iso(SIGMA_YOUNG * 1.6, seed=0)
    assert iso.rho_lambda > 0.95
    assert not iso.passes


def test_single_axis_widening_passes():
    """Softest axis 3× wider, others unchanged: one r_k ≈ log(1/9)
    against a median ≈ 0 → S ≫ null → pass."""
    sig_b = SIGMA_YOUNG.copy()
    sig_b[-1] *= 3.0
    for seed in range(3):
        iso = _iso(sig_b, seed=seed)
        assert iso.passes, (
            f"seed {seed}: single-axis 3× widening must PASS — "
            f"S={iso.S:.3f}, null_q95={iso.null_q95:.3f}")
        assert iso.S > 1.5                         # ≈ log 9 = 2.2


def test_identical_gaussians_fail():
    for seed in range(3):
        iso = _iso(SIGMA_YOUNG, seed=seed)
        assert not iso.passes, (
            f"seed {seed}: identical groups must FAIL — "
            f"S={iso.S:.3f}, null_q95={iso.null_q95:.3f}")


def test_statistic_is_zero_for_exact_uniform_scaling():
    J_a = -np.diag(1.0 / SIGMA_YOUNG ** 2)
    S, r, _ = isotropy_statistic(J_a, J_a / 2.56)
    assert S < 1e-12
    assert np.allclose(r, np.log(1 / 2.56))


def test_anticipation_zero_shift_fails_and_soft_axis_shift_passes():
    rng = np.random.default_rng(1)
    J_a = -np.diag(1.0 / SIGMA_YOUNG ** 2)
    z_a, _ = _cells(SIGMA_YOUNG, 1000, rng)
    z_b0, _ = _cells(SIGMA_YOUNG * 1.6, 1000, rng)
    assert not anticipation_test(z_a, z_b0, J_a, n_null=500, seed=0).passes

    mu_b = np.zeros(D); mu_b[-1] = 3.0 * SIGMA_YOUNG[-1]
    z_b, _ = _cells(SIGMA_YOUNG, 1000, rng, mu=mu_b)
    ant = anticipation_test(z_a, z_b, J_a, n_null=500, seed=0)
    assert ant.passes, f"|cos|={ant.cos_abs:.3f}, q95={ant.null_q95:.3f}"
    assert abs(ant.u_a[-1]) > 0.9


def test_match_directions_identity_is_identity():
    """When J_a == J_b, every direction matches itself with cos = 1."""
    rng = np.random.default_rng(2)
    d = 6
    A = rng.standard_normal((d, d))
    J = -0.5 * (A @ A.T + np.eye(d))                                # NSD
    md = match_directions(J, J)
    for i_a, i_b, c in md.pairs:
        assert i_a == i_b
        assert abs(c - 1.0) < 1e-6
    assert np.allclose(md.eig_a, md.eig_b)


def _toy_adata():
    rng = np.random.default_rng(3)
    zs, Js, obs = [], [], []
    sig_axis = SIGMA_YOUNG.copy(); sig_axis[-1] *= 3.0
    for ct, sig_old in [("axis_loss", sig_axis), ("uniform", SIGMA_YOUNG * 1.6)]:
        for grp, sig in [("young", SIGMA_YOUNG), ("old", sig_old)]:
            z, J = _cells(sig, 250, rng)
            zs.append(z); Js.append(J)
            obs += [(ct, grp)] * 250
    a = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32),
                   obs=pd.DataFrame(obs, columns=["cell_type", "age_group"],
                                    index=[str(i) for i in range(len(obs))]))
    a.obsm["X_scVI_shared"] = np.vstack(zs)
    a.uns["scjdo2"] = {"J_latent": np.vstack(Js)}
    return a


def test_compare_types_columns_and_reproducible():
    a = _toy_adata()
    kw = dict(use_rep="X_scVI_shared", n_null=200, seed=0)
    df1 = compare_types(a, **kw)
    df2 = compare_types(a, **kw)
    pd.testing.assert_frame_equal(df1, df2)
    for col in ["S", "iso_null_q95", "iso_pass", "rho", "cv_ratio",
                "cos_abs_u_young_d", "ant_null_q95", "ant_pass"]:
        assert col in df1.columns
    res = df1.set_index("cell_type")["iso_pass"]
    assert bool(res["axis_loss"]) and not bool(res["uniform"])


def test_aging_modes_are_seeded_and_descriptively_labelled():
    """A03 mode helper uses fixed-seed silhouette K-means and spec labels."""
    from scjdo2.compare import aging_modes

    rows = []
    base = np.array([-9.0, -6.0, -4.0, -2.0])
    for i in range(4):
        rows.append({
            "kind": "uniform", "spectrum_young": base,
            "spectrum_old": base * (0.70 + 0.01 * i),
            "median_abs_lambda_young": 5.0,
            "median_abs_lambda_old": 3.5 + 0.05 * i,
            "anisotropy_young": 2.0, "anisotropy_old": 2.04,
            "n_soft_young": 0, "n_soft_old": 0,
            "basin_index_young": 1.0, "basin_index_old": 0.7,
            "displacement": 1.0, "split_half_noise": 0.1,
        })
    for i in range(4):
        rows.append({
            "kind": "axis", "spectrum_young": base,
            "spectrum_old": np.array([-13.5 - i, -6.0, -4.0, -2.0]),
            "median_abs_lambda_young": 5.0,
            "median_abs_lambda_old": 5.1,
            "anisotropy_young": 2.0, "anisotropy_old": 2.7,
            "n_soft_young": 0, "n_soft_old": 1,
            "basin_index_young": 1.0, "basin_index_old": 0.8,
            "displacement": 1.0, "split_half_noise": 0.1,
        })
    for i in range(4):
        rows.append({
            "kind": "unchanged", "spectrum_young": base,
            "spectrum_old": base * (1.0 + 0.002 * i),
            "median_abs_lambda_young": 5.0,
            "median_abs_lambda_old": 5.0,
            "anisotropy_young": 2.0, "anisotropy_old": 2.0,
            "n_soft_young": 0, "n_soft_old": 0,
            "basin_index_young": 1.0, "basin_index_old": 1.0,
            "displacement": 0.01, "split_half_noise": 0.1,
        })
    inp = pd.DataFrame(rows)
    out1 = aging_modes(inp, seed=0)
    out2 = aging_modes(inp, seed=0)
    pd.testing.assert_frame_equal(out1, out2)
    labels = pd.concat([inp[["kind"]], out1], axis=1).groupby("kind")["mode_label"].agg(set)
    assert labels["uniform"] == {"uniform softening"}
    assert labels["axis"] == {"single-axis loss"}
    assert labels["unchanged"] == {"unchanged"}
    assert 2 <= int(out1["silhouette_k"].iloc[0]) <= 5
