"""Behaviour-landscape contract tests (operator-space occupancy figures).

Cheap synthetic adata with planted operators so the contract runs in CI:

1. ``behaviour_fractions``: a planted basin cell gives frac_neg = 1, a
   planted saddle gives frac_pos = 1/d, fractions sum to 1 everywhere.
2. ``oracle_tol``: positive, does not rescale with the operator spread.
3. ``fisher_mass``: ones without a ledger; ledger sums with one.
4. ``flow_scrambled_null`` (mode="flow"): returns pooled arrays of the
   right length and erases the planted basin/saddle separation.
5. ``figure_behaviour_landscape`` and ``figure_behaviour_ternary``
   write PDF+PNG, have labelled axes, and the composition bars show the
   planted young→aged shift in the right type.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scjdo2 import opspace
from scjdo2.plot import (
    behaviour_composition,
    figure_behaviour_landscape,
    figure_behaviour_ternary,
)

D = 6
N_PER = 60


def _planted_adata(seed: int = 0) -> AnnData:
    """Three types × two conditions; aged ``saddle`` type gains fork share."""
    rng = np.random.default_rng(seed)
    rows = []
    J = []
    for cond in ("young", "aged"):
        for ctype in ("basin", "saddle", "flat"):
            for _ in range(N_PER):
                if ctype == "basin":
                    ev = -np.abs(rng.normal(1.0, 0.1, D))
                elif ctype == "saddle":
                    ev = -np.abs(rng.normal(1.0, 0.1, D))
                    n_pos = 1 if cond == "young" else 2
                    ev[:n_pos] = np.abs(rng.normal(0.8, 0.1, n_pos))
                else:  # flat
                    ev = rng.normal(0.0, 0.02, D)
                # random orthogonal basis → symmetric J with those eigenvalues
                Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
                J.append((Q * ev) @ Q.T)
                rows.append({"cluster": ctype, "age": cond})
    J = np.stack(J).astype(np.float32)
    n = J.shape[0]
    obs = pd.DataFrame(rows, index=[f"c{i}" for i in range(n)])
    z = rng.normal(size=(n, D)).astype(np.float32)
    # give each cluster its own centroid so the oracle has structure
    for i, ctype in enumerate(("basin", "saddle", "flat")):
        z[obs["cluster"].to_numpy() == ctype] += 4.0 * np.eye(D)[i]
    ad = AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs)
    ad.obsm["X_pca"] = z
    ad.uns["scjdo2"] = {"J": J, "meta": {"clock_calibrated": False, "use_rep": "X_pca"}}
    desc = opspace.landscape_descriptors(J)
    for k, v in desc.items():
        ad.obs[k] = v
    return ad


class _ConstAdapter:
    """Stand-in drift adapter: returns a fixed J tensor regardless of z."""

    def __init__(self, J):
        self._J = J

    def jacobian_autodiff(self, z):
        return self._J


def test_behaviour_fractions_planted():
    ad = _planted_adata()
    df = opspace.behaviour_fractions(ad, tol=0.1)
    s = df.sum(axis=1).to_numpy()
    assert np.allclose(s, 1.0, atol=1e-6)
    cl = ad.obs["cluster"].to_numpy()
    age = ad.obs["age"].to_numpy()
    assert np.allclose(df.loc[cl == "basin", "op_frac_neg"], 1.0)
    assert np.allclose(df.loc[(cl == "saddle") & (age == "young"), "op_frac_pos"], 1.0 / D)
    assert np.allclose(df.loc[(cl == "saddle") & (age == "aged"), "op_frac_pos"], 2.0 / D)
    assert np.allclose(df.loc[cl == "flat", "op_frac_zero"], 1.0)
    assert ad.uns["scjdo2"]["behaviour_fractions_tol"] == 0.1


def test_oracle_tol_scale_free():
    ad = _planted_adata()
    tol = opspace.oracle_tol(ad, use_rep="X_pca", cluster_key="cluster")
    assert tol > 0
    # scaling the operators must not change the oracle (it reads z, not J)
    ad.uns["scjdo2"]["J"] = ad.uns["scjdo2"]["J"] * 10.0
    tol2 = opspace.oracle_tol(ad, use_rep="X_pca", cluster_key="cluster")
    assert tol2 == pytest.approx(tol)
    assert ad.uns["scjdo2"]["oracle_tol"] == pytest.approx(tol2)


def test_fisher_mass_default_and_ledger():
    from scipy.sparse import csr_matrix

    ad = _planted_adata()
    assert np.all(opspace.fisher_mass(ad) == 1.0)
    n = ad.n_obs
    F = [csr_matrix(np.full((2, 2), float(i + 1))) for i in range(n)]
    ad.uns["scjdo2"]["fisher_ledger"] = {"fisher": F}
    m = opspace.fisher_mass(ad)
    assert m.shape == (n,)
    assert m[0] == pytest.approx(4.0) and m[-1] == pytest.approx(4.0 * n)


def test_flow_scrambled_null_erases_structure():
    ad = _planted_adata()
    J = ad.uns["scjdo2"]["J"]
    z = ad.obsm["X_pca"]
    v = np.random.default_rng(1).normal(size=z.shape)
    n = ad.n_obs
    cl = ad.obs["cluster"].to_numpy()

    def fit_fn(z_in, v_perm, seed):
        # a "refit" on scrambled flow: assign each cell the operator of a
        # random other cell — the planted cluster structure must vanish.
        rng = np.random.default_rng(seed)
        return _ConstAdapter(J[rng.permutation(n)])

    null = opspace.flow_scrambled_null(z, v, fit_fn, n_perm=3, mode="flow", seed=0)
    assert null["lead_re"].shape == (3 * n,)
    assert null["n_perm"] == 3 and null["mode"] == "flow"
    real = ad.obs["lead_re"].to_numpy()
    sep_real = real[cl == "saddle"].mean() - real[cl == "basin"].mean()
    nl = null["lead_re"][:n]
    sep_null = nl[cl == "saddle"].mean() - nl[cl == "basin"].mean()
    assert sep_real > 1.0
    assert abs(sep_null) < 0.5 * sep_real


def test_figures_write_and_label():
    ad = _planted_adata()
    tol = opspace.oracle_tol(ad, use_rep="X_pca", cluster_key="cluster")
    opspace.behaviour_fractions(ad, tol=0.1)
    J = ad.uns["scjdo2"]["J"]
    n = ad.n_obs

    def fit_fn(z_in, v_perm, seed):
        rng = np.random.default_rng(seed)
        return _ConstAdapter(J[rng.permutation(n)])

    null = opspace.flow_scrambled_null(
        ad.obsm["X_pca"], np.zeros_like(ad.obsm["X_pca"]), fit_fn, n_perm=2,
    )
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "landscape"
        fig = figure_behaviour_landscape(ad, y="spectral_gap", null=null, out=out)
        assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
        ax = fig.axes[0]
        assert "(relative)" in ax.get_xlabel()
        assert ax.get_ylabel()
        fig2 = figure_behaviour_landscape(ad, y="henrici", null=null, subtract=True)
        assert fig2.axes[0].get_ylabel() == "Henrici index"

        out2 = Path(tmp) / "ternary"
        fig3 = figure_behaviour_ternary(ad, color_key="cluster", condition_key="age",
                                        conditions=("young", "aged"), out=out2)
        assert out2.with_suffix(".pdf").exists() and out2.with_suffix(".png").exists()
        assert len(fig3.axes) >= 2
        assert fig3.axes[1].get_ylabel()

    comp = behaviour_composition(ad, by="cluster", condition_key="age").set_index(
        ["cluster", "condition"],
    )
    assert comp.loc[("saddle", "aged"), "frac_pos"] > comp.loc[("saddle", "young"), "frac_pos"]
    assert comp.loc[("basin", "aged"), "frac_pos"] == pytest.approx(
        comp.loc[("basin", "young"), "frac_pos"],
    )
    assert tol > 0


def test_stiffness_is_not_a_function_of_lead_re():
    ad = _planted_adata()
    desc = opspace.landscape_descriptors(ad.uns["scjdo2"]["J"])
    cl = ad.obs["cluster"].to_numpy()
    # basin and saddle cells share the same holding stiffness (~1.0) but
    # differ in lead_re by ~1.8: height must separate from the floor.
    assert desc["stiffness"][cl == "basin"].mean() == pytest.approx(1.0, abs=0.1)
    assert desc["stiffness"][cl == "saddle"].mean() == pytest.approx(1.0, abs=0.1)
    assert desc["stiffness"][cl == "flat"].mean() < 0.1
    assert abs(desc["lead_re"][cl == "saddle"].mean()
               - desc["lead_re"][cl == "basin"].mean()) > 1.0
    # basin_index = -lead_re exactly (a plane over the floor)
    s = opspace.well_stiffness(ad)
    assert "op_stiffness" in ad.obs and "op_lambda_min" in ad.obs
    assert np.allclose(s, ad.obs["op_stiffness"].to_numpy())


def test_figure_3d_writes_and_warns():
    import warnings

    from scjdo2.plot import figure_behaviour_landscape_3d

    ad = _planted_adata()
    J = ad.uns["scjdo2"]["J"]
    n = ad.n_obs

    def fit_fn(z_in, v_perm, seed):
        rng = np.random.default_rng(seed)
        return _ConstAdapter(J[rng.permutation(n)])

    null = opspace.flow_scrambled_null(
        ad.obsm["X_pca"], np.zeros_like(ad.obsm["X_pca"]), fit_fn, n_perm=2,
    )
    assert "stiffness" in null and null["stiffness"].shape == (2 * n,)
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "landscape3d"
        fig = figure_behaviour_landscape_3d(ad, y="spectral_gap", null=null, out=out)
        assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
        ax = fig.axes[0]
        assert "(relative)" in ax.get_xlabel()
        assert "stiffness" in ax.get_zlabel()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            figure_behaviour_landscape_3d(ad, height="basin_index")
        assert any("plane" in str(r.message) for r in rec)


def test_energy_landscape_writes():
    from scjdo2.plot import figure_energy_landscape

    ad = _planted_adata()
    rng = np.random.default_rng(3)
    n = ad.n_obs
    # two-valley toy embedding + an energy that is low where cells are dense
    umap = np.concatenate([rng.normal((0, 0), 0.4, (n // 2, 2)),
                           rng.normal((3, 1), 0.4, (n - n // 2, 2))])
    ad.obsm["X_umap"] = umap.astype(np.float32)

    class _E:
        def energy(self, z):
            return -np.exp(-np.sum(z[:, :2] ** 2, axis=1)) - np.exp(
                -np.sum((z[:, :2] - np.array([3.0, 1.0])) ** 2, axis=1))

    with pytest.raises(TypeError):
        opspace.store_energy(ad, object(), use_rep="X_umap")
    E = opspace.store_energy(ad, _E(), use_rep="X_umap")
    assert E.shape == (n,) and "energy" in ad.obs
    opspace.classify_basin_latent(ad, tol=0.1, use_rep="X_pca")
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "energy"
        fig = figure_energy_landscape(ad, use_rep="X_umap", out=out)
        assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
        assert fig.axes[0].get_zlabel() == "energy E(z)"
