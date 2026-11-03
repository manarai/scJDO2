"""Finite-time influence: drivers vs responders (scjdo2.influence)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.linalg import expm

from scjdo2 import influence as inf


def _adata_with_archetype(J: np.ndarray, P: np.ndarray, reason=None, names=None) -> AnnData:
    G = P.shape[1]
    ad = AnnData(np.zeros((3, G), dtype=np.float32))
    ad.var_names = names or [f"g{i}" for i in range(G)]
    block = P @ J @ P.T
    ad.uns["scjdo2"] = {"archetypes": {0: {
        "behaviour": "holder",
        "eigenvalues": np.sort(np.linalg.eigvals(block))[::-1],
        "block": block,
        "projector": P,
        "fisher": np.ones(G),
        "reason": np.array(reason if reason is not None else ["ok"] * G, dtype=object),
        "n_cells": 3,
    }}}
    return ad


def _star(G: int = 10, strength: float = 1.0) -> np.ndarray:
    """Gene 0 drives genes 1..4; everything decays at rate 1."""
    J = -np.eye(G)
    J[1:5, 0] = strength
    return J


def test_scores_match_brute_force():
    rng = np.random.default_rng(0)
    G, r = 12, 5
    Q, _ = np.linalg.qr(rng.standard_normal((G, r)))
    P = Q.T                                          # orthonormal rows
    J = rng.standard_normal((G, G)) * 0.4 - np.eye(G)
    B = P @ J @ P.T
    t = 0.7
    M = P.T @ (expm(B * t) - np.eye(r)) @ P
    off = M - np.diag(np.diag(M))
    s = inf.influence_scores(B, P, t)
    np.testing.assert_allclose(s["out"], np.linalg.norm(off, axis=0), atol=1e-10)
    np.testing.assert_allclose(s["in"], np.linalg.norm(off, axis=1), atol=1e-10)
    np.testing.assert_allclose(s["self"], np.diag(M), atol=1e-10)


def test_star_driver_and_responders():
    G = 10
    ad = _adata_with_archetype(_star(G), np.eye(G))
    tbl = inf.archetype_influence(ad, t=1.0, quantile=0.5)
    role = dict(zip(tbl["gene"], tbl["role"]))
    assert role["g0"] == "driver"
    assert all(role[f"g{i}"] == "responder" for i in range(1, 5))
    assert all(role[f"g{i}"] == "passenger" for i in range(5, G))
    assert tbl.iloc[0]["gene"] == "g0"               # sorted by out-influence


def test_unmeasured_genes_are_flagged_not_scored_as_weak():
    G = 10
    reason = ["capped"] + ["ok"] * (G - 1)           # the driver's column is capped
    ad = _adata_with_archetype(_star(G), np.eye(G), reason=reason)
    tbl = inf.archetype_influence(ad, t=1.0)
    assert tbl.set_index("gene").loc["g0", "role"] == "unmeasured"


def test_off_tangent_gene_is_unmeasured():
    G = 6
    P = np.eye(G)[:5]                                # gene 5 not in the tangent
    ad = _adata_with_archetype(-np.eye(G), P)
    tbl = inf.archetype_influence(ad, t=1.0)
    assert tbl.set_index("gene").loc["g5", "role"] == "unmeasured"


def test_edges_carry_sign():
    G = 6
    J = -np.eye(G)
    J[1, 0] = 1.0                                    # 0 activates 1
    J[2, 0] = -1.0                                   # 0 represses 2
    ad = _adata_with_archetype(J, np.eye(G))
    inf.archetype_influence(ad, t=1.0)
    e = inf.influence_edges(ad, 0, sources=["g0"], n_targets=3).set_index("target")
    assert e.loc["g1", "sign"] == 1
    assert e.loc["g2", "sign"] == -1
    assert "g0" not in e.index                       # no self edge


def test_reference_annotation_case_insensitive():
    edges = pd.DataFrame({"source": ["Gata1", "Spi1"], "target": ["Klf1", "Cebpa"],
                          "weight": [1.0, 1.0], "abs_weight": [1.0, 1.0], "sign": [1, 1], "t": [1.0, 1.0]})
    ref = pd.DataFrame({"source": ["GATA1"], "target": ["KLF1"]})
    e = inf.annotate_reference(edges, ref)
    assert e["in_reference"].tolist() == [True, False]


def test_default_time_uses_transient_peak_for_non_normal_block():
    B = np.array([[-1.0, 20.0], [0.0, -2.0]])        # strong shear → transient growth
    t, rule = inf.default_time(B)
    assert rule == "transient_peak"
    assert inf.excess_gain(B, [t])[0] > 1.05
    # Normal block: no excess gain → lead time scale.
    t2, rule2 = inf.default_time(np.diag([-1.0, -3.0]))
    assert rule2 == "lead_timescale"
    assert t2 == pytest.approx(1.0)


def test_near_normal_operator_still_splits_by_polarity():
    """out ≈ in for most genes (symmetric coupling) plus one asymmetric pair:
    the asymmetric source is a driver, its target a responder."""
    rng = np.random.default_rng(1)
    G = 30
    A = rng.standard_normal((G, G)) * 0.05
    J = (A + A.T) / 2 - np.eye(G)
    J[1, 0] += 0.6                                   # one-way 0 → 1
    ad = _adata_with_archetype(J, np.eye(G))
    tbl = inf.archetype_influence(ad, t=1.0, quantile=0.8).set_index("gene")
    assert tbl.loc["g0", "role"] == "driver"
    assert tbl.loc["g1", "role"] == "responder"


def test_unstable_block_uses_lead_timescale():
    B = np.array([[0.5, 20.0], [0.0, -2.0]])         # fork with strong shear
    t, rule = inf.default_time(B)
    assert rule == "lead_timescale"
    assert t == pytest.approx(2.0)
    assert inf.time_grid(B).max() <= 5.0 / 0.5 + 1e-9  # capped at 5 e-foldings


def test_missing_block_raises_clear_error():
    G = 4
    ad = _adata_with_archetype(-np.eye(G), np.eye(G))
    del ad.uns["scjdo2"]["archetypes"][0]["block"]
    with pytest.raises(RuntimeError, match="archetype_operators"):
        inf.archetype_influence(ad)


def test_figure_influence_network_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2.plot import figure_influence_network

    G = 10
    ad = _adata_with_archetype(_star(G), np.eye(G))
    ref = pd.DataFrame({"source": ["g0"], "target": ["g1"]})
    out = tmp_path / "influence.png"
    fig = figure_influence_network(ad, ref_network=ref, save=out)
    assert out.exists() and out.stat().st_size > 0
    assert len(fig.axes) == 3
