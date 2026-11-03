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
    with pytest.raises(RuntimeError, match="drivers.influence"):
        figure_influence_network(ad)                 # draw-only: nothing stored yet
    inf.archetype_influence(ad)
    out = tmp_path / "influence"
    fig = figure_influence_network(ad, ref_network=ref, out=out)
    assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
    assert len(fig.axes) == 3


# ---------------------------------------------------------------- timing QC


def _cascade_adata(driver_t=0.3, responder_t=0.7, n=600, seed=0):
    """Star network (g0 → g1..g4) whose expression rises at known times."""
    rng = np.random.default_rng(seed)
    G = 10
    ad = _adata_with_archetype(_star(G), np.eye(G))
    inf.archetype_influence(ad, t=1.0, quantile=0.5)
    t = np.sort(rng.random(n))
    sig = lambda c: 1.0 / (1.0 + np.exp(-(t - c) / 0.03))
    X = rng.normal(0, 0.02, (n, G))
    X[:, 0] += 2 * sig(driver_t)
    for g in range(1, 5):
        X[:, g] += 2 * sig(responder_t + 0.02 * g)
    X[:, 5] += 2 * (1 - sig(0.5))                   # a falling passenger
    new = AnnData(X.astype(np.float32))
    new.var_names = ad.var_names
    new.obs["pseudotime"] = t
    new.obsm["archetype_weights"] = np.ones((n, 1))
    new.uns["scjdo2"] = ad.uns["scjdo2"]
    return new


def test_onsets_recover_known_times_and_direction():
    from scjdo2.timing import expression_timing
    ad = _cascade_adata()
    res = expression_timing(ad, 0, n_perm=200)
    g = res["genes"].set_index("gene")
    assert g.loc["g0", "onset"] == pytest.approx(0.3, abs=0.05)
    assert g.loc["g1", "onset"] == pytest.approx(0.72, abs=0.05)
    assert g.loc["g5", "direction"] == -1
    assert g.loc["g5", "onset"] == pytest.approx(0.5, abs=0.05)
    assert np.isnan(g.loc["g9", "onset"])           # flat gene → no onset


def test_edges_and_roles_show_driver_first():
    from scjdo2.timing import expression_timing
    res = expression_timing(_cascade_adata(), 0, n_perm=200)
    assert res["edge_test"]["frac_source_first"] == 1.0
    assert res["edge_test"]["median_lag"] > 0.3
    # Reversed timing flips the answer.
    rev = expression_timing(_cascade_adata(driver_t=0.8, responder_t=0.2), 0, n_perm=200)
    assert rev["edge_test"]["frac_source_first"] == 0.0


def test_figure_expression_cascade_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2 import drivers as D
    from scjdo2.plot import figure_expression_cascade
    ad = _cascade_adata()
    with pytest.raises(RuntimeError, match="drivers.timing"):
        figure_expression_cascade(ad)
    summary = D.timing(ad, n_perm=50)["summary"]
    out = tmp_path / "cascade"
    fig = figure_expression_cascade(ad, out=out)
    assert out.with_suffix(".png").exists() and len(summary) == 1
    assert len(fig.axes) == 3


# ---------------------------------------------------------------- one-call API


def test_drivers_api_end_to_end(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2 import drivers as D

    ad = _cascade_adata()
    ref = D.load_reference(pd.DataFrame({"source": ["g0", "g0"], "target": ["g1", "g2"],
                                         "weight": [1.0, 1.0]}))
    assert set(ref.columns) == {"source", "target", "mode", "weight"}
    res = D.analyse(ad, ref_network=ref, quantile=0.5, n_perm=50)
    for key in ("influence", "roles", "key_genes", "edges", "reference_summary",
                "reference_per_tf", "timing_summary", "timing_onsets"):
        assert key in res, key
    # g1, g2 are reference targets; g3, g4 tie with them (all driven by g0).
    assert res["reference_summary"]["mean_auroc_jacobian"].iloc[0] == pytest.approx(6 / 7)
    assert res["timing_summary"]["edge_frac_source_first"].iloc[0] == 1.0
    tables = D.save_tables(res, tmp_path)
    assert all(p.exists() for p in tables.values())
    paths = D.figures(ad, tmp_path, ref_network=ref, results=res, formats=("png",))
    assert set(paths) == {"influence_network", "expression_cascade", "reference_auroc"}
    assert all(p.exists() for p in paths.values())


# ---------------------------------------------------------------- branch timing


def _fork_adata(n=800, seed=0):
    """g0 drives g1..g4. In fate A all rise, in fate B all fall; g0 first."""
    rng = np.random.default_rng(seed)
    G = 10
    base = _adata_with_archetype(_star(G), np.eye(G))
    inf.archetype_influence(base, t=1.0, quantile=0.5)
    t = rng.random(n)
    fate = np.where(t < 0.2, "root", np.where(rng.random(n) < 0.5, "A", "B"))
    sgn = np.where(fate == "B", -1.0, 1.0)
    sig = lambda c: 1.0 / (1.0 + np.exp(-(t - c) / 0.03))
    X = rng.normal(0, 0.02, (n, G)) + 1.0
    X[:, 0] += sgn * sig(0.4)
    for g in range(1, 5):
        X[:, g] += sgn * sig(0.6 + 0.02 * g)
    ad = AnnData(X.astype(np.float32))
    ad.var_names = base.var_names
    ad.obs["pseudotime"] = t
    ad.obs["fate"] = fate
    ad.obsm["archetype_weights"] = np.ones((n, 1))
    ad.uns["scjdo2"] = base.uns["scjdo2"]
    return ad


def test_branch_timing_recovers_order_that_mixing_hides():
    from scjdo2.timing import branch_timing, expression_timing
    ad = _fork_adata()
    mixed = expression_timing(ad, 0, n_perm=100)
    # Averaged over both fates the cascade cancels: the true driver onset
    # (0.4) is lost — either no onset or a noise crossing far from it.
    on_mixed = mixed["genes"].set_index("gene").loc["g0", "onset"]
    assert np.isnan(on_mixed) or abs(on_mixed - 0.4) > 0.2
    res = branch_timing(ad, 0, {"A": ["A"], "B": ["B"]}, "fate", root=["root"], n_perm=200)
    pb = res["per_branch"].set_index("branch")
    assert pb.loc["A", "frac_source_first"] == 1.0
    assert pb.loc["B", "frac_source_first"] == 1.0
    assert res["pooled"]["frac_source_first"] == 1.0
    assert res["pooled"]["p_perm"] < 0.05


def test_branch_timing_skips_small_branches():
    from scjdo2.timing import branch_timing
    res = branch_timing(_fork_adata(), 0, {"A": ["A"], "tiny": ["nope"]}, "fate",
                        root=[], n_perm=50)
    pb = res["per_branch"].set_index("branch")
    assert pb.loc["tiny", "n_cells"] == 0 and np.isnan(pb.loc["tiny", "frac_source_first"])


def test_drivers_branch_api_and_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2 import drivers as D
    ad = _fork_adata()
    res = D.analyse(ad, branches={"A": ["A"], "B": ["B"]}, cluster_key="fate",
                    root=["root"], quantile=0.5, n_perm=50)
    assert {"branch_summary", "branch_overall", "branch_members_summary"} <= set(res)
    assert res["branch_overall"]["frac_source_first"].iloc[0] == 1.0
    paths = D.figures(ad, tmp_path, results=res, formats=("png",))
    assert {"branch_timing", "branch_members_timing"} <= set(paths)


# ---------------------------------------------------------- network × states


def _state_adata(driver_first=True, n=900, seed=0):
    """Star network; 6 ordered states s0..s5. Driver g0 switches on at s1,
    targets g1..g4 at s3 (or reversed)."""
    rng = np.random.default_rng(seed)
    G = 10
    base = _adata_with_archetype(_star(G), np.eye(G))
    inf.archetype_influence(base, t=1.0, quantile=0.5)
    st = rng.integers(0, 6, n)
    on_d, on_t = (1, 3) if driver_first else (3, 1)
    X = rng.normal(0, 0.05, (n, G)) + 0.2
    X[:, 0] += 2.0 * (st >= on_d)
    for g in range(1, 5):
        X[:, g] += 2.0 * (st >= on_t)
    ad = AnnData(np.clip(X, 0, None).astype(np.float32))
    ad.var_names = base.var_names
    ad.obs["state"] = [f"s{k}" for k in st]
    ad.obs["pseudotime"] = st + rng.random(n) * 0.5
    ad.obsm["archetype_weights"] = np.ones((n, 1))
    ad.uns["scjdo2"] = base.uns["scjdo2"]
    return ad


def test_network_expression_driver_first_and_reversed():
    from scjdo2 import drivers as D
    states = [f"s{k}" for k in range(6)]
    res = D.network_expression(_state_adata(True), 0, "state", states=states)
    s = res["summary"].set_index("source").loc["g0"]
    assert s["n_targets"] == 4 and s["n_driver_first"] == 4
    assert res["genes"].set_index("gene").loc["g0", "on_state"] < 1.0 + 1e-9
    rev = D.network_expression(_state_adata(False), 0, "state", states=states)
    assert rev["summary"].set_index("source").loc["g0", "n_driver_first"] == 0


def test_default_states_ordered_by_pseudotime():
    from scjdo2.timing import default_states
    assert default_states(_state_adata(), 0, "state") == [f"s{k}" for k in range(6)]


def test_plot_network_expression_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2 import drivers as D
    ad = _state_adata()
    out = tmp_path / "netexpr"
    D.plot_network_expression(ad, 0, "state", out=out)
    assert out.with_suffix(".png").exists()
    # Stored result is reused when no cluster_key is given.
    D.plot_network_expression(ad, 0)
    paths = D.figures(ad, tmp_path, time_key="pseudotime", cluster_key="state",
                      formats=("png",))
    assert "network_expression_arch0" in paths


def test_plot_package_exports_resolve():
    import scjdo2.plot as P
    missing = [n for n in P.__all__ if not hasattr(P, n)]
    assert not missing, missing
    # Pre-reorganisation names are aliases of the new ones.
    assert P.spectrum_ladder is P.figure_spectrum_ladder
    assert P.perturb_paired is P.figure_perturb_paired


def test_save_figure_writes_each_format(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scjdo2.plot import save_figure
    fig = plt.figure()
    paths = save_figure(fig, tmp_path / "sub" / "x.png", formats=("pdf", "png"))
    assert [p.name for p in paths] == ["x.pdf", "x.png"] and all(p.exists() for p in paths)
    assert save_figure(fig, None) == []


def test_opspace_drivers_kind_selects_left_eigenvector():
    """Regression: drivers(kind="drivers") used to return the responder
    table because the storage kind from _get_J overwrote the argument."""
    from scipy.sparse import csr_matrix

    from scjdo2 import opspace
    G = 6
    ad = AnnData(np.zeros((2, G), dtype=np.float32))
    ad.var_names = [f"g{i}" for i in range(G)]
    right = np.array([1.0, 0.5, 0.1, 0.0, 0.0, 0.0], dtype=complex)
    left = np.array([0.0, 0.1, 0.2, 1.0, 0.3, 0.0], dtype=complex)
    ad.uns["scjdo2"] = {
        "J": [csr_matrix(np.eye(G)), csr_matrix(np.eye(G))],
        "op_eigvecs": np.zeros((2, 1, G), dtype=complex),
        "archetypes": {0: {"eigvec": right, "eigvec_left": left, "fisher": np.ones(G),
                           "reason": np.array(["ok"] * G, dtype=object)}},
    }
    resp = opspace.drivers(ad, which="archetype", archetype_id=0, kind="responders", top=1)
    drv = opspace.drivers(ad, which="archetype", archetype_id=0, kind="drivers", top=1)
    assert resp["gene"].iloc[0] == "g0"
    assert drv["gene"].iloc[0] == "g3"
