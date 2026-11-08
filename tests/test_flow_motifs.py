"""scjdo2.flow_motifs: OT plan kept by flow, archetypes chained along it."""

from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("ot")
pytest.importorskip("torch")

import scjdo2 as sj  # noqa: E402
from scjdo2.flow_operator import load_drift, ot_velocity  # noqa: E402


def _two_lanes(n_per_time: int = 120, n_times: int = 4, seed: int = 0):
    """Cells drift along x in two separated lanes (y = -2 or +2); a cell never
    changes lane, so the lane label at one time fixes it at the next. Both lanes
    hold half the cells at every time: balanced transport would otherwise have to
    move the surplus across lanes."""
    rng = np.random.default_rng(seed)
    z, t, lane = [], [], []
    for k in range(n_times):
        side = rng.permutation(np.repeat([-1.0, 1.0], n_per_time // 2))   # equal lanes: balanced OT
        x = k + 0.15 * rng.standard_normal(n_per_time)
        y = 2.0 * side + 0.15 * rng.standard_normal(n_per_time)
        w = 0.1 * rng.standard_normal((n_per_time, 2))
        z.append(np.column_stack([x, y, w]))
        t.append(np.full(n_per_time, float(k)))
        lane.append(np.where(side > 0, "up", "down"))
    return np.vstack(z), np.concatenate(t), np.concatenate(lane)


@pytest.fixture(scope="module")
def lanes():
    z, t, lane = _two_lanes()
    a = anndata.AnnData(X=np.zeros((len(z), 1), np.float32))
    a.obsm["X_pca"] = z
    a.obs["day"] = t
    a.obs["lane"] = lane
    sj.flow(a, time_key="day", n_comps=4, n_epochs=300, device="cpu", keep_plan=20)
    return a


def test_plan_rows_are_distributions_over_the_next_time(lanes):
    a = lanes
    P = a.obsp["flow_plan"].tocsr()
    t = a.obs["day"].to_numpy()
    rows = np.asarray(P.sum(axis=1)).ravel()
    last = t == t.max()
    assert np.allclose(rows[~last], 1.0, atol=1e-5)
    assert np.all(rows[last] == 0)
    r, c = P.nonzero()
    assert np.all(t[c] == t[r] + 1)                     # only to the next sampling time
    assert np.diff(P.indptr)[~last].max() <= 20
    pairs = a.uns["flow"]["ot_pairs"]
    assert ((pairs["plan_kept_mass_median"] > 0) & (pairs["plan_kept_mass_median"] <= 1)).all()


def test_minibatch_plan_matches_dense_label_flow():
    z, t, lane = _two_lanes(n_per_time=200)
    up = lane == "up"
    _, _, Pd = ot_velocity(z, t, solver="dense", plan_top_k=20)
    # Each mini-batch is balanced on its own, so batch-to-batch differences in lane
    # composition push some mass across lanes; the leak shrinks with batch size.
    _, info, Pm = ot_velocity(z, t, solver="minibatch", batch_size=180, coverage=20, plan_top_k=20)
    src = t < t.max()
    to_up_d = np.asarray(Pd[src] @ up.astype(float)).ravel()
    to_up_m = np.asarray(Pm[src] @ up.astype(float)).ravel()
    assert np.allclose(np.asarray(Pm.sum(axis=1)).ravel()[src], 1.0, atol=1e-5)
    assert np.median(np.abs(to_up_d - to_up_m)) < 0.05
    assert np.mean(np.abs(to_up_d - to_up_m)) < 0.15
    assert all("plan_kept_mass_median" in p for p in info)


def test_exact_contraction_matches_full_dense_plan():
    import ot
    z, t, lane = _two_lanes(n_per_time=50, n_times=2)
    up = (lane == "up").astype(float)[:, None]
    _, _, Q = ot_velocity(z, t, solver="dense", plan_apply=up)
    x, y = z[t == 0], z[t == 1]
    C = ot.dist(x, y)
    P = ot.sinkhorn(np.full(50, 1 / 50), np.full(50, 1 / 50), C, reg=0.05 * np.median(C))
    ref = (P @ up[t == 1]) / P.sum(1, keepdims=True)
    assert np.allclose(Q[t == 0], ref, atol=1e-8)
    assert np.isnan(Q[t == 1]).all()


def test_exact_and_stored_plan_agree_on_lanes(lanes):
    ex = lanes.copy()
    te = sj.flow_transitions(ex, labels="lane", n_perm=0)
    ts = sj.flow_transitions(ex, labels="lane", n_perm=0, exact=False)
    assert np.allclose(te["prob"], ts["prob"], atol=0.02)
    bad = lanes.copy()
    bad.obsm["X_pca"] = bad.obsm["X_pca"] * 1.5         # (a shift would not change OT velocities)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        sj.flow_transitions(bad, labels="lane", n_perm=0)


def test_load_drift_reproduces_stored_drift(lanes):
    import torch
    a = lanes
    model = load_drift(a)
    z = np.asarray(a.obsm["X_pca"], np.float32)[:, :4]
    with torch.no_grad():
        f = model(torch.as_tensor(z)).numpy()
    assert np.allclose(f, a.obsm["flow_drift"], atol=1e-4)


def test_transitions_recover_lanes_and_null_is_flat(lanes):
    a = lanes
    tab = sj.flow_transitions(a, labels="lane", n_perm=200)
    same = tab[tab.from_label == tab.to_label]
    other = tab[tab.from_label != tab.to_label]
    assert (same["prob"] > 0.95).all()
    assert (same["p_enriched"] <= 1.0 / 201 + 1e-12).all()
    assert (other["p_depleted"] <= 1.0 / 201 + 1e-12).all()
    assert tab.groupby(["from_time", "from_label"])["prob"].sum().round(6).eq(1).all()
    assert tab[tab.to_time == 3.0]["to_extrapolated"].all()
    a.obs["one"] = "x"
    single = sj.flow_transitions(a, labels="one", n_perm=50)
    assert single["prob"].eq(1).all() and single["p_enriched"].isna().all()

    rng = np.random.default_rng(1)
    a.obs["noise"] = rng.choice(["p", "q"], a.n_obs)
    tn = sj.flow_transitions(a, labels="noise", n_perm=200)
    assert (tn["p_enriched"] > 0.001).mean() > 0.8
    assert np.nanmax(np.abs(tn["z"])) < 5


def test_paths_follow_lanes_and_match_markov(lanes):
    tab = sj.flow_paths(lanes, labels="lane", n_paths=2000)
    top = set(tab["path"].head(2))
    assert top == {"up > up > up > up", "down > down > down > down"}
    assert tab["frequency"].head(2).sum() > 0.95
    assert np.allclose(tab["obs_over_expected"].head(2), 1.0, atol=0.15)
    col = sj.flow_paths(lanes, labels="lane", n_paths=500, collapse=True)
    assert set(col["path"].head(2)) == {"up", "down"}


def test_archetypes_integrate_and_figures(lanes, tmp_path):
    a = lanes
    H = sj.flow_archetypes(a, n_archetypes=2, max_fit_cells=200)
    assert H.shape == (2, 4, 4)
    assert a.obsm["flow_archetype_weights"].shape == (a.n_obs, 2)
    assert list(a.obs["flow_archetype"].cat.categories) == ["A1", "A2"]
    counts = a.obs["flow_archetype"].value_counts()
    assert counts["A1"] >= counts["A2"]
    tab = sj.flow_transitions(a, n_perm=50)
    assert set(tab["from_label"]) <= {"A1", "A2"}

    res = sj.flow_integrate(a, np.flatnonzero(a.obs["day"].to_numpy() == 0)[:5], n_steps=30)
    assert res["z"].shape == (5, 31, 4)
    assert np.median(res["z"][:, -1, 0] - res["z"][:, 0, 0]) > 1.5   # moves forward along x
    assert res["archetype_weights"].shape == (5, 31, 2)
    assert np.isfinite(res["lambda1"]).all()

    from scjdo2.plot import figure_flow_archetypes, figure_flow_transitions
    figure_flow_archetypes(a, out=tmp_path / "arch.png")
    figure_flow_transitions(a, out=tmp_path / "trans.png")
    assert (tmp_path / "arch.png").exists() and (tmp_path / "trans.png").exists()


def test_h5ad_round_trip_keeps_plan_and_drift(lanes, tmp_path):
    a = lanes.copy()
    del a.uns["scjdo2"]                                  # J goes to a sidecar via scjdo2.io
    a.write_h5ad(tmp_path / "x.h5ad")
    b = anndata.read_h5ad(tmp_path / "x.h5ad")
    assert (b.obsp["flow_plan"] != lanes.obsp["flow_plan"]).nnz == 0
    load_drift(b)
    tab = sj.flow_transitions(b, labels="lane", n_perm=20)
    assert (tab[tab.from_label == tab.to_label]["prob"] > 0.95).all()


def test_keep_plan_none_drops_plan():
    z, t, _ = _two_lanes(n_per_time=60, n_times=3)
    a = anndata.AnnData(X=np.zeros((len(z), 1), np.float32))
    a.obsm["X_pca"] = z
    a.obs["day"] = t
    sj.flow(a, time_key="day", n_comps=4, n_epochs=50, device="cpu", keep_plan=None)
    assert "flow_plan" not in a.obsp
    with pytest.raises(KeyError, match="keep_plan"):
        sj.flow_transitions(a, labels="day", exact=False)
    with pytest.raises(KeyError, match="keep_plan"):
        sj.flow_paths(a, labels="day")
    assert len(sj.flow_transitions(a, labels="day", n_perm=0))      # exact needs no stored plan
