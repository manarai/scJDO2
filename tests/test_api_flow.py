"""Top-level flow API: flow_groups, flow_stability, run(time_key=...), and the
guards that keep the flow and energy operators (same uns key) apart."""

from __future__ import annotations

import json

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("ot")
pytest.importorskip("torch")

import scjdo2 as sj  # noqa: E402
from scjdo2.flow_operator import operator_kind  # noqa: E402

FAST = dict(n_comps=4, n_epochs=150, device="cpu")


def _fork(n_per_time: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    z, t = [], []
    for k in range(5):
        x = k + 0.15 * rng.standard_normal(n_per_time)
        arm = rng.choice([-1.0, 1.0], n_per_time)
        y = arm * max(k - 2, 0) * 0.8 + 0.15 * rng.standard_normal(n_per_time)
        z.append(np.column_stack([x, y, 0.1 * rng.standard_normal((n_per_time, 2))]))
        t.append(np.full(n_per_time, float(k)))
    a = anndata.AnnData(X=np.zeros((5 * n_per_time, 1), np.float32))
    a.obsm["X_pca"] = np.vstack(z).astype(np.float32)
    a.obs["day"] = np.concatenate(t)
    a.obs["stage"] = np.where(a.obs["day"] == 2, "branch", "d" + a.obs["day"].astype(int).astype(str))
    return a


def test_names_exported():
    for name in ("flow", "flow_groups", "flow_stability", "fit", "run"):
        assert callable(getattr(sj, name))
    assert "flow_groups" in sj.FIGURES


def test_flow_groups_table_and_store():
    a = _fork()
    sj.flow(a, "day", **FAST)
    assert operator_kind(a) == "flow"
    tab = sj.flow_groups(a, "stage", n_boot=200)
    assert list(tab["rank"]) == list(range(1, len(tab) + 1))
    assert (np.diff(tab["median_lambda1"]) <= 0).all()
    assert (tab["ci_low"] <= tab["median_lambda1"]).all() and (tab["median_lambda1"] <= tab["ci_high"]).all()
    last = tab.set_index("group").loc["d4"]
    assert last["frac_extrapolated"] == 1.0          # last sampling time has no OT velocity
    assert tab.set_index("group").loc["d0", "frac_extrapolated"] == 0.0
    assert a.uns["flow"]["groups"]["groupby"] == "stage"
    # same seed, same interval
    again = sj.flow_groups(a, "stage", n_boot=200)
    assert np.allclose(again["ci_low"], tab["ci_low"])


def test_flow_groups_needs_flow():
    with pytest.raises(KeyError, match="run scjdo2.flow"):
        sj.flow_groups(_fork(), "stage")


def test_flow_stability_leaves_stored_operator():
    a = _fork()
    sj.flow(a, "day", **FAST)
    J0 = a.uns["scjdo2"]["J"].copy()
    st = sj.flow_stability(a, seeds=(0, 1), groupby="stage", device="cpu")
    assert set(st["per_cell_spearman"]) == {"0-1"}
    assert -1 <= st["per_cell_spearman_min"] <= st["per_cell_spearman_median"] <= 1
    assert st["params"]["n_epochs"] == 150 and st["params"]["n_comps"] == 4   # taken from the stored run
    assert len(st["groups"]) == 5 and isinstance(st["top_group_same"], bool)
    sd = a.obs["flow_lambda1_rank_sd"].to_numpy()
    assert np.isfinite(sd).all() and (sd >= 0).all()
    assert np.array_equal(a.uns["scjdo2"]["J"], J0)
    assert "stability" in a.uns["flow"]


def test_flow_stability_seed0_matches_stored_fit():
    a = _fork()
    sj.flow(a, "day", **FAST)
    b = a.copy()
    sj.flow_stability(b, seeds=(0, 0), device="cpu")
    assert b.uns["flow"]["stability"]["per_cell_spearman_min"] > 0.999


def test_guards_between_operators():
    a = _fork()
    sj.fit(a, use_rep="X_pca", n_epochs=20)
    assert operator_kind(a) == "energy"
    with pytest.raises(ValueError, match="energy operator"):
        sj.flow(a, "day", **FAST)
    sj.flow(a, "day", overwrite=True, **FAST)
    assert operator_kind(a) == "flow"
    with pytest.raises(ValueError, match="flow operator"):
        sj.fit(a, use_rep="X_pca", n_epochs=20)
    with pytest.raises(ValueError, match="flow operator"):
        sj.analyse(a, cluster_key="stage")
    sj.flow(a, "day", **FAST)                         # re-running flow on flow is fine


def test_run_time_course(tmp_path):
    a = _fork()
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2].copy()
    summary = sj.run(a, tmp_path / "r", cluster_key="stage", time_key="day",
                     flow_kwargs=FAST, stability_seeds=(0, 1))
    assert summary["route"] == "flow" and summary["figures"] == ["flow_groups"]
    for f in ("groups.csv", "per_cell.csv", "summary.json", "figures/flow_groups.png", "adata.h5ad"):
        assert (tmp_path / "r" / f).exists(), f
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert 0 < s["frac_with_velocity"] < 1 and "per_cell_spearman_median" in s["stability"]
    back = sj.io.load(tmp_path / "r")
    assert back.uns["scjdo2"]["J"].shape == (a.n_obs, 4, 4)
    with pytest.raises(ValueError, match="energy route"):
        sj.run(a, tmp_path / "r2", cluster_key="stage", time_key="day", sources=["lineage"])


def test_dense_energy_J_survives_save_load(tmp_path):
    a = _fork()
    sj.fit(a, use_rep="X_pca", n_epochs=20)
    sj.io.save(a, tmp_path)
    back = sj.io.load(tmp_path)
    assert np.allclose(back.uns["scjdo2"]["J"], a.uns["scjdo2"]["J"])
    assert operator_kind(back) == "energy"
