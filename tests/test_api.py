"""scjdo2 top-level API: fit → analyse → orient → figures → run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scjdo2 as sj
from scjdo2.data.synthetic_oriented import make_oriented_pitchfork


def _data():
    a = make_oriented_pitchfork(seed=0)
    a.obs["cluster"] = np.where(a.obs["day"] <= a.obs["day"].median(), "early", "late")
    a.obsm["X_umap"] = a.obsm["X_lat"][:, :2].copy()
    return a


def test_top_level_names():
    for name in ("fit", "analyse", "compare_conditions", "orient", "figures", "run", "FIGURES"):
        assert callable(getattr(sj, name)) or name == "FIGURES"
    with pytest.raises(AttributeError):
        sj.no_such_function
    import scjdo2.orient                     # the subpackage shares the name
    assert callable(sj.orient) and hasattr(sj.orient, "OrientConfig")


def test_step_by_step(tmp_path):
    a = _data()
    drift = sj.fit(a, use_rep="X_lat", n_epochs=30)
    J = a.uns["scjdo2"]["J"]
    assert J.shape == (a.n_obs, 2, 2) or J.shape[0] == a.n_obs
    assert np.allclose(J, np.transpose(J, (0, 2, 1)), atol=1e-4)
    assert "energy" in a.obs and hasattr(drift, "energy")
    info = sj.analyse(a, cluster_key="cluster", drift=drift)
    assert info["tol"] > 0 and "op_behaviour_basin" in a.obs
    out = sj.orient(a, ["labelling", "lineage"], labelling_key="velocity_labelling")
    assert out["meta"]["hash_unchanged"] and "flow_orientation_class" in a.obs
    paths = sj.figures(a, tmp_path, cluster_key="cluster")
    for name in ("operator_space", "ternary", "energy", "basin_index_by_type", "spectrum_ladder",
                 "soft_axis_map", "orientation"):
        assert name in paths, (name, sorted(paths))
        assert Path(str(paths[name]) + ".png").exists()


def test_figures_skip_what_is_missing(tmp_path):
    a = _data()
    sj.fit(a, use_rep="X_lat", n_epochs=10)
    paths = sj.figures(a, tmp_path, include=["orientation", "ternary"])
    assert paths == {}                       # no analyse, no orient yet
    with pytest.raises(ValueError):
        sj.figures(a, tmp_path, include=["nope"])


def test_run_writes_everything(tmp_path):
    a = _data()
    s = sj.run(a, tmp_path, cluster_key="cluster", use_rep="X_lat",
               sources=["labelling", "lineage", "pseudotime"],
               orient_config={"labelling_key": "velocity_labelling"},
               fit_kwargs={"n_epochs": 30})
    assert (tmp_path / "summary.json").exists() and (tmp_path / "per_cell.csv").exists()
    assert (tmp_path / "adata.h5ad").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["orient"]["J_unchanged"]
    assert "orientation" in s["figures"] and s["analyse"]["tol"] > 0
    # a second run reuses the stored operator
    s2 = sj.run(a, tmp_path / "again", cluster_key="cluster", use_rep="X_lat")
    assert s2["fit"] == "reused stored J"


def test_compare_conditions(tmp_path):
    pytest.importorskip("scjdo2.compare")
    a = _data()
    rng = np.random.default_rng(0)
    a.obs["cell_type"] = rng.choice(["A", "B"], a.n_obs)
    a.obs["age_group"] = rng.choice(["young", "old"], a.n_obs)
    sj.fit(a, use_rep="X_lat", n_epochs=30)
    t = sj.compare_conditions(a, "cell_type", "age_group", ("young", "old"),
                              min_cells=100, n_null=20)
    assert len(t) == 2 and "conditions" in a.uns["scjdo2"]
    sj.analyse(a, cluster_key="cell_type")
    paths = sj.figures(a, tmp_path, cluster_key="cell_type", condition_key="age_group",
                       conditions=("young", "old"), include=["ternary", "spectrum_ladder"])
    assert set(paths) == {"ternary", "spectrum_ladder"}


def test_compare_conditions_absent_gives_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "scjdo2.compare", None)    # simulate a release without it
    with pytest.raises(ImportError, match="not included in this release"):
        sj.compare_conditions(_data(), "cluster", "cluster", ("early", "late"))
