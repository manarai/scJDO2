"""scjdo2.flow: OT velocity -> drift -> flow operator -> descriptors."""

from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("ot")
pytest.importorskip("torch")

import scjdo2 as sj  # noqa: E402
from scjdo2.flow_operator import DESCRIPTOR_NAMES, ot_velocity  # noqa: E402


def _fork(n_per_time: int = 150, seed: int = 0):
    """Cells drift along x and split in y after t = 2 (a pitchfork in time)."""
    rng = np.random.default_rng(seed)
    z, t = [], []
    for k in range(5):
        x = k + 0.15 * rng.standard_normal(n_per_time)
        arm = rng.choice([-1.0, 1.0], n_per_time)
        y = arm * max(k - 2, 0) * 0.8 + 0.15 * rng.standard_normal(n_per_time)
        w = 0.1 * rng.standard_normal((n_per_time, 2))
        z.append(np.column_stack([x, y, w]))
        t.append(np.full(n_per_time, float(k)))
    return np.vstack(z), np.concatenate(t)


def test_ot_velocity_moves_forward_and_skips_last_time():
    z, t = _fork()
    v, info = ot_velocity(z, t)
    last = t == t.max()
    assert np.isnan(v[last]).all()
    assert np.isfinite(v[~last]).all()
    assert np.median(v[~last, 0]) == pytest.approx(1.0, abs=0.15)   # one unit of x per unit time
    assert [p["solver"] for p in info] == ["dense"] * 4


def test_minibatch_matches_dense():
    z, t = _fork(n_per_time=300)
    vd, _ = ot_velocity(z, t, solver="dense")
    vm, info = ot_velocity(z, t, solver="minibatch", batch_size=100, coverage=10)
    ok = np.isfinite(vd).all(1)
    cos = np.sum(vd[ok] * vm[ok], 1) / (np.linalg.norm(vd[ok], axis=1) * np.linalg.norm(vm[ok], axis=1))
    assert np.median(cos) > 0.98
    assert all(p["coverage_min"] >= 1 for p in info)


def test_flow_writes_outputs(tmp_path):
    z, t = _fork()
    a = anndata.AnnData(X=np.zeros((len(z), 1), np.float32))
    a.obsm["X_pca"] = z
    a.obs["day"] = t
    sj.flow(a, time_key="day", n_comps=4, n_epochs=200, device="cpu")
    n, d = len(z), 4
    assert a.uns["scjdo2"]["J"].shape == (n, d, d)
    assert np.isfinite(a.uns["scjdo2"]["J"]).all()
    assert a.obs["flow_lambda1"].shape == (n,)
    assert a.obs["flow_has_velocity"].sum() == n - (t == t.max()).sum()
    assert a.obsm["flow_drift"].shape == (n, d)
    assert a.obsm["flow_descriptor_ranks"].shape == (n, len(DESCRIPTOR_NAMES))
    r = a.obsm["flow_descriptor_ranks"]
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 1
    assert a.uns["scjdo2"]["meta"]["operator"] == "flow"
    a.write_h5ad(tmp_path / "flow.h5ad")   # outputs must be serialisable


def test_flow_copy_and_bad_inputs():
    z, t = _fork(n_per_time=60)
    a = anndata.AnnData(X=np.zeros((len(z), 1), np.float32))
    a.obsm["X_pca"] = z
    a.obs["day"] = t
    b = sj.flow(a, time_key="day", n_comps=4, n_epochs=50, device="cpu", descriptors=False, copy=True)
    assert "flow_lambda1" in b.obs and "flow_lambda1" not in a.obs
    with pytest.raises(KeyError):
        sj.flow(a, time_key="missing")
    with pytest.raises(ValueError):
        sj.flow(a, time_key="day", n_comps=99)
    a.obs["one"] = 0.0
    with pytest.raises(ValueError):
        sj.flow(a, time_key="one", n_comps=4)
