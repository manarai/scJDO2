"""Fast checks of scjdo2.routes on small synthetic data (no operator fit)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scjdo2.routes import direction as dirn
from scjdo2.routes.report import render_page, warnings
from scjdo2.routes.walk import absorption, cell_walk, drift_kernel


def _line(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.linspace(0, 10, n), 0.1 * rng.standard_normal(n)]
    return X, np.tile([1.0, 0.0], (n, 1))


def test_drift_kernel_follows_drift():
    X, V = _line()
    nb, cw = drift_kernel(X, V, k=10, sigma=0.25)
    w = np.diff(np.c_[np.zeros(len(cw)), cw], axis=1)
    dx = (w * (X[nb, 0] - X[:, None, 0])).sum(1)
    assert np.median(dx) > 0
    nb, cw = drift_kernel(X, V, k=10, sigma=np.inf)
    w = np.diff(np.c_[np.zeros(len(cw)), cw], axis=1)
    assert np.allclose(w, 0.1)


def test_cell_walk_moves_forward_and_stays_on_cells():
    X, V = _line()
    nb, cw = drift_kernel(X, V, k=10, sigma=0.25)
    cells = cell_walk(nb, cw, start_idx=np.arange(20), n_walkers=50, n_frames=100, hops=100, seed=0)
    assert cells.shape == (101, 50)
    assert cells.min() >= 0 and cells.max() < len(X)
    assert X[cells[-1], 0].mean() > X[cells[0], 0].mean() + 2


def test_absorption_rows_sum_to_one():
    rng = np.random.default_rng(1)
    # a fork: stem along x, then two arms up / down
    stem = np.c_[np.linspace(0, 5, 200), 0.05 * rng.standard_normal(200)]
    up = np.c_[np.linspace(5, 8, 100), np.linspace(0, 3, 100)]
    dn = np.c_[np.linspace(5, 8, 100), np.linspace(0, -3, 100)]
    X = np.r_[stem, up, dn]
    V = np.tile([1.0, 0.0], (len(X), 1))
    lab = -np.ones(len(X), int)
    lab[200 + 80:300] = 0
    lab[300 + 80:] = 1
    Y = absorption(X, V, lab, k=10, sigma=0.25, iters=800)
    assert np.allclose(Y[lab >= 0].sum(1), 1)
    s = Y[:50].sum(1)
    assert np.all(s > 0.95)
    assert abs(Y[:50, 0].mean() - 0.5) < 0.25      # symmetric fork: no strong preference


def test_clone_and_ot_velocity_recover_a_shift():
    rng = np.random.default_rng(2)
    n = 300
    Z0 = rng.standard_normal((n, 3))
    Z = np.r_[Z0, Z0 + [2.0, 0, 0]]
    t = np.r_[np.zeros(n), 2 * np.ones(n)]
    clone = np.r_[np.arange(n) // 3, np.arange(n) // 3]
    V = dirn.clone_velocity(Z, t, clone)
    assert np.allclose(np.nanmedian(V, 0), [1.0, 0, 0], atol=1e-6)
    Vo = dirn.ot_velocity(Z, t, eps=0.01)
    assert np.median(Vo[:, 0]) == pytest.approx(1.0, abs=0.15)


def test_alignment_and_choose_source():
    X, V = _line()
    assert dirn.alignment(X, V, X[:, 0]) > 0.9
    assert dirn.alignment(X, -V, X[:, 0]) < -0.9
    a = ad.AnnData(np.zeros((4, 2)), obs=pd.DataFrame({"day": [0, 0, 2, 2], "pt": [0, 1, 2, 3],
                                                        "c": ["a", "a", "b", "b"]}))
    assert dirn.choose_source(a, clone_key="c", time_key="day") == "clone"
    assert dirn.choose_source(a, time_key="day") == "ot"
    assert dirn.choose_source(a, pseudotime_key="pt") == "pseudotime"
    with pytest.raises(ValueError):
        dirn.choose_source(a)


def _diag(**kw):
    d = {"n_cells_total": 1000, "n_cells_operator": 500, "n_genes": 200, "latent_dim": 10,
         "source": "pseudotime", "drift_fit_cells": 1000, "drift_fit_median_cos": 0.9,
         "clock_calibrated": False, "states": ["A", "B", "C"], "merged_states": [], "start_state": "A",
         "n_walkers": 10, "hops": 50, "sigma": 0.25, "k": 30, "final": {"A": 2, "B": 8, "C": 0},
         "visited": {"A": 10, "B": 8, "C": 0}, "never_reached": ["C"], "floor": "UMAP",
         "backdrop_phi_residual": 0.3, "routes": [],
         "archetypes": [{"id": 0, "behaviour": "fork", "n_cells": 100, "n_driver": 0, "n_hub": 9,
                         "n_responder": 0, "asym": 0.2, "top_drivers": "", "top_responders": "",
                         "frac_unmeasured": 0.95}],
         "alignment_pseudotime": 0.1}
    d.update(kw)
    return d


def test_warnings_flag_weak_runs():
    w = " ".join(warnings(_diag()))
    for phrase in ("barely points", "Never reached", "no operator weight", "hubs", "symmetric",
                   "No reference network", "not checked against true fates"):
        assert phrase in w


def test_render_page_fills_every_placeholder():
    view = {"x": [0, 1], "y": [0, 1], "U": [[0, 0], [0, 0]], "inside": [[1, 1], [1, 1]], "names": ["A", "B", "C"],
            "centroids": [], "cells": [[0, 0, 0]], "traj": [[[0, 0]]], "states": [[0]],
            "meta": {"cap": 1, "n_steps": 50, "dt": 1, "final": [2, 8, 0]}}
    page = render_page(view, {"routes": [], "reference": [], "archetypes": []}, _diag(), title="Test Page")
    for ph in ("__TITLE__", "__HEADLINE__", "__SUB__", "__NMEAS__", "__METHODS__", "__CAVEAT__",
               "__DATA__", "__ROUTES__"):
        assert ph not in page
    assert "<title>Test Page</title>" in page and "not a regulatory network" in page
