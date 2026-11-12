"""One call: from an AnnData with raw counts to a route page and a report.

    from scjdo2.routes import simulate
    res = simulate(adata, state_key="cell_type", start_state="HSC", out_dir="results",
                   time_key="day", clone_key="clone")

Steps (each a public function in :mod:`scjdo2.routes`):

1. latent space: normalise → log1p → HVGs → PCA, unless ``latent_key`` is given
   (required with ``velocity_key``, which must live in that space);
2. direction: ``direction.choose_source`` (clone > velocity > ot > pseudotime)
   turned into a latent velocity; ``MeasuredDrift`` fitted to it;
3. operator: ``n_cells`` cells (stratified by state), Phase C gene field
   (``pipeline.fit_rate_and_gene_field``) and ``opspace.compute``;
4. walk: walkers hop between those cells along the drift (``walk``);
5. routes: archetypes, genes and gene coupling along each route (``dynamics``);
   with time points, also the flow operator's λ₁ per cell and along each
   route (``flow_layer``: expanding vs contracting neighbourhoods);
6. checks: drift alignment with time / pseudotime, operator directionality,
   gene coverage, reference-network support, and — with clones and time —
   the held-out clone fate test (``fate``);
7. outputs: ``route_page.html``, ``report.md``, CSV tables, ``diagnostics.json``.
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import direction as dirn
from . import flow_layer
from .backdrop import surface
from .dynamics import operator_tables, route_dynamics
from .report import write_outputs
from .walk import cell_walk, drift_kernel, smooth_positions

MAX_STATES = 8   # the viewer has eight state colours


def _log(msg):
    print(f"[scjdo2.routes] {msg}", flush=True)


def _counts(adata):
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = sp.csr_matrix(X)
    sample = X.data[: min(len(X.data), 100_000)]
    if len(sample) and (np.any(sample < 0) or not np.allclose(sample, np.round(sample))):
        raise ValueError("needs raw counts in layers['counts'] or X (found non-integer or negative values)")
    return X.astype(np.float32)


def _merge_states(st, start_state):
    vc = pd.Series(st).value_counts()
    keep = [start_state] + [s for s in vc.index if s != start_state][: MAX_STATES - 2]
    if len(vc) <= MAX_STATES:
        return st, []
    merged = [s for s in vc.index if s not in keep]
    return np.where(np.isin(st, keep), st, "Other"), merged


def simulate(adata, state_key, start_state, out_dir, *, clone_key=None, time_key=None,
             velocity_key=None, latent_key=None, pseudotime_key=None, source="auto",
             basis=None, fate_states=None, reference=None, n_cells=2000, n_pcs=10,
             n_top_genes=2000, keep_genes=(), n_walkers=300, hops=150, sigma=0.25, k=30,
             drift_epochs=1500, rate_epochs=150, n_anchors=300, anchor_max_live=200,
             fate_test=True, fate_max_cells=60000, max_fit_cells=60000, title=None,
             dataset_name=None, seed=0, save_run=False, backdrop_steps=6000,
             flow_lambda="auto", flow_max_cells=60000) -> dict:
    """Run the whole route simulation; see the module docstring. Returns the diagnostics dict.

    ``flow_lambda``: ``"auto"`` (default) adds flow λ₁ when ``time_key`` gives at least two
    time points and the POT package is installed; ``True`` requires it; ``False`` skips it.
    """
    import scanpy as sc
    from .. import io as scjdo2_io
    from .. import opspace
    from ..measured_drift import MeasuredDrift, MeasuredDriftConfig
    from ..pipeline import fit_rate_and_gene_field

    t_start = _time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if state_key not in adata.obs:
        raise ValueError(f"state_key {state_key!r} not in adata.obs")
    st_raw = adata.obs[state_key].astype(str).to_numpy()
    if start_state not in set(st_raw):
        raise ValueError(f"start_state {start_state!r} not among {sorted(set(st_raw))[:20]}")
    st, merged = _merge_states(st_raw, start_state)
    if merged:
        _log(f"more than {MAX_STATES} states: merged {len(merged)} rarest into 'Other'")
    counts = _counts(adata)
    tm = adata.obs[time_key].astype(float).to_numpy() if time_key else None

    # 1. latent space and genes
    tmp = sc.AnnData(counts.copy(), obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names))
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in keep_genes:
        if g in tmp.var_names:
            hv[tmp.var_names.get_loc(g)] = True
    if latent_key:
        Z = np.asarray(adata.obsm[latent_key], float)
    else:
        t2 = tmp[:, hv].copy()
        sc.pp.pca(t2, n_comps=n_pcs, random_state=seed)
        Z = np.asarray(t2.obsm["X_pca"], float)
    if velocity_key and not latent_key:
        raise ValueError("velocity_key needs latent_key: the velocity must be in that latent space")
    _log(f"{adata.n_obs} cells, {int(hv.sum())} genes kept, latent dim {Z.shape[1]}")

    # 2. direction → drift
    if source == "auto":
        source = dirn.choose_source(adata, clone_key, time_key, velocity_key, pseudotime_key)
    clone = dirn.clone_labels(adata, clone_key) if clone_key else None
    if source == "clone":
        V = dirn.clone_velocity(Z, tm, clone)
    elif source == "velocity":
        V = np.asarray(adata.obsm[velocity_key], float)
    elif source == "ot":
        V = dirn.ot_velocity(Z, tm, seed=seed)
    else:
        pt = adata.obs[pseudotime_key].astype(float).to_numpy()
        V = dirn.gradient_velocity(Z, pt, k=k)
    fit = np.flatnonzero(np.isfinite(V).all(1))
    if len(fit) < 200:
        raise ValueError(f"only {len(fit)} cells have a {source} velocity")
    if len(fit) > max_fit_cells:
        fit = np.sort(rng.choice(fit, max_fit_cells, replace=False))
    calibrated = source in ("clone", "ot")

    def fit_drift(rows, vel, s=seed):
        return MeasuredDrift(dim=Z.shape[1], config=MeasuredDriftConfig(
            n_epochs=drift_epochs, seed=s, clock_calibrated=calibrated)).fit(
            Z[rows].astype(np.float32), vel[rows].astype(np.float32))
    drift = fit_drift(fit, V)
    vp = drift.velocity(Z[fit].astype(np.float32))
    fit_cos = float(np.median((vp * V[fit]).sum(1) / (np.linalg.norm(vp, axis=1) * np.linalg.norm(V[fit], axis=1) + 1e-12)))
    _log(f"direction source: {source}; MeasuredDrift on {len(fit)} cells, median cosine {fit_cos:.3f}")

    # 3. operator on a stratified subsample
    n_cells = min(n_cells, adata.n_obs)
    idx = []
    for s, c in pd.Series(st).value_counts().items():
        pool = np.flatnonzero(st == s)
        idx.append(rng.choice(pool, min(len(pool), max(30, int(round(n_cells * c / len(st))))), replace=False))
    idx = np.sort(np.concatenate(idx))
    if len(idx) > n_cells:
        idx = np.sort(rng.choice(idx, n_cells, replace=False))
    Zs = Z[idx]
    Vs = drift.velocity(Zs.astype(np.float32)).astype(float)
    x = sc.AnnData(counts[idx][:, np.flatnonzero(hv)], obs=pd.DataFrame({"state": st[idx]}, index=adata.obs_names[idx]),
                   var=pd.DataFrame(index=adata.var_names[hv]))
    x.layers["counts"] = x.X.copy()
    x.obsm["X_pca"] = Zs.astype(np.float32)
    x.obsm["X_drift_v"] = Vs.astype(np.float32)
    x.uns["scjdo2"] = {"J": drift.jacobian_autodiff(Zs.astype(np.float32)),
                       "meta": {"clock_calibrated": calibrated, "use_rep": "X_pca", "time_key": time_key,
                                "jacobian_method": "autodiff",
                                "drift": {"method": "measured_mlp_regression", "source": source},
                                "field": {"method": "autodiff"}},
                       "field_reason_summary": {"ok": 1.0, "autodiff": True}}
    _log(f"operator fit on {len(idx)} cells (Phase C; memory grows with cells × genes × {anchor_max_live})")
    t0 = _time.monotonic()
    fit_rate_and_gene_field(x, use_rep="X_pca", n_epochs_rate=rate_epochs, k=k, cv_sample_cells=20, seed=seed,
                            anchor_fit=True, n_anchors=min(n_anchors, len(idx) // 5), anchor_top_k=10,
                            anchor_max_live=anchor_max_live)
    opspace.compute(x)
    _log(f"operator done in {_time.monotonic() - t0:.0f}s")
    ops = operator_tables(x, reference, seed=seed)

    # 4. walk
    names = [start_state] + [s for s in pd.Series(st[idx]).value_counts().index if s != start_state]
    lab = np.array([names.index(s) for s in st[idx]])
    start = np.flatnonzero(lab == 0)
    if tm is not None:
        early = start[tm[idx][start] == tm[idx][start].min()]
        start = early if len(early) >= 20 else start
    nb, cw = drift_kernel(Zs, Vs, k=k, sigma=sigma)
    cells = cell_walk(nb, cw, start, n_walkers=n_walkers, hops=hops, seed=seed)
    states = lab[cells]
    final = np.bincount(states[-1], minlength=len(names))
    visited = (np.stack([(states == j).any(0) for j in range(len(names))])).sum(1)
    _log("walkers end in: " + ", ".join(f"{n} {c}" for n, c in zip(names, final) if c))

    # floor + backdrop
    if basis and basis in adata.obsm:
        u = np.asarray(adata.obsm[basis], float)[idx][:, :2]
        floor_name = basis
    else:
        sc.pp.neighbors(x, use_rep="X_pca", n_neighbors=30, random_state=seed)
        sc.tl.umap(x, random_state=seed)
        u, floor_name = np.asarray(x.obsm["X_umap"], float), "UMAP of the latent space"
    u = (u - u.mean(0)) / u.std()
    bd = surface(u, Zs, Vs, seed=seed, potential_steps=backdrop_steps)
    traj = smooth_positions(cells, u, seed=seed)
    cent = [{"name": n, "x": float(np.median(u[lab == j, 0])), "y": float(np.median(u[lab == j, 1])),
             "n": int((lab == j).sum())} for j, n in enumerate(names)]
    view = {"x": bd["x"], "y": bd["y"], "U": bd["U"], "inside": bd["inside"], "names": names, "centroids": cent,
            "cells": np.c_[u.round(3), lab].tolist(), "traj": traj.round(3).tolist(), "states": states.tolist(),
            "meta": {"cap": bd["cap"], "n_steps": hops, "dt": 1.0, "clock_label": "hops", "clock_digits": 0,
                     "n_flow": n_walkers, "final": final.tolist()}}

    # flow λ₁ (time course only)
    lam, flow_info = None, None
    n_times = np.unique(tm[np.isfinite(tm)]).size if tm is not None else 0
    want = flow_lambda is True or (flow_lambda == "auto" and n_times >= 2)
    if flow_lambda not in (True, False, "auto"):
        raise ValueError("flow_lambda must be 'auto', True or False")
    if want and not flow_layer.available():
        if flow_lambda is True:
            raise ImportError("flow λ₁ needs the POT package (pip install pot)")
        _log("flow λ₁ skipped: the POT package is not installed")
        want = False
    if flow_lambda is True and n_times < 2:
        raise ValueError("flow_lambda=True needs time_key with at least two time points")
    if want:
        _log(f"flow λ₁: optimal transport over {n_times} time points, up to {flow_max_cells} cells")
        lam, flow_info = flow_layer.flow_lambda(tmp.X[:, np.flatnonzero(hv)], tm, idx, max_cells=flow_max_cells,
                                                n_epochs=drift_epochs, seed=seed)
        lim = float(np.quantile(np.abs(lam), 0.98)) or 1.0
        view["lambda"] = np.round(lam, 3).tolist()
        view["meta"]["lambda"] = {"lim": round(lim, 3), "units": f"per unit of {time_key}"}
        _log(f"flow λ₁ done in {flow_info['seconds']:.0f}s; median {np.median(lam):.2f}")

    # 5. routes
    routes = route_dynamics(cells, states, names, start_state, Zs, ops, reference, lam=lam)

    # 6. checks
    diag = {"n_cells_total": int(adata.n_obs), "n_cells_operator": int(len(idx)), "n_genes": int(hv.sum()),
            "latent_dim": int(Z.shape[1]), "source": source, "drift_fit_cells": int(len(fit)),
            "drift_fit_median_cos": round(fit_cos, 3), "clock_calibrated": calibrated,
            "states": names, "merged_states": merged, "start_state": start_state,
            "n_walkers": n_walkers, "hops": hops, "sigma": sigma, "k": k,
            "final": dict(zip(names, final.tolist())), "visited": dict(zip(names, visited.tolist())),
            "never_reached": [n for n, v in zip(names, visited) if v == 0],
            "floor": floor_name, "backdrop_phi_residual": bd["phi_residual"],
            "routes": [{"path": r["states"], "n": r["n"], "top_archetypes": r["top"]} for r in routes["routes"]]}
    if lam is not None:
        diag["flow_lambda"] = {**flow_info, "time_key": time_key, "units": f"per unit of {time_key}",
                               **flow_layer.summarise(lam, lab, names, tm[idx], routes["routes"])}
    # alignment needs a continuous key: a local gradient of a few time points is noise
    for key, name in ((time_key, "time"), (pseudotime_key, "pseudotime")):
        if key and key in adata.obs and adata.obs[key].nunique() > 20:
            diag[f"alignment_{name}"] = round(dirn.alignment(Zs, Vs, adata.obs[key].astype(float).to_numpy()[idx], k=k), 3)
    arch_rows = []
    for a_ in routes["archetypes"]:
        tbl = ops["arch"][a_["id"]]["influence"]
        arch_rows.append({**{kk: a_[kk] for kk in ("id", "behaviour", "n_cells", "n_driver", "n_hub",
                                                   "n_responder", "asym", "top_drivers", "top_responders")},
                          "frac_unmeasured": round(float((tbl["role"] == "unmeasured").mean()), 3)})
    diag["archetypes"] = arch_rows
    if ops["ref_summary"] is not None:
        rs = ops["ref_summary"]
        diag["reference"] = {"n_archetypes": int(len(rs)),
                             "above_random": int((rs["p_jacobian"] < 0.05).sum()),
                             "above_coexpression": int((rs["mean_auroc_jacobian"] > rs["mean_auroc_coexpr"]).sum()),
                             "auroc_range": [round(float(rs["mean_auroc_jacobian"].min()), 3),
                                             round(float(rs["mean_auroc_jacobian"].max()), 3)]}
    fate_tbl = None
    if fate_test and clone is not None and tm is not None:
        fs = list(fate_states) if fate_states else [s for s in names if s not in (start_state, "Other")]
        rows = np.arange(len(Z))
        if len(rows) > fate_max_cells:
            must = np.flatnonzero((clone >= 0) & (tm == tm.min()))
            rest = np.setdiff1d(rows, must)
            rows = np.sort(np.concatenate([must, rng.choice(rest, max(0, fate_max_cells - len(must)), replace=False)]))
        Zf, cf, tf_, sf = Z[rows], clone[rows], tm[rows], st[rows]

        def vel_fn(train):
            if train is None:
                return drift.velocity(Zf.astype(np.float32)).astype(float)
            Vt = dirn.clone_velocity(Z, tm, clone, clones=train)
            fr = np.flatnonzero(np.isfinite(Vt).all(1))
            return fit_drift(fr, Vt).velocity(Zf.astype(np.float32)).astype(float)
        from .fate import clone_fate_test
        _log("held-out clone fate test")
        try:
            fate_tbl, fate_sum = clone_fate_test(Zf, tf_, cf, sf, fs, vel_fn, refit_on_train=(source == "clone"),
                                                 k=k, seed=seed)
            diag["fate_test"] = {"summary": json.loads(fate_sum.round(3).to_json(orient="index")),
                                 "fate_states": fs, "refit_on_train": source == "clone"}
        except ValueError as err:
            diag["fate_test"] = {"skipped": str(err)}
    diag["runtime_s"] = round(_time.monotonic() - t_start, 1)

    # 7. outputs
    write_outputs(out, view, routes, diag, fate_tbl, title=title, dataset_name=dataset_name)
    if save_run:
        scjdo2_io.save(x, out / "run")
    _log(f"wrote {out} in {diag['runtime_s']:.0f}s")
    return diag
