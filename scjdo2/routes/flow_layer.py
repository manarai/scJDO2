"""Flow λ₁ for route pages: how strongly nearby cells spread apart or converge.

With a time course, :func:`scjdo2.flow_operator.flow` gives every cell a flow
operator ``J_flow`` (optimal transport between consecutive time points →
smooth drift → its Jacobian). Its leading eigenvalue λ₁ (largest real part)
is shown per cell and along each route:

* λ₁ > 0 — **expanding**: nearby cells move apart, as at a fate split;
* λ₁ < 0 — **contracting**: nearby cells converge, as when cells commit.

Why only λ₁ and not flow-J archetypes: on LARRY (atlas/F01, F02) per-cell λ₁
reproduced across replicate wells (Spearman 0.97), but archetypes of
``J_flow`` reproduced only as a two-way split (expanding vs contracting),
which λ₁ alone separates (AUROC 0.90). The values of ``J_flow`` do not
recover a true Jacobian and depend on the transport space; the reference
settings (PCA of log-normalised counts, 15 components) are used.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

N_COMPS = 15


def available() -> bool:
    try:
        import ot  # noqa: F401
        return True
    except ImportError:
        return False


def flow_lambda(Xlog, tm, keep, *, n_comps=N_COMPS, max_cells=60000, n_epochs=1500, seed=0):
    """λ₁ of the flow operator at cells ``keep``.

    ``Xlog``: log-normalised expression (cells × genes, the run's variable genes);
    ``tm``: numeric time point per cell. The flow is fitted on ``keep`` plus a
    random sample of the other cells, up to ``max_cells`` in all.
    Returns (λ₁ at ``keep``, info dict).
    """
    import anndata as ad
    import scanpy as sc

    from ..flow_operator import flow

    tm = np.asarray(tm, float)
    keep = np.asarray(keep, int)
    ok = np.flatnonzero(np.isfinite(tm))
    if np.unique(tm[ok]).size < 2:
        raise ValueError("flow λ₁ needs at least two time points")
    if not np.isfinite(tm[keep]).all():
        raise ValueError("every shown cell needs a time point for flow λ₁")
    rng = np.random.default_rng(seed)
    rest = np.setdiff1d(ok, keep)
    extra = rng.choice(rest, min(len(rest), max(0, max_cells - len(keep))), replace=False)
    rows = np.sort(np.concatenate([keep, extra]))
    a = ad.AnnData(Xlog[rows], obs=pd.DataFrame({"t": tm[rows]}, index=[str(i) for i in rows]))
    n_comps = int(min(n_comps, a.n_vars - 1, a.n_obs - 1))
    sc.pp.pca(a, n_comps=n_comps, random_state=seed)
    tic = _time.monotonic()
    flow(a, "t", use_rep="X_pca", n_comps=n_comps, n_epochs=n_epochs, seed=seed,
         descriptors=False, keep_plan=None)
    lam = a.obs["flow_lambda1"].to_numpy(float)[np.searchsorted(rows, keep)]
    times = np.unique(tm[rows])
    info = {"fit_cells": int(len(rows)), "n_comps": n_comps, "n_times": int(times.size),
            "times": [float(t) for t in times], "seconds": round(_time.monotonic() - tic, 1),
            "drift_fit": {k: v for k, v in a.uns["flow"]["drift_fit"].items()
                          if isinstance(v, (int, float, str, bool))}}
    return lam, info


def summarise(lam, states, names, tm_shown, routes) -> dict:
    """Numbers for the report: λ₁ by state, along each route, and coverage of the last time point."""
    lam = np.asarray(lam, float)
    by_state = {}
    for j, n in enumerate(names):
        v = lam[states == j]
        if len(v):
            by_state[n] = {"median": round(float(np.median(v)), 3), "n": int(len(v)),
                           "frac_expanding": round(float((v > 0).mean()), 3)}
    last = np.nanmax(tm_shown)
    per_route = []
    for r in routes:
        c = [x for x in r.get("lambda_curve", []) if x is not None]
        if c:
            per_route.append({"path": r["states"], "start": round(c[0], 3), "end": round(c[-1], 3)})
    q = np.quantile(lam, [0.05, 0.5, 0.95])
    return {"quantiles_5_50_95": [round(float(x), 3) for x in q],
            "frac_expanding": round(float((lam > 0).mean()), 3),
            "frac_last_time": round(float((tm_shown == last).mean()), 3),
            "by_state": by_state, "routes": per_route}
