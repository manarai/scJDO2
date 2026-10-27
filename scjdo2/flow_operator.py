"""Flow operator from a sampled time course: ``scjdo2.flow(adata, time_key=...)``.

One call runs the label-free pipeline evaluated in the v57 manuscript (Results 3-5):

1. **OT velocity.** Balanced entropic optimal transport (Sinkhorn) between cells of
   consecutive sampling times, with ``epsilon = eps x median squared distance`` of the
   pair. Each cell of the earlier time gets the barycentric velocity
   ``(barycentre of its image - z) / dt``. Cells of the last time point get none.
2. **Drift.** A smooth drift ``f(z)`` (:class:`scjdo2.measured_drift.MeasuredDrift`,
   an MLP) is fitted to those velocities. It is evaluated at every cell, so the last
   time point is covered by extrapolation.
3. **Flow operator.** ``J_flow = df/dz`` at every cell, by automatic differentiation.
4. **Descriptors.** The leading eigenvalue ``lambda_1`` (max Re) and the
   magnitude-free, basis-invariant operator-space descriptors, rank-transformed
   within the dataset.

The defaults are the a-priori "reference" configuration of the manuscript: PCA of
log-normalised counts with 15 components, ``eps = 0.05``, MLP hidden (96, 96, 96),
weight decay 1e-4, one fit seed. Large time pairs switch from dense to mini-batch
transport (4,000-cell batches, each source cell in ten batches), which reproduced
dense transport with velocity cosine 0.9998 (manuscript Results 5).

What this operator is and is not (manuscript Results 3): ``lambda_1`` ranks cells
at a fate bifurcation above committed cells. The values of ``J_flow`` do **not**
recover the true Jacobian, and the operator depends on the transport metric (it
inverts on Factor Analysis). Use a variance-weighted space such as PCA.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["flow", "ot_velocity", "drift_jacobian", "flow_descriptors", "DESCRIPTOR_NAMES"]

DENSE_MAX_PAIRS = 64_000_000   # n_source x n_target above which mini-batch OT is used
BATCH_SIZE = 4000
MINIBATCH_COVERAGE = 10

DESCRIPTOR_NAMES = (
    "sign_lead_re",
    "n_positive_real",
    "lead_re_over_max_abs_lambda",
    "max_abs_imag_over_max_abs_lambda",
    "spectral_gap_over_max_abs_lambda",
    "henrici",
    "transient_gain",
    "trace_over_sum_abs_lambda",
)


# ---------------------------------------------------------------------------
# 1. OT velocity
# ---------------------------------------------------------------------------
def _median_sq_dist(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, str]:
    if x.shape[0] * y.shape[0] <= DENSE_MAX_PAIRS:
        import ot
        return float(np.median(ot.dist(x, y, metric="sqeuclidean"))), "exact"
    rng = np.random.default_rng(seed)
    n = 2_000_000
    r, c = rng.integers(0, len(x), n), rng.integers(0, len(y), n)
    return float(np.median(np.sum((x[r] - y[c]) ** 2, axis=1))), f"{n:,}-pair sample"


def _dense_bary(x: np.ndarray, y: np.ndarray, eps: float) -> tuple[np.ndarray, dict[str, Any]]:
    import ot
    C = ot.dist(x, y, metric="sqeuclidean")
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        P, log = ot.sinkhorn(a, b, C, reg=eps, log=True)
    rows = P.sum(axis=1, keepdims=True)
    err = log.get("err", [])
    return (P @ y) / np.maximum(rows, 1e-300), {
        "solver": "dense", "n_iter": len(err), "final_err": float(err[-1]) if len(err) else None,
        "rows_zero": int((rows[:, 0] <= 0).sum()), "warnings": [str(w.message)[:200] for w in caught]}


def _minibatch_bary(x: np.ndarray, y: np.ndarray, eps: float, seed: int, batch_size: int,
                    coverage: int) -> tuple[np.ndarray, dict[str, Any]]:
    import ot
    rng = np.random.default_rng(seed)
    schedule = np.tile(np.arange(len(x)), coverage)
    rng.shuffle(schedule)
    bary = np.zeros_like(x)
    seen = np.zeros(len(x), dtype=np.int64)
    n_batches = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for lo in range(0, len(schedule), batch_size):
            si = schedule[lo:lo + batch_size]
            ti = rng.choice(len(y), size=batch_size, replace=len(y) < batch_size)
            xb, yb = x[si], y[ti]
            P = ot.sinkhorn(np.full(len(xb), 1.0 / len(xb)), np.full(len(yb), 1.0 / len(yb)),
                            ot.dist(xb, yb, metric="sqeuclidean"), reg=eps)
            mass = P.sum(axis=1, keepdims=True)
            np.add.at(bary, si, (P @ yb) / np.maximum(mass, 1e-12))
            np.add.at(seen, si, 1)
            n_batches += 1
    bary /= seen[:, None]
    return bary, {"solver": "minibatch", "batch_size": batch_size, "n_batches": n_batches,
                  "coverage_min": int(seen.min()), "warnings": sorted({str(w.message)[:200] for w in caught})}


def ot_velocity(z: np.ndarray, times: np.ndarray, *, eps: float = 0.05, solver: str = "auto",
                batch_size: int = BATCH_SIZE, coverage: int = MINIBATCH_COVERAGE, seed: int = 0,
                ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Barycentric entropic-OT velocity between consecutive sampling times.

    Parameters
    ----------
    z : (n, d) coordinates. The transport cost is squared Euclidean distance in ``z``.
    times : (n,) numeric sampling time per cell. ``dt`` between two times is their
        difference, so velocities are per unit of ``times``.
    eps : entropic regularisation as a fraction of the pair's median squared distance.
    solver : ``"dense"``, ``"minibatch"`` or ``"auto"`` (mini-batch when
        ``n_source x n_target > 64e6``).

    Returns
    -------
    v : (n, d) float32 velocity, NaN for cells of the last time point.
    info : one record per time pair (epsilon, dt, solver, convergence).
    """
    if solver not in ("auto", "dense", "minibatch"):
        raise ValueError(f"solver must be 'auto', 'dense' or 'minibatch'; got {solver!r}")
    z = np.asarray(z, np.float64)
    times = np.asarray(times, np.float64)
    if z.ndim != 2 or times.shape != (len(z),):
        raise ValueError(f"z must be (n, d) and times (n,); got {z.shape} and {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times contains NaN or inf")
    uniq = np.unique(times)
    if len(uniq) < 2:
        raise ValueError("need at least two sampling times")
    v = np.full(z.shape, np.nan)
    info: list[dict[str, Any]] = []
    for t0, t1 in zip(uniq[:-1], uniq[1:]):
        i0, i1 = np.flatnonzero(times == t0), np.flatnonzero(times == t1)
        x, y = z[i0], z[i1]
        med, how = _median_sq_dist(x, y, seed)
        e = float(eps * med)
        if not np.isfinite(e) or e <= 0:
            raise RuntimeError(f"invalid epsilon {e} for time pair {t0:g} -> {t1:g}")
        use = solver if solver != "auto" else ("dense" if len(x) * len(y) <= DENSE_MAX_PAIRS else "minibatch")
        tic = time.time()
        if use == "dense":
            bary, rec = _dense_bary(x, y, e)
        else:
            bary, rec = _minibatch_bary(x, y, e, seed, batch_size, coverage)
        dt = float(t1 - t0)
        v[i0] = (bary - x) / dt
        info.append({"from_time": float(t0), "to_time": float(t1), "dt": dt, "n_source": int(len(i0)),
                     "n_target": int(len(i1)), "epsilon": e, "eps_factor": float(eps), "median_method": how,
                     "seconds": round(time.time() - tic, 2), **rec})
    return v.astype(np.float32), info


# ---------------------------------------------------------------------------
# 2-3. Drift and its Jacobian
# ---------------------------------------------------------------------------
def _batched_jacobian(model, z: np.ndarray, chunk: int = 4096) -> np.ndarray:
    import torch
    from torch.func import jacrev, vmap
    model = model.to("cpu").eval()
    jac = vmap(jacrev(lambda zi: model(zi[None])[0]))
    out = np.empty((len(z), z.shape[1], z.shape[1]), np.float32)
    with torch.no_grad():
        for lo in range(0, len(z), chunk):
            zt = torch.as_tensor(z[lo:lo + chunk], dtype=torch.float32)
            out[lo:lo + chunk] = jac(zt).numpy()
    return out


def drift_jacobian(z: np.ndarray, v: np.ndarray, *, hidden: Sequence[int] = (96, 96, 96),
                   weight_decay: float = 1e-4, n_epochs: int = 1500, n_seeds: int = 1, seed: int = 0,
                   device: Optional[str] = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit ``MeasuredDrift`` to the cells with a finite velocity; return the per-cell
    Jacobian ``df/dz`` (n, d, d) and fitted drift ``f(z)`` (n, d) at **every** cell.

    With ``n_seeds > 1`` the Jacobians and drifts of seeds ``seed .. seed+n_seeds-1``
    are averaged (the manuscript's "ensemble" estimator).
    """
    from .measured_drift import MeasuredDrift
    z32 = np.asarray(z, np.float32)
    v32 = np.asarray(v, np.float32)
    ok = np.isfinite(v32).all(axis=1)
    if ok.sum() < 2:
        raise ValueError("fewer than two cells have a finite velocity")
    Js, fs, devices = [], [], []
    tic = time.time()
    for s in range(seed, seed + n_seeds):
        md = MeasuredDrift(z32.shape[1], hidden=tuple(hidden), weight_decay=float(weight_decay),
                           n_epochs=int(n_epochs), seed=int(s), device=device)
        md.fit(z32[ok], v32[ok])
        devices.append(str(next(md._model.parameters()).device))
        md._model.to("cpu")
        fs.append(md.velocity(z32).astype(np.float64))
        Js.append(_batched_jacobian(md._model, z32).astype(np.float64))
    J = np.mean(Js, axis=0).astype(np.float32)
    f = np.mean(fs, axis=0).astype(np.float32)
    return J, f, {"n_fit_cells": int(ok.sum()), "hidden": list(hidden), "weight_decay": float(weight_decay),
                  "n_epochs": int(n_epochs), "seeds": list(range(seed, seed + n_seeds)), "fit_device": devices[0],
                  "jacobian": "autodiff (torch.func.jacrev), CPU", "seconds": round(time.time() - tic, 2)}


# ---------------------------------------------------------------------------
# 4. Descriptors
# ---------------------------------------------------------------------------
def _rank_quantiles(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, np.float64)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() == 1:
        out[ok] = 0.5
    elif ok.sum() > 1:
        out[ok] = (rankdata(x[ok], method="average") - 1.0) / (ok.sum() - 1.0)
    return out


def flow_descriptors(J: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Magnitude-free operator-space descriptors of per-cell operators (n, d, d).

    Returns ``(raw, ranks, meta)``: ``raw`` (n, 8) in the order of
    :data:`DESCRIPTOR_NAMES`; ``ranks`` the within-dataset rank quantiles in [0, 1];
    Henrici non-normality and transient gain come from :mod:`scjdo2.opspace`.
    """
    import anndata as ad
    from . import opspace
    J = np.asarray(J, np.float32)
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        raise ValueError(f"J must be (n, d, d); got {J.shape}")
    a = ad.AnnData(X=np.zeros((len(J), 0), dtype=np.float32))
    a.uns["scjdo2"] = {"J": J}
    opspace.spectrum(a, n_eig=int(J.shape[1]), fisher_mask=False)
    opspace.invariants(a)
    eig = np.linalg.eigvals(J.astype(np.float64))
    eig = np.take_along_axis(eig, np.argsort(-eig.real, axis=1, kind="stable"), axis=1)
    max_abs = np.maximum(np.abs(eig).max(axis=1), 1e-300)
    sum_abs = np.maximum(np.abs(eig).sum(axis=1), 1e-300)
    lead = eig[:, 0].real
    gap = lead - eig[:, 1].real if eig.shape[1] > 1 else np.zeros(len(eig))
    cols = {
        "sign_lead_re": np.sign(lead),
        "n_positive_real": (eig.real > 0).sum(axis=1).astype(np.float64),
        "lead_re_over_max_abs_lambda": lead / max_abs,
        "max_abs_imag_over_max_abs_lambda": np.abs(eig.imag).max(axis=1) / max_abs,
        "spectral_gap_over_max_abs_lambda": gap / max_abs,
        "henrici": np.asarray(a.obs["henrici"], np.float64),
        "transient_gain": np.asarray(a.obs["transient_gain"], np.float64),
        "trace_over_sum_abs_lambda": np.trace(J.astype(np.float64), axis1=1, axis2=2) / sum_abs,
    }
    raw = np.column_stack([cols[k] for k in DESCRIPTOR_NAMES])
    ranks = np.column_stack([_rank_quantiles(cols[k]) for k in DESCRIPTOR_NAMES])
    return raw, ranks, {"lambda1": lead, "eigenvalues": eig}


def _pairs_table(pairs: list[dict[str, Any]]):
    """Per-pair OT records as a DataFrame (h5ad-writable)."""
    import pandas as pd
    rows = [{k: ("; ".join(v) if isinstance(v, list) else v) for k, v in p.items()} for p in pairs]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-call API
# ---------------------------------------------------------------------------
def flow(adata, time_key: str, *, use_rep: str = "X_pca", n_comps: Optional[int] = 15,
         eps: float = 0.05, solver: str = "auto", hidden: Sequence[int] = (96, 96, 96),
         weight_decay: float = 1e-4, n_epochs: int = 1500, n_seeds: int = 1, seed: int = 0,
         device: Optional[str] = None, descriptors: bool = True, key_added: str = "flow",
         copy: bool = False):
    """Label-free flow operator of every cell from a sampled time course.

    Parameters
    ----------
    adata : AnnData with a numeric sampling time in ``adata.obs[time_key]``.
    time_key : obs column with the sampling time (e.g. day or hours). Velocities and
        ``lambda_1`` are per unit of this column.
    use_rep : ``obsm`` key of the transport space. Default ``"X_pca"``; if missing it
        is computed with ``scanpy.tl.pca`` on ``adata.X`` (which should then hold
        log-normalised counts). Avoid Factor Analysis: the operator inverts on it.
    n_comps : leading components of ``use_rep`` to use (manuscript: 10-20, default 15).
        ``None`` uses all.
    eps, solver : see :func:`ot_velocity`.
    hidden, weight_decay, n_epochs, n_seeds, seed, device : see :func:`drift_jacobian`.
    descriptors : also compute the operator-space descriptors (needs ``n_comps >= 2``).
    key_added : prefix of the output keys.
    copy : return a modified copy instead of writing to ``adata``.

    Writes
    ------
    ``obs[f"{key}_lambda1"]``           leading eigenvalue (max Re) of ``J_flow``
    ``obs[f"{key}_has_velocity"]``      cell had an OT velocity (False on the last time)
    ``obsm[f"{key}_velocity"]``         OT velocity (NaN on the last time point)
    ``obsm[f"{key}_drift"]``            fitted drift ``f(z)`` at every cell
    ``obsm[f"{key}_descriptors"]``      raw descriptors (columns: ``uns[...]['descriptor_names']``)
    ``obsm[f"{key}_descriptor_ranks"]`` rank quantiles of the descriptors
    ``uns["scjdo2"]["J"]``              ``J_flow`` (n, d, d) float32, read by :mod:`scjdo2.opspace`
    ``uns["scjdo2"]["meta"]``           operator kind, clock units, provenance
    ``uns[key]``                        parameters and per-pair OT diagnostics

    Returns the AnnData if ``copy`` else ``None``.
    """
    adata = adata.copy() if copy else adata
    if time_key not in adata.obs:
        raise KeyError(f"adata.obs[{time_key!r}] not found")
    times = np.asarray(adata.obs[time_key], dtype=np.float64)
    if use_rep not in adata.obsm:
        if use_rep != "X_pca":
            raise KeyError(f"adata.obsm[{use_rep!r}] not found")
        import scanpy as sc
        sc.tl.pca(adata, n_comps=max(int(n_comps or 50), 2))
    z = np.asarray(adata.obsm[use_rep], np.float64)
    if n_comps is not None:
        if n_comps > z.shape[1]:
            raise ValueError(f"n_comps={n_comps} > {z.shape[1]} columns in obsm[{use_rep!r}]")
        z = z[:, :n_comps]
    tic = time.time()
    v, pairs = ot_velocity(z, times, eps=eps, solver=solver, seed=seed)
    J, f, fit = drift_jacobian(z, v, hidden=hidden, weight_decay=weight_decay, n_epochs=n_epochs,
                               n_seeds=n_seeds, seed=seed, device=device)

    adata.obsm[f"{key_added}_velocity"] = v
    adata.obsm[f"{key_added}_drift"] = f
    adata.obs[f"{key_added}_has_velocity"] = np.isfinite(v).all(axis=1)
    eig_lead = np.linalg.eigvals(J.astype(np.float64)).real.max(axis=1)
    adata.obs[f"{key_added}_lambda1"] = eig_lead.astype(np.float32)
    if descriptors:
        raw, ranks, _ = flow_descriptors(J)
        adata.obsm[f"{key_added}_descriptors"] = raw.astype(np.float32)
        adata.obsm[f"{key_added}_descriptor_ranks"] = ranks.astype(np.float32)

    sj = adata.uns.setdefault("scjdo2", {})
    sj["J"] = J
    sj["meta"] = {**dict(sj.get("meta", {})), "operator": "flow", "source": "entropic OT time course",
                  "use_rep": use_rep, "n_comps": int(z.shape[1]), "time_key": time_key,
                  "clock_calibrated": False, "velocity_units": f"per unit of obs[{time_key!r}]"}
    adata.uns[key_added] = {
        "params": {"time_key": time_key, "use_rep": use_rep, "n_comps": int(z.shape[1]), "eps": float(eps),
                   "solver": solver, "hidden": list(hidden), "weight_decay": float(weight_decay),
                   "n_epochs": int(n_epochs), "n_seeds": int(n_seeds), "seed": int(seed)},
        "ot_pairs": _pairs_table(pairs), "drift_fit": fit, "descriptor_names": list(DESCRIPTOR_NAMES),
        "seconds": round(time.time() - tic, 2)}
    return adata if copy else None
