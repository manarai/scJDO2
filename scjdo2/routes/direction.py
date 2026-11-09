"""Where a cell's direction of travel comes from.

Every source is turned into a velocity in the latent space (per unit of the
time key when there is one), so that one ``MeasuredDrift`` can be fitted to
it. In order of preference (``choose_source``):

``clone``      lineage barcodes + time points: (late − early clone centroid)
               / Δtime, given to every cell of the clone. Not a gradient
               field, so its Jacobian can be directional. On LARRY held-out
               clones a cell walk on this drift matched a supervised fate
               classifier.
``velocity``   a measured velocity already in the latent space (e.g. from
               metabolic labelling, ``obsm['velocity_pca']``).
``ot``         time points without lineage: entropic optimal transport
               between consecutive time points, barycentric displacement /
               Δtime. Near-gradient (Brenier), so its Jacobian is mostly
               symmetric.
``pseudotime`` a snapshot: the local gradient of pseudotime. A gradient by
               construction; the weakest source, gene roles are mostly hubs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

SOURCES = ("clone", "velocity", "ot", "pseudotime")


def clone_labels(adata, clone_key: str) -> np.ndarray:
    """One integer clone id per cell (−1 = none) from ``obs`` or a one-hot ``obsm``."""
    if clone_key in adata.obs:
        s = adata.obs[clone_key]
        codes = pd.Categorical(s.where(s.notna() & (s.astype(str) != "-1"))).codes
        return np.asarray(codes, dtype=int)
    cm = sp.csr_matrix(adata.obsm[clone_key])
    lab = np.full(adata.n_obs, -1)
    has = np.diff(cm.indptr) > 0
    lab[has] = [cm.indices[cm.indptr[i]] for i in np.flatnonzero(has)]
    return lab


def choose_source(adata, clone_key=None, time_key=None, velocity_key=None, pseudotime_key=None) -> str:
    """Strongest available direction source (see module docstring)."""
    if clone_key and time_key and (clone_key in adata.obs or clone_key in adata.obsm):
        return "clone"
    if velocity_key and velocity_key in adata.obsm:
        return "velocity"
    if time_key and time_key in adata.obs and adata.obs[time_key].nunique() >= 2:
        return "ot"
    if pseudotime_key and pseudotime_key in adata.obs:
        return "pseudotime"
    raise ValueError("no direction source: give clone_key + time_key, velocity_key, "
                     "time_key (≥ 2 time points) or pseudotime_key")


def clone_velocity(Z, time, clone, clones=None, min_per_end=2):
    """(late − early centroid) / Δtime per clone; NaN for cells without one.

    ``clones`` restricts to those clone ids (used to fit on training clones only).
    """
    V = np.full_like(Z, np.nan, dtype=float)
    df = pd.DataFrame({"c": clone, "t": time})
    df = df[df.c >= 0]
    if clones is not None:
        df = df[df.c.isin(set(clones))]
    for _, g in df.groupby("c"):
        ts = np.unique(g.t)
        if len(ts) < 2:
            continue
        idx = g.index.to_numpy()
        e, l = idx[g.t.to_numpy() == ts.min()], idx[g.t.to_numpy() == ts.max()]
        if min(len(e), len(l)) < min_per_end:
            continue
        V[idx] = (Z[l].mean(0) - Z[e].mean(0)) / (ts.max() - ts.min())
    return V


def sinkhorn_log(C, eps, n_iter=2000, tol=1e-7):
    """Balanced entropic OT plan between uniform marginals, log domain."""
    from scipy.special import logsumexp
    n, m = C.shape
    la, lb = -np.log(n) * np.ones(n), -np.log(m) * np.ones(m)
    f, g = np.zeros(n), np.zeros(m)
    K = -C / eps
    for _ in range(n_iter):
        f_old = f
        f = la - logsumexp(K + g[None, :], axis=1)
        g = lb - logsumexp(K + f[:, None], axis=0)
        if np.max(np.abs(f - f_old)) < tol:
            break
    return np.exp(K + f[:, None] + g[None, :])


def ot_velocity(Z, time, eps=0.05, max_per_time=4000, seed=0):
    """Displacement / Δtime from OT between consecutive time points.

    At most ``max_per_time`` cells per time point enter each plan; other cells
    take the mean velocity of their 10 nearest planned cells of the same time.
    Cells on the last time point use the backward image.
    """
    rng = np.random.default_rng(seed)
    ts = np.unique(time)
    V = np.full_like(Z, np.nan, dtype=float)
    pick = {}
    for t in ts:
        idx = np.flatnonzero(time == t)
        pick[t] = idx if len(idx) <= max_per_time else np.sort(rng.choice(idx, max_per_time, replace=False))
    for i in range(len(ts) - 1):
        a, b = pick[ts[i]], pick[ts[i + 1]]
        C = ((Z[a][:, None, :] - Z[b][None, :, :]) ** 2).sum(-1)
        T = sinkhorn_log(C / np.median(C), eps)
        dt = ts[i + 1] - ts[i]
        V[a] = (T @ Z[b] / T.sum(1, keepdims=True) - Z[a]) / dt
        if i == len(ts) - 2:
            V[b] = (Z[b] - T.T @ Z[a] / T.sum(0)[:, None]) / dt
    for t in ts:
        idx = np.flatnonzero(time == t)
        done = idx[np.isfinite(V[idx, 0])]
        miss = idx[~np.isfinite(V[idx, 0])]
        if len(miss) and len(done):
            nb = cKDTree(Z[done]).query(Z[miss], k=min(10, len(done)))[1]
            V[miss] = V[done][np.atleast_2d(nb)].mean(1)
    return V


def gradient_velocity(Z, scalar, k=30):
    """Local least-squares gradient of a scalar (pseudotime) over each cell's k neighbours."""
    nb = cKDTree(Z).query(Z, k=k + 1)[1][:, 1:]
    D = Z[nb] - Z[:, None, :]
    dy = scalar[nb] - scalar[:, None]
    return np.stack([np.linalg.lstsq(D[i], dy[i], rcond=None)[0] for i in range(len(Z))])


def alignment(Z, V, scalar, k=30) -> float:
    """Median cosine between a velocity and the local gradient of a scalar (time, pseudotime)."""
    g = gradient_velocity(Z, np.asarray(scalar, float), k)
    cos = (g * V).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(V, axis=1) + 1e-12)
    return float(np.median(cos))
