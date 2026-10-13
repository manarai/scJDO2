"""Clone arm — the direction-free lineage signal from LARRY barcodes.

Step 2's drift adapter is fit on an unbarcoded dataset and never
consumes a root cell. The clone arm turns clone-barcoded sampling-day
data into two measurements of the resulting drift:

* :func:`clone_split_map` — for each clone whose members span two
  annotated fates, the latent position where the clone's cells sit
  before the split (direction-free; see V2_PLAN R17).
* :func:`clone_consistency` — the fraction of cross-day clone pairs
  whose later-day member is reached by integrating ``f`` forward from
  the earlier-day member.

Both quantities depend only on physical orderings (clone nesting +
sampling-day). Neither needs a root cell, an annotation-based arrow,
or a trained velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


# ------------------------------------------------------------------ clone pairs


def clone_cross_day_pairs(
    adata: AnnData,
    *,
    day_key: str = "day",
    clone_key: str = "clone_matrix",
) -> np.ndarray:
    """Return ``(c_early, c_late)`` integer index pairs where the two
    cells share a clone barcode and sit on different sampling days.

    Earlier day = smaller ``day``. Pairs within the same day are
    skipped. Direction-free: the pair is a physical ordering (clone
    nesting + sampling-day), not an annotation-driven arrow.
    """
    cm = adata.obsm.get(clone_key)
    if cm is None:
        raise KeyError(f"adata.obsm[{clone_key!r}] missing")
    if not sp.issparse(cm):
        cm = sp.csr_matrix(cm)
    cm_t = cm.T.tocsr()
    day = adata.obs[day_key].to_numpy()
    pairs: list[tuple[int, int]] = []
    for k in range(cm_t.shape[0]):
        members = cm_t.indices[cm_t.indptr[k]:cm_t.indptr[k + 1]]
        if members.size < 2:
            continue
        d = day[members]
        for i in range(members.size):
            di = d[i]
            if pd.isna(di):
                continue
            for j in range(members.size):
                if i == j:
                    continue
                dj = d[j]
                if pd.isna(dj) or di >= dj:
                    continue
                pairs.append((int(members[i]), int(members[j])))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


# -------------------------------------------------------------- clone-split map


@dataclass
class CloneSplit:
    """A single clone's bifurcation descriptor on the latent manifold."""
    clone_id: int
    fates: tuple[str, str]
    counts: tuple[int, int]
    pre_split_mean: np.ndarray          # (d,) mean of pre-split members
    pre_split_members: np.ndarray       # cell indices (early day)
    post_split_members_a: np.ndarray
    post_split_members_b: np.ndarray


def clone_split_map(
    adata: AnnData,
    *,
    annotation_key: str = "cell_type_annotation",
    use_rep: str = "X_pca",
    day_key: str = "day",
    min_split_count: int = 1,
) -> list[CloneSplit]:
    """For each clone whose members span two distinct annotated fates on
    later days, return the mean latent coordinate of the clone's
    pre-split (earliest-day) members.

    Direction-free: the "pre-split" day is the minimum ``day`` label in
    the clone; "post-split" days are strictly greater. No root cell is
    used.

    Parameters
    ----------
    annotation_key
        obs column with Weinreb's cell-type annotation.
    use_rep
        obsm key for the latent embedding where the split is reported.
    day_key
        obs column with sampling-day labels (physical time ordering).
    min_split_count
        Minimum number of cells of each fate required to count the
        clone as a split event.
    """
    cm = adata.obsm.get("clone_matrix")
    if cm is None:
        raise KeyError("adata.obsm['clone_matrix'] missing")
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] missing")
    if not sp.issparse(cm):
        cm = sp.csr_matrix(cm)
    cm_t = cm.T.tocsr()
    day = adata.obs[day_key].to_numpy()
    fate = adata.obs[annotation_key].astype(str).to_numpy()
    z = np.asarray(adata.obsm[use_rep])

    out: list[CloneSplit] = []
    for k in range(cm_t.shape[0]):
        members = cm_t.indices[cm_t.indptr[k]:cm_t.indptr[k + 1]]
        if members.size < 2:
            continue
        d = day[members]
        finite = ~pd.isna(d)
        if finite.sum() < 2:
            continue
        d_arr = d[finite].astype(float)
        min_day = float(d_arr.min())
        if (d_arr > min_day).sum() == 0:
            continue
        pre_mask = finite & (d == min_day)
        post_mask = finite & (d > min_day)
        pre_idx = members[pre_mask]
        post_idx = members[post_mask]
        if pre_idx.size == 0 or post_idx.size < 2:
            continue
        post_fates = fate[post_idx]
        post_fates = post_fates[post_fates != "nan"]
        if post_fates.size < 2:
            continue
        counts = pd.Series(fate[post_idx]).value_counts()
        counts = counts[counts.index != "nan"]
        if len(counts) < 2:
            continue
        top = counts.head(2)
        if int(top.iloc[1]) < min_split_count:
            continue
        fa, fb = str(top.index[0]), str(top.index[1])
        a_mask = fate[post_idx] == fa
        b_mask = fate[post_idx] == fb
        out.append(CloneSplit(
            clone_id=int(k),
            fates=(fa, fb),
            counts=(int(top.iloc[0]), int(top.iloc[1])),
            pre_split_mean=np.asarray(z[pre_idx].mean(axis=0), dtype=np.float64),
            pre_split_members=pre_idx.astype(np.int64),
            post_split_members_a=post_idx[a_mask].astype(np.int64),
            post_split_members_b=post_idx[b_mask].astype(np.int64),
        ))
    return out


# ----------------------------------------------------- clone-consistency score


def _integrate_forward(
    f_fn, z0: np.ndarray, t0: float, t1: float, *, n_steps: int = 20,
) -> np.ndarray:
    """Fourth-order Runge–Kutta integration of ``dz/dt = f(z, t)``.

    ``f_fn(z, t)`` accepts ``(z: (d,), t: float) → (d,)`` or batched
    ``(z: (n, d), t: float) → (n, d)``. We vectorise over batch when
    ``z0`` is 2-D.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    squeeze = z0.ndim == 1
    if squeeze:
        z0 = z0[None, :]
    dt = (t1 - t0) / float(max(1, n_steps))
    z = z0.copy()
    t = float(t0)
    for _ in range(int(n_steps)):
        k1 = np.asarray(f_fn(z, t))
        k2 = np.asarray(f_fn(z + 0.5 * dt * k1, t + 0.5 * dt))
        k3 = np.asarray(f_fn(z + 0.5 * dt * k2, t + 0.5 * dt))
        k4 = np.asarray(f_fn(z + dt * k3, t + dt))
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
    return z[0] if squeeze else z


def clone_consistency(
    adata: AnnData,
    drift,
    *,
    use_rep: str = "X_pca",
    day_key: str = "day",
    time_key: Optional[str] = "pseudotime",
    radius: Optional[float] = None,
    k_nearest: int = 5,
    n_rk_steps: int = 20,
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Fraction of cross-day clone pairs where integrating the drift
    forward from the earlier-day member lands within ``radius`` of the
    later-day member in latent space.

    The score is **direction-free in the behaviour sense**: it tests
    whether the fitted ``drift`` is a consistent flow on the latent
    manifold *given* the clone-nesting + sampling-day ordering — it is
    not a classifier output. Returned keys:

    * ``n_pairs`` : number of evaluated clone pairs.
    * ``n_consistent`` : how many landed within ``radius``.
    * ``consistency`` : ``n_consistent / n_pairs``.
    * ``residuals`` : per-pair distance from the integrated endpoint
      to the later-day target in latent space (``(n_pairs,)``).
    * ``radius`` : the acceptance threshold actually used.
    """
    pairs = clone_cross_day_pairs(adata, day_key=day_key)
    if pairs.size == 0:
        return {"n_pairs": 0, "n_consistent": 0, "consistency": float("nan"),
                "residuals": np.zeros(0), "radius": float("nan")}
    rng = np.random.default_rng(int(seed))
    if max_pairs is not None and pairs.shape[0] > int(max_pairs):
        sel = rng.choice(pairs.shape[0], size=int(max_pairs), replace=False)
        pairs = pairs[sel]
    z = np.asarray(adata.obsm[use_rep])
    day = adata.obs[day_key].astype(float).to_numpy()
    # Use pseudotime as the time coordinate if present and finite;
    # otherwise map day → float directly as the time axis (the clock is
    # a check per V2_PLAN R18, not a signal — this just gives the drift
    # a monotone t).
    if time_key is not None and time_key in adata.obs:
        t_series = adata.obs[time_key].to_numpy()
        if not np.isfinite(np.asarray(t_series, dtype=float)).any():
            t_series = day.copy()
    else:
        t_series = day.copy()
    t_series = np.asarray(t_series, dtype=np.float64)

    # Radius: median kNN distance across all cells at the earlier-day
    # end of each pair, so the acceptance threshold scales with local
    # density. ``k_nearest`` tunes locality.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_nearest + 1).fit(z)
    _, _ = nn.kneighbors(z[pairs[:, 1]][:1], return_distance=True)   # warm-up
    rad = radius
    if rad is None or not np.isfinite(rad) or rad <= 0:
        d_arr, _ = nn.kneighbors(z, return_distance=True)
        rad = float(np.median(d_arr[:, -1]))

    def f_batch(zb: np.ndarray, t: float) -> np.ndarray:
        tv = np.full(zb.shape[0], t, dtype=np.float64)
        return drift.velocity(zb, tv)

    z0 = z[pairs[:, 0]]
    z1 = z[pairs[:, 1]]
    t0 = t_series[pairs[:, 0]]
    t1 = t_series[pairs[:, 1]]
    pred = np.empty_like(z0)
    for i in range(z0.shape[0]):
        pred[i] = _integrate_forward(f_batch, z0[i:i + 1], float(t0[i]),
                                     float(t1[i]), n_steps=n_rk_steps)
    residuals = np.linalg.norm(pred - z1, axis=1)
    hits = int((residuals <= rad).sum())
    return {
        "n_pairs": int(pairs.shape[0]),
        "n_consistent": hits,
        "consistency": float(hits) / float(pairs.shape[0]),
        "residuals": residuals,
        "radius": float(rad),
        "pairs": pairs,
    }
