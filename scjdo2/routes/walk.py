"""Walkers that hop between measured cells, steered by a drift.

Each cell links to its ``k`` nearest cells in the latent space; a hop from
i to neighbour j has probability ∝ exp(cos(x_j − x_i, v_i) / σ), the
velocity kernel CellRank uses. Walkers cannot leave the data, unlike walkers
integrated freely in the latent space (which drifted off the data on LARRY
and scored below a proximity baseline) or rolling down a 2-D landscape
(which a 2-D layout can trap, e.g. iPSC reprogramming sent to Trophoblast).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def drift_kernel(X, V, k=30, sigma=0.25):
    """(neighbour index, cumulative hop probabilities), both n × k. σ = ∞ gives an undirected walk."""
    X = np.asarray(X, float)
    V = np.asarray(V, float)
    nb = cKDTree(X).query(X, k=k + 1)[1][:, 1:]
    if np.isfinite(sigma):
        D = X[nb] - X[:, None, :]
        cos = (D * V[:, None, :]).sum(-1) / (np.linalg.norm(D, axis=-1) * np.linalg.norm(V, axis=1)[:, None] + 1e-12)
        w = np.exp((cos - cos.max(1, keepdims=True)) / sigma)
    else:
        w = np.ones(nb.shape)
    w /= w.sum(1, keepdims=True)
    return nb, np.cumsum(w, 1)


def _hop(cell, nb, cw, rng, move):
    r = rng.random(len(cell))[:, None]
    j = (r > cw[cell]).sum(1).clip(max=nb.shape[1] - 1)
    return np.where(move, nb[cell, j], cell)


def cell_walk(nb, cw, start_idx, n_walkers=300, n_frames=360, hops=150, seed=0):
    """Cell index of every walker at every frame, (n_frames + 1) × n_walkers.

    Each frame a walker hops with probability hops / n_frames.
    """
    rng = np.random.default_rng(seed)
    cell = rng.choice(np.asarray(start_idx), n_walkers)
    p_hop = min(1.0, hops / n_frames)
    out = [cell.copy()]
    for _ in range(n_frames):
        cell = _hop(cell, nb, cw, rng, rng.random(len(cell)) < p_hop)
        out.append(cell.copy())
    return np.stack(out)


def smooth_positions(cells, floor, ease=0.35, jitter=0.01, seed=0):
    """Display positions on a 2-D floor: each frame eases toward the current cell."""
    rng = np.random.default_rng(seed)
    pos = floor[cells[0]].astype(float).copy()
    out = []
    for c in cells:
        pos = pos + ease * (floor[c] - pos)
        out.append(pos + jitter * rng.standard_normal(pos.shape))
    return np.stack(out)


def absorption(X, V, absorbing_label, k=30, sigma=0.25, iters=600):
    """Absorption probabilities of the drift-kernel walk.

    ``absorbing_label``: per cell, the fate index of absorbing cells, −1 for
    transient cells. Returns n × n_fates (rows of absorbing cells are one-hot).
    """
    nb, cw = drift_kernel(X, V, k, sigma)
    w = np.diff(np.concatenate([np.zeros((len(cw), 1)), cw], 1), axis=1)
    n = len(X)
    P = sp.csr_matrix((w.ravel(), nb.ravel(), np.arange(0, nb.size + 1, nb.shape[1])), shape=(n, n))
    lab = np.asarray(absorbing_label)
    n_f = int(lab.max()) + 1
    Y = np.zeros((n, n_f))
    ab = lab >= 0
    Y[ab, lab[ab]] = 1.0
    tr = ~ab
    Ptr = P[tr]
    for _ in range(iters):
        Y[tr] = Ptr @ Y
    return Y
