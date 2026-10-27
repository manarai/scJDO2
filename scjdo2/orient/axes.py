"""§2 Axes: per-cell eigenvectors of the symmetric operator, a fixed sign
convention, and the eigen-gap rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Axes:
    eigvals: np.ndarray   # (n, k), largest first (axis 1 = softest)
    U: np.ndarray         # (n, d, k), columns = eigenvectors
    usable: np.ndarray    # (n, k) bool, passes the gap rule
    rel_gap: np.ndarray   # (n, k)


def compute_axes(J: np.ndarray, n_axes: int, gap_min: float) -> Axes:
    S = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    w, V = np.linalg.eigh(S)                    # ascending
    w, V = w[:, ::-1], V[:, :, ::-1]            # largest first
    d = w.shape[1]
    k = min(n_axes, d)
    # Sign convention: largest-magnitude component of each eigenvector positive.
    lead = np.argmax(np.abs(V), axis=1)                        # (n, d)
    s = np.sign(np.take_along_axis(V, lead[:, None, :], axis=1))[:, 0, :]
    s[s == 0] = 1.0
    V = V * s[:, None, :]
    # Gap rule: relative gap to the nearest neighbouring eigenvalue.
    gaps = np.full((w.shape[0], d), np.inf)
    diff = np.abs(np.diff(w, axis=1))
    gaps[:, :-1] = np.minimum(gaps[:, :-1], diff)
    gaps[:, 1:] = np.minimum(gaps[:, 1:], diff)
    scale = np.maximum(np.median(np.abs(w), axis=1), 1e-300)[:, None]
    rel = gaps / scale
    return Axes(eigvals=w[:, :k], U=V[:, :, :k], usable=(rel >= gap_min)[:, :k], rel_gap=rel[:, :k])
