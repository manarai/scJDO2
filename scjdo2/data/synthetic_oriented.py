"""Synthetic data with known orientation, for the orientation-layer tests
(docs/ORIENTATION_SPEC.md §8).

Both generators use an analytic energy ``E`` so the operator is exact,
``J = −Hess E`` (symmetric), and the score is ``f = −∇E``. Positions are
rotated by a random orthogonal matrix so axes are not coordinate-aligned.

* :func:`make_oriented_pitchfork` — cells on clone trajectories that leave a
  progenitor state along a pitchfork, sampled on three days; known
  developmental velocity, clones, days, pseudotime and a start cell.
* :func:`make_rotating_two_well` — cells in a two-well density with a
  rotating flow ``v = −∇E + ω S w``; the known rotation is ``ω S``.

Measured inputs are written under the keys the orientation sources read:
``obsm["velocity_labelling"]`` (per hour), ``obsm["velocity_lat"]``
(scVelo-style, for ``embedding_key="X_lat"``), ``obs["palantir_pseudotime"]``
+ ``uns["palantir_start_cell"]``, ``obs["clone"]``, ``obs["day"]``,
``obsm["X_drift_v"]`` (score) and ``obs["energy"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def _rotation(d: int, rng) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def _noisy(v: np.ndarray, rel: float, rng) -> np.ndarray:
    scale = rel * np.linalg.norm(v, axis=1, keepdims=True) / np.sqrt(v.shape[1])
    return v + scale * rng.standard_normal(v.shape)


def _finish(w, vw, Hw, fw, E, R, obs, rng, label_noise, velo_noise, velo_scale):
    z = w @ R.T
    v = vw @ R.T
    J = -np.einsum("ij,njk,lk->nil", R, Hw, R)
    a = AnnData(np.zeros((len(z), 1), dtype=np.float32), obs=obs)
    a.obs_names = [f"c{i}" for i in range(len(z))]
    a.obsm["X_lat"] = z
    a.obsm["velocity_true"] = v
    a.obsm["velocity_labelling"] = _noisy(v, label_noise, rng)
    a.obsm["velocity_lat"] = velo_scale * _noisy(v, velo_noise, rng)
    a.obsm["X_drift_v"] = fw @ R.T
    a.obs["energy"] = E
    a.uns["scjdo2"] = {"J": J}
    a.uns["synthetic"] = {"R": R}
    return a


def make_oriented_pitchfork(n_clones: int = 160, cells_per_day: int = 4,
                            days=(2.0, 4.0, 6.0), d: int = 5, clone_coverage: float = 1.0,
                            label_noise: float = 0.1, velo_noise: float = 0.3,
                            seed: int = 0) -> AnnData:
    """Pitchfork in ``w0`` (wells at ±1), a slow drift along ``w1``, and
    decaying stable directions ``w2…``. Energy
    ``E = w0⁴/4 − w0²/2 + w1² + Σ c_k w_k²/2``; developmental velocity
    ``v = (w0(1 − w0²), 0.3, −0.5 w_k)``. Clones are trajectories from
    the progenitor state; each is sampled ``cells_per_day`` times per day,
    at continuous times within ± half the day spacing of the collection day.
    ``clone_coverage`` keeps clone labels on that fraction of clones."""
    rng = np.random.default_rng(seed)
    c = np.arange(3.0, 3.0 + d - 2)
    dt, T = 0.02, max(days)

    def vel(w):
        out = np.empty_like(w)
        out[:, 0] = w[:, 0] * (1 - w[:, 0] ** 2)
        out[:, 1] = 0.3
        out[:, 2:] = -0.5 * w[:, 2:]
        return out

    w = np.zeros((n_clones, d))
    w[:, 0] = rng.choice([-1, 1], n_clones) * np.abs(rng.normal(0.08, 0.03, n_clones))
    w[:, 1] = rng.normal(0, 0.1, n_clones)
    w[:, 2:] = rng.normal(0, 0.6, (n_clones, d - 2))
    # Integrate every clone's trajectory and keep each step, so cells can be
    # sampled at continuous times: real samples hold a continuum of states
    # around their collection day (three exact snapshots would leave gaps in
    # state space that no local pseudotime gradient can bridge).
    n_steps = int(round(T / dt)) + 1
    traj = np.empty((n_steps, n_clones, d))
    for step in range(n_steps):
        traj[step] = w
        w = w + dt * vel(w) + np.sqrt(dt) * 0.02 * rng.standard_normal(w.shape)
    half = 0.5 * float(np.min(np.diff(days))) if len(days) > 1 else 1.0
    pos, clone, day_col, t_col = [], [], [], []
    for day in days:
        cl = np.repeat(np.arange(n_clones), cells_per_day)
        tt = np.clip(day + rng.uniform(-half, half, cl.size), dt, T)
        steps = np.minimum(np.round(tt / dt).astype(int), n_steps - 1)
        base = traj[steps, cl]
        pos.append(base + rng.normal(0, 0.03, base.shape))
        clone += list(cl)
        day_col += [day] * cl.size
        t_col += list(tt)
    w = np.vstack(pos)
    clone = np.asarray(clone)
    day_col = np.asarray(day_col)
    t_true = np.asarray(t_col)
    vw = vel(w)
    Hw = np.zeros((len(w), d, d))
    Hw[:, 0, 0] = 3 * w[:, 0] ** 2 - 1
    Hw[:, 1, 1] = 2.0
    for k in range(2, d):
        Hw[:, k, k] = c[k - 2]
    fw = -np.column_stack([w[:, 0] ** 3 - w[:, 0], 2 * w[:, 1], c * w[:, 2:]])
    E = w[:, 0] ** 4 / 4 - w[:, 0] ** 2 / 2 + w[:, 1] ** 2 + 0.5 * np.sum(c * w[:, 2:] ** 2, axis=1)
    keep = rng.random(n_clones) < clone_coverage
    clone_lab = pd.array([str(cl) if keep[cl] else None for cl in clone], dtype=object)
    pt = t_true + rng.normal(0, 0.15, len(w))
    obs = pd.DataFrame({"clone": clone_lab, "day": day_col, "palantir_pseudotime": pt,
                        "clone_true": clone.astype(str)})
    obs.index = [f"c{i}" for i in range(len(w))]
    R = _rotation(d, rng)
    a = _finish(w, vw, Hw, fw, E, R, obs, rng, label_noise, velo_noise, velo_scale=3.0)
    a.obs["clone_covered"] = np.asarray([keep[cl] for cl in clone])
    a.uns["palantir_start_cell"] = a.obs_names[int(np.argmin(pt))]
    return a


def make_rotating_two_well(n: int = 2000, d: int = 4, omega: float = 3.0,
                           label_noise: float = 0.1, seed: int = 0) -> AnnData:
    """Two wells in ``w0`` (``E = (w0² − 1)² + w1²/2 + Σ c_k w_k²/2``); flow
    ``v = −∇E + ω S w`` with ``S`` the 90° rotation in the (w0, w1) plane, so
    the antisymmetric part of the velocity Jacobian is exactly ``ω S``.
    ``omega = 0`` gives a pure gradient (equilibrium) system."""
    rng = np.random.default_rng(seed)
    c = np.arange(2.0, 2.0 + d - 2)
    w = np.zeros((n, d))
    w[:, 0] = rng.choice([-1, 1], n) * rng.normal(1.0, 0.25, n)
    w[:, 1] = rng.normal(0, 1.0, n)
    w[:, 2:] = rng.normal(0, 1 / np.sqrt(c), (n, d - 2))
    grad = np.column_stack([4 * w[:, 0] * (w[:, 0] ** 2 - 1), w[:, 1], c * w[:, 2:]])
    S = np.zeros((d, d))
    S[0, 1], S[1, 0] = -1.0, 1.0
    vw = -grad + omega * (w @ S.T)
    Hw = np.zeros((n, d, d))
    Hw[:, 0, 0] = 12 * w[:, 0] ** 2 - 4
    Hw[:, 1, 1] = 1.0
    for k in range(2, d):
        Hw[:, k, k] = c[k - 2]
    E = (w[:, 0] ** 2 - 1) ** 2 + 0.5 * w[:, 1] ** 2 + 0.5 * np.sum(c * w[:, 2:] ** 2, axis=1)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    R = _rotation(d, rng)
    a = _finish(w, vw, Hw, -grad, E, R, obs, rng, label_noise, 0.3, velo_scale=1.0)
    a.uns["synthetic"]["omega_true"] = omega * (R @ S @ R.T)
    return a
