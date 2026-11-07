"""Operator motifs along the time course: archetypes of ``J_flow`` chained by the OT plan.

:func:`scjdo2.flow` gives every cell an operator ``J_flow`` but, on its own, no order:
the operators of a day-2 cell and of the day-4 cells it becomes are not linked.
The link is the entropic transport plan between consecutive sampling times, which
:func:`scjdo2.flow` keeps in ``obsp["flow_plan"]`` (``keep_plan``). This module puts
the two together.

1. :func:`flow_archetypes` -- a small set of operator shapes (semi-NMF of ``J_flow``,
   :mod:`scjdo2.archetypes`) and an archetype label per cell.
2. :func:`flow_transitions` -- **the default motif readout.** For each pair of
   consecutive times, the label x label matrix of transported mass
   ``T[k, l] = sum_{i in k} sum_{j in l} P(j | i)``, the plan contracted with the
   labels. Exact: the transport is re-run with the stored settings (deterministic)
   and contracted with the labels as it goes, so nothing is truncated. Tested
   against a null that shuffles the source labels within the source time.
3. :func:`flow_paths` -- label sequences of chains sampled through the plan, each
   compared with the frequency the per-pair matrices predict for a Markov chain.
   A robustness check: it shows rare paths and memory the matrices cannot. Chains
   step through the stored plan, which keeps each cell's most likely destinations
   only (``keep_plan``; kept mass in ``uns["flow"]["ot_pairs"]``), so paths favour near
   destinations; the Markov expectation uses the same stored plan, so
   ``obs_over_expected`` compares like with like.
4. :func:`flow_integrate` -- integrate the fitted drift from chosen start points,
   with the archetype profile along the path. A continuous picture; a deterministic
   path cannot branch, so it is not a test.

Any categorical ``obs`` column can replace the archetype labels (e.g. cell types),
which makes (2) and (3) a label-free-operator x annotation readout as well.

Caveats. Cells of the last sampling time have no OT velocity; their operator is the
drift extrapolated there (``obs["flow_has_velocity"]`` is False), so their labels are
less certain, and transitions into the last time are flagged. The plan depends on
the transport space and ``eps`` exactly as the velocities do. Balanced transport
assumes no large proliferation or death across a time step.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["flow_archetypes", "flow_transitions", "flow_paths", "flow_integrate",
           "archetype_eigenvalues"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flow_rec(adata, key: str) -> dict[str, Any]:
    if key not in adata.uns or "params" not in adata.uns[key]:
        raise KeyError(f"uns[{key!r}] missing: run scjdo2.flow first")
    return adata.uns[key]


def _plan(adata, key: str):
    name = f"{key}_plan"
    if name not in adata.obsp:
        raise KeyError(f"obsp[{name!r}] missing: run scjdo2.flow with keep_plan (default 50)")
    from scipy import sparse
    return sparse.csr_matrix(adata.obsp[name])


def _labels(adata, key: str, labels: Optional[str]) -> tuple[str, np.ndarray, list[str]]:
    col = labels or f"{key}_archetype"
    if col not in adata.obs:
        hint = " (run scjdo2.flow_archetypes first)" if labels is None else ""
        raise KeyError(f"obs[{col!r}] missing{hint}")
    s = adata.obs[col]
    levels = list(s.cat.categories) if hasattr(s, "cat") else sorted(map(str, s.unique()))
    lab = s.astype(str).to_numpy()
    present = set(lab)
    levels = [str(x) for x in levels if str(x) in present]
    index = {x: i for i, x in enumerate(levels)}
    return col, np.array([index[x] for x in lab], np.int64), levels


def _times(adata, rec: dict[str, Any]) -> np.ndarray:
    return np.asarray(adata.obs[rec["params"]["time_key"]], np.float64)


def _normalise(J: np.ndarray, how: Optional[str]) -> np.ndarray:
    J = np.asarray(J, np.float64)
    if how is None:
        return J
    if how != "frobenius":
        raise ValueError(f"normalize must be 'frobenius' or None; got {how!r}")
    nrm = np.linalg.norm(J.reshape(len(J), -1), axis=1)
    return J / np.maximum(nrm, 1e-300)[:, None, None]


def archetype_eigenvalues(patterns: np.ndarray) -> np.ndarray:
    """Eigenvalues of each archetype matrix (K, d), sorted by real part, largest first."""
    eig = np.linalg.eigvals(np.asarray(patterns, np.float64))
    return np.take_along_axis(eig, np.argsort(-eig.real, axis=1, kind="stable"), axis=1)


# ---------------------------------------------------------------------------
# 1. archetypes
# ---------------------------------------------------------------------------
def flow_archetypes(adata, *, key: str = "flow", n_archetypes: int | str = "auto",
                    k_candidates: Sequence[int] = (2, 3, 4, 5, 6, 8), normalize: Optional[str] = "frobenius",
                    max_fit_cells: Optional[int] = 4000, n_restarts: int = 3, seed: int = 0):
    """Operator archetypes of ``J_flow`` and an archetype label per cell.

    Semi-NMF ``J_c ~ sum_k W[c, k] H_k`` (``W >= 0``, ``H_k`` signed; see
    :class:`scjdo2.archetypes.ArchetypeDecomposer`), ``K`` by held-out Frobenius error
    when ``n_archetypes="auto"``.

    Parameters
    ----------
    normalize : ``"frobenius"`` (default) divides each cell's operator by its Frobenius
        norm first, so archetypes are operator *shapes* and the arbitrary clock of the
        time course does not matter; ``None`` keeps magnitudes.
    max_fit_cells : fit on at most this many cells (random, ``seed``), then assign every
        cell by non-negative least squares on the fitted archetypes. ``None`` fits on all.

    Writes
    ------
    ``obsm[f"{key}_archetype_weights"]`` (n, K) non-negative weights
    ``obs[f"{key}_archetype"]``          label ``A1..AK`` of the largest weight; A1 is the
                                         most common
    ``obs[f"{key}_archetype_share"]``    largest weight / sum of weights (1 = pure)
    ``uns[key]["archetypes"]``           patterns (K, d, d), their eigenvalues (real and
                                         imaginary parts), K curve, settings

    Returns the patterns (K, d, d).
    """
    import pandas as pd

    from .archetypes import ArchetypeConfig, ArchetypeDecomposer, _nnls_rows

    rec = _flow_rec(adata, key)
    J = adata.uns.get("scjdo2", {}).get("J")
    if J is None or adata.uns["scjdo2"].get("meta", {}).get("operator") != "flow":
        raise KeyError('uns["scjdo2"]["J"] does not hold a flow operator: run scjdo2.flow first')
    Jn = _normalise(J, normalize)
    n, d, _ = Jn.shape
    rng = np.random.default_rng(seed)
    fit_idx = np.arange(n) if (max_fit_cells is None or n <= max_fit_cells) else \
        np.sort(rng.choice(n, int(max_fit_cells), replace=False))
    res = ArchetypeDecomposer(ArchetypeConfig(n_archetypes=n_archetypes, k_candidates=tuple(k_candidates),
                                              n_restarts=int(n_restarts), seed=int(seed))).fit(Jn[fit_idx])
    H = res.patterns.reshape(res.k_selected, d * d)
    W = _nnls_rows(H, Jn.reshape(n, d * d).astype(np.float32)).astype(np.float64)

    # order archetypes by how many cells they label, most first
    hard = W.argmax(axis=1)
    order = np.argsort(-np.bincount(hard, minlength=res.k_selected), kind="stable")
    W, patterns = W[:, order], res.patterns[order]
    hard = W.argmax(axis=1)
    names = [f"A{k + 1}" for k in range(len(order))]
    adata.obsm[f"{key}_archetype_weights"] = W.astype(np.float32)
    adata.obs[f"{key}_archetype"] = pd.Categorical([names[k] for k in hard], categories=names)
    tot = W.sum(axis=1)
    adata.obs[f"{key}_archetype_share"] = np.where(tot > 0, W.max(axis=1) / np.maximum(tot, 1e-300),
                                                   np.nan).astype(np.float32)
    eig = archetype_eigenvalues(patterns)
    if n_archetypes == "auto" and res.k_selected == max(k_candidates):
        import warnings
        warnings.warn(f"K = {res.k_selected} is the largest candidate (held-out error still falling); "
                      "pass larger k_candidates or fix n_archetypes", stacklevel=2)
    rec["archetypes"] = {
        "names": names, "patterns": patterns.astype(np.float32),
        "eig_real": eig.real.astype(np.float32), "eig_imag": eig.imag.astype(np.float32),
        "k_selected": int(res.k_selected),
        "k_curve": {"k": [int(k) for k, _ in res.k_curve], "heldout_frobenius": [float(e) for _, e in res.k_curve]},
        "normalize": normalize or "none", "n_fit_cells": int(len(fit_idx)), "seed": int(seed),
        "frobenius_error_fit": float(res.frobenius_error)}
    return patterns


# ---------------------------------------------------------------------------
# 2. transition matrices
# ---------------------------------------------------------------------------
def _destination_mix(adata, key: str, rec: dict[str, Any], onehot: np.ndarray, exact: bool) -> np.ndarray:
    """(n, L) label mix of each cell's destinations at the next time (NaN on the last)."""
    if not exact:
        P = _plan(adata, key)
        Q = np.asarray(P @ onehot)
        Q[np.diff(P.indptr) == 0] = np.nan
        return Q
    from .flow_operator import ot_velocity
    p = rec["params"]
    z = np.asarray(adata.obsm[p["use_rep"]], np.float64)
    if p.get("n_comps") is not None:
        z = z[:, :int(p["n_comps"])]
    v, _, Q = ot_velocity(z, _times(adata, rec), eps=float(p["eps"]), solver=p["solver"], seed=int(p["seed"]),
                          plan_apply=onehot)
    stored = f"{key}_velocity"
    if stored in adata.obsm:
        ok = np.isfinite(v).all(axis=1)
        ref = np.asarray(adata.obsm[stored], np.float64)
        scale = max(float(np.abs(ref[ok]).max()), 1e-12)
        if not np.allclose(v[ok], ref[ok], atol=1e-3 * scale, rtol=0):
            raise RuntimeError(f"re-run transport does not reproduce obsm[{stored!r}]: the cells, "
                               "transport space or settings changed since scjdo2.flow")
    return Q


def flow_transitions(adata, labels: Optional[str] = None, *, key: str = "flow", n_perm: int = 1000,
                     seed: int = 0, exact: bool = True):
    """Label x label transport matrix per pair of consecutive sampling times.

    Each source cell ``i`` carries one unit of mass, split over the next time by its
    plan row ``P(j | i)``; ``mass[k, l]`` sums what cells labelled ``k`` send to cells
    labelled ``l``, and ``prob[k, l] = mass[k, l] / n_k`` (rows sum to 1).

    **Null.** Source labels are permuted among the source cells of that time
    (``n_perm`` times), which keeps each label's size and the target composition and
    breaks only the link between a cell's label and where it goes. ``expected`` is the
    null mean of ``mass`` (``expected_prob`` the same as a probability), ``z`` = (mass - null mean) / null sd, and ``p_enriched`` /
    ``p_depleted`` are one-sided permutation p-values ``(1 + #) / (1 + n_perm)``.
    They are per cell of the fitted plan: they do not include refit variation.

    Parameters
    ----------
    labels : ``obs`` column; default ``obs[f"{key}_archetype"]``.
    exact : re-run the transport of :func:`scjdo2.flow` (same space, ``eps``, solver and
        seed; checked against the stored velocities) and contract the full plan with the
        labels. ``False`` uses the stored top-``keep_plan`` plan instead: no transport
        cost, but each cell's mass outside its kept destinations is redistributed.

    Returns a long DataFrame (one row per time pair x source label x target label) with
    ``to_extrapolated`` = True where the target time is the last one (target operators
    extrapolated). Also stored in ``uns[key]["transitions"]``.
    """
    import pandas as pd

    rec = _flow_rec(adata, key)
    col, code, levels = _labels(adata, key, labels)
    L = len(levels)
    t = _times(adata, rec)
    uniq = np.unique(t)
    onehot = np.zeros((len(code), L))
    onehot[np.arange(len(code)), code] = 1.0
    Qall = _destination_mix(adata, key, rec, onehot, exact)
    rng = np.random.default_rng(seed)
    rows = []
    for t0, t1 in zip(uniq[:-1], uniq[1:]):
        src = np.flatnonzero((t == t0) & np.isfinite(Qall).all(axis=1))
        if len(src) == 0:
            continue
        Q = Qall[src]                                         # (n_src, L) destination label mix
        c = code[src]
        def _contract(lab):
            return np.column_stack([np.bincount(lab, weights=Q[:, l], minlength=L) for l in range(L)])

        mass = _contract(c)
        n_k = np.bincount(c, minlength=L).astype(np.float64)
        ge = np.zeros((L, L))
        le = np.zeros((L, L))
        s1 = np.zeros((L, L))
        s2 = np.zeros((L, L))
        for _ in range(int(n_perm)):
            Mp = _contract(rng.permutation(c))
            ge += Mp >= mass - 1e-12
            le += Mp <= mass + 1e-12
            s1 += Mp
            s2 += Mp ** 2
        mu = s1 / max(n_perm, 1)
        sd = np.sqrt(np.maximum(s2 / max(n_perm, 1) - mu ** 2, 0.0))
        last = bool(t1 == uniq[-1])
        for k in range(L):
            if n_k[k] == 0:
                continue
            for l in range(L):
                rows.append({
                    "from_time": float(t0), "to_time": float(t1), "from_label": levels[k], "to_label": levels[l],
                    "n_from": int(n_k[k]), "mass": float(mass[k, l]), "prob": float(mass[k, l] / n_k[k]),
                    "expected": float(mu[k, l]), "expected_prob": float(mu[k, l] / n_k[k]),
                    "z": float((mass[k, l] - mu[k, l]) / sd[k, l]) if sd[k, l] > 0 else np.nan,
                    "p_enriched": float((1 + ge[k, l]) / (1 + n_perm)) if n_perm else np.nan,
                    "p_depleted": float((1 + le[k, l]) / (1 + n_perm)) if n_perm else np.nan,
                    "to_extrapolated": last})
    table = pd.DataFrame(rows)
    rec["transitions"] = {"labels": col, "levels": levels, "n_perm": int(n_perm), "seed": int(seed),
                          "exact": bool(exact),
                          "table": {c_: table[c_].tolist() for c_ in table.columns}}
    return table


def _prob_matrices(adata, key: str, code: np.ndarray, L: int, t: np.ndarray, P) -> dict[float, np.ndarray]:
    onehot = np.zeros((len(code), L))
    onehot[np.arange(len(code)), code] = 1.0
    out = {}
    uniq = np.unique(t)
    for t0 in uniq[:-1]:
        src = np.flatnonzero(t == t0)
        src = src[np.diff(P.indptr)[src] > 0]
        Q = np.asarray(P[src] @ onehot)
        M = np.zeros((L, L))
        np.add.at(M, code[src], Q)
        out[float(t0)] = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-300)
    return out


# ---------------------------------------------------------------------------
# 3. sampled paths
# ---------------------------------------------------------------------------
def flow_paths(adata, labels: Optional[str] = None, *, key: str = "flow", n_paths: int = 10_000,
               start_time: Optional[float] = None, seed: int = 0, collapse: bool = False):
    """Label sequences of chains sampled through the transport plan.

    Each chain starts at a cell drawn uniformly from ``start_time`` (default: the
    first time) and steps to a destination drawn from its plan row, until the last
    time. The sequence of labels visited is counted and compared with the frequency
    a Markov chain on the per-pair matrices of :func:`flow_transitions` predicts
    (``expected``). ``obs_over_expected`` far from 1 means the path depends on more
    than the current label (memory, or a rare route the matrices average away).

    ``collapse=True`` merges repeated labels (``A1 > A1 > A2`` becomes ``A1 > A2``)
    before counting; the Markov expectation is then summed over the sequences that
    collapse to the same one.

    Returns a DataFrame sorted by count (``path``, ``count``, ``frequency``,
    ``expected``, ``obs_over_expected``), also stored in ``uns[key]["paths"]``.
    """
    import itertools

    import pandas as pd

    rec = _flow_rec(adata, key)
    P = _plan(adata, key)
    col, code, levels = _labels(adata, key, labels)
    L = len(levels)
    t = _times(adata, rec)
    uniq = np.unique(t)
    t_start = uniq[0] if start_time is None else float(start_time)
    if t_start not in set(uniq[:-1]):
        raise ValueError(f"start_time must be a sampling time before the last; got {start_time}")
    steps = [u for u in uniq if u >= t_start]
    start = np.flatnonzero(t == t_start)
    rng = np.random.default_rng(seed)
    cur = start[rng.integers(0, len(start), int(n_paths))]
    seqs = [code[cur]]
    alive = np.ones(len(cur), bool)
    for _ in steps[:-1]:
        nxt = np.full(len(cur), -1)
        u = rng.random(len(cur))
        for p in np.flatnonzero(alive):
            lo, hi = P.indptr[cur[p]], P.indptr[cur[p] + 1]
            if hi == lo:
                alive[p] = False
                continue
            cdf = np.cumsum(P.data[lo:hi])
            nxt[p] = P.indices[lo + min(int(np.searchsorted(cdf, u[p] * cdf[-1])), hi - lo - 1)]
        cur = np.where(alive, nxt, cur)
        seqs.append(np.where(alive, code[cur], -1))
    seqs = np.vstack(seqs).T[alive]

    def _name(s):
        s = [levels[x] for x in s]
        if collapse:
            s = [x for i, x in enumerate(s) if i == 0 or x != s[i - 1]]
        return " > ".join(s)

    names, counts = np.unique([_name(s) for s in seqs], return_counts=True)

    # Markov expectation over every full-length sequence, then collapse.
    Pm = _prob_matrices(adata, key, code, L, t, P)
    pi0 = np.bincount(code[start], minlength=L) / len(start)
    expected: dict[str, float] = {}
    if L ** len(steps) > 2_000_000:
        raise ValueError(f"{L} labels over {len(steps)} times is too many sequences for the Markov "
                         "expectation; use fewer labels or a later start_time")
    for s in itertools.product(range(L), repeat=len(steps)):
        p = pi0[s[0]]
        for a, (tt, b) in enumerate(zip(steps[:-1], s[1:])):
            p *= Pm[float(tt)][s[a], b]
            if p == 0:
                break
        if p > 0:
            nm = _name(s)
            expected[nm] = expected.get(nm, 0.0) + p
    n_ok = int(len(seqs))
    table = pd.DataFrame({"path": names, "count": counts.astype(int), "frequency": counts / max(n_ok, 1)})
    table["expected"] = [expected.get(nm, 0.0) for nm in table["path"]]
    table["obs_over_expected"] = table["frequency"] / table["expected"].where(table["expected"] > 0)
    table = table.sort_values("count", ascending=False, ignore_index=True)
    rec["paths"] = {"labels": col, "n_paths": int(n_paths), "n_complete": n_ok, "start_time": float(t_start),
                    "collapse": bool(collapse), "seed": int(seed),
                    "table": {c_: table[c_].tolist() for c_ in table.columns}}
    return table


# ---------------------------------------------------------------------------
# 4. drift integration
# ---------------------------------------------------------------------------
def flow_integrate(adata, start, *, key: str = "flow", duration: Optional[float] = None,
                   n_steps: int = 100) -> dict[str, np.ndarray]:
    """Integrate the fitted drift ``dz/dt = f(z)`` from start points (RK4).

    Parameters
    ----------
    start : cell indices (int array), a boolean obs mask, or points (m, d) in the
        transport space.
    duration : integration time in units of ``obs[time_key]``; default last minus
        first sampling time.

    Returns ``{"t": (n_steps+1,), "z": (m, n_steps+1, d), "lambda1": (m, n_steps+1),
    "archetype_weights": (m, n_steps+1, K)}``; the weights (only after
    :func:`flow_archetypes`) are the non-negative least-squares fit of the operator
    along the path, normalised as the archetypes were. Outside the cloud of cells the
    drift is extrapolated; a deterministic path also cannot branch, so read it as a
    picture of the mean flow, not as a test.
    """
    import torch
    from torch.func import jacrev, vmap

    from .flow_operator import load_drift

    rec = _flow_rec(adata, key)
    model = load_drift(adata, key)
    p = rec["params"]
    zall = np.asarray(adata.obsm[p["use_rep"]], np.float64)[:, :int(p["n_comps"])]
    start = np.asarray(start)
    if start.dtype == bool:
        z0 = zall[start]
    elif start.ndim == 1 and np.issubdtype(start.dtype, np.integer):
        z0 = zall[start]
    else:
        z0 = np.atleast_2d(start).astype(np.float64)
    if z0.shape[1] != zall.shape[1]:
        raise ValueError(f"start points must have {zall.shape[1]} columns; got {z0.shape[1]}")
    t = _times(adata, rec)
    T = float(t.max() - t.min()) if duration is None else float(duration)
    h = T / int(n_steps)

    def f(z):
        with torch.no_grad():
            return model(torch.as_tensor(z, dtype=torch.float32)).numpy().astype(np.float64)

    Z = np.empty((len(z0), int(n_steps) + 1, z0.shape[1]))
    Z[:, 0] = z0
    for s in range(int(n_steps)):
        z = Z[:, s]
        k1 = f(z)
        k2 = f(z + 0.5 * h * k1)
        k3 = f(z + 0.5 * h * k2)
        k4 = f(z + h * k3)
        Z[:, s + 1] = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    flat = torch.as_tensor(Z.reshape(-1, Z.shape[2]), dtype=torch.float32)
    jac = vmap(jacrev(lambda zi: model(zi[None])[0]))
    with torch.no_grad():
        Jp = np.concatenate([jac(flat[i:i + 4096]).numpy() for i in range(0, len(flat), 4096)]).astype(np.float64)
    out = {"t": np.linspace(0.0, T, int(n_steps) + 1), "z": Z,
           "lambda1": np.linalg.eigvals(Jp).real.max(axis=1).reshape(Z.shape[:2])}
    arch = rec.get("archetypes")
    if arch is not None:
        from .archetypes import _nnls_rows
        norm = arch["normalize"]
        Jn = _normalise(Jp, None if norm == "none" else norm)
        H = np.asarray(arch["patterns"]).reshape(len(arch["names"]), -1)
        out["archetype_weights"] = _nnls_rows(H, Jn.reshape(len(Jn), -1).astype(np.float32)) \
            .reshape(Z.shape[0], Z.shape[1], -1)
    return out
