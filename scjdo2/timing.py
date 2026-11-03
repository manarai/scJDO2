"""Expression-timing QC: do drivers change before responders?

For each archetype, member cells (``W[:, k] ≥ 0.5·max``, the same rule
as :func:`scjdo2.opspace.archetype_operators`) are ordered by an existing
pseudotime (``obs[time_key]``; no new ordering is built). Each gene's
expression is averaged in quantile bins of pseudotime (weighted by the
archetype weight), smoothed with a short moving average, and min-max
scaled to ``[0, 1]`` so every gene is on the same scale.

**Onset** = pseudotime where the scaled curve crosses 0.5 on its way
from its low to its high (rising genes) or from its high to its low
(falling genes). Direction is set by which extreme comes first. A gene
whose smoothed range is below ``min_range`` (in the units of ``X``) is
``flat`` and gets no onset.

Two tests per archetype:

* **role test** — one-sided Mann–Whitney, driver onsets < responder
  onsets (needs ≥ 2 genes in each role).
* **edge test** — for each ``source → target`` edge from
  :func:`scjdo2.influence.influence_edges`, ``lag = onset(target) −
  onset(source)``. Reports the fraction with ``lag > 0`` and a
  permutation p against random pairs of measurable genes with onsets.

Caveat: if the drift / velocity was fitted with this same pseudotime
(Paul15 uses it as a velocity prior), the check is a consistency test,
not independent validation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import mannwhitneyu


def _dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _members(adata: AnnData, archetype_id: int, cutoff: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(adata.obsm["archetype_weights"])
    w = W[:, int(archetype_id)]
    wmax = float(w.max())
    idx = np.where(w >= cutoff * wmax)[0] if wmax > 0 else np.arange(w.size)
    return idx, w[idx]


def smoothed_profiles(
    X: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None,
    n_bins: int = 25, window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean per quantile bin of ``t``, then a centred moving average.

    Returns ``(bin_centres (n_bins,), profiles (n_bins, G))`` in the units of ``X``.
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    w = np.ones(t.size) if w is None else np.asarray(w, dtype=np.float64)
    edges = np.quantile(t, np.linspace(0, 1, int(n_bins) + 1))
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, int(n_bins) - 1)
    prof = np.full((int(n_bins), X.shape[1]), np.nan)
    centres = np.full(int(n_bins), np.nan)
    for k in range(int(n_bins)):
        m = b == k
        if not m.any():
            continue
        wk = w[m] / w[m].sum()
        prof[k] = wk @ X[m]
        centres[k] = float(np.average(t[m], weights=wk))
    ok = np.isfinite(centres)
    centres, prof = centres[ok], prof[ok]
    if window > 1 and prof.shape[0] >= window:
        pad = window // 2
        padded = np.pad(prof, ((pad, pad), (0, 0)), mode="edge")
        kern = np.ones(window) / window
        prof = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="valid"), 0, padded)
    return centres, prof


def onsets(centres: np.ndarray, prof: np.ndarray, min_range: float = 0.05
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Half-way crossing time per gene.

    Returns ``(onset, direction, scaled (n_bins, G), range)`` with
    ``direction`` +1 rising / −1 falling / 0 flat.
    """
    lo, hi = prof.min(axis=0), prof.max(axis=0)
    rng = hi - lo
    scaled = (prof - lo) / np.where(rng > 0, rng, 1.0)
    G = prof.shape[1]
    onset = np.full(G, np.nan)
    direction = np.zeros(G, dtype=int)
    for g in range(G):
        if rng[g] < min_range:
            continue
        s = scaled[:, g]
        i_lo, i_hi = int(np.argmin(s)), int(np.argmax(s))
        rising = i_lo < i_hi
        start = i_lo if rising else i_hi
        seg = s[start:] if rising else 1.0 - s[start:]
        cross = np.where(seg >= 0.5)[0]
        if cross.size == 0:
            continue
        k = start + int(cross[0])
        if k == start:
            onset[g] = centres[k]
        else:
            # Linear interpolation between bins k-1 and k.
            y0, y1 = seg[k - start - 1], seg[k - start]
            f = (0.5 - y0) / (y1 - y0) if y1 != y0 else 0.0
            onset[g] = centres[k - 1] + f * (centres[k] - centres[k - 1])
        direction[g] = 1 if rising else -1
    return onset, direction, scaled, rng


def expression_timing(
    adata: AnnData,
    archetype_id: int,
    time_key: str = "pseudotime",
    layer: Optional[str] = None,
    n_bins: int = 25,
    window: int = 3,
    min_range: float = 0.05,
    n_perm: int = 2000,
    edges: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> dict:
    """Onset timing of drivers vs responders for one archetype.

    Needs :func:`scjdo2.influence.archetype_influence` to have run.
    ``layer=None`` uses ``adata.X`` (expected log-normalised).

    Returns a dict with ``genes`` (DataFrame: gene, role, onset,
    direction, range), ``centres`` / ``scaled`` (``(n_bins, G)``),
    ``role_test`` and ``edge_test`` dicts, and ``edges`` (with ``lag``).
    """
    from . import influence

    info = adata.uns["scjdo2"]["archetypes"][int(archetype_id)]
    if "influence" not in info:
        influence.archetype_influence(adata, archetype_ids=[archetype_id])
    if time_key not in adata.obs:
        raise KeyError(f"adata.obs[{time_key!r}] not found")
    idx, w = _members(adata, archetype_id)
    X = _dense(adata.layers[layer] if layer else adata.X)[idx]
    t = adata.obs[time_key].to_numpy(dtype=np.float64)[idx]
    ok = np.isfinite(t)
    centres, prof = smoothed_profiles(X[ok], t[ok], w[ok], n_bins=n_bins, window=window)
    on, direction, scaled, rng = onsets(centres, prof, min_range=min_range)

    genes = np.asarray(adata.var_names)
    role = info["influence"].set_index("gene")["role"].reindex(genes).to_numpy()
    tbl = pd.DataFrame({"gene": genes, "role": role, "onset": on,
                        "direction": direction, "range": rng})

    d = tbl.loc[(tbl["role"] == "driver") & tbl["onset"].notna(), "onset"].to_numpy()
    r = tbl.loc[(tbl["role"] == "responder") & tbl["onset"].notna(), "onset"].to_numpy()
    role_test = {"n_driver": int(d.size), "n_responder": int(r.size),
                 "median_driver": float(np.median(d)) if d.size else np.nan,
                 "median_responder": float(np.median(r)) if r.size else np.nan,
                 "p_driver_earlier": np.nan}
    if d.size >= 2 and r.size >= 2:
        role_test["p_driver_earlier"] = float(mannwhitneyu(d, r, alternative="less").pvalue)

    if edges is None:
        edges = influence.influence_edges(adata, archetype_id)
    e = edges.copy()
    on_map = dict(zip(genes, on))
    e["lag"] = [on_map.get(tg, np.nan) - on_map.get(s, np.nan)
                for s, tg in zip(e["source"], e["target"])]
    lags = e["lag"].dropna().to_numpy()
    edge_test = {"n_edges": int(lags.size), "frac_source_first": np.nan,
                 "median_lag": np.nan, "p_perm": np.nan, "null_frac": np.nan}
    pool = np.where(np.isfinite(on) & (role != "unmeasured"))[0]
    if lags.size and pool.size > 2:
        frac = float(np.mean(lags > 0))
        rs = np.random.default_rng(seed)
        null = np.empty(int(n_perm))
        for b in range(int(n_perm)):
            a = rs.choice(pool, lags.size)
            c = rs.choice(pool, lags.size)
            dl = on[c] - on[a]
            dl = dl[a != c]
            null[b] = np.mean(dl > 0) if dl.size else np.nan
        edge_test.update(
            frac_source_first=frac, median_lag=float(np.median(lags)),
            null_frac=float(np.nanmean(null)),
            p_perm=float((1 + np.sum(null >= frac)) / (1 + n_perm)),
        )
    return {"genes": tbl, "centres": centres, "scaled": scaled,
            "role_test": role_test, "edge_test": edge_test, "edges": e,
            "time_key": time_key, "n_cells": int(ok.sum())}


# ---------------------------------------------------------------- branches


def _edge_lags(e: pd.DataFrame, on_map: dict) -> np.ndarray:
    return np.array([on_map.get(t, np.nan) - on_map.get(s, np.nan)
                     for s, t in zip(e["source"], e["target"])], dtype=np.float64)


def branch_timing(
    adata: AnnData,
    archetype_id: int,
    branches: dict[str, Sequence[str]],
    cluster_key: str,
    root: Sequence[str] = (),
    time_key: str = "pseudotime",
    cells: str = "path",
    layer: Optional[str] = None,
    n_bins: int = 20,
    window: int = 3,
    min_range: float = 0.05,
    min_cells: int = 50,
    n_perm: int = 2000,
    edges: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> dict:
    """Edge timing along each branch separately.

    A fork archetype acts on the split, so averaging expression over both
    fates can cancel the driver's effect. Here each branch path is
    ``root`` clusters + that branch's clusters (``obs[cluster_key]``),
    ordered by the existing ``obs[time_key]``.

    ``cells="path"`` (default) uses every cell on the branch path, so the
    timing covers the whole lineage the archetype's edges act along;
    ``cells="members"`` restricts to the archetype's member cells on the
    path (weighted by archetype weight). Branches with fewer than
    ``min_cells`` cells are skipped.

    Returns ``{"per_branch"`` (one row per branch: n_cells, n_edges,
    frac_source_first, median_lag, null_frac, p_perm), ``"pooled"``
    (all edge × branch pairs against a per-branch null), ``"lags"``
    (edge × branch long table), ``"onsets"`` (gene × branch)``}``.
    """
    from . import influence

    if cells not in ("path", "members"):
        raise ValueError("cells must be 'path' or 'members'")
    info = adata.uns["scjdo2"]["archetypes"][int(archetype_id)]
    if "influence" not in info:
        influence.archetype_influence(adata, archetype_ids=[archetype_id])
    if edges is None:
        edges = influence.influence_edges(adata, archetype_id)
    genes = np.asarray(adata.var_names)
    role = info["influence"].set_index("gene")["role"].reindex(genes).to_numpy()
    X_all = _dense(adata.layers[layer] if layer else adata.X)
    t_all = adata.obs[time_key].to_numpy(dtype=np.float64)
    cl = adata.obs[cluster_key].astype(str).to_numpy()
    W = np.asarray(adata.obsm["archetype_weights"])[:, int(archetype_id)]
    wmax = float(W.max())
    member = W >= 0.5 * wmax if wmax > 0 else np.ones(W.size, bool)

    rs = np.random.default_rng(seed)
    per_branch, lag_rows, onset_rows = [], [], []
    obs_pos, obs_n, null_pos = 0, 0, np.zeros(int(n_perm))
    for name, clusters in branches.items():
        path = np.isin(cl, list(root) + list(clusters)) & np.isfinite(t_all)
        sel = path & member if cells == "members" else path
        idx = np.where(sel)[0]
        row = {"branch": name, "n_cells": int(idx.size), "n_edges": 0,
               "frac_source_first": np.nan, "median_lag": np.nan,
               "null_frac": np.nan, "p_perm": np.nan}
        if idx.size < int(min_cells):
            per_branch.append(row)
            continue
        w = W[idx] if cells == "members" else np.ones(idx.size)
        cen, prof = smoothed_profiles(X_all[idx], t_all[idx], w, n_bins=n_bins, window=window)
        on, direction, _, rng_ = onsets(cen, prof, min_range=min_range)
        onset_rows.append(pd.DataFrame({"gene": genes, "branch": name, "role": role,
                                        "onset": on, "direction": direction, "range": rng_}))
        lags = _edge_lags(edges, dict(zip(genes, on)))
        for (s, tg), lg in zip(zip(edges["source"], edges["target"]), lags):
            lag_rows.append({"branch": name, "source": s, "target": tg, "lag": lg})
        lags = lags[np.isfinite(lags)]
        pool = np.where(np.isfinite(on) & (role != "unmeasured"))[0]
        if lags.size == 0 or pool.size < 3:
            per_branch.append(row)
            continue
        null = np.empty(int(n_perm))
        for b in range(int(n_perm)):
            a = rs.choice(pool, lags.size)
            c = rs.choice(pool, lags.size)
            dl = on[c] - on[a]
            dl = dl[a != c]
            null[b] = np.mean(dl > 0) if dl.size else np.nan
        frac = float(np.mean(lags > 0))
        row.update(n_edges=int(lags.size), frac_source_first=frac,
                   median_lag=float(np.median(lags)), null_frac=float(np.nanmean(null)),
                   p_perm=float((1 + np.sum(null >= frac)) / (1 + n_perm)))
        per_branch.append(row)
        obs_pos += int(np.sum(lags > 0))
        obs_n += int(lags.size)
        null_pos += np.nan_to_num(null) * lags.size
    pooled = {"n_pairs": obs_n, "frac_source_first": np.nan, "null_frac": np.nan, "p_perm": np.nan}
    if obs_n:
        frac = obs_pos / obs_n
        null_frac = null_pos / obs_n
        pooled.update(frac_source_first=float(frac), null_frac=float(np.mean(null_frac)),
                      p_perm=float((1 + np.sum(null_frac >= frac)) / (1 + n_perm)))
    return {"per_branch": pd.DataFrame(per_branch), "pooled": pooled,
            "lags": pd.DataFrame(lag_rows),
            "onsets": pd.concat(onset_rows, ignore_index=True) if onset_rows else pd.DataFrame()}


# ------------------------------------------------------------ cell states


def default_states(adata: AnnData, archetype_id: int, cluster_key: str,
                   time_key: str = "pseudotime", min_frac: float = 0.05) -> list[str]:
    """Clusters holding ≥ ``min_frac`` of the archetype's member cells,
    ordered by their median ``obs[time_key]`` (over all their cells)."""
    idx, _ = _members(adata, archetype_id)
    cl = adata.obs[cluster_key].astype(str)
    share = cl.iloc[idx].value_counts(normalize=True)
    keep = share[share >= float(min_frac)].index
    med = adata.obs.loc[cl.isin(keep), time_key].groupby(cl[cl.isin(keep)]).median()
    return list(med.sort_values().index)


def network_state_expression(
    adata: AnnData,
    archetype_id: int,
    cluster_key: str,
    states: Optional[Sequence[str]] = None,
    time_key: str = "pseudotime",
    edges: Optional[pd.DataFrame] = None,
    layer: Optional[str] = None,
    min_range: float = 0.05,
    expr_threshold: float = 0.0,
) -> dict:
    """Expression of one archetype's network genes across ordered cell states.

    Genes = sources and targets of the archetype's influence edges
    (:func:`scjdo2.influence.influence_edges` unless ``edges`` is given).
    States = ``states`` in that order, else :func:`default_states`.
    Uses every cell of each state (not only archetype members), so the
    whole progression is visible.

    For each gene: mean per state, fraction of cells with expression >
    ``expr_threshold``, the mean min-max scaled across states, and the
    **on-state** — fractional state position where the scaled mean
    crosses 0.5 from its low side (rising) or high side (falling), as in
    :func:`onsets`. Per edge: ``driver_first`` = source's on-state is
    earlier than the target's (ties count as not first).

    Returns ``{"states", "genes"`` (gene, role, on_state, direction,
    range), ``"mean"``, ``"scaled"``, ``"frac"`` (DataFrames gene ×
    state), ``"edges"`` (with on-states and ``driver_first``),
    ``"summary"`` (per source: n_targets, n_driver_first)``}``.
    """
    from . import influence

    info = adata.uns["scjdo2"]["archetypes"][int(archetype_id)]
    if "influence" not in info:
        influence.archetype_influence(adata, archetype_ids=[archetype_id])
    if edges is None:
        edges = influence.influence_edges(adata, archetype_id)
    if states is None:
        states = default_states(adata, archetype_id, cluster_key, time_key)
    states = [str(s) for s in states]
    genes = list(dict.fromkeys(list(edges["source"]) + list(edges["target"])))
    gidx = [int(np.where(adata.var_names == g)[0][0]) for g in genes]
    cl = adata.obs[cluster_key].astype(str).to_numpy()
    X = adata.layers[layer] if layer else adata.X
    mean = np.full((len(states), len(genes)), np.nan)
    frac = np.full_like(mean, np.nan)
    for k, st in enumerate(states):
        m = cl == st
        if not m.any():
            continue
        Xs = _dense(X[m][:, gidx])
        mean[k] = Xs.mean(axis=0)
        frac[k] = (Xs > expr_threshold).mean(axis=0)
    ok = np.isfinite(mean).all(axis=1)
    pos = np.arange(len(states), dtype=np.float64)
    on, direction, scaled_ok, rng = onsets(pos[ok], mean[ok], min_range=min_range)
    scaled = np.full_like(mean, np.nan)
    scaled[ok] = scaled_ok
    role = info["influence"].set_index("gene")["role"]
    gtbl = pd.DataFrame({"gene": genes, "role": [role.get(g, "?") for g in genes],
                         "on_state": on, "direction": direction, "range": rng})
    on_map = dict(zip(genes, on))
    e = edges.copy()
    e["on_source"] = [on_map[s] for s in e["source"]]
    e["on_target"] = [on_map[t] for t in e["target"]]
    e["driver_first"] = np.where(e["on_source"].notna() & e["on_target"].notna(),
                                 e["on_source"] < e["on_target"], np.nan)
    summ = (e.dropna(subset=["driver_first"]).groupby("source")["driver_first"]
            .agg(n_targets="size", n_driver_first="sum").reset_index())
    summ["n_driver_first"] = summ["n_driver_first"].astype(int)
    as_df = lambda A: pd.DataFrame(A.T, index=genes, columns=states)
    return {"states": states, "genes": gtbl, "mean": as_df(mean), "scaled": as_df(scaled),
            "frac": as_df(frac), "edges": e, "summary": summ}
