"""Figure for the flow operator: lambda_1 per group, and on the embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from .figures_v05 import _save
from .style import style

__all__ = ["figure_flow_groups", "figure_flow_archetypes", "figure_flow_transitions"]


def figure_flow_groups(adata, groupby: str, *, key: str = "flow", basis: Optional[str] = None,
                       out: Optional[str | Path] = None) -> Figure:
    """Two panels: median ``lambda_1`` per group with its bootstrap interval
    (from :func:`scjdo2.flow_groups`, computed here if missing), and the cells on a
    2-D basis coloured by ``lambda_1``.

    ``basis`` defaults to ``obsm["X_umap"]`` when present, else the first two
    columns of the transport space. Groups with fewer than ``min_cells`` cells have
    no interval and are drawn hollow.
    """
    import matplotlib.pyplot as plt

    from ..api import flow_groups

    col = f"{key}_lambda1"
    if col not in adata.obs:
        raise RuntimeError(f"obs[{col!r}] missing: run scjdo2.flow first")
    stored = adata.uns.get(key, {}).get("groups")
    if stored is None or stored.get("groupby") != groupby:
        flow_groups(adata, groupby, key=key)
        stored = adata.uns[key]["groups"]
    g = stored["table"]
    order = np.argsort(-np.asarray(g["median_lambda1"], float))

    if basis is None:
        basis = "X_umap" if "X_umap" in adata.obsm else adata.uns[key]["params"]["use_rep"]
    xy = np.asarray(adata.obsm[basis])[:, :2]
    lam = adata.obs[col].to_numpy(float)

    with style():
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 0.25 * len(order) + 2.2),
                                         gridspec_kw={"width_ratios": [1, 1.1]})
        for row, i in enumerate(order):
            med, lo, hi = g["median_lambda1"][i], g["ci_low"][i], g["ci_high"][i]
            ok = np.isfinite(lo)
            if ok:
                ax_l.plot([lo, hi], [row, row], color="0.3", lw=1)
            ax_l.plot(med, row, "o", ms=4, color="#d55e00" if ok else "white",
                      mec="#d55e00")
        ax_l.set_yticks(range(len(order)))
        ax_l.set_yticklabels([f"{g['group'][i]}  (n={g['n_cells'][i]})" for i in order], fontsize=7)
        ax_l.invert_yaxis()
        ax_l.set_xlabel(r"median $\lambda_1$ (95% bootstrap CI)")
        ax_l.set_title(f"flow operator by {groupby}", fontsize=9)

        lo, hi = np.nanpercentile(lam, [2, 98])
        sc = ax_r.scatter(xy[:, 0], xy[:, 1], c=np.clip(lam, lo, hi), s=2, cmap="viridis",
                          rasterized=True, linewidths=0)
        ax_r.set_xticks([]); ax_r.set_yticks([])
        ax_r.set_xlabel(f"{basis} 1"); ax_r.set_ylabel(f"{basis} 2")
        ax_r.set_title(r"$\lambda_1$ per cell (2–98% clipped)", fontsize=9)
        fig.colorbar(sc, ax=ax_r, shrink=0.7, label=r"$\lambda_1$")
        fig.text(0.01, -0.02, r"$\lambda_1$ is a within-dataset branch-point score for groups of "
                 "cells; check per-cell values with scjdo2.flow_stability.", fontsize=6, color="0.4")
        _save(fig, out)
    return fig


def figure_flow_archetypes(adata, *, key: str = "flow", out: Optional[str | Path] = None) -> Figure:
    """The shape of each flow-operator archetype (:func:`scjdo2.flow_archetypes`), two
    panels per archetype on shared axes:

    * its eigenvalues on the complex plane (real part across, imaginary part up);
      a positive real part is a growing direction, a non-zero imaginary part rotation;
    * the matrix itself as a component x component heatmap; asymmetry across the
      diagonal is the non-normal part (stretching, transient gain), which the
      eigenvalues do not show.
    """
    import matplotlib.pyplot as plt

    arch = adata.uns.get(key, {}).get("archetypes")
    if arch is None:
        raise RuntimeError(f"uns[{key!r}]['archetypes'] missing: run scjdo2.flow_archetypes first")
    H = np.asarray(arch["patterns"], float)
    re, im = np.asarray(arch["eig_real"], float), np.asarray(arch["eig_imag"], float)
    names = list(arch["names"])
    K, d = H.shape[0], H.shape[1]
    counts = adata.obs[f"{key}_archetype"].value_counts()
    lim_re = 1.1 * max(np.abs(re).max(), 1e-12)
    lim_im = 1.1 * max(np.abs(im).max(), lim_re * 0.25)
    vmax = np.abs(H).max()
    with style():
        fig, axes = plt.subplots(2, K, figsize=(1.9 * K + 0.6, 4.0), squeeze=False)
        for k in range(K):
            ax = axes[0, k]
            ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
            ax.scatter(re[k], im[k], s=14, color=np.where(re[k] > 0, "#d55e00", "#0072b2"), zorder=3)
            ax.set_xlim(-lim_re, lim_re); ax.set_ylim(-lim_im, lim_im)
            ax.set_title(f"{names[k]}  (n={int(counts.get(names[k], 0))})", fontsize=8)
            ax.set_xlabel(r"Re $\lambda$", fontsize=7)
            if k == 0:
                ax.set_ylabel(r"Im $\lambda$", fontsize=7)
            ax.tick_params(labelsize=6)
            ax = axes[1, k]
            img = ax.imshow(H[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([0, d - 1]); ax.set_xticklabels(["1", f"{d}"], fontsize=6)
            ax.set_yticks([0, d - 1]); ax.set_yticklabels(["1", f"{d}"] if k == 0 else [], fontsize=6)
            ax.set_xlabel("component", fontsize=6)
        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.45, anchor=(0.0, 0.0), label="archetype entry")
        norm = arch.get("normalize", "none")
        fig.text(0.01, -0.04, f"Archetypes of J_flow (semi-NMF, K={K}, cells normalised: {norm}). "
                 "Rows: eigenvalues; matrix (row = d f_i, column = d z_j).", fontsize=6, color="0.4")
        _save(fig, out)
    return fig


def figure_flow_transitions(adata, *, key: str = "flow", out: Optional[str | Path] = None) -> Figure:
    """One heatmap per time pair from :func:`scjdo2.flow_transitions`: the row-normalised
    transport ``P(to | from)``, with cells marked where the permutation p-value for
    enrichment is below 0.01 (``*``) and transitions into the last time hatched
    (target operators extrapolated)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    tr = adata.uns.get(key, {}).get("transitions")
    if tr is None:
        raise RuntimeError(f"uns[{key!r}]['transitions'] missing: run scjdo2.flow_transitions first")
    tab = pd.DataFrame(tr["table"])
    levels = list(tr["levels"])
    pairs = tab[["from_time", "to_time"]].drop_duplicates().to_numpy()
    with style():
        fig, axes = plt.subplots(1, len(pairs), figsize=(2.4 * len(pairs) + 1.6, 3.0), squeeze=False)
        for n_ax, (ax, (t0, t1)) in enumerate(zip(axes[0], pairs)):
            sub = tab[(tab.from_time == t0) & (tab.to_time == t1)]
            M = sub.pivot(index="from_label", columns="to_label", values="prob").reindex(index=levels, columns=levels)
            Pv = sub.pivot(index="from_label", columns="to_label", values="p_enriched").reindex(index=levels,
                                                                                              columns=levels)
            img = ax.imshow(M.to_numpy(float), cmap="Greys", vmin=0, vmax=1)
            for i in range(len(levels)):
                for j in range(len(levels)):
                    if np.isfinite(Pv.iat[i, j]) and Pv.iat[i, j] < 0.01:
                        ax.text(j, i, "*", ha="center", va="center", color="#d55e00", fontsize=9)
            ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, fontsize=6, rotation=90)
            ax.set_yticks(range(len(levels))); ax.set_yticklabels(levels if n_ax == 0 else [], fontsize=6)
            ax.set_xlabel(f"to (t = {t1:g})", fontsize=7)
            if n_ax == 0:
                ax.set_ylabel("from", fontsize=7)
            last = bool(sub["to_extrapolated"].iloc[0])
            ax.set_title(f"t = {t0:g} \u2192 {t1:g}" + ("\n(target extrapolated)" if last else ""), fontsize=8)
        fig.colorbar(img, ax=list(axes[0]), shrink=0.8, label="P(to | from)")
        fig.text(0.01, -0.2, f"* enriched vs source-label shuffle, p < 0.01 ({tr['n_perm']} permutations). "
                 f"Labels: obs[{tr['labels']!r}].", fontsize=6, color="0.4")
        _save(fig, out)
    return fig
