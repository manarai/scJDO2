"""Figure for the flow operator: lambda_1 per group, and on the embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from .figures_v05 import _save
from .style import style

__all__ = ["figure_flow_groups"]


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
