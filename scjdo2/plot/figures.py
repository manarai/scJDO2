"""Three publication figures for scJDO2 operator-space analysis.

Phase B scope:
    - :func:`figure_operator_space` renders Panels A and B. Panel C
      (gene-level drivers) needs a gene-space Jacobian from Phase C and
      raises ``NotImplementedError`` for now; the figure is still produced
      with an explanatory panel in its place.
    - :func:`figure_archetype_sheet` raises ``NotImplementedError`` until
      the archetype decomposer (``scjdo2.archetypes``) lands.
    - :func:`figure_v1_vs_v2` consumes the Phase A v1 baseline and the
      scJDO2 state on the same dataset.

The plotting layer reads from AnnData only; it never fits or estimates.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure

from .style import (
    BEHAVIOUR_COLOURS,
    BEHAVIOUR_MARKERS,
    DIVERGING_CMAP,
    OPSPACE_FIG_HEIGHT_IN,
    OPSPACE_FIG_WIDTH_IN,
    behaviour_legend_handles,
    style,
)


def _axis_tag(adata: AnnData) -> str:
    meta = dict(adata.uns.get("scjdo2", {}).get("meta", {}))
    return "" if meta.get("clock_calibrated", False) else " (relative)"


def _confidence_alpha(c: np.ndarray, low: float = 0.15, high: float = 1.0) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if not np.isfinite(c).any():
        return np.full_like(c, high)
    lo, hi = np.nanmin(c), np.nanmax(c)
    if hi - lo < 1e-12:
        return np.full_like(c, high)
    return low + (high - low) * (c - lo) / (hi - lo)


def _scatter_by_behaviour(ax, x, y, beh, alpha, size=10, edgewidth=0.6):
    """Scatter points, grouped by behaviour; per-point alpha is applied via
    an explicit RGBA facecolor array (so matplotlib doesn't reject arrays)."""
    import matplotlib.colors as mcolors

    for behaviour, colour in BEHAVIOUR_COLOURS.items():
        marker = BEHAVIOUR_MARKERS.get(behaviour, "o")
        mask = np.asarray(beh == behaviour)
        if not mask.any():
            continue
        rgba = np.tile(mcolors.to_rgba(colour), (int(mask.sum()), 1))
        rgba[:, 3] = np.clip(alpha[mask], 0.1, 1.0)
        if marker == "o+dot":
            ring_rgba = np.tile(mcolors.to_rgba(colour), (int(mask.sum()), 1))
            ring_rgba[:, 3] = rgba[:, 3]
            ax.scatter(x[mask], y[mask], s=size * 1.6, facecolors="none",
                       edgecolors=ring_rgba, linewidths=edgewidth, zorder=3)
            ax.scatter(x[mask], y[mask], s=size * 0.3, c=rgba,
                       edgecolors="none", zorder=4, marker="o")
        elif marker == "o+":
            ring_rgba = rgba.copy()
            ax.scatter(x[mask], y[mask], s=size * 1.6, facecolors="none",
                       edgecolors=ring_rgba, linewidths=edgewidth, zorder=3)
        else:
            ax.scatter(x[mask], y[mask], s=size, c=rgba,
                       edgecolors="none", zorder=2)


# ======================================================================
# Panel A — operator field on the embedding
# ======================================================================


def _panel_a(ax, adata: AnnData, use_rep: Optional[str],
             lineages: Optional[list], label_fork: bool = True) -> None:
    if use_rep is None:
        use_rep = "X_umap" if "X_umap" in adata.obsm else "X_pca"
    emb = np.asarray(adata.obsm[use_rep])[:, :2]
    lead_re = adata.obs["lead_re"].to_numpy()
    lead_im = adata.obs["lead_im"].to_numpy()
    conf = adata.obs.get("op_confidence", np.ones(adata.n_obs)).to_numpy()
    alpha = _confidence_alpha(conf)

    # Symmetric colour limits at the 98th percentile of |lead_re|
    finite = lead_re[np.isfinite(lead_re)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)

    # Build per-point RGBA so the per-cell confidence alpha can vary cleanly.
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=DIVERGING_CMAP)
    rgba = sm.to_rgba(lead_re)
    rgba[:, 3] = np.clip(alpha, 0.1, 1.0)
    sm.set_array(lead_re)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=rgba, s=6, edgecolors="none", zorder=2)
    # Outline rings on oscillatory cells
    osc = lead_im > 1e-6
    if osc.any():
        ax.scatter(emb[osc, 0], emb[osc, 1], s=14, facecolors="none",
                   edgecolors="black", linewidths=0.4, alpha=0.6, zorder=3)

    # Lineage paths
    if lineages:
        for path in lineages:
            z = path["z"]
            # Project back onto the embedding by nearest-neighbour cell index
            cells = path["cells"]
            ax.plot(emb[cells, 0], emb[cells, 1], color="0.3", linewidth=0.6,
                    alpha=0.7, zorder=4)
            ax.scatter(emb[cells[-1], 0], emb[cells[-1], 1], marker=">",
                       color="0.1", s=18, zorder=5)

    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=18, shrink=0.55,
                        location="right")
    cbar.set_label(f"Re(λ_max){_axis_tag(adata)}", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel(f"{use_rep} 1")
    ax.set_ylabel(f"{use_rep} 2")
    ax.set_aspect("equal", adjustable="datalim")

    if label_fork and "op_behaviour" in adata.obs:
        beh = adata.obs["op_behaviour"].astype(str).to_numpy()
        fork = (beh == "fork") & np.isfinite(lead_re)
        if fork.any():
            idx = np.argmax(np.where(fork, lead_re, -np.inf))
            ax.annotate("fork", xy=(emb[idx, 0], emb[idx, 1]),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=7, color=BEHAVIOUR_COLOURS["fork"],
                        arrowprops=dict(arrowstyle="-", color="0.2",
                                        linewidth=0.4))

    handles, labels = behaviour_legend_handles()
    ax.legend(handles, labels, loc="upper right", fontsize=6,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3)


# ======================================================================
# Panel B — complex plane of leading eigenvalues
# ======================================================================


def _panel_b(ax, adata: AnnData, lineages: Optional[list]) -> None:
    lead_re = adata.obs["lead_re"].to_numpy()
    lead_im = adata.obs["lead_im"].to_numpy()
    beh = adata.obs["op_behaviour"].astype(str).to_numpy() \
        if "op_behaviour" in adata.obs else np.array([""] * adata.n_obs)
    conf = adata.obs.get("op_confidence", np.ones(adata.n_obs)).to_numpy()
    alpha = _confidence_alpha(conf)
    _scatter_by_behaviour(ax, lead_re, lead_im, beh, alpha, size=14)

    ax.axhline(0.0, color="0.7", linewidth=0.4, zorder=1)
    ax.axvline(0.0, color="0.7", linewidth=0.4, linestyle="--", zorder=1)
    ax.text(0.02, 0.03, "Re = 0", transform=ax.get_xaxis_transform(),
            fontsize=6, color="0.3", rotation=90, va="bottom")

    if lineages:
        for path in lineages:
            re = path["lead_re"]
            im = path["lead_im"]
            ax.plot(re, im, color="0.3", linewidth=0.6, alpha=0.7, zorder=3)
            if re.size:
                ax.annotate("", xy=(re[-1], im[-1]), xytext=(re[-2], im[-2]),
                            arrowprops=dict(arrowstyle="->", color="0.2", lw=0.5),
                            zorder=4)

    ax.set_xlabel(f"Re (growth rate{_axis_tag(adata)})")
    ax.set_ylabel(f"Im (rotation rate{_axis_tag(adata)})")

    # Region labels in quiet space
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xpad = 0.05 * (xlim[1] - xlim[0])
    ypad = 0.05 * (ylim[1] - ylim[0])
    ax.text(xlim[0] + xpad, ylim[0] + ypad, "stable", fontsize=6, color="0.4")
    ax.text(xlim[1] - xpad, ylim[0] + ypad, "unstable",
            fontsize=6, color="0.4", ha="right")
    ax.text(xlim[0] + xpad, ylim[1] - ypad, "oscillating",
            fontsize=6, color="0.4", va="top")

    # Footnote under the panel
    ax.text(0.5, -0.18, "conjugate partner at −Im omitted",
            transform=ax.transAxes, fontsize=6, color="0.4", ha="center")


# ======================================================================
# Panel C — eigenvector of one archetype (Phase C)
# ======================================================================


def _panel_c_stub(ax, reason: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, "Panel C — eigenvector drivers\n(" + reason + ")",
            transform=ax.transAxes, fontsize=7, color="0.3",
            ha="center", va="center", wrap=True)


# ======================================================================
# Public: figure_operator_space
# ======================================================================


def figure_operator_space(
    adata: AnnData,
    lineages: Optional[Sequence[dict]] = None,
    archetype: Optional[int] = None,
    use_rep: Optional[str] = None,
    out: Optional[str | Path] = None,
) -> Figure:
    """Three-panel operator-space figure.

    Panels A and B are produced from the stored opspace invariants; Panel C
    is a placeholder in Phase B — needs gene-space Jacobians (Phase C work).

    Parameters
    ----------
    adata : AnnData
        Must have ``lead_re``, ``lead_im``, ``op_behaviour`` populated
        (call :func:`scjdo2.opspace.spectrum`, ``invariants``, ``behaviour``
        first).
    lineages : list of dicts, optional
        Output of :func:`scjdo2.opspace.lineage_path`.
    archetype : int, optional
        Archetype id for Panel C. If None, picks the fork archetype or the
        cell with maximum ``lead_re`` as a fallback.
    use_rep : str, optional
        AnnData.obsm key for Panel A's embedding (default: ``X_umap`` if
        present, else ``X_pca``).
    out : path, optional
        If given, save ``out.pdf`` and ``out.png`` at 300 dpi.
    """
    for col in ("lead_re", "lead_im"):
        if col not in adata.obs:
            raise RuntimeError(
                f"adata.obs['{col}'] missing — call scjdo2.opspace.invariants first"
            )

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN, OPSPACE_FIG_HEIGHT_IN * 1.7))
        gs = fig.add_gridspec(
            nrows=2, ncols=2, height_ratios=[1.0, 0.55], hspace=0.5, wspace=0.55,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, :])

        _panel_a(ax_a, adata, use_rep=use_rep, lineages=list(lineages) if lineages else None)
        ax_a.set_title("A", loc="left", fontsize=10, fontweight="bold")

        _panel_b(ax_b, adata, lineages=list(lineages) if lineages else None)
        ax_b.set_title("B", loc="left", fontsize=10, fontweight="bold")

        # Panel C — gene drivers via gene-space eigenvectors. Phase C work.
        J_shape = adata.uns.get("scjdo2", {}).get("J")
        if J_shape is not None and np.asarray(J_shape).shape[1] == adata.n_vars:
            _panel_c_stub(ax_c, "driver bar chart pending adapter wire-up")
        else:
            _panel_c_stub(
                ax_c,
                "needs gene-space Jacobian from Phase C (NB rate model + ∇_z μ_g)."
            )
        ax_c.set_title("C", loc="left", fontsize=10, fontweight="bold")

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


# ======================================================================
# figure_archetype_sheet — pends Phase B Step 8 adapter
# ======================================================================


def figure_archetype_sheet(
    adata: AnnData,
    out: Optional[str | Path] = None,
) -> Figure:
    """One row per archetype: eigenvalue inset + driver bars + stat block.

    Needs :mod:`scjdo2.archetypes` (semi-NMF adapter around v1
    ``_seminmf``) and gene-space eigenvectors. Both pending.
    """
    raise NotImplementedError(
        "figure_archetype_sheet() needs scjdo2.archetypes (Phase B Step 8 "
        "adapter) and gene-space Jacobians (Phase C). Both pending."
    )


# ======================================================================
# figure_v1_vs_v2
# ======================================================================


def figure_v1_vs_v2(
    adata_v2: AnnData,
    v1_results: str | Path | dict,
    out: Optional[str | Path] = None,
    use_rep: Optional[str] = None,
) -> Figure:
    """Side-by-side comparison of v1 slicing archetypes vs. scJDO2 Panel A.

    ``v1_results`` is either a path to a v1 baseline directory (produced by
    ``compare/run_v1.py``, containing ``adata.h5ad`` with
    ``uns["scjdo"]["activations"]`` and ``uns["scjdo"]["t_centers"]``) or
    a pre-loaded dict with keys ``activations``, ``t_centers``.
    """
    import anndata as ad

    if isinstance(v1_results, (str, Path)):
        v1_adata = ad.read_h5ad(Path(v1_results) / "adata.h5ad")
        v1 = {
            "activations": np.asarray(v1_adata.uns["scjdo"]["activations"]),
            "t_centers": np.asarray(v1_adata.uns["scjdo"]["t_centers"]),
        }
    else:
        v1 = dict(v1_results)

    acts = np.asarray(v1["activations"])          # (T, K)
    tc = np.asarray(v1["t_centers"])              # (T,)
    K = acts.shape[1]

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN, OPSPACE_FIG_HEIGHT_IN))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.3)
        ax_v1 = fig.add_subplot(gs[0, 0])
        ax_v2 = fig.add_subplot(gs[0, 1])

        for k in range(K):
            ax_v1.plot(tc, acts[:, k], linewidth=1.0, label=f"archetype {k}")
        ax_v1.set_xlabel("pseudotime")
        ax_v1.set_ylabel("archetype activation")
        ax_v1.set_title("v1 slicing", loc="left", fontsize=9)
        ax_v1.legend(loc="upper right", fontsize=6)

        _panel_a(ax_v2, adata_v2, use_rep=use_rep, lineages=None, label_fork=True)
        ax_v2.set_title("scJDO2 operator field", loc="left", fontsize=9)

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig
