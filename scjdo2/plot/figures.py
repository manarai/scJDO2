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
    CYCLIC,
    DIVERGING,
    DIVERGING_CMAP,
    OPSPACE_FIG_HEIGHT_IN,
    OPSPACE_FIG_WIDTH_IN,
    behaviour_legend_handles,
    confidence_alpha as _style_confidence_alpha,
    relative_label as _relative_label,
    sensitive_ring as _sensitive_ring,
    style,
)


def _axis_tag(adata: AnnData) -> str:
    """Deprecated — kept for back-compat. Prefer :func:`style.relative_label`.

    Returns ``" (relative)"`` for uncalibrated adata, else ``""``. The
    newer :func:`style.relative_label` returns ``f"{base} (relative)"``
    (no leading space).
    """
    return "" if _is_clock_calibrated(adata) else " (relative)"


def _is_clock_calibrated(adata: AnnData) -> bool:
    try:
        return bool(adata.uns["scjdo2"].get("clock_calibrated", False))
    except (KeyError, AttributeError, TypeError):
        return False


def _confidence_alpha(c: np.ndarray, low: float = 0.15, high: float = 1.0) -> np.ndarray:
    """Thin wrapper over :func:`style.confidence_alpha` (rank-based rescale)."""
    return _style_confidence_alpha(c, lo=low, hi=high)


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

    # Sensitive ring (§0 addition 2): dashed outer circle on high-henrici cells.
    if "op_sensitive" in adata.obs:
        sens = adata.obs["op_sensitive"].to_numpy().astype(bool)
        if sens.any():
            # Convert the scatter marker size s=6 into data units via the
            # axis's x-range heuristic (good enough for a mark; precise
            # placement is in display coords inside sensitive_ring).
            xr = float(np.ptp(emb[:, 0]))
            r = 0.012 * (xr if xr > 0 else 1.0)
            if "op_behaviour" in adata.obs:
                beh_s = adata.obs["op_behaviour"].astype(str).to_numpy()
                for b, col in BEHAVIOUR_COLOURS.items():
                    m = sens & (beh_s == b)
                    if m.any():
                        _sensitive_ring(ax, emb[m, 0], emb[m, 1], col, r)
            else:
                _sensitive_ring(ax, emb[sens, 0], emb[sens, 1], "#404040", r)

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

    # Sensitive ring on Panel B too (§0 addition 2).
    if "op_sensitive" in adata.obs:
        sens = adata.obs["op_sensitive"].to_numpy().astype(bool)
        if sens.any():
            xr = float(np.ptp(lead_re[np.isfinite(lead_re)])) if np.isfinite(lead_re).any() else 1.0
            r = 0.015 * (xr if xr > 0 else 1.0)
            for b, col in BEHAVIOUR_COLOURS.items():
                m = sens & (beh == b)
                if m.any():
                    _sensitive_ring(ax, lead_re[m], lead_im[m], col, r)

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
# Panel C — eigenvector drivers (gene-space, R15 lifted when available)
# ======================================================================


def _panel_c_stub(ax, reason: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, "Panel C — eigenvector drivers\n(" + reason + ")",
            transform=ax.transAxes, fontsize=7, color="0.3",
            ha="center", va="center", wrap=True)


def plot_drivers(
    ax,
    driver_table: "Any",
    mode: Optional[str] = None,
    colour: str = "#333333",
    top: int = 15,
) -> None:
    """Render a driver bar chart from :func:`scjdo2.opspace.drivers`.

    Parameters
    ----------
    driver_table
        pandas DataFrame as returned by :func:`scjdo2.opspace.drivers`.
    mode
        ``"real"``: signed horizontal bars (left = negative, right =
        positive). ``"complex"``: one-sided bars coloured by phase on
        the cyclic ``twilight`` map. ``None`` auto-selects: ``"complex"``
        if any ``phase`` entry is finite, else ``"real"``.
    colour
        Fallback bar colour for the real mode. Complex mode uses the
        cyclic phase colormap instead.
    top
        Number of rows to render (table is already sorted by magnitude).

    Non-``ok`` reason-code rows are drawn hollow with the reason tag at
    the bar end, as per §3.4.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    df = driver_table.head(int(top)).reset_index(drop=True)
    n = len(df)
    if n == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no drivers above threshold",
                transform=ax.transAxes, fontsize=7, color="0.5",
                ha="center", va="center")
        return
    if mode is None:
        phase_vals = df["phase"].to_numpy() if "phase" in df.columns else np.array([])
        mode = "complex" if np.any(np.isfinite(phase_vals)) else "real"

    y = np.arange(n)[::-1]       # top-of-table at the top of the plot
    reasons = df["reason"].to_numpy() if "reason" in df.columns else np.full(n, "ok")
    fisher = df["fisher"].to_numpy() if "fisher" in df.columns else np.full(n, np.nan)
    fisher_se = (
        df["fisher_se"].to_numpy() if "fisher_se" in df.columns else np.full(n, np.nan)
    )
    alpha = _style_confidence_alpha(fisher, lo=0.2, hi=1.0)

    if mode == "real":
        weights = df["weight"].to_numpy()
        pos_block = weights >= 0
        order = np.lexsort((-np.abs(weights), ~pos_block))
        df = df.iloc[order].reset_index(drop=True)
        weights = weights[order]
        reasons = reasons[order]
        fisher_se = fisher_se[order]
        alpha = alpha[order]
        for i, w in enumerate(weights):
            is_ok = str(reasons[i]) == "ok"
            face = colour if is_ok else "none"
            edge = colour
            ax.barh(y[i], w, color=face, edgecolor=edge, linewidth=0.5,
                    alpha=max(float(alpha[i]), 0.1), zorder=2)
            if np.isfinite(fisher_se[i]):
                xerr = float(fisher_se[i])
                ax.errorbar(w, y[i], xerr=xerr, fmt="none",
                            ecolor="0.3", elinewidth=0.3, capsize=1.5, zorder=3)
            if not is_ok:
                ax.text(w, y[i], f" {reasons[i]}", color="0.4", fontsize=6,
                        va="center", ha="left" if w >= 0 else "right")
        # Thin rule between the sign blocks.
        if np.any(weights < 0) and np.any(weights >= 0):
            boundary = float(np.searchsorted(weights >= 0, True))
            boundary = n - boundary - 0.5
            ax.axhline(boundary, color="0.75", linewidth=0.3, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(df["gene"].astype(str).to_numpy())
        ax.axvline(0.0, color="0.6", linewidth=0.4, zorder=1)
        ax.set_xlabel("weight  (← down together | up together →)", fontsize=7)
    else:
        magnitudes = df["magnitude"].to_numpy() if "magnitude" in df.columns else np.abs(df["weight"].to_numpy())
        phases = df["phase"].to_numpy()
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        sm = cm.ScalarMappable(norm=norm, cmap=CYCLIC)
        for i, (m, ph) in enumerate(zip(magnitudes, phases)):
            is_ok = str(reasons[i]) == "ok"
            face = sm.to_rgba(float(ph) if np.isfinite(ph) else 0.0)
            face = (face[0], face[1], face[2], max(float(alpha[i]), 0.1))
            if is_ok:
                ax.barh(y[i], m, color=face, edgecolor="0.3", linewidth=0.4, zorder=2)
            else:
                ax.barh(y[i], m, color="none", edgecolor="0.6", linewidth=0.4, zorder=2)
                ax.text(m, y[i], f" {reasons[i]}", color="0.4", fontsize=6,
                        va="center", ha="left")
            if np.isfinite(fisher_se[i]):
                ax.errorbar(m, y[i], xerr=float(fisher_se[i]), fmt="none",
                            ecolor="0.3", elinewidth=0.3, capsize=1.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(df["gene"].astype(str).to_numpy())
        ax.set_xlabel("|v_g| (magnitude)", fontsize=7)
        # Thin cyclic colorbar just below.
        sm.set_array(phases[np.isfinite(phases)])
        cbar = ax.figure.colorbar(sm, ax=ax, orientation="horizontal",
                                  fraction=0.05, pad=0.18, aspect=25)
        cbar.set_label("early in cycle → late in cycle", fontsize=6)
        cbar.ax.tick_params(labelsize=6)


def _panel_c(ax, adata: AnnData, archetype: Optional[int]) -> None:
    """Render Panel C from :func:`scjdo2.opspace.drivers`.

    Default: an archetype if archetype weights + archetype operator are
    available (first fork archetype, else first archetype). Fallback:
    the fork-class cell with the highest ``lead_re`` (or the overall
    maximum ``lead_re`` when there is no fork cell). Uses the R15 lifted
    eigvec when ``uns["scjdo2"]["op_eigvec_lifted"]`` is populated.
    """
    from .. import opspace

    uns = adata.uns.get("scjdo2", {})
    arch = uns.get("archetypes")
    if archetype is None and arch:
        # First fork archetype, else first archetype.
        arch_items = list(arch.items()) if isinstance(arch, dict) else list(enumerate(arch))
        fork_items = [i for i, a in arch_items if a.get("behaviour") == "fork"]
        archetype = (fork_items[0] if fork_items else arch_items[0][0]) if arch_items else None

    try:
        if archetype is not None and arch:
            df = opspace.drivers(adata, which="archetype", archetype_id=archetype, top=15)
            colour = BEHAVIOUR_COLOURS.get(
                arch[archetype].get("behaviour", "fork"), BEHAVIOUR_COLOURS["fork"],
            )
        else:
            df = opspace.drivers(adata, which="lead", top=15)
            # Colour by the chosen cell's behaviour, if we can recover it.
            if "op_behaviour" in adata.obs and "lead_re" in adata.obs:
                mask = adata.obs["op_behaviour"].astype(str) == "fork"
                lead = adata.obs["lead_re"].to_numpy()
                c_i = int(np.argmax(np.where(mask, lead, -np.inf))) if mask.any() \
                    else int(np.nanargmax(lead))
                beh = str(adata.obs["op_behaviour"].astype(str).iloc[c_i])
            else:
                beh = "fork"
            colour = BEHAVIOUR_COLOURS.get(beh, BEHAVIOUR_COLOURS["fork"])
    except (RuntimeError, NotImplementedError) as err:
        _panel_c_stub(ax, str(err))
        return

    plot_drivers(ax, df, colour=colour, top=15)


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

        # Panel C — gene drivers via the gene-space eigenvector (R15: the
        # ∇η-lift of the latent lead when op_eigvec_lifted is stored).
        _panel_c(ax_c, adata, archetype=archetype)
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
    top: int = 10,
) -> Figure:
    """One row per archetype: eigenvalue inset + driver bars + stat block.

    Reads ``uns["scjdo2"]["archetypes"]`` populated by
    :func:`scjdo2.opspace.archetype_operators`. Rows are sorted by
    behaviour then by cell count. Raises :class:`RuntimeError` if the
    archetype operators have not been run.
    """
    from .. import opspace          # noqa: F401 — kept for symmetry

    arch = adata.uns.get("scjdo2", {}).get("archetypes")
    if not arch:
        raise RuntimeError(
            "figure_archetype_sheet needs uns['scjdo2']['archetypes']; "
            "run scjdo2.opspace.archetype_operators(adata) first."
        )

    items = list(arch.items()) if isinstance(arch, dict) else list(enumerate(arch))
    # Sort: behaviour then -n_cells.
    beh_order = {b: i for i, b in enumerate(BEHAVIOUR_COLOURS.keys())}
    items.sort(key=lambda kv: (beh_order.get(kv[1].get("behaviour", "drift"), 99),
                               -int(kv[1].get("n_cells", 0))))

    n_rows = len(items)
    with style():
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(OPSPACE_FIG_WIDTH_IN, 0.9 * n_rows + 0.5),
            gridspec_kw={"width_ratios": [0.9, 1.4, 1.2], "hspace": 0.6, "wspace": 0.5},
        )
        if n_rows == 1:
            axes = np.array([axes])
        for row, (arch_id, info) in enumerate(items):
            beh = info.get("behaviour", "drift")
            colour = BEHAVIOUR_COLOURS.get(beh, "#555555")

            # (i) mini complex plane
            ax0 = axes[row, 0]
            ev = np.asarray(info.get("eigenvalues", []), dtype=np.complex128)
            if ev.size:
                ax0.scatter(ev.real, ev.imag, s=18, c=colour, edgecolors="0.2",
                            linewidths=0.3, zorder=2)
                ax0.axvline(0.0, color="0.6", linestyle="--", linewidth=0.4)
                ax0.axhline(0.0, color="0.8", linewidth=0.3)
            ax0.set_title(f"archetype {arch_id}", fontsize=7, loc="left")
            ax0.set_xlabel(_relative_label(adata, "Re"), fontsize=6)
            ax0.set_ylabel(_relative_label(adata, "Im"), fontsize=6)
            ax0.tick_params(labelsize=6)

            # (ii) driver bars
            ax1 = axes[row, 1]
            df = info.get("drivers")
            if df is None:
                from .. import opspace as _op
                try:
                    df = _op.drivers(adata, which="archetype", archetype_id=arch_id, top=top)
                except Exception:
                    df = None
            if df is not None and len(df):
                plot_drivers(ax1, df, colour=colour, top=top)
            else:
                ax1.axis("off")
                ax1.text(0.5, 0.5, "no drivers", transform=ax1.transAxes,
                         fontsize=7, color="0.5", ha="center", va="center")

            # (iii) text stat block
            ax2 = axes[row, 2]
            ax2.axis("off")
            n_cells = int(info.get("n_cells", 0))
            mean_conf = float(info.get("mean_confidence", float("nan")))
            henrici = float(info.get("henrici", float("nan")))
            tgain = float(info.get("transient_gain", float("nan")))
            sens = bool(info.get("sensitive", False))
            lines = [
                f"behaviour: {beh}",
                f"sensitive: {'yes' if sens else 'no'}",
                f"eigenvalues: " + (
                    ", ".join(f"{e.real:+.2f}{e.imag:+.2f}j" for e in ev[:3])
                    if ev.size else "—"
                ),
                f"henrici: {henrici:.3g}",
                f"transient gain: {tgain:.3g}",
                f"n cells: {n_cells}",
                f"mean conf: {mean_conf:.3g}",
            ]
            ax2.text(0.0, 1.0, "\n".join(lines),
                     transform=ax2.transAxes, fontsize=7, va="top", ha="left")

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


# ======================================================================
# figure_ledger_summary — stacked bar + rank histogram
# ======================================================================


def figure_ledger_summary(
    adata: AnnData,
    out: Optional[str | Path] = None,
) -> Figure:
    """Supplementary: fraction of reason codes per confidence decile +
    histogram of per-cell tangent rank.

    Reads ``uns["scjdo2"]["fisher_ledger"]`` (per-cell reason arrays) and
    ``obs["op_confidence"]``. If no ledger / confidence is present, the
    corresponding axis is blanked and the figure still writes."""
    from .. import opspace          # for REASON_CODES

    with style():
        fig, axes = plt.subplots(
            1, 2,
            figsize=(OPSPACE_FIG_WIDTH_IN, 0.45 * OPSPACE_FIG_WIDTH_IN),
            gridspec_kw={"wspace": 0.3},
        )
        ax_bar, ax_hist = axes

        # Stacked reason-code fractions per confidence decile.
        ledger = adata.uns.get("scjdo2", {}).get("fisher_ledger")
        reason_per_cell = None
        if ledger is not None:
            reason_per_cell = (
                ledger.get("reason_per_cell") if isinstance(ledger, dict)
                else getattr(ledger, "reason_per_cell", None)
            )
        conf = adata.obs.get("op_confidence")
        if reason_per_cell is not None and conf is not None:
            conf_arr = np.asarray(conf.to_numpy() if hasattr(conf, "to_numpy") else conf)
            finite = np.isfinite(conf_arr)
            deciles = np.full(len(conf_arr), -1, dtype=int)
            if finite.any():
                qs = np.nanquantile(conf_arr[finite], np.linspace(0, 1, 11))
                deciles[finite] = np.clip(np.searchsorted(qs[1:-1], conf_arr[finite]), 0, 9)
            codes = opspace.REASON_CODES
            counts = np.zeros((10, len(codes)), dtype=float)
            for c, dec in enumerate(deciles):
                if dec < 0:
                    continue
                r = np.asarray(reason_per_cell[c])
                for k, code in enumerate(codes):
                    counts[dec, k] += float(np.sum(r == k)) if r.dtype.kind in "iu" else \
                                      float(np.sum(r == code))
            totals = counts.sum(axis=1, keepdims=True)
            fracs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
            left = np.zeros(10)
            cmap = plt.get_cmap("Set2")
            for k, code in enumerate(codes):
                ax_bar.barh(np.arange(10), fracs[:, k], left=left,
                            color=cmap(k), label=code, height=0.8)
                left += fracs[:, k]
            ax_bar.set_yticks(np.arange(10))
            ax_bar.set_yticklabels([f"d{i+1}" for i in range(10)], fontsize=6)
            ax_bar.set_xlabel("fraction of entries", fontsize=7)
            ax_bar.set_ylabel("op_confidence decile", fontsize=7)
            ax_bar.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            ax_bar.axis("off")
            ax_bar.text(0.5, 0.5, "no Fisher ledger / confidence",
                        transform=ax_bar.transAxes, fontsize=7, color="0.5",
                        ha="center", va="center")

        # op_rank histogram.
        if "op_rank" in adata.obs:
            r = adata.obs["op_rank"].to_numpy()
            ax_hist.hist(r[np.isfinite(r)], bins=np.arange(np.nanmax(r) + 2) - 0.5,
                         color="#378ADD", edgecolor="0.2", linewidth=0.3)
            ax_hist.set_xlabel("op_rank (r_c)", fontsize=7)
            ax_hist.set_ylabel("cells", fontsize=7)
            ax_hist.tick_params(labelsize=6)
        else:
            ax_hist.axis("off")
            ax_hist.text(0.5, 0.5, "no op_rank stored",
                         transform=ax_hist.transAxes, fontsize=7, color="0.5",
                         ha="center", va="center")

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


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
