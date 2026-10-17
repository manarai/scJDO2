"""Behaviour-landscape figures — occupancy of operator space.

Two figures that ask "which operator behaviours does this population
actually occupy, and how densely", rather than "what is each cell":

* :func:`figure_behaviour_landscape` — a density map on an
  orientation-free plane of operator-space descriptors
  (``x = lead_re``, ``y = spectral_gap`` or ``henrici``), each cell
  weighted by its Fisher mass, drawn as topographic contours. A
  flow-scrambled null (``scjdo2.opspace.flow_scrambled_null``) is
  overlaid in grey or subtracted, so only density rising above the
  null reads as landscape.
* :func:`figure_behaviour_ternary` — Panel A: exact ternary with
  holder / drift / fork at the corners, each cell placed by the fraction
  of its latent eigenvalues below ``−tol``, within ``±tol``, above
  ``+tol`` (``scjdo2.opspace.behaviour_fractions``), a density layer
  behind the dots, dots coloured by cell type. Panel B: stacked bars of
  the three fractions per cell type, split by condition (young vs aged).

Both read from AnnData only and never fit. Coordinates are never the
raw ``J`` entries or the semi-NMF archetype weights, so the maps do
not inherit the embedding's axis orientation and are comparable across
datasets and conditions. The eigenvalue axis carries "(relative)"
unless the run is clock-calibrated. Intended first for the
measured-flow arms (pitchfork truth, scNT-seq, LARRY); on a snapshot
run the landscape is the Hessian of a density and the null must be
``mode="marginal"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from .style import (
    BEHAVIOUR_COLOURS,
    OPSPACE_FIG_HEIGHT_IN,
    OPSPACE_FIG_WIDTH_IN,
    relative_label,
    style,
)

_SQRT3_2 = float(np.sqrt(3.0) / 2.0)
_TERNARY_CORNERS = {
    "holder": (0.5, _SQRT3_2),
    "drift": (0.0, 0.0),
    "fork": (1.0, 0.0),
}
# Categorical palette for cell types (colour-blind safe, ≤ 10 classes;
# wraps beyond that).
_TYPE_PALETTE = [
    "#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#EF9F27",
    "#D4537E", "#639922", "#888780", "#185FA5", "#993C1D",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _weighted_kde_grid(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
    xlim: tuple[float, float], ylim: tuple[float, float],
    n_grid: int = 120, bw: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fisher-weighted Gaussian KDE on a regular grid. Returns
    ``(X, Y, Z)`` with ``Z`` normalised to integrate to 1 over the grid.
    Falls back to a weighted 2-D histogram when fewer than 5 finite
    points are available (KDE needs a covariance)."""
    from scipy.stats import gaussian_kde

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    if m.sum() < 5:
        H, _, _ = np.histogram2d(
            x[m], y[m], bins=[xs, ys], weights=w[m],
        )
        Z = np.zeros_like(X)
        Z[:-1, :-1] = H.T
    else:
        try:
            kde = gaussian_kde(
                np.vstack([x[m], y[m]]), weights=w[m],
                bw_method=bw if bw is not None else "scott",
            )
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        except np.linalg.LinAlgError:
            # Degenerate covariance (all points on a line) — histogram.
            H, _, _ = np.histogram2d(x[m], y[m], bins=[xs, ys], weights=w[m])
            Z = np.zeros_like(X)
            Z[:-1, :-1] = H.T
    total = float(np.sum(Z)) * (xs[1] - xs[0]) * (ys[1] - ys[0])
    if total > 0:
        Z = Z / total
    return X, Y, Z


def _shared_limits(
    a: np.ndarray, b: Optional[np.ndarray], pad: float = 0.08,
) -> tuple[float, float]:
    vals = a[np.isfinite(a)]
    if b is not None:
        vals = np.concatenate([vals, b[np.isfinite(b)]])
    if vals.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def _ternary_xy(neg: np.ndarray, zero: np.ndarray, pos: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric (holder, drift, fork) → Cartesian on the unit triangle."""
    s = neg + zero + pos
    s = np.where(s > 0, s, 1.0)
    a, b, c = neg / s, zero / s, pos / s
    x = (a * _TERNARY_CORNERS["holder"][0] + b * _TERNARY_CORNERS["drift"][0]
         + c * _TERNARY_CORNERS["fork"][0])
    y = (a * _TERNARY_CORNERS["holder"][1] + b * _TERNARY_CORNERS["drift"][1]
         + c * _TERNARY_CORNERS["fork"][1])
    return x, y


def _draw_ternary_frame(ax, tol: float, grid: int = 4) -> None:
    corners = _TERNARY_CORNERS
    tri = np.array([corners["drift"], corners["fork"], corners["holder"], corners["drift"]])
    ax.plot(tri[:, 0], tri[:, 1], color="#404040", lw=0.6, zorder=5)
    # light interior grid at 25 % steps
    for k in range(1, grid):
        f = k / grid
        for (p, q), (r, s) in (
            (("drift", "fork"), ("drift", "holder")),
            (("fork", "holder"), ("fork", "drift")),
            (("holder", "drift"), ("holder", "fork")),
        ):
            p0, q0 = np.array(corners[p]), np.array(corners[q])
            r0, s0 = np.array(corners[r]), np.array(corners[s])
            A = p0 + f * (q0 - p0)
            B = r0 + f * (s0 - r0)
            ax.plot([A[0], B[0]], [A[1], B[1]], color="#C8C8C8", lw=0.3, zorder=1)
    ax.text(*corners["holder"], "holder\n(λ < −tol)", ha="center", va="bottom",
            color=BEHAVIOUR_COLOURS["holder"], fontsize=7,
            transform=ax.transData)
    ax.text(corners["drift"][0], corners["drift"][1] - 0.02, "drift\n(|λ| ≤ tol)",
            ha="center", va="top", color=BEHAVIOUR_COLOURS["drift"], fontsize=7)
    ax.text(corners["fork"][0], corners["fork"][1] - 0.02, "fork\n(λ > tol)",
            ha="center", va="top", color=BEHAVIOUR_COLOURS["fork"], fontsize=7)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.16, _SQRT3_2 + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.13, f"tol = {tol:.3g}", ha="center", va="top",
            fontsize=6, color="#606060")


def _footer(fig: Figure, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=6, color="#606060", ha="left", va="bottom")


# ----------------------------------------------------------------------
# figure 1 — landscape
# ----------------------------------------------------------------------


def figure_behaviour_landscape(
    adata: AnnData,
    y: str = "spectral_gap",
    null: Optional[dict] = None,
    weight: str = "fisher",
    subtract: bool = False,
    n_levels: int = 8,
    n_grid: int = 120,
    bw: Optional[float] = None,
    show_points: bool = True,
    out: Optional[str | Path] = None,
) -> Figure:
    """Fisher-weighted density of cells on the operator-space plane
    ``(lead_re, y)``, drawn as contours, with an optional scrambled-flow
    null.

    Parameters
    ----------
    adata
        Needs ``obs['lead_re']`` and ``obs[y]`` (from
        :func:`scjdo2.opspace.invariants`). ``y`` is ``"spectral_gap"``
        (how sharply the leading behaviour is separated from the next
        mode) or ``"henrici"`` (non-normality; identically ~0 for a
        symmetric snapshot J, so choose ``spectral_gap`` there).
    null
        Output of :func:`scjdo2.opspace.flow_scrambled_null`. Drawn as
        grey contour lines when ``subtract=False``; when
        ``subtract=True`` the plotted field is ``max(real − null, 0)``
        so only density above the null is shown.
    weight
        ``"fisher"`` (per-cell Fisher mass from the ledger; ones when
        no ledger is stored) or ``"none"``.
    show_points
        Overlay a faint scatter of the cells on top of the contours.
    out
        If given, save ``out.pdf`` and ``out.png``.
    """
    for col in ("lead_re", y):
        if col not in adata.obs:
            raise RuntimeError(
                f"adata.obs[{col!r}] missing — call scjdo2.opspace.invariants first",
            )
    from .. import opspace as _opspace

    x_real = adata.obs["lead_re"].to_numpy(dtype=float)
    y_real = adata.obs[y].to_numpy(dtype=float)
    w_real = (_opspace.fisher_mass(adata) if weight == "fisher"
              else np.ones(adata.n_obs))

    x_null = y_null = w_null = None
    if null is not None:
        x_null = np.asarray(null["lead_re"], dtype=float)
        y_null = np.asarray(null[y], dtype=float)
        w_null = np.asarray(null.get("weight", np.ones_like(x_null)), dtype=float)

    xlim = _shared_limits(x_real, x_null)
    ylim = _shared_limits(y_real, y_null)
    X, Y, Z = _weighted_kde_grid(x_real, y_real, w_real, xlim, ylim, n_grid, bw)
    Zn = None
    if x_null is not None:
        _, _, Zn = _weighted_kde_grid(x_null, y_null, w_null, xlim, ylim, n_grid, bw)

    with style():
        fig, ax = plt.subplots(
            figsize=(OPSPACE_FIG_WIDTH_IN * 0.6, OPSPACE_FIG_HEIGHT_IN),
        )
        if subtract and Zn is not None:
            field = np.clip(Z - Zn, 0.0, None)
            title = "density above scrambled-flow null"
        else:
            field = Z
            title = "Fisher-weighted density" if weight == "fisher" else "cell density"
        vmax = float(np.max(field)) if np.isfinite(field).any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        levels = np.linspace(vmax / n_levels, vmax, n_levels)
        cf = ax.contourf(X, Y, field, levels=levels, cmap="Blues", alpha=0.9,
                         extend="max")
        ax.contour(X, Y, field, levels=levels, colors="#1F4E79", linewidths=0.3)
        if Zn is not None and not subtract:
            nl = np.linspace(float(np.max(Zn)) / n_levels, float(np.max(Zn)), n_levels)
            ax.contour(X, Y, Zn, levels=nl, colors="#808080",
                       linewidths=0.5, linestyles="dashed")
        if show_points:
            ax.scatter(x_real, y_real, s=2, c="#1F1F1F", alpha=0.15,
                       edgecolors="none", zorder=3)
        ax.axvline(0.0, color="#404040", lw=0.5, ls=":")
        ax.set_xlabel(relative_label(adata, "leading eigenvalue Re(λ₁)"))
        ylab = {"spectral_gap": "spectral gap Re(λ₁) − Re(λ₂)",
                "henrici": "Henrici index"}.get(y, y)
        if y == "spectral_gap":
            ylab = relative_label(adata, ylab)
        ax.set_ylabel(ylab)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.96, "holder ←", transform=ax.transAxes, fontsize=7,
                color=BEHAVIOUR_COLOURS["holder"], ha="left", va="top")
        ax.text(0.98, 0.96, "→ fork", transform=ax.transAxes, fontsize=7,
                color=BEHAVIOUR_COLOURS["fork"], ha="right", va="top")
        cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("density", fontsize=7)
        if Zn is not None and not subtract:
            import matplotlib.lines as mlines
            h = mlines.Line2D([], [], color="#808080", lw=0.8, ls="--")
            ax.legend([h], [f"scrambled-flow null ({null.get('mode', 'flow')}, "
                            f"n_perm={null.get('n_perm', '?')})"],
                      loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)
        _footer(fig, "coordinates: operator-space descriptors (orientation-free); "
                     f"weight: {weight}; null: {'none' if null is None else null.get('mode', 'flow')}")
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


# ----------------------------------------------------------------------
# figure 2 — ternary + composition bars
# ----------------------------------------------------------------------


def behaviour_composition(
    adata: AnnData,
    by: str = "cluster",
    condition_key: Optional[str] = None,
) -> pd.DataFrame:
    """Mean of the three eigenvalue-sign fractions per ``by`` (× condition).
    Returns a tidy DataFrame with columns ``by``, ``condition``, ``n``,
    ``frac_neg``, ``frac_zero``, ``frac_pos``."""
    for col in ("op_frac_neg", "op_frac_zero", "op_frac_pos"):
        if col not in adata.obs:
            raise RuntimeError(
                f"adata.obs[{col!r}] missing — call scjdo2.opspace.behaviour_fractions first",
            )
    df = adata.obs[[by, "op_frac_neg", "op_frac_zero", "op_frac_pos"]].copy()
    df[by] = df[by].astype(str)
    if condition_key is not None:
        df["condition"] = adata.obs[condition_key].astype(str).to_numpy()
    else:
        df["condition"] = "all"
    g = df.groupby([by, "condition"], observed=True)
    out = g[["op_frac_neg", "op_frac_zero", "op_frac_pos"]].mean()
    out["n"] = g.size()
    out = out.reset_index().rename(columns={
        "op_frac_neg": "frac_neg", "op_frac_zero": "frac_zero", "op_frac_pos": "frac_pos",
    })
    return out[[by, "condition", "n", "frac_neg", "frac_zero", "frac_pos"]]


def figure_behaviour_ternary(
    adata: AnnData,
    color_key: str = "cluster",
    condition_key: Optional[str] = None,
    conditions: Optional[Sequence[str]] = None,
    density: bool = True,
    hex_grid: int = 30,
    point_size: float = 4.0,
    out: Optional[str | Path] = None,
) -> Figure:
    """Panel A: exact ternary of eigenvalue-sign fractions, one dot per
    cell, coloured by ``color_key``, with a Fisher-weighted hexbin
    density behind the dots. Panel B: stacked composition bars per
    ``color_key`` group, split by ``condition_key`` (e.g. young vs aged).

    Needs ``obs['op_frac_neg' / 'op_frac_zero' / 'op_frac_pos']`` from
    :func:`scjdo2.opspace.behaviour_fractions` and the tolerance in
    ``uns['scjdo2']['behaviour_fractions_tol']``.
    """
    for col in ("op_frac_neg", "op_frac_zero", "op_frac_pos"):
        if col not in adata.obs:
            raise RuntimeError(
                f"adata.obs[{col!r}] missing — call scjdo2.opspace.behaviour_fractions first",
            )
    if color_key not in adata.obs:
        raise KeyError(f"obs[{color_key!r}] missing")
    from .. import opspace as _opspace

    tol = float(adata.uns.get("scjdo2", {}).get("behaviour_fractions_tol", float("nan")))
    neg = adata.obs["op_frac_neg"].to_numpy(dtype=float)
    zero = adata.obs["op_frac_zero"].to_numpy(dtype=float)
    pos = adata.obs["op_frac_pos"].to_numpy(dtype=float)
    x, yv = _ternary_xy(neg, zero, pos)
    w = _opspace.fisher_mass(adata)
    types = adata.obs[color_key].astype(str).to_numpy()
    type_levels = list(pd.unique(types))
    colour_of = {t: _TYPE_PALETTE[i % len(_TYPE_PALETTE)] for i, t in enumerate(type_levels)}

    comp = behaviour_composition(adata, by=color_key, condition_key=condition_key)
    if conditions is None:
        conditions = list(pd.unique(comp["condition"]))
    conditions = [str(c) for c in conditions]

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN, OPSPACE_FIG_HEIGHT_IN * 0.9))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.15], wspace=0.35)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])

        # Panel A — ternary
        finite = np.isfinite(x) & np.isfinite(yv)
        if density and finite.sum() > 0:
            ax_a.hexbin(x[finite], yv[finite], C=w[finite], reduce_C_function=np.sum,
                        gridsize=hex_grid, cmap="Greys", mincnt=1, linewidths=0.0,
                        alpha=0.55, zorder=0,
                        extent=(-0.02, 1.02, -0.02, _SQRT3_2 + 0.02))
        for t in type_levels:
            m = finite & (types == t)
            if not m.any():
                continue
            ax_a.scatter(x[m], yv[m], s=point_size, c=colour_of[t],
                         edgecolors="none", alpha=0.8, zorder=3, label=t)
        _draw_ternary_frame(ax_a, tol=tol)
        ax_a.set_title("A", loc="left", fontsize=10, fontweight="bold")
        if len(type_levels) <= 12:
            ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                        markerscale=2.0, ncol=1, fontsize=6, handletextpad=0.3)

        # Panel B — stacked composition bars per type × condition
        n_types = len(type_levels)
        n_cond = max(1, len(conditions))
        width = 0.8 / n_cond
        xs_base = np.arange(n_types)
        for ci, cond in enumerate(conditions):
            offs = xs_base + (ci - (n_cond - 1) / 2.0) * width
            sub = comp[comp["condition"] == cond].set_index(color_key)
            fn = np.array([sub["frac_neg"].get(t, np.nan) for t in type_levels])
            fz = np.array([sub["frac_zero"].get(t, np.nan) for t in type_levels])
            fp = np.array([sub["frac_pos"].get(t, np.nan) for t in type_levels])
            fn, fz, fp = (np.nan_to_num(a) for a in (fn, fz, fp))
            hatch = "" if ci == 0 else ["//", "..", "xx"][(ci - 1) % 3]
            ax_b.bar(offs, fn, width, color=BEHAVIOUR_COLOURS["holder"],
                     edgecolor="white", linewidth=0.3, hatch=hatch)
            ax_b.bar(offs, fz, width, bottom=fn, color=BEHAVIOUR_COLOURS["drift"],
                     edgecolor="white", linewidth=0.3, hatch=hatch)
            ax_b.bar(offs, fp, width, bottom=fn + fz, color=BEHAVIOUR_COLOURS["fork"],
                     edgecolor="white", linewidth=0.3, hatch=hatch)
        ax_b.set_xticks(xs_base)
        ax_b.set_xticklabels(type_levels, rotation=45, ha="right")
        ax_b.set_ylim(0, 1.0)
        ax_b.set_ylabel("mean fraction of eigenvalues")
        ax_b.set_title("B", loc="left", fontsize=10, fontweight="bold")
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=BEHAVIOUR_COLOURS["holder"], label="λ < −tol (holder)"),
            mpatches.Patch(color=BEHAVIOUR_COLOURS["drift"], label="|λ| ≤ tol (drift)"),
            mpatches.Patch(color=BEHAVIOUR_COLOURS["fork"], label="λ > tol (fork)"),
        ]
        if n_cond > 1:
            for ci, cond in enumerate(conditions):
                hatch = "" if ci == 0 else ["//", "..", "xx"][(ci - 1) % 3]
                handles.append(mpatches.Patch(facecolor="white", edgecolor="#404040",
                                              hatch=hatch, label=str(cond)))
        ax_b.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=6, frameon=False)
        _footer(fig, f"fractions of latent eigenvalues at tol = {tol:.3g} "
                     f"(oracle); density weight: Fisher mass; "
                     f"condition: {condition_key or 'none'}")
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


# ----------------------------------------------------------------------
# figure 3 — 3-D landscape: descriptor floor, well depth as height
# ----------------------------------------------------------------------

_HEIGHT_LABELS = {
    "stiffness": "well stiffness −mean(Re λ < 0)",
    "lambda_min": "stiffest axis −min Re λ",
    "basin_index": "basin index −Re(λ₁)",
    "energy": "energy E(z)",
}


def _binned_height(
    x: np.ndarray, y: np.ndarray, h: np.ndarray, w: np.ndarray,
    xlim: tuple[float, float], ylim: tuple[float, float],
    n_bins: int = 36, min_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fisher-weighted mean height per (x, y) bin. Returns ``(X, Y, H,
    W)`` on bin centres; ``H`` is NaN where no cell lands (so the
    surface is masked there rather than interpolated across gaps)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(h) & np.isfinite(w) & (w > 0)
    xe = np.linspace(xlim[0], xlim[1], n_bins + 1)
    ye = np.linspace(ylim[0], ylim[1], n_bins + 1)
    Wsum, _, _ = np.histogram2d(x[m], y[m], bins=[xe, ye], weights=w[m])
    Hsum, _, _ = np.histogram2d(x[m], y[m], bins=[xe, ye], weights=w[m] * h[m])
    with np.errstate(invalid="ignore", divide="ignore"):
        H = Hsum / Wsum
    H[Wsum <= min_weight] = np.nan
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    return X, Y, H.T, Wsum.T


def figure_behaviour_landscape_3d(
    adata: AnnData,
    y: str = "spectral_gap",
    height: str = "stiffness",
    null: Optional[dict] = None,
    weight: str = "fisher",
    n_bins: int = 36,
    n_grid: int = 120,
    bw: Optional[float] = None,
    n_levels: int = 8,
    elev: float = 32.0,
    azim: float = -58.0,
    out: Optional[str | Path] = None,
) -> Figure:
    """3-D behaviour landscape: operator-space floor ``(lead_re, y)``,
    height = curvature-derived well depth, Fisher-weighted density
    contours projected onto the floor.

    The point of the third axis is to carry something the floor and the
    density do not already encode, so that places where depth and
    crowding disagree stand out: a tall, sparsely occupied cell of the
    surface is a deep well few cells sit in; a crowded floor region
    under a low surface is a shallow well many cells sit in.

    ``height``:

    * ``"stiffness"`` (default) — ``−mean(Re λ < 0)`` per cell, the
      depth along the directions that hold the cell. Computed from the
      dense latent J via :func:`scjdo2.opspace.well_stiffness` when
      ``obs['op_stiffness']`` is absent.
    * ``"lambda_min"`` — the stiffest single axis.
    * ``"basin_index"`` — ``−lead_re``. Allowed but warned against: it
      is ``−x``, a plane over this floor.
    * ``"energy"`` — ``obs['energy']`` (``E(z)`` from the energy drift).
      Warned against: on the snapshot path ``E ≈ −log p``, so its
      valleys are the density contours redrawn as height.
    * any other string — an ``obs`` column to use as height.

    ``null`` (from :func:`scjdo2.opspace.flow_scrambled_null`) adds the
    null's binned height as grey ticks and its density as dashed floor
    contours.
    """
    import warnings

    from .. import opspace as _opspace

    for col in ("lead_re", y):
        if col not in adata.obs:
            raise RuntimeError(
                f"adata.obs[{col!r}] missing — call scjdo2.opspace.invariants first",
            )
    if height == "stiffness" and "op_stiffness" not in adata.obs:
        _opspace.well_stiffness(adata)
    if height == "lambda_min" and "op_lambda_min" not in adata.obs:
        _opspace.well_stiffness(adata)
    if height == "basin_index":
        warnings.warn(
            "height='basin_index' is −lead_re, i.e. −x over this floor: the "
            "surface will be a plane. Use 'stiffness' or 'lambda_min'.",
            RuntimeWarning, stacklevel=2,
        )
    if height == "energy":
        warnings.warn(
            "height='energy' is ≈ −log density on the snapshot path; the "
            "surface will restate the floor contours.",
            RuntimeWarning, stacklevel=2,
        )
    height_col = {
        "stiffness": "op_stiffness", "lambda_min": "op_lambda_min",
        "basin_index": "op_basin_index", "energy": "energy",
    }.get(height, height)
    if height_col not in adata.obs:
        if height == "basin_index" and "lead_re" in adata.obs:
            adata.obs["op_basin_index"] = -adata.obs["lead_re"].to_numpy(dtype=float)
        else:
            raise RuntimeError(f"adata.obs[{height_col!r}] missing for height={height!r}")

    x_real = adata.obs["lead_re"].to_numpy(dtype=float)
    y_real = adata.obs[y].to_numpy(dtype=float)
    h_real = adata.obs[height_col].to_numpy(dtype=float)
    w_real = (_opspace.fisher_mass(adata) if weight == "fisher"
              else np.ones(adata.n_obs))

    x_null = y_null = h_null = w_null = None
    null_key = {"stiffness": "stiffness", "lambda_min": "lambda_min",
                "basin_index": "lead_re"}.get(height)
    if null is not None:
        x_null = np.asarray(null["lead_re"], dtype=float)
        y_null = np.asarray(null[y], dtype=float)
        w_null = np.asarray(null.get("weight", np.ones_like(x_null)), dtype=float)
        if null_key is not None and null_key in null:
            h_null = np.asarray(null[null_key], dtype=float)
            if height == "basin_index":
                h_null = -h_null

    xlim = _shared_limits(x_real, x_null)
    ylim = _shared_limits(y_real, y_null)
    X, Y, H, Wb = _binned_height(x_real, y_real, h_real, w_real, xlim, ylim, n_bins)
    Xd, Yd, Z = _weighted_kde_grid(x_real, y_real, w_real, xlim, ylim, n_grid, bw)
    Hn = Zn = None
    if x_null is not None:
        _, _, Zn = _weighted_kde_grid(x_null, y_null, w_null, xlim, ylim, n_grid, bw)
        if h_null is not None:
            _, _, Hn, _ = _binned_height(x_null, y_null, h_null, w_null, xlim, ylim, n_bins)

    h_finite = H[np.isfinite(H)]
    if Hn is not None:
        h_finite = np.concatenate([h_finite, Hn[np.isfinite(Hn)]])
    if h_finite.size == 0:
        h_lo, h_hi = 0.0, 1.0
    else:
        h_lo, h_hi = float(np.min(h_finite)), float(np.max(h_finite))
        if h_hi <= h_lo:
            h_hi = h_lo + 1.0
    floor = h_lo - 0.25 * (h_hi - h_lo)

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN * 0.75, OPSPACE_FIG_HEIGHT_IN * 1.15))
        ax = fig.add_subplot(111, projection="3d")
        # Height map: one bar per occupied bin (Fisher-weighted mean
        # height), shaded by occupancy. Bars, not a surface, so empty
        # bins stay empty instead of being interpolated across.
        occ = np.where(np.isfinite(H), Wb, 0.0)
        occ_max = float(occ.max()) if occ.size and float(occ.max()) > 0 else 1.0
        cmap = plt.get_cmap("Blues")
        dx = (xlim[1] - xlim[0]) / n_bins
        dy = (ylim[1] - ylim[0]) / n_bins
        ii, jj = np.where(np.isfinite(H))
        if ii.size:
            xs_b = X[ii, jj] - dx / 2.0
            ys_b = Y[ii, jj] - dy / 2.0
            hs_b = H[ii, jj] - floor
            cols = cmap(0.25 + 0.75 * occ[ii, jj] / occ_max)
            ax.bar3d(xs_b, ys_b, np.full(ii.size, floor), dx * 0.95, dy * 0.95,
                     hs_b, color=cols, edgecolor="#1F4E79", linewidth=0.15,
                     shade=False, alpha=0.9)
        if Hn is not None:
            ni, nj = np.where(np.isfinite(Hn))
            if ni.size:
                ax.scatter(X[ni, nj], Y[ni, nj], Hn[ni, nj], s=3, c="#808080",
                           marker="_", linewidths=0.6, alpha=0.8, depthshade=False)
        # Floor: density contours (and null contours dashed).
        vmax = float(np.max(Z)) if np.isfinite(Z).any() and float(np.max(Z)) > 0 else 1.0
        levels = np.linspace(vmax / n_levels, vmax, n_levels)
        ax.contour(Xd, Yd, Z, levels=levels, zdir="z", offset=floor,
                   colors="#1F4E79", linewidths=0.5)
        if Zn is not None:
            nmax = float(np.max(Zn)) if float(np.max(Zn)) > 0 else 1.0
            ax.contour(Xd, Yd, Zn, levels=np.linspace(nmax / n_levels, nmax, n_levels),
                       zdir="z", offset=floor, colors="#808080",
                       linewidths=0.5, linestyles="dashed")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(floor, h_hi + 0.05 * (h_hi - h_lo))
        ax.set_xlabel(relative_label(adata, "Re(λ₁)"), labelpad=4)
        ylab = {"spectral_gap": "spectral gap", "henrici": "Henrici"}.get(y, y)
        if y == "spectral_gap":
            ylab = relative_label(adata, ylab)
        ax.set_ylabel(ylab, labelpad=4)
        zlab = _HEIGHT_LABELS.get(height, height)
        if height in ("stiffness", "lambda_min", "basin_index"):
            zlab = relative_label(adata, zlab)
        ax.set_zlabel(zlab, labelpad=4)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("behaviour landscape — bars: well depth, floor: density",
                     loc="left", fontsize=8)
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(facecolor=cmap(0.95), edgecolor="#1F4E79",
                           label="bars: well depth per bin (shade = occupancy)"),
            mlines.Line2D([], [], color="#1F4E79", lw=0.8, label="floor: density contours"),
        ]
        if null is not None:
            handles.append(mlines.Line2D([], [], color="#808080", lw=0.8, ls="--",
                                         label="scrambled-flow null (floor contours; "
                                               "grey ticks = null height)"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                  fontsize=6, frameon=False)
        _footer(fig, f"height: {height} (curvature-derived, not density); "
                     f"weight: {weight}; bins: {n_bins}; "
                     f"null: {'none' if null is None else null.get('mode', 'flow')}")
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig
