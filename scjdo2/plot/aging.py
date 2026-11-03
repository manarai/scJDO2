"""Aging figures: young-to-old vectors and energy transects (A04)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from ._save import save_figure
from .style import style


# =====================================================================
# 7. aging comparison figures
# =====================================================================
AGING_FOOTER = (
    "v0.5-basin-lock · aging-basins · classify_basin_latent(rule='lead_eig') "
    "· scArches shared space · d_latent=15 · units relative"
)


def add_aging_footer(fig: Figure, extra: str = "") -> None:
    """Add the fixed A04 aging-analysis footer to a rendered figure."""
    text = AGING_FOOTER if not extra else f"{AGING_FOOTER} · {extra}"
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=5, color="0.35")


def figure_aging_vectors(results: pd.DataFrame,
                         axes: tuple[str, str] = ("median_abs_lambda", "anisotropy"),
                         style_by: str = "tissue",
                         colour_by: str = "mode",
                         out: Optional[str | Path] = None) -> Figure:
    """Plot stored young-to-old per-type vectors in log-scaled summary space."""
    x_key, y_key = axes
    req = {"cell_type", style_by, colour_by,
           f"young_{x_key}", f"old_{x_key}", f"young_{y_key}", f"old_{y_key}"}
    missing = sorted(req.difference(results.columns))
    if missing:
        raise KeyError(f"aging_vectors missing columns: {missing}")
    work = results.copy()
    for col in (f"young_{x_key}", f"old_{x_key}", f"young_{y_key}", f"old_{y_key}"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    tissues = sorted(work[style_by].astype(str).dropna().unique())
    if not tissues:
        raise ValueError("aging_vectors needs at least one tissue")
    colours = {"mixed": "#666666", "softening": "#378ADD", "stiffening": "#D85A30"}

    def _decade_limits(values: np.ndarray) -> tuple[float, float]:
        """Centre a positive log axis on the geometric mean with ≥1 decade span."""
        vals = values[np.isfinite(values) & (values > 0)]
        if not len(vals):
            return 0.1, 1.0
        centre = float(np.exp(np.mean(np.log(vals))))
        half_span = max(0.5, float(np.max(np.abs(np.log10(vals / centre)))) + 0.04)
        return centre * 10 ** (-half_span), centre * 10 ** half_span

    modes = sorted(work[colour_by].astype(str).dropna().unique())
    for i, mode in enumerate(modes):
        colours.setdefault(mode, plt.get_cmap("tab10")(i % 10))
    with style():
        fig, axs = plt.subplots(1, len(tissues), figsize=(5.3 * len(tissues), 4.6),
                                squeeze=False, sharey=False)
        fig.subplots_adjust(bottom=0.27, top=0.86)
        for ax, tissue in zip(axs[0], tissues):
            sub = work[work[style_by].astype(str) == tissue]
            for row in sub.itertuples(index=False):
                d = row._asdict()
                x0, x1 = d[f"young_{x_key}"], d[f"old_{x_key}"]
                y0, y1 = d[f"young_{y_key}"], d[f"old_{y_key}"]
                if not np.isfinite([x0, x1, y0, y1]).all() or min(x0, x1, y0, y1) <= 0:
                    continue
                colour = colours[str(d[colour_by])]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=colour,
                                            linewidth=1.2, alpha=0.85))
                ax.scatter([x0], [y0], s=22, facecolor="white", edgecolor=colour,
                           linewidth=0.8, zorder=3)
                ax.scatter([x1], [y1], s=22, color=colour, edgecolor="white",
                           linewidth=0.4, zorder=3)
                ax.text(x1, y1, f" {d['cell_type']}", fontsize=6, va="center")
                ax.text(np.sqrt(x0 * x1), np.sqrt(y0 * y1), f"{x1 / x0:.2f}×",
                        fontsize=5.5, ha="center", va="bottom", color=colour,
                        bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                                  edgecolor="none", alpha=0.75))
            xvals = sub[[f"young_{x_key}", f"old_{x_key}"]].to_numpy(float).ravel()
            yvals = sub[[f"young_{y_key}", f"old_{y_key}"]].to_numpy(float).ravel()
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*_decade_limits(xvals)); ax.set_ylim(*_decade_limits(yvals))
            ax.set_xlabel("overall stiffness (relative) → stiffer")
            ax.set_ylabel("anisotropy (soft axis / median)")
            ax.set_title(str(tissue))
            ax.grid(True, which="both", lw=0.25, color="0.88")
        fig.suptitle("Aging vectors: young → old by cell type", fontsize=10)
        fig.text(0.5, 0.095,
                 "2-D projection; descriptive modes were clustered on full coordinates.",
                 ha="center", fontsize=6, color="0.35")
        add_aging_footer(fig)
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig


def figure_energy_transect(adata: AnnData,
                           cell_type: str,
                           neighbour: str,
                           energy,
                           groups: Sequence[str] = ("young", "old"),
                           by: str = "cell_type",
                           group_key: str = "age_group",
                           use_rep: Optional[str] = None,
                           n_cells_centroid: int = 30,
                           out: Optional[str | Path] = None) -> Figure:
    """Render stored energy along a young soft-axis transect.

    The line begins at the young centroid of ``cell_type`` and follows its
    leading eigendirection to the ``neighbour`` centroid.  Those two stored
    transect endpoints are the floors; the old centroid is separately projected
    onto that line. A saddle is the highest strict local maximum between the
    endpoint floors. ``energy`` is a fitted adapter's method
    or an object exposing ``energy(z)``; it is never trained here.
    """
    if by not in adata.obs or group_key not in adata.obs:
        raise KeyError(f"energy_transect requires obs[{by!r}] and obs[{group_key!r}]")
    use_rep = use_rep or (adata.uns.get("scjdo2", {}).get("meta", {}) or {}).get(
        "use_rep", "X_scVI" if "X_scVI" in adata.obsm else "X_pca",
    )
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"].get("J_latent", adata.uns["scjdo2"]["J"]))
    labels = adata.obs[by].astype(str).to_numpy()
    ages = adata.obs[group_key].astype(str).to_numpy()
    young, old = map(str, groups[:2])
    target_y = (labels == str(cell_type)) & (ages == young)
    target_o = (labels == str(cell_type)) & (ages == old)
    nb = labels == str(neighbour)
    if target_y.sum() < 5 or target_o.sum() < 5 or nb.sum() < 5:
        raise ValueError("energy_transect needs >=5 young, old, and neighbour cells")
    z_y = z[target_y].mean(axis=0)
    z_o = z[target_o].mean(axis=0)
    z_nb = z[nb].mean(axis=0)
    pool = np.flatnonzero(target_y)
    dist = np.linalg.norm(z[pool] - z_y, axis=1)
    take = min(int(n_cells_centroid), len(pool))
    idx = pool[np.argpartition(dist, take - 1)[:take]]
    J_y = J[idx].mean(axis=0)
    ev, V = np.linalg.eigh(0.5 * (J_y + J_y.T))
    u = V[:, int(np.argmax(ev))]
    toward = z_nb - z_y
    if float(np.dot(u, toward)) < 0:
        u = -u
    length = float(np.linalg.norm(toward))
    if length <= 1e-12:
        raise ValueError("neighbour centroid is coincident with young centroid")
    grid_t = np.linspace(-0.25, 1.25, 240)
    grid_z = z_y[None, :] + grid_t[:, None] * length * u[None, :]
    energy_fn = energy.energy if hasattr(energy, "energy") else energy
    curve = np.asarray(energy_fn(grid_z), dtype=float).reshape(-1)
    cell_t = ((z - z_y) @ u) / length
    cell_E = np.asarray(energy_fn(z), dtype=float).reshape(-1)
    old_t = float(np.dot(z_o - z_y, u) / length)
    young_E = float(np.asarray(energy_fn(z_y[None, :])).reshape(-1)[0])
    old_E = float(np.asarray(energy_fn(z_o[None, :])).reshape(-1)[0])
    neighbour_E = float(np.asarray(energy_fn(z_nb[None, :])).reshape(-1)[0])
    # The displayed line is the stored young-to-neighbour transect: its floors
    # are fixed at t=0 and t=1. The old centroid is an overlaid projection.
    lo, hi = 0.0, 1.0
    strictly_between = (grid_t > lo) & (grid_t < hi)
    local_max = np.zeros_like(strictly_between, dtype=bool)
    local_max[1:-1] = ((curve[1:-1] > curve[:-2]) & (curve[1:-1] > curve[2:]))
    candidates = np.flatnonzero(strictly_between & local_max)
    saddle_idx = int(candidates[np.argmax(curve[candidates])]) if len(candidates) else None
    old_pool = np.flatnonzero(target_o)
    old_dist = np.linalg.norm(z[old_pool] - z_o, axis=1)
    old_take = min(take, len(old_pool))
    old_idx = old_pool[np.argpartition(old_dist, old_take - 1)[:old_take]]
    J_o = J[old_idx].mean(axis=0)
    old_ev = np.linalg.eigvalsh(0.5 * (J_o + J_o.T))
    with style():
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(grid_t, curve, color="black", lw=1.3, label="stored E(z)")
        for age, mask, color in ((young, target_y, "#378ADD"), (old, target_o, "#D85A30")):
            show = mask & (cell_t >= -0.3) & (cell_t <= 1.3)
            ax.scatter(cell_t[show], cell_E[show], s=5, alpha=0.22, color=color,
                       edgecolors="none", label=f"{age} projected cells")
        ax.scatter([0.0, 1.0], [young_E, neighbour_E], s=42,
                   color=["#378ADD", "#888780"], zorder=4, label="transect endpoint floors")
        ax.scatter([old_t], [old_E], marker="D", s=32, color="#D85A30", zorder=4,
                   label="old centroid projection")
        if saddle_idx is not None:
            ax.scatter([grid_t[saddle_idx]], [curve[saddle_idx]], marker="^", s=45,
                       color="#888780", zorder=4, label="intervening saddle")
        ax.axvline(0.0, color="#378ADD", lw=0.6, ls="--")
        ax.axvline(1.0, color="#888780", lw=0.6, ls="--")
        ax.axvline(old_t, color="#D85A30", lw=0.6, ls=":")
        ax.text(0.0, young_E, f"  young λ={float(ev.max()):+.3g}", fontsize=7, va="bottom")
        ax.text(old_t, old_E, f"  old λ={float(old_ev.max()):+.3g}", fontsize=7, va="bottom")
        barrier_text = (f"barrier height = {curve[saddle_idx] - young_E:+.3g}\n"
                        if saddle_idx is not None else "no barrier along this transect\n")
        ax.text(0.02, 0.96, barrier_text + f"ΔE(old − young) = {old_E - young_E:+.3g}\n"
                "ΔE includes abundance in the learned energy",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7"))
        ax.set_xlabel(f"distance along young soft axis toward {neighbour}")
        ax.set_ylabel("stored energy E(z) (relative)")
        ax.set_title(f"energy transect — {cell_type}")
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        add_aging_footer(fig)
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig
