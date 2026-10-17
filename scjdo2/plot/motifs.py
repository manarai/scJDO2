"""Motif × context occurrence matrix (atlas arm).

:func:`figure_motif_matrix` draws the output of
:mod:`scjdo2.motifs`: rows = spectral motifs, columns = contexts
grouped by the last context key (tissue), cell fill = occupancy where
the motif is enriched above the shuffled-context null (blank where it
is not), a recurrence bar at the right (number of enriched contexts
per motif), and — when driver similarity has been computed — a second
panel with the same layout showing the per-(motif, context) cosine to
the motif's consensus driver vector.

Row shapes carry the reading: a fully filled row is a universal motif,
a row filled only in one tissue block is local, a row lit at
structurally similar positions across tissues is a motif tied to a
situation rather than a tissue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from .style import OPSPACE_FIG_HEIGHT_IN, OPSPACE_FIG_WIDTH_IN, style


def _motif_name(adata: AnnData, i: int, tol: Optional[float] = None) -> str:
    """Short descriptive name from the motif's raw-unit summary
    (``uns["scjdo2"]["motifs"]["summary"]``): sign of the mean leading
    eigenvalue against ``tol`` (the stored oracle tol, else
    ``behaviour_fractions_tol``, else 0.1 × spread), plus deep/shallow
    from the mean stiffness relative to the median across motifs."""
    try:
        summ = pd.DataFrame(adata.uns["scjdo2"]["motifs"]["summary"]).set_index("motif")
        lead = float(summ.loc[i, "lead_re"])
        stiff = float(summ.loc[i, "stiffness"])
        if tol is None:
            scj = adata.uns.get("scjdo2", {})
            tol = scj.get("oracle_tol", scj.get("behaviour_fractions_tol"))
        if tol is None:
            tol = 0.1 * float(np.nanstd(summ["lead_re"])) or 0.1
        if lead < -tol:
            kind = "basin"
        elif lead > tol:
            kind = "saddle"
        else:
            kind = "flat"
        med = float(np.nanmedian(summ["stiffness"]))
        if kind != "flat":
            depth = ("deep " if stiff > 1.25 * med else
                     ("shallow " if stiff < 0.75 * med else ""))
        else:
            depth = ""
        return f"{i}: {depth}{kind}"
    except (KeyError, IndexError, TypeError, ValueError):
        return f"motif {i}"


def figure_motif_matrix(
    adata: AnnData,
    show_drivers: Optional[bool] = None,
    min_cells: int = 20,
    out: Optional[str | Path] = None,
) -> Figure:
    """Draw the motif × context occurrence matrix (and the driver-
    similarity twin when available).

    Needs ``uns["scjdo2"]["motifs"]["occurrence"]`` and
    ``["recurrence"]`` from :func:`scjdo2.motifs.motif_occurrence`;
    uses ``["driver_similarity"]`` when present (or when
    ``show_drivers=True``).
    """
    uns = adata.uns.get("scjdo2", {}).get("motifs", {})
    occ = uns.get("occurrence")
    rec = uns.get("recurrence")
    if occ is None or rec is None:
        raise RuntimeError("call scjdo2.motifs.motif_occurrence first")
    occ = pd.DataFrame(occ)
    rec = pd.DataFrame(rec).set_index("motif")
    ctx_keys = list(uns.get("context_keys", ["cluster", "tissue"]))
    tissue_key = ctx_keys[-1]
    drivers = uns.get("driver_similarity")
    if show_drivers is None:
        show_drivers = drivers is not None and len(drivers) > 0
    if show_drivers and drivers is None:
        raise RuntimeError("show_drivers=True but no driver_similarity stored")

    occ = occ[occ["n"] >= min_cells]
    motifs = sorted(occ["motif"].unique())
    # contexts ordered by tissue block, then by name
    ctx_df = occ[["context", tissue_key]].drop_duplicates()
    ctx_df = ctx_df.sort_values([tissue_key, "context"])
    contexts = ctx_df["context"].tolist()
    tissues = ctx_df[tissue_key].tolist()
    n_m, n_c = len(motifs), len(contexts)

    occ_mat = np.full((n_m, n_c), np.nan)
    enr_mat = np.zeros((n_m, n_c), dtype=bool)
    for r in occ.itertuples(index=False):
        i = motifs.index(r.motif)
        j = contexts.index(r.context)
        occ_mat[i, j] = r.occupancy
        enr_mat[i, j] = bool(r.enriched)
    shown = np.where(enr_mat, occ_mat, np.nan)

    drv_mat = None
    if show_drivers:
        drv = pd.DataFrame(drivers)
        drv_mat = np.full((n_m, n_c), np.nan)
        for r in drv.itertuples(index=False):
            if r.motif in motifs and r.context in contexts:
                drv_mat[motifs.index(r.motif), contexts.index(r.context)] = r.driver_cosine

    with style():
        n_panels = 2 if show_drivers else 1
        fig_w = OPSPACE_FIG_WIDTH_IN
        fig_h = OPSPACE_FIG_HEIGHT_IN * (0.55 + 0.08 * n_m) * (1.6 if show_drivers else 1.0)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            nrows=n_panels, ncols=4, width_ratios=[1.0, 0.03, 0.10, 0.18],
            wspace=0.08, hspace=0.45,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[0, 3], sharey=ax)
        im = ax.imshow(shown, aspect="auto", cmap="Purples", vmin=0.0,
                       vmax=float(np.nanmax(occ_mat)) if np.isfinite(occ_mat).any() else 1.0,
                       interpolation="nearest")
        # faint outline for every tested cell so blank = tested-not-enriched
        for i in range(n_m):
            for j in range(n_c):
                if np.isfinite(occ_mat[i, j]) and not enr_mat[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#D3D1C7", linewidth=0.3))
        _tissue_blocks(ax, tissues, n_m)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([_motif_name(adata, m) for m in motifs])
        ax.set_xticks(range(n_c))
        ax.set_xticklabels([c.split(" | ")[0] for c in contexts], rotation=90, fontsize=6)
        ax.set_title("A  motif occurrence (filled = enriched above shuffled-context null)",
                     loc="left")
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("occupancy", fontsize=7)

        n_enr = np.array([int(rec.loc[m, "n_enriched"]) if m in rec.index else 0 for m in motifs])
        n_tot = int(rec["n_contexts"].iloc[0]) if "n_contexts" in rec else n_c
        ax_bar.barh(range(n_m), n_enr, color="#444441", height=0.6)
        for i, v in enumerate(n_enr):
            ax_bar.text(v + 0.2, i, f"{v}/{n_tot}", va="center", fontsize=6)
        ax_bar.set_xlim(0, n_tot * 1.35)
        ax_bar.set_xlabel("recurrence", fontsize=7)
        ax_bar.tick_params(labelleft=False)
        ax_bar.spines["left"].set_visible(False)

        if show_drivers and drv_mat is not None:
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
            cax2 = fig.add_subplot(gs[1, 1])
            im2 = ax2.imshow(drv_mat, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0,
                             interpolation="nearest")
            _tissue_blocks(ax2, tissues, n_m)
            ax2.set_yticks(range(n_m))
            ax2.set_yticklabels([_motif_name(adata, m) for m in motifs])
            ax2.set_xticks(range(n_c))
            ax2.set_xticklabels([c.split(" | ")[0] for c in contexts], rotation=90, fontsize=6)
            ax2.set_title("B  driver similarity (cosine to motif consensus driver vector)",
                          loc="left")
            cb2 = fig.colorbar(im2, cax=cax2)
            cb2.set_label("cosine", fontsize=7)
            fig.add_subplot(gs[1, 3]).axis("off")

        null = uns.get("null", {})
        fig.text(0.01, 0.005,
                 f"contexts: {' × '.join(ctx_keys)}; null: context labels shuffled within "
                 f"{tissue_key} ({null.get('n_perm', '?')} perms, α={null.get('alpha', '?')}, "
                 f"replicate={null.get('replicate_key') or 'none'}); "
                 "pooled operators carry batch unless aligned (scOpAlign) — run within-dataset first",
                 fontsize=6, color="#606060", ha="left", va="bottom")
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=300)
    return fig


def _tissue_blocks(ax, tissues: list[str], n_rows: int) -> None:
    """Vertical separators and top labels for the tissue blocks."""
    starts = [0] + [j for j in range(1, len(tissues)) if tissues[j] != tissues[j - 1]]
    ends = starts[1:] + [len(tissues)]
    for s, e in zip(starts, ends):
        if s > 0:
            ax.axvline(s - 0.5, color="#404040", linewidth=0.6)
        ax.text((s + e - 1) / 2.0, -0.9, tissues[s], ha="center", va="bottom",
                fontsize=7, color="#404040", clip_on=False)
    ax.set_ylim(n_rows - 0.5, -0.5)
