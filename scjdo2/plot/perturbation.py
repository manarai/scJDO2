"""Perturb-seq (K562) operator-vs-DE figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ._save import save_figure
from .style import style
from .symmetric import _footer


def figure_perturb_paired(result: dict | pd.DataFrame | Path | str,
                           out: Optional[str | Path] = None) -> Figure:
    """Two-panel K562 figure.

    ``result`` can be a per-knockdown DataFrame OR a path to the CSV OR
    a dict pointing at both the per-KD and generic-control tables.

    Left panel: operator cos vs DE-only cos as paired points per KD +
    y=x diagonal + win fraction annotation.
    Right panel: generic-response decomposition — mean cosine for
    full / generic-component / residual, across operator, DE-only
    and random.
    """
    if isinstance(result, (str, Path)):
        pk = pd.read_csv(result)
        gen = None
    elif isinstance(result, pd.DataFrame):
        pk = result
        gen = None
    else:
        pk = result.get("per_kd")
        gen = result.get("generic_table")
        if isinstance(pk, (str, Path)):
            pk = pd.read_csv(pk)
        if isinstance(gen, (str, Path)):
            gen = pd.read_csv(gen)

    valid = pk.dropna(subset=["cos_scjdo2", "cos_de_only"])
    win_frac = float((valid["cos_scjdo2"] > valid["cos_de_only"]).mean())

    with style():
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
        # Left: paired scatter.
        ax_l.scatter(valid["cos_de_only"], valid["cos_scjdo2"], s=6, alpha=0.4,
                      color="0.3")
        lim = max(float(valid[["cos_de_only", "cos_scjdo2"]].quantile(0.99).max()),
                   abs(float(valid[["cos_de_only", "cos_scjdo2"]].quantile(0.01).min())))
        ax_l.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
        ax_l.axhline(0, color="k", lw=0.3); ax_l.axvline(0, color="k", lw=0.3)
        ax_l.set_xlim(-lim, lim); ax_l.set_ylim(-lim, lim)
        ax_l.set_xlabel("DE-only  cos(b_g, δx_obs)")
        ax_l.set_ylabel("operator  cos(δx_pred, δx_obs)")
        ax_l.set_title(f"K562 per-KD paired  —  frac(op > DE) = {win_frac:.2%}  "
                        f"(n = {len(valid)})", fontsize=9)
        ax_l.set_aspect("equal", adjustable="box")

        # Right: generic decomposition bars.
        bars = []
        if gen is not None and not gen.empty:
            # Expect rows for arm in {full_space_all, residual_all}.
            try:
                full = gen[gen["arm"] == "full_space_all"].iloc[0]
                res = gen[gen["arm"] == "residual_all"].iloc[0]
                bars.append(("full", "operator", float(full["op_mean"])))
                bars.append(("full", "DE-only",  float(full["de_mean"])))
                bars.append(("residual", "operator", float(res["op_mean"])))
                bars.append(("residual", "DE-only",  float(res["de_mean"])))
            except (IndexError, KeyError):
                pass
        if bars:
            labels_x = ["full\noperator", "full\nDE-only",
                        "residual\noperator", "residual\nDE-only"]
            vals = [b[2] for b in bars]
            colors = ["#d55e00", "#0072b2", "#d55e00", "#0072b2"]
            ax_r.bar(range(len(bars)), vals, color=colors)
            ax_r.set_xticks(range(len(bars)))
            ax_r.set_xticklabels(labels_x, fontsize=7)
            ax_r.axhline(0, color="k", lw=0.5)
            ax_r.set_ylabel("mean cosine")
            ax_r.set_title("generic-response decomposition  "
                            "(full → residual = KD-specific)",
                            fontsize=9)
            for i, v in enumerate(vals):
                ax_r.text(i, v, f"{v:+.3f}", ha="center",
                           va="bottom" if v >= 0 else "top", fontsize=7)
        else:
            ax_r.text(0.5, 0.5,
                      "generic_control_comparison.csv not provided",
                      ha="center", va="center", transform=ax_r.transAxes,
                      fontsize=8, color="0.5")
            ax_r.set_axis_off()
        _footer(fig, None,
                 extra="K562 Perturb-seq  ·  controls per V2_PLAN §9f-g")
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig


# =====================================================================
# 6. basin_index_by_type
# =====================================================================


